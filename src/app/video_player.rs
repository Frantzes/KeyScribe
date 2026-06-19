use std::io::Read;
use std::process::{Stdio, ChildStdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, sync_channel, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use eframe::egui;

const VIDEO_WIDTH: usize = 1280;
const VIDEO_HEIGHT: usize = 720;
const FRAME_BYTES: usize = VIDEO_WIDTH * VIDEO_HEIGHT * 4;

/// VLC-style sync thresholds (in seconds). The video follows the master audio
/// clock and corrects drift in three tiers, exactly like VLC's audio-master
/// synchronization:
///
/// * `LATE_THRESHOLD`  — if a decoded frame's PTS is older than the audio
///   clock by more than this, the frame is dropped (skipped) to catch up.
/// * `EARLY_THRESHOLD` — if the next frame's PTS is ahead of the audio clock
///   by more than this, the current frame is held (repeated) until audio
///   catches up. Kept small (10 ms) so the video doesn't run noticeably
///   ahead of the audio.
/// * `SEEK_THRESHOLD`  — if the cumulative drift exceeds this, a hard seek of
///   the decoder is triggered. Small drift is corrected by drop/repeat;
///   only large drift (e.g. after a user jump) restarts ffmpeg.
const LATE_THRESHOLD: f32 = 0.020;
const EARLY_THRESHOLD: f32 = 0.010;
const SEEK_THRESHOLD: f32 = 0.40;

pub struct VideoPlayer {
    _path: String,
    _fps: f64,
    _frame_duration: f32,
    current_time: f32,
    frame_rx: Receiver<(f32, egui::ColorImage)>,
    seek_tx: Sender<f32>,
    texture: Option<egui::TextureHandle>,
    /// PTS (seconds) of the frame currently displayed on screen.
    displayed_pts: f32,
    /// The next decoded frame waiting to be shown.
    next_frame: Option<(f32, egui::ColorImage)>,
    cancel_flag: Arc<AtomicBool>,
}

impl VideoPlayer {
    pub fn new(path: String) -> Self {
        let (frame_tx, frame_rx) = sync_channel(30);
        let (seek_tx, seek_rx) = channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));

        // Probe the real frame rate so PTS is accurate. Fall back to 30 fps
        // if ffprobe is unavailable — the sync logic still works, just with
        // slightly less precise frame timing.
        let probed = crate::dsp::probe_video_stream(std::path::Path::new(&path));
        let fps = probed.as_ref().map(|i| i.fps).unwrap_or(30.0);
        let frame_duration = (1.0 / fps) as f32;

        let cancel_clone = cancel_flag.clone();
        let path_clone = path.clone();

        thread::spawn(move || {
            video_decoder_thread(path_clone, fps, frame_tx, seek_rx, cancel_clone);
        });

        Self {
            _path: path,
            _fps: fps,
            _frame_duration: frame_duration,
            current_time: 0.0,
            frame_rx,
            seek_tx,
            texture: None,
            displayed_pts: 0.0,
            next_frame: None,
            cancel_flag,
        }
    }

    /// Feed the master audio clock into the video sync engine.
    ///
    /// `audio_clock_sec` is the latency-compensated audio position — i.e. the
    /// time of the sample currently *audible* through the speakers. The video
    /// aligns its displayed frame to this clock using VLC-style threshold
    /// correction (drop late frames, hold early frames, seek on large drift).
    fn sync_to_audio_clock(&mut self, audio_clock_sec: f32) {
        // If the audio clock jumped far from where the video is, trigger a
        // hard seek of the decoder. This handles user-initiated jumps and
        // large cumulative drift that frame drop/repeat can't fix.
        let drift = audio_clock_sec - self.displayed_pts;
        if drift.abs() > SEEK_THRESHOLD {
            let _ = self.seek_tx.send(audio_clock_sec);
            self.next_frame = None;
            // Drain stale frames from the channel so we don't display frames
            // from the old decoder position after the seek.
            while self.frame_rx.try_recv().is_ok() {}
            self.displayed_pts = audio_clock_sec;
            self.current_time = audio_clock_sec;
            return;
        }

        self.current_time = audio_clock_sec;
    }

    pub fn draw(&mut self, ui: &mut egui::Ui, audio_clock_sec: f32, _playing: bool) {
        self.sync_to_audio_clock(audio_clock_sec);

        // Ensure we have a frame queued for the scheduling loop.
        while self.next_frame.is_none() {
            match self.frame_rx.try_recv() {
                Ok(frame) => self.next_frame = Some(frame),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        }

        // VLC-style frame scheduling: walk the queued frames and decide which
        // one to display based on its PTS relative to the master audio clock.
        loop {
            let Some((pts, _)) = self.next_frame.as_ref() else {
                break;
            };

            // Late frame: its PTS is behind the audio clock by more than the
            // threshold. Drop it and pull the next one — this is how VLC
            // catches up when the decoder falls behind real-time.
            if *pts < audio_clock_sec - LATE_THRESHOLD {
                self.next_frame.take();
                while self.next_frame.is_none() {
                    match self.frame_rx.try_recv() {
                        Ok(frame) => self.next_frame = Some(frame),
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                    }
                }
                continue;
            }

            // The frame's PTS is at or ahead of the audio clock. Display it
            // (or keep displaying it) until the audio clock reaches the next
            // frame's PTS. This naturally repeats frames when video is early.
            if *pts <= audio_clock_sec + EARLY_THRESHOLD || self.texture.is_none() {
                let (pts_val, image) = self.next_frame.take().unwrap();
                self.texture = Some(ui.ctx().load_texture(
                    "video_frame",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
                self.displayed_pts = pts_val;

                // Pull the next frame and continue the loop so that if
                // multiple frames are due (e.g. after a brief decoder stall),
                // they are all processed in a single draw call and only the
                // latest one is shown. This provides fast catch-up instead of
                // recovering one frame per UI tick (~16 ms).
                while self.next_frame.is_none() {
                    match self.frame_rx.try_recv() {
                        Ok(frame) => self.next_frame = Some(frame),
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                    }
                }
                continue;
            }
            break;
        }

        if let Some(tex) = &self.texture {
            let avail_size = ui.available_size();
            let aspect_ratio = VIDEO_WIDTH as f32 / VIDEO_HEIGHT as f32;
            let mut width = avail_size.x;
            let mut height = width / aspect_ratio;

            if height > avail_size.y {
                height = avail_size.y;
                width = height * aspect_ratio;
            }

            let (rect, _) = ui.allocate_exact_size(avail_size, egui::Sense::hover());
            let img_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(width, height));
            ui.painter().image(
                tex.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            let avail_size = ui.available_size();
            let (rect, _) = ui.allocate_exact_size(avail_size, egui::Sense::hover());
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Loading video...",
                egui::FontId::proportional(16.0),
                egui::Color32::WHITE,
            );
        }
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }
}

fn video_decoder_thread(
    path: String,
    fps: f64,
    frame_tx: SyncSender<(f32, egui::ColorImage)>,
    seek_rx: Receiver<f32>,
    cancel_flag: Arc<AtomicBool>,
) {
    let mut current_process: Option<std::process::Child> = None;
    let mut stdout: Option<ChildStdout> = None;
    let mut frame_index: u64 = 0;
    let mut seek_base_sec: f32 = 0.0;
    let frame_duration = (1.0 / fps.max(1.0)) as f32;

    let spawn_ffmpeg = |start_time: f32| -> Option<(std::process::Child, ChildStdout)> {
        let mut cmd = crate::dsp::get_ffmpeg_command();
        cmd.args([
            "-ignore_editlist", "1",
            "-ss", &start_time.to_string(),
            "-i", &path,
            "-f", "image2pipe",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgba",
            "-vf", &format!("scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=0x00000000", VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_HEIGHT),
            // Use the exact probed frame rate (not rounded) so that the
            // output frame interval matches the PTS computation
            // (frame_index / fps). Rounding here — e.g. 29.97 → 30 — would
            // make ffmpeg output at 30 fps while PTS advances at 29.97 fps,
            // causing the video to drift ~0.3 s every 5 minutes.
            "-r", &format!("{fps}"),
            "-"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }

        let mut child = cmd.spawn().ok()?;
        let out = child.stdout.take()?;
        Some((child, out))
    };

    if let Some((child, out)) = spawn_ffmpeg(0.0) {
        current_process = Some(child);
        stdout = Some(out);
    }

    let mut buf = vec![0u8; FRAME_BYTES];

    loop {
        if cancel_flag.load(Ordering::SeqCst) {
            if let Some(mut child) = current_process.take() {
                let _ = child.kill();
            }
            break;
        }

        // Check for seek requests. Drain to the latest request so multiple
        // rapid seeks coalesce into one ffmpeg restart.
        if let Ok(seek_time) = seek_rx.try_recv() {
            let mut final_seek = seek_time;
            while let Ok(t) = seek_rx.try_recv() {
                final_seek = t;
            }

            if let Some(mut child) = current_process.take() {
                let _ = child.kill();
            }
            stdout = None;
            frame_index = 0;
            seek_base_sec = final_seek.max(0.0);
            if let Some((child, out)) = spawn_ffmpeg(final_seek) {
                current_process = Some(child);
                stdout = Some(out);
            }
        }

        if let Some(out) = stdout.as_mut() {
            if let Ok(()) = out.read_exact(&mut buf) {
                let pixels: Vec<egui::Color32> = buf.chunks_exact(4).map(|chunk| {
                    egui::Color32::from_rgba_unmultiplied(chunk[0], chunk[1], chunk[2], chunk[3])
                }).collect();

                let image = egui::ColorImage {
                    size: [VIDEO_WIDTH, VIDEO_HEIGHT],
                    pixels,
                };

                // Compute PTS from the real probed frame rate. Combined with
                // the seek base, this gives an accurate presentation timestamp
                // that the UI thread uses for VLC-style drop/hold scheduling.
                let pts = seek_base_sec + (frame_index as f32 * frame_duration);
                if frame_tx.send((pts, image)).is_err() {
                    break;
                }
                frame_index += 1;
            } else {
                // EOF or read error — wait for a seek or cancel.
                thread::sleep(Duration::from_millis(50));
            }
        } else {
            thread::sleep(Duration::from_millis(50));
        }
    }
}
