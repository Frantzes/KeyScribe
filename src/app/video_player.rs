use std::io::Read;
use std::process::{Command, Stdio, ChildStdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, sync_channel, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;

const VIDEO_WIDTH: usize = 1280;
const VIDEO_HEIGHT: usize = 720;
const FRAME_BYTES: usize = VIDEO_WIDTH * VIDEO_HEIGHT * 4;

pub struct VideoPlayer {
    path: String,
    fps: f32,
    current_time: f32,
    frame_rx: Receiver<(f32, egui::ColorImage)>,
    seek_tx: Sender<f32>,
    texture: Option<egui::TextureHandle>,
    last_frame_time: f32,
    playback_start_time: Option<(Instant, f32)>, // (real time, video time)
    cancel_flag: Arc<AtomicBool>,
    next_frame: Option<(f32, egui::ColorImage)>,
}

impl VideoPlayer {
    pub fn new(path: String) -> Self {
        let (frame_tx, frame_rx) = sync_channel(30);
        let (seek_tx, seek_rx) = channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));

        let cancel_clone = cancel_flag.clone();
        let path_clone = path.clone();

        thread::spawn(move || {
            video_decoder_thread(path_clone, frame_tx, seek_rx, cancel_clone);
        });

        Self {
            path,
            fps: 30.0, // Default, could be parsed from ffprobe
            current_time: 0.0,
            frame_rx,
            seek_tx,
            texture: None,
            last_frame_time: 0.0,
            playback_start_time: None,
            cancel_flag,
            next_frame: None,
        }
    }

    pub fn set_time(&mut self, time_sec: f32) {
        if (time_sec - self.current_time).abs() > 0.5 {
            let _ = self.seek_tx.send(time_sec);
            // Drain the frame buffer so the background thread unblocks
            self.next_frame = None;
            while self.frame_rx.try_recv().is_ok() {}
        }
        self.current_time = time_sec;
    }

    pub fn draw(&mut self, ui: &mut egui::Ui, time_sec: f32, _playing: bool) {
        self.set_time(time_sec);

        // Fetch frames into the buffer if it's empty
        while self.next_frame.is_none() {
            match self.frame_rx.try_recv() {
                Ok(frame) => self.next_frame = Some(frame),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        }

        // Only display if the next frame's PTS is reached
        while let Some((pts, _)) = &self.next_frame {
            if *pts <= self.current_time + 0.03 { // Small margin
                let (pts_val, image) = self.next_frame.take().unwrap();
                self.texture = Some(ui.ctx().load_texture(
                    "video_frame",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
                self.last_frame_time = pts_val;

                // Grab the next frame
                match self.frame_rx.try_recv() {
                    Ok(frame) => self.next_frame = Some(frame),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                }
            } else {
                break;
            }
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
    frame_tx: SyncSender<(f32, egui::ColorImage)>,
    seek_rx: Receiver<f32>,
    cancel_flag: Arc<AtomicBool>,
) {
    let mut current_process: Option<std::process::Child> = None;
    let mut stdout: Option<ChildStdout> = None;
    let mut current_pts = 0.0;
    let fps = 30.0; // Assume 30 fps for PTS calculation
    let frame_duration = 1.0 / fps;

    let spawn_ffmpeg = |start_time: f32| -> Option<(std::process::Child, ChildStdout)> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-ignore_editlist", "1",
            "-ss", &start_time.to_string(),
            "-i", &path,
            "-f", "image2pipe",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgba",
            "-vf", &format!("scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=0x00000000", VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_HEIGHT),
            "-r", "30",
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

        // Check for seek requests
        if let Ok(seek_time) = seek_rx.try_recv() {
            // Drain remaining seeks
            let mut final_seek = seek_time;
            while let Ok(t) = seek_rx.try_recv() {
                final_seek = t;
            }

            if let Some(mut child) = current_process.take() {
                let _ = child.kill();
            }
            if let Some((child, out)) = spawn_ffmpeg(final_seek) {
                current_process = Some(child);
                stdout = Some(out);
                current_pts = final_seek;
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

                if frame_tx.send((current_pts, image)).is_err() {
                    break;
                }
                current_pts += frame_duration;
            } else {
                // EOF or error, wait for seek or cancel
                thread::sleep(Duration::from_millis(50));
            }
        } else {
            thread::sleep(Duration::from_millis(50));
        }
    }
}
