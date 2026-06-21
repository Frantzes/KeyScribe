use std::sync::atomic::{AtomicBool, Ordering};

use signalsmith_stretch::Stretch;

pub fn apply_speed_and_pitch(
    samples: &[f32],
    sample_rate: u32,
    speed: f32,
    pitch_semitones: f32,
) -> Vec<f32> {
    apply_speed_and_pitch_interleaved(samples, 1, sample_rate, speed, pitch_semitones)
}

pub(crate) fn apply_speed_and_pitch_with_cancel(
    samples: &[f32],
    sample_rate: u32,
    speed: f32,
    pitch_semitones: f32,
    cancel: &AtomicBool,
) -> Option<Vec<f32>> {
    apply_speed_and_pitch_interleaved_with_cancel(samples, 1, sample_rate, speed, pitch_semitones, cancel)
}

fn apply_speed_and_pitch_mono_with_cancel(
    samples: &[f32],
    sample_rate: u32,
    speed: f32,
    pitch_semitones: f32,
    cancel: &AtomicBool,
) -> Option<Vec<f32>> {
    if samples.is_empty() {
        return Some(Vec::new());
    }

    let clamped_speed = speed.clamp(0.25, 4.0);
    let speed_is_unity = (clamped_speed - 1.0).abs() < 1.0e-4;
    let pitch_is_zero = pitch_semitones.abs() < 1.0e-4;

    if speed_is_unity && pitch_is_zero {
        return Some(samples.to_vec());
    }

    if sample_rate == 0 {
        return Some(samples.to_vec());
    }

    let mut stretch = Stretch::preset_default(1, sample_rate);
    stretch.set_transpose_factor_semitones(pitch_semitones, None);

    let target_frames = ((samples.len() as f32) / clamped_speed).round().max(1.0) as usize;
    let input_latency = stretch.input_latency();
    let output_latency = stretch.output_latency();
    let silence_out_len = if input_latency > 0 {
        ((input_latency as f32) / clamped_speed).ceil().max(1.0) as usize
    } else {
        0
    };

    if input_latency > 0 {
        let mut seek_input = vec![0.0f32; input_latency];
        let copy = input_latency.min(samples.len());
        seek_input[..copy].copy_from_slice(&samples[..copy]);
        stretch.seek(&seek_input, clamped_speed as f64);
    }

    const BLOCK_OUT: usize = 4096;
    let mut output = Vec::with_capacity(target_frames + BLOCK_OUT);
    let mut in_pos = 0usize;
    let mut rendered = 0usize;
    let max_in_len = ((BLOCK_OUT as f32) * clamped_speed).ceil().max(1.0) as usize;
    let scratch_in_len = max_in_len.max(input_latency.max(1));
    let scratch_out_len = BLOCK_OUT.max(output_latency.max(silence_out_len).max(1));
    let mut input_scratch = vec![0.0f32; scratch_in_len];
    let mut output_chunk = vec![0.0f32; scratch_out_len];
    let mut skip_front = output_latency;

    #[inline]
    fn push_chunk_with_skip(output: &mut Vec<f32>, chunk: &[f32], skip_front: &mut usize) {
        if chunk.is_empty() {
            return;
        }

        let skip = (*skip_front).min(chunk.len());
        *skip_front -= skip;
        output.extend_from_slice(&chunk[skip..]);
    }

    while rendered < target_frames {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }

        let out_len = (target_frames - rendered).min(BLOCK_OUT);
        let in_len = ((out_len as f32) * clamped_speed).ceil().max(1.0) as usize;

        if in_pos + in_len <= samples.len() {
            let input_chunk = &samples[in_pos..in_pos + in_len];
            in_pos += in_len;
            stretch.process(input_chunk, &mut output_chunk[..out_len]);
        } else {
            let available = samples.len().saturating_sub(in_pos).min(in_len);
            if available > 0 {
                input_scratch[..available].copy_from_slice(&samples[in_pos..in_pos + available]);
                in_pos += available;
            }
            if available < in_len {
                input_scratch[available..in_len].fill(0.0);
            }
            stretch.process(&input_scratch[..in_len], &mut output_chunk[..out_len]);
        }

        push_chunk_with_skip(&mut output, &output_chunk[..out_len], &mut skip_front);
        rendered += out_len;
    }

    if input_latency > 0 && silence_out_len > 0 {
        let input_chunk = &input_scratch[..input_latency];
        stretch.process(input_chunk, &mut output_chunk[..silence_out_len]);
        push_chunk_with_skip(
            &mut output,
            &output_chunk[..silence_out_len],
            &mut skip_front,
        );
    }

    if output_latency > 0 {
        let mut flushed = 0usize;
        while flushed < output_latency {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let len = (output_latency - flushed).min(output_chunk.len());
            stretch.flush(&mut output_chunk[..len]);
            push_chunk_with_skip(&mut output, &output_chunk[..len], &mut skip_front);
            flushed += len;
        }
    }

    if output.len() < target_frames {
        output.resize(target_frames, 0.0);
    } else {
        output.truncate(target_frames);
    }

    Some(output)
}

pub fn apply_speed_and_pitch_interleaved(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    speed: f32,
    pitch_semitones: f32,
) -> Vec<f32> {
    apply_speed_and_pitch_interleaved_with_cancel(samples, channels, sample_rate, speed, pitch_semitones, &AtomicBool::new(false))
        .unwrap()
}

pub(crate) fn apply_speed_and_pitch_interleaved_with_cancel(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    speed: f32,
    pitch_semitones: f32,
    cancel: &AtomicBool,
) -> Option<Vec<f32>> {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return apply_speed_and_pitch_mono_with_cancel(samples, sample_rate, speed, pitch_semitones, cancel);
    }

    if samples.is_empty() {
        return Some(Vec::new());
    }

    let frame_count = samples.len() / channels;
    if frame_count == 0 {
        return Some(Vec::new());
    }

    let trimmed_len = frame_count * channels;
    let mut separated = vec![vec![0.0f32; frame_count]; channels];
    for frame_idx in 0..frame_count {
        let base = frame_idx * channels;
        for ch in 0..channels {
            separated[ch][frame_idx] = samples[base + ch];
        }
    }

    use rayon::prelude::*;
    let processed_channels: Vec<Option<Vec<f32>>> = separated
        .into_par_iter()
        .map(|channel| {
            apply_speed_and_pitch_mono_with_cancel(
                &channel,
                sample_rate,
                speed,
                pitch_semitones,
                cancel,
            )
        })
        .collect();

    if processed_channels.iter().any(Option::is_none) {
        return None;
    }

    let out_frames = processed_channels.iter().filter_map(|c| c.as_ref().map(Vec::len)).min().unwrap_or(0);
    if out_frames == 0 {
        return Some(Vec::new());
    }

    let mut out = Vec::with_capacity(out_frames * channels);
    for frame_idx in 0..out_frames {
        for ch in 0..channels {
            out.push(processed_channels[ch].as_ref().unwrap()[frame_idx]);
        }
    }
    // Keep output aligned to whole frames, even if input had a partial tail.
    let _ = trimmed_len;
    Some(out)
}

/// Streaming time-stretch: processes interleaved audio in chunks and sends
/// each output chunk via the provided sender. Uses a single multi-channel
/// `Stretch` instance, preserving the original channel count (no mono bug).
///
/// The first chunk arrives after just a few milliseconds of processing,
/// enabling instant playback start while the rest streams in the background.
/// Subsequent chunks are appended to the running sink, creating a seamless
/// pitch-preserved speed change with no UI freeze.
pub fn stretch_streaming_interleaved(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    start_frame: usize,
    speed: f32,
    pitch_semitones: f32,
    cancel: &AtomicBool,
    tx: &std::sync::mpsc::Sender<Vec<f32>>,
) -> Option<()> {
    let ch = channels.max(1) as usize;
    let clamped_speed = speed.clamp(0.25, 4.0);

    if samples.is_empty() || sample_rate == 0 {
        return Some(());
    }

    let total_input_frames = samples.len() / ch;
    if start_frame >= total_input_frames {
        return Some(());
    }

    let remaining_input_frames = total_input_frames - start_frame;
    let target_output_frames =
        ((remaining_input_frames as f32) / clamped_speed).round().max(1.0) as usize;

    let mut stretch = Stretch::preset_default(ch as u32, sample_rate);
    stretch.set_transpose_factor_semitones(pitch_semitones, None);

    let input_latency = stretch.input_latency();
    let output_latency = stretch.output_latency();

    // Seek: prime the stretcher with the first `input_latency` frames.
    if input_latency > 0 {
        let seek_end = (start_frame + input_latency).min(total_input_frames);
        let seek_len = seek_end - start_frame;
        let mut seek_input = vec![0.0f32; input_latency * ch];
        if seek_len > 0 {
            let src = &samples[start_frame * ch..seek_end * ch];
            seek_input[..seek_len * ch].copy_from_slice(src);
        }
        stretch.seek(&seek_input, clamped_speed as f64);
    }

    const BLOCK_OUT: usize = 4096;
    let mut in_pos = start_frame + input_latency.min(remaining_input_frames);
    let mut rendered = 0usize;
    let mut skip_front = output_latency;

    let max_in_len = ((BLOCK_OUT as f32) * clamped_speed).ceil().max(1.0) as usize;
    let scratch_in_len = max_in_len.max(input_latency.max(1));
    let mut input_scratch = vec![0.0f32; scratch_in_len * ch];
    let mut output_chunk = vec![0.0f32; BLOCK_OUT * ch];

    while rendered < target_output_frames {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }

        let out_len = (target_output_frames - rendered).min(BLOCK_OUT);
        let in_len = ((out_len as f32) * clamped_speed).ceil().max(1.0) as usize;

        if in_pos + in_len <= total_input_frames {
            let input_chunk = &samples[in_pos * ch..(in_pos + in_len) * ch];
            in_pos += in_len;
            stretch.process(input_chunk, &mut output_chunk[..out_len * ch]);
        } else {
            let available = total_input_frames.saturating_sub(in_pos).min(in_len);
            if available > 0 {
                let src = &samples[in_pos * ch..(in_pos + available) * ch];
                input_scratch[..available * ch].copy_from_slice(src);
                in_pos += available;
            }
            if available < in_len {
                input_scratch[available * ch..in_len * ch].fill(0.0);
            }
            stretch.process(&input_scratch[..in_len * ch], &mut output_chunk[..out_len * ch]);
        }

        let skip_frames = skip_front.min(out_len);
        skip_front -= skip_frames;
        let usable = &output_chunk[skip_frames * ch..out_len * ch];
        if !usable.is_empty() {
            if tx.send(usable.to_vec()).is_err() {
                return None;
            }
        }
        rendered += out_len;
    }

    // Flush remaining output latency
    if output_latency > 0 {
        let mut flushed = 0usize;
        while flushed < output_latency {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let len = (output_latency - flushed).min(BLOCK_OUT);
            stretch.flush(&mut output_chunk[..len * ch]);
            let skip_frames = skip_front.min(len);
            skip_front -= skip_frames;
            let usable = &output_chunk[skip_frames * ch..len * ch];
            if !usable.is_empty() {
                if tx.send(usable.to_vec()).is_err() {
                    return None;
                }
            }
            flushed += len;
        }
    }

    Some(())
}

pub fn get_ffmpeg_command() -> std::process::Command {
    let ffmpeg_exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    // 1. Try next to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let local_ffmpeg = parent.join(ffmpeg_exe);
            if local_ffmpeg.exists() {
                return std::process::Command::new(local_ffmpeg);
            }
        }
    }

    // 2. Try current working directory
    let local_ffmpeg = std::path::PathBuf::from(ffmpeg_exe);
    if local_ffmpeg.exists() {
        return std::process::Command::new(local_ffmpeg);
    }

    // 3. Fallback to system PATH
    std::process::Command::new(ffmpeg_exe)
}

/// Locate the `ffprobe` executable using the same resolution order as
/// `get_ffmpeg_command` (next to exe → cwd → PATH).
pub fn get_ffprobe_command() -> std::process::Command {
    let ffprobe_exe = if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let local = parent.join(ffprobe_exe);
            if local.exists() {
                return std::process::Command::new(local);
            }
        }
    }

    let local = std::path::PathBuf::from(ffprobe_exe);
    if local.exists() {
        return std::process::Command::new(local);
    }

    std::process::Command::new(ffprobe_exe)
}

/// Probed video stream properties needed for accurate audio/video sync.
#[derive(Debug, Clone)]
pub struct VideoStreamInfo {
    /// Average frames per second as a floating-point value. For fractional
    /// rates (e.g. 30000/1001 ≈ 29.97) the exact quotient is returned.
    pub fps: f64,
    /// Total duration in seconds, if known.
    pub duration_sec: Option<f64>,
}

/// Probe a video file for its frame rate and duration using `ffprobe`.
///
/// Returns `None` if ffprobe is unavailable or the file cannot be parsed.
/// The caller should fall back to a sane default (30 fps) in that case.
pub fn probe_video_stream(path: &std::path::Path) -> Option<VideoStreamInfo> {
    let mut cmd = get_ffprobe_command();
    cmd.args([
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        &path.to_string_lossy(),
    ])
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }

    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines();

    // ffprobe outputs values in declaration order: avg_frame_rate,
    // r_frame_rate, stream duration, format duration.
    let avg_fps = parse_fps_line(lines.next());
    let r_fps = parse_fps_line(lines.next());
    let stream_dur = lines.next().and_then(parse_duration_line);
    let format_dur = lines.next().and_then(parse_duration_line);

    // Prefer r_frame_rate (the actual base frame rate) for PTS math; fall
    // back to avg_frame_rate if it is missing or zero.
    let fps = r_fps.or(avg_fps).filter(|f| *f > 0.0).unwrap_or(30.0);
    let duration_sec = stream_dur.or(format_dur);

    Some(VideoStreamInfo { fps, duration_sec })
}

fn parse_fps_line(line: Option<&str>) -> Option<f64> {
    let line = line?.trim();
    if line.is_empty() || line == "N/A" {
        return None;
    }
    if let Some((num, den)) = line.split_once('/') {
        let n: f64 = num.trim().parse().ok()?;
        let d: f64 = den.trim().parse().ok()?;
        if d == 0.0 {
            return None;
        }
        return Some(n / d);
    }
    line.parse::<f64>().ok()
}

fn parse_duration_line(line: &str) -> Option<f64> {
    let line = line.trim();
    if line.is_empty() || line == "N/A" {
        return None;
    }
    line.parse::<f64>().ok()
}
