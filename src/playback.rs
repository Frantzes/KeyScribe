use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use rodio::buffer::SamplesBuffer;
use rodio::cpal::traits::{DeviceTrait, HostTrait};
use rodio::cpal::Device;
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};

/// A point-in-time snapshot of the master audio clock.
///
/// This is the single source of truth for playback position. All consumers
/// (video, keyboard transcription, waveform playhead) must read from this
/// snapshot rather than independently estimating time, so they can never
/// drift apart — the same approach VLC uses with its audio-master clock.
#[derive(Debug, Clone, Copy)]
pub struct ClockSnapshot {
    /// Position in timeline seconds, already compensated for output latency
    /// so it reflects the sample currently *audible* through the speakers,
    /// not the sample merely queued into the device buffer.
    pub position_sec: f32,
    /// Configured playback rate (speed factor applied to the timeline).
    pub timeline_rate: f32,
    /// Whether the engine is actively producing sound.
    pub is_playing: bool,
    /// Wall-clock instant at which this snapshot was captured.
    pub captured_at: Instant,
    /// Estimated output device latency (seconds) baked into `position_sec`.
    pub latency_sec: f32,
}

impl ClockSnapshot {
    /// A neutral snapshot representing a stopped clock at the given position.
    pub fn stopped(position_sec: f32) -> Self {
        Self {
            position_sec,
            timeline_rate: 1.0,
            is_playing: false,
            captured_at: Instant::now(),
            latency_sec: 0.0,
        }
    }
}

// Stream samples from a shared buffer without copying.
struct ArcSamplesSource {
    samples: Arc<Vec<f32>>,
    idx: usize,
    start: usize,
    end: usize,
    channels: u16,
    sample_rate: u32,
    fade_frames: usize,
    fade_in: bool,
    fade_out: bool,
    consumed: Arc<AtomicUsize>,
}

impl ArcSamplesSource {
    fn new(
        samples: Arc<Vec<f32>>,
        start_idx: usize,
        end_idx: usize,
        channels: u16,
        sample_rate: u32,
        consumed: Arc<AtomicUsize>,
    ) -> Self {
        let fade_frames = (sample_rate as f32 * 0.005) as usize; // 5ms fade
        Self {
            samples,
            idx: start_idx,
            start: start_idx,
            end: end_idx,
            channels,
            sample_rate,
            fade_frames,
            fade_in: true,
            fade_out: true,
            consumed,
        }
    }

    /// Create a source with explicit fade control. Used for streaming
    /// time-stretch where chunk boundaries must be seamless (no fades)
    /// to prevent amplitude modulation artifacts.
    fn new_with_fades(
        samples: Arc<Vec<f32>>,
        start_idx: usize,
        end_idx: usize,
        channels: u16,
        sample_rate: u32,
        consumed: Arc<AtomicUsize>,
        fade_in: bool,
        fade_out: bool,
    ) -> Self {
        let fade_frames = (sample_rate as f32 * 0.005) as usize; // 5ms fade
        Self {
            samples,
            idx: start_idx,
            start: start_idx,
            end: end_idx,
            channels,
            sample_rate,
            fade_frames,
            fade_in,
            fade_out,
            consumed,
        }
    }
}

impl Iterator for ArcSamplesSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.end {
            return None;
        }

        self.consumed.fetch_add(1, Ordering::Relaxed);

        let mut value = self.samples[self.idx];
        
        let ch = self.channels.max(1) as usize;
        let frame_idx = (self.idx - self.start) / ch;
        let total_frames = (self.end - self.start) / ch;
        
        if self.fade_in && frame_idx < self.fade_frames {
            let fade = frame_idx as f32 / self.fade_frames as f32;
            value *= fade;
        } else if self.fade_out && total_frames.saturating_sub(frame_idx) <= self.fade_frames {
            let remain = total_frames.saturating_sub(frame_idx);
            let fade = remain as f32 / self.fade_frames as f32;
            value *= fade;
        }

        self.idx += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}

impl Source for ArcSamplesSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        self.channels.max(1)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        let channels = self.channels.max(1) as usize;
        let frames = self.end.saturating_sub(self.idx) / channels;
        let denom = self.sample_rate.max(1) as f32;
        Some(Duration::from_secs_f32(frames as f32 / denom))
    }
}

/// Default estimated output latency used when no device-specific value is
/// available. cpal/rodio typically buffer ~1-3 periods of ~10-30 ms each, so
/// 30 ms is a conservative middle ground. This keeps the master clock
/// tracking the audible signal rather than the queued one.
const DEFAULT_OUTPUT_LATENCY_SEC: f32 = 0.030;

#[derive(Debug, Clone)]
pub struct AudioOutputDeviceOption {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

fn collect_output_devices() -> Vec<(String, Device)> {
    let host = rodio::cpal::default_host();
    let Ok(devices) = host.output_devices() else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for device in devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "Unknown output device".to_string());
        out.push((name, device));
    }
    out
}

pub fn available_output_devices() -> Vec<AudioOutputDeviceOption> {
    let host = rodio::cpal::default_host();
    let default_name = host
        .default_output_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    collect_output_devices()
        .into_iter()
        .enumerate()
        .map(|(idx, (name, _device))| AudioOutputDeviceOption {
            id: format!("{idx}:{name}"),
            is_default: !default_name.is_empty() && name == default_name,
            name,
        })
        .collect()
}

fn open_output_stream(device_id: Option<&str>) -> Result<(OutputStream, OutputStreamHandle)> {
    if let Some(id) = device_id {
        for (idx, (name, device)) in collect_output_devices().into_iter().enumerate() {
            let current_id = format!("{idx}:{name}");
            if current_id == id {
                return OutputStream::try_from_device(&device)
                    .context("Failed to open selected audio output device");
            }
        }
        bail!("Selected output device is no longer available");
    }

    OutputStream::try_default().context("No audio output device found")
}

pub struct AudioEngine {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Option<Sink>,
    started_at: Option<Instant>,
    start_pos_sec: f32,
    is_playing: bool,
    duration_sec: f32,
    timeline_rate: f32,
    volume: f32,
    consumed_samples: Option<Arc<AtomicUsize>>,
    playback_channels: u16,
    playback_sample_rate: u32,
    /// Estimated output device latency in seconds. The master clock subtracts
    /// this from the raw consumed-sample position so that `position_sec`
    /// tracks the sample currently *audible*, not the sample merely pulled
    /// into the device buffer. This is the single biggest source of audio/
    /// video and audio/keyboard drift, and compensating for it mirrors how
    /// VLC aligns its clock to the audible audio endpoint.
    output_latency_sec: f32,
}

impl AudioEngine {
    pub fn new() -> Result<Self> {
        Self::new_with_output_device(None)
    }

    pub fn new_with_output_device(device_id: Option<&str>) -> Result<Self> {
        let (stream, stream_handle) = open_output_stream(device_id)?;
        Ok(Self {
            _stream: stream,
            stream_handle,
            sink: None,
            started_at: None,
            start_pos_sec: 0.0,
            is_playing: false,
            duration_sec: 0.0,
            timeline_rate: 1.0,
            volume: 0.8,
            consumed_samples: None,
            playback_channels: 1,
            playback_sample_rate: 44100,
            output_latency_sec: DEFAULT_OUTPUT_LATENCY_SEC,
        })
    }

    pub fn play_from(
        &mut self,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        start_pos_sec: f32,
        timeline_rate: f32,
    ) -> Result<()> {
        self.play_range(
            samples,
            channels,
            sample_rate,
            start_pos_sec,
            None,
            timeline_rate,
        )
    }

    pub fn play_range(
        &mut self,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        start_pos_sec: f32,
        end_pos_sec: Option<f32>,
        timeline_rate: f32,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let channels = channels.max(1);
        let channels_usize = channels as usize;
        let frame_count = samples.len() / channels_usize;
        if frame_count == 0 {
            return Ok(());
        }

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);

        let start_frame = ((start_pos_sec.max(0.0) / timeline_rate) * sample_rate as f32) as usize;
        if start_frame >= frame_count {
            return Ok(());
        }
        let start_idx = start_frame * channels_usize;

        let mut end_frame = frame_count;
        if let Some(end) = end_pos_sec {
            end_frame = (((end.max(start_pos_sec) / timeline_rate) * sample_rate as f32) as usize)
                .min(frame_count);
        }
        let end_idx = end_frame * channels_usize;

        if end_idx <= start_idx {
            return Ok(());
        }

        let sink = Sink::try_new(&self.stream_handle).context("Failed to create playback sink")?;
        sink.set_volume(self.volume.clamp(0.0, 1.5));
        let data = samples[start_idx..end_idx].to_vec();
        let source = SamplesBuffer::new(channels, sample_rate, data);

        sink.append(source);
        sink.play();

        self.duration_sec = if let Some(end) = end_pos_sec {
            end.max(start_pos_sec)
        } else {
            let played_frames = end_frame.saturating_sub(start_frame);
            start_pos_sec + (played_frames as f32 / sample_rate as f32) * timeline_rate
        };
        self.start_pos_sec = start_pos_sec;
        self.timeline_rate = timeline_rate;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);

        Ok(())
    }

    pub fn play_arc_range(
        &mut self,
        samples: Arc<Vec<f32>>,
        channels: u16,
        sample_rate: u32,
        start_pos_sec: f32,
        end_pos_sec: Option<f32>,
        timeline_rate: f32,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let channels = channels.max(1);
        let channels_usize = channels as usize;
        let frame_count = samples.len() / channels_usize;
        if frame_count == 0 {
            return Ok(());
        }

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        let start_frame = ((start_pos_sec.max(0.0) / timeline_rate) * sample_rate as f32) as usize;
        if start_frame >= frame_count {
            return Ok(());
        }
        let start_idx = start_frame * channels_usize;

        let mut end_frame = frame_count;
        if let Some(end) = end_pos_sec {
            end_frame = (((end.max(start_pos_sec) / timeline_rate) * sample_rate as f32) as usize)
                .min(frame_count);
        }
        let end_idx = end_frame * channels_usize;

        if end_idx <= start_idx {
            return Ok(());
        }

        let sink = Sink::try_new(&self.stream_handle).context("Failed to create playback sink")?;
        sink.set_volume(self.volume.clamp(0.0, 1.5));
        let consumed = Arc::new(AtomicUsize::new(0));
        let source = ArcSamplesSource::new(
            samples,
            start_idx,
            end_idx,
            channels,
            sample_rate,
            Arc::clone(&consumed),
        );

        sink.append(source);
        sink.play();

        self.duration_sec = if let Some(end) = end_pos_sec {
            end.max(start_pos_sec)
        } else {
            let played_frames = end_frame.saturating_sub(start_frame);
            start_pos_sec + (played_frames as f32 / sample_rate as f32) * timeline_rate
        };
        self.start_pos_sec = start_pos_sec;
        self.timeline_rate = timeline_rate;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);
        self.consumed_samples = Some(consumed);
        self.playback_channels = channels;
        self.playback_sample_rate = sample_rate;

        Ok(())
    }

    pub fn play_chunk_at_timeline(
        &mut self,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        timeline_start_sec: f32,
        timeline_rate: f32,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let channels = channels.max(1);
        let channels_usize = channels as usize;
        let frame_count = samples.len() / channels_usize;
        if frame_count == 0 {
            return Ok(());
        }

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        let sink = Sink::try_new(&self.stream_handle).context("Failed to create playback sink")?;
        sink.set_volume(self.volume.clamp(0.0, 1.5));
        let consumed = Arc::new(AtomicUsize::new(0));
        let data = Arc::new(samples[..frame_count * channels_usize].to_vec());
        let total = data.len();
        let source = ArcSamplesSource::new(
            data,
            0,
            total,
            channels,
            sample_rate,
            Arc::clone(&consumed),
        );

        sink.append(source);
        sink.play();

        let duration = frame_count as f32 / sample_rate as f32;
        self.start_pos_sec = timeline_start_sec.max(0.0);
        self.duration_sec = self.start_pos_sec + duration * timeline_rate;
        self.timeline_rate = timeline_rate;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);
        self.consumed_samples = Some(consumed);
        self.playback_channels = channels;
        self.playback_sample_rate = sample_rate;

        Ok(())
    }

    pub fn has_active_sink(&self) -> bool {
        self.sink.is_some()
    }

    pub fn append_samples(
        &mut self,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        timeline_rate: f32,
    ) -> Result<()> {
        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let channels = channels.max(1);
        let channels_usize = channels as usize;
        let frame_count = samples.len() / channels_usize;
        if frame_count == 0 {
            return Ok(());
        }

        let Some(sink) = self.sink.as_ref() else {
            return Ok(());
        };

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        let data = Arc::new(samples[..frame_count * channels_usize].to_vec());
        let total = data.len();
        let shared_consumed = self
            .consumed_samples
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicUsize::new(0)));
        let source = ArcSamplesSource::new(
            data,
            0,
            total,
            channels,
            sample_rate,
            Arc::clone(&shared_consumed),
        );
        sink.append(source);
        self.duration_sec += (frame_count as f32 / sample_rate as f32) * timeline_rate;
        self.timeline_rate = timeline_rate;
        self.consumed_samples = Some(shared_consumed);
        self.playback_channels = channels;
        self.playback_sample_rate = sample_rate;

        // If playback reached queue end and new audio arrives, resume timeline tracking.
        if !self.is_playing && !sink.is_paused() {
            self.started_at = Some(Instant::now());
            self.is_playing = true;
        }

        Ok(())
    }

    /// Append a chunk of stretched audio to the running sink for streaming
    /// time-stretch. Unlike `append_samples`, this uses NO fades at chunk
    /// boundaries to prevent amplitude modulation artifacts. The consumed
    /// counter stays continuous across chunks for accurate clock tracking.
    pub fn append_streaming_chunk(
        &mut self,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        timeline_rate: f32,
    ) -> Result<()> {
        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let channels = channels.max(1);
        let channels_usize = channels as usize;
        let frame_count = samples.len() / channels_usize;
        if frame_count == 0 {
            return Ok(());
        }

        let Some(sink) = self.sink.as_ref() else {
            return Ok(());
        };

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        let data = Arc::new(samples[..frame_count * channels_usize].to_vec());
        let total = data.len();
        let shared_consumed = self
            .consumed_samples
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicUsize::new(0)));
        // No fade-in or fade-out — seamless chunk boundaries
        let source = ArcSamplesSource::new_with_fades(
            data,
            0,
            total,
            channels,
            sample_rate,
            Arc::clone(&shared_consumed),
            false,
            false,
        );
        sink.append(source);
        self.duration_sec += (frame_count as f32 / sample_rate as f32) * timeline_rate;
        self.timeline_rate = timeline_rate;
        self.consumed_samples = Some(shared_consumed);
        self.playback_channels = channels;
        self.playback_sample_rate = sample_rate;

        // If playback reached queue end and new audio arrives, resume timeline tracking.
        if !self.is_playing && !sink.is_paused() {
            self.started_at = Some(Instant::now());
            self.is_playing = true;
        }

        Ok(())
    }

    /// Start playback of the first streaming chunk. Uses a fade-in to prevent
    /// a click at playback start, but NO fade-out so the next chunk can be
    /// appended seamlessly.
    pub fn play_streaming_first_chunk(
        &mut self,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        timeline_start_sec: f32,
        timeline_rate: f32,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let channels = channels.max(1);
        let channels_usize = channels as usize;
        let frame_count = samples.len() / channels_usize;
        if frame_count == 0 {
            return Ok(());
        }

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        let sink = Sink::try_new(&self.stream_handle).context("Failed to create playback sink")?;
        sink.set_volume(self.volume.clamp(0.0, 1.5));
        let consumed = Arc::new(AtomicUsize::new(0));
        let data = Arc::new(samples[..frame_count * channels_usize].to_vec());
        let total = data.len();
        // Fade-in only, no fade-out — next chunk will be appended seamlessly
        let source = ArcSamplesSource::new_with_fades(
            data,
            0,
            total,
            channels,
            sample_rate,
            Arc::clone(&consumed),
            true,
            false,
        );

        sink.append(source);
        sink.play();

        let duration = frame_count as f32 / sample_rate as f32;
        self.start_pos_sec = timeline_start_sec.max(0.0);
        self.duration_sec = self.start_pos_sec + duration * timeline_rate;
        self.timeline_rate = timeline_rate;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);
        self.consumed_samples = Some(consumed);
        self.playback_channels = channels;
        self.playback_sample_rate = sample_rate;

        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(s) = self.sink.take() {
            s.stop();
        }
        self.is_playing = false;
        self.started_at = None;
        self.start_pos_sec = 0.0;
        self.timeline_rate = 1.0;
        self.consumed_samples = None;
    }

    pub fn pause(&mut self) {
        if !self.is_playing {
            return;
        }
        if let Some(s) = &self.sink {
            s.pause();
        }
        // When using the consumed-samples clock (hardware sample clock), do
        // NOT touch consumed_samples or start_pos_sec — the consumed counter
        // simply stops incrementing while the sink is paused and resumes when
        // playback continues, so the clock stays locked to the hardware with
        // zero drift.
        //
        // Only the wall-time fallback path needs to freeze start_pos_sec.
        // We use raw_position() (not current_position()) because start_pos_sec
        // stores the uncompensated value; using the compensated one would
        // double-subtract latency on resume.
        if self.consumed_samples.is_none() {
            self.start_pos_sec = self.raw_position();
        }
        self.started_at = None;
        self.is_playing = false;
    }

    pub fn resume(&mut self) {
        if self.is_playing {
            return;
        }
        if let Some(s) = &self.sink {
            s.play();
        }
        self.started_at = Some(Instant::now());
        self.is_playing = true;
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    /// Raw position (timeline seconds) of the playback cursor **without**
    /// latency compensation. This counts samples that have been pulled into
    /// the audio device buffer, which is ahead of the audible signal by the
    /// output latency. Use `current_position()` or `clock_snapshot()` for
    /// latency-compensated positions that consumers should display.
    fn raw_position(&self) -> f32 {
        if let Some(ref consumed) = self.consumed_samples {
            let count = consumed.load(Ordering::Relaxed) as f32;
            let channels = self.playback_channels.max(1) as f32;
            let sr = self.playback_sample_rate.max(1) as f32;
            let elapsed = count / channels / sr;
            let pos = self.start_pos_sec + elapsed * self.timeline_rate;
            return pos.min(self.duration_sec.max(self.start_pos_sec));
        }
        if self.is_playing {
            if let Some(t0) = self.started_at {
                let pos = self.start_pos_sec + t0.elapsed().as_secs_f32() * self.timeline_rate;
                return pos.min(self.duration_sec.max(self.start_pos_sec));
            }
        }
        self.start_pos_sec
    }

    /// Latency-compensated master clock position in timeline seconds.
    ///
    /// The output latency is subtracted so this reflects the sample
    /// currently **audible** through the speakers. This is the canonical
    /// position that all visual consumers (video, keyboard, waveform
    /// playhead) must use to stay in sync with what the user hears.
    pub fn current_position(&self) -> f32 {
        let raw = self.raw_position();
        let compensated = raw - self.output_latency_sec * self.timeline_rate;
        // Clamp to the valid timeline range [0, duration]. The raw
        // start_pos_sec is *not* used as the lower bound because it is an
        // uncompensated value — using it here would re-add the latency we
        // just subtracted and defeat the compensation.
        compensated.clamp(0.0, self.duration_sec.max(0.0))
    }

    /// Capture a point-in-time snapshot of the master audio clock.
    ///
    /// This is the single source of truth for all playback-synchronized
    /// consumers. Capturing once per UI frame and distributing the snapshot
    /// guarantees that video, keyboard transcription, and the waveform
    /// playhead can never drift relative to each other — they all read the
    /// same compensated audio time.
    pub fn clock_snapshot(&self) -> ClockSnapshot {
        ClockSnapshot {
            position_sec: self.current_position(),
            timeline_rate: self.timeline_rate,
            is_playing: self.is_playing,
            captured_at: Instant::now(),
            latency_sec: self.output_latency_sec,
        }
    }

    /// Estimated output device latency (seconds) applied to the master clock.
    pub fn output_latency_sec(&self) -> f32 {
        self.output_latency_sec
    }

    /// Override the estimated output latency. Useful for calibration or when
    /// a device reports its own latency. Values are clamped to a sane range.
    pub fn set_output_latency(&mut self, latency_sec: f32) {
        self.output_latency_sec = latency_sec.clamp(0.0, 0.5);
    }

    pub fn sync_finished(&mut self) {
        let is_empty = self.sink.as_ref().map(|s| s.empty()).unwrap_or(false);
        if is_empty {
            self.sink = None;
            self.is_playing = false;
            self.started_at = None;
            self.start_pos_sec = self.duration_sec;
            self.consumed_samples = None;
        }
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.5);
        if let Some(s) = &self.sink {
            s.set_volume(self.volume);
        }
    }
}
