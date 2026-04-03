use std::time::Instant;

use anyhow::{bail, Context, Result};
use rodio::buffer::SamplesBuffer;
use rodio::cpal::traits::{DeviceTrait, HostTrait};
use rodio::cpal::Device;
use rodio::{OutputStream, OutputStreamHandle, Sink};

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
        })
    }

    pub fn play_from(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        start_pos_sec: f32,
        timeline_rate: f32,
    ) -> Result<()> {
        self.play_range(samples, sample_rate, start_pos_sec, None, timeline_rate)
    }

    pub fn play_range(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        start_pos_sec: f32,
        end_pos_sec: Option<f32>,
        timeline_rate: f32,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);

        let start_idx = ((start_pos_sec.max(0.0) / timeline_rate) * sample_rate as f32) as usize;
        if start_idx >= samples.len() {
            return Ok(());
        }

        let mut end_idx = samples.len();
        if let Some(end) = end_pos_sec {
            end_idx = (((end.max(start_pos_sec) / timeline_rate) * sample_rate as f32) as usize)
                .min(samples.len());
        }

        if end_idx <= start_idx {
            return Ok(());
        }

        let sink = Sink::try_new(&self.stream_handle).context("Failed to create playback sink")?;
        sink.set_volume(self.volume.clamp(0.0, 1.5));
        let data = samples[start_idx..end_idx].to_vec();
        let source = SamplesBuffer::new(1, sample_rate, data);

        sink.append(source);
        sink.play();

        self.duration_sec = if let Some(end) = end_pos_sec {
            end.max(start_pos_sec)
        } else {
            start_pos_sec + ((end_idx - start_idx) as f32 / sample_rate as f32) * timeline_rate
        };
        self.start_pos_sec = start_pos_sec;
        self.timeline_rate = timeline_rate;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);

        Ok(())
    }

    pub fn play_chunk_at_timeline(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        timeline_start_sec: f32,
        timeline_rate: f32,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        let sink = Sink::try_new(&self.stream_handle).context("Failed to create playback sink")?;
        sink.set_volume(self.volume.clamp(0.0, 1.5));
        let data = samples.to_vec();
        let source = SamplesBuffer::new(1, sample_rate, data);

        sink.append(source);
        sink.play();

        let duration = samples.len() as f32 / sample_rate as f32;
        self.start_pos_sec = timeline_start_sec.max(0.0);
        self.duration_sec = self.start_pos_sec + duration * timeline_rate;
        self.timeline_rate = timeline_rate;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);

        Ok(())
    }

    pub fn has_active_sink(&self) -> bool {
        self.sink.is_some()
    }

    pub fn append_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        timeline_rate: f32,
    ) -> Result<()> {
        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let Some(sink) = self.sink.as_ref() else {
            return Ok(());
        };

        let timeline_rate = timeline_rate.clamp(0.25, 4.0);
        sink.append(SamplesBuffer::new(1, sample_rate, samples.to_vec()));
        self.duration_sec += (samples.len() as f32 / sample_rate as f32) * timeline_rate;
        self.timeline_rate = timeline_rate;

        // If playback reached queue end and new audio arrives, resume timeline tracking.
        if !self.is_playing && !sink.is_paused() {
            self.started_at = Some(Instant::now());
            self.is_playing = true;
        }

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
    }

    pub fn pause(&mut self) {
        if !self.is_playing {
            return;
        }
        if let Some(s) = &self.sink {
            s.pause();
        }
        self.start_pos_sec = self.current_position();
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

    pub fn current_position(&self) -> f32 {
        if self.is_playing {
            if let Some(t0) = self.started_at {
                let pos = self.start_pos_sec + t0.elapsed().as_secs_f32() * self.timeline_rate;
                return pos.min(self.duration_sec.max(self.start_pos_sec));
            }
        }
        self.start_pos_sec
    }

    pub fn sync_finished(&mut self) {
        let is_empty = self.sink.as_ref().map(|s| s.empty()).unwrap_or(false);
        if is_empty {
            self.sink = None;
            self.is_playing = false;
            self.started_at = None;
            self.start_pos_sec = self.duration_sec;
        }
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.5);
        if let Some(s) = &self.sink {
            s.set_volume(self.volume);
        }
    }
}
