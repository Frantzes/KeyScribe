use std::time::Instant;

use anyhow::{Context, Result};
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, OutputStreamHandle, Sink};

pub struct AudioEngine {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Option<Sink>,
    started_at: Option<Instant>,
    start_pos_sec: f32,
    is_playing: bool,
    duration_sec: f32,
    volume: f32,
}

impl AudioEngine {
    pub fn new() -> Result<Self> {
        let (stream, stream_handle) =
            OutputStream::try_default().context("No audio output device found")?;
        Ok(Self {
            _stream: stream,
            stream_handle,
            sink: None,
            started_at: None,
            start_pos_sec: 0.0,
            is_playing: false,
            duration_sec: 0.0,
            volume: 0.8,
        })
    }

    pub fn play_from(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        start_pos_sec: f32,
    ) -> Result<()> {
        self.play_range(samples, sample_rate, start_pos_sec, None)
    }

    pub fn play_range(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        start_pos_sec: f32,
        end_pos_sec: Option<f32>,
    ) -> Result<()> {
        self.stop();

        if samples.is_empty() || sample_rate == 0 {
            return Ok(());
        }

        let start_idx = (start_pos_sec.max(0.0) * sample_rate as f32) as usize;
        if start_idx >= samples.len() {
            return Ok(());
        }

        let mut end_idx = samples.len();
        if let Some(end) = end_pos_sec {
            end_idx = ((end.max(start_pos_sec) * sample_rate as f32) as usize).min(samples.len());
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

        self.duration_sec = end_idx as f32 / sample_rate as f32;
        self.start_pos_sec = start_pos_sec;
        self.started_at = Some(Instant::now());
        self.is_playing = true;
        self.sink = Some(sink);

        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(s) = self.sink.take() {
            s.stop();
        }
        self.is_playing = false;
        self.started_at = None;
        self.start_pos_sec = 0.0;
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
                let pos = self.start_pos_sec + t0.elapsed().as_secs_f32();
                return pos.min(self.duration_sec.max(self.start_pos_sec));
            }
        }
        self.start_pos_sec
    }

    pub fn sync_finished(&mut self) {
        if let Some(s) = &self.sink {
            if s.empty() {
                self.is_playing = false;
                self.started_at = None;
                self.start_pos_sec = self.duration_sec;
            }
        }
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.5);
        if let Some(s) = &self.sink {
            s.set_volume(self.volume);
        }
    }

}
