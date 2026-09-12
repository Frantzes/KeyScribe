use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};

const MEL_MODEL: &str = "mel_spectrogram.onnx";
const BEAT_MODEL: &str = "beat_this_small.onnx";

#[derive(Debug, Clone)]
pub struct BeatResult {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

impl From<beat_this::BeatAnalysis> for BeatResult {
    fn from(a: beat_this::BeatAnalysis) -> Self {
        Self {
            beats: a.beats,
            downbeats: a.downbeats,
        }
    }
}

pub fn create_tracker() -> Result<beat_this::BeatThis<impl beat_this::Model>> {
    let mel_path = resolve_model_path(MEL_MODEL)
        .ok_or_else(|| anyhow!("Could not find {MEL_MODEL}. Download it from https://github.com/danigb/beat-this-rs"))?;
    let beat_path = resolve_model_path(BEAT_MODEL)
        .ok_or_else(|| anyhow!("Could not find {BEAT_MODEL}. Download it from https://github.com/danigb/beat-this-rs"))?;
    beat_this::BeatThis::new(&beat_this::RtenRuntime, &mel_path, &beat_path)
        .context("Failed to initialize beat-this tracker")
}

fn resolve_model_path(filename: &str) -> Option<PathBuf> {
    crate::platform::resolve_model_path(filename)
}
