#![allow(dead_code)]

use crate::inference::{BasicPitchInference, InferenceConfig};
use crate::preprocessing::PreprocessingConfig;
use crate::viterbi::{ViterbiConfig, ViterbiDecoder};
use anyhow::Result;

/// Result from inference - note probabilities for a frame
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub timestamp_ms: f32,
    pub note_probs: Vec<f32>,            // Raw probabilities from model
    pub note_activations: Vec<bool>,     // Binary on/off (thresholded)
    pub active_notes: Vec<(usize, f32)>, // (MIDI note index, confidence)
}

/// Configuration for the entire pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub sample_rate: u32,
    pub chunk_size: usize,       // 200-500ms chunks
    pub lookahead_frames: usize, // For Viterbi smoothing
    pub preprocessing: PreprocessingConfig,
    pub viterbi: ViterbiConfig,
    pub model_path: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            chunk_size: 4410, // 100ms at 44.1kHz
            lookahead_frames: 5,
            preprocessing: PreprocessingConfig::default(),
            viterbi: ViterbiConfig::default(),
            model_path: "models/basic-pitch.onnx".to_string(),
        }
    }
}

/// Simplified pipeline result for file-based processing
pub struct PipelineResult {
    pub note_probs_sequence: Vec<Vec<f32>>,
    pub smoothed_notes: Vec<Vec<bool>>,
}

/// Simplified pipeline for file-based transcription
/// Processes audio through HPSS + Viterbi smoothing pipeline
pub struct AudioPipeline {
    config: PipelineConfig,
}

impl AudioPipeline {
    /// Create a new audio pipeline
    pub fn new(config: PipelineConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Process audio file with full CQT + HPSS + Viterbi pipeline
    pub fn process_audio(&self, samples: &[f32]) -> Result<PipelineResult> {
        if samples.is_empty() {
            return Ok(PipelineResult {
                note_probs_sequence: vec![],
                smoothed_notes: vec![],
            });
        }

        let mut inference = BasicPitchInference::new(InferenceConfig {
            model_path: self.config.model_path.clone(),
            ..Default::default()
        })?;
        let viterbi = ViterbiDecoder::new(self.config.viterbi.clone());

        // Basic Pitch expects mono 22.05kHz windows of exactly 43,844 samples.
        let model_sr = inference.config().model_sample_rate;
        let model_input_samples = inference.config().input_samples;
        let model_output_frames = inference.config().output_frames;
        let overlap_hop = (model_input_samples / 2).max(1);
        let overlap_output = (model_output_frames / 2).max(1);

        let model_samples =
            BasicPitchInference::resample_linear(samples, self.config.sample_rate, model_sr);
        if model_samples.is_empty() {
            return Ok(PipelineResult {
                note_probs_sequence: vec![],
                smoothed_notes: vec![],
            });
        }

        let mut note_probs_sequence = Vec::new();
        let mut start = 0usize;
        let mut first = true;

        loop {
            let end = (start + model_input_samples).min(model_samples.len());
            let window = &model_samples[start..end];
            let mut window_probs = inference.infer_audio_window(window)?;

            if first {
                note_probs_sequence.append(&mut window_probs);
                first = false;
            } else {
                let keep_from = overlap_output.min(window_probs.len());
                note_probs_sequence.extend(window_probs.into_iter().skip(keep_from));
            }

            if end >= model_samples.len() {
                break;
            }

            start += overlap_hop;
            if start >= model_samples.len() {
                break;
            }
        }

        if note_probs_sequence.is_empty() {
            return Ok(PipelineResult {
                note_probs_sequence,
                smoothed_notes: vec![],
            });
        }

        // Viterbi post-processing for smoothing and ghost note elimination.
        let smoothed = viterbi.decode(&note_probs_sequence, self.config.lookahead_frames);

        // Apply temporal smoothing to enforce minimum note duration.
        let smoothed = viterbi.apply_temporal_smoothing(&smoothed, 2); // Minimum 2 frames

        Ok(PipelineResult {
            note_probs_sequence,
            smoothed_notes: smoothed,
        })
    }

    /// Get pipeline config
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Create a Viterbi decoder for post-processing
    pub fn create_viterbi_decoder(&self) -> ViterbiDecoder {
        ViterbiDecoder::new(self.config.viterbi.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = AudioPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_process_empty_audio() {
        let config = PipelineConfig::default();
        let pipeline = AudioPipeline::new(config).unwrap();
        let samples = vec![0.0; 0];
        let result = pipeline.process_audio(&samples);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.note_probs_sequence.len(), 0);
    }

    #[test]
    fn test_process_simple_audio() {
        if !Path::new("models/basic-pitch.onnx").exists() {
            return;
        }

        let config = PipelineConfig::default();
        let pipeline = AudioPipeline::new(config).unwrap();
        // Create 1 second of audio at 44.1kHz
        let samples = vec![0.1; 44100];
        let result = pipeline.process_audio(&samples);
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should have some frames
        assert!(result.note_probs_sequence.len() > 0);
        // Each frame should have 88 notes (piano keys)
        for frame in &result.note_probs_sequence {
            assert_eq!(frame.len(), 88);
        }
    }
}
