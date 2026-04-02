#![allow(dead_code)]

use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;

/// Configuration for Spotify Basic Pitch ONNX inference.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to ONNX model file.
    pub model_path: String,
    /// Model input size in mono samples.
    pub input_samples: usize,
    /// Number of MIDI notes (A0..C8).
    pub num_notes: usize,
    /// Number of frame steps produced by the model per window.
    pub output_frames: usize,
    /// Model operating sample rate.
    pub model_sample_rate: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: "models/basic-pitch.onnx".to_string(),
            input_samples: 43_844,
            num_notes: 88,
            output_frames: 172,
            model_sample_rate: 22_050,
        }
    }
}

/// Basic Pitch ONNX inference engine.
pub struct BasicPitchInference {
    config: InferenceConfig,
    session: Session,
    input_name: String,
}

impl BasicPitchInference {
    /// Create a new Basic Pitch inference engine.
    pub fn new(config: InferenceConfig) -> Result<Self> {
        if !Path::new(&config.model_path).exists() {
            return Err(anyhow!(
                "Basic Pitch ONNX model not found at: {}",
                config.model_path
            ));
        }

        let session = Session::builder()?.commit_from_file(&config.model_path)?;
        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("ONNX model has no inputs"))?
            .name()
            .to_string();

        Ok(Self {
            config,
            session,
            input_name,
        })
    }

    /// Infer note probabilities for a single Basic Pitch window.
    ///
    /// Returns shape (output_frames, 88 notes).
    pub fn infer_audio_window(&mut self, audio_window: &[f32]) -> Result<Vec<Vec<f32>>> {
        let prepared = Self::prepare_audio_window(audio_window, self.config.input_samples);

        let input_tensor = Tensor::from_array((
            [1usize, self.config.input_samples, 1usize],
            prepared.into_boxed_slice(),
        ))?;

        let outputs = self
            .session
            .run(ort::inputs! { self.input_name.as_str() => input_tensor })?;

        // Basic Pitch exports two 88-note heads (note/onset) and one 264-bin contour head.
        // To be robust across export variants, we merge every (1, frames, 88) output via max().
        let mut note_heads: Vec<Vec<f32>> = Vec::new();
        for (_, output) in &outputs {
            let arr = output.try_extract_array::<f32>()?;
            let shape = arr.shape();
            if shape.len() == 3
                && shape[0] == 1
                && shape[1] == self.config.output_frames
                && shape[2] == self.config.num_notes
            {
                let mut flat = vec![0.0f32; self.config.output_frames * self.config.num_notes];
                for t in 0..self.config.output_frames {
                    for n in 0..self.config.num_notes {
                        flat[t * self.config.num_notes + n] = arr[[0, t, n]];
                    }
                }
                note_heads.push(flat);
            }
        }

        if note_heads.is_empty() {
            let output_names: Vec<String> = outputs.keys().map(|name| name.to_string()).collect();
            return Err(anyhow!(
                "Basic Pitch outputs did not include any (1, {}, {}) tensors. Outputs: {:?}",
                self.config.output_frames,
                self.config.num_notes,
                output_names
            ));
        }

        let mut note_probs = vec![vec![0.0f32; self.config.num_notes]; self.config.output_frames];
        for t in 0..self.config.output_frames {
            for n in 0..self.config.num_notes {
                let idx = t * self.config.num_notes + n;
                let mut v = 0.0f32;
                for head in &note_heads {
                    v = v.max(head[idx]);
                }
                note_probs[t][n] = v.clamp(0.0, 1.0);
            }
        }

        Ok(note_probs)
    }

    /// Resample with linear interpolation.
    pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if samples.is_empty() || src_rate == 0 || dst_rate == 0 {
            return Vec::new();
        }
        if src_rate == dst_rate {
            return samples.to_vec();
        }

        let ratio = dst_rate as f64 / src_rate as f64;
        let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
        let mut out = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let src_pos = (i as f64) / ratio;
            let idx0 = src_pos.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = (src_pos - idx0 as f64) as f32;

            let s0 = samples[idx0];
            let s1 = samples[idx1];
            out.push(s0 + (s1 - s0) * frac);
        }

        out
    }

    /// Pad or trim an audio window to the exact model input length.
    pub fn prepare_audio_window(samples: &[f32], target_len: usize) -> Vec<f32> {
        if samples.len() >= target_len {
            samples[..target_len].to_vec()
        } else {
            let mut out = Vec::with_capacity(target_len);
            out.extend_from_slice(samples);
            out.resize(target_len, 0.0);
            out
        }
    }

    /// Apply confidence threshold to predictions.
    pub fn threshold_predictions(note_probs: &[Vec<f32>], threshold: f32) -> Vec<Vec<bool>> {
        note_probs
            .iter()
            .map(|frame| frame.iter().map(|&p| p >= threshold).collect())
            .collect()
    }

    /// Get confidence-weighted note indices for each frame.
    pub fn get_active_notes(note_probs: &[Vec<f32>], threshold: f32) -> Vec<Vec<(usize, f32)>> {
        note_probs
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &prob)| {
                        if prob >= threshold {
                            Some((idx, prob))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_audio_window() {
        let src = vec![1.0f32, 2.0, 3.0];
        let padded = BasicPitchInference::prepare_audio_window(&src, 6);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);

        let trimmed = BasicPitchInference::prepare_audio_window(&src, 2);
        assert_eq!(trimmed, vec![1.0, 2.0]);
    }

    #[test]
    fn test_threshold_predictions() {
        let probs = vec![vec![0.1, 0.7, 0.2], vec![0.5, 0.3, 0.8]];

        let predictions = BasicPitchInference::threshold_predictions(&probs, 0.5);
        assert_eq!(predictions[0], vec![false, true, false]);
        assert_eq!(predictions[1], vec![true, false, true]);
    }

    #[test]
    fn test_get_active_notes() {
        let probs = vec![vec![0.1, 0.7, 0.2], vec![0.5, 0.3, 0.8]];

        let active = BasicPitchInference::get_active_notes(&probs, 0.5);
        assert_eq!(active[0].len(), 1);
        assert_eq!(active[0][0].0, 1); // Note index
        assert!((active[0][0].1 - 0.7).abs() < 1e-5);
    }
}
