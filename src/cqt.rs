#![allow(dead_code)]

use rustfft::num_complex::Complex;
use std::f32::consts::PI;

/// Constant-Q Transform configuration
/// Each bin represents one semitone across the audio spectrum
#[derive(Debug, Clone)]
pub struct CQTConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Lowest frequency to analyze (Hz)
    pub fmin: f32,
    /// Highest frequency to analyze (Hz)
    pub fmax: f32,
    /// Bins per semitone (default: 1, can increase for finer resolution)
    pub bins_per_semitone: usize,
}

impl CQTConfig {
    /// Create CQT config optimized for piano/guitar (A0 to C8)
    pub fn piano_range(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fmin: 27.5,   // A0
            fmax: 4186.0, // C8
            bins_per_semitone: 1,
        }
    }

    /// Calculate the number of CQT bins
    pub fn num_bins(&self) -> usize {
        let octaves = (self.fmax / self.fmin).log2();
        (octaves * 12.0 * self.bins_per_semitone as f32).ceil() as usize
    }

    /// Get the center frequency for bin k
    pub fn bin_frequency(&self, k: usize) -> f32 {
        self.fmin * 2.0_f32.powf(k as f32 / (12.0 * self.bins_per_semitone as f32))
    }
}

/// Constant-Q Transform engine
pub struct CQTransform {
    config: CQTConfig,
    // Precomputed kernels for each bin
    kernels: Vec<Vec<Complex<f32>>>,
}

impl CQTransform {
    /// Create a new CQT engine
    pub fn new(config: CQTConfig) -> Self {
        let kernels = Self::compute_kernels(&config);
        Self { config, kernels }
    }

    /// Compute CQT kernels (precomputed Gabor wavelets)
    fn compute_kernels(config: &CQTConfig) -> Vec<Vec<Complex<f32>>> {
        let num_bins = config.num_bins();
        let sample_rate = config.sample_rate as f32;
        let mut kernels = Vec::with_capacity(num_bins);

        for k in 0..num_bins {
            let fk = config.bin_frequency(k);
            // Quality factor Q = fk / bandwidth
            // For music: Q ≈ 12-20 (higher Q = narrower filter, better frequency resolution)
            let q = 20.0;
            let bandwidth = fk / q;

            // Kernel length in samples (3 periods of the lowest frequency)
            let kernel_len = ((3.0 * sample_rate) / fk).ceil() as usize;

            let mut kernel = vec![Complex::new(0.0, 0.0); kernel_len];

            for n in 0..kernel_len {
                let t = n as f32 / sample_rate;
                // Gaussian window
                let alpha = 2.0 * PI * bandwidth;
                let window = (-alpha * t * t).exp();
                // Complex exponential at fk
                let phase = 2.0 * PI * fk * t;
                let complex_exp = Complex::new(phase.cos(), phase.sin());

                kernel[n] = window * complex_exp;
            }

            kernels.push(kernel);
        }

        kernels
    }

    /// Compute CQT from input samples
    /// Returns magnitude spectrogram of shape (num_bins, num_frames)
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let num_bins = self.config.num_bins();
        let num_frames = samples.len();

        let mut result = vec![vec![0.0; num_frames]; num_bins];

        for (k, kernel) in self.kernels.iter().enumerate() {
            let kernel_len = kernel.len();
            if samples.len() < kernel_len {
                continue;
            }

            // Convolve samples with kernel
            for n in 0..(samples.len() - kernel_len) {
                let mut sum = Complex::new(0.0, 0.0);
                for m in 0..kernel_len {
                    sum += samples[n + m] * kernel[m];
                }

                // Normalize by kernel energy
                let kernel_energy: f32 = kernel.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
                let magnitude = sum.norm() / (kernel_energy + 1e-7);
                result[k][n] = magnitude;
            }
        }

        result
    }

    /// Compute CQT for a single frame of audio
    /// More efficient if you only need one time frame
    pub fn compute_frame(&self, samples: &[f32]) -> Vec<f32> {
        let num_bins = self.config.num_bins();
        let mut result = vec![0.0; num_bins];

        let frame_len = samples.len();

        for (k, kernel) in self.kernels.iter().enumerate() {
            if frame_len < kernel.len() {
                continue;
            }

            let mut sum = Complex::new(0.0, 0.0);

            // Circular convolution: if kernel is longer than frame, use the frame repeatedly
            for m in 0..kernel.len() {
                let sample_idx = m % frame_len;
                sum += samples[sample_idx] * kernel[m];
            }

            let kernel_energy: f32 = kernel.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
            result[k] = sum.norm() / (kernel_energy + 1e-7);
        }

        result
    }

    /// Get CQT config
    pub fn config(&self) -> &CQTConfig {
        &self.config
    }

    /// Convert CQT magnitude to log scale (dB)
    pub fn to_log_scale(magnitude: &[f32], ref_power: f32) -> Vec<f32> {
        magnitude
            .iter()
            .map(|&m| {
                let power = m * m;
                if power > 0.0 {
                    10.0 * (power / ref_power).log10()
                } else {
                    -80.0 // Floor at -80 dB
                }
            })
            .collect()
    }

    /// Apply mel-scale warping to CQT (already log-frequency, but this further warps)
    pub fn apply_mel_warping(cqt_frame: &[f32], num_mel_bins: usize) -> Vec<f32> {
        if num_mel_bins == 0 || cqt_frame.is_empty() {
            return vec![];
        }

        // Simple averaging pooling if cqt_frame.len() > num_mel_bins
        let pool_size = cqt_frame.len() / num_mel_bins;
        if pool_size == 0 {
            return cqt_frame.to_vec();
        }

        let mut mel_frame = vec![0.0; num_mel_bins];
        for i in 0..num_mel_bins {
            let start = i * pool_size;
            let end = ((i + 1) * pool_size).min(cqt_frame.len());
            let avg = cqt_frame[start..end].iter().sum::<f32>() / (end - start) as f32;
            mel_frame[i] = avg;
        }

        mel_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cqt_config_piano() {
        let config = CQTConfig::piano_range(44100);
        assert_eq!(config.sample_rate, 44100);
        assert!(config.num_bins() > 0);
        assert!(config.fmin < config.fmax);
    }

    #[test]
    fn test_bin_frequency() {
        let config = CQTConfig::piano_range(44100);
        let f0 = config.bin_frequency(0);
        let f1 = config.bin_frequency(12); // One octave higher
        assert!((f1 - f0 * 2.0).abs() < 0.1);
    }

    #[test]
    fn test_cqt_compute() {
        let config = CQTConfig::piano_range(44100);
        let cqt = CQTransform::new(config);
        let samples = vec![0.1; 4410]; // 0.1 seconds
        let result = cqt.compute(&samples);
        assert!(result.len() > 0);
    }
}
