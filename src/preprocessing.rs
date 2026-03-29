use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Configuration for preprocessing stage
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// FFT/STFT size
    pub fft_size: usize,
    /// Hop size (advance per frame)
    pub hop_size: usize,
    /// HPSS kernel size for harmonic median filter (time)
    pub harmonic_kernel_time: usize,
    /// HPSS kernel size for harmonic median filter (frequency)
    pub harmonic_kernel_freq: usize,
    /// HPSS kernel size for percussive median filter (time)
    pub percussive_kernel_time: usize,
    /// HPSS kernel size for percussive median filter (frequency)
    pub percussive_kernel_freq: usize,
    /// Margin parameter for HPSS (balance between H and P)
    pub margin: f32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            harmonic_kernel_time: 31,
            harmonic_kernel_freq: 31,
            percussive_kernel_time: 7,
            percussive_kernel_freq: 31,
            margin: 1.0,
        }
    }
}

/// Preprocessing stage: HPSS, windowing, normalization
pub struct Preprocessor {
    config: PreprocessingConfig,
    window: Vec<f32>,
    fft_planner: FftPlanner<f32>,
}

impl Preprocessor {
    /// Create a new preprocessor with given config
    pub fn new(config: PreprocessingConfig) -> Self {
        let window = Self::hann_window(config.fft_size);
        let fft_planner = FftPlanner::new();

        Self {
            config,
            window,
            fft_planner,
        }
    }

    /// Generate Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|n| {
                let phase = 2.0 * PI * n as f32 / (size - 1) as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect()
    }

    /// Apply Hann window to audio frame
    pub fn apply_window(&self, frame: &[f32]) -> Vec<f32> {
        frame
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| s * w)
            .collect()
    }

    /// Compute STFT of input signal
    /// Returns complex spectrogram of shape (num_frames, fft_size/2+1)
    pub fn compute_stft(&mut self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        if samples.len() < self.config.fft_size {
            return vec![];
        }

        let hop_size = self.config.hop_size;
        let fft_size = self.config.fft_size;
        let num_frames = (samples.len() - fft_size) / hop_size + 1;

        let mut stft = Vec::with_capacity(num_frames);
        let fft = self.fft_planner.plan_fft_forward(fft_size);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let frame = &samples[start..start + fft_size];

            // Apply window
            let windowed: Vec<Complex<f32>> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(s, w)| Complex::new(s * w, 0.0))
                .collect();

            let mut buffer = windowed;
            fft.process(&mut buffer);

            // Keep only positive frequencies (0 to fft_size/2)
            let positive_freqs: Vec<Complex<f32>> =
                buffer.iter().take(fft_size / 2 + 1).cloned().collect();
            stft.push(positive_freqs);
        }

        stft
    }

    /// Harmonic-Percussive Source Separation (HPSS)
    /// Separates harmonic and percussive components from STFT
    /// Returns (harmonic_magnitude, percussive_magnitude) where both are real-valued
    pub fn hpss(&self, stft: &[Vec<Complex<f32>>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if stft.is_empty() || stft[0].is_empty() {
            return (vec![], vec![]);
        }

        let num_frames = stft.len();
        let num_freqs = stft[0].len();

        // Convert to magnitude spectrogram
        let mut magnitude: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| frame.iter().map(|c| c.norm()).collect())
            .collect();

        // Apply log scaling to make separation easier
        magnitude.iter_mut().for_each(|frame| {
            frame.iter_mut().for_each(|mag| {
                *mag = (*mag + 1e-7).ln().max(-80.0);
            });
        });

        let cfg = &self.config;

        // Horizontal median filter (time direction) for harmonic
        let horizontal_filtered = self.median_filter_time(&magnitude, cfg.harmonic_kernel_time);

        // Vertical median filter (frequency direction) for harmonic
        let harmonic_filtered =
            self.median_filter_freq(&horizontal_filtered, cfg.harmonic_kernel_freq);

        // Horizontal median filter for percussive
        let horizontal_filtered_p = self.median_filter_time(&magnitude, cfg.percussive_kernel_time);

        // Vertical median filter for percussive
        let percussive_filtered =
            self.median_filter_freq(&horizontal_filtered_p, cfg.percussive_kernel_freq);

        // Soft masking with margin
        let mut harmonic_mask = vec![vec![0.0; num_freqs]; num_frames];
        let mut percussive_mask = vec![vec![0.0; num_freqs]; num_frames];

        for t in 0..num_frames {
            for f in 0..num_freqs {
                let h = harmonic_filtered[t][f];
                let p = percussive_filtered[t][f];

                let h_margin = h + cfg.margin;
                let p_margin = p + cfg.margin;

                if h_margin > p_margin {
                    harmonic_mask[t][f] = 1.0;
                    percussive_mask[t][f] = 0.0;
                } else {
                    harmonic_mask[t][f] = 0.0;
                    percussive_mask[t][f] = 1.0;
                }
            }
        }

        // Apply masks to original magnitude
        let harmonic: Vec<Vec<f32>> = magnitude
            .iter()
            .enumerate()
            .map(|(t, frame)| {
                frame
                    .iter()
                    .enumerate()
                    .map(|(f, mag)| mag * harmonic_mask[t][f])
                    .collect()
            })
            .collect();

        let percussive: Vec<Vec<f32>> = magnitude
            .iter()
            .enumerate()
            .map(|(t, frame)| {
                frame
                    .iter()
                    .enumerate()
                    .map(|(f, mag)| mag * percussive_mask[t][f])
                    .collect()
            })
            .collect();

        (harmonic, percussive)
    }

    /// Median filter in time direction
    fn median_filter_time(&self, spectrogram: &[Vec<f32>], kernel_size: usize) -> Vec<Vec<f32>> {
        if spectrogram.is_empty() {
            return vec![];
        }

        let num_frames = spectrogram.len();
        let num_freqs = spectrogram[0].len();
        let half_kernel = kernel_size / 2;

        let mut result = vec![vec![0.0; num_freqs]; num_frames];

        for t in 0..num_frames {
            for f in 0..num_freqs {
                let start_t = t.saturating_sub(half_kernel);
                let end_t = (t + half_kernel + 1).min(num_frames);

                let mut values: Vec<f32> = spectrogram[start_t..end_t]
                    .iter()
                    .map(|frame| frame[f])
                    .collect();

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                result[t][f] = values[values.len() / 2];
            }
        }

        result
    }

    /// Median filter in frequency direction
    fn median_filter_freq(&self, spectrogram: &[Vec<f32>], kernel_size: usize) -> Vec<Vec<f32>> {
        if spectrogram.is_empty() || spectrogram[0].is_empty() {
            return spectrogram.to_vec();
        }

        let num_frames = spectrogram.len();
        let num_freqs = spectrogram[0].len();
        let half_kernel = kernel_size / 2;

        let mut result = vec![vec![0.0; num_freqs]; num_frames];

        for t in 0..num_frames {
            for f in 0..num_freqs {
                let start_f = f.saturating_sub(half_kernel);
                let end_f = (f + half_kernel + 1).min(num_freqs);

                let mut values: Vec<f32> = spectrogram[t][start_f..end_f].to_vec();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                result[t][f] = values[values.len() / 2];
            }
        }

        result
    }

    /// Normalize spectrogram to [0, 1] range
    pub fn normalize_spectrogram(spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if spec.is_empty() || spec[0].is_empty() {
            return spec.to_vec();
        }

        let max_val = spec
            .iter()
            .flat_map(|frame| frame.iter())
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let min_val = spec
            .iter()
            .flat_map(|frame| frame.iter())
            .copied()
            .fold(f32::INFINITY, f32::min);

        let range = max_val - min_val;
        if range.abs() < 1e-7 {
            return spec.to_vec();
        }

        spec.iter()
            .map(|frame| frame.iter().map(|val| (val - min_val) / range).collect())
            .collect()
    }

    /// Get the FFT size
    pub fn fft_size(&self) -> usize {
        self.config.fft_size
    }

    /// Get the hop size
    pub fn hop_size(&self) -> usize {
        self.config.hop_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = Preprocessor::hann_window(512);
        assert_eq!(window.len(), 512);
        assert!(window[0] > 0.0);
        assert!(window[256] > 0.99); // Peak near center
    }

    #[test]
    fn test_preprocessor_creation() {
        let config = PreprocessingConfig::default();
        let preprocessor = Preprocessor::new(config);
        assert_eq!(preprocessor.fft_size(), 2048);
    }

    #[test]
    fn test_stft() {
        let config = PreprocessingConfig::default();
        let mut preprocessor = Preprocessor::new(config);
        let samples = vec![0.1; 4410]; // 0.1 seconds at 44.1kHz
        let stft = preprocessor.compute_stft(&samples);
        assert!(stft.len() > 0);
        assert_eq!(stft[0].len(), 1025); // 2048/2 + 1
    }
}
