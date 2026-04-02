#![allow(dead_code)]

use crate::cqt::CQTransform;
use crate::inference::{BasicPitchInference, InferenceConfig};
use crate::pipeline::{AudioPipeline, PipelineConfig};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::{Mutex, OnceLock};

static BASIC_PITCH_ENGINE: OnceLock<Mutex<Option<BasicPitchInference>>> = OnceLock::new();

pub const PIANO_LOW_MIDI: u8 = 21;
pub const PIANO_HIGH_MIDI: u8 = 108;
pub const PIANO_KEY_COUNT: usize = 88;
const _: [(); PIANO_KEY_COUNT] = [(); (PIANO_HIGH_MIDI as usize - PIANO_LOW_MIDI as usize + 1)];

pub fn waveform_points(samples: &[f32], sample_rate: u32, max_points: usize) -> Vec<[f64; 2]> {
    if samples.is_empty() || sample_rate == 0 || max_points == 0 {
        return Vec::new();
    }

    let step = (samples.len() / max_points.max(1)).max(1);
    let mut points = Vec::with_capacity(samples.len() / step + 1);

    for i in (0..samples.len()).step_by(step) {
        let t = i as f64 / sample_rate as f64;
        points.push([t, samples[i] as f64]);
    }

    points
}

pub fn detect_note_probabilities(
    samples: &[f32],
    sample_rate: u32,
    center_sample: usize,
    fft_size: usize,
) -> Vec<f32> {
    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    let mut probs = vec![0.0f32; note_count];

    if samples.len() < fft_size || sample_rate == 0 || fft_size < 32 {
        return probs;
    }

    let half = fft_size / 2;
    let start = center_sample
        .saturating_sub(half)
        .min(samples.len() - fft_size);
    let slice = &samples[start..start + fft_size];

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut buffer: Vec<Complex<f32>> = slice
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos();
            Complex::new(s * w, 0.0)
        })
        .collect();

    fft.process(&mut buffer);
    let mags: Vec<f32> = buffer.iter().take(fft_size / 2).map(|c| c.norm()).collect();
    let power: Vec<f32> = mags.iter().map(|m| m * m).collect();

    let bin_hz = sample_rate as f32 / fft_size as f32;
    let nyquist_hz = sample_rate as f32 * 0.5;
    let split_hz = 3200.0f32.min(nyquist_hz * 0.95);
    let split_bin = ((split_hz / bin_hz) as usize).min(power.len().saturating_sub(1));

    let low_energy: f32 = power.iter().take(split_bin.max(1)).copied().sum();
    let high_energy: f32 = power.iter().skip(split_bin.max(1)).copied().sum();
    let high_ratio = high_energy / (low_energy + high_energy + 1.0e-9);

    // Spectral flatness: close to 0 => tonal/harmonic, close to 1 => noisy/percussive.
    let flatness = if split_bin > 16 {
        let bins = &power[8..split_bin];
        let n = bins.len() as f32;
        let geo = (bins.iter().map(|v| (v + 1.0e-12).ln()).sum::<f32>() / n).exp();
        let arith = bins.iter().copied().sum::<f32>() / n + 1.0e-12;
        (geo / arith).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let tonal_factor =
        (1.0 - flatness).clamp(0.0, 1.0).powf(1.35) * (1.0 - high_ratio).clamp(0.0, 1.0).powf(0.65);

    if tonal_factor < 0.05 {
        return probs;
    }

    for (idx, midi) in (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI).enumerate() {
        let f0 = midi_to_freq(midi);
        let mut score = 0.0;
        let mut fundamental = 0.0;
        let mut second = 0.0;

        for harmonic in 1..=7 {
            let target = f0 * harmonic as f32;
            if target >= nyquist_hz * 0.98 {
                break;
            }
            let bin = (target / bin_hz).round() as usize;
            if bin < power.len() {
                let peak = weighted_peak(&power, bin);
                let background = local_noise_floor(&power, bin);
                let peakiness = (peak - background * 0.9).max(0.0);
                let h_w = 1.0 / (harmonic as f32).powf(1.2);
                score += peakiness * h_w;
                if harmonic == 1 {
                    fundamental = peakiness;
                } else if harmonic == 2 {
                    second = peakiness;
                }
            }
        }

        // Suppress notes with weak fundamental support (common in transients/noise).
        let support = fundamental + second;
        if support < 1.0e-9 {
            score *= 0.2;
        } else if fundamental < second * 0.25 {
            score *= 0.65;
        }

        probs[idx] = score;
    }

    let max_v = probs
        .iter()
        .copied()
        .fold(0.0f32, |a, b| if a > b { a } else { b });

    if max_v > 1.0e-9 {
        for p in &mut probs {
            *p /= max_v;
        }

        // Keep the strongest candidates and suppress weak "always-on" tails.
        let mut ranked = probs.clone();
        ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let sparse_floor = ranked.get(9).copied().unwrap_or(0.0) * 0.55;

        for p in &mut probs {
            let s = ((*p - sparse_floor).max(0.0) / (1.0 - sparse_floor + 1.0e-6)).powf(1.8);
            let gated = (s * tonal_factor).clamp(0.0, 1.0);
            *p = if gated >= 0.06 { gated } else { 0.0 };
        }

        let max_after = probs
            .iter()
            .copied()
            .fold(0.0f32, |a, b| if a > b { a } else { b });
        if max_after > 1.0e-9 {
            for p in &mut probs {
                *p /= max_after;
            }
        }
    }

    probs
}

pub fn pitch_track(samples: &[f32], sample_rate: u32, points: usize) -> Vec<[f64; 2]> {
    if samples.is_empty() || sample_rate == 0 || points == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(points);
    let last_idx = samples.len().saturating_sub(1);

    for i in 0..points {
        let frac = i as f32 / points as f32;
        let center = (frac * last_idx as f32) as usize;
        let probs = detect_note_probabilities(samples, sample_rate, center, 4096);

        let (best_idx, best_prob) = probs.iter().enumerate().fold(
            (0usize, 0.0f32),
            |acc, (idx, &p)| if p > acc.1 { (idx, p) } else { acc },
        );

        let best_midi = PIANO_LOW_MIDI as f64 + best_idx as f64;
        let t = center as f64 / sample_rate as f64;

        // Confidence gates low-energy/noisy windows so the track is less jumpy.
        let y = if best_prob > 0.28 {
            best_midi
        } else {
            f64::NAN
        };
        out.push([t, y]);
    }

    out
}

fn midi_to_freq(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

fn weighted_peak(power: &[f32], center_bin: usize) -> f32 {
    if power.is_empty() {
        return 0.0;
    }

    const W: [f32; 5] = [0.15, 0.6, 1.0, 0.6, 0.15];
    let mut acc = 0.0;
    let mut wsum = 0.0;

    for (i, w) in W.iter().enumerate() {
        let offset = i as isize - 2;
        let bin = center_bin as isize + offset;
        if bin >= 0 && (bin as usize) < power.len() {
            acc += power[bin as usize] * *w;
            wsum += *w;
        }
    }

    if wsum > 0.0 {
        acc / wsum
    } else {
        0.0
    }
}

fn local_noise_floor(power: &[f32], center_bin: usize) -> f32 {
    if power.is_empty() {
        return 0.0;
    }

    let mut acc = 0.0;
    let mut n = 0usize;
    for d in 3..=7 {
        if center_bin >= d {
            acc += power[center_bin - d];
            n += 1;
        }
        if center_bin + d < power.len() {
            acc += power[center_bin + d];
            n += 1;
        }
    }

    if n > 0 {
        acc / n as f32
    } else {
        0.0
    }
}

// ============================================================================
// Pro analysis using Constant-Q Transform / Basic Pitch pipeline
// ============================================================================

/// Compute Constant-Q Transform based note probabilities
/// This provides better frequency resolution for polyphonic transcription
pub fn detect_note_probabilities_cqt(
    samples: &[f32],
    sample_rate: u32,
    center_sample: usize,
    fft_size: usize,
) -> Vec<f32> {
    if let Some(probs) = detect_note_probabilities_basic_pitch(samples, sample_rate, center_sample)
    {
        return probs;
    }

    detect_note_probabilities_cqt_fallback(samples, sample_rate, fft_size)
}

fn detect_note_probabilities_basic_pitch(
    samples: &[f32],
    sample_rate: u32,
    center_sample: usize,
) -> Option<Vec<f32>> {
    if samples.is_empty() || sample_rate == 0 {
        return None;
    }

    let config = InferenceConfig::default();
    let source_window_samples = ((config.input_samples as f64) * (sample_rate as f64)
        / (config.model_sample_rate as f64))
        .round()
        .max(1.0) as usize;

    let mut start = center_sample.saturating_sub(source_window_samples / 2);
    let mut end = (start + source_window_samples).min(samples.len());
    if end - start < source_window_samples {
        start = end.saturating_sub(source_window_samples);
    }
    end = end.max(start);

    let window_src = &samples[start..end];
    let window_model =
        BasicPitchInference::resample_linear(window_src, sample_rate, config.model_sample_rate);

    let engine =
        BASIC_PITCH_ENGINE.get_or_init(|| Mutex::new(BasicPitchInference::new(config).ok()));

    let mut guard = engine.lock().ok()?;
    let model = guard.as_mut()?;
    let note_frames = model.infer_audio_window(&window_model).ok()?;
    if note_frames.is_empty() {
        return None;
    }

    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    let mut probs = vec![0.0f32; note_count];
    for frame in &note_frames {
        for (i, p) in frame.iter().take(note_count).enumerate() {
            probs[i] = probs[i].max(*p);
        }
    }

    Some(probs)
}

fn detect_note_probabilities_cqt_fallback(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
) -> Vec<f32> {
    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    let mut probs = vec![0.0f32; note_count];

    if samples.len() < fft_size {
        return probs;
    }

    // Use CQT instead of linear FFT
    let cqt_config = crate::cqt::CQTConfig::piano_range(sample_rate);
    let cqt = CQTransform::new(cqt_config.clone());

    let frame_slice = &samples[..fft_size.min(samples.len())];
    let cqt_frame = cqt.compute_frame(frame_slice);

    // Map CQT bins to MIDI notes
    // Since CQT is already in semitone units, mapping is direct
    for (cqt_idx, &mag) in cqt_frame.iter().enumerate() {
        let note_idx = (cqt_idx / cqt_config.bins_per_semitone).min(note_count - 1);
        probs[note_idx] += mag;
    }

    // Normalize
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    if max_prob > 1e-9 {
        for p in &mut probs {
            *p = (*p / max_prob).clamp(0.0, 1.0);
        }
    }

    probs
}

/// Create or get a reference to the Basic Pitch pipeline.
pub fn create_basic_pitch_pipeline(sample_rate: u32) -> anyhow::Result<AudioPipeline> {
    let config = PipelineConfig {
        sample_rate,
        chunk_size: (sample_rate as usize / 10), // 100ms chunks
        lookahead_frames: 5,
        ..Default::default()
    };

    AudioPipeline::new(config)
}

/// Backward-compatible alias.
pub fn create_hfsformer_pipeline(sample_rate: u32) -> anyhow::Result<AudioPipeline> {
    create_basic_pitch_pipeline(sample_rate)
}

/// Compute log-magnitude CQT spectrogram
pub fn compute_cqt_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    hop_size: usize,
) -> Vec<Vec<f32>> {
    if samples.len() < hop_size {
        return vec![];
    }

    let cqt_config = crate::cqt::CQTConfig::piano_range(sample_rate);
    let cqt = CQTransform::new(cqt_config);

    let num_frames = (samples.len() - hop_size) / hop_size + 1;
    let mut spectrogram = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = (start + hop_size).min(samples.len());
        let frame = &samples[start..end];

        let cqt_frame = cqt.compute_frame(frame);
        let log_frame = CQTransform::to_log_scale(&cqt_frame, 1.0);

        spectrogram.push(log_frame);
    }

    spectrogram
}

/// Pitch track using CQT (better for polyphonic audio)
pub fn pitch_track_cqt(samples: &[f32], sample_rate: u32, points: usize) -> Vec<[f64; 2]> {
    if samples.is_empty() || sample_rate == 0 || points == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(points);
    let last_idx = samples.len().saturating_sub(1);

    for i in 0..points {
        let frac = i as f32 / points as f32;
        let center = (frac * last_idx as f32) as usize;
        let probs = detect_note_probabilities_cqt(samples, sample_rate, center, 4096);

        let (best_idx, best_prob) = probs.iter().enumerate().fold(
            (0usize, 0.0f32),
            |acc, (idx, &p)| if p > acc.1 { (idx, p) } else { acc },
        );

        let best_midi = PIANO_LOW_MIDI as f64 + best_idx as f64;
        let t = center as f64 / sample_rate as f64;

        // CQT-based confidence gating is slightly more lenient since CQT inherently
        // provides better separation of notes
        let y = if best_prob > 0.25 {
            best_midi
        } else {
            f64::NAN
        };
        out.push([t, y]);
    }

    out
}

/// Full professional pipeline: HPSS + CQT + Viterbi smoothing
/// Returns smoothed note activations and raw probabilities
pub fn analyze_with_full_pipeline(
    samples: &[f32],
    sample_rate: u32,
) -> anyhow::Result<(Vec<Vec<bool>>, Vec<Vec<f32>>)> {
    use crate::pipeline::{AudioPipeline, PipelineConfig};

    let config = PipelineConfig {
        sample_rate,
        chunk_size: (sample_rate as usize / 10), // 100ms chunks
        lookahead_frames: 5,
        ..Default::default()
    };

    let pipeline = AudioPipeline::new(config)?;
    let result = pipeline.process_audio(samples)?;

    Ok((result.smoothed_notes, result.note_probs_sequence))
}
