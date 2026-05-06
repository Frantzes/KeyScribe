#![allow(dead_code)]

use crate::cqt::CQTransform;
use crate::inference::{BasicPitchInference, InferenceConfig};
use crate::pipeline::{AudioPipeline, PipelineConfig};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

static BASIC_PITCH_ENGINE: OnceLock<Mutex<Option<BasicPitchInference>>> = OnceLock::new();
static CQT_PREVIEW_ENGINE: OnceLock<Mutex<Option<(u32, Arc<crate::cqt::CQTransform>)>>> =
    OnceLock::new();

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
    if let Some(probs) = detect_note_probabilities_basic_pitch(samples, sample_rate, center_sample)
    {
        return probs;
    }
    let _ = (samples, sample_rate, center_sample, fft_size);
    vec![0.0f32; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize]
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
#[deprecated(note = "CQT analysis is deprecated. Use Basic Pitch only.")]
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

    let _ = fft_size;
    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    vec![0.0f32; note_count]
}

/// Lightweight CQT detector for responsive live previews.
///
/// This intentionally avoids Basic Pitch ONNX inference so UI-driven updates remain smooth
/// while loading. Full Pro analysis still runs in the background pipeline.
#[deprecated(note = "CQT preview is deprecated. Use Basic Pitch only.")]
pub fn detect_note_probabilities_cqt_preview(
    samples: &[f32],
    sample_rate: u32,
    center_sample: usize,
    fft_size: usize,
) -> Vec<f32> {
    if let Some(probs) = detect_note_probabilities_basic_pitch(samples, sample_rate, center_sample)
    {
        return probs;
    }
    let _ = fft_size;
    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    vec![0.0f32; note_count]
}

fn compute_cached_cqt_preview_frame(
    sample_rate: u32,
    frame_slice: &[f32],
) -> Option<(Vec<f32>, usize)> {
    let cache = CQT_PREVIEW_ENGINE.get_or_init(|| Mutex::new(None));
    let cqt = {
        let mut guard = cache.lock().ok()?;

        let rebuild = guard
            .as_ref()
            .map(|(cached_rate, _)| *cached_rate != sample_rate)
            .unwrap_or(true);
        if rebuild {
            let cqt_config = crate::cqt::CQTConfig::piano_range(sample_rate);
            *guard = Some((sample_rate, Arc::new(CQTransform::new(cqt_config))));
        }

        guard.as_ref().map(|(_, cqt)| Arc::clone(cqt))?
    };

    Some((
        cqt.compute_frame(frame_slice),
        cqt.config().bins_per_semitone.max(1),
    ))
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
    let center_idx = note_frames.len() / 2;
    let start_idx = center_idx.saturating_sub(1);
    let end_idx = (center_idx + 1).min(note_frames.len().saturating_sub(1));
    let span = end_idx.saturating_sub(start_idx) + 1;

    for frame_idx in start_idx..=end_idx {
        let frame = &note_frames[frame_idx];
        for (i, p) in frame.iter().take(note_count).enumerate() {
            probs[i] += *p;
        }
    }

    if span > 1 {
        let inv = 1.0 / span as f32;
        for p in &mut probs {
            *p *= inv;
        }
    }

    Some(probs)
}

#[deprecated(note = "CQT fallback is deprecated. Use Basic Pitch only.")]
fn detect_note_probabilities_cqt_fallback(
    samples: &[f32],
    sample_rate: u32,
    center_sample: usize,
    fft_size: usize,
) -> Vec<f32> {
    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    let mut probs = vec![0.0f32; note_count];

    if samples.is_empty() || sample_rate == 0 || fft_size < 32 {
        return probs;
    }

    let window_size = fft_size.min(samples.len());
    let half_window = window_size / 2;
    let start = center_sample
        .saturating_sub(half_window)
        .min(samples.len().saturating_sub(window_size));
    let end = start + window_size;

    let frame_slice = &samples[start..end];
    let (cqt_frame, bins_per_semitone) = if let Some(cached) =
        compute_cached_cqt_preview_frame(sample_rate, frame_slice)
    {
        cached
    } else {
        let cqt_config = crate::cqt::CQTConfig::piano_range(sample_rate);
        let bins = cqt_config.bins_per_semitone.max(1);
        let cqt = CQTransform::new(cqt_config);
        (cqt.compute_frame(frame_slice), bins)
    };

    if cqt_frame.is_empty() {
        return probs;
    }

    let cqt_sum = cqt_frame.iter().copied().sum::<f32>();
    if cqt_sum <= 1.0e-12 {
        return probs;
    }

    // Similar to the FFT path, gate noisy/percussive frames so we don't paint the full keyboard.
    let n = cqt_frame.len() as f32;
    let geo = (cqt_frame.iter().map(|v| (v + 1.0e-12).ln()).sum::<f32>() / n).exp();
    let arith = (cqt_sum / n).max(1.0e-12);
    let flatness = (geo / arith).clamp(0.0, 1.0);
    let tonal_factor = (1.0 - flatness).clamp(0.0, 1.0).powf(1.2);
    if tonal_factor < 0.08 {
        return probs;
    }

    // Map CQT bins to MIDI notes. Since CQT is already in semitone units,
    // map each bin directly into the note bucket and accumulate energy.
    for (cqt_idx, &mag) in cqt_frame.iter().enumerate() {
        let note_idx = (cqt_idx / bins_per_semitone).min(note_count - 1);
        probs[note_idx] += mag * mag;
    }

    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    if max_prob <= 1.0e-9 {
        return probs;
    }

    for p in &mut probs {
        *p = (*p / max_prob).clamp(0.0, 1.0).powf(1.25);
    }

    // Keep only a sparse set of strongest candidates (same principle as FFT detector).
    let mut ranked = probs.clone();
    ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let sparse_floor = ranked.get(11).copied().unwrap_or(0.0) * 0.78;

    for p in &mut probs {
        let s = ((*p - sparse_floor).max(0.0) / (1.0 - sparse_floor + 1.0e-6)).powf(1.7);
        let gated = (s * tonal_factor).clamp(0.0, 1.0);
        *p = if gated >= 0.08 { gated } else { 0.0 };
    }

    // Suppress shoulders around peaks so adjacent notes don't all stay lit.
    let snapshot = probs.clone();
    for i in 0..note_count {
        let left = if i > 0 { snapshot[i - 1] } else { 0.0 };
        let right = if i + 1 < note_count {
            snapshot[i + 1]
        } else {
            0.0
        };
        if snapshot[i] < left.max(right) * 0.92 {
            probs[i] *= 0.45;
        }
        if probs[i] < 0.08 {
            probs[i] = 0.0;
        }
    }

    let max_after = probs.iter().copied().fold(0.0f32, f32::max);
    if max_after > 1.0e-9 {
        for p in &mut probs {
            *p /= max_after;
        }
    }

    probs
}

/// Create or get a reference to the Basic Pitch pipeline.
pub fn create_basic_pitch_pipeline(sample_rate: u32) -> anyhow::Result<AudioPipeline> {
    let config = PipelineConfig {
        sample_rate,
        chunk_size: sample_rate as usize / 10, // 100ms chunks
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
#[deprecated(note = "CQT spectrogram is deprecated. Use Basic Pitch only.")]
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
#[deprecated(note = "CQT pitch tracking is deprecated. Use Basic Pitch only.")]
pub fn pitch_track_cqt(samples: &[f32], sample_rate: u32, points: usize) -> Vec<[f64; 2]> {
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
