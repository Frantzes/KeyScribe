use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::leadsheet::beat_association::associate_note_events;
use crate::leadsheet::bpm::{detect_bpm, detect_bpm_from_audio, BpmDetectionConfig, TempoEstimate};
use crate::leadsheet::NoteEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatTrackResult {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BeatTrackConfig {
    pub model: String,
    pub device: BeatTrackDevice,
    pub dbn: bool,
}

impl Default for BeatTrackConfig {
    fn default() -> Self {
        Self {
            model: "final0".to_string(),
            device: BeatTrackDevice::Auto,
            dbn: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BeatTrackDevice {
    Auto,
    Cpu,
    Cuda,
}

pub fn run_beat_this(audio_path: &Path, config: &BeatTrackConfig) -> Result<BeatTrackResult> {
    let mut results = run_beat_this_multi(&[audio_path], config)?;
    results.pop().ok_or_else(|| anyhow!("No result from beat tracker"))
}

/// Run beat_this on multiple audio files.
/// The tracker is created once and reused across all files.
pub fn run_beat_this_multi(
    audio_paths: &[&Path],
    _config: &BeatTrackConfig,
) -> Result<Vec<BeatTrackResult>> {
    if audio_paths.is_empty() {
        return Err(anyhow!("No audio files provided for beat tracking"));
    }
    for &p in audio_paths {
        if !p.is_file() {
            return Err(anyhow!("Beat tracking input is not a file: {}", p.display()));
        }
    }

    let mut tracker = crate::beat_this::create_tracker()
        .context("Failed to initialize beat-this tracker")?;

    let mut results = Vec::with_capacity(audio_paths.len());
    for &p in audio_paths {
        let analysis = tracker.analyze_file(p)
            .map_err(|e| anyhow!("beat-this failed on {}: {}", p.display(), e))?;
        results.push(BeatTrackResult {
            beats: analysis.beats,
            downbeats: analysis.downbeats,
        });
    }

    for r in &mut results {
        correct_beat_metric_level(r);
    }

    Ok(results)
}



/// Run beat_this on combined drum+bass stems (or fall back to full mix).
pub fn run_beat_this_combined(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
    sample_rate: u32,
    _config: &BeatTrackConfig,
) -> Result<BeatTrackResult> {
    let combined = combined_audio(bass_samples, drum_samples, full_mix_samples)?;
    let mut tracker = crate::beat_this::create_tracker()
        .context("Failed to initialize beat-this tracker")?;
    let analysis = tracker.analyze_audio(&combined, sample_rate)?;
    let mut result = BeatTrackResult {
        beats: analysis.beats,
        downbeats: analysis.downbeats,
    };
    correct_beat_metric_level(&mut result);
    Ok(result)
}

fn infer_beats_per_bar(downbeats: &[f32], beats: &[f32]) -> u32 {
    if downbeats.len() < 2 || beats.len() < 2 {
        return 4;
    }
    let dbi = median_of_values(downbeats);
    let bi = median_of_values(beats);
    if bi < 0.001 {
        return 4;
    }
    (dbi / bi).round() as u32
}

fn median_of_values(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let samples: Vec<f32> = if values.len() >= 2 {
        values.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0.001).collect()
    } else {
        return 0.5;
    };
    if samples.is_empty() {
        return 0.5;
    }
    let mut sorted = samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

impl From<BeatTrackResult> for CrossValidatedBeats {
    fn from(bt: BeatTrackResult) -> Self {
        let bi = median_of_values(&bt.beats);
        let bpb = infer_beats_per_bar(&bt.downbeats, &bt.beats).clamp(2, 8);
        Self {
            beats: bt.beats,
            downbeats: bt.downbeats,
            beats_per_bar: bpb,
            bpm: (60.0 / bi.max(0.001)).clamp(40.0, 260.0),
            confidence: 0.5,
            source_count: 1,
        }
    }
}

impl From<CrossValidatedBeats> for BeatTrackResult {
    fn from(cv: CrossValidatedBeats) -> Self {
        BeatTrackResult {
            beats: cv.beats,
            downbeats: cv.downbeats,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CrossValidatedBeats {
    /// Consensus downbeat positions (seconds).
    pub downbeats: Vec<f32>,
    /// Consensus beat positions (seconds).
    pub beats: Vec<f32>,
    /// Beats per bar inferred from the downbeat/beat intervals.
    pub beats_per_bar: u32,
    /// Median BPM across all sources.
    pub bpm: f32,
    /// Confidence 0.0–1.0 based on cross-source agreement.
    pub confidence: f32,
    /// Number of audio sources that contributed (1–3).
    pub source_count: u32,
}

const RHYTHM_PHASE_TARGETS: [f32; 6] = [0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0];

/// Score how well a candidate beat grid aligns the melody onsets to the rhythm
/// vocabulary, weighted toward strong beats. Onbeat (0.0) and half-beat (0.5)
/// positions are far more common in real music than passing 16ths/triplets, so
/// a grid that displaces strong-beat notes must cost more than one that only
/// misplaces subdivisions.
pub(crate) fn score_phase_alignment(
    notes: &[NoteEvent],
    beats: &[f32],
    downbeats: &[f32],
) -> f32 {
    let aligned = associate_note_events(notes, beats, downbeats);
    if aligned.is_empty() {
        return f32::INFINITY;
    }
    let total: f32 = aligned
        .iter()
        .map(|n| {
            let (min_dist, closest) = RHYTHM_PHASE_TARGETS
                .iter()
                .map(|target| ((n.intra_beat_pos - target).abs(), *target))
                .fold((f32::INFINITY, 0.0f32), |a, b| if b.0 < a.0 { b } else { a });
            // Strong-beat weighting: onbeat and half-beat positions are much
            // more common in real music — a grid that aligns them correctly
            // should score much better than one that only aligns passing
            // subdivisions.
            let weight = if closest.abs() < 0.01 {
                3.0 // onbeat (beat 1, 2, 3, 4)
            } else if (closest - 0.5).abs() < 0.01 {
                2.0 // offbeat 8th ("and")
            } else {
                1.0 // subdivisions (16ths, triplets)
            };
            weight * min_dist
        })
        .sum();
    total / aligned.len() as f32
}

/// Refine a tracker grid's metric and phase using detected melody onsets.
/// BeatThis can return a musically plausible half-time grid, or place its first
/// beat a fraction late/early. Testing the base and doubled metric at a small
/// phase neighborhood prevents either error from reaching quantization.
pub fn refine_beat_phase(notes: &[NoteEvent], base: &CrossValidatedBeats) -> CrossValidatedBeats {
    if notes.is_empty() || base.beats.len() < 2 {
        return base.clone();
    }
    let mut intervals: Vec<f32> = base
        .beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d > 0.001)
        .collect();
    if intervals.is_empty() {
        return base.clone();
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let period = intervals[intervals.len() / 2];

let score = |beats: &[f32], downbeats: &[f32]| -> f32 {
        score_phase_alignment(notes, beats, downbeats)
    };

    let doubled_beats = || {
        let mut out = Vec::with_capacity(base.beats.len() * 2);
        for pair in base.beats.windows(2) {
            out.push(pair[0]);
            out.push((pair[0] + pair[1]) * 0.5);
        }
        if let Some(last) = base.beats.last().copied() {
            out.push(last);
        }
        out
    };

    let estimate_bpb = |downbeats: &[f32], candidate_period: f32| -> u32 {
        let mut values: Vec<u32> = downbeats
            .windows(2)
            .map(|w| ((w[1] - w[0]) / candidate_period).round() as u32)
            .filter(|&n| (2..=8).contains(&n))
            .collect();
        values.sort_unstable();
        values
            .get(values.len() / 2)
            .copied()
            .unwrap_or(base.beats_per_bar.max(2))
    };

    struct GridSource {
        doubled: bool,
        beats: Vec<f32>,
        downbeats: Vec<f32>,
        period: f32,
        bpm: f32,
        beats_per_bar: u32,
    }

    let mut sources = vec![GridSource {
        doubled: false,
        beats: base.beats.clone(),
        downbeats: base.downbeats.clone(),
        period,
        bpm: base.bpm,
        beats_per_bar: base.beats_per_bar,
    }];
    sources.push(GridSource {
        doubled: true,
        beats: doubled_beats(),
        downbeats: base.downbeats.clone(),
        period: period * 0.5,
        bpm: base.bpm * 2.0,
        beats_per_bar: estimate_bpb(&base.downbeats, period * 0.5),
    });

    // The existing onset autocorrelation is a useful independent period
    // proposal. Keep it only when it differs meaningfully from the tracker so
    // this remains a cheap candidate search rather than a second full tracker.
    if let Some(note_tempo) = detect_bpm(
        notes,
        BpmDetectionConfig {
            max_bpm: 400.0,
            ..BpmDetectionConfig::default()
        },
    ) {
        let note_period = note_tempo.beat_duration_sec;
        if note_period.is_finite() && note_period > 0.001
            && ((note_period - period) / period).abs() > 0.02
            && ((note_period - period * 0.5) / period).abs() > 0.02
        {
            let anchor = base.beats[0];
            let scale = note_period / period;
            sources.push(GridSource {
                doubled: false,
                beats: base
                    .beats
                    .iter()
                    .map(|t| anchor + (*t - anchor) * scale)
                    .collect(),
                downbeats: base
                    .downbeats
                    .iter()
                    .map(|t| anchor + (*t - anchor) * scale)
                    .collect(),
                period: note_period,
                bpm: note_tempo.bpm,
                beats_per_bar: estimate_bpb(&base.downbeats, note_period),
            });
        }
    }

let mut best_score = score(&base.beats, &base.downbeats);
    let mut best_beats = base.beats.clone();
    let mut best_downbeats = base.downbeats.clone();
    let mut best_bpb = base.beats_per_bar;
    let mut best_bpm = base.bpm;
    let mut best_period = period;
    let mut best_shift = 0.0f32;
    let mut best_doubled = false;
    let mut changed = false;
    let mut candidates: Vec<(f32, f32, bool)> = Vec::new(); // (score, shift, doubled)

    for source in sources {
        let candidate_period = source.period;
        // 16 candidates: -0.5 to +0.4375 in steps of 1/16 of the period.
        // At 208 BPM this is 18 ms resolution — smaller than onset detection
        // noise (~30 ms) — so the optimal phase is always reachable.
        let n_steps = 16i32;
        for step in (-n_steps / 2)..=(n_steps / 2 - 1) {
            let shift = step as f32 / n_steps as f32 * candidate_period;
            let beats: Vec<f32> = source.beats.iter().map(|t| *t + shift).collect();
            let downbeats: Vec<f32> = source.downbeats.iter().map(|t| *t + shift).collect();
            let candidate_score = score(&beats, &downbeats);
            candidates.push((candidate_score, shift, source.doubled));
            // Accept a candidate only when it is meaningfully better (the +0.01
            // stability threshold), or when it ties the current winner and is
            // closer to the source's natural grid — a tie-break that keeps an
            // equally-good grid from flipping to an arbitrary phase-shifted one.
            let is_better = candidate_score + 0.01 < best_score;
            // Ties resolve by closeness to the base grid: prefer the period
            // nearest the tracker's own period (the base grid is the most
            // likely), then the non-doubled metric level, then the smallest
            // phase shift. This keeps an equally-good 16th or 8th grid from
            // displacing a musically-plausible quarter grid.
            let is_tie_preferred = candidate_score <= best_score + 1e-6
                && {
                    let cand_dist = (candidate_period - period).abs();
                    let best_dist = (best_period - period).abs();
                    if cand_dist + 1e-6 < best_dist {
                        true
                    } else if cand_dist > best_dist + 1e-6 {
                        false
                    } else if best_doubled != source.doubled {
                        !source.doubled // prefer the non-doubled metric level on ties
                    } else {
                        shift.abs() < best_shift.abs()
                    }
                };
            if is_better || is_tie_preferred {
                best_score = candidate_score;
                best_beats = beats;
                best_downbeats = downbeats;
                best_bpb = source.beats_per_bar;
                best_bpm = source.bpm;
                best_period = candidate_period;
                best_shift = shift;
                best_doubled = source.doubled;
                changed = true;
            }
        }
    }

    // Local gradient refinement: test ±period/32 in 4 sub-steps around the
    // winning shift to find the precise optimum. At 208 BPM, period/32 ≈ 9ms.
    if changed {
        let fine_step = period / 32.0;
        for sub in [-2.0, -1.0, 1.0, 2.0] {
            let fine_shift = sub * fine_step;
            let fine_beats: Vec<f32> = best_beats.iter().map(|t| *t + fine_shift).collect();
            let fine_downbeats: Vec<f32> = best_downbeats.iter().map(|t| *t + fine_shift).collect();
            let fine_score = score(&fine_beats, &fine_downbeats);
            if fine_score + 0.005 < best_score {
                best_score = fine_score;
                best_beats = fine_beats;
                best_downbeats = fine_downbeats;
                best_shift += fine_shift;
            }
        }
    }

    if !changed {
        return base.clone();
    }

    let mut refined = base.clone();
    refined.beats = best_beats;
    refined.downbeats = best_downbeats;
    refined.beats_per_bar = best_bpb;
    refined.bpm = best_bpm;
    if std::env::var_os("KEYSCRIBE_PHASE_DEBUG").is_some() {
        eprintln!(
            "[phase] candidates={} best_score={best_score:.4} shift_ms={:.1} doubled={best_doubled}",
            candidates.len(),
            best_shift * 1000.0
        );
        let mut ranked: Vec<(f32, f32, bool)> = candidates.clone();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (sc, sh, db)) in ranked.iter().take(3).enumerate() {
            eprintln!(
                "[phase] top{} score={sc:.4} shift_ms={:.1} doubled={db}",
                rank + 1,
                sh * 1000.0
            );
        }
    }
if std::env::var_os("KEYSCRIBE_BEAT_PHASE_DEBUG").is_some() {
        eprintln!(
            "[beats] grid refinement doubled={best_doubled} shift={best_shift:.4}s period={period:.4}s score={best_score:.4} bpb={best_bpb}"
        );
    }
    recalibrate_beat_grid_elastic(notes, &refined)
}

/// Refine a fixed-BPM synthetic beat grid's phase (offset) using note onsets.
/// Keeps the tempo (BPM) and beats-per-bar fixed, and only searches for the
/// optimal phase shift in `[-0.5*period, +0.5*period]`.
pub fn refine_beat_phase_fixed_bpm(
    notes: &[NoteEvent],
    base: &CrossValidatedBeats,
) -> CrossValidatedBeats {
    if notes.is_empty() || base.beats.len() < 2 {
        return base.clone();
    }
    let period = 60.0 / base.bpm.clamp(30.0, 400.0);
    let score = |beats: &[f32], downbeats: &[f32]| -> f32 {
        score_phase_alignment(notes, beats, downbeats)
    };

    let mut best_score = score(&base.beats, &base.downbeats);
    let mut best_beats = base.beats.clone();
    let mut best_downbeats = base.downbeats.clone();
    let mut best_shift = 0.0f32;
    let mut changed = false;

    let n_steps = 16i32;
    for step in (-n_steps / 2)..=(n_steps / 2 - 1) {
        let shift = step as f32 / n_steps as f32 * period;
        let beats: Vec<f32> = base.beats.iter().map(|t| *t + shift).collect();
        let downbeats: Vec<f32> = base.downbeats.iter().map(|t| *t + shift).collect();
        let candidate_score = score(&beats, &downbeats);
        if std::env::var_os("KEYSCRIBE_PHASE_DEBUG").is_some() {
            eprintln!(
                "[phase-fixed] step={step} shift_ms={:.1} score={candidate_score:.4} vs best={best_score:.4}",
                shift * 1000.0
            );
        }
        if candidate_score + 0.002 < best_score {
            best_score = candidate_score;
            best_beats = beats;
            best_downbeats = downbeats;
            best_shift = shift;
            changed = true;
        }
    }

    // Local gradient refinement: test ±period/32 around the winning shift
    if changed {
        let fine_step = period / 32.0;
        for sub in [-2.0, -1.0, 1.0, 2.0] {
            let fine_shift = sub * fine_step;
            let fine_beats: Vec<f32> = best_beats.iter().map(|t| *t + fine_shift).collect();
            let fine_downbeats: Vec<f32> = best_downbeats.iter().map(|t| *t + fine_shift).collect();
            let fine_score = score(&fine_beats, &fine_downbeats);
            if fine_score + 0.002 < best_score {
                best_score = fine_score;
                best_beats = fine_beats;
                best_downbeats = fine_downbeats;
                best_shift += fine_shift;
            }
        }
    }

    let refined = if changed {
        let mut r = base.clone();
        r.beats = best_beats;
        r.downbeats = best_downbeats;
        if std::env::var_os("KEYSCRIBE_PHASE_DEBUG").is_some() {
            eprintln!(
                "[phase-fixed] best_score={best_score:.4} shift_ms={:.1}",
                best_shift * 1000.0
            );
        }
        r
    } else {
        base.clone()
    };

    recalibrate_beat_grid_elastic(notes, &refined)
}

/// Dynamically recalibrate beat positions across measures using an inertial
/// elastic grid. Real performances (and rendered audio) can experience micro-drift
/// or expressive timing variations across dozens of measures. This tracks a
/// measure-by-measure bounded phase adjustment with momentum smoothing,
/// preserving continuous monotonic beat spacing.
pub fn recalibrate_beat_grid_elastic(
    notes: &[NoteEvent],
    base: &CrossValidatedBeats,
) -> CrossValidatedBeats {
    if notes.is_empty() || base.beats.len() < 4 {
        return base.clone();
    }

    let bpb = base.beats_per_bar.max(2) as usize;
    let n_beats = base.beats.len();
    let n_measures = (n_beats + bpb - 1) / bpb;
    if n_measures < 2 {
        return base.clone();
    }

    let base_score = score_phase_alignment(notes, &base.beats, &base.downbeats);
    if !base_score.is_finite() {
        return base.clone();
    }

    let mut intervals: Vec<f32> = base
        .beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d > 0.001)
        .collect();
    if intervals.is_empty() {
        return base.clone();
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let period = intervals[intervals.len() / 2];

    // Max local shift per measure: ±15% of a beat (~25-45ms)
    let max_shift = 0.15 * period;
    let n_steps = 8i32;

    let mut measure_shifts = vec![0.0f32; n_measures];
    let mut running_phase = 0.0f32;

    for m in 0..n_measures {
        let b_start = m * bpb;
        let b_end = ((m + 1) * bpb).min(n_beats);
        if b_start >= n_beats {
            break;
        }

        let b_ctx_start = b_start.saturating_sub(1);
        let b_ctx_end = (b_end + 1).min(n_beats);
        let ctx_beats = &base.beats[b_ctx_start..b_ctx_end];
        if ctx_beats.len() < 2 {
            measure_shifts[m] = running_phase;
            continue;
        }

        let t_start = ctx_beats.first().copied().unwrap_or(0.0) - 0.05;
        let t_end = ctx_beats.last().copied().unwrap_or(0.0) + period + 0.05;

        let measure_notes: Vec<NoteEvent> = notes
            .iter()
            .filter(|n| n.start_time >= t_start && n.start_time <= t_end)
            .cloned()
            .collect();

        if measure_notes.is_empty() {
            running_phase *= 0.9;
            measure_shifts[m] = running_phase;
            continue;
        }

        let measure_start_beat = base.beats[b_start];
        let mut best_m_score = f32::INFINITY;
        let mut best_m_shift = running_phase;

        for step in -n_steps..=n_steps {
            let shift = (step as f32 / n_steps as f32) * max_shift;
            let mut m_total_err = 0.0f32;
            let mut m_weight_sum = 0.0f32;

            for n in &measure_notes {
                let dt = (n.start_time - (measure_start_beat + shift)) / period;
                let intra_pos = dt.rem_euclid(1.0);
                let (min_dist, closest) = RHYTHM_PHASE_TARGETS
                    .iter()
                    .map(|target| {
                        let d = (intra_pos - target).abs().min((intra_pos - (target + 1.0)).abs()).min((intra_pos - (target - 1.0)).abs());
                        (d, *target)
                    })
                    .fold((f32::INFINITY, 0.0f32), |a, b| if b.0 < a.0 { b } else { a });
                let weight = if closest.abs() < 0.01 {
                    3.0
                } else if (closest - 0.5).abs() < 0.01 {
                    2.0
                } else {
                    1.0
                };
                m_total_err += weight * min_dist;
                m_weight_sum += weight;
            }

            if m_weight_sum > 0.0 {
                let raw_score = m_total_err / m_weight_sum;
                let diff = (shift - running_phase) / period;
                let inertia_cost = 0.20 * diff * diff;
                let total_m_score = raw_score + inertia_cost;
                if total_m_score < best_m_score {
                    best_m_score = total_m_score;
                    best_m_shift = shift;
                }
            }
        }

        // Momentum update (EMA with alpha = 0.70)
        running_phase = 0.70 * running_phase + 0.30 * best_m_shift;
        measure_shifts[m] = running_phase;
    }

    // Interpolate beat shifts smoothly to prevent discontinuities
    let mut elastic_beats = Vec::with_capacity(n_beats);
    for i in 0..n_beats {
        let m = i / bpb;
        let j = i % bpb;
        let u = j as f32 / bpb as f32;
        let next_m = (m + 1).min(n_measures - 1);
        let shift = (1.0 - u) * measure_shifts[m] + u * measure_shifts[next_m];
        elastic_beats.push(base.beats[i] + shift);
    }

    // Enforce strict monotonicity: each beat interval must be >= 0.5 * period
    for i in 1..elastic_beats.len() {
        if elastic_beats[i] < elastic_beats[i - 1] + 0.5 * period {
            elastic_beats[i] = elastic_beats[i - 1] + 0.5 * period;
        }
    }

    // Recompute downbeats from elastic beat positions
    let mut elastic_downbeats = Vec::new();
    for i in (0..elastic_beats.len()).step_by(bpb) {
        elastic_downbeats.push(elastic_beats[i]);
    }

    let elastic_score = score_phase_alignment(notes, &elastic_beats, &elastic_downbeats);
    if std::env::var_os("KEYSCRIBE_PHASE_DEBUG").is_some() {
        eprintln!(
            "[phase-elastic] base_score={base_score:.4} elastic_score={elastic_score:.4}"
        );
    }

    // Only accept if strictly better than base
    if elastic_score + 0.001 < base_score {
        let mut refined = base.clone();
        refined.beats = elastic_beats;
        refined.downbeats = elastic_downbeats;
        refined
    } else {
        base.clone()
    }
}

/// Validate and optionally rotate the downbeat assignment within the bar.
/// BeatThis can place beat 1 on the wrong beat of the bar (common in jazz
/// without strong drums on beat 1), shifting every note by 1-3 beats. This
/// tests 0..beats_per_bar rotations and picks the one where the designated
/// "beat 1" positions carry the highest aggregate onset energy.
///
/// Only applies the rotation if it wins by a clear margin (`min_margin`,
/// default 0.15 = 15%) to avoid rotating a correct grid on weak evidence.
pub fn validate_downbeat_rotation(
    notes: &[NoteEvent],
    beats: &mut CrossValidatedBeats,
    min_margin: f32,
) {
    let bpb = beats.beats_per_bar.max(2) as usize;
    if notes.is_empty() || beats.beats.len() < 4 || bpb < 2 {
        return;
    }

    // Median beat duration for a tempo-adaptive onset window.
    let mut intervals: Vec<f32> = beats
        .beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d > 0.001)
        .collect();
    if intervals.is_empty() {
        return;
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let beat_duration = intervals[intervals.len() / 2];
    let window = 0.15 * beat_duration;

    // Per-beat onset energy profile: sum of note velocities within ±0.15 ×
    // beat_duration of each beat.
    let mut onset_energy: Vec<f32> = vec![0.0; beats.beats.len()];
    for n in notes {
        if !n.start_time.is_finite() {
            continue;
        }
        for (i, &b) in beats.beats.iter().enumerate() {
            if (n.start_time - b).abs() <= window {
                onset_energy[i] += n.velocity as f32 / 127.0;
            }
        }
    }

    // Downbeat strength per rotation r in 0..beats_per_bar: the candidate
    // downbeats for rotation r are indices r, r+bpb, ... plus a bonus for the
    // secondary strong beat (beat 3 at r + bpb/2 in 4/4).
    let mut strength = vec![0.0f32; bpb];
    for r in 0..bpb {
        let mut db_sum = 0.0f32;
        let mut db_count = 0usize;
        let mut b3_sum = 0.0f32;
        let mut b3_count = 0usize;
        let mut idx = r;
        while idx < onset_energy.len() {
            db_sum += onset_energy[idx];
            db_count += 1;
            let b3 = idx + bpb / 2;
            if b3 < onset_energy.len() {
                b3_sum += onset_energy[b3];
                b3_count += 1;
            }
            idx += bpb;
        }
        let db_mean = db_sum / db_count.max(1) as f32;
        let b3_mean = b3_sum / b3_count.max(1) as f32;
        strength[r] = db_mean + 0.3 * b3_mean;
    }

    let mut best_r = 0usize;
    for (r, &s) in strength.iter().enumerate() {
        if s > strength[best_r] {
            best_r = r;
        }
    }

    if std::env::var_os("KEYSCRIBE_DOWNBEAT_DEBUG").is_some() {
        eprintln!(
            "[downbeat] strength={} best_r={} best={:.3} current={:.3} margin={}",
            strength
                .iter()
                .enumerate()
                .map(|(r, s)| format!("r{r}={s:.3}"))
                .collect::<Vec<String>>()
                .join(" "),
            best_r,
            strength[best_r],
            strength[0],
            min_margin
        );
    }

    // Margin check: the rotation must win by at least `min_margin` over the
    // current grid (r = 0) or we leave a correct grid alone.
    if best_r == 0 || strength[best_r] < (1.0 + min_margin) * strength[0] {
        return;
    }

    // Apply the rotation: set new downbeats starting at the winning rotation
    // offset (best_r). Keeping the full beats array preserves absolute timing
    // and allows find_structural_position to assign beats 0..best_r to the
    // anacrusis / pickup bar (measure 0).
    let mut new_downbeats: Vec<f32> = beats.beats.iter().skip(best_r).step_by(bpb).copied().collect();
    if new_downbeats.len() >= 2 {
        new_downbeats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        beats.downbeats = new_downbeats;
    }
}

/// Run beat-this on up to three audio sources (combined, drums-only, bass-only)
/// and cross-validate the results for robust downbeat detection.
///
/// Audio source priority:
///   1. drums + bass combined (always created if both stems present)
///   2. drums-only (if available)
///   3. bass-only (if available)
///   4. full mix (fallback when no stems)
///
/// Downbeats that appear in multiple sources are retained with higher confidence.
pub fn cross_validate_beat_sources(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
    sample_rate: u32,
    _config: &BeatTrackConfig,
) -> Result<CrossValidatedBeats> {
    // ---- collect audio sources to analyse ----
    let mut source_labels: Vec<String> = Vec::new();
    let mut source_audios: Vec<Vec<f32>> = Vec::new();

    // always create the combined source (drums+bass or fallback)
    let combined = combined_audio(bass_samples, drum_samples, full_mix_samples)?;
    source_labels.push("combined".into());
    source_audios.push(combined);

    // individual drums
    if let Some(d) = drum_samples {
        source_labels.push("drums".into());
        source_audios.push(d.to_vec());
    }

    // individual bass
    if let Some(b) = bass_samples {
        source_labels.push("bass".into());
        source_audios.push(b.to_vec());
    }

    // ---- run beat-this on all sources ----
    let mut tracker = crate::beat_this::create_tracker()
        .context("Failed to initialize beat-this tracker")?;
    let mut results: Vec<BeatTrackResult> = Vec::with_capacity(source_audios.len());
    for (label, samples) in source_labels.iter().zip(source_audios.iter()) {
        let analysis = tracker.analyze_audio(samples, sample_rate)
            .map_err(|e| anyhow!("beat-this failed on {label}: {e}"))?;
        let mut r = BeatTrackResult {
            beats: analysis.beats,
            downbeats: analysis.downbeats,
        };
        correct_beat_metric_level(&mut r);
        results.push(r);
    }

    // ---- cross-validate ----
    cross_validate_results(&results, &source_labels)
}

fn combined_audio(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
) -> Result<Vec<f32>> {
    match (bass_samples, drum_samples) {
        (Some(b), Some(d)) => {
            let len = b.len().max(d.len());
            let mut buf = Vec::with_capacity(len);
            for i in 0..len {
                buf.push(b.get(i).copied().unwrap_or(0.0) + d.get(i).copied().unwrap_or(0.0));
            }
            Ok(buf)
        }
        (Some(b), None) => Ok(b.to_vec()),
        (None, Some(d)) => Ok(d.to_vec()),
        (None, None) => full_mix_samples
            .ok_or_else(|| anyhow!("No audio available for beat tracking"))
            .map(|s| s.to_vec()),
    }
}

fn cross_validate_results(
    results: &[BeatTrackResult],
    _labels: &[String],
) -> Result<CrossValidatedBeats> {
    if results.is_empty() {
        return Err(anyhow!("No beat tracking results to cross-validate"));
    }

    // ---- compute BPM from each source ----
    let mut bpms: Vec<f32> = Vec::new();
    for r in results {
        if r.beats.len() >= 2 {
            let intervals: Vec<f32> = r.beats.windows(2).map(|w| w[1] - w[0]).collect();
            let med = median_of(&intervals);
            if med > 0.001 {
                bpms.push(60.0 / med);
            }
        }
    }
    let bpm = if bpms.is_empty() {
        120.0
    } else {
        median_of(&mut bpms)
    };
    let beat_interval = 60.0 / bpm.max(1.0);

    // ---- cross-validate downbeats ----
    // Pair up downbeats across sources (within 70ms window)
    let tolerance = 0.070f32;
    let primary = &results[0]; // combined result is the reference

    let mut consensus_downbeats: Vec<f32> = Vec::new();
    let mut hit_counts: Vec<u32> = Vec::new();

    for &db in &primary.downbeats {
        let mut count = 1u32; // always present in combined
        // check other sources
        for other in &results[1..] {
            if other.downbeats.iter().any(|&od| (od - db).abs() < tolerance) {
                count += 1;
            }
        }
        consensus_downbeats.push(db);
        hit_counts.push(count);
    }

    // If less than 2 downbeats, beats-per-bar unknown; infer from BPM
    let beats_per_bar = if consensus_downbeats.len() >= 2 {
        let db_intervals: Vec<f32> = consensus_downbeats
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let med_db = median_of(&db_intervals);
        let bpb = (med_db / beat_interval).round() as u32;
        bpb.clamp(2, 8)
    } else {
        4
    };

    // ---- cross-validate beats ----
    // Use primary beats (combined source) as the reference
    let mut consensus_beats: Vec<f32> = primary.beats.clone();

    // If we have too few beats, generate from BPM grid
    if consensus_beats.len() < 4 && bpms.len() >= 1 {
        let duration = primary.beats.last().copied().unwrap_or(30.0) + 2.0;
        let mut t = 0.0f32;
        while t <= duration {
            consensus_beats.push(t);
            t += beat_interval;
        }
        consensus_beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        consensus_beats.dedup_by(|a, b| (*a - *b).abs() < 0.001);
    }

    // ---- confidence ----
    let source_count = results.len() as u32;
    let avg_hits = if consensus_downbeats.is_empty() {
        1.0
    } else {
        hit_counts.iter().sum::<u32>() as f32 / consensus_downbeats.len() as f32
    };
    let confidence = ((avg_hits - 1.0) / (source_count as f32 - 1.0).max(1.0)).clamp(0.0, 1.0);

    // ---- post-process downbeats for consistency ----
    postprocess_downbeats(&mut consensus_beats, &mut consensus_downbeats, beats_per_bar);

    Ok(CrossValidatedBeats {
        downbeats: consensus_downbeats,
        beats: consensus_beats,
        beats_per_bar,
        bpm,
        confidence,
        source_count,
    })
}

/// Post-process beat_this downbeats to fill gaps and ensure consistent bar spacing.
/// beat_this can sometimes miss downbeats in sections with weak percussion, leaving
/// large gaps between downbeats that cause wonky engraving with extended measures.
/// This function detects such gaps and inserts missing downbeats at regular intervals.
/// Also handles anacrusis by propagating a downbeat back to time 0 when the first
/// downbeat arrives after the music has already started.
fn postprocess_downbeats(beats: &mut Vec<f32>, downbeats: &mut Vec<f32>, beats_per_bar: u32) {
    if downbeats.len() < 2 || beats.len() < 4 {
        return;
    }

    // Compute median beat interval from the full beat sequence
    let bi: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0.001).collect();
    if bi.is_empty() {
        return;
    }
    let mut bi_sorted = bi.clone();
    bi_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_beat = bi_sorted[bi_sorted.len() / 2];
    let bar_duration = beats_per_bar as f32 * median_beat;

    // Build a new downbeat list ensuring no gap exceeds 1.5x the expected bar duration
    let mut new_downbeats: Vec<f32> = Vec::new();
    let tolerance = bar_duration * 1.5;

    // Handle anacrusis: if the first downbeat is far from time 0, insert a
    // downbeat at time 0 so the pickup notes have a proper bar reference.
    if downbeats[0] > bar_duration * 0.5 {
        new_downbeats.push(0.0);
    }

    for i in 0..downbeats.len() {
        let current = downbeats[i];
        new_downbeats.push(current);

        if i + 1 < downbeats.len() {
            let next = downbeats[i + 1];
            let gap = next - current;
            if gap > tolerance {
                // Insert missing downbeats at regular bar intervals
                let mut t = current + bar_duration;
                while t + median_beat < next {
                    new_downbeats.push(t);
                    // Also insert the beat positions that belong to these filled bars
                    for b in 1..beats_per_bar {
                        let beat_t = t + b as f32 * median_beat;
                        if beat_t < next && !beats.iter().any(|&x| (x - beat_t).abs() < median_beat * 0.3) {
                            beats.push(beat_t);
                        }
                    }
                    t += bar_duration;
                }
            }
        }
    }

    // Sort and deduplicate
    new_downbeats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    new_downbeats.dedup_by(|a, b| (*a - *b).abs() < median_beat * 0.3);
    *downbeats = new_downbeats;

    // Also fill in beats for the pickup region
    if downbeats.len() >= 2 && downbeats[0] < 0.001 {
        let first_real_db = downbeats[1];
        let mut t = median_beat;
        while t < first_real_db - median_beat * 0.3 {
            if !beats.iter().any(|&x| (x - t).abs() < median_beat * 0.3) {
                beats.push(t);
            }
            t += median_beat;
        }
    }

    beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    beats.dedup_by(|a, b| (*a - *b).abs() < median_beat * 0.3);
}

fn median_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn median_value(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Detect beats directly from note event onsets using autocorrelation.
/// This replaces the external beat-this Python model with a pure-Rust
/// algorithm that uses the transcribed note data for BPM detection.
pub fn detect_beats_from_notes(notes: &[NoteEvent]) -> Option<BeatTrackResult> {
    let config = BpmDetectionConfig::default();
    let tempo = detect_bpm(notes, config)?;

    let beat_duration = 60.0 / tempo.bpm;

    // Find first and last onset for phase alignment and range
    let first_onset = notes
        .iter()
        .filter_map(|n| {
            if n.start_time.is_finite() && n.start_time >= 0.0 {
                Some(n.start_time)
            } else {
                None
            }
        })
        .fold(f32::MAX, |a, b| a.min(b));

    let last_onset = notes
        .iter()
        .filter_map(|n| {
            if n.start_time.is_finite() && n.start_time >= 0.0 {
                Some(n.start_time)
            } else {
                None
            }
        })
        .fold(0.0f32, |a, b| a.max(b));

    if first_onset >= f32::MAX || last_onset <= 0.0 {
        return None;
    }

    // Align first beat: snap to the nearest beat grid position before first onset
    let phase = (first_onset / beat_duration).floor() * beat_duration;

    let end_time = last_onset + beat_duration;
    let mut beats: Vec<f32> = Vec::new();
    let mut t = phase;
    while t <= end_time + 1e-3 {
        beats.push(t);
        t += beat_duration;
    }

    if beats.len() < 4 {
        return None;
    }

    let downbeats: Vec<f32> = Vec::new();

    Some(BeatTrackResult { beats, downbeats })
}

/// Detect beats from bass/drum stem audio for the most reliable BPM reference.
/// Falls back to full mix audio if no stems are available.
pub fn detect_beats_from_stems(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
    sample_rate: u32,
    audio_duration_sec: f32,
) -> Option<(BeatTrackResult, String)> {
    let config = BpmDetectionConfig {
        min_bpm: 40.0,
        max_bpm: 200.0,
        ..Default::default()
    };

    // Try bass + drums combined first (best rhythmic reference)
    let audio = match (bass_samples, drum_samples) {
        (Some(b), Some(d)) => {
            let len = b.len().max(d.len());
            let mut combined = vec![0.0f32; len];
            for (i, &s) in b.iter().enumerate() {
                combined[i] += s;
            }
            for (i, &s) in d.iter().enumerate() {
                combined[i] += s;
            }
            Some((combined, "Bass + Drums".to_string()))
        }
        (Some(b), None) => Some((b.to_vec(), "Bass".to_string())),
        (None, Some(d)) => Some((d.to_vec(), "Drums".to_string())),
        (None, None) => full_mix_samples.map(|s| (s.to_vec(), "Full mix".to_string())),
    };

    let (samples, source_label) = audio?;
    let tempo = detect_bpm_from_audio(&samples, sample_rate, config)?;
    Some((
        generate_beats_from_tempo(&tempo, audio_duration_sec),
        source_label,
    ))
}

fn generate_beats_from_tempo(tempo: &TempoEstimate, duration_sec: f32) -> BeatTrackResult {
    let total_sec = duration_sec.max(10.0) + 2.0;
    let mut beats: Vec<f32> = Vec::new();
    let mut t = 0.0f32;
    while t <= total_sec + 1e-3 {
        beats.push(t);
        t += tempo.beat_duration_sec;
    }
    let downbeats: Vec<f32> = beats.iter().step_by(4).copied().collect();
    BeatTrackResult { beats, downbeats }
}

fn correct_beat_metric_level(result: &mut BeatTrackResult) {
    if result.beats.len() < 4 {
        return;
    }

    // Filter and sort beats
    result.beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.beats.dedup_by(|a, b| (*a - *b).abs() < 1.0e-3);
    if result.beats.len() < 4 {
        return;
    }

    // Compute median beat interval
    let intervals: Vec<f32> = result.beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.001)
        .collect();
    if intervals.len() < 3 {
        return;
    }
    let mut intervals_copy = intervals.clone();
    let beat_interval = median_value(&mut intervals_copy);
    let bpm = 60.0 / beat_interval;

    // Use downbeats to cross-validate the metric level
    if result.downbeats.len() >= 2 {
        let db_intervals: Vec<f32> = result.downbeats
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&d| d > 0.001)
            .collect();
        if !db_intervals.is_empty() {
            let mut db_copy = db_intervals.clone();
            let db_interval = median_value(&mut db_copy);
            let beats_per_bar = (db_interval / beat_interval).round();

            // If beats-per-bar is implausible (< 2.5 or > 6.0), the metric level is wrong.
            // Try doubling (half-time) or halving (double-time) the beat count.
            // beat-this regularly reports half-time on fast music (e.g. 104 BPM
            // for a 208 BPM bebop head), so the doubling guard must cover the
            // full range of plausible half-time reports (up to ~130 BPM, i.e.
            // true 260 BPM — the upper bound of jazz tempos).
            if beats_per_bar < 2.5 && bpm < 130.0 {
                // Too few beats detected per measure → likely half-time
                // Double the number of beats by interpolating midpoints
                let mut new_beats = Vec::with_capacity(result.beats.len() * 2 - 1);
                for w in result.beats.windows(2) {
                    new_beats.push(w[0]);
                    new_beats.push((w[0] + w[1]) * 0.5);
                }
                new_beats.push(*result.beats.last().unwrap());
                result.beats = new_beats;
            } else if beats_per_bar > 6.0 && bpm > 160.0 {
                // Too many beats per bar → likely double-time
                // Halve the number of beats
                result.beats = result.beats.iter().step_by(2).copied().collect();
                if result.beats.len() < 4 {
                    result.beats = intervals_copy.into_iter().step_by(2).collect();
                }
            }
        }
    } else {
        // No downbeats: use simple BPM range heuristic
        if bpm < 130.0 {
            // Likely half-time: double the beat count
            let mut new_beats = Vec::with_capacity(result.beats.len() * 2 - 1);
            for w in result.beats.windows(2) {
                new_beats.push(w[0]);
                new_beats.push((w[0] + w[1]) * 0.5);
            }
            new_beats.push(*result.beats.last().unwrap());
            result.beats = new_beats;
        } else if bpm > 200.0 {
            // Likely double-time: halve the beat count
            result.beats = result.beats.iter().step_by(2).copied().collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

fn note(start: f32) -> NoteEvent {
        NoteEvent {
            id: 0,
            pitch: 60,
            start_time: start,
            end_time: start + 0.2,
            velocity: 100,
            channel: None, is_rearticulation: false,
        }
    }

    fn note_v(start: f32, velocity: u8) -> NoteEvent {
        NoteEvent {
            id: 0,
            pitch: 60,
            start_time: start,
            end_time: start + 0.2,
            velocity,
            channel: None, is_rearticulation: false,
        }
    }

    #[test]
    fn phase_refinement_corrects_quarter_beat_offset() {
        let base = CrossValidatedBeats {
            beats: vec![0.125, 0.625, 1.125, 1.625, 2.125],
            downbeats: vec![0.125, 2.125],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![note(0.0), note(0.5), note(1.0), note(1.5)];
        let refined = refine_beat_phase(&notes, &base);
        assert!((refined.beats[0] - 0.0).abs() < 1e-4);
        assert!((refined.beats[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn phase_refinement_keeps_already_aligned_grid() {
        let base = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            downbeats: vec![0.0, 2.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![note(0.0), note(0.5), note(1.0), note(1.5)];
        let refined = refine_beat_phase(&notes, &base);
        assert_eq!(refined.beats, base.beats);
        assert_eq!(refined.downbeats, base.downbeats);
    }

    #[test]
    fn phase_refinement_corrects_half_time_grid() {
        let base = CrossValidatedBeats {
            beats: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            downbeats: vec![0.0, 2.0, 4.0],
            beats_per_bar: 2,
            bpm: 60.0,
            confidence: 1.0,
            source_count: 1,
        };
let notes = vec![note(0.25), note(0.5), note(0.75), note(1.0), note(1.25)];
        let refined = refine_beat_phase(&notes, &base);
        assert_eq!(refined.beats_per_bar, 4);
        assert!((refined.beats[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn phase_refinement_corrects_sixteenth_beat_offset() {
        // Base grid offset by +0.0625 of the period (a 16th at 120 BPM). The
        // legacy 3-candidate search (shift -0.25/0/+0.25 period) cannot reach
        // this; the 16-candidate sweep shifts by exactly -0.03125.
        let base = CrossValidatedBeats {
            beats: vec![0.03125, 0.53125, 1.03125, 1.53125, 2.03125],
            downbeats: vec![0.03125, 2.03125],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![note(0.0), note(0.5), note(1.0), note(1.5)];
        let refined = refine_beat_phase(&notes, &base);
        assert!((refined.beats[0] - 0.0).abs() < 1e-4);
        assert!((refined.beats[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn score_prefers_onbeat_alignment() {
        // A grid aligned to the onbeat must score better than one aligned to a
        // 16th slot for the same notes.
        let notes = vec![note(0.0), note(0.5), note(1.0), note(1.5)];
        let onbeat = (vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![0.0, 2.0]);
        let sixteenth = (vec![0.125, 0.625, 1.125, 1.625, 2.125], vec![0.125, 2.125]);
        let on_score = score_phase_alignment(&notes, &onbeat.0, &onbeat.1);
        let si_score = score_phase_alignment(&notes, &sixteenth.0, &sixteenth.1);
        assert!(on_score < si_score);
    }

    #[test]
    fn local_refinement_improves_on_coarse_winner() {
        // Base offset of 0.09375s (0.1875 period) at 120 BPM. The 16-candidate
        // sweep finds the exact correction (shift -0.09375) and the refined
        // score is strictly better than the base grid's.
        let base = CrossValidatedBeats {
            beats: vec![0.09375, 0.59375, 1.09375, 1.59375, 2.09375],
            downbeats: vec![0.09375, 2.09375],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![note(0.0), note(0.5), note(1.0), note(1.5)];
        let refined = refine_beat_phase(&notes, &base);
        assert!((refined.beats[0] - 0.0).abs() < 1e-4);
        let base_score = score_phase_alignment(&notes, &base.beats, &base.downbeats);
        let refined_score = score_phase_alignment(&notes, &refined.beats, &refined.downbeats);
        assert!(refined_score + 0.001 < base_score);
    }

    #[test]
    fn downbeat_rotation_keeps_correct_grid() {
        let mut beats = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            downbeats: vec![0.0, 2.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![note(0.0), note(1.0), note(2.0)];
        validate_downbeat_rotation(&notes, &mut beats, 0.15);
        assert_eq!(beats.downbeats, vec![0.0, 2.0]);
        assert_eq!(beats.beats, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn downbeat_rotation_rotates_when_notes_on_beat_2() {
        let mut beats = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            downbeats: vec![0.0, 2.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![note(0.5), note(1.5), note(2.5)];
        validate_downbeat_rotation(&notes, &mut beats, 0.15);
        assert_eq!(beats.downbeats, vec![0.5, 2.5]);
        assert_eq!(beats.beats, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn downbeat_rotation_blocked_by_weak_margin() {
        // Uniform onsets with a slight accent on beats 1 and 5: rotation 1
        // wins narrowly (~10%) but below the 15% margin, so the grid stays.
        let mut beats = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            downbeats: vec![0.0, 2.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![
            note(0.0),
            note_v(0.5, 113),
            note(1.0),
            note(1.5),
            note(2.0),
            note_v(2.5, 113),
        ];
        validate_downbeat_rotation(&notes, &mut beats, 0.15);
        assert_eq!(beats.downbeats, vec![0.0, 2.0]);
        assert_eq!(beats.beats, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

        // With a tiny margin the same evidence is enough to rotate.
        let mut beats2 = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            downbeats: vec![0.0, 2.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        validate_downbeat_rotation(&notes, &mut beats2, 0.05);
        assert_eq!(beats2.downbeats, vec![0.5, 2.5]);
    }

    #[test]
    fn fixed_bpm_refinement_shifts_phase_without_changing_tempo() {
        // Base grid at 120 BPM starting at 0.0, but notes start at 0.057s (57ms offset)
        let base = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            downbeats: vec![0.0, 2.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![
            note(0.057),
            note(0.557),
            note(1.057),
            note(1.557),
        ];
        let refined = refine_beat_phase_fixed_bpm(&notes, &base);
        assert_eq!(refined.bpm, 120.0);
        assert_eq!(refined.beats_per_bar, 4);
        assert!((refined.beats[0] - 0.057).abs() < 0.015);
    }

    #[test]
    fn elastic_grid_recalibrates_tempo_drift_across_measures() {
        // Measure 0: 120 BPM (period 0.5s) at offset 0.0s -> beats at 0.0, 0.5, 1.0, 1.5
        // Measure 1: notes drifted by +30ms -> notes at 2.03, 2.53, 3.03, 3.53
        let base = CrossValidatedBeats {
            beats: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            downbeats: vec![0.0, 2.0, 4.0],
            beats_per_bar: 4,
            bpm: 120.0,
            confidence: 1.0,
            source_count: 1,
        };
        let notes = vec![
            note(0.0),
            note(0.5),
            note(1.0),
            note(1.5),
            note(2.03),
            note(2.53),
            note(3.03),
            note(3.53),
        ];
        let refined = recalibrate_beat_grid_elastic(&notes, &base);
        assert_eq!(refined.bpm, 120.0);
        // Measure 0 beat should stay close to 0.0
        assert!((refined.beats[0] - 0.0).abs() < 0.01);
        // Measure 1 beats should adapt toward +0.03
        assert!(refined.beats[4] > 2.005);
    }
}

