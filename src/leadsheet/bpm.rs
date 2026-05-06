use crate::leadsheet::types::NoteEvent;

#[derive(Debug, Clone, Copy)]
pub struct BpmDetectionConfig {
    pub min_bpm: f32,
    pub max_bpm: f32,
    pub onset_bin_size_sec: f32,
    pub harmonic_score_ratio: f32,
    pub subdivision_tolerance_beats: f32,
}

impl Default for BpmDetectionConfig {
    fn default() -> Self {
        Self {
            min_bpm: 60.0,
            max_bpm: 180.0,
            onset_bin_size_sec: 0.01,
            harmonic_score_ratio: 0.9,
            subdivision_tolerance_beats: 0.07,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEstimate {
    pub bpm: f32,
    pub beat_duration_sec: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
struct TempoCandidate {
    bpm: f32,
    raw_score: f32,
}

pub fn detect_bpm(notes: &[NoteEvent], config: BpmDetectionConfig) -> Option<TempoEstimate> {
    let onsets = collect_onsets(notes);
    if onsets.len() < 3 || config.min_bpm <= 0.0 || config.max_bpm <= config.min_bpm {
        return None;
    }

    let primary = best_candidate_from_autocorrelation(&onsets, config)?;
    let resolved = resolve_harmonic_ambiguity(primary, &onsets, config);

    let beat_duration_sec = 60.0 / resolved.bpm;
    let confidence = (resolved.raw_score / onsets.len() as f32).clamp(0.0, 1.0);

    Some(TempoEstimate {
        bpm: resolved.bpm,
        beat_duration_sec,
        confidence,
    })
}

fn collect_onsets(notes: &[NoteEvent]) -> Vec<f32> {
    let mut onsets: Vec<f32> = notes
        .iter()
        .filter_map(|n| {
            if n.start_time.is_finite() && n.start_time >= 0.0 {
                Some(n.start_time)
            } else {
                None
            }
        })
        .collect();

    onsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Merge near-identical onsets to reduce duplicate-hit bias from stacked notes.
    let mut deduped = Vec::with_capacity(onsets.len());
    for onset in onsets {
        let keep = deduped
            .last()
            .map(|last: &f32| (onset - *last).abs() > 0.005)
            .unwrap_or(true);
        if keep {
            deduped.push(onset);
        }
    }

    deduped
}

fn best_candidate_from_autocorrelation(
    onsets: &[f32],
    config: BpmDetectionConfig,
) -> Option<TempoCandidate> {
    let period_min = 60.0 / config.max_bpm;
    let period_max = 60.0 / config.min_bpm;
    let duration = *onsets.last()?;
    if duration <= 0.0 {
        return None;
    }

    let bin_size = config.onset_bin_size_sec.max(0.001);
    let signal_len = (duration / bin_size).ceil() as usize + 2;

    let mut onset_bins = Vec::with_capacity(onsets.len());
    let mut signal = vec![false; signal_len + 1];
    for &t in onsets {
        let idx = ((t / bin_size).round() as usize).min(signal_len);
        signal[idx] = true;
        onset_bins.push(idx);
    }

    let lag_min = ((period_min / bin_size).round() as usize).max(1);
    let lag_max = ((period_max / bin_size).round() as usize).max(lag_min + 1);
    let tol_bins = ((0.02 / bin_size).round() as usize).max(1);

    let mut best: Option<TempoCandidate> = None;
    for lag in lag_min..=lag_max {
        let score = autocorrelation_lag_score(&signal, &onset_bins, lag, tol_bins);
        let bpm = 60.0 / (lag as f32 * bin_size);
        if bpm < config.min_bpm || bpm > config.max_bpm {
            continue;
        }

        match best {
            Some(existing) if score <= existing.raw_score => {}
            _ => {
                best = Some(TempoCandidate {
                    bpm,
                    raw_score: score,
                })
            }
        }
    }

    best
}

fn autocorrelation_lag_score(
    signal: &[bool],
    onset_bins: &[usize],
    lag: usize,
    tol_bins: usize,
) -> f32 {
    let mut score = 0.0f32;

    for &idx in onset_bins {
        let target = idx + lag;
        if target >= signal.len() {
            continue;
        }

        let start = target.saturating_sub(tol_bins);
        let end = (target + tol_bins).min(signal.len() - 1);
        let mut best_dist: Option<usize> = None;
        for j in start..=end {
            if !signal[j] {
                continue;
            }

            let dist = j.abs_diff(target);
            best_dist = Some(match best_dist {
                Some(current) => current.min(dist),
                None => dist,
            });
        }

        if let Some(dist) = best_dist {
            // Exact lag alignment gets full credit, nearby bins are discounted.
            let local = 1.0 - (dist as f32 / (tol_bins as f32 + 1.0));
            score += local.max(0.0);
        }
    }

    score
}

fn autocorrelation_score_for_bpm(onsets: &[f32], bpm: f32, config: BpmDetectionConfig) -> f32 {
    if onsets.is_empty() || bpm <= 0.0 {
        return 0.0;
    }

    let duration = match onsets.last() {
        Some(v) => *v,
        None => return 0.0,
    };
    if duration <= 0.0 {
        return 0.0;
    }

    let bin_size = config.onset_bin_size_sec.max(0.001);
    let signal_len = (duration / bin_size).ceil() as usize + 2;
    let mut onset_bins = Vec::with_capacity(onsets.len());
    let mut signal = vec![false; signal_len + 1];

    for &t in onsets {
        let idx = ((t / bin_size).round() as usize).min(signal_len);
        signal[idx] = true;
        onset_bins.push(idx);
    }

    let lag = ((60.0 / bpm) / bin_size).round() as usize;
    let tol_bins = ((0.02 / bin_size).round() as usize).max(1);
    autocorrelation_lag_score(&signal, &onset_bins, lag.max(1), tol_bins)
}

fn resolve_harmonic_ambiguity(
    primary: TempoCandidate,
    onsets: &[f32],
    config: BpmDetectionConfig,
) -> TempoCandidate {
    let mut candidates = vec![primary];

    let half_bpm = primary.bpm / 2.0;
    if half_bpm >= config.min_bpm {
        candidates.push(TempoCandidate {
            bpm: half_bpm,
            raw_score: autocorrelation_score_for_bpm(onsets, half_bpm, config),
        });
    }

    let double_bpm = primary.bpm * 2.0;
    if double_bpm <= config.max_bpm {
        candidates.push(TempoCandidate {
            bpm: double_bpm,
            raw_score: autocorrelation_score_for_bpm(onsets, double_bpm, config),
        });
    }

    let primary_score = primary.raw_score;
    let mut best = primary;
    let mut best_alignment = subdivision_alignment_score(
        onsets,
        60.0 / primary.bpm,
        config.subdivision_tolerance_beats,
    );

    for candidate in candidates {
        if candidate.bpm == primary.bpm {
            continue;
        }

        if candidate.raw_score < primary_score * config.harmonic_score_ratio {
            continue;
        }

        let alignment = subdivision_alignment_score(
            onsets,
            60.0 / candidate.bpm,
            config.subdivision_tolerance_beats,
        );
        if alignment > best_alignment {
            best = candidate;
            best_alignment = alignment;
        }
    }

    best
}

fn subdivision_alignment_score(onsets: &[f32], beat_duration_sec: f32, tolerance_beats: f32) -> f32 {
    if beat_duration_sec <= 0.0 || onsets.is_empty() {
        return 0.0;
    }

    let subdivisions = [1.0f32, 0.5, 0.25, 0.125, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 12.0];
    let mut score = 0.0;

    for &onset in onsets {
        let beat_pos = onset / beat_duration_sec;
        let mut best_err = 1.0f32;

        for &grid in &subdivisions {
            let snapped = (beat_pos / grid).round() * grid;
            let err = (beat_pos - snapped).abs();
            if err < best_err {
                best_err = err;
            }
        }

        if best_err <= tolerance_beats {
            score += 1.0 - (best_err / tolerance_beats);
        }
    }

    score / onsets.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metronome_notes(bpm: f32, beats: usize) -> Vec<NoteEvent> {
        let mut notes = Vec::with_capacity(beats);
        let beat = 60.0 / bpm;
        for i in 0..beats {
            let start = i as f32 * beat;
            notes.push(NoteEvent {
                pitch: 60,
                start_time: start,
                end_time: start + 0.1,
                velocity: 100,
                channel: None,
            });
        }
        notes
    }

    #[test]
    fn detects_120_bpm_for_regular_quarter_onsets() {
        let notes = make_metronome_notes(120.0, 64);
        let tempo = detect_bpm(&notes, BpmDetectionConfig::default()).unwrap();
        assert!((tempo.bpm - 120.0).abs() < 1.5);
    }

    #[test]
    fn returns_none_for_too_few_notes() {
        let notes = make_metronome_notes(120.0, 2);
        assert!(detect_bpm(&notes, BpmDetectionConfig::default()).is_none());
    }
}
