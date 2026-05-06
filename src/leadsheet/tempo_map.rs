use crate::leadsheet::bpm::{detect_bpm, BpmDetectionConfig, TempoEstimate};
use crate::leadsheet::types::{MeterClass, NoteEvent, TempoSegment, TimeSignatureSegment};

#[derive(Debug, Clone, Copy)]
pub struct TempoMapConfig {
    pub window_sec: f32,
    pub step_sec: f32,
    pub bpm_change_ratio_threshold: f32,
    pub min_segment_sec: f32,
    pub min_window_notes: usize,
}

impl Default for TempoMapConfig {
    fn default() -> Self {
        Self {
            window_sec: 8.0,
            step_sec: 2.0,
            bpm_change_ratio_threshold: 0.12,
            min_segment_sec: 4.0,
            min_window_notes: 5,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TimeSignatureConfig {
    pub min_notes_per_segment: usize,
    pub downbeat_emphasis: f32,
}

impl Default for TimeSignatureConfig {
    fn default() -> Self {
        Self {
            min_notes_per_segment: 8,
            downbeat_emphasis: 1.35,
        }
    }
}

#[derive(Debug, Clone)]
struct TempoWindowSample {
    start_sec: f32,
    bpm: f32,
}

pub fn detect_tempo_map(
    notes: &[NoteEvent],
    bpm_config: BpmDetectionConfig,
    map_config: TempoMapConfig,
) -> Option<(TempoEstimate, Vec<TempoSegment>)> {
    let global = detect_bpm(notes, bpm_config)?;
    let duration_sec = notes
        .iter()
        .map(|n| n.end_time.max(n.start_time))
        .fold(0.0f32, f32::max)
        .max(0.01);

    let mut samples = collect_window_samples(notes, bpm_config, map_config, global.bpm);
    if samples.is_empty() {
        return Some((
            global,
            vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: duration_sec,
                bpm: global.bpm,
                beat_duration_sec: global.beat_duration_sec,
                beat_offset: 0.0,
            }],
        ));
    }

    smooth_samples(samples.as_mut_slice());
    let mut segments = samples_to_segments(samples.as_slice(), duration_sec, map_config);
    if segments.is_empty() {
        segments.push(TempoSegment {
            start_time_sec: 0.0,
            end_time_sec: duration_sec,
            bpm: global.bpm,
            beat_duration_sec: global.beat_duration_sec,
            beat_offset: 0.0,
        });
    }

    normalize_segment_offsets(segments.as_mut_slice());
    Some((global, segments))
}

pub fn tempo_map_from_beats(beat_times: &[f32]) -> Option<(TempoEstimate, Vec<TempoSegment>)> {
    let mut beats: Vec<f32> = beat_times
        .iter()
        .copied()
        .filter(|t| t.is_finite() && *t >= 0.0)
        .collect();
    if beats.len() < 2 {
        return None;
    }

    beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    beats.dedup_by(|a, b| (*a - *b).abs() < 1.0e-3);
    if beats.len() < 2 {
        return None;
    }

    let mut intervals: Vec<f32> = beats
        .windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .filter(|dt| *dt > 1.0e-3)
        .collect();
    if intervals.is_empty() {
        return None;
    }

    let median_interval = median_value(intervals.as_mut_slice());
    let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
    let variance = intervals
        .iter()
        .map(|dt| (dt - mean_interval).powi(2))
        .sum::<f32>()
        / intervals.len() as f32;
    let stddev = variance.sqrt();
    let cv = (stddev / mean_interval.max(1.0e-3)).clamp(0.0, 1.0);
    let confidence = (1.0 - cv).clamp(0.1, 0.99);

    let global_bpm = (60.0 / median_interval).clamp(40.0, 260.0);
    let tempo = TempoEstimate {
        bpm: global_bpm,
        beat_duration_sec: 60.0 / global_bpm,
        confidence,
    };

    let mut segments = Vec::new();
    let first_interval = intervals.first().copied().unwrap_or(median_interval);
    let first_beat = beats[0];
    if first_beat > 1.0e-3 && first_interval > 0.0 {
        let beat_offset = -first_beat / first_interval;
        segments.push(TempoSegment {
            start_time_sec: 0.0,
            end_time_sec: first_beat,
            bpm: (60.0 / first_interval).clamp(40.0, 260.0),
            beat_duration_sec: first_interval,
            beat_offset,
        });
    }

    let mut beat_offset = 0.0f32;
    for window in beats.windows(2) {
        let start = window[0];
        let end = window[1];
        let interval = end - start;
        if interval <= 1.0e-3 {
            continue;
        }

        segments.push(TempoSegment {
            start_time_sec: start,
            end_time_sec: end,
            bpm: (60.0 / interval).clamp(40.0, 260.0),
            beat_duration_sec: interval,
            beat_offset,
        });
        beat_offset += 1.0;
    }

    let last_beat = *beats.last().unwrap_or(&0.0);
    let tail_interval = intervals.last().copied().unwrap_or(median_interval);
    if tail_interval > 0.0 {
        segments.push(TempoSegment {
            start_time_sec: last_beat,
            end_time_sec: last_beat + tail_interval,
            bpm: (60.0 / tail_interval).clamp(40.0, 260.0),
            beat_duration_sec: tail_interval,
            beat_offset,
        });
    }

    if segments.is_empty() {
        None
    } else {
        Some((tempo, segments))
    }
}

pub fn beat_at_time(time_sec: f32, tempo_map: &[TempoSegment]) -> f32 {
    if tempo_map.is_empty() {
        return 0.0;
    }

    let t = time_sec.max(0.0);
    if let Some(first) = tempo_map.first() {
        if t < first.start_time_sec {
            return first.beat_offset + (t - first.start_time_sec) / first.beat_duration_sec;
        }
    }
    for segment in tempo_map {
        if t >= segment.start_time_sec && t < segment.end_time_sec {
            return segment.beat_offset + (t - segment.start_time_sec) / segment.beat_duration_sec;
        }
    }

    if let Some(last) = tempo_map.last() {
        return last.beat_offset + (t - last.start_time_sec).max(0.0) / last.beat_duration_sec;
    }

    0.0
}

fn median_value(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

pub fn detect_time_signature_segments(
    notes: &[NoteEvent],
    tempo_map: &[TempoSegment],
    config: TimeSignatureConfig,
) -> Vec<TimeSignatureSegment> {
    if tempo_map.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::with_capacity(tempo_map.len());
    for tempo_segment in tempo_map {
        let start_beat = tempo_segment.beat_offset;
        let end_beat = start_beat
            + (tempo_segment.end_time_sec - tempo_segment.start_time_sec).max(0.0)
                / tempo_segment.beat_duration_sec.max(0.001);

        let (numerator, denominator, confidence) = infer_signature_for_tempo_segment(
            notes,
            tempo_map,
            tempo_segment,
            start_beat,
            end_beat,
            config,
        );

        segments.push(TimeSignatureSegment {
            start_beat,
            end_beat,
            numerator,
            denominator,
            confidence,
            meter_class: MeterClass::from_signature(numerator, denominator),
        });
    }

    merge_adjacent_time_signatures(segments.as_mut_slice());
    segments
}

pub fn build_barline_beats(
    time_signature_segments: &[TimeSignatureSegment],
    max_beat: f32,
) -> Vec<f32> {
    let target_max = max_beat.max(0.0);
    let mut out = vec![0.0f32];

    if time_signature_segments.is_empty() {
        let mut beat = 4.0f32;
        while beat <= target_max + 1.0e-4 {
            out.push(beat);
            beat += 4.0;
        }
        if out.last().copied().unwrap_or(0.0) < target_max {
            out.push(target_max);
        }
        return out;
    }

    let mut sorted = time_signature_segments.to_vec();
    sorted.sort_by(|a, b| {
        a.start_beat
            .partial_cmp(&b.start_beat)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cursor = 0.0f32;
    for seg in &sorted {
        let seg_start = seg.start_beat.max(cursor);
        let seg_end = seg.end_beat.max(seg_start).min(target_max.max(seg_start));
        let beats_per_measure = seg.numerator.max(1) as f32 * (4.0 / seg.denominator.max(1) as f32);
        let step = beats_per_measure.max(0.25);

        if out.last().copied().unwrap_or(0.0) < seg_start {
            out.push(seg_start);
        }

        let mut beat = seg_start + step;
        while beat < seg_end - 1.0e-4 {
            out.push(beat);
            beat += step;
        }

        cursor = seg_end;
    }

    let mut tail = out.last().copied().unwrap_or(0.0);
    let default_step = 4.0;
    while tail + default_step <= target_max + 1.0e-4 {
        tail += default_step;
        out.push(tail);
    }
    if out.last().copied().unwrap_or(0.0) < target_max {
        out.push(target_max);
    }

    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1.0e-4);
    out
}

fn infer_signature_for_tempo_segment(
    notes: &[NoteEvent],
    tempo_map: &[TempoSegment],
    tempo_segment: &TempoSegment,
    start_beat: f32,
    end_beat: f32,
    config: TimeSignatureConfig,
) -> (u8, u8, f32) {
    let segment_notes: Vec<&NoteEvent> = notes
        .iter()
        .filter(|n| {
            n.start_time >= tempo_segment.start_time_sec && n.start_time < tempo_segment.end_time_sec
        })
        .collect();

    if segment_notes.len() < config.min_notes_per_segment {
        return (4, 4, 0.20);
    }

    let candidates: [(u8, u8, f32); 5] = [
        (2, 4, 0.88),
        (3, 4, 0.84),
        (4, 4, 1.00),
        (6, 8, 0.78),
        (12, 8, 0.74),
    ];

    let mut best = (4u8, 4u8, 0.20f32);
    let mut best_score = f32::MIN;

    for (numerator, denominator, prior) in candidates {
        let score = score_time_signature_candidate(
            segment_notes.as_slice(),
            tempo_map,
            start_beat,
            end_beat,
            numerator,
            config,
        ) + prior;

        if score > best_score {
            best_score = score;
            let confidence = ((score - 0.2) / 3.0).clamp(0.05, 0.99);
            best = (numerator, denominator, confidence);
        }
    }

    best
}

fn score_time_signature_candidate(
    notes: &[&NoteEvent],
    tempo_map: &[TempoSegment],
    start_beat: f32,
    end_beat: f32,
    numerator: u8,
    config: TimeSignatureConfig,
) -> f32 {
    let bins = numerator.max(1) as usize;
    let mut histogram = vec![0.0f32; bins];
    let mut total_weight = 0.0f32;

    for note in notes {
        let beat = beat_at_time(note.start_time.max(0.0), tempo_map);
        if beat < start_beat || beat >= end_beat {
            continue;
        }

        let local_beat = beat - start_beat;
        let phase = local_beat.rem_euclid(numerator as f32);
        let bin = phase.round() as usize % bins;
        let quant_error = (phase - bin as f32).abs();

        let velocity_weight = (note.velocity as f32 / 127.0).clamp(0.2, 1.0);
        let timing_weight = (1.0 - quant_error * 0.7).clamp(0.15, 1.0);
        let weight = velocity_weight * timing_weight;

        histogram[bin] += weight;
        total_weight += weight;
    }

    if total_weight <= 1.0e-5 {
        return 0.0;
    }

    let mut best_rotation_score = f32::MIN;
    for rot in 0..bins {
        let downbeat_weight = histogram[rot] * config.downbeat_emphasis.max(1.0);
        let mut other_sum = 0.0f32;
        for (idx, value) in histogram.iter().enumerate() {
            if idx != rot {
                other_sum += *value;
            }
        }
        let other_mean = other_sum / (bins.saturating_sub(1)).max(1) as f32;
        let rotation_score = downbeat_weight - other_mean + total_weight / bins as f32;
        if rotation_score > best_rotation_score {
            best_rotation_score = rotation_score;
        }
    }

    best_rotation_score / total_weight.max(1.0)
}

fn merge_adjacent_time_signatures(segments: &mut [TimeSignatureSegment]) {
    if segments.len() < 2 {
        return;
    }

    for i in 1..segments.len() {
        let prev = segments[i - 1].clone();
        let curr = &mut segments[i];
        if prev.numerator == curr.numerator && prev.denominator == curr.denominator {
            curr.start_beat = prev.start_beat;
            curr.confidence = curr.confidence.max(prev.confidence);
        }
    }
}

fn collect_window_samples(
    notes: &[NoteEvent],
    bpm_config: BpmDetectionConfig,
    map_config: TempoMapConfig,
    fallback_bpm: f32,
) -> Vec<TempoWindowSample> {
    let duration_sec = notes
        .iter()
        .map(|n| n.end_time.max(n.start_time))
        .fold(0.0f32, f32::max);
    if duration_sec <= 0.0 {
        return vec![];
    }

    let mut out = Vec::new();
    let window = map_config.window_sec.max(2.0);
    let step = map_config.step_sec.max(0.5);

    let mut t = 0.0f32;
    while t <= duration_sec {
        let end = (t + window).min(duration_sec + 0.001);
        let window_notes: Vec<NoteEvent> = notes
            .iter()
            .filter(|n| n.start_time >= t && n.start_time < end)
            .cloned()
            .collect();

        let bpm = if window_notes.len() >= map_config.min_window_notes {
            detect_bpm(window_notes.as_slice(), bpm_config)
                .map(|tempo| tempo.bpm)
                .unwrap_or(fallback_bpm)
        } else {
            fallback_bpm
        };

        out.push(TempoWindowSample { start_sec: t, bpm });
        t += step;
    }

    out
}

fn smooth_samples(samples: &mut [TempoWindowSample]) {
    if samples.len() < 3 {
        return;
    }

    for i in 1..samples.len() - 1 {
        let a = samples[i - 1].bpm;
        let b = samples[i].bpm;
        let c = samples[i + 1].bpm;
        let mut vals = [a, b, c];
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        samples[i].bpm = vals[1];
    }
}

fn samples_to_segments(
    samples: &[TempoWindowSample],
    duration_sec: f32,
    map_config: TempoMapConfig,
) -> Vec<TempoSegment> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_start = 0.0f32;
    let mut current_bpm = samples[0].bpm;

    for sample in samples.iter().skip(1) {
        let ratio = ((sample.bpm - current_bpm).abs() / current_bpm.max(1.0)).max(0.0);
        let seg_len = sample.start_sec - current_start;

        if ratio >= map_config.bpm_change_ratio_threshold && seg_len >= map_config.min_segment_sec {
            segments.push(build_segment(current_start, sample.start_sec, current_bpm));
            current_start = sample.start_sec;
            current_bpm = sample.bpm;
        } else {
            // Blend gradually to avoid jitter while retaining local tempo tendency.
            current_bpm = current_bpm * 0.75 + sample.bpm * 0.25;
        }
    }

    segments.push(build_segment(current_start, duration_sec.max(current_start + 0.001), current_bpm));

    merge_short_segments(segments.as_mut_slice(), map_config.min_segment_sec);
    segments
}

fn build_segment(start_time_sec: f32, end_time_sec: f32, bpm: f32) -> TempoSegment {
    let clamped_bpm = bpm.clamp(40.0, 260.0);
    TempoSegment {
        start_time_sec,
        end_time_sec,
        bpm: clamped_bpm,
        beat_duration_sec: 60.0 / clamped_bpm,
        beat_offset: 0.0,
    }
}

fn merge_short_segments(segments: &mut [TempoSegment], min_segment_sec: f32) {
    if segments.len() < 2 {
        return;
    }

    for i in 0..segments.len() {
        let seg_len = segments[i].end_time_sec - segments[i].start_time_sec;
        if seg_len < min_segment_sec {
            if i > 0 {
                segments[i].bpm = (segments[i].bpm + segments[i - 1].bpm) * 0.5;
                segments[i].beat_duration_sec = 60.0 / segments[i].bpm.max(1.0);
                segments[i].start_time_sec = segments[i - 1].end_time_sec;
            }
        }
    }
}

fn normalize_segment_offsets(segments: &mut [TempoSegment]) {
    if segments.is_empty() {
        return;
    }

    segments[0].start_time_sec = 0.0;
    segments[0].beat_offset = 0.0;
    for i in 1..segments.len() {
        segments[i].start_time_sec = segments[i - 1].end_time_sec;
    }

    for i in 0..segments.len() {
        let prev_end = if i == 0 {
            segments[i].start_time_sec
        } else {
            segments[i - 1].end_time_sec
        };
        if segments[i].end_time_sec <= prev_end {
            segments[i].end_time_sec = prev_end + 0.001;
        }
    }

    let mut beat_cursor = 0.0f32;
    for seg in segments {
        seg.beat_offset = beat_cursor;
        let duration = (seg.end_time_sec - seg.start_time_sec).max(0.0);
        beat_cursor += duration / seg.beat_duration_sec.max(0.001);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notes_with_tempo_change() -> Vec<NoteEvent> {
        let mut out = Vec::new();
        for i in 0..16 {
            let t = i as f32 * 0.5;
            out.push(NoteEvent {
                pitch: 64,
                start_time: t,
                end_time: t + 0.15,
                velocity: 100,
                channel: None,
            });
        }
        let base = 8.0;
        for i in 0..16 {
            let t = base + i as f32 * 0.4;
            out.push(NoteEvent {
                pitch: 67,
                start_time: t,
                end_time: t + 0.12,
                velocity: 96,
                channel: None,
            });
        }
        out
    }

    #[test]
    fn detects_multiple_tempo_segments() {
        let notes = notes_with_tempo_change();
        let (_, map) = detect_tempo_map(
            notes.as_slice(),
            BpmDetectionConfig::default(),
            TempoMapConfig::default(),
        )
        .expect("tempo map should exist");

        assert!(!map.is_empty());
        assert!(map.len() >= 2);
    }

    #[test]
    fn converts_time_to_piecewise_beats() {
        let map = vec![
            TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 8.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            },
            TempoSegment {
                start_time_sec: 8.0,
                end_time_sec: 16.0,
                bpm: 150.0,
                beat_duration_sec: 0.4,
                beat_offset: 16.0,
            },
        ];

        let beat = beat_at_time(9.2, map.as_slice());
        assert!((beat - 19.0).abs() < 0.1);
    }

    #[test]
    fn detects_34_from_regular_three_beat_accent_pattern() {
        let mut notes = Vec::new();
        for i in 0..12 {
            let t = i as f32 * 1.5;
            notes.push(NoteEvent {
                pitch: 64,
                start_time: t,
                end_time: t + 0.15,
                velocity: 118,
                channel: None,
            });
        }

        let tempo_map = vec![TempoSegment {
            start_time_sec: 0.0,
            end_time_sec: 18.0,
            bpm: 120.0,
            beat_duration_sec: 0.5,
            beat_offset: 0.0,
        }];

        let signatures = detect_time_signature_segments(
            notes.as_slice(),
            tempo_map.as_slice(),
            TimeSignatureConfig {
                min_notes_per_segment: 4,
                ..TimeSignatureConfig::default()
            },
        );
        assert!(!signatures.is_empty());
        assert!(signatures.iter().any(|s| {
            (s.numerator == 3 && s.denominator == 4)
                || (s.numerator == 6 && s.denominator == 8)
        }));
    }
}
