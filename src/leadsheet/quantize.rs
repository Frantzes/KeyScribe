use crate::leadsheet::tempo_map::beat_at_time;
use crate::leadsheet::types::{
    Articulation, BeatAlignedNote, MeterClass, NoteEvent, QuantizedNote, SwingSection, SwingStyle,
    TempoSegment, TimeSignatureSegment,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TiedNote {
    pub pitch: u8,
    pub beat_start: f32,
    pub beat_duration: f32,
    pub velocity: u8,
    pub channel: Option<u8>,
    pub tie_start: bool,
    pub tie_stop: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Beat subdivisions, from coarse to fine.
    pub grids: Vec<f32>,
    /// A finer grid is only chosen when this much better (in beats) than current best.
    pub finer_grid_improvement_threshold: f32,
    /// Duration candidate values (in beats), from coarse to fine.
    pub duration_grids: Vec<f32>,
    /// A finer duration is only chosen when this much better than current best.
    pub duration_finer_grid_improvement_threshold: f32,
    /// Minimum output note duration in beats.
    pub min_duration_beats: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            grids: vec![1.0, 0.5, 0.25, 0.125, 1.0 / 12.0],
            finer_grid_improvement_threshold: 0.03,
            duration_grids: vec![
                4.0,
                3.0,
                2.0,
                1.5,
                1.0,
                0.75,
                0.5,
                0.375,
                0.25,
            ],
            duration_finer_grid_improvement_threshold: 0.015,
            min_duration_beats: 0.25,
        }
    }
}

pub fn quantize_notes(
    notes: &[NoteEvent],
    beat_duration_sec: f32,
    config: &QuantizationConfig,
) -> Vec<QuantizedNote> {
    if beat_duration_sec <= 0.0 || notes.is_empty() {
        return Vec::new();
    }

    let mut quantized = Vec::with_capacity(notes.len());
    for note in notes {
        if !note.start_time.is_finite() || !note.end_time.is_finite() {
            continue;
        }

        let start = note.start_time.max(0.0);
        let end = note.end_time.max(start);
        let beat_start_raw = start / beat_duration_sec;
        let beat_end_raw = end / beat_duration_sec;

        let beat_start = snap_with_coarse_preference(
            beat_start_raw,
            config.grids.as_slice(),
            config.finer_grid_improvement_threshold,
        );
        let raw_duration = (beat_end_raw - beat_start_raw).max(config.min_duration_beats);
        let beat_duration = snap_duration_with_preference(
            raw_duration,
            config.duration_grids.as_slice(),
            config.duration_finer_grid_improvement_threshold,
            config.min_duration_beats,
        )
        .max(config.min_duration_beats);

        quantized.push(QuantizedNote {
            pitch: note.pitch,
            beat_start,
            beat_duration,
            velocity: note.velocity,
            channel: note.channel,
            confidence: 1.0,
            bar_index: 0,
            beat_index: 0,
            intra_beat_pos: 0.0,
            articulation: Articulation::Normal,
            swing_style: SwingStyle::Straight,
            swing_feel: false,
        });
    }

    quantized.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    quantized
}

pub fn quantize_notes_with_tempo_map(
    notes: &[NoteEvent],
    tempo_map: &[TempoSegment],
    config: &QuantizationConfig,
) -> Vec<QuantizedNote> {
    quantize_notes_with_rhythm_map(notes, tempo_map, &[], config)
}

pub fn quantize_notes_with_rhythm_map(
    notes: &[NoteEvent],
    tempo_map: &[TempoSegment],
    time_signature_segments: &[TimeSignatureSegment],
    config: &QuantizationConfig,
) -> Vec<QuantizedNote> {
    if tempo_map.is_empty() || notes.is_empty() {
        return Vec::new();
    }

    let mut quantized = Vec::with_capacity(notes.len());
    for note in notes {
        if !note.start_time.is_finite() || !note.end_time.is_finite() {
            continue;
        }

        let start_time = note.start_time.max(0.0);
        let end_time = note.end_time.max(start_time);

        let beat_start_raw = beat_at_time(start_time, tempo_map);
        let beat_end_raw = beat_at_time(end_time, tempo_map);

        let meter = meter_class_at_beat(beat_start_raw, time_signature_segments);
        let (start_grids, duration_grids) = meter_specific_grids(config, meter);

        let beat_start = snap_with_coarse_preference(
            beat_start_raw,
            start_grids.as_slice(),
            config.finer_grid_improvement_threshold,
        );
        let raw_duration = (beat_end_raw - beat_start_raw).max(config.min_duration_beats);
        let beat_duration = snap_duration_with_preference(
            raw_duration,
            duration_grids.as_slice(),
            config.duration_finer_grid_improvement_threshold,
            config.min_duration_beats,
        )
        .max(config.min_duration_beats);

        quantized.push(QuantizedNote {
            pitch: note.pitch,
            beat_start,
            beat_duration,
            velocity: note.velocity,
            channel: note.channel,
            confidence: 1.0,
            bar_index: 0,
            beat_index: 0,
            intra_beat_pos: 0.0,
            articulation: Articulation::Normal,
            swing_style: SwingStyle::Straight,
            swing_feel: false,
        });
    }

    quantized.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    quantized
}

pub fn quantize_notes_with_ties(
    notes: &[NoteEvent],
    tempo_map: &[TempoSegment],
    time_signature_segments: &[TimeSignatureSegment],
    config: &QuantizationConfig,
) -> Vec<TiedNote> {
    if tempo_map.is_empty() || notes.is_empty() {
        return Vec::new();
    }

    let barline_positions = build_barline_positions(time_signature_segments);
    let mut tied_notes = Vec::with_capacity(notes.len() * 2);

    for note in notes {
        if !note.start_time.is_finite() || !note.end_time.is_finite() {
            continue;
        }

        let start_time = note.start_time.max(0.0);
        let end_time = note.end_time.max(start_time);

        let beat_start_raw = beat_at_time(start_time, tempo_map);
        let beat_end_raw = beat_at_time(end_time, tempo_map);

        let meter = meter_class_at_beat(beat_start_raw, time_signature_segments);
        let (start_grids, duration_grids) = meter_specific_grids(config, meter);

        let snapped_start = snap_with_coarse_preference(
            beat_start_raw,
            start_grids.as_slice(),
            config.finer_grid_improvement_threshold,
        );
        let snapped_end = snap_with_coarse_preference(
            beat_end_raw,
            start_grids.as_slice(),
            config.finer_grid_improvement_threshold,
        );

        let barline_crossings = find_barline_crossings(
            snapped_start,
            snapped_end,
            barline_positions.as_slice(),
        );

        if barline_crossings.is_empty() {
            let raw_duration = (snapped_end - snapped_start).max(config.min_duration_beats);
            let beat_duration = snap_duration_with_preference(
                raw_duration,
                duration_grids.as_slice(),
                config.duration_finer_grid_improvement_threshold,
                config.min_duration_beats,
            )
            .max(config.min_duration_beats);

            tied_notes.push(TiedNote {
                pitch: note.pitch,
                beat_start: snapped_start,
                beat_duration,
                velocity: note.velocity,
                channel: note.channel,
                tie_start: false,
                tie_stop: false,
                confidence: 1.0,
            });
        } else {
            let mut cursor = snapped_start;
            let mut is_first = true;

            for barline in barline_crossings {
                let segment_end = barline;
                let seg_duration = (segment_end - cursor).max(config.min_duration_beats);
                let snapped_duration = snap_duration_with_preference(
                    seg_duration,
                    duration_grids.as_slice(),
                    config.duration_finer_grid_improvement_threshold,
                    config.min_duration_beats,
                )
                .max(config.min_duration_beats);

                tied_notes.push(TiedNote {
                    pitch: note.pitch,
                    beat_start: cursor,
                    beat_duration: snapped_duration,
                    velocity: note.velocity,
                    channel: note.channel,
                    tie_start: !is_first,
                    tie_stop: true,
                    confidence: 1.0,
                });

                cursor = segment_end;
                is_first = false;
            }

            let seg_duration = (snapped_end - cursor).max(config.min_duration_beats);
            let snapped_duration = snap_duration_with_preference(
                seg_duration,
                duration_grids.as_slice(),
                config.duration_finer_grid_improvement_threshold,
                config.min_duration_beats,
            )
            .max(config.min_duration_beats);

            tied_notes.push(TiedNote {
                pitch: note.pitch,
                beat_start: cursor,
                beat_duration: snapped_duration,
                velocity: note.velocity,
                channel: note.channel,
                tie_start: true,
                tie_stop: false,
                confidence: 1.0,
            });
        }
    }

    tied_notes.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    tied_notes
}

fn find_barline_crossings(start_beat: f32, end_beat: f32, barlines: &[f32]) -> Vec<f32> {
    barlines
        .iter()
        .filter(|&&b| b > start_beat && b < end_beat)
        .copied()
        .collect()
}

fn build_barline_positions(time_signature_segments: &[TimeSignatureSegment]) -> Vec<f32> {
    if time_signature_segments.is_empty() {
        return Vec::new();
    }

    let max_beat = time_signature_segments
        .iter()
        .map(|s| s.end_beat)
        .fold(0.0f32, f32::max)
        .min(32768.0);

    let mut barlines = Vec::new();
    let mut sorted = time_signature_segments.to_vec();
    sorted.sort_by(|a, b| {
        a.start_beat
            .partial_cmp(&b.start_beat)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, seg) in sorted.iter().enumerate() {
        let beats_per_measure = seg.beats_per_measure();
        if beats_per_measure <= 0.0 {
            continue;
        }

        let next_start = sorted
            .get(i + 1)
            .map(|s| s.start_beat)
            .unwrap_or(max_beat);

        let mut barline = seg.start_beat + beats_per_measure;
        while barline < next_start && barline <= max_beat {
            barlines.push(barline);
            barline += beats_per_measure;
        }
    }

    barlines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    barlines.dedup_by(|a, b| (*a - *b).abs() < 1.0e-4);
    barlines
}

fn meter_class_at_beat(beat: f32, time_signature_segments: &[TimeSignatureSegment]) -> MeterClass {
    for segment in time_signature_segments {
        if segment.contains_beat(beat) {
            return segment.meter_class();
        }
    }

    MeterClass::SimpleQuadruple
}

fn meter_specific_grids(config: &QuantizationConfig, meter: MeterClass) -> (Vec<f32>, Vec<f32>) {
    let (mut grids, mut durations) = if meter.is_compound() {
        (
            vec![1.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 6.0],
            vec![
                4.0,
                3.0,
                2.0,
                1.5,
                1.0,
                2.0 / 3.0,
                0.5,
                0.25,
            ],
        )
    } else {
        (
            vec![1.0, 0.5, 0.25],
            vec![4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25],
        )
    };

    for g in &config.grids {
        if *g > 0.0 && !grids.iter().any(|x| (*x - *g).abs() < 1.0e-5) {
            grids.push(*g);
        }
    }
    for d in &config.duration_grids {
        if *d > 0.0 && !durations.iter().any(|x| (*x - *d).abs() < 1.0e-5) {
            durations.push(*d);
        }
    }

    (grids, durations)
}

fn snap_with_coarse_preference(value: f32, grids: &[f32], finer_grid_improvement_threshold: f32) -> f32 {
    if !value.is_finite() || grids.is_empty() {
        return value;
    }

    let mut grids = grids
        .iter()
        .copied()
        .filter(|g| *g > 0.0)
        .collect::<Vec<_>>();
    if grids.is_empty() {
        return value;
    }
    grids.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_snap = snap_to_grid(value, grids[0]);
    let mut best_error = (value - best_snap).abs();

    for grid in grids.into_iter().skip(1) {
        let candidate_snap = snap_to_grid(value, grid);
        let candidate_error = (value - candidate_snap).abs();

        if best_error - candidate_error > finer_grid_improvement_threshold {
            best_snap = candidate_snap;
            best_error = candidate_error;
        }
    }

    best_snap
}

fn snap_to_grid(value: f32, grid: f32) -> f32 {
    (value / grid).round() * grid
}

fn snap_duration_with_preference(
    value: f32,
    duration_grids: &[f32],
    duration_finer_grid_improvement_threshold: f32,
    min_duration_beats: f32,
) -> f32 {
    if !value.is_finite() {
        return min_duration_beats;
    }

    let grids = duration_grids
        .iter()
        .copied()
        .filter(|g| *g > 0.0)
        .collect::<Vec<_>>();
    if grids.is_empty() {
        return value.max(min_duration_beats);
    }

    let mut best = grids[0];
    let mut best_error = (value - best).abs();

    for grid in grids.into_iter().skip(1) {
        let error = (value - grid).abs();
        if error + duration_finer_grid_improvement_threshold < best_error {
            best = grid;
            best_error = error;
        }
    }

    best.max(min_duration_beats)
}

// ── Phase 3: Swing Detection ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SwingDetectionConfig {
    pub intra_beat_bin_count: usize,
    pub swing_cluster_threshold: f32,
    pub triplet_cluster_threshold: f32,
    pub min_bars_per_section: u32,
    pub min_notes_for_analysis: usize,
    pub swing_ratio_target: f32,
    pub triplet_positions: Vec<f32>,
}

impl Default for SwingDetectionConfig {
    fn default() -> Self {
        Self {
            intra_beat_bin_count: 20,
            swing_cluster_threshold: 0.55,
            triplet_cluster_threshold: 0.45,
            min_bars_per_section: 4,
            min_notes_for_analysis: 4,
            swing_ratio_target: 2.0 / 3.0,
            triplet_positions: vec![0.0, 1.0 / 3.0, 2.0 / 3.0],
        }
    }
}

pub fn detect_swing(
    aligned: &[BeatAlignedNote],
    _beats_per_bar: u32,
    config: &SwingDetectionConfig,
) -> Vec<SwingSection> {
    if aligned.len() < config.min_notes_for_analysis {
        return vec![SwingSection {
            bar_start: 0,
            bar_end: 1,
            style: SwingStyle::Straight,
            confidence: 0.5,
            swing_ratio: None,
        }];
    }

    let max_bar = aligned.iter().map(|n| n.bar_index).max().unwrap_or(0) + 1;
    if max_bar < 2 {
        return vec![SwingSection {
            bar_start: 0,
            bar_end: max_bar.max(1),
            style: SwingStyle::Straight,
            confidence: 0.5,
            swing_ratio: None,
        }];
    }

    let section_size = config.min_bars_per_section.max(1);
    let num_sections = ((max_bar + section_size - 1) / section_size).max(1);
    let mut sections = Vec::with_capacity(num_sections as usize);

    for s in 0..num_sections {
        let bar_start = s * section_size;
        let bar_end = ((s + 1) * section_size).min(max_bar);
        let section_notes: Vec<&BeatAlignedNote> = aligned
            .iter()
            .filter(|n| n.bar_index >= bar_start && n.bar_index < bar_end)
            .collect();

        if section_notes.len() < config.min_notes_for_analysis {
            sections.push(SwingSection {
                bar_start,
                bar_end,
                style: SwingStyle::Straight,
                confidence: 0.4,
                swing_ratio: None,
            });
            continue;
        }

        let intra_positions: Vec<f32> =
            section_notes.iter().map(|n| n.intra_beat_pos).collect();
        let (style, confidence, swing_ratio) =
            classify_intra_beat_distribution(&intra_positions, config);
        sections.push(SwingSection {
            bar_start,
            bar_end,
            style,
            confidence,
            swing_ratio,
        });
    }

    sections
}

fn classify_intra_beat_distribution(
    positions: &[f32],
    config: &SwingDetectionConfig,
) -> (SwingStyle, f32, Option<f32>) {
    if positions.is_empty() {
        return (SwingStyle::Straight, 0.5, None);
    }

    let bins = config.intra_beat_bin_count.max(10);
    let mut hist = vec![0usize; bins];
    for &pos in positions {
        let idx = ((pos * bins as f32).round() as usize).min(bins - 1);
        hist[idx] += 1;
    }

    let total: usize = hist.iter().sum();
    if total == 0 {
        return (SwingStyle::Straight, 0.5, None);
    }

    let half_bin = (bins as f32 * config.swing_ratio_target).round() as usize;
    let on_beat_bin = 0;

    let on_beat_energy = hist[on_beat_bin] as f32
        + hist.get(1).copied().unwrap_or(0) as f32;
    let swing_energy = hist.get(half_bin).copied().unwrap_or(0) as f32
        + hist.get(half_bin.saturating_sub(1)).copied().unwrap_or(0) as f32
        + hist.get((half_bin + 1).min(bins - 1)).copied().unwrap_or(0) as f32;

    let straight_energy = if bins >= 2 {
        let mid = bins / 2;
        hist[mid] as f32
            + hist.get(mid.saturating_sub(1)).copied().unwrap_or(0) as f32
            + hist.get((mid + 1).min(bins - 1)).copied().unwrap_or(0) as f32
    } else {
        0.0
    };

    let mut triplet_bins = std::collections::BTreeSet::new();
    for &tp in &config.triplet_positions {
        let bin = (tp * bins as f32).round() as usize;
        triplet_bins.insert(bin);
        triplet_bins.insert(bin.saturating_sub(1));
        triplet_bins.insert((bin + 1).min(bins - 1));
    }
    let triplet_energy: f32 = triplet_bins
        .iter()
        .map(|&b| hist.get(b).copied().unwrap_or(0) as f32)
        .sum();

    let total_f = total as f32;
    let swing_score = (on_beat_energy + swing_energy) / total_f;
    let straight_score = (on_beat_energy + straight_energy) / total_f;
    let triplet_score = triplet_energy / total_f;

    let spread = |bin: usize| -> f32 {
        let vals: Vec<f32> = positions
            .iter()
            .copied()
            .filter(|&p| {
                let b = (p * bins as f32).round() as usize;
                b.abs_diff(bin) <= 1
            })
            .collect();
        if vals.len() < 2 {
            return 1.0;
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let variance: f32 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        1.0 - (variance.sqrt() * 5.0).clamp(0.0, 1.0)
    };

    if triplet_score > config.triplet_cluster_threshold && triplet_score > straight_score {
        let conf = (triplet_score * spread(half_bin)).clamp(0.3, 0.98);
        return (SwingStyle::Triplet, conf, Some(1.0 / 3.0));
    }

    if swing_score > config.swing_cluster_threshold && swing_score > straight_score {
        let conf = (swing_score * spread(half_bin)).clamp(0.3, 0.98);
        return (SwingStyle::Swing, conf, Some(config.swing_ratio_target));
    }

    let conf = (straight_score * 0.7 + 0.3).clamp(0.3, 0.95);
    (SwingStyle::Straight, conf, None)
}

// ── Phase 4-5: Context-Aware Quantization ──────────────────────────────────

fn subdivision_grid(swing_style: SwingStyle) -> Vec<f32> {
    match swing_style {
        SwingStyle::Straight => {
            vec![0.0, 0.25, 0.5, 0.75]
        }
        SwingStyle::Swing => {
            vec![0.0, 2.0 / 3.0]
        }
        SwingStyle::Triplet => {
            vec![0.0, 1.0 / 3.0, 2.0 / 3.0]
        }
    }
}

fn snap_intra_beat_pos(pos: f32, grid: &[f32]) -> (f32, f32) {
    if grid.is_empty() {
        return (0.0, 1.0);
    }
    let mut best = grid[0];
    let mut best_err = (pos - best).abs();
    for &g in grid.iter().skip(1) {
        let err = (pos - g).abs();
        if err < best_err {
            best = g;
            best_err = err;
        }
    }
    (best, best_err)
}

fn compute_next_onset_duration(
    note: &BeatAlignedNote,
    next_notes: &[&BeatAlignedNote],
) -> f32 {
    let next_onset = next_notes
        .iter()
        .filter(|n| n.pitch != note.pitch || n.original_start_time > note.original_start_time + 0.001)
        .map(|n| n.original_start_time)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match next_onset {
        Some(t) if t > note.original_start_time => {
            let dur_sec = t - note.original_start_time;
            dur_sec / note.beat_duration_sec.max(0.001)
        }
        _ => {
            let dur_sec = (note.original_end_time - note.original_start_time).max(0.05);
            dur_sec / note.beat_duration_sec.max(0.001)
        }
    }
}

fn quantize_duration(
    raw_duration: f32,
    _subdivision: f32,
    swing_style: SwingStyle,
) -> f32 {
    let candidates: Vec<f32> = match swing_style {
        SwingStyle::Straight => {
            vec![4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25]
        }
        SwingStyle::Swing => {
            vec![4.0, 2.0, 1.0, 0.5, 2.0 / 3.0, 0.25]
        }
        SwingStyle::Triplet => {
            vec![4.0, 2.0, 1.0, 0.5, 2.0 / 3.0, 0.25, 1.0 / 3.0, 1.0 / 6.0]
        }
    };

    let mut best = candidates[0];
    let mut best_err = (raw_duration - best).abs();
    for &c in candidates.iter().skip(1) {
        let err = (raw_duration - c).abs();
        if err < best_err {
            best = c;
            best_err = err;
        }
    }
    best.max(0.25)
}

// ── Main Aligned Quantization Entry Point ──────────────────────────────────

pub fn quantize_aligned_notes(
    aligned: &[BeatAlignedNote],
    swing_sections: &[SwingSection],
    beats_per_bar: u32,
) -> Vec<QuantizedNote> {
    if aligned.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<&BeatAlignedNote> = aligned.iter().collect();
    sorted.sort_by(|a, b| {
        a.original_start_time
            .partial_cmp(&b.original_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut quantized: Vec<QuantizedNote> = Vec::with_capacity(sorted.len());

    for (i, note) in sorted.iter().enumerate() {
        let swing_section = swing_sections
            .iter()
            .find(|s| s.contains_bar(note.bar_index))
            .or_else(|| swing_sections.first());
        let style = swing_section.map(|s| s.style).unwrap_or(SwingStyle::Straight);
        let swing_feel = style == SwingStyle::Swing;

        let sub_grid = subdivision_grid(style);
        let (snapped_pos, snap_error) = snap_intra_beat_pos(note.intra_beat_pos, &sub_grid);

        let subdivision_pos = snapped_pos;
        let beat_start_in_bar = note.beat_index as f32 + subdivision_pos;
        let beat_start = note.bar_index as f32 * beats_per_bar as f32 + beat_start_in_bar;

        let next_notes: Vec<&BeatAlignedNote> = sorted
            .iter()
            .skip(i + 1)
            .filter(|n| n.pitch == note.pitch)
            .copied()
            .collect();
        let raw_dur = compute_next_onset_duration(note, &next_notes);
        let duration = quantize_duration(raw_dur, subdivision_pos, style);

        let articulation = if note.original_end_time - note.original_start_time < 0.08 {
            Articulation::Staccato
        } else {
            Articulation::Normal
        };

        let confidence = (1.0 - snap_error * 2.0).clamp(0.3, 1.0);

        quantized.push(QuantizedNote {
            pitch: note.pitch,
            beat_start,
            beat_duration: duration,
            velocity: note.velocity,
            channel: note.channel,
            confidence,
            bar_index: note.bar_index,
            beat_index: note.beat_index,
            intra_beat_pos: note.intra_beat_pos,
            articulation,
            swing_style: style,
            swing_feel,
        });
    }

    quantized.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    quantized
}

// ── Phase 9: Grace Notes ────────────────────────────────────────────────────

pub fn detect_grace_notes(
    quantized: &[QuantizedNote],
    aligned: &[BeatAlignedNote],
) -> Vec<QuantizedNote> {
    if quantized.is_empty() {
        return Vec::new();
    }

    let mut result = quantized.to_vec();
    let mut grace_indices: Vec<usize> = Vec::new();

    for i in 0..result.len().saturating_sub(1) {
        let current = &result[i];
        let next = &result[i + 1];

        let (start_time, end_time) = if i < aligned.len() {
            (aligned[i].original_start_time, aligned[i].original_end_time)
        } else {
            (current.beat_start, current.beat_start + current.beat_duration)
        };

        let next_start = if i + 1 < aligned.len() {
            aligned[i + 1].original_start_time
        } else {
            next.beat_start
        };

        let is_short = (end_time - start_time) < 0.08;
        let is_close_to_next = (next_start - start_time).abs() < 0.6;

        if is_short && is_close_to_next && current.pitch != next.pitch {
            grace_indices.push(i);
        }
    }

    for &idx in grace_indices.iter().rev() {
        if let Some(note) = result.get_mut(idx) {
            note.articulation = Articulation::Grace;
            note.beat_duration = 0.0;
        }
    }

    result
}

// ── Phase 9: Articulation Detection ─────────────────────────────────────────

pub fn detect_articulation(
    quantized: &[QuantizedNote],
    aligned: &[BeatAlignedNote],
) -> Vec<QuantizedNote> {
    let mut result = quantized.to_vec();

    for note in result.iter_mut() {
        let aligned_note = aligned
            .iter()
            .filter(|a| a.pitch == note.pitch)
            .min_by(|a, b| {
                let da = (a.original_start_time - note.beat_start * a.beat_duration_sec).abs();
                let db = (b.original_start_time - note.beat_start * b.beat_duration_sec).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(a) = aligned_note {
            let actual_dur_sec = (a.original_end_time - a.original_start_time).max(0.001);
            let notated_dur_sec = (note.beat_duration * a.beat_duration_sec).max(0.001);
            let ratio = actual_dur_sec / notated_dur_sec;

            if actual_dur_sec < 0.08 || ratio < 0.5 {
                note.articulation = Articulation::Staccato;
            } else if ratio > 0.9 {
                note.articulation = Articulation::Tenuto;
            } else {
                note.articulation = Articulation::Normal;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start: f32, end: f32) -> NoteEvent {
        NoteEvent {
            pitch: 60,
            start_time: start,
            end_time: end,
            velocity: 100,
            channel: None,
        }
    }

    #[test]
    fn uses_finer_grid_when_error_improves_enough() {
        let config = QuantizationConfig::default();
        let snapped = snap_with_coarse_preference(
            1.48,
            config.grids.as_slice(),
            config.finer_grid_improvement_threshold,
        );
        assert!((snapped - 1.5).abs() < 0.0001);
    }

    #[test]
    fn keeps_coarse_grid_for_small_gain() {
        let mut config = QuantizationConfig::default();
        config.finer_grid_improvement_threshold = 0.05;

        let snapped = snap_with_coarse_preference(
            1.04,
            config.grids.as_slice(),
            config.finer_grid_improvement_threshold,
        );
        assert!((snapped - 1.0).abs() < 0.0001);
    }

    #[test]
    fn quantizes_note_start_and_duration() {
        let notes = vec![note(0.49, 1.01)];
        let quantized = quantize_notes(&notes, 0.5, &QuantizationConfig::default());
        assert_eq!(quantized.len(), 1);
        assert!((quantized[0].beat_start - 1.0).abs() < 0.0001);
        assert!((quantized[0].beat_duration - 1.0).abs() < 0.0001);
    }

    #[test]
    fn quantizes_with_piecewise_tempo_map() {
        let notes = vec![note(7.9, 8.4)];
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

        let quantized = quantize_notes_with_tempo_map(&notes, map.as_slice(), &QuantizationConfig::default());
        assert_eq!(quantized.len(), 1);
        assert!(quantized[0].beat_start >= 15.5);
    }

    #[test]
    fn prefers_triplet_snap_in_compound_meter() {
        let notes = vec![note(0.0, 0.165)];
        let map = vec![TempoSegment {
            start_time_sec: 0.0,
            end_time_sec: 4.0,
            bpm: 120.0,
            beat_duration_sec: 0.5,
            beat_offset: 0.0,
        }];
        let signatures = vec![TimeSignatureSegment {
            start_beat: 0.0,
            end_beat: 8.0,
            numerator: 6,
            denominator: 8,
            confidence: 0.9,
            meter_class: MeterClass::CompoundDuple,
        }];

        let quantized = quantize_notes_with_rhythm_map(
            &notes,
            map.as_slice(),
            signatures.as_slice(),
            &QuantizationConfig::default(),
        );
        assert_eq!(quantized.len(), 1);
        assert!((quantized[0].beat_duration - (1.0 / 3.0)).abs() < 0.06);
    }

    #[test]
    fn splits_note_across_barline_into_tied_segments() {
        let notes = vec![note(1.4, 2.6)];
        let map = vec![TempoSegment {
            start_time_sec: 0.0,
            end_time_sec: 4.0,
            bpm: 120.0,
            beat_duration_sec: 0.5,
            beat_offset: 0.0,
        }];
        let signatures = vec![TimeSignatureSegment {
            start_beat: 0.0,
            end_beat: 8.0,
            numerator: 4,
            denominator: 4,
            confidence: 0.9,
            meter_class: MeterClass::SimpleQuadruple,
        }];

        let tied = quantize_notes_with_ties(
            &notes,
            map.as_slice(),
            signatures.as_slice(),
            &QuantizationConfig::default(),
        );

        assert!(tied.len() >= 2, "Note crossing barline at beat 4 should split into multiple segments");
        let first_segment = &tied[0];
        assert!(first_segment.beat_duration <= 4.0, "First segment should end at or before barline");
        let last_segment = tied.last().unwrap();
        assert!(last_segment.tie_start, "Last segment should have tie_start=true");
    }

    #[test]
    fn no_ties_when_note_within_single_bar() {
        let notes = vec![note(0.1, 0.4)];
        let map = vec![TempoSegment {
            start_time_sec: 0.0,
            end_time_sec: 4.0,
            bpm: 120.0,
            beat_duration_sec: 0.5,
            beat_offset: 0.0,
        }];
        let signatures = vec![TimeSignatureSegment {
            start_beat: 0.0,
            end_beat: 8.0,
            numerator: 4,
            denominator: 4,
            confidence: 0.9,
            meter_class: MeterClass::SimpleQuadruple,
        }];

        let tied = quantize_notes_with_ties(
            &notes,
            map.as_slice(),
            signatures.as_slice(),
            &QuantizationConfig::default(),
        );

        assert_eq!(tied.len(), 1);
        assert!(!tied[0].tie_start && !tied[0].tie_stop);
    }

    // ── New Pipeline Tests ──

    fn aligned_note(
        pitch: u8,
        start: f32,
        end: f32,
        beat_idx: u32,
        bar: u32,
        intra: f32,
    ) -> BeatAlignedNote {
        BeatAlignedNote {
            pitch,
            velocity: 100,
            channel: None,
            original_start_time: start,
            original_end_time: end,
            beat_index: beat_idx,
            bar_index: bar,
            intra_beat_pos: intra,
            prev_beat_time: start - 0.1,
            next_beat_time: start + 0.4,
            beat_duration_sec: 0.5,
        }
    }

    #[test]
    fn detects_straight_section() {
        let notes: Vec<BeatAlignedNote> = (0..16)
            .map(|i| aligned_note(60, i as f32 * 0.5, i as f32 * 0.5 + 0.2, i % 4, i / 4, 0.0))
            .collect();
        let sections = detect_swing(&notes, 4, &SwingDetectionConfig::default());
        assert!(!sections.is_empty());
        assert_eq!(sections[0].style, SwingStyle::Straight);
    }

    #[test]
    fn snap_intra_pos_to_on_beat() {
        let (pos, err) = snap_intra_beat_pos(0.03, &[0.0, 0.5]);
        assert!((pos - 0.0).abs() < 0.001);
        assert!(err < 0.05);
    }

    #[test]
    fn snap_intra_pos_to_swing() {
        let grid = vec![0.0, 2.0 / 3.0];
        let (pos, err) = snap_intra_beat_pos(0.65, &grid);
        assert!((pos - 2.0 / 3.0).abs() < 0.001);
        assert!(err < 0.05);
    }

    #[test]
    fn quantize_aligned_produces_output() {
        let notes = vec![
            aligned_note(60, 0.0, 0.4, 0, 0, 0.0),
            aligned_note(62, 0.5, 0.9, 1, 0, 0.0),
        ];
        let swing = vec![SwingSection {
            bar_start: 0,
            bar_end: 1,
            style: SwingStyle::Straight,
            confidence: 0.9,
            swing_ratio: None,
        }];
        let q = quantize_aligned_notes(&notes, &swing, 4);
        assert_eq!(q.len(), 2);
        assert!((q[0].beat_start - 0.0).abs() < 0.001);
        assert!((q[1].beat_start - 1.0).abs() < 0.001);
        assert_eq!(q[0].bar_index, 0);
        assert_eq!(q[0].swing_style, SwingStyle::Straight);
    }

    #[test]
    fn quantize_with_swing_sets_swing_feel_flag() {
        let notes = vec![
            aligned_note(60, 0.0, 0.4, 0, 0, 0.0),
            aligned_note(62, 0.35, 0.7, 0, 0, 0.65),
        ];
        let swing = vec![SwingSection {
            bar_start: 0,
            bar_end: 1,
            style: SwingStyle::Swing,
            confidence: 0.8,
            swing_ratio: Some(2.0 / 3.0),
        }];
        let q = quantize_aligned_notes(&notes, &swing, 4);
        assert!(q.iter().any(|n| n.swing_feel));
        assert!(q.iter().any(|n| n.swing_style == SwingStyle::Swing));
    }

    #[test]
    fn grace_note_detection_marks_short_notes() {
        let notes = vec![
            aligned_note(60, 0.0, 0.03, 0, 0, 0.0),
            aligned_note(62, 0.5, 0.9, 1, 0, 0.0),
        ];
        let swing = vec![SwingSection {
            bar_start: 0,
            bar_end: 1,
            style: SwingStyle::Straight,
            confidence: 0.9,
            swing_ratio: None,
        }];
        let q = quantize_aligned_notes(&notes, &swing, 4);
        let with_grace = detect_grace_notes(&q, &notes);
        let grace_count = with_grace
            .iter()
            .filter(|n| n.articulation == Articulation::Grace)
            .count();
        assert_eq!(grace_count, 1);
    }

    #[test]
    fn articulation_staccato_when_short() {
        let notes = vec![aligned_note(60, 0.0, 0.05, 0, 0, 0.0)];
        let swing = vec![SwingSection {
            bar_start: 0,
            bar_end: 1,
            style: SwingStyle::Straight,
            confidence: 0.9,
            swing_ratio: None,
        }];
        let q = quantize_aligned_notes(&notes, &swing, 4);
        let with_art = detect_articulation(&q, &notes);
        assert_eq!(with_art[0].articulation, Articulation::Staccato);
    }
}
