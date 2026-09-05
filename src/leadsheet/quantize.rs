use crate::leadsheet::tempo_map::beat_at_time;
use crate::leadsheet::types::{
    Articulation, BeatAlignedNote, MeterClass, NoteEvent, QuantizedNote, SwingSection, SwingStyle,
    TempoSegment, TimeSignatureSegment,
};
use std::path::Path;

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
            grids: vec![1.0, 0.5, 0.25],
            finer_grid_improvement_threshold: 0.03,
            duration_grids: vec![4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25],
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
            id: note.id,
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
            id: note.id,
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
                    tie_start: true,
                    tie_stop: !is_first,
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
                tie_start: false,
                tie_stop: true,
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
    let mut grids = match meter {
        MeterClass::CompoundDuple | MeterClass::CompoundQuadruple => {
            vec![1.0, 0.5, 1.0/3.0, 2.0/3.0]
        }
        _ => vec![1.0, 0.5, 0.25],
    };
    let mut durations = match meter {
        MeterClass::CompoundDuple | MeterClass::CompoundQuadruple => {
            vec![1.5, 1.0, 0.5, 1.0/3.0]
        }
        _ => vec![4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25],
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

    let mut best = (value / grids[0]).round() * grids[0];
    let mut best_error = (value - best).abs();

    for grid in grids.into_iter().skip(1) {
        let candidate = (value / grid).round() * grid;
        let error = (value - candidate).abs();
        if error + duration_finer_grid_improvement_threshold < best_error {
            best = candidate;
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

    let total_f = total as f32;
    let swing_score = (on_beat_energy + swing_energy) / total_f;
    let straight_score = (on_beat_energy + straight_energy) / total_f;

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
            // Straight feel still has to admit triplet 8ths and 16ths inside a
            // beat: the omnibook line mixes straight 8ths with occasional
            // triplet runs. `snap_intra_beat_pos` keeps coarse (8th/quarter)
            // precedence, so a slightly-late straight 8th stays on the 8th.
            vec![
                0.0,
                1.0 / 6.0,
                1.0 / 3.0,
                0.5,
                2.0 / 3.0,
                5.0 / 6.0,
                1.0,
            ]
        }
        SwingStyle::Swing => {
            vec![0.0, 2.0 / 3.0, 1.0]
        }
        SwingStyle::Triplet => {
            vec![
                0.0,
                1.0 / 6.0,
                1.0 / 3.0,
                0.5,
                2.0 / 3.0,
                5.0 / 6.0,
                1.0,
            ]
        }
    }
}

/// Subdivision vocabulary per rhythmic context. Selecting the grid from local
/// onset spacing (P2) instead of one mixed nearest-slot grid removes the
/// ±1/6-beat slips: straight 8ths/16ths only ever hit straight slots, while
/// triplet runs get their own clean slots.
fn straight_grid() -> Vec<f32> {
    vec![0.0, 0.25, 0.5, 0.75, 1.0]
}

fn eighth_triplet_grid() -> Vec<f32> {
    vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
}

fn sixteenth_triplet_grid() -> Vec<f32> {
    vec![0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0]
}

fn snap_intra_beat_pos(
    pos: f32,
    grid: &[f32],
    beat_in_bar: u32,
    beats_per_bar: u32,
) -> (f32, f32) {
    if grid.is_empty() {
        return (0.0, 1.0);
    }
    // Complexity-aware snapping: coarse slots (onbeat / straight 8th) are
    // preferred over fine slots (16th / triplet) by a small margin, so
    // ±60 ms of onset jitter at fast tempos cannot flip a straight 8th onto
    // the dotted-8th/16th slot (measured on Confirmation: clean melody
    // onsets jitter ±20-75 ms while a 16th slot at 208 BPM is 72 ms). The
    // margin is far smaller than a genuine 16th's distance from the coarse
    // grid, so real subdivisions still win.
    let slot_penalty = |g: f32| -> f32 {
        let frac = (g * 12.0).round() / 12.0;
        let base = if frac.abs() < 1e-3 || (frac - 0.5).abs() < 1e-3 || (frac - 1.0).abs() < 1e-3 {
            0.0 // onbeats and straight 8ths
        } else if (frac - 0.25).abs() < 1e-3 || (frac - 0.75).abs() < 1e-3 {
            0.045 // 16ths
        } else if (frac - 1.0 / 3.0).abs() < 2e-2 || (frac - 2.0 / 3.0).abs() < 2e-2 {
            0.03 // triplet 8ths
        } else {
            0.06 // 16th triplets
        };
        // Metrical strength prior: on strong beats (1 and 3 in 4/4), the
        // onbeat slot (0.0) gets a small extra pull. A genuine 16th at large
        // distance still wins, but when the onset is ambiguously between an
        // 8th and a 16th, the onbeat wins on strong beats.
        let metrical_bonus = if frac.abs() < 1e-3 || (frac - 1.0).abs() < 1e-3 {
            match beat_in_bar {
                0 => 0.02, // beat 1
                b if beats_per_bar > 2 && b == beats_per_bar / 2 => 0.01, // beat 3
                _ => 0.0,
            }
        } else {
            0.0
        };
        base - metrical_bonus
    };
    let mut best = grid[0];
    let mut best_cost = (pos - grid[0]).abs() + slot_penalty(grid[0]);
    for &g in grid.iter().skip(1) {
        let cost = (pos - g).abs() + slot_penalty(g);
        if cost < best_cost {
            best = g;
            best_cost = cost;
        }
    }
    (best, (pos - best).abs())
}

/// Tier B1 MERGE_INTO_PREVIOUS decoding: given one vocabulary index per
/// detection, return a keep mask. A detection whose token index is outside
/// `LEARNED_TOKEN_TABLE` (the 13th class) is a fragment of the PREVIOUS
/// detection and is not emitted (a leading merge with no previous note is
/// dropped). A v1 12-class model produces no out-of-vocab indices, so this
/// is a no-op passthrough for it.
pub(crate) fn merge_keep_mask(tokens: &[u32]) -> Vec<bool> {
    let vocab = LEARNED_TOKEN_TABLE.len() as u32;
    tokens.iter().map(|&t| t < vocab).collect()
}

// ── Main Aligned Quantization Entry Point ──────────────────────────────────

/// Duration fill (rhythm alternative 1): monophonic lead-sheet lines are
/// mostly continuous — a note lasts until the next onset. Detected note ends
/// (adaptive-release times) are far noisier than onsets, so duration error is
/// dominated by under-filling. This pass extends each note's duration to the
/// next snapped onset UNLESS there is substantial audible silence after it
/// (rest evidence): `raw_gap_beats > fill_max_gap` keeps the note short.
/// `KEYSCRIBE_FILL_GAP=<beats>` overrides the threshold (debug/sweep), and
/// `KEYSCRIBE_FILL_GAP=0` disables the pass.
pub fn fill_melody_durations(
    quantized: &mut [QuantizedNote],
    aligned: &[BeatAlignedNote],
) {
    if quantized.len() < 2 {
        return;
    }
    let mut max_gap = 0.30f32;
    let mut enabled = true;
    if let Ok(v) = std::env::var("KEYSCRIBE_FILL_GAP") {
        if let Ok(f) = v.parse::<f32>() {
            if f <= 0.0 {
                enabled = false;
            } else {
                max_gap = f;
            }
        }
    }
    if !enabled {
        return;
    }
    let raw_by_id: std::collections::HashMap<u32, &BeatAlignedNote> =
        aligned.iter().map(|n| (n.id, n)).collect();

    let mut debug_filled = 0usize;
    for i in 0..quantized.len().saturating_sub(1) {
        let cur = &quantized[i];
        let next = &quantized[i + 1];
        let snapped_gap = next.beat_start - cur.beat_start;
        if snapped_gap <= cur.beat_duration || snapped_gap <= 0.0 {
            continue; // already spans, or next onset is not after this note
        }
        let Some(raw) = raw_by_id.get(&cur.id).copied() else { continue };
        let raw_gap_beats = (next_raw_start(&quantized[i + 1], &raw_by_id)
            - raw.original_end_time)
            / raw.beat_duration_sec.max(0.001);
        if raw_gap_beats <= max_gap {
            quantized[i].beat_duration = snapped_gap.min(4.0);
            debug_filled += 1;
        }
    }
    if std::env::var_os("KEYSCRIBE_QUANT_DEBUG").is_some() {
        eprintln!(
            "[quant-debug] duration fill: extended {} of {} notes (max gap {:.2} beats)",
            debug_filled,
            quantized.len(),
            max_gap
        );
    }
}

/// Raw onset (seconds) of the note following `q`, via the aligned table.
fn next_raw_start(
    q: &QuantizedNote,
    raw_by_id: &std::collections::HashMap<u32, &BeatAlignedNote>,
) -> f32 {
    raw_by_id
        .get(&q.id)
        .map(|n| n.original_start_time)
        .unwrap_or(f32::INFINITY)
}

/// Sequence-level subdivision grid selection: dynamically identifies straight,
/// 8th-triplet, and 16th-triplet rhythmic regimes across note sequences so that
/// coherent runs (e.g. 3-note triplet licks or straight 16th runs) do not suffer
/// single-note grid hopping.
fn compute_subdivision_grids(beat_pos: &[f32]) -> Vec<Vec<f32>> {
    let n = beat_pos.len();
    if n == 0 {
        return Vec::new();
    }
    let mut grids = vec![straight_grid(); n];
    let gap = |i: usize| -> f32 {
        if i + 1 < n {
            (beat_pos[i + 1] - beat_pos[i]).abs()
        } else {
            f32::INFINITY
        }
    };
    let near = |g: f32, target: f32, tol: f32| (g - target).abs() <= tol;

    // 16th triplets (target 1/6):
    for i in 0..n {
        let prev_gap = if i > 0 { gap(i - 1) } else { f32::INFINITY };
        let next_gap = gap(i);
        let is_triplet_16th = (prev_gap.is_finite() && next_gap.is_finite() && near(prev_gap + next_gap, 1.0 / 3.0, 0.07))
            || near(prev_gap, 1.0 / 6.0, 0.045)
            || near(next_gap, 1.0 / 6.0, 0.045);
        if is_triplet_16th {
            grids[i] = sixteenth_triplet_grid();
        }
    }

    // 8th triplets (target 1/3):
    for i in 0..n {
        if grids[i] == sixteenth_triplet_grid() {
            continue;
        }
        let prev_gap = if i > 0 { gap(i - 1) } else { f32::INFINITY };
        let next_gap = gap(i);
        let is_triplet_8th = (prev_gap.is_finite() && next_gap.is_finite() && near(prev_gap + next_gap, 2.0 / 3.0, 0.10))
            || near(prev_gap, 1.0 / 3.0, 0.055)
            || near(next_gap, 1.0 / 3.0, 0.055);
        if is_triplet_8th {
            grids[i] = eighth_triplet_grid();
        }
    }

    // Triplet bridge: if note i-1 and note i+1 are 8th triplets, note i is also 8th triplet
    for i in 1..n.saturating_sub(1) {
        if grids[i - 1] == eighth_triplet_grid() && grids[i + 1] == eighth_triplet_grid() {
            grids[i] = eighth_triplet_grid();
        }
    }

    grids
}

pub fn quantize_aligned_notes(
    aligned: &[BeatAlignedNote],
    swing_sections: &[SwingSection],
    beats_per_bar: u32,
) -> Vec<QuantizedNote> {
    if aligned.is_empty() {
        return Vec::new();
    }

    // Pass 1: snap every onset position to the subdivision grid. Durations are
    // derived afterwards from the gaps between CONSECUTIVE SNAPPED onsets,
    // not from raw onset deltas: a triplet 8th run (0.096s spacing at 208 BPM)
    // snaps to 1/3-beat slots, and the gap between two snapped triplet onsets
    // is exactly 1/3 beat — no extra quantization needed.
    let mut sorted: Vec<&BeatAlignedNote> = aligned.iter().collect();
    sorted.sort_by(|a, b| {
        a.original_start_time
            .partial_cmp(&b.original_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    struct Snapped<'a> {
        note: &'a BeatAlignedNote,
        style: SwingStyle,
        snapped_pos: f32,
        beat_start: f32,
        snap_error: f32,
    }

    // P2: choose the subdivision vocabulary from LOCAL onset spacing. Using a
    // single mixed grid makes every note nearest-snap independently, which
    // lands inside a triplet run on the wrong straight slot (the ±1/6-beat
    let beat_pos: Vec<f32> = sorted
        .iter()
        .map(|n| n.beat_index as f32 + n.intra_beat_pos)
        .collect();
    let sub_grids = compute_subdivision_grids(&beat_pos);

    let mut snapped: Vec<Snapped> = Vec::with_capacity(sorted.len());
    for (i, note) in sorted.iter().enumerate() {
        let swing_section = swing_sections
            .iter()
            .find(|s| s.contains_bar(note.bar_index))
            .or_else(|| swing_sections.first());
        let style = swing_section.map(|s| s.style).unwrap_or(SwingStyle::Straight);
        let sub_grid = if style == SwingStyle::Swing {
            subdivision_grid(style)
        } else {
            sub_grids.get(i).cloned().unwrap_or_else(straight_grid)
        };
        let (snapped_pos, snap_error) = snap_intra_beat_pos(
            note.intra_beat_pos,
            &sub_grid,
            note.beat_index % beats_per_bar,
            beats_per_bar,
        );
        snapped.push(Snapped {
            note,
            style,
            snapped_pos,
            beat_start: note.beat_index as f32 + snapped_pos,
            snap_error,
        });
    }
    snapped.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.note.pitch.cmp(&b.note.pitch))
    });

    if std::env::var_os("KEYSCRIBE_QUANT_DEBUG").is_some() {
        for (i, s) in snapped.iter().take(12).enumerate() {
            let note = s.note;
            let sub_grid = if s.style == SwingStyle::Swing {
                subdivision_grid(s.style)
            } else {
                sub_grids.get(i).cloned().unwrap_or_else(straight_grid)
            };
            eprintln!(
                "[quant-debug] pitch={} raw={:.4}s intra={:.3} beat_idx={} style={:?} grid={:?} -> snapped={:.3} beat_start={:.3}",
                note.pitch, note.original_start_time, note.intra_beat_pos, note.beat_index, s.style, sub_grid, s.snapped_pos, s.beat_start
            );
        }
        // Evidence summary for rhythm diagnosis: where do raw onsets actually
        // sit inside the beat (24 bins), and where do they snap?
        let mut hist = vec![0usize; 24];
        let mut snap_counts: std::collections::BTreeMap<u32, usize> =
            std::collections::BTreeMap::new();
        let mut snap_err_sum = 0.0f32;
        for s in &snapped {
            let bin = (s.note.intra_beat_pos.clamp(0.0, 0.9999) * 24.0) as usize;
            hist[bin] += 1;
            *snap_counts.entry((s.snapped_pos * 24.0).round() as u32).or_default() += 1;
            snap_err_sum += s.snap_error;
        }
        for (bin, &count) in hist.iter().enumerate() {
            if count > 0 {
                eprintln!(
                    "[quant-debug] intra hist bin {:.3}-{:.3}: {}",
                    bin as f32 / 24.0,
                    (bin + 1) as f32 / 24.0,
                    count
                );
            }
        }
        let mut dests: Vec<String> = snap_counts
            .iter()
            .map(|(slot, count)| format!("{:.3}x{}", *slot as f32 / 24.0, count))
            .collect();
        dests.sort();
        eprintln!(
            "[quant-debug] snap destinations: {} | mean snap err {:.4} over {} notes",
            dests.join(" "),
            snap_err_sum / snapped.len().max(1) as f32,
            snapped.len()
        );
    }

    // Pass 2: infer duration from the local sequence. For contiguous attacks,
    // use the gap between snapped onsets so triplet runs remain coherent. When
    // there is a real performed gap after the note ends, preserve its detected
    // duration instead of stretching it to the next unrelated onset.
    let mut quantized: Vec<QuantizedNote> = Vec::with_capacity(snapped.len());
    for (i, s) in snapped.iter().enumerate() {
        let raw_duration = (s.note.original_end_time - s.note.original_start_time)
            .max(0.05)
            / s.note.beat_duration_sec.max(0.001);
        let duration_value = if let Some(next) = snapped.get(i + 1) {
            let snapped_gap = (next.beat_start - s.beat_start).max(0.0);
            let raw_gap_after_note = (next.note.original_start_time - s.note.original_end_time)
                / s.note.beat_duration_sec.max(0.001);
            let raw_onset_spacing = (next.note.original_start_time - s.note.original_start_time)
                / s.note.beat_duration_sec.max(0.001);
            // Contiguous attacks (straight 8ths, triplet runs) use the snapped
            // onset gap so triplet runs keep exact 1/3 durations. The end-of-
            // note test alone is unreliable because staccato/triplet note
            // detections end early, so require a dense onset spacing (<= ~0.6
            // beat) — a genuinely sparse gap (a rest) falls through to the
            // note's own detected duration.
            let contiguous = snapped_gap > 0.0
                && raw_onset_spacing <= 0.6
                && (raw_onset_spacing - snapped_gap).abs() <= 0.3;
            if (raw_gap_after_note <= 0.10 || contiguous) && snapped_gap > 0.0 {
                snapped_gap
            } else {
                raw_duration.min(snapped_gap.max(1.0 / 12.0))
            }
        } else {
            raw_duration
        };
        let duration = snap_duration_with_preference(
            duration_value,
            &duration_grid_for(s.style),
            0.03,
            1.0 / 12.0,
        )
        .max(1.0 / 12.0);

        let confidence = (1.0 - s.snap_error * 2.0).clamp(0.3, 1.0);

        quantized.push(QuantizedNote {
            id: s.note.id,
            pitch: s.note.pitch,
            beat_start: s.beat_start,
            beat_duration: duration,
            velocity: s.note.velocity,
            channel: s.note.channel,
            confidence,
            bar_index: s.note.bar_index,
            beat_index: s.note.beat_index % beats_per_bar,
            intra_beat_pos: s.note.intra_beat_pos,
            articulation: Articulation::Normal,
            swing_style: s.style,
            swing_feel: s.style == SwingStyle::Swing,
        });
    }

    quantized.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    // Strictly-monophonic guarantee: a lead-sheet melody has at most one note
    // per grid slot. Two fast notes can quantize to the same slot (collisions);
    // keep the better-aligned one (higher confidence, or longer for same
    // pitch) so the writer produces a single voice instead of splitting into
    // spurious parallel voices.
    let mut mono: Vec<QuantizedNote> = Vec::with_capacity(quantized.len());
    for note in quantized {
        match mono.last_mut() {
            Some(last) if (last.beat_start - note.beat_start).abs() < 1e-4 => {
                let replace = if last.pitch == note.pitch {
                    note.beat_duration > last.beat_duration
                } else {
                    note.confidence > last.confidence
                };
                if replace {
                    *last = note;
                }
            }
            _ =>         mono.push(note),
        }
    }

    mono
}

// ── Tier A1: rhythm merge & coarsening pass ────────────────────────────────

/// Post-quantization rhythm merge & coarsening configuration.
#[derive(Debug, Clone)]
pub struct RhythmCoarsenConfig {
    /// Whether the pass runs at all. `--no-rhythm-coarsen` sets this false so
    /// the legacy (unchanged) output is produced.
    pub enabled: bool,
    /// Minimum fraction of inter-onset gaps that must be 16th-level (in
    /// (0.05, 0.45) beats) for a bar to be voted COARSE and re-snapped to the
    /// 8th-note grid. Raising this makes coarsening more conservative.
    pub coarse_vote_ratio: f32,
}

impl Default for RhythmCoarsenConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            coarse_vote_ratio: 0.70,
        }
    }
}

/// Collect the distinct snapped onset offsets within one bar, collapsing
/// offsets that are < 0.05 beats apart.
fn distinct_bar_offsets(notes: &[QuantizedNote], beats_per_bar: u32) -> Vec<f32> {
    let mut offsets: Vec<f32> = notes
        .iter()
        .map(|n| n.beat_start - (n.bar_index as f32 * beats_per_bar as f32))
        .collect();
    offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<f32> = Vec::with_capacity(offsets.len());
    for o in offsets {
        if out.last().map_or(true, |last| o - last >= 0.05) {
            out.push(o);
        }
    }
    out
}

/// Grid vote: a bar is COARSE (8th-note grid) when it contains more onsets
/// than a full 8th grid (guarding genuine swing 8ths whose swung onsets land
/// on 16th slots), its inter-onset gaps are dominated by 8th-or-larger
/// spacing, AND it contains at least one 16th-spaced gap (the over-split
/// signature). Requiring `min_notes` prevents the pass from chopping genuine
/// swing passages whose swung onsets snap to 16th slots.
fn bar_is_coarse(offsets: &[f32], coarse_vote_ratio: f32, min_notes: usize) -> bool {
    if offsets.len() < min_notes.max(2) {
        return false;
    }
    let mut coarse = 0usize;
    let mut fine = 0usize;
    let mut has_16th_gap = false;
    for w in offsets.windows(2) {
        let gap = w[1] - w[0];
        if gap >= 0.45 {
            coarse += 1;
        } else if gap > 0.05 {
            fine += 1;
        }
        if (gap - 0.25).abs() < 0.02 {
            has_16th_gap = true;
        }
    }
    let total = (coarse + fine) as f32;
    if total <= 0.0 {
        return false;
    }
    has_16th_gap && coarse as f32 / total >= coarse_vote_ratio
}

/// Re-snap a COARSE bar's onsets to the 0.5 grid, merge collisions and
/// same-pitch fragments, and recompute durations from the merged onset gaps.
fn coarsen_bar(notes: Vec<QuantizedNote>, bpb: f32) -> Vec<QuantizedNote> {
    // Re-snap every onset onto the 0.5 grid. Offsets already sitting on a 0.5
    // multiple keep their exact position; 16th positions in (0.05, 0.45) are
    // "zeroed out" onto the containing beat start (e.g. 0.25 -> 0, 0.75 ->
    // 0.5, 1.25 -> 1.0), so over-split fragments collapse into the 8th grid.
    let mut snapped: Vec<QuantizedNote> = Vec::with_capacity(notes.len());
    for mut n in notes {
        let offset = n.beat_start - n.bar_index as f32 * bpb;
        let half = (offset / 0.5).floor() * 0.5;
        let new_offset = if (offset - (offset / 0.5).round() * 0.5).abs() < 1e-4 {
            offset
        } else {
            half.clamp(0.0, bpb - 0.5)
        };
        n.beat_start = n.bar_index as f32 * bpb + new_offset;
        snapped.push(n);
    }
    snapped.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    // Collision merge: two notes now share a snapped onset (monophonic melody)
    // → keep the higher confidence, tie-break by lower pitch.
    let mut merged: Vec<QuantizedNote> = Vec::with_capacity(snapped.len());
    for n in snapped {
        match merged.last_mut() {
            Some(last) if (last.beat_start - n.beat_start).abs() < 1e-4 => {
                let keep_last = if (last.confidence - n.confidence).abs() < 1e-6 {
                    last.pitch <= n.pitch
                } else {
                    last.confidence >= n.confidence
                };
                if !keep_last {
                    *last = n;
                }
            }
            _ => merged.push(n),
        }
    }

    // Same-pitch adjacent merge: note B starts where note A starts + snapped
    // gap and A's duration ≈ that gap → A extends over both, B dropped.
    let mut i = 0;
    while i + 1 < merged.len() {
        let gap = merged[i + 1].beat_start - merged[i].beat_start;
        if merged[i].pitch == merged[i + 1].pitch
            && gap > 0.0
            && (merged[i].beat_duration - gap).abs() <= 0.1
        {
            let a = merged[i].clone();
            let b = merged[i + 1].clone();
            merged[i] = QuantizedNote {
                beat_duration: b.beat_start + b.beat_duration - a.beat_start,
                ..a
            };
            merged.remove(i + 1);
        } else {
            i += 1;
        }
    }

    // Recompute durations from the final distinct snapped onsets. For the last
    // note in the bar the horizon is the next 0.5 slot (or the bar end when
    // closer), clamped to ≥ 1/6 beat (the smallest token).
    let offsets: Vec<f32> = merged
        .iter()
        .map(|n| n.beat_start - n.bar_index as f32 * bpb)
        .collect();
    for (i, n) in merged.iter_mut().enumerate() {
        let next = if i + 1 < offsets.len() {
            offsets[i + 1] - offsets[i]
        } else {
            (0.5_f32).min(bpb - offsets[i])
        };
        n.beat_duration = next.max(1.0 / 6.0);
    }
    merged
}

/// Post-quantization pass that merges 16th-note over-segmentation fragments
/// back onto the 8th-note grid, per-bar. Bars whose inter-onset gaps are NOT
/// dominated by 16th spacing (genuine 16th passages) are left untouched.
pub fn coarsen_rhythm(
    notes: Vec<QuantizedNote>,
    beats_per_bar: u32,
    cfg: &RhythmCoarsenConfig,
) -> Vec<QuantizedNote> {
    if !cfg.enabled || notes.is_empty() || beats_per_bar == 0 {
        return notes;
    }
    let bpb = beats_per_bar.max(2) as f32;
    let debug = std::env::var_os("KEYSCRIBE_COARSEN_DEBUG").is_some();
    if debug {
        eprintln!(
            "[coarsen-debug] {} notes, {} beats/bar",
            notes.len(),
            beats_per_bar
        );
    }

    let max_bar = notes.iter().map(|n| n.bar_index).max().unwrap_or(0);
    let mut out: Vec<QuantizedNote> = Vec::with_capacity(notes.len());
    for bar in 0..=max_bar {
        let group: Vec<QuantizedNote> = notes
            .iter()
            .filter(|n| n.bar_index == bar)
            .cloned()
            .collect();
        if group.is_empty() {
            continue;
        }
        let offsets = distinct_bar_offsets(&group, beats_per_bar);
        let min_notes = beats_per_bar as usize * 2 + 1;
        let coarse = bar_is_coarse(&offsets, cfg.coarse_vote_ratio, min_notes);
        if debug {
            eprintln!(
                "[coarsen-debug] bar {}: {} notes, {} distinct offsets, coarse={}",
                bar,
                group.len(),
                offsets.len(),
                coarse
            );
            if coarse {
                for n in &group {
                    eprintln!(
                        "[coarsen-debug]   pitch={} beat_start={:.3} dur={:.3} conf={:.3}",
                        n.pitch, n.beat_start, n.beat_duration, n.confidence
                    );
                }
            }
        }
        if coarse {
            out.extend(coarsen_bar(group, bpb));
        } else {
            out.extend(group);
        }
    }

    out.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });
    out
}

/// Duration grid including triplet values, coarse-to-fine. The engraver can
/// notate any of these (8th triplet = 1/3, 16th triplet = 1/6, etc.), so
/// snapping to them yields clean triplet beams instead of a 16th + tie.
fn duration_grid_for(style: SwingStyle) -> Vec<f32> {
    match style {
        SwingStyle::Swing => vec![4.0, 2.0, 1.0, 0.5, 2.0 / 3.0, 1.0 / 3.0],
        _ => vec![
            4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 2.0 / 3.0, 1.0 / 3.0, 0.25, 1.0 / 6.0, 1.0 / 8.0,
        ],
    }
}

// ── Learned Melody Quantizer ────────────────────────────────────────────────

/// Which rhythm-quantization engine to use for the melody.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerEngine {
    /// Rule-based grid snapping (`quantize_aligned_notes`).
    LegacyGrid,
    /// Learned MIDI-to-score tokenizer (needs `melody_quantizer.onnx`); falls
    /// back to `LegacyGrid` when the model file is absent or inference fails.
    LearnedOnnx,
}

impl Default for QuantizerEngine {
    fn default() -> Self {
        Self::LearnedOnnx
    }
}

impl QuantizerEngine {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "legacy" | "grid" => Some(Self::LegacyGrid),
            "learned" | "onnx" => Some(Self::LearnedOnnx),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LegacyGrid => "legacy",
            Self::LearnedOnnx => "learned",
        }
    }
}

/// Output of the learned quantizer: one rhythmic-value token per note, drawn
/// from the same vocabulary the engraver uses (`musicxml.rs:1210`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizerToken {
    /// Duration in beats (quarter = 1.0).
    pub beats: f32,
    pub note_type: &'static str,
    pub dots: u8,
    pub time_mod: Option<(u8, u8)>,
}

/// Vocabulary order. Must match the training export
/// (`tools/melody_corpus/export_quantizer_onnx.py`) and the `DurationToken`
/// candidate order in `musicxml.rs:1210`.
pub const LEARNED_TOKEN_TABLE: [QuantizerToken; 12] = [
    QuantizerToken { beats: 4.0, note_type: "whole", dots: 0, time_mod: None },
    QuantizerToken { beats: 3.0, note_type: "half", dots: 1, time_mod: None },
    QuantizerToken { beats: 2.0, note_type: "half", dots: 0, time_mod: None },
    QuantizerToken { beats: 1.5, note_type: "quarter", dots: 1, time_mod: None },
    QuantizerToken { beats: 1.0, note_type: "quarter", dots: 0, time_mod: None },
    QuantizerToken { beats: 0.75, note_type: "eighth", dots: 1, time_mod: None },
    QuantizerToken { beats: 0.5, note_type: "eighth", dots: 0, time_mod: None },
    QuantizerToken { beats: 0.375, note_type: "16th", dots: 1, time_mod: None },
    QuantizerToken { beats: 0.25, note_type: "16th", dots: 0, time_mod: None },
    QuantizerToken { beats: 2.0 / 3.0, note_type: "quarter", dots: 0, time_mod: Some((3, 2)) },
    QuantizerToken { beats: 1.0 / 3.0, note_type: "eighth", dots: 0, time_mod: Some((3, 2)) },
    QuantizerToken { beats: 1.0 / 6.0, note_type: "16th", dots: 0, time_mod: Some((3, 2)) },
];

/// Per-note feature vector fed to the learned quantizer. Must match the
/// featurizer used to train (`tools/melody_corpus/build_features.py`,
/// Phase 1.2 of `MELODY_TRANSCRIPTION_PLAN.md`): 9 floats = intra-beat pos,
/// raw duration beats, normalized tempo, beat-within-bar, swing one-hot[3],
/// normalized pitch, normalized velocity.
pub fn learned_note_features(
    note: &BeatAlignedNote,
    swing_style: SwingStyle,
    beats_per_bar: u32,
    tempo_bpm: f32,
) -> [f32; 9] {
    let swing = match swing_style {
        SwingStyle::Straight => [1.0, 0.0, 0.0],
        SwingStyle::Swing => [0.0, 1.0, 0.0],
        SwingStyle::Triplet => [0.0, 0.0, 1.0],
    };
    [
        note.intra_beat_pos.clamp(0.0, 0.9999),
        ((note.original_end_time - note.original_start_time)
            / note.beat_duration_sec.max(0.001))
            .max(0.0)
            .min(32.0),
        ((tempo_bpm.clamp(40.0, 260.0) - 40.0) / 220.0),
        (note.beat_index % beats_per_bar.max(1)) as f32 / beats_per_bar.max(1) as f32,
        swing[0],
        swing[1],
        swing[2],
        note.pitch as f32 / 127.0,
        note.velocity as f32 / 127.0,
    ]
}

/// Resolve `melody_quantizer.onnx` next to the executable, then the working
/// directory (mirrors the Demucs/Basic Pitch model resolution).
pub fn resolve_quantizer_model_path() -> Option<std::path::PathBuf> {
    let filename = "melody_quantizer.onnx";
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for p in [parent.join("models").join(filename), parent.join(filename)] {
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    for p in [
        std::path::PathBuf::from("models").join(filename),
        std::path::PathBuf::from(filename),
    ] {
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Learned melody quantization: per-note rhythmic-value tokens from the ONNX
/// model. Falls back to the rule grid when the model file is absent or
/// inference fails, so the learned engine is a strict improvement when it
/// works and never a regression.
pub fn quantize_aligned_notes_learned(
    aligned: &[BeatAlignedNote],
    swing_sections: &[SwingSection],
    beats_per_bar: u32,
    model_path: Option<&Path>,
) -> Vec<QuantizedNote> {
    let model_path = match model_path
        .map(|p| p.to_path_buf())
        .or_else(resolve_quantizer_model_path)
    {
        Some(p) if p.exists() => p,
        _ => {
            if std::env::var_os("KEYSCRIBE_QUANTIZER_DEBUG").is_some() {
                eprintln!("[quantizer] melody_quantizer.onnx not found — using legacy grid");
            }
            return quantize_aligned_notes(aligned, swing_sections, beats_per_bar);
        }
    };

    let mut infer = match crate::inference::MelodyQuantizerInference::new(&model_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!(
                "[quantizer] failed to load {}: {:#}; using legacy grid",
                model_path.display(),
                e
            );
            return quantize_aligned_notes(aligned, swing_sections, beats_per_bar);
        }
    };

    let mut tempo_values: Vec<f32> = aligned
        .iter()
        .filter_map(|n| {
            let d = n.next_beat_time - n.prev_beat_time;
            if d > 1e-3 {
                Some(60.0 / d)
            } else {
                None
            }
        })
        .collect();
    tempo_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tempo_bpm = tempo_values
        .get(tempo_values.len() / 2)
        .copied()
        .unwrap_or(120.0);

    let mut sorted: Vec<&BeatAlignedNote> = aligned.iter().collect();
    sorted.sort_by(|a, b| {
        a.original_start_time
            .partial_cmp(&b.original_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let section_for = |note: &BeatAlignedNote| {
        swing_sections
            .iter()
            .find(|s| s.contains_bar(note.bar_index))
            .or_else(|| swing_sections.first())
    };

    let features: Vec<Vec<f32>> = sorted
        .iter()
        .map(|n| {
            let style = section_for(n).map(|s| s.style).unwrap_or(SwingStyle::Straight);
            learned_note_features(n, style, beats_per_bar, tempo_bpm).to_vec()
        })
        .collect();

    let tokens = match infer.infer(&features) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[quantizer] inference failed: {:#}; using legacy grid", e);
            return quantize_aligned_notes(aligned, swing_sections, beats_per_bar);
        }
    };

    // Tier B1: vocabulary index >= LEARNED_TOKEN_TABLE.len() is the
    // MERGE_INTO_PREVIOUS class — the detection is a spurious fragment of
    // the previous detection. Merged detections emit no note; the previous
    // note's duration is extended to span them.
    //
    // Gap guard: training fragments are CONTIGUOUS (near-zero gap), but a
    // genuine staccato re-articulation has real silence between the notes
    // and must NOT be merged even when the model says so (corpus A/B:
    // unguarded merging cost recall 0.38 -> 0.34). Only honor the merge
    // when the detection starts within MERGE_MAX_GAP_BEATS of the previous
    // kept detection's end.
    const MERGE_MAX_GAP_BEATS: f32 = 0.08;
    let mut keep_mask = merge_keep_mask(&tokens);
    let mut last_kept_end: Option<(usize, f32)> = None; // (sorted idx, end beats)
    for i in 0..sorted.len() {
        let note = sorted[i];
        let raw_start_beats = note.original_start_time / note.beat_duration_sec.max(0.001);

        if keep_mask[i] {
            // Algorithmic Tier B1 Heuristic: Over-segmentation merge.
            if let Some((prev_idx, prev_end)) = last_kept_end {
                let prev_note = sorted[prev_idx];
                if note.pitch == prev_note.pitch 
                    && (raw_start_beats - prev_end).abs() <= MERGE_MAX_GAP_BEATS 
                    && !note.is_rearticulation {
                    keep_mask[i] = false; // Merge it!
                }
            }
        }

        if !keep_mask[i] {
            // A merge token (from model or heuristic): require contiguity with the previous KEPT note.
            if let Some((_, prev_end)) = last_kept_end {
                if raw_start_beats - prev_end > MERGE_MAX_GAP_BEATS {
                    keep_mask[i] = true; // real silence: keep as its own note
                }
            } else {
                keep_mask[i] = true; // Leading merge: keep as its own note
            }
        }

        if keep_mask[i] {
            let raw_end_beats = note.original_end_time / note.beat_duration_sec.max(0.001);
            last_kept_end = Some((i, raw_end_beats));
        }
    }

    // Pre-compute each detection's snapped onset so merged spans can extend
    // the previous kept note to the next kept onset.
    let beat_pos: Vec<f32> = sorted
        .iter()
        .map(|n| n.beat_index as f32 + n.intra_beat_pos)
        .collect();
    let sub_grids = compute_subdivision_grids(&beat_pos);

    let snapped_starts: Vec<f32> = sorted
        .iter()
        .enumerate()
        .map(|(i, note)| {
            let style = section_for(note).map(|s| s.style).unwrap_or(SwingStyle::Straight);
            let sub_grid = if style == SwingStyle::Swing {
                subdivision_grid(style)
            } else {
                sub_grids.get(i).cloned().unwrap_or_else(straight_grid)
            };
            let (snapped_pos, _) = snap_intra_beat_pos(
                note.intra_beat_pos,
                &sub_grid,
                note.beat_index % beats_per_bar,
                beats_per_bar,
            );
            note.beat_index as f32 + snapped_pos
        })
        .collect();

    let mut quantized: Vec<QuantizedNote> = Vec::with_capacity(sorted.len());
    for (i, (note, idx)) in sorted.iter().zip(tokens.iter()).enumerate() {
        if !keep_mask[i] {
            continue;
        }
        let style = section_for(note).map(|s| s.style).unwrap_or(SwingStyle::Straight);
        let token = LEARNED_TOKEN_TABLE
            .get(*idx as usize)
            .copied()
            .unwrap_or(LEARNED_TOKEN_TABLE[4]); // quarter fallback for out-of-vocab
        let sub_grid = if style == SwingStyle::Swing {
            subdivision_grid(style)
        } else {
            sub_grids.get(i).cloned().unwrap_or_else(straight_grid)
        };
        let (snapped_pos, snap_error) = snap_intra_beat_pos(
            note.intra_beat_pos,
            &sub_grid,
            note.beat_index % beats_per_bar,
            beats_per_bar,
        );
        let beat_start = note.beat_index as f32 + snapped_pos;
        let mut beat_duration = token.beats.max(1.0 / 6.0);
        // Span extension: when the detections FOLLOWING this note were
        // merged into it, cover up to the next KEPT detection's snapped
        // onset (same bar only) or the bar end — never past it.
        let mut k = i + 1;
        while k < sorted.len() && !keep_mask[k] {
            k += 1;
        }
        if k > i + 1 {
            let bar_end = ((beat_start / beats_per_bar.max(1) as f32).floor() + 1.0)
                * beats_per_bar.max(1) as f32;
            let mut span_target = bar_end;
            if k < sorted.len() && sorted[k].bar_index == note.bar_index {
                span_target = snapped_starts[k];
            }
            let span = (span_target - beat_start).max(0.0);
            if span > beat_duration {
                beat_duration = span.min(bar_end - beat_start).max(1.0 / 6.0);
            }
        }
        quantized.push(QuantizedNote {
            id: note.id,
            pitch: note.pitch,
            beat_start,
            beat_duration,
            velocity: note.velocity,
            channel: note.channel,
            confidence: (1.0 - snap_error * 2.0).clamp(0.3, 1.0),
            bar_index: note.bar_index,
            beat_index: note.beat_index % beats_per_bar,
            intra_beat_pos: note.intra_beat_pos,
            articulation: Articulation::Normal,
            swing_style: style,
            swing_feel: style == SwingStyle::Swing,
        });
    }
    if std::env::var_os("KEYSCRIBE_QUANT_DEBUG").is_some() {
        let merged = keep_mask.iter().filter(|&&k| !k).count();
        eprintln!(
            "[quant-debug] learned engine: {} of {} detections merged into previous ({} notes emitted)",
            merged,
            tokens.len(),
            quantized.len()
        );
    }

    quantized.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    // Independent token predictions can make a note overlap the next snapped
    // onset. Reject only the affected bars and use the sequence-aware legacy
    // result there; this preserves learned gains without letting one bad token
    // shift every later MusicXML cursor position.
    let mut invalid_bars = std::collections::BTreeSet::new();
    let by_id: std::collections::HashMap<u32, &QuantizedNote> = quantized
        .iter()
        .map(|note| (note.id, note))
        .collect();
    for pair in sorted.windows(2) {
        let current = pair[0];
        let next = pair[1];
        let Some(current_q) = by_id.get(&current.id) else { continue };
        let Some(next_q) = by_id.get(&next.id) else { continue };
        if current.bar_index != next.bar_index {
            continue;
        }
        let raw_gap_after_note =
            (next.original_start_time - current.original_end_time) / current.beat_duration_sec.max(0.001);
        if raw_gap_after_note <= 0.08 {
            let snapped_gap = (next_q.beat_start - current_q.beat_start).max(0.0);
            if snapped_gap > 0.0 && (current_q.beat_duration - snapped_gap).abs() > 0.08 {
                invalid_bars.insert(current.bar_index);
            }
        }
    }
    if std::env::var_os("KEYSCRIBE_QUANT_DEBUG").is_some() {
        let total_bars = quantized.iter().map(|n| n.bar_index).collect::<std::collections::BTreeSet<_>>().len();
        eprintln!(
            "[quant-debug] learned engine: {}/{} bars fell back to the legacy grid",
            invalid_bars.len(),
            total_bars
        );
    }
    if !invalid_bars.is_empty() {
        let legacy = quantize_aligned_notes(aligned, swing_sections, beats_per_bar);
        let legacy_by_id: std::collections::HashMap<u32, QuantizedNote> = legacy
            .into_iter()
            .map(|note| (note.id, note))
            .collect();
        for note in &mut quantized {
            if invalid_bars.contains(&note.bar_index) {
                if let Some(replacement) = legacy_by_id.get(&note.id) {
                    *note = replacement.clone();
                }
            }
        }
    }

    quantized.sort_by(|a, b| {
        a.beat_start
            .partial_cmp(&b.beat_start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    // Fill melody durations across audible gaps
    fill_melody_durations(&mut quantized, aligned);

    // Monophonic collision deduplication
    let mut mono: Vec<QuantizedNote> = Vec::with_capacity(quantized.len());
    for note in quantized {
        match mono.last_mut() {
            Some(last) if (last.beat_start - note.beat_start).abs() < 1e-4 => {
                let replace = if last.pitch == note.pitch {
                    note.beat_duration > last.beat_duration
                } else {
                    note.confidence > last.confidence
                };
                if replace {
                    *last = note;
                }
            }
            _ => mono.push(note),
        }
    }
    quantized = mono;

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

        let matched_current = aligned.iter().find(|a| a.id == current.id);
        let (start_time, end_time) = if let Some(a) = matched_current {
            (a.original_start_time, a.original_end_time)
        } else {
            (current.beat_start, current.beat_start + current.beat_duration)
        };

        let matched_next = aligned.iter().find(|a| a.id == next.id);
        let next_start = if let Some(a) = matched_next {
            a.original_start_time
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
            id: 0,
            pitch: 60,
            start_time: start,
            end_time: end,
            velocity: 100,
            channel: None, is_rearticulation: false,
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
        let notes = vec![NoteEvent {
            id: 1,
            pitch: 60,
            start_time: 0.49,
            end_time: 1.01,
            velocity: 100,
            channel: None, is_rearticulation: false,
        }];
        let quantized = quantize_notes(&notes, 0.5, &QuantizationConfig::default());
        assert_eq!(quantized.len(), 1);
        assert!((quantized[0].beat_start - 1.0).abs() < 0.0001);
        assert!((quantized[0].beat_duration - 1.0).abs() < 0.0001);
    }

    #[test]
    fn quantizes_with_piecewise_tempo_map() {
        let notes = vec![NoteEvent {
            id: 1,
            pitch: 60,
            start_time: 7.9,
            end_time: 8.4,
            velocity: 100,
            channel: None, is_rearticulation: false,
        }];
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

        let quantized =
            quantize_notes_with_tempo_map(&notes, map.as_slice(), &QuantizationConfig::default());
        assert_eq!(quantized.len(), 1);
        assert!(quantized[0].beat_start >= 15.5);
    }

    #[test]
    fn snaps_to_quarter_or_eighth_in_compound_meter() {
        let notes = vec![NoteEvent {
            id: 1,
            pitch: 60,
            start_time: 0.0,
            end_time: 0.165,
            velocity: 100,
            channel: None, is_rearticulation: false,
        }];
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
        // DEBUG: now snaps to 0.333 (1/3 beat)
        assert!((quantized[0].beat_duration - 1.0/3.0).abs() < 0.06);
    }

    #[test]
    fn straight_grid_admits_sixteenth_triplet_positions() {
        let grid = subdivision_grid(SwingStyle::Straight);
        assert!(grid.iter().any(|p| (*p - 1.0 / 6.0).abs() < 1e-6));
        assert!(grid.iter().any(|p| (*p - 5.0 / 6.0).abs() < 1e-6));
        let (snapped, _) = snap_intra_beat_pos(0.17, &grid, 0, 4);
        assert!((snapped - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn duration_uses_snapped_gap_for_contiguous_triplets() {
        let notes = vec![
            aligned_note(60, 0.0, 1.0 / 6.0, 0, 0, 0.0),
            aligned_note(61, 1.0 / 6.0, 1.0 / 3.0, 0, 0, 1.0 / 3.0),
            aligned_note(62, 1.0 / 3.0, 0.5, 0, 0, 2.0 / 3.0),
        ];
        let quantized = quantize_aligned_notes(&notes, &[], 4);
        assert_eq!(quantized.len(), 3);
        assert!((quantized[0].beat_duration - 1.0 / 3.0).abs() < 1e-5);
        assert!((quantized[1].beat_duration - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn duration_does_not_fill_a_real_gap_to_next_onset() {
        let notes = vec![
            aligned_note(60, 0.0, 0.125, 0, 0, 0.0),
            aligned_note(61, 0.5, 0.75, 1, 0, 0.0),
        ];
        let quantized = quantize_aligned_notes(&notes, &[], 4);
        assert_eq!(quantized.len(), 2);
        assert!((quantized[0].beat_duration - 0.25).abs() < 1e-5);
    }

    #[test]
    fn learned_duration_feature_is_in_beats() {
        let n = aligned_note(60, 0.0, 0.25, 0, 0, 0.0);
        let features = learned_note_features(&n, SwingStyle::Straight, 4, 120.0);
        assert!((features[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn splits_note_across_barline_into_tied_segments() {
        let notes = vec![NoteEvent {
            id: 1,
            pitch: 60,
            start_time: 1.4,
            end_time: 2.6,
            velocity: 100,
            channel: None, is_rearticulation: false,
        }];
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

        assert!(
            tied.len() >= 2,
            "Note crossing barline at beat 4 should split into multiple segments"
        );
        let first_segment = &tied[0];
        assert!(
            first_segment.beat_duration <= 4.0,
            "First segment should end at or before barline"
        );
        let last_segment = tied.last().unwrap();
        assert!(
            last_segment.tie_stop,
            "Last segment should have tie_stop=true"
        );
    }

    #[test]
    fn no_ties_when_note_within_single_bar() {
        let notes = vec![NoteEvent {
            id: 1,
            pitch: 60,
            start_time: 0.1,
            end_time: 0.4,
            velocity: 100,
            channel: None, is_rearticulation: false,
        }];
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
            id: 1,
            pitch,
            velocity: 100,
            channel: None, is_rearticulation: false,
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
        let (pos, err) = snap_intra_beat_pos(0.03, &[0.0, 0.5], 0, 4);
        assert!((pos - 0.0).abs() < 0.001);
        assert!(err < 0.05);
    }

    #[test]
    fn snap_intra_pos_to_swing() {
        let grid = vec![0.0, 2.0 / 3.0];
        let (pos, err) = snap_intra_beat_pos(0.65, &grid, 0, 4);
        assert!((pos - 2.0 / 3.0).abs() < 0.001);
        assert!(err < 0.05);
    }

    #[test]
    fn metrical_bonus_pulls_ambiguity_to_onbeat_on_strong_beat() {
        let grid = straight_grid();
        // p=0.14 on beat 1 (4/4): cost to 0.0 = 0.14 - 0.02 = 0.12, cost to
        // 0.25 = 0.11 + 0.045 = 0.155 → onbeat wins.
        let (pos, _) = snap_intra_beat_pos(0.14, &grid, 0, 4);
        assert!((pos - 0.0).abs() < 1e-4);
    }

    #[test]
    fn metrical_bonus_is_a_tiebreaker_not_a_decider() {
        let grid = straight_grid();
        // p=0.15: beat 1 (bonus 0.02) → 0.0 wins (0.13 vs 0.145); beat 2
        // (no bonus) → 0.25 wins (0.15 vs 0.145). Same onset, different snap
        // purely from metrical position.
        let (pos1, _) = snap_intra_beat_pos(0.15, &grid, 0, 4);
        assert!((pos1 - 0.0).abs() < 1e-4);
        let (pos2, _) = snap_intra_beat_pos(0.15, &grid, 1, 4);
        assert!((pos2 - 0.25).abs() < 1e-4);
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

    // ── Tier A1: rhythm coarsening ──

    fn q_note(pitch: u8, bar: u32, offset: f32, duration: f32, confidence: f32) -> QuantizedNote {
        QuantizedNote {
            id: pitch as u32 * 100 + (offset * 100.0) as u32,
            pitch,
            beat_start: bar as f32 * 4.0 + offset,
            beat_duration: duration,
            confidence,
            bar_index: bar,
            beat_index: (offset % 4.0).floor() as u32,
            intra_beat_pos: offset % 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn coarsen_rhythm_collapses_stray_sixteenths_onto_eighths() {
        // Over-split bar: a full 8th grid (8 notes) plus a 16th fragment at the
        // "and of 2" (1.25). The vote passes because the bar has > a full 8th
        // grid (9 notes) and is dominated by 8th-or-larger gaps.
        let offsets = [0.0f32, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5];
        let notes: Vec<QuantizedNote> = offsets
            .iter()
            .enumerate()
            .map(|(i, &o)| q_note(60 + (i % 2) as u8, 0, o, 0.25, 0.5))
            .collect();
        let out = coarsen_rhythm(notes, 4, &RhythmCoarsenConfig::default());
        // 1.25 -> 1.0 collides with the existing note at 1.0 (kept): 9 input
        // notes collapse to 8 on the 0.5 grid, each with a 0.5 duration.
        assert_eq!(out.len(), 8);
        for n in &out {
            assert!((n.beat_duration - 0.5).abs() < 1e-4, "dur {} wrong", n.beat_start);
        }
    }

    #[test]
    fn coarsen_rhythm_leaves_genuine_sixteenths_untouched() {
        // Mixed 0.25 / 0.75 gaps: fine vote fails → bar is not coarse.
        let offsets = [0.0f32, 0.25, 1.0, 1.25, 2.0, 2.25];
        let notes: Vec<QuantizedNote> = offsets
            .iter()
            .enumerate()
            .map(|(i, &o)| q_note(60 + i as u8, 0, o, 0.25, 0.8))
            .collect();
        let out = coarsen_rhythm(notes.clone(), 4, &RhythmCoarsenConfig::default());
        assert_eq!(out.len(), notes.len());
        for (a, b) in out.iter().zip(notes.iter()) {
            assert!((a.beat_start - b.beat_start).abs() < 1e-4);
            assert!((a.beat_duration - b.beat_duration).abs() < 1e-4);
        }
    }

    #[test]
    fn coarsen_rhythm_collision_keeps_higher_confidence() {
        // Over-split bar (60 on the beat + a 16th fragment at 0.25) sitting on
        // a coarse-dominated 8th grid. The two notes collide on the 0.5 snap;
        // the higher-confidence one wins.
        let mut notes = vec![
            q_note(60, 0, 0.0, 0.25, 0.9),
            q_note(62, 0, 0.25, 0.25, 0.4),
        ];
        for (i, off) in [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5].iter().enumerate() {
            notes.push(q_note(64 + i as u8, 0, *off, 0.25, 0.8));
        }
        let out = coarsen_rhythm(notes, 4, &RhythmCoarsenConfig::default());
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].pitch, 60);
        assert!((out[0].beat_duration - 0.5).abs() < 1e-4);
    }

    #[test]
    fn coarsen_rhythm_leaves_single_note_and_empty_unchanged() {
        let single = vec![q_note(60, 0, 0.5, 0.5, 0.9)];
        let out = coarsen_rhythm(single.clone(), 4, &RhythmCoarsenConfig::default());
        assert_eq!(out.len(), 1);
        assert!((out[0].beat_start - single[0].beat_start).abs() < 1e-4);

        let out = coarsen_rhythm(Vec::new(), 4, &RhythmCoarsenConfig::default());
        assert!(out.is_empty());
    }

    #[test]
    fn coarsen_rhythm_leaves_pure_eighth_bar_unchanged() {
        let notes: Vec<QuantizedNote> = (0..8)
            .map(|i| q_note(60 + i as u8, 0, i as f32 * 0.5, 0.5, 0.8))
            .collect();
        let out = coarsen_rhythm(notes.clone(), 4, &RhythmCoarsenConfig::default());
        assert_eq!(out.len(), 8);
        for (a, b) in out.iter().zip(notes.iter()) {
            assert!((a.beat_start - b.beat_start).abs() < 1e-4);
            assert!((a.beat_duration - b.beat_duration).abs() < 1e-4);
        }
    }

    #[test]
    fn coarsen_rhythm_disabled_passes_through() {
        let notes: Vec<QuantizedNote> = (0..8)
            .map(|i| q_note(60, 0, i as f32 * 0.25, 0.25, 0.5))
            .collect();
        let cfg = RhythmCoarsenConfig {
            enabled: false,
            coarse_vote_ratio: 0.70,
        };
        let out = coarsen_rhythm(notes.clone(), 4, &cfg);
        assert_eq!(out.len(), 8);
        for (a, b) in out.iter().zip(notes.iter()) {
            assert!((a.beat_start - b.beat_start).abs() < 1e-4);
        }
    }

    #[test]
    fn merge_keep_mask_drops_merge_tokens() {
        // quarter, merge, eighth, merge, merge -> only two notes survive.
        let mask = merge_keep_mask(&[4, 12, 6, 12, 12]);
        assert_eq!(mask, vec![true, false, true, false, false]);
    }

    #[test]
    fn merge_keep_mask_leading_merge_is_dropped() {
        let mask = merge_keep_mask(&[12, 4, 12, 4]);
        assert_eq!(mask, vec![false, true, false, true]);
    }

    #[test]
    fn merge_keep_mask_v1_vocab_passthrough() {
        // A v1 12-class model never emits index >= 12: keep everything.
        let tokens: Vec<u32> = (0..12u32).collect();
        assert!(merge_keep_mask(&tokens).iter().all(|&k| k));
        assert_eq!(merge_keep_mask(&tokens).len(), 12);
    }

    #[test]
    fn merge_keep_mask_empty() {
        assert!(merge_keep_mask(&[]).is_empty());
    }
}
