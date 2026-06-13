use std::collections::BTreeSet;

#[allow(unused_imports)]
use crate::leadsheet::types::{Articulation, ChordSymbolChange, QuantizedNote, SwingStyle};

#[derive(Debug, Clone, Copy)]
pub struct ChordAnalysisConfig {
    pub step_beats: f32,
    pub min_active_notes: usize,
    pub skip: bool,
    pub max_chords_per_bar: usize,
    pub chord_min_simultaneous: usize,
}

impl Default for ChordAnalysisConfig {
    fn default() -> Self {
        Self {
            step_beats: 1.0,
            min_active_notes: 2,
            skip: false,
            max_chords_per_bar: 2,
            chord_min_simultaneous: 3,
        }
    }
}

pub fn detect_chord_changes(
    quantized_notes: &[QuantizedNote],
    config: ChordAnalysisConfig,
) -> Vec<ChordSymbolChange> {
    if config.skip || quantized_notes.is_empty() {
        return Vec::new();
    }

    let max_beat = quantized_notes
        .iter()
        .map(|n| n.beat_start + n.beat_duration)
        .fold(0.0f32, f32::max);

    let mut out = Vec::new();
    let mut last_symbol = String::new();

    let mut sorted_notes = quantized_notes.to_vec();
    sorted_notes.sort_by(|a, b| a.beat_start.partial_cmp(&b.beat_start).unwrap_or(std::cmp::Ordering::Equal));

    let mut beat = 0.0f32;
    let step = config.step_beats.max(0.25);
    let mut note_idx = 0;
    let mut active_notes: Vec<&QuantizedNote> = Vec::new();

    while beat <= max_beat + 1.0e-4 {
        while note_idx < sorted_notes.len() && sorted_notes[note_idx].beat_start <= beat + 1.0e-5 {
            active_notes.push(&sorted_notes[note_idx]);
            note_idx += 1;
        }

        active_notes.retain(|n| n.beat_start + n.beat_duration > beat);

        let mut pcs = BTreeSet::<u8>::new();
        let mut bass_pitch: Option<u8> = None;

        for note in &active_notes {
            pcs.insert(note.pitch % 12);
            bass_pitch = Some(match bass_pitch {
                Some(existing) => existing.min(note.pitch),
                None => note.pitch,
            });
        }

        if pcs.len() < config.min_active_notes {
            beat += step;
            continue;
        }

        if let Some(symbol) = choose_chord_symbol(pcs.iter().copied().collect::<Vec<_>>().as_slice(), bass_pitch) {
            if symbol != last_symbol {
                out.push(ChordSymbolChange {
                    beat_start: beat,
                    symbol: symbol.clone(),
                });
                last_symbol = symbol;
            }
        }

        beat += step;
    }

    out
}

/// Detect chord changes using per-bar analysis.
/// Scans each bar at a fine resolution to find the moment with the most simultaneous
/// notes, then aligns the resulting chord to the downbeat. Emits at most
/// `max_chords_per_bar` chords per bar. Favors positions with 3+ simultaneous notes
/// and longer note durations. Only places off-beat chords when the harmony is very
/// clearly rooted at an off-beat position.
pub fn detect_chord_changes_per_bar(
    quantized_notes: &[QuantizedNote],
    beats_per_bar: u32,
    config: ChordAnalysisConfig,
) -> Vec<ChordSymbolChange> {
    if config.skip || quantized_notes.is_empty() {
        return Vec::new();
    }

    let bpb = beats_per_bar.max(2) as f32;
    let max_chords = config.max_chords_per_bar.max(1);
    let min_simultaneous = config.chord_min_simultaneous.max(1);
    let scan_step = 0.25;

    let max_beat = quantized_notes
        .iter()
        .map(|n| n.beat_start + n.beat_duration)
        .fold(0.0f32, f32::max);
    let num_bars = (max_beat / bpb).ceil() as u32;

    let mut out = Vec::new();
    let mut last_symbol = String::new();

    for bar_idx in 0..num_bars {
        let bar_start = bar_idx as f32 * bpb;
        let bar_end = bar_start + bpb;
        let half_bar = bar_start + bpb * 0.5;

        let bar_notes: Vec<&QuantizedNote> = quantized_notes
            .iter()
            .filter(|n| n.beat_start < bar_end && n.beat_start + n.beat_duration > bar_start)
            .collect();

        if bar_notes.is_empty() {
            continue;
        }

        struct Candidate {
            pos: f32,
            pcs: Vec<u8>,
            bass_pitch: Option<u8>,
            num_pcs: usize,
            score: f32,
        }

        fn collect_at(notes: &[&QuantizedNote], pos: f32) -> (BTreeSet<u8>, Option<u8>, f32, usize) {
            let mut pcs_set = BTreeSet::<u8>::new();
            let mut bass_pitch: Option<u8> = None;
            let mut total_duration = 0.0f32;
            let mut active_count = 0;
            for note in notes {
                let note_end = note.beat_start + note.beat_duration;
                if pos + 1e-5 < note.beat_start || pos >= note_end {
                    continue;
                }
                pcs_set.insert(note.pitch % 12);
                bass_pitch = Some(match bass_pitch {
                    Some(existing) => existing.min(note.pitch),
                    None => note.pitch,
                });
                total_duration += note.beat_duration;
                active_count += 1;
            }
            (pcs_set, bass_pitch, total_duration, active_count)
        }

        // Scan first half of bar at fine resolution
        let mut best_first: Option<Candidate> = None;
        let mut pos = bar_start;
        while pos < half_bar - 1e-5 {
            let (pcs_set, bass_pitch, total_dur, active) = collect_at(&bar_notes, pos);
            let num_pcs = pcs_set.len();
            if num_pcs >= 1 {
                let score = num_pcs as f32 * 5.0 + total_dur * 2.0 + active as f32;
                let is_better = match &best_first {
                    Some(b) => num_pcs > b.num_pcs || (num_pcs == b.num_pcs && score > b.score),
                    None => true,
                };
                if is_better {
                    best_first = Some(Candidate {
                        pos,
                        pcs: pcs_set.iter().copied().collect(),
                        bass_pitch,
                        num_pcs,
                        score,
                    });
                }
            }
            pos += scan_step;
        }

        // Scan second half of bar at fine resolution
        let mut best_second: Option<Candidate> = None;
        pos = half_bar;
        while pos < bar_end - 1e-5 {
            let (pcs_set, bass_pitch, total_dur, active) = collect_at(&bar_notes, pos);
            let num_pcs = pcs_set.len();
            if num_pcs >= 1 {
                let score = num_pcs as f32 * 5.0 + total_dur * 2.0 + active as f32;
                let is_better = match &best_second {
                    Some(b) => num_pcs > b.num_pcs || (num_pcs == b.num_pcs && score > b.score),
                    None => true,
                };
                if is_better {
                    best_second = Some(Candidate {
                        pos,
                        pcs: pcs_set.iter().copied().collect(),
                        bass_pitch,
                        num_pcs,
                        score,
                    });
                }
            }
            pos += scan_step;
        }

        // Helper: emit a chord if symbol changed, return true if emitted
        let mut chords_added = 0;

        // Decide where to place the chord: snap to downbeat unless clearly off-beat
        let mut try_emit = |cand: &Candidate, default_placement: f32, min_pcs: usize| -> bool {
            if cand.num_pcs < min_pcs {
                return false;
            }
            if let Some(symbol) = choose_chord_symbol(&cand.pcs, cand.bass_pitch) {
                if symbol == last_symbol {
                    return false;
                }
                // Snap to nearest beat unless the chord is clearly off-beat
                // (off-beat defined as > 0.3 beats from any beat boundary)
                let min_dist = (cand.pos - bar_start).abs().min((cand.pos - half_bar).abs());
                let placement = if min_dist > 0.3 && cand.num_pcs >= min_simultaneous {
                    // Very clear off-beat: place at actual position
                    cand.pos
                } else {
                    default_placement
                };
                out.push(ChordSymbolChange {
                    beat_start: placement,
                    symbol: symbol.clone(),
                });
                last_symbol = symbol;
                true
            } else {
                false
            }
        };

        // Try primary chord from first half, placed at downbeat
        if let Some(ref best) = best_first {
            if try_emit(best, bar_start, min_simultaneous) {
                chords_added += 1;
            }
        }

        // Try secondary chord from second half, placed at half-bar
        if chords_added < max_chords {
            if let Some(ref best) = best_second {
                if try_emit(best, half_bar, min_simultaneous) {
                    chords_added += 1;
                }
            }
        }

        // Fallback: no chord scored well enough — try downbeat with min_active_notes threshold
        if chords_added == 0 {
            let (pcs_set, bass_pitch, _, _) = collect_at(&bar_notes, bar_start);
            if pcs_set.len() >= config.min_active_notes {
                let pcs: Vec<u8> = pcs_set.iter().copied().collect();
                if let Some(symbol) = choose_chord_symbol(&pcs, bass_pitch) {
                    if symbol != last_symbol {
                        out.push(ChordSymbolChange {
                            beat_start: bar_start,
                            symbol: symbol.clone(),
                        });
                        last_symbol = symbol;
                    }
                }
            }
        }
    }

    out
}

pub fn detect_chord_from_active_notes(active_midi: &[u8]) -> Option<String> {
    if active_midi.len() < 2 {
        return None;
    }
    let mut pcs: Vec<u8> = active_midi.iter().map(|m| m % 12).collect();
    pcs.sort();
    pcs.dedup();
    let bass_pitch = active_midi.iter().min().copied();
    choose_chord_symbol(&pcs, bass_pitch)
}

fn score_template(pcs: &[u8], root: u8, intervals: &[u8]) -> Option<(i32, i32)> {
    let mut score = 0i32;
    let mut covered = 0i32;
    for &pc in pcs {
        let rel = (pc + 12 - root) % 12;
        if intervals.contains(&rel) {
            score += 3;
            covered += 1;
        } else {
            score -= 1;
        }
    }
    for &interval in intervals {
        let target = (root + interval) % 12;
        if pcs.contains(&target) {
            score += 1;
        }
    }
    if covered < 2 {
        None
    } else {
        Some((score, covered))
    }
}

fn best_template_for_root<'a>(
    pcs: &[u8],
    root: u8,
    templates: &'a [(&str, &[u8])],
) -> Option<(&'a str, i32, i32)> {
    let mut best = None;
    let mut best_score = i32::MIN;
    for (suffix, intervals) in templates {
        if let Some((score, covered)) = score_template(pcs, root, intervals) {
            // Prefer simpler chords (fewer extensions) when scores are close.
            // Simplicity bonus: triads get +1, 7th chords 0, extended chords negative.
            let simplicity = match intervals.len() {
                3 => 1,
                4 => 0,
                _ => -(intervals.len() as i32 - 4),
            };
            let adjusted = score + simplicity;
            if adjusted > best_score {
                best_score = adjusted;
                best = Some((*suffix, adjusted, covered));
            }
        }
    }
    best
}

pub(crate) fn choose_chord_symbol(pcs: &[u8], bass_pitch: Option<u8>) -> Option<String> {
    if pcs.is_empty() {
        return None;
    }

    // Octave: all notes are the same pitch class
    if pcs.len() == 1 {
        let root_name = pitch_class_name_flat(pcs[0]);
        return if let Some(bass) = bass_pitch.map(|b| b % 12) {
            if bass != pcs[0] {
                Some(format!("{} (8ve)/{}", root_name, pitch_class_name_bass(bass, pcs[0])))
            } else {
                Some(format!("{} (8ve)", root_name))
            }
        } else {
            Some(format!("{} (8ve)", root_name))
        };
    }

    // Power chord: root + perfect 5th
    if pcs.len() == 2 {
        let interval = (pcs[1] + 12 - pcs[0]) % 12;
        if interval == 7 {
            let root_name = pitch_class_name_flat(pcs[0]);
            return if let Some(bass) = bass_pitch.map(|b| b % 12) {
                if bass != pcs[0] {
                    Some(format!("{}{}/{}", root_name, 5, pitch_class_name_bass(bass, pcs[0])))
                } else {
                    Some(format!("{}{}", root_name, 5))
                }
            } else {
                Some(format!("{}{}", root_name, 5))
            };
        }
    }

    const TEMPLATES: [(&str, &[u8]); 25] = [
        ("", &[0, 4, 7]),
        ("-", &[0, 3, 7]),
        ("dim", &[0, 3, 6]),
        ("aug", &[0, 4, 8]),
        ("sus2", &[0, 2, 7]),
        ("sus4", &[0, 5, 7]),
        ("7", &[0, 4, 7, 10]),
        ("\u{0394}7", &[0, 4, 7, 11]),
        ("-7", &[0, 3, 7, 10]),
        ("-\u{0394}7", &[0, 3, 7, 11]),
        ("dim7", &[0, 3, 6, 9]),
        ("-7b5", &[0, 3, 6, 10]),
        ("7#5", &[0, 4, 8, 10]),
        ("9", &[0, 4, 7, 10, 2]),
        ("\u{0394}9", &[0, 4, 7, 11, 2]),
        ("-9", &[0, 3, 7, 10, 2]),
        ("7b9", &[0, 4, 7, 10, 1]),
        ("7#9", &[0, 4, 7, 10, 3]),
        ("7#11", &[0, 4, 7, 10, 6]),
        ("\u{0394}7#11", &[0, 4, 7, 11, 6]),
        ("-11", &[0, 3, 7, 10, 2, 5]),
        ("13", &[0, 4, 7, 10, 2, 9]),
        ("\u{0394}13", &[0, 4, 7, 11, 2, 9]),
        ("-13", &[0, 3, 7, 10, 2, 9]),
        ("\u{0394}9#11", &[0, 4, 7, 11, 2, 6]),
    ];

    let mut best_root = 0u8;
    let mut best_suffix = "";
    let mut best_score = i32::MIN;
    for root in 0u8..12u8 {
        if let Some((suffix, score, _)) = best_template_for_root(pcs, root, &TEMPLATES) {
            let bass_bonus = match bass_pitch {
                Some(bass) if bass % 12 == root => 2,
                _ => 0,
            };
            if score + bass_bonus > best_score {
                best_score = score + bass_bonus;
                best_root = root;
                best_suffix = suffix;
            }
        }
    }
    if best_score == i32::MIN {
        return None;
    }

    let bass_pc = bass_pitch.map(|b| b % 12);

    let mut chord = format!("{}{}", pitch_class_name_flat(best_root), best_suffix);

    if let Some(bass) = bass_pc {
        if bass != best_root {
            chord.push('/');
            chord.push_str(pitch_class_name_bass(bass, best_root));
        }
    }

    Some(chord)
}

fn pitch_class_name_flat(pc: u8) -> &'static str {
    match pc % 12 {
        0 => "C",
        1 => "Db",
        2 => "D",
        3 => "Eb",
        4 => "E",
        5 => "F",
        6 => "Gb",
        7 => "G",
        8 => "Ab",
        9 => "A",
        10 => "Bb",
        _ => "B",
    }
}

fn pitch_class_name_bass(bass_pc: u8, root_pc: u8) -> &'static str {
    let interval = (bass_pc + 12 - root_pc) % 12;
    let degree: u8 = match interval {
        0 => 0,
        1 | 2 => 1,
        3 | 4 => 2,
        5 => 3,
        6 => 4,
        7 => 4,
        8 | 9 => 5,
        10 | 11 => 6,
        _ => 0,
    };
    let root_letter: u8 = match root_pc {
        0 => 0,
        1 | 2 => 1,
        3 | 4 => 2,
        5 => 3,
        6 | 7 => 4,
        8 | 9 => 5,
        10 | 11 => 6,
        _ => 0,
    };
    let bass_letter = (root_letter + degree) % 7;
    let natural_pc: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
    let natural = natural_pc[bass_letter as usize];

    let name = if bass_pc == natural {
        ["C", "D", "E", "F", "G", "A", "B"][bass_letter as usize]
    } else if (bass_pc + 12 - natural) % 12 == 1 {
        ["C#", "D#", "E#", "F#", "G#", "A#", "B#"][bass_letter as usize]
    } else {
        ["Cb", "Db", "Eb", "Fb", "Gb", "Ab", "Bb"][bass_letter as usize]
    };
    match name {
        "Cb" => "B",
        "Fb" => "E",
        "B#" => "C",
        "E#" => "F",
        _ => name,
    }
}

fn midi_pitch_name(pitch: u8) -> String {
    const NOTES: &[&str] = &["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let octave = (pitch / 12).saturating_sub(1);
    let note = NOTES[(pitch % 12) as usize];
    format!("{}{}", note, octave)
}

/// Writes `chord_debug.json` with per-bar chord details: candidate positions,
/// pitch classes, and all notes that contributed to each chord.
pub fn debug_chord_notes_to_json(
    quantized_notes: &[QuantizedNote],
    beats_per_bar: u32,
    config: ChordAnalysisConfig,
) {
    if quantized_notes.is_empty() {
        return;
    }

    let bpb = beats_per_bar.max(2) as f32;
    let min_simultaneous = config.chord_min_simultaneous.max(1);
    let scan_step = 0.25;

    let max_beat = quantized_notes
        .iter()
        .map(|n| n.beat_start + n.beat_duration)
        .fold(0.0f32, f32::max);
    let num_bars = (max_beat / bpb).ceil() as u32;

    let mut bars_json = Vec::new();

    for bar_idx in 0..num_bars {
        let bar_start = bar_idx as f32 * bpb;
        let bar_end = bar_start + bpb;
        let half_bar = bar_start + bpb * 0.5;

        let bar_notes: Vec<&QuantizedNote> = quantized_notes
            .iter()
            .filter(|n| n.beat_start < bar_end && n.beat_start + n.beat_duration > bar_start)
            .collect();

        if bar_notes.is_empty() {
            continue;
        }

        let notes_at = |pos: f32| -> Vec<&QuantizedNote> {
            bar_notes
                .iter()
                .filter(|n| {
                    let ne = n.beat_start + n.beat_duration;
                    pos + 1e-5 >= n.beat_start && pos < ne
                })
                .copied()
                .collect()
        };

        fn scan_half(notes: &[&QuantizedNote], start: f32, end: f32, step: f32) -> (f32, Vec<u8>, f32) {
            let mut best_pos = start;
            let mut best_pcs = Vec::new();
            let mut best_score = -1.0f32;
            let mut p = start;
            while p < end - 1e-5 {
                let active: Vec<&QuantizedNote> = notes
                    .iter()
                    .filter(|n| {
                        let ne = n.beat_start + n.beat_duration;
                        p + 1e-5 >= n.beat_start && p < ne
                    })
                    .copied()
                    .collect();
                let mut pcs: Vec<u8> = active.iter().map(|n| n.pitch % 12).collect();
                pcs.sort();
                pcs.dedup();
                let count = pcs.len();
                if count >= 1 {
                    let td: f32 = active.iter().map(|n| n.beat_duration).sum();
                    let score = count as f32 * 5.0 + td * 2.0 + active.len() as f32;
                    if score > best_score {
                        best_score = score;
                        best_pos = p;
                        best_pcs = pcs;
                    }
                }
                p += step;
            }
            (best_pos, best_pcs, best_score)
        }

        let (first_pos, first_pcs, _) = scan_half(&bar_notes, bar_start, half_bar, scan_step);
        let (second_pos, second_pcs, _) = scan_half(&bar_notes, half_bar, bar_end, scan_step);

        let mut chords_json = Vec::new();

        fn chord_entry(pos: f32, pcs: &[u8], label: &str, placement: f32, notes: &[&QuantizedNote]) -> serde_json::Value {
            let bp = notes.iter().map(|n| n.pitch).min();
            let symbol = choose_chord_symbol(pcs, bp);
            serde_json::json!({
                "type": label,
                "symbol": symbol,
                "placed_at": placement,
                "found_at": pos,
                "num_pitch_classes": pcs.len(),
                "pitch_classes": pcs,
                "notes": notes.iter().map(|n| serde_json::json!({
                    "id": n.id,
                    "pitch": n.pitch,
                    "pitch_class": n.pitch % 12,
                    "midi_name": midi_pitch_name(n.pitch),
                    "beat_start": n.beat_start,
                    "beat_duration": n.beat_duration,
                    "velocity": n.velocity,
                    "bar_index": n.bar_index,
                    "beat_index": n.beat_index,
                })).collect::<Vec<_>>(),
            })
        }

        // Primary chord from first half
        if first_pcs.len() >= min_simultaneous {
            let md = (first_pos - bar_start).abs().min((first_pos - half_bar).abs());
            let place = if md > 0.3 && first_pcs.len() >= min_simultaneous { first_pos } else { bar_start };
            let ns = notes_at(first_pos);
            chords_json.push(chord_entry(first_pos, &first_pcs, "primary", place, &ns));
        }

        // Secondary chord from second half
        if chords_json.len() < config.max_chords_per_bar && second_pcs.len() >= min_simultaneous {
            let md = (second_pos - bar_start).abs().min((second_pos - half_bar).abs());
            let place = if md > 0.3 && second_pcs.len() >= min_simultaneous { second_pos } else { half_bar };
            let ns = notes_at(second_pos);
            chords_json.push(chord_entry(second_pos, &second_pcs, "secondary", place, &ns));
        }

        // Fallback: downbeat
        if chords_json.is_empty() {
            let ns = notes_at(bar_start);
            if ns.len() >= config.min_active_notes {
                let mut pcs: Vec<u8> = ns.iter().map(|n| n.pitch % 12).collect();
                pcs.sort();
                pcs.dedup();
                chords_json.push(chord_entry(bar_start, &pcs, "fallback", bar_start, &ns));
            }
        }

        bars_json.push(serde_json::json!({
            "bar": bar_idx,
            "beat_start": bar_start,
            "beat_end": bar_end,
            "total_notes_in_bar": bar_notes.len(),
            "chords": chords_json,
        }));
    }

    let debug = serde_json::json!({
        "beats_per_bar": beats_per_bar,
        "chord_min_simultaneous": min_simultaneous,
        "bars": bars_json,
    });

    if let Ok(json_str) = serde_json::to_string_pretty(&debug) {
        let temp_dir = std::env::temp_dir();
        let _ = std::fs::write(temp_dir.join("chord_debug.json"), json_str);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_chord_progression() {
        let notes = vec![
            QuantizedNote {
                id: 1,
                pitch: 60,
                beat_start: 0.0,
                beat_duration: 2.0,
                velocity: 100,
                channel: None,
                confidence: 1.0,
                bar_index: 0,
                beat_index: 0,
                intra_beat_pos: 0.0,
                articulation: Articulation::Normal,
                swing_style: SwingStyle::Straight,
                swing_feel: false,
            },
            QuantizedNote {
                id: 2,
                pitch: 64,
                beat_start: 0.0,
                beat_duration: 2.0,
                velocity: 100,
                channel: None,
                confidence: 1.0,
                bar_index: 0,
                beat_index: 0,
                intra_beat_pos: 0.0,
                articulation: Articulation::Normal,
                swing_style: SwingStyle::Straight,
                swing_feel: false,
            },
            QuantizedNote {
                id: 3,
                pitch: 67,
                beat_start: 0.0,
                beat_duration: 2.0,
                velocity: 100,
                channel: None,
                confidence: 1.0,
                bar_index: 0,
                beat_index: 0,
                intra_beat_pos: 0.0,
                articulation: Articulation::Normal,
                swing_style: SwingStyle::Straight,
                swing_feel: false,
            },
            QuantizedNote {
                id: 4,
                pitch: 62,
                beat_start: 2.0,
                beat_duration: 2.0,
                velocity: 100,
                channel: None,
                confidence: 1.0,
                bar_index: 0,
                beat_index: 0,
                intra_beat_pos: 0.0,
                articulation: Articulation::Normal,
                swing_style: SwingStyle::Straight,
                swing_feel: false,
            },
            QuantizedNote {
                id: 5,
                pitch: 65,
                beat_start: 2.0,
                beat_duration: 2.0,
                velocity: 100,
                channel: None,
                confidence: 1.0,
                bar_index: 0,
                beat_index: 0,
                intra_beat_pos: 0.0,
                articulation: Articulation::Normal,
                swing_style: SwingStyle::Straight,
                swing_feel: false,
            },
            QuantizedNote {
                id: 6,
                pitch: 69,
                beat_start: 2.0,
                beat_duration: 2.0,
                velocity: 100,
                channel: None,
                confidence: 1.0,
                bar_index: 0,
                beat_index: 0,
                intra_beat_pos: 0.0,
                articulation: Articulation::Normal,
                swing_style: SwingStyle::Straight,
                swing_feel: false,
            },
        ];

        let chords = detect_chord_changes(notes.as_slice(), ChordAnalysisConfig::default());
        assert!(chords.len() >= 2);
        assert_eq!(chords[0].symbol, "C");
    }
}
