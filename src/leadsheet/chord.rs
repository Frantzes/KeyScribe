use std::collections::BTreeSet;

#[allow(unused_imports)]
use crate::leadsheet::types::{Articulation, ChordSymbolChange, QuantizedNote, SwingStyle};

#[derive(Debug, Clone, Copy)]
pub struct ChordAnalysisConfig {
    pub step_beats: f32,
    pub min_active_notes: usize,
}

impl Default for ChordAnalysisConfig {
    fn default() -> Self {
        Self {
            step_beats: 1.0,
            min_active_notes: 2,
        }
    }
}

pub fn detect_chord_changes(
    quantized_notes: &[QuantizedNote],
    config: ChordAnalysisConfig,
) -> Vec<ChordSymbolChange> {
    if quantized_notes.is_empty() {
        return Vec::new();
    }

    let max_beat = quantized_notes
        .iter()
        .map(|n| n.beat_start + n.beat_duration)
        .fold(0.0f32, f32::max);

    let mut out = Vec::new();
    let mut last_symbol = String::new();

    let mut beat = 0.0f32;
    let step = config.step_beats.max(0.25);
    while beat <= max_beat + 1.0e-4 {
        let mut pcs = BTreeSet::<u8>::new();
        let mut bass_pitch: Option<u8> = None;

        for note in quantized_notes {
            let note_end = note.beat_start + note.beat_duration;
            if beat + 1.0e-5 < note.beat_start || beat >= note_end {
                continue;
            }

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
            if score > best_score {
                best_score = score;
                best = Some((*suffix, score, covered));
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
            if score > best_score {
                best_score = score;
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

    // Alternative: chord rooted on the bass note
    if let Some(bass) = bass_pc {
        if let Some((alt_suffix, alt_score, _)) = best_template_for_root(pcs, bass, &TEMPLATES) {
            let threshold = (best_score as f32 * 0.75).ceil() as i32;
            let is_different = alt_suffix != best_suffix || bass != best_root;
            if alt_score >= threshold && is_different {
                let alt = format!("{}{}", pitch_class_name_flat(bass), alt_suffix);
                chord = format!("{} or {}", alt, chord);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_chord_progression() {
        let notes = vec![
            QuantizedNote {
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
