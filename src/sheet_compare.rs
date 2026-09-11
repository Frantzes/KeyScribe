//! Headless sheet comparison for the round-trip test loop.
//!
//! Extracts note pitches/onsets/durations from MusicXML (via roxmltree) and from
//! MIDI (via midly), then computes pitch/rhythm accuracy metrics so an agent can
//! self-evaluate how well a transcription survives a sheet -> audio -> sheet loop.

use std::path::Path;

use crate::midi::{parse_midi_notes, MidiNote};

/// Parse a MusicXML document, tolerating the DOCTYPE MuseScore emits.
fn parse_doc(xml: &str) -> anyhow::Result<roxmltree::Document<'_>> {
    roxmltree::Document::parse_with_options(
        xml,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("failed to parse MusicXML: {e}"))
}

/// Extract notes from a MusicXML string. Onsets/durations are expressed in
/// beats (divisions-normalized) so they compare directly with MIDI notes.
/// Rests are skipped. Multiple parts are merged in document order.
pub fn parse_musicxml_notes(xml: &str) -> anyhow::Result<Vec<MidiNote>> {
    let doc = parse_doc(xml)?;

    let divisions: f32 = doc
        .descendants()
        .find(|n| n.tag_name().name() == "divisions")
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);

    let mut notes: Vec<MidiNote> = Vec::new();

    // MusicXML voices are serialized sequentially inside each measure. A
    // `<backup>` returns the cursor to the start of the next voice; ignoring it
    // makes every later onset drift by the duration of preceding voices.
    for part in doc.descendants().filter(|n| n.tag_name().name() == "part") {
        let mut cursor_div = 0.0f32;
        let mut last_onset_beats = 0.0f32;
        let mut last_dur_beats = 0.0f32;

        for node in part.descendants() {
            match node.tag_name().name() {
                "backup" | "forward" => {
                    let duration_div = node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "duration")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.0);
                    if node.tag_name().name() == "backup" {
                        cursor_div -= duration_div;
                    } else {
                        cursor_div += duration_div;
                    }
                }
                "note" => {
                    let is_chord = node
                        .children()
                        .any(|c| c.is_element() && c.tag_name().name() == "chord");
                    let is_rest = node
                        .children()
                        .any(|c| c.is_element() && c.tag_name().name() == "rest");
                    let is_grace = node
                        .children()
                        .any(|c| c.is_element() && c.tag_name().name() == "grace");
                    let duration_div = node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "duration")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.0);

                    let onset_beats = if is_chord {
                        last_onset_beats
                    } else {
                        cursor_div / divisions
                    };
                    let dur_beats = duration_div / divisions;

                    if !is_chord && !is_grace {
                        last_onset_beats = onset_beats;
                        last_dur_beats = dur_beats;
                        cursor_div += duration_div;
                    }

                    if is_chord || is_rest || is_grace {
                        if is_chord && !is_rest {
                            if let Some(pitch) = note_pitch_midi(&node) {
                                notes.push(MidiNote {
                                    pitch,
                                    onset_beats,
                                    dur_beats: last_dur_beats,
                                });
                            }
                        }
                        continue;
                    }

                    if let Some(pitch) = note_pitch_midi(&node) {
                        notes.push(MidiNote {
                            pitch,
                            onset_beats,
                            dur_beats,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    notes.sort_by(|a, b| {
        a.onset_beats
            .partial_cmp(&b.onset_beats)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });
    Ok(notes)
}

/// A chord symbol parsed from a MusicXML `<harmony>` element, with its onset in
/// beats for alignment with the ground truth.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Harmony {
    pub onset_beats: f32,
    /// Root note name e.g. "C", "F#", "Bb" ("" if absent).
    pub root: String,
    /// MusicXML kind value e.g. "major-seventh", "minor" ("" if absent).
    pub kind: String,
    /// Displayed suffix from the `text` attribute (may be empty).
    pub suffix: String,
}

impl Harmony {
    /// Canonical symbol used for exact matching: root + kind, e.g. "Cmajor-seventh".
    pub fn canonical(&self) -> String {
        format!("{}{}", self.root, self.kind)
    }

    /// Human-readable symbol, e.g. "Cmaj7", "B-7", "G-13". Maps the MusicXML
    /// kind value (not the display `text` attribute) to conventional jazz
    /// shorthand.
    pub fn display(&self) -> String {
        let kind_display = match self.kind.as_str() {
            "" | "major" => "",
            "minor" => "-",
            "dominant" => "7",
            "major-seventh" => "maj7",
            "minor-seventh" => "-7",
            "diminished" => "dim",
            "augmented" => "aug",
            "diminished-seventh" => "dim7",
            "half-diminished" => "-7b5",
            "minor-major-seventh" => "-maj7",
            "major-sixth" => "6",
            "minor-sixth" => "-6",
            "suspended-fourth" => "sus4",
            "suspended-second" => "sus2",
            "major-ninth" => "maj9",
            "minor-ninth" => "-9",
            "dominant-ninth" => "9",
            "dominant-11th" => "11",
            "dominant-13th" => "13",
            "minor-13th" => "-13",
            other => other,
        };
        format!("{}{}", self.root, kind_display)
    }
}

/// Extract chord symbols from a MusicXML string in document order, with onsets
/// in beats. Walking order tracks cumulative note durations so a harmony
/// placed mid-bar gets the correct onset.
pub fn parse_musicxml_harmonies(xml: &str) -> anyhow::Result<Vec<Harmony>> {
    let doc = parse_doc(xml)?;

    let divisions: f32 = doc
        .descendants()
        .find(|n| n.tag_name().name() == "divisions")
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);

    let mut out: Vec<Harmony> = Vec::new();
    let mut cumulative: f32 = 0.0;

    for node in doc.descendants() {
        match node.tag_name().name() {
            "note" => {
                let is_chord = node
                    .children()
                    .any(|c| c.is_element() && c.tag_name().name() == "chord");
                let is_grace = node
                    .children()
                    .any(|c| c.is_element() && c.tag_name().name() == "grace");
                if !is_chord && !is_grace {
                    let duration_div: f32 = node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "duration")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.0);
                    cumulative += duration_div;
                }
            }
            "backup" => {
                let dur_div: f32 = node
                    .children()
                    .find(|c| c.is_element() && c.tag_name().name() == "duration")
                    .and_then(|c| c.text())
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                cumulative -= dur_div;
            }
            "forward" => {
                let dur_div: f32 = node
                    .children()
                    .find(|c| c.is_element() && c.tag_name().name() == "duration")
                    .and_then(|c| c.text())
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                cumulative += dur_div;
            }
            "harmony" => {
                let mut root = String::new();
                let mut kind = String::new();
                let mut suffix = String::new();
                for child in node.children().filter(|c| c.is_element()) {
                    match child.tag_name().name() {
                        "root" => {
                            if let Some(step) = child
                                .descendants()
                                .find(|n| n.tag_name().name() == "root-step")
                                .and_then(|n| n.text())
                            {
                                root = step.to_string();
                            }
                            if let Some(alter) = child
                                .descendants()
                                .find(|n| n.tag_name().name() == "root-alter")
                                .and_then(|n| n.text())
                            {
                                let a: i32 = alter.parse().unwrap_or(0);
                                if a == 1 {
                                    root.push('#');
                                } else if a == -1 {
                                    root.push('b');
                                }
                            }
                        }
                        "kind" => {
                            if let Some(text) = child.text() {
                                kind = text.to_string();
                            }
                            if let Some(text) = child.attribute("text") {
                                suffix = text.to_string();
                            }
                        }
                        _ => {}
                    }
                }
                out.push(Harmony {
                    onset_beats: cumulative / divisions,
                    root,
                    kind,
                    suffix,
                });
            }
            _ => {}
        }
    }

    out.sort_by(|a, b| a.onset_beats.partial_cmp(&b.onset_beats).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

/// Chord-symbol accuracy report between a reference and a transcription.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChordReport {
    pub reference_count: usize,
    pub transcription_count: usize,
    /// Fraction of transcription chords whose (root+kind) and onset match a
    /// reference chord within `onset_tolerance_beats`.
    pub exact_match_rate: f32,
    /// Fraction of transcription chords matching the reference root (ignoring
    /// kind/extension) within the same onset tolerance.
    pub root_match_rate: f32,
    pub mean_onset_error_beats: f32,
    pub onset_tolerance_beats: f32,
    /// Reference symbols in order (root + suffix, for human inspection).
    pub reference_symbols: Vec<String>,
    /// Transcription symbols in order (root + suffix).
    pub transcription_symbols: Vec<String>,
}

/// Greedy chord matching by canonical symbol (root+kind) within an onset
/// tolerance, plus root-only matching.
pub fn compare_harmonies(
    reference: &[Harmony],
    transcription: &[Harmony],
    onset_tolerance_beats: f32,
) -> ChordReport {
    let mut used_exact = vec![false; reference.len()];
    let mut used_root = vec![false; reference.len()];
    let mut exact = 0usize;
    let mut root_matches = 0usize;
    let mut onset_err_sum = 0.0f32;

    for th in transcription {
        let mut best_exact: Option<(usize, f32)> = None;
        let mut best_root: Option<(usize, f32)> = None;
        for (ri, rh) in reference.iter().enumerate() {
            let err = (rh.onset_beats - th.onset_beats).abs();
            if err > onset_tolerance_beats {
                continue;
            }
            if !used_exact[ri] && !rh.canonical().is_empty() && rh.canonical() == th.canonical() {
                if best_exact.map(|(_, be)| err < be).unwrap_or(true) {
                    best_exact = Some((ri, err));
                }
            }
            if !used_root[ri] && !rh.root.is_empty() && rh.root == th.root {
                if best_root.map(|(_, be)| err < be).unwrap_or(true) {
                    best_root = Some((ri, err));
                }
            }
        }
        if let Some((ri, err)) = best_exact {
            used_exact[ri] = true;
            used_root[ri] = true;
            exact += 1;
            onset_err_sum += err;
        } else if let Some((ri, _)) = best_root {
            used_root[ri] = true;
            root_matches += 1;
        }
    }

    let tc = transcription.len().max(1);
    ChordReport {
        reference_count: reference.len(),
        transcription_count: transcription.len(),
        exact_match_rate: exact as f32 / tc as f32,
        root_match_rate: (exact + root_matches) as f32 / tc as f32,
        mean_onset_error_beats: if exact > 0 {
            onset_err_sum / exact as f32
        } else {
            f32::NAN
        },
        onset_tolerance_beats,
        reference_symbols: reference
            .iter()
            .map(|h| h.display())
            .collect(),
        transcription_symbols: transcription
            .iter()
            .map(|h| h.display())
            .collect(),
    }
}

/// Load a note list from either a `.musicxml` or a `.mid` file based on the
/// extension.
fn note_pitch_midi(note: &roxmltree::Node) -> Option<u8> {
    let pitch = note
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "pitch")?;

    let step = pitch
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "step")?
        .text()?;
    let octave: i32 = pitch
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "octave")?
        .text()?
        .parse()
        .ok()?;
    let alter: i32 = pitch
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "alter")
        .and_then(|c| c.text())
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);

    let step_pc: i32 = match step.to_ascii_uppercase().as_str() {
        "C" => 0,
        "D" => 2,
        "E" => 4,
        "F" => 5,
        "G" => 7,
        "A" => 9,
        "B" => 11,
        _ => return None,
    };

    let midi = (octave + 1) * 12 + step_pc + alter;
    if (0..=127).contains(&midi) {
        Some(midi as u8)
    } else {
        None
    }
}

/// Accuracy report for comparing a ground-truth/reference note list against a
/// transcription.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompareReport {
    /// Total notes in the reference (ground truth).
    pub reference_note_count: usize,
    /// Total notes in the transcription (candidate).
    pub transcription_note_count: usize,
    /// Fraction of transcription notes whose pitch exists anywhere in the
    /// reference (pitch-level accuracy, onset-independent).
    pub pitch_accuracy: f32,
    /// Fraction of transcription notes that match a reference note on both
    /// pitch and onset (within `onset_tolerance_beats`).
    pub note_accuracy: f32,
    /// Fraction of reference notes that were recovered by the transcription.
    pub recall: f32,
    /// Mean |onset_transcribed - onset_reference| in beats for pitch-matched
    /// notes (only the nearest reference note per pitch).
    pub mean_onset_error_beats: f32,
    /// Mean relative duration error for pitch+onset matched notes (0 = perfect).
    pub mean_duration_error: f32,
    pub onset_tolerance_beats: f32,
}

/// Compare a reference note list against a transcribed note list.
///
/// Matching is greedy: for each transcription note we look for the nearest
/// reference note with the same pitch whose onset is within
/// `onset_tolerance_beats`. Each reference note may be matched at most once.
pub fn compare_note_lists(
    reference: &[MidiNote],
    transcription: &[MidiNote],
    onset_tolerance_beats: f32,
) -> CompareReport {
    let mut used = vec![false; reference.len()];
    let mut onset_matches = 0usize;
    let mut onset_err_sum = 0.0f32;
    let mut dur_err_sum = 0.0f32;

    for tn in transcription {
        let mut best: Option<(usize, f32)> = None;
        for (ri, rn) in reference.iter().enumerate() {
            if used[ri] || rn.pitch != tn.pitch {
                continue;
            }
            let err = (rn.onset_beats - tn.onset_beats).abs();
            if err <= onset_tolerance_beats {
                if best.map(|(_, be)| err < be).unwrap_or(true) {
                    best = Some((ri, err));
                }
            }
        }
        if let Some((ri, err)) = best {
            used[ri] = true;
            onset_matches += 1;
            onset_err_sum += err;
            let r_dur = reference[ri].dur_beats.max(1e-3);
            dur_err_sum += ((tn.dur_beats - reference[ri].dur_beats).abs() / r_dur).min(2.0);
        }
    }

    let transcription_note_count = transcription.len().max(1);
    let reference_note_count = reference.len().max(1);

    // Pitch-only accuracy: what fraction of the transcription's note instances
    // have a matching pitch somewhere in the reference (greedy, onset-independent)?
    let mut pitch_matches = 0usize;
    let mut used_pitch = vec![false; reference.len()];
    for tn in transcription {
        let found = reference
            .iter()
            .enumerate()
            .find(|(ri, rn)| !used_pitch[*ri] && rn.pitch == tn.pitch);
        if let Some((ri, _)) = found {
            used_pitch[ri] = true;
            pitch_matches += 1;
        }
    }

    CompareReport {
        reference_note_count: reference.len(),
        transcription_note_count: transcription.len(),
        pitch_accuracy: pitch_matches as f32 / transcription_note_count as f32,
        note_accuracy: onset_matches as f32 / transcription_note_count as f32,
        recall: onset_matches as f32 / reference_note_count as f32,
        mean_onset_error_beats: if onset_matches > 0 {
            onset_err_sum / onset_matches as f32
        } else {
            f32::NAN
        },
        mean_duration_error: if onset_matches > 0 {
            dur_err_sum / onset_matches as f32
        } else {
            f32::NAN
        },
        onset_tolerance_beats,
    }
}

#[cfg(test)]
mod a1_diag {
    use super::*;
    use std::path::Path;

    fn per_bar_matches(refn: &[MidiNote], trans: &[MidiNote], tol: f32) -> Vec<(u32, usize, f32)> {
        let mut out = Vec::new();
        for bar in 0..100 {
            let r: Vec<&MidiNote> = refn.iter().filter(|n| (n.onset_beats / 4.0).floor() as u32 == bar).collect();
            let t: Vec<&MidiNote> = trans.iter().filter(|n| (n.onset_beats / 4.0).floor() as u32 == bar).collect();
            if r.is_empty() && t.is_empty() { continue; }
            let rep = compare_note_lists(
                &r.iter().map(|n| (**n).clone()).collect::<Vec<_>>(),
                &t.iter().map(|n| (**n).clone()).collect::<Vec<_>>(),
                tol,
            );
            out.push((bar, rep.transcription_note_count, rep.note_accuracy * rep.transcription_note_count as f32));
        }
        out
    }

    #[test]
    #[ignore]
    fn diag() {
        let refn = load_notes(Path::new("out/omnibook/Confirmation.musicxml")).unwrap();
        let no = load_notes(Path::new("C:/Users/Fran/AppData/Local/Temp/opencode/conf_no_a1.musicxml")).unwrap();
        let co = load_notes(Path::new("C:/Users/Fran/AppData/Local/Temp/opencode/conf_a1c.musicxml")).unwrap();
        eprintln!("DIAG ref={} no={} co={}", refn.len(), no.len(), co.len());
        for bar in [0u32, 1, 7, 30, 31, 35] {
            let r: Vec<MidiNote> = refn.iter().filter(|n| (n.onset_beats / 4.0).floor() as u32 == bar).cloned().collect();
            let n: Vec<MidiNote> = no.iter().filter(|n| (n.onset_beats / 4.0).floor() as u32 == bar).cloned().collect();
            let c: Vec<MidiNote> = co.iter().filter(|n| (n.onset_beats / 4.0).floor() as u32 == bar).cloned().collect();
            eprintln!("DIAG bar{} REF: {:?}", bar, r.iter().map(|n| format!("{}({:.2}/{:.2})", n.pitch, n.onset_beats, n.dur_beats)).collect::<Vec<_>>().join(" "));
            eprintln!("DIAG bar{} NO : {:?}", bar, n.iter().map(|n| format!("{}({:.2}/{:.2})", n.pitch, n.onset_beats, n.dur_beats)).collect::<Vec<_>>().join(" "));
            eprintln!("DIAG bar{} CO : {:?}", bar, c.iter().map(|n| format!("{}({:.2}/{:.2})", n.pitch, n.onset_beats, n.dur_beats)).collect::<Vec<_>>().join(" "));
        }
        let pn = per_bar_matches(&refn, &no, 0.25);
        let pc = per_bar_matches(&refn, &co, 0.25);
        for (bar, cnt_n, m_n) in &pn {
            let m_c = pc.iter().find(|(b, _, _)| b == bar).map(|(_, _, m)| *m).unwrap_or(0.0);
            let cnt_c = pc.iter().find(|(b, _, _)| b == bar).map(|(_, c, _)| *c).unwrap_or(0);
            let delta = m_c as i32 - *m_n as i32;
            if *cnt_n > 0 && delta != 0 {
                eprintln!("DIAG bar {} no={}matches({}) co={}matches({}) delta={}",
                    bar, cnt_n, m_n, cnt_c, m_c, delta);
            }
        }
    }
}
pub fn parse_beats_per_bar(xml: &str) -> anyhow::Result<u32> {
    let doc = parse_doc(xml)?;
    let beats = doc
        .descendants()
        .filter(|n| n.tag_name().name() == "time")
        .filter_map(|t| {
            t.children()
                .find(|c| c.is_element() && c.tag_name().name() == "beats")
        })
        .filter_map(|b| b.text().and_then(|s| s.parse::<u32>().ok()))
        .next()
        .unwrap_or(4);
    Ok(beats)
}

/// Downbeat / bar-placement report between a reference and a transcription.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BarPlacementReport {
    pub reference_bpb: u32,
    pub transcription_bpb: u32,
    /// Note pairs matched on pitch + onset (same greedy rule as
    /// `compare_note_lists`).
    pub matched_pairs: usize,
    /// Matched pairs whose position within the bar agrees.
    pub in_bar_matches: usize,
    /// Fraction of matched pairs whose in-bar beat position agrees. Detects
    /// wrong downbeat phase / half-time grids that would still pass a plain
    /// onset comparison.
    pub in_bar_rate: f32,
    /// Total bars spanned by the reference and transcription.
    pub reference_bars: u32,
    pub transcription_bars: u32,
    /// 1 - |trans_bars - ref_bars| / ref_bars; catches doubled/halved tempos.
    pub bar_count_score: f32,
}

/// Compare the beat position of matched notes *within their bar* (onset mod
/// beats-per-bar) to check the transcription's downbeats land correctly and
/// the melody falls in the right place in the bar.
pub fn compare_bar_placement(
    reference: &[MidiNote],
    transcription: &[MidiNote],
    reference_bpb: u32,
    transcription_bpb: u32,
    onset_tolerance_beats: f32,
    in_bar_tolerance_beats: f32,
) -> BarPlacementReport {
    let ref_bpb = reference_bpb.max(1);
    let trans_bpb = transcription_bpb.max(1);

    let mut used = vec![false; reference.len()];
    let mut matched = 0usize;
    let mut in_bar = 0usize;
    for tn in transcription {
        let mut best: Option<(usize, f32)> = None;
        for (ri, rn) in reference.iter().enumerate() {
            if used[ri] || rn.pitch != tn.pitch {
                continue;
            }
            let err = (rn.onset_beats - tn.onset_beats).abs();
            if err <= onset_tolerance_beats && best.map(|(_, be)| err < be).unwrap_or(true) {
                best = Some((ri, err));
            }
        }
        if let Some((ri, _)) = best {
            used[ri] = true;
            matched += 1;
            let r_pos = reference[ri].onset_beats.rem_euclid(ref_bpb as f32);
            let t_pos = tn.onset_beats.rem_euclid(trans_bpb as f32);
            if (r_pos - t_pos).abs() <= in_bar_tolerance_beats {
                in_bar += 1;
            }
        }
    }

    let ref_last = reference
        .iter()
        .map(|n| n.onset_beats)
        .fold(0.0f32, f32::max);
    let trans_last = transcription
        .iter()
        .map(|n| n.onset_beats)
        .fold(0.0f32, f32::max);
    let ref_bars = (ref_last / ref_bpb as f32).ceil() as u32;
    let trans_bars = (trans_last / trans_bpb as f32).ceil() as u32;
    let bar_count_score = if ref_bars > 0 {
        (1.0 - ((trans_bars as f32 - ref_bars as f32).abs() / ref_bars as f32).min(1.0)).max(0.0)
    } else {
        0.0
    };

    BarPlacementReport {
        reference_bpb,
        transcription_bpb,
        matched_pairs: matched,
        in_bar_matches: in_bar,
        in_bar_rate: if matched > 0 {
            in_bar as f32 / matched as f32
        } else {
            0.0
        },
        reference_bars: ref_bars,
        transcription_bars: trans_bars,
        bar_count_score,
    }
}

/// Load a note list from either a `.musicxml` or a `.mid` file based on the
/// extension.
pub fn load_notes(path: &Path) -> anyhow::Result<Vec<MidiNote>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "mid" | "midi" => parse_midi_notes(path),
        "musicxml" | "xml" | "mxl" => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
            parse_musicxml_notes(&text)
        }
        _ => Err(anyhow::anyhow!(
            "unsupported format '{}' (expected .musicxml or .mid)",
            path.display()
        )),
    }
}

/// Load chord symbols from a MusicXML/MXL file. Returns an empty list for MIDI
/// (which has no harmony data).
pub fn load_harmonies(path: &Path) -> anyhow::Result<Vec<Harmony>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "mid" | "midi" => Ok(Vec::new()),
        "musicxml" | "xml" | "mxl" => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
            parse_musicxml_harmonies(&text)
        }
        _ => Ok(Vec::new()),
    }
}
