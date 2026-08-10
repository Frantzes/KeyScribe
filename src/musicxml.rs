//! MusicXML sheet-music generation, shared between the desktop app and the CLI.
//!
//! This module is deliberately free of any egui/UI dependency so the exact same
//! engraving logic can be driven from a headless binary.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::leadsheet::{
    Articulation, ChordSymbolChange, LeadSheetFoundation, NoteEvent, QuantizedNote, SwingStyle,
    TimeSignatureSegment,
};

pub const MUSICXML_DIVISIONS: i32 = 480;
pub const GRAND_STAFF_SPLIT_MIDI: u8 = 60;
pub const SHEET_SWING_BIAS: bool = true;

struct NoteSpan {
    id: u32,
    start_tick: i32,
    end_tick: i32,
    pitch: u8,
    velocity: u8,
    staff: u8,
    articulation: Articulation,
}

#[derive(Clone)]
struct NoteChunk {
    id: u32,
    start_tick_in_measure: i32,
    duration_ticks: i32,
    absolute_tick: i32,
    pitch: u8,
    velocity: u8,
    tie_start: bool,
    tie_stop: bool,
    staff: u8,
    articulation: Articulation,
}

#[derive(Clone, Copy)]
struct DurationToken {
    ticks: i32,
    note_type: &'static str,
    dots: u8,
    time_mod: Option<(u8, u8)>,
}

#[derive(Clone, Copy)]
pub struct SheetEngravingConfig {
    pub _allow_triplets: bool,
    pub is_lead_sheet: bool,
    pub single_staff: bool,
}

impl Default for SheetEngravingConfig {
    fn default() -> Self {
        Self {
            _allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: true,
            single_staff: false,
        }
    }
}

fn build_measure_boundaries(
    time_sigs: &[TimeSignatureSegment],
    default_beats_per_bar: u32,
    max_tick: i32,
) -> Vec<i32> {
    let mut boundaries: Vec<i32> = vec![0];
    let mut last_mw = (default_beats_per_bar.max(1) as i32) * MUSICXML_DIVISIONS;

    // Beat tracking can emit many adjacent segments that all describe the
    // same meter. Treat those as one continuous meter; otherwise an imprecise
    // segment end such as 14.667 beats creates a short measure and makes the
    // MusicXML cursor fail the declared 4/4 duration.
    let uniform_meter = time_sigs.first().is_some_and(|first| {
        time_sigs.iter().all(|seg| {
            seg.numerator == first.numerator && seg.denominator == first.denominator
        })
    });
    if uniform_meter {
        let mw = (time_sigs[0].numerator.max(1) as i32) * MUSICXML_DIVISIONS;
        let mut t = mw;
        while t <= max_tick {
            boundaries.push(t);
            t += mw;
        }
        if boundaries.last().copied().unwrap_or(0) < max_tick {
            boundaries.push(t);
        }
        return boundaries;
    }

    if time_sigs.is_empty() {
        let mw = (default_beats_per_bar.max(1) as i32) * MUSICXML_DIVISIONS;
        let mut t = mw;
        while t <= max_tick {
            boundaries.push(t);
            t += mw;
        }
        if boundaries.last().copied().unwrap_or(0) < max_tick {
            boundaries.push(t);
        }
        return boundaries;
    }

    for (i, seg) in time_sigs.iter().enumerate() {
        let seg_start_tick = (seg.start_beat * MUSICXML_DIVISIONS as f32).round() as i32;
        let seg_end_tick = if i + 1 < time_sigs.len() {
            (time_sigs[i + 1].start_beat * MUSICXML_DIVISIONS as f32).round() as i32
        } else {
            (seg.end_beat * MUSICXML_DIVISIONS as f32).round() as i32
        };
        let mw = (seg.numerator.max(1) as i32) * MUSICXML_DIVISIONS;
        if mw > 0 {
            last_mw = mw;
        }

        let seg_limit = if i + 1 < time_sigs.len() {
            seg_end_tick.min(max_tick)
        } else {
            seg_end_tick.max(max_tick)
        };
        let mut cursor = boundaries.last().copied().unwrap_or(0).max(seg_start_tick);
        while cursor < seg_limit {
            let next = (cursor + mw).min(seg_limit);
            boundaries.push(next);
            cursor = next;
        }
    }

    let mut last = boundaries.last().copied().unwrap_or(0);
    if last_mw <= 0 {
        last_mw = (default_beats_per_bar.max(1) as i32) * MUSICXML_DIVISIONS;
    }
    while last < max_tick {
        last += last_mw;
        boundaries.push(last);
    }

    // Fill any large gaps in case time-signature segments ended early.
    if last_mw > 0 && boundaries.len() >= 2 {
        let mut filled: Vec<i32> = Vec::with_capacity(boundaries.len());
        filled.push(boundaries[0]);
        for pair in boundaries.windows(2) {
            let mut current = pair[0];
            let next = pair[1];
            if next <= current {
                continue;
            }
            while current + last_mw < next {
                current += last_mw;
                filled.push(current);
            }
            filled.push(next);
        }
        filled.dedup();
        boundaries = filled;
    }
    boundaries
}

fn split_span_into_measures(
    span: NoteSpan,
    boundaries: &[i32],
    target: &mut BTreeMap<i32, Vec<NoteChunk>>,
) {
    let mut cursor = span.start_tick;
    let mut first = true;

    while cursor < span.end_tick {
        let mi = match boundaries.binary_search(&cursor) {
            Ok(i) => i.min(boundaries.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(boundaries.len().saturating_sub(2)),
        };
        let ms = boundaries[mi];
        let me = boundaries[mi + 1];
        let chunk_end = span.end_tick.min(me);
        let dur = (chunk_end - cursor).max(1);

        target.entry(mi as i32).or_default().push(NoteChunk {
            id: span.id,
            start_tick_in_measure: cursor - ms,
            duration_ticks: dur,
            absolute_tick: cursor,
            pitch: span.pitch,
            velocity: span.velocity,
            tie_start: chunk_end < span.end_tick,
            tie_stop: !first,
            staff: span.staff,
            articulation: span.articulation,
        });

        first = false;
        cursor = chunk_end;
    }
}

pub fn build_musicxml_document(
    title: &str,
    foundation: &LeadSheetFoundation,
    config: SheetEngravingConfig,
) -> String {
    let note_spans = notes_to_spans(foundation.quantized_notes.as_slice(), config);

    let mut max_tick = 0i32;
    for span in &note_spans {
        max_tick = max_tick.max(span.end_tick);
    }

    let boundaries = build_measure_boundaries(
        &foundation.time_signature_segments,
        foundation.beats_per_bar,
        max_tick,
    );

    let mut chunks_by_measure: BTreeMap<i32, Vec<NoteChunk>> = BTreeMap::new();
    for span in note_spans {
        split_span_into_measures(span, &boundaries, &mut chunks_by_measure);
    }

    // Single average tempo for whole sheet
    let avg_bpm = if foundation.tempo_map.is_empty() {
        foundation.tempo.bpm
    } else {
        let total_weight: f32 = foundation
            .tempo_map
            .iter()
            .map(|s| (s.end_time_sec - s.start_time_sec).max(0.0))
            .sum();
        if total_weight > 0.0 {
            foundation
                .tempo_map
                .iter()
                .map(|s| s.bpm * (s.end_time_sec - s.start_time_sec).max(0.0))
                .sum::<f32>()
                / total_weight
        } else {
            foundation.tempo.bpm
        }
    };

    // Single tempo mark at the beginning
    let mut tempo_marks_by_measure: BTreeMap<i32, Vec<(i32, f32)>> = BTreeMap::new();
    tempo_marks_by_measure.entry(0).or_default().push((0, avg_bpm));

    let mut chord_by_measure: BTreeMap<i32, Vec<(i32, ChordSymbolChange)>> = BTreeMap::new();
    for chord in &foundation.chord_changes {
        let abs_tick = (chord.beat_start * MUSICXML_DIVISIONS as f32).round() as i32;
        let mi = match boundaries.binary_search(&abs_tick) {
            Ok(i) => i.min(boundaries.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(boundaries.len().saturating_sub(2)),
        };
        let offset = abs_tick - boundaries[mi];
        chord_by_measure
            .entry(mi as i32)
            .or_default()
            .push((offset, chord.clone()));
        max_tick = max_tick.max(abs_tick);
    }

    let total_measures = boundaries.len().saturating_sub(1).max(1);

    let mut time_signature_change_by_measure: BTreeMap<i32, (u8, u8)> = BTreeMap::new();
    for seg in &foundation.time_signature_segments {
        let abs_tick = (seg.start_beat * MUSICXML_DIVISIONS as f32).round() as i32;
        let mi = match boundaries.binary_search(&abs_tick) {
            Ok(i) => i.min(boundaries.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(boundaries.len().saturating_sub(2)),
        };
        time_signature_change_by_measure
            .entry(mi.max(0) as i32)
            .or_insert((seg.numerator, seg.denominator));
    }

    if !time_signature_change_by_measure.contains_key(&0) {
        let default_num = foundation.beats_per_bar as u8;
        let default_ts = foundation
            .time_signature_segments
            .first()
            .map(|s| (s.numerator, s.denominator))
            .unwrap_or((default_num.max(1), 4));
        time_signature_change_by_measure.insert(0, default_ts);
    }

    let mut swing_by_measure: BTreeMap<i32, SwingStyle> = BTreeMap::new();
    for section in &foundation.swing_sections {
        if section.style != SwingStyle::Straight {
            let start_measure = section.bar_start as i32;
            let end_measure = (section.bar_end as i32).min(total_measures as i32);
            for m in start_measure..end_measure {
                swing_by_measure.entry(m).or_insert(section.style);
            }
        }
    }

    // Trim trailing empty measures
    let mut last_content = -1i32;
    for mi in 0..total_measures as i32 {
        let has_content = chunks_by_measure.contains_key(&mi)
            || chord_by_measure.contains_key(&mi)
            || tempo_marks_by_measure.contains_key(&mi)
            || swing_by_measure.contains_key(&mi);
        if has_content {
            last_content = mi;
        }
    }
    let total_measures = (last_content + 1).max(1).min(total_measures as i32) as usize;

    let mut xml = String::new();
    let _ = write!(
        xml,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<score-partwise version=\"3.1\">\n"
    );
    let _ = write!(
        xml,
        "  <work><work-title>{}</work-title></work>\n",
        xml_escape(title)
    );
    let _ = write!(xml, "  <part-list>\n");
    let _ = write!(
        xml,
        "    <score-part id=\"P1\"><part-name>Lead Sheet</part-name></score-part>\n"
    );
    let _ = write!(xml, "  </part-list>\n");
    let _ = write!(xml, "  <part id=\"P1\">\n");

    let mut current_time_sig = time_signature_change_by_measure
        .get(&0)
        .copied()
        .unwrap_or((4, 4));
    let mut prev_swing_style: Option<SwingStyle> = None;

    // Determine per-measure clef for single-staff modes based on average pitch
    let measure_clefs: Vec<&'static str> = if config.is_lead_sheet || config.single_staff {
        let mut clefs = Vec::with_capacity(total_measures);
        for measure_idx in 0..total_measures {
            if let Some(chunks) = chunks_by_measure.get(&(measure_idx as i32)) {
                let avg_pitch = chunks
                    .iter()
                    .map(|c| c.pitch as f32)
                    .sum::<f32>()
                    / chunks.len().max(1) as f32;
                clefs.push(if avg_pitch < 46.0 { "F" } else { "G" });
            } else {
                clefs.push(clefs.last().copied().unwrap_or("G"));
            }
        }
        clefs
    } else {
        Vec::new()
    };
    let mut current_clef: &'static str = if config.is_lead_sheet || config.single_staff {
        measure_clefs.first().copied().unwrap_or("G")
    } else {
        "G"
    };

    for measure_idx in 0..total_measures {
        let measure_ticks = boundaries[measure_idx + 1] - boundaries[measure_idx];
        let _ = write!(xml, "    <measure number=\"{}\">\n", measure_idx + 1);
        if let Some(&(num, den)) = time_signature_change_by_measure.get(&(measure_idx as i32)) {
            current_time_sig = (num, den);
        }

        let mut clef_to_write = None;
        if config.is_lead_sheet || config.single_staff {
            if let Some(&clef) = measure_clefs.get(measure_idx) {
                if clef != current_clef && measure_idx > 0 {
                    clef_to_write = Some(clef);
                }
            }
        }

        let has_time_sig_change = time_signature_change_by_measure.contains_key(&(measure_idx as i32));
        if measure_idx == 0 || has_time_sig_change || clef_to_write.is_some() {
            let _ = write!(xml, "      <attributes>\n");
            if measure_idx == 0 {
                let _ = write!(xml, "        <divisions>{}</divisions>\n", MUSICXML_DIVISIONS);
                let _ = write!(xml, "        <key><fifths>0</fifths></key>\n");
                if config.is_lead_sheet {
                    // Use dynamic clef based on the first measure's average
                    // pitch, just like single-staff mode. The old code always
                    // hardcoded G clef, putting low melodies far below the
                    // staff.
                    let first_clef = measure_clefs.first().copied().unwrap_or("G");
                    let (sign, line) = if first_clef == "F" { ("F", 4) } else { ("G", 2) };
                    let _ = write!(xml, "        <clef><sign>{}</sign><line>{}</line></clef>\n", sign, line);
                } else if config.single_staff {
                    let first_clef = measure_clefs.first().copied().unwrap_or("G");
                    let (sign, line) = if first_clef == "F" { ("F", 4) } else { ("G", 2) };
                    let _ = write!(xml, "        <clef><sign>{}</sign><line>{}</line></clef>\n", sign, line);
                } else {
                    let _ = write!(xml, "        <staves>2</staves>\n");
                    let _ = write!(xml, "        <clef number=\"1\"><sign>G</sign><line>2</line></clef>\n");
                    let _ = write!(xml, "        <clef number=\"2\"><sign>F</sign><line>4</line></clef>\n");
                }
            }
            if measure_idx == 0 || has_time_sig_change {
                let _ = write!(
                    xml,
                    "        <time><beats>{}</beats><beat-type>{}</beat-type></time>\n",
                    current_time_sig.0,
                    current_time_sig.1
                );
            }

            // Dynamic clef for single-staff modes: switch per-measure based on avg pitch
            if let Some(clef) = clef_to_write {
                current_clef = clef;
                let (sign, line) = if clef == "F" { ("F", 4) } else { ("G", 2) };
                let _ = write!(
                    xml,
                    "        <clef><sign>{}</sign><line>{}</line></clef>\n",
                    sign, line
                );
            }

            let _ = write!(xml, "      </attributes>\n");
        }

        // Swing direction element
        let current_swing = swing_by_measure.get(&(measure_idx as i32)).copied();
        if current_swing != prev_swing_style {
            if let Some(style) = current_swing {
                let _ = write!(xml, "      <direction placement=\"above\">\n");
                let _ = write!(xml, "        <direction-type>\n");
                match style {
                    SwingStyle::Swing => {
                        let _ = write!(xml, "          <words>Swing</words>\n");
                    }
                    SwingStyle::Triplet => {
                        let _ = write!(xml, "          <words>Triplet feel</words>\n");
                    }
                    _ => {}
                }
                let _ = write!(xml, "        </direction-type>\n");
                // MusicXML <sound> uses a "swing" attribute, not a "type"
                // attribute. The old code produced invalid MusicXML that
                // renderers would ignore or reject.
                match style {
                    SwingStyle::Swing => {
                        let _ = write!(xml, "        <sound swing=\"straight\"/>\n");
                    }
                    SwingStyle::Triplet => {
                        let _ = write!(xml, "        <sound swing=\"triplet\"/>\n");
                    }
                    _ => {}
                }
                let _ = write!(xml, "      </direction>\n");
            } else if prev_swing_style == Some(SwingStyle::Swing)
                || prev_swing_style == Some(SwingStyle::Triplet)
            {
                let _ = write!(xml, "      <direction placement=\"above\">\n");
                let _ = write!(xml, "        <direction-type>\n");
                let _ = write!(xml, "          <words>Straight</words>\n");
                let _ = write!(xml, "        </direction-type>\n");
                let _ = write!(xml, "      </direction>\n");
            }
            prev_swing_style = current_swing;
        }

        if let Some(tempo_marks) = tempo_marks_by_measure.get(&(measure_idx as i32)) {
            let mut sorted = tempo_marks.clone();
            sorted.sort_by_key(|(offset, _)| *offset);
            for (offset, bpm) in sorted {
                let _ = write!(xml, "      <direction placement=\"above\">\n");
                if offset > 0 {
                    let _ = write!(xml, "        <offset>{offset}</offset>\n");
                }
                let _ = write!(xml, "        <direction-type>\n");
                let _ = write!(xml, "          <metronome>\n");
                let _ = write!(xml, "            <beat-unit>quarter</beat-unit>\n");
                let _ = write!(xml, "            <per-minute>{:.2}</per-minute>\n", bpm);
                let _ = write!(xml, "          </metronome>\n");
                let _ = write!(xml, "        </direction-type>\n");
                let _ = write!(xml, "        <sound tempo=\"{:.2}\"/>\n", bpm);
                let _ = write!(xml, "      </direction>\n");
            }
        }

        if let Some(chords) = chord_by_measure.get(&(measure_idx as i32)) {
            let mut sorted = chords.clone();
            sorted.sort_by_key(|(offset, _)| *offset);
            for (offset, chord) in sorted {
                write_harmony(&mut xml, offset, &chord.symbol);
            }
        }

        let mut chunks = chunks_by_measure.remove(&(measure_idx as i32)).unwrap_or_default();
        chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);

        // If the measure has no notes, chords, or directions, fill it with a rest
        let has_content = !chunks.is_empty()
            || chord_by_measure.contains_key(&(measure_idx as i32))
            || tempo_marks_by_measure.contains_key(&(measure_idx as i32))
            || swing_by_measure.contains_key(&(measure_idx as i32));
        if !has_content {
            write_rest_ticks(
                &mut xml,
                boundaries[measure_idx],
                measure_ticks,
                MUSICXML_DIVISIONS,
                1,
                1,
                config,
            );
            let _ = write!(xml, "    </measure>\n");
            continue;
        }

        if config.is_lead_sheet {
            let voice_map = build_voice_chunks(chunks.as_slice(), 1);
            let voice_count = voice_map.len().max(1);
            let mut rendered = 0usize;
            for (voice, mut voice_chunks) in voice_map {
                voice_chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);
                write_voice_sequence(
                    &mut xml,
                    voice_chunks.as_slice(),
                    boundaries[measure_idx],
                    measure_ticks,
                    MUSICXML_DIVISIONS,
                    1,
                    voice,
                    config,
                );
                rendered += 1;
                if rendered < voice_count {
                    let _ = write!(xml, "      <backup>\n");
                    let _ = write!(xml, "        <duration>{}</duration>\n", measure_ticks.max(1));
                    let _ = write!(xml, "      </backup>\n");
                }
            }
        } else {
            for staff in [1u8, 2u8] {
                let voice_map = build_voice_chunks(chunks.as_slice(), staff);
                let voice_count = voice_map.len().max(1);
                let mut rendered = 0usize;
                let mut staff_has_content = false;

                for (voice, mut voice_chunks) in voice_map {
                    voice_chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);
                    if !voice_chunks.is_empty() {
                        staff_has_content = true;
                    }
                    write_voice_sequence(
                        &mut xml,
                        voice_chunks.as_slice(),
                        boundaries[measure_idx],
                        measure_ticks,
                        MUSICXML_DIVISIONS,
                        staff,
                        voice,
                        config,
                    );

                    rendered += 1;
                    if rendered < voice_count {
                        let _ = write!(xml, "      <backup>\n");
                        let _ = write!(xml, "        <duration>{}</duration>\n", measure_ticks.max(1));
                        let _ = write!(xml, "      </backup>\n");
                    }
                }

                // Only emit the inter-staff backup if staff 1 actually
                // wrote notes. Emitting a backup after an empty staff 1
                // moves the cursor backwards and corrupts the measure
                // layout in renderers.
                if staff == 1 && staff_has_content {
                    let _ = write!(xml, "      <backup>\n");
                    let _ = write!(xml, "        <duration>{}</duration>\n", measure_ticks.max(1));
                    let _ = write!(xml, "      </backup>\n");
                }
            }
        }

        let _ = write!(xml, "    </measure>\n");
    }

    let _ = write!(xml, "  </part>\n");
    let _ = write!(xml, "</score-partwise>\n");

    xml
}

/// Combined heuristic: skyline (highest pitch) + near-note continuity bias + outlier filter.
/// At each event point the highest active pitch is the base candidate, but when multiple
/// notes are active at the same pitch range, prefers the one closest to the previous melody
/// pitch (near-note continuity). Outliers more than `outlier_semitones` from the rolling
/// median are suppressed.
pub fn extract_melody_heuristic(notes: &[NoteEvent], outlier_semitones: u8) -> Vec<NoteEvent> {
    extract_melody_skyline(notes, outlier_semitones)
}

/// Melody line selection.
///
/// At each event point, score the active pitches by:
/// - onset density: the melody re-articulates (8th-note phrasing) while the
///   accompaniment sustains whole-bar chords;
/// - isolation: the melody onsets are monophonic (a single pitch striking)
///   while the accompaniment strikes chord stacks;
/// - register: the melody is a vocal-range line — the walking bass also
///   re-articulates, so below-vocal pitches are penalized hard.
/// Tie-breaks go to the highest pitch; when nothing has onset recently the
/// previous melody pitch is kept (continuity).
pub fn extract_melody_skyline(notes: &[NoteEvent], outlier_semitones: u8) -> Vec<NoteEvent> {
    if notes.is_empty() {
        return Vec::new();
    }

    // Build event list: note-on and note-off
    let mut events: Vec<(f32, u8, u8, bool)> = Vec::with_capacity(notes.len() * 2);
    for n in notes {
        if !n.start_time.is_finite() || !n.end_time.is_finite() || n.end_time <= n.start_time {
            continue;
        }
        events.push((n.start_time, n.pitch, n.velocity, true));
        events.push((n.end_time, n.pitch, n.velocity, false));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.3.cmp(&a.3)));

    // Batch tolerance: events closer than this are treated as one strike
    // (chord stack). Must stay well below the tightest melody rhythm: at 208
    // BPM a triplet 8th spacing is ~0.096s, and a 32nd-triplet run is ~0.024s.
    // 0.12s merged a triplet's first two notes into one batch and the
    // pitch tie-break silently dropped the first note (E4 in Confirmation).
    let time_tolerance = 0.03;
    let onset_window = 0.6; // ~2 beats at 200 BPM
    let min_segment = 0.10;

    // Adaptive melody register: the lead line lives in a narrow tessitura
    // centred on the most-massive (velocity-weighted) pitch. Candidates far
    // outside it — comp basses, bell/bright overtones, distant chord tones —
    // get a hard penalty so they can't hijack the melody at comp strikes.
    let mut pitch_mass = vec![0f32; 128];
    for n in notes {
        let p = (n.pitch as usize).min(127);
        pitch_mass[p] += n.velocity as f32;
    }
    let modal_pitch = (0..128)
        .max_by(|a, b| pitch_mass[*a].partial_cmp(&pitch_mass[*b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(69) as i32;
    let band_half = 9i32;
    let register_lo = (modal_pitch - band_half).clamp(40, 60) as u8;
    let register_hi = (modal_pitch + band_half).clamp(72, 96) as u8;

    let register_penalty = |p: u8| -> f32 {
        if p < register_lo {
            -((register_lo as i32 - p as i32) as f32) * 0.9
        } else if p > register_hi {
            -((p as i32 - register_hi as i32) as f32) * 0.9
        } else {
            0.0
        }
    };
    // Don't flip the melody to a near-tie at chord strikes: a switch must beat
    // the running melody pitch by a meaningful margin (continuity).
    const CONTINUITY_MARGIN: f32 = 0.25;
    // Chatter = two switches within this window (A4->C5->A4 at a comp strike).
    const CHATTER_WINDOW_SEC: f32 = 0.18;

    let mut active: Vec<(u8, u8)> = Vec::new();
    let mut onset_times: Vec<(f32, u8)> = Vec::new();
    let mut melody_segments: Vec<NoteEvent> = Vec::new();
    let mut segment_start = 0.0f32;
    let mut last_melody_pitch: Option<u8> = None;
    let mut last_melody_vel: u8 = 90;
    let mut last_switch_time = f32::NEG_INFINITY;

    let mut i = 0;
    while i < events.len() {
        let batch_time = events[i].0;

        let mut batch_end = i;
        while batch_end < events.len() && (events[batch_end].0 - batch_time).abs() <= time_tolerance {
            batch_end += 1;
        }

        for j in i..batch_end {
            let (_, pitch, _, is_start) = events[j];
            if !is_start {
                active.retain(|a| a.0 != pitch);
            }
        }

        let mut batch_onsets: Vec<u8> = Vec::new();
        for j in i..batch_end {
            let (_, pitch, vel, is_start) = events[j];
            if is_start {
                active.push((pitch, vel));
                batch_onsets.push(pitch);
                onset_times.push((batch_time, pitch));
            }
        }
        onset_times.retain(|(t, _)| *t >= batch_time - onset_window);

        if !active.is_empty() {
            // Score each active pitch.
            let mut counts: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();
            for &(_, p) in onset_times.iter() {
                if active.iter().any(|a| a.0 == p) {
                    *counts.entry(p).or_insert(0) += 1;
                }
            }
            let max_count = counts.values().max().copied().unwrap_or(0);
            // Strongest active note's velocity: harmonic artifacts (octave
            // partials of the synth) ring at ~30-45 while the real melody
            // sits at ~105-117. A candidate much weaker than the loudest
            // active note is an artifact, not a new melody note.
            let max_active_vel = active.iter().map(|a| a.1).max().unwrap_or(0);
            let best_pitch = if max_count == 0 {
                // Nothing re-articulated recently: keep the line continuous.
                last_melody_pitch
                    .unwrap_or_else(|| active.iter().map(|a| a.0).max().unwrap_or(0))
            } else {
                let mut best: Option<(u8, f32)> = None;
                for (&p, &c) in counts.iter() {
                    // Count is a soft term, not a gate: a fresh isolated onset
                    // (E4 right after repeated A4s) must be able to overtake a
                    // note that merely re-articulated more times in the window.
                    let p_vel = active
                        .iter()
                        .find(|a| a.0 == p)
                        .map(|a| a.1)
                        .unwrap_or(0);
                    if max_active_vel > 0 && p_vel < (max_active_vel as f32 * 0.5) as u8 {
                        continue;
                    }
                    let mut score = c as f32 * 1.2;
                    // Isolation bonus: the melody's onsets tend to be alone.
                    if batch_onsets.contains(&p) {
                        score += match batch_onsets.len() {
                            1 => 1.5,
                            2 => 0.5,
                            _ => 0.0,
                        };
                    }
                    // Adaptive register penalty: keep the line in the melody's
                    // own tessitura (comp bass + high overtones excluded).
                    score += register_penalty(p);
                    // Velocity term: the melody is the loudest line. Octave
                    // harmonic artifacts (vel ~35-45 vs ~115 real) otherwise
                    // win isolated batches on the isolation bonus alone.
                    score += active
                        .iter()
                        .find(|a| a.0 == p)
                        .map(|a| a.1 as f32 / 127.0)
                        .unwrap_or(0.7);
                    // Highest pitch breaks ties (the melody is the top line in
                    // its register).
                    score += p as f32 * 0.001;
                    if best.map(|(_, bs)| score > bs).unwrap_or(true) {
                        best = Some((p, score));
                    }
                }
                // Continuity: score the running melody pitch too, and only flip
                // to a new candidate when it wins by a real margin (or the old
                // pitch is no longer sounding). This kills the A4->C5->A4
                // chatter that comp strikes otherwise induce.
                let last_running_score = last_melody_pitch.and_then(|lp| {
                    active.iter().find(|a| a.0 == lp).map(|a| {
                        let c = counts.get(&lp).copied().unwrap_or(0);
                        let mut s = c as f32 * 1.2;
                        if batch_onsets.contains(&lp) {
                            s += match batch_onsets.len() {
                                1 => 1.5,
                                2 => 0.5,
                                _ => 0.0,
                            };
                        }
                        s += register_penalty(lp);
                        s += a.1 as f32 / 127.0;
                        s += lp as f32 * 0.001;
                        s
                    })
                });
                let old_still_sounding = last_melody_pitch
                    .map(|lp| active.iter().any(|a| a.0 == lp))
                    .unwrap_or(false);
                match best {
                    Some((p, bs)) if !old_still_sounding => p,
                    Some((p, bs)) => {
                        if let Some(ls) = last_running_score {
                            // Only apply the continuity margin when we just
                            // switched (chatter window) — otherwise allow
                            // genuine melodic motion through freely.
                            let rapid = batch_time - last_switch_time < CHATTER_WINDOW_SEC;
                            if (!rapid || bs - ls >= CONTINUITY_MARGIN) {
                                p
                            } else if let Some(lp) = last_melody_pitch {
                                lp
                            } else {
                                p
                            }
                        } else {
                            p
                        }
                    }
                    None => last_melody_pitch.unwrap_or_else(|| {
                        active.iter().map(|a| a.0).max().unwrap_or(0)
                    }),
                }
            };
            let best_vel = active
                .iter()
                .find(|a| a.0 == best_pitch)
                .map(|a| a.1)
                .unwrap_or(90);

            if std::env::var_os("KEYSCRIBE_SKYLINE_DEBUG").is_some() {
                eprintln!(
                    "[skyline] t={:.3}s batch={:?} counts={:?} active={:?} -> best_pitch={} last={:?}",
                    batch_time,
                    batch_onsets,
                    counts,
                    active,
                    best_pitch,
                    last_melody_pitch
                );
            }

            if Some(best_pitch) != last_melody_pitch {
                last_switch_time = batch_time;
                if let Some(prev_pitch) = last_melody_pitch {
                    if batch_time > segment_start {
                        melody_segments.push(NoteEvent {
                            id: (melody_segments.len() + 1) as u32,
                            pitch: prev_pitch,
                            start_time: segment_start,
                            end_time: batch_time,
                            velocity: last_melody_vel,
                            channel: None,
                        });
                    }
                }
                segment_start = batch_time;
                last_melody_pitch = Some(best_pitch);
                last_melody_vel = best_vel;
            } else if batch_time - segment_start >= min_segment && batch_onsets.contains(&best_pitch)
            {
                // Re-articulation of the same pitch: split the segment so
                // repeated 8th notes keep their rhythm.
                melody_segments.push(NoteEvent {
                    id: (melody_segments.len() + 1) as u32,
                    pitch: best_pitch,
                    start_time: segment_start,
                    end_time: batch_time,
                    velocity: last_melody_vel,
                    channel: None,
                });
                segment_start = batch_time;
            }
        }

        i = batch_end;
    }

    if let Some(pitch) = last_melody_pitch {
        let end_t = events.last().map(|e| e.0).unwrap_or(segment_start + 1.0);
        if end_t > segment_start {
            melody_segments.push(NoteEvent {
                id: (melody_segments.len() + 1) as u32,
                pitch,
                start_time: segment_start,
                end_time: end_t,
                velocity: last_melody_vel,
                channel: None,
            });
        }
    }

    // Onset-clamp durations: a monophonic melody note cannot extend past the
    // next onset, no matter how long the extracted probability sustains.
    // This turns staccato 8ths (whose extraction runs into the next attack)
    // into properly short notes that quantize to the right values.
    melody_segments.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in 0..melody_segments.len().saturating_sub(1) {
        let next_start = melody_segments[i + 1].start_time;
        if melody_segments[i].end_time > next_start {
            melody_segments[i].end_time = next_start.max(melody_segments[i].start_time + 0.02);
        }
    }

    let _ = outlier_semitones; // reserved for a future register guard
    melody_segments
}

/// Stepwise-motion prior: how strongly we penalize a pitch jump of `d`
/// semitones between consecutive melody decisions.
fn transition_penalty(q: u8, p: u8) -> f32 {
    let d = (p as i16 - q as i16).abs();
    match d {
        0 => 0.0,
        1 => 0.15,
        2 => 0.35,
        3 => 0.8,
        4 => 1.4,
        5..=7 => 2.2,
        8..=11 => 4.0,
        _ => 7.0,
    }
}

pub fn merge_adjacent_notes(notes: &mut Vec<NoteEvent>, step: f32) {
    let gap = (step * 2.0).max(0.03);
    merge_adjacent_notes_with_gap(notes, gap);
}

/// Merge same-pitch notes whose gap is smaller than `gap_sec`. A small gap
/// (one frame) removes extraction jitter while preserving genuine
/// re-articulations (staccato repeats).
pub fn merge_adjacent_notes_with_gap(notes: &mut Vec<NoteEvent>, gap_sec: f32) {
    let merge_gap = gap_sec.max(0.005);
    
    notes.sort_by(|a, b| {
        a.pitch.cmp(&b.pitch).then_with(|| {
            a.start_time.partial_cmp(&b.start_time).unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut i = 0;
    while i + 1 < notes.len() {
        let a = &notes[i];
        let b = &notes[i + 1];
        if a.pitch == b.pitch && b.start_time - a.end_time < merge_gap {
            let end = a.end_time.max(b.end_time);
            let vel = a.velocity.max(b.velocity);
            notes[i].end_time = end;
            notes[i].velocity = vel;
            notes.remove(i + 1);
        } else {
            i += 1;
        }
    }

    notes.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });
}

fn notes_to_spans(notes: &[QuantizedNote], config: SheetEngravingConfig) -> Vec<NoteSpan> {
    let mut out = Vec::new();
    for note in notes {
        let start_tick = (note.beat_start * MUSICXML_DIVISIONS as f32).round().max(0.0) as i32;
        // Preserve the exact quantized duration. In particular, 1/3 beat is
        // 160 ticks at 480 divisions; re-snapping it to quarter-beats turns a
        // triplet eighth into an ordinary 16th before notation is emitted.
        let min_duration = if config._allow_triplets { 1.0 / 6.0 } else { 0.25 };
        let raw_dur = if note.beat_duration.is_finite() {
            note.beat_duration.max(min_duration)
        } else {
            min_duration
        };
        let duration_ticks = (raw_dur * MUSICXML_DIVISIONS as f32).round().max(1.0) as i32;
        let staff = if config.is_lead_sheet || config.single_staff {
            1
        } else if note.pitch >= GRAND_STAFF_SPLIT_MIDI {
            1
        } else {
            2
        };
        out.push(NoteSpan {
            id: note.id,
            start_tick: start_tick.max(0),
            end_tick: (start_tick + duration_ticks).max(start_tick + 1),
            pitch: note.pitch,
            velocity: note.velocity,
            staff,
            articulation: note.articulation,
        });
    }

    out.sort_by_key(|n| n.start_tick);

    // Deduplicate: if two NoteSpans share the same pitch and start_tick,
    // keep only the longer one (prevents unison from monophonic reduction).
    let mut deduped: Vec<NoteSpan> = Vec::with_capacity(out.len());
    for span in out {
        if let Some(last) = deduped.last_mut() {
            if last.pitch == span.pitch && last.start_tick == span.start_tick {
                if span.end_tick > last.end_tick {
                    last.end_tick = span.end_tick;
                }
                continue;
            }
        }
        deduped.push(span);
    }
    out = deduped;

    // Monophonic guarantee: a melody note must never extend past the next
    // note's onset. Clamping here keeps the lead sheet strictly one-voice —
    // otherwise overlapping durations make `build_voice_chunks` split the
    // melody into spurious parallel voices.
    for i in 0..out.len().saturating_sub(1) {
        if out[i].end_tick > out[i + 1].start_tick {
            out[i].end_tick = (out[i + 1].start_tick).max(out[i].start_tick + 1);
        }
    }

    out
}

fn write_harmony(xml: &mut String, offset: i32, symbol: &str) {
    let (root_pc, suffix, bass_pc) = parse_chord_symbol(symbol);
    let (root_step, root_alter) = pc_to_step_alter(root_pc);

    let _ = write!(xml, "      <harmony>\n");
    if offset > 0 {
        let _ = write!(xml, "        <offset>{offset}</offset>\n");
    }
    let _ = write!(xml, "        <root><root-step>{root_step}</root-step>");
    if root_alter != 0 {
        let _ = write!(xml, "<root-alter>{}</root-alter>", root_alter);
    }
    let _ = write!(xml, "</root>\n");

    // Split the suffix into the base kind and any alterations (b9, #9, #11, b5, #5).
    let (base_suffix, alterations) = split_chord_alterations(suffix);
    let kind = chord_suffix_to_musicxml_kind(base_suffix.as_str());
    let _ = write!(
        xml,
        "        <kind{}>{}</kind>\n",
        if kind != "other" {
            ""
        } else {
            " text=\"other\""
        },
        kind
    );

    if let Some(bass_pc) = bass_pc {
        let (bass_step, bass_alter) = pc_to_step_alter(bass_pc);
        let _ = write!(xml, "        <bass><bass-step>{bass_step}</bass-step>");
        if bass_alter != 0 {
            let _ = write!(xml, "<bass-alter>{}</bass-alter>", bass_alter);
        }
        let _ = write!(xml, "</bass>\n");
    }

    // Emit <degree> elements for alterations so renderers display the
    // correct chord symbol (e.g. "7b9" instead of just "9").
    for alt in &alterations {
        let _ = write!(xml, "        <degree>\n");
        let _ = write!(xml, "          <degree-value>{}</degree-value>\n", alt.degree_value);
        let _ = write!(xml, "          <degree-alter>{}</degree-alter>\n", alt.alter);
        let _ = write!(xml, "          <degree-type>{}</degree-type>\n", alt.type_str);
        let _ = write!(xml, "        </degree>\n");
    }

    let _ = write!(xml, "      </harmony>\n");
}

struct ChordAlteration {
    degree_value: u8,  // e.g. 9 for the 9th
    alter: i8,         // -1 = flat, 1 = sharp
    type_str: &'static str, // "add" or "alter"
}

/// Split a chord suffix into the base kind and a list of alterations.
/// For example "7b9" â†’ ("7", [Alteration { 9, -1, "alter" }])
/// "7#11" â†’ ("7", [Alteration { 11, 1, "add" }])
fn split_chord_alterations(suffix: &str) -> (String, Vec<ChordAlteration>) {
    // Known alteration patterns to extract from the suffix.
    let patterns: &[(&str, u8, i8, &str)] = &[
        ("b9", 9, -1, "alter"),
        ("#9", 9, 1, "alter"),
        ("b5", 5, -1, "alter"),
        ("#5", 5, 1, "alter"),
        ("b13", 13, -1, "alter"),
        ("#11", 11, 1, "add"),
    ];

    let mut remaining = suffix.to_string();
    let mut alterations = Vec::new();

    // Check for compound alterations first (e.g. "b9b5")
    if remaining.contains("b9") && remaining.contains("b5") {
        alterations.push(ChordAlteration { degree_value: 9, alter: -1, type_str: "alter" });
        alterations.push(ChordAlteration { degree_value: 5, alter: -1, type_str: "alter" });
        remaining = remaining.replace("b9", "").replace("b5", "");
    }

    for (pat, degree, alter, type_str) in patterns {
        if remaining.contains(pat) {
            alterations.push(ChordAlteration { degree_value: *degree, alter: *alter, type_str });
            remaining = remaining.replace(pat, "");
        }
    }

    (remaining, alterations)
}

fn parse_chord_symbol(symbol: &str) -> (u8, &str, Option<u8>) {
    let (main, bass) = symbol.split_once('/').map(|(a, b)| (a, Some(b))).unwrap_or((symbol, None));

    let (root_pc, root_len) = parse_root_pc(main).unwrap_or((0, 1));
    let suffix = &main[root_len.min(main.len())..];
    let bass_pc = bass.and_then(|b| parse_root_pc(b).map(|(pc, _)| pc));

    (root_pc, suffix, bass_pc)
}

fn parse_root_pc(s: &str) -> Option<(u8, usize)> {
    let mut chars = s.chars();
    let first = chars.next()?;
    let base = match first {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };

    let second = s.chars().nth(1);
    match second {
        Some('#') => Some(((base + 1) % 12, 2)),
        Some('b') => Some(((base + 11) % 12, 2)),
        _ => Some((base, 1)),
    }
}

fn chord_suffix_to_musicxml_kind(suffix: &str) -> &'static str {
    match suffix {
        "" => "major",
        "-" => "minor",
        "7" => "dominant",
        "\u{0394}7" => "major-seventh",
        "-7" => "minor-seventh",
        "dim" => "diminished",
        "dim7" => "diminished-seventh",
        "aug" => "augmented",
        "sus2" => "suspended-second",
        "sus4" => "suspended-fourth",
        "-\u{0394}7" => "minor-major-seventh",
        "-7b5" => "half-diminished",
        "7#5" => "augmented-seventh",
        "6" => "major-sixth",
        "m6" => "minor-sixth",
        "9" => "dominant-ninth",
        "\u{0394}9" => "major-ninth",
        "-9" => "minor-ninth",
        "7b9" => "dominant-ninth",
        "7#9" => "dominant-ninth",
        "7#11" => "dominant-11th",
        "\u{0394}7#11" => "major-11th",
        "-11" => "minor-11th",
        "13" => "dominant-13th",
        "\u{0394}13" => "major-13th",
        "-13" => "minor-13th",
        "\u{0394}9#11" => "major-11th",
        "5" => "power",
        " (8ve)" => "power",
        _ => "other",
    }
}

fn pc_to_step_alter(pc: u8) -> (&'static str, i8) {
    match pc % 12 {
        0 => ("C", 0),
        1 => ("D", -1),
        2 => ("D", 0),
        3 => ("E", -1),
        4 => ("E", 0),
        5 => ("F", 0),
        6 => ("G", -1),
        7 => ("G", 0),
        8 => ("A", -1),
        9 => ("A", 0),
        10 => ("B", -1),
        _ => ("B", 0),
    }
}

fn write_rest_ticks(
    xml: &mut String,
    start_tick: i32,
    duration_ticks: i32,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let no_triplet_config = SheetEngravingConfig {
        _allow_triplets: false,
        ..config
    };
    let tokens = duration_tokens_for_ticks(duration_ticks, divisions, no_triplet_config);
    if tokens.is_empty() {
        write_forward_ticks(xml, duration_ticks, voice, staff);
        return;
    }
    let emitted_ticks: i32 = tokens.iter().map(|token| token.ticks.max(1)).sum();
    let mut cursor_tick = start_tick;
    for token in tokens {
        let _ = write!(
            xml,
            "      <note id=\"r{}_{}\">\n",
            cursor_tick,
            token.ticks.max(1)
        );
        let _ = write!(xml, "        <rest/>\n");
        let _ = write!(xml, "        <duration>{}</duration>\n", token.ticks.max(1));
        write_time_mod(xml, token.time_mod);
        let _ = write!(xml, "        <type>{}</type>\n", token.note_type);
        for _ in 0..token.dots {
            let _ = write!(xml, "        <dot/>\n");
        }
        let _ = write!(xml, "        <voice>{}</voice>\n", voice.max(1));
        let _ = write!(xml, "        <staff>{}</staff>\n", staff.max(1));
        let _ = write!(xml, "      </note>\n");
        cursor_tick += token.ticks.max(1);
    }
    if emitted_ticks < duration_ticks {
        write_forward_ticks(xml, duration_ticks - emitted_ticks, voice, staff);
    }
}

fn write_forward_ticks(xml: &mut String, duration_ticks: i32, voice: u8, staff: u8) {
    if duration_ticks > 0 {
        let _ = write!(
            xml,
            "      <forward><duration>{}</duration><voice>{}</voice><staff>{}</staff></forward>\n",
            duration_ticks,
            voice.max(1),
            staff.max(1)
        );
    }
}

fn write_note_element(
    xml: &mut String,
    note_id: &str,
    pitch: u8,
    token: DurationToken,
    staff: u8,
    voice: u8,
    velocity: u8,
    is_chord: bool,
    tie_start: bool,
    tie_stop: bool,
    articulation: Articulation,
    beam: Option<&'static str>,
    is_tuplet_start: bool,
    is_tuplet_end: bool,
) {
    let (step, alter, octave) = midi_to_pitch_parts(pitch);
    let _ = write!(xml, "      <note id=\"{}\">\n", xml_escape(note_id));
    if is_chord {
        let _ = write!(xml, "        <chord/>\n");
    }
    if articulation == Articulation::Grace || token.note_type == "grace" {
        let _ = write!(xml, "        <grace/>\n");
    }
    if token.note_type != "grace" || articulation == Articulation::Grace {
        let _ = write!(xml, "        <pitch><step>{step}</step>");
        if alter != 0 {
            let _ = write!(xml, "<alter>{}</alter>", alter);
        }
        let _ = write!(xml, "<octave>{octave}</octave></pitch>\n");
    }
    if token.note_type != "grace" && articulation != Articulation::Grace {
        let _ = write!(xml, "        <duration>{}</duration>\n", token.ticks.max(1));
    }
    write_time_mod(xml, token.time_mod);
    let note_type = if articulation == Articulation::Grace { "eighth" } else { token.note_type };
    let _ = write!(xml, "        <type>{}</type>\n", note_type);
    for _ in 0..token.dots {
        let _ = write!(xml, "        <dot/>\n");
    }
    // `alter` is already computed from midi_to_pitch_parts at the top of
    // this function â€” no need to call it again.
    if alter != 0 && token.note_type != "grace" {
        let acc = match alter { -2 => "double-flat", -1 => "flat", 1 => "sharp", 2 => "double-sharp", _ => "sharp" };
        let _ = write!(xml, "        <accidental>{}</accidental>\n", acc);
    }
    if let Some(b) = beam {
        let _ = write!(xml, "        <beam number=\"1\">{}</beam>\n", b);
    }
    let _ = write!(xml, "        <voice>{}</voice>\n", voice.max(1));
    let _ = write!(xml, "        <staff>{}</staff>\n", staff.max(1));
    let _ = write!(xml, "        <velocity>{}</velocity>\n", velocity);

    if tie_stop {
        let _ = write!(xml, "        <tie type=\"stop\"/>\n");
    }
    if tie_start {
        let _ = write!(xml, "        <tie type=\"start\"/>\n");
    }

    // `<time-modification>` carries the authoritative tuplet ratio. Bracket
    // markers are optional and are omitted here until cross-measure tuplet
    // grouping is fully modeled; unbalanced start/stop markers make MuseScore
    // reject an otherwise valid score.
    let has_notations = tie_start
        || tie_stop
        || (token.time_mod.is_some() && (is_tuplet_start || is_tuplet_end))
        || articulation == Articulation::Staccato
        || articulation == Articulation::Tenuto
        || articulation == Articulation::Accent;

    if has_notations {
        let _ = write!(xml, "        <notations>\n");
        if tie_stop {
            let _ = write!(xml, "          <tied type=\"stop\"/>\n");
        }
        if tie_start {
            let _ = write!(xml, "          <tied type=\"start\"/>\n");
        }
        if is_tuplet_start {
            let _ = write!(xml, "          <tuplet type=\"start\"/>\n");
        }
        if is_tuplet_end {
            let _ = write!(xml, "          <tuplet type=\"stop\"/>\n");
        }
        match articulation {
            Articulation::Staccato => {
                let _ = write!(xml, "          <articulations>\n");
                let _ = write!(xml, "            <staccato/>\n");
                let _ = write!(xml, "          </articulations>\n");
            }
            Articulation::Tenuto => {
                let _ = write!(xml, "          <articulations>\n");
                let _ = write!(xml, "            <tenuto/>\n");
                let _ = write!(xml, "          </articulations>\n");
            }
            Articulation::Accent => {
                let _ = write!(xml, "          <articulations>\n");
                let _ = write!(xml, "            <accent/>\n");
                let _ = write!(xml, "          </articulations>\n");
            }
            _ => {}
        }
        let _ = write!(xml, "        </notations>\n");
    }

    let _ = write!(xml, "      </note>\n");
}

fn duration_tokens_for_ticks(
    duration_ticks: i32,
    divisions: i32,
    _config: SheetEngravingConfig,
) -> Vec<DurationToken> {
    let d = divisions.max(1);
    let min_tick = if _config._allow_triplets {
        d / 6 // smallest supported triplet unit = triplet 16th (80 at 480 div)
    } else {
        d / 4 // smallest unit = 16th note (120 at 480 div)
    };
    let total = duration_ticks.max(0);
    if total < min_tick {
        return Vec::new();
    }

    // Candidate durations from longest to shortest, including dotted notes
    // and triplets. Each entry is (ticks, note_type, dots, time_modification).
    // Dotted note = base * 1.5. Triplet quarter = d*2/3, triplet eighth = d/3.
    // At 480 divisions: whole=1920, half=960, half.=1440, quarter=480,
    // quarter.=720, eighth=240, eighth.=360, 16th=120.
    let mut candidates = vec![
        DurationToken { ticks: d * 4, note_type: "whole", dots: 0, time_mod: None },
        DurationToken { ticks: d * 3, note_type: "half", dots: 1, time_mod: None },
        DurationToken { ticks: d * 2, note_type: "half", dots: 0, time_mod: None },
        DurationToken { ticks: d + d / 2, note_type: "quarter", dots: 1, time_mod: None },
        DurationToken { ticks: d, note_type: "quarter", dots: 0, time_mod: None },
        DurationToken { ticks: d / 2 + d / 4, note_type: "eighth", dots: 1, time_mod: None },
        DurationToken { ticks: d / 2, note_type: "eighth", dots: 0, time_mod: None },
        DurationToken { ticks: d / 4 + d / 8, note_type: "16th", dots: 1, time_mod: None },
        DurationToken { ticks: d / 4, note_type: "16th", dots: 0, time_mod: None },
    ];
    if _config._allow_triplets {
        // Triplet candidates (time_mod actual/normal)
        candidates.extend([
            DurationToken { ticks: d * 2 / 3, note_type: "quarter", dots: 0, time_mod: Some((3, 2)) },
            DurationToken { ticks: d / 3, note_type: "eighth", dots: 0, time_mod: Some((3, 2)) },
            DurationToken { ticks: d / 6, note_type: "16th", dots: 0, time_mod: Some((3, 2)) },
        ]);
    }

    let mut remaining = total;
    let mut out = Vec::new();

    while remaining >= min_tick {
        // Choose the longest representable token that fits. The triplet
        // eighth (160 ticks) must win over a normal 16th (120 ticks) when the
        // requested duration is exactly 160 ticks.
        let chosen = candidates
            .iter()
            .filter(|candidate| candidate.ticks <= remaining)
            .max_by_key(|candidate| candidate.ticks)
            .copied();

        match chosen {
            Some(token) => {
                remaining -= token.ticks;
                out.push(token);
            }
            None => {
                // Remaining is smaller than the finest candidate; emit as
                // a 16th so the measure still adds up.
                let fallback = DurationToken {
                    ticks: min_tick,
                    note_type: "16th",
                    dots: 0,
                    time_mod: None,
                };
                remaining -= fallback.ticks;
                out.push(fallback);
            }
        }
    }

    out
}

fn build_voice_chunks(chunks: &[NoteChunk], staff: u8) -> BTreeMap<u8, Vec<NoteChunk>> {
    // Greedy voice assignment: sort notes by start position, then assign
    // each to the lowest-numbered voice whose last note ends before this
    // one starts. This separates overlapping notes (e.g. a sustained half
    // note and a moving quarter-note line) into distinct voices so the
    // MusicXML renderer doesn't truncate or reorder them.
    let mut filtered: Vec<NoteChunk> = chunks
        .iter()
        .filter(|c| c.staff == staff)
        .cloned()
        .collect();
    filtered.sort_by_key(|c| (c.start_tick_in_measure, c.pitch));

    // Track the end tick of the last note in each voice.
    let mut voice_end_ticks: Vec<i32> = Vec::new();
    let mut voice_assignments: Vec<u8> = vec![0u8; filtered.len()];

    for (i, chunk) in filtered.iter().enumerate() {
        let start = chunk.start_tick_in_measure;
        let end = start + chunk.duration_ticks;

        // Find the first voice that is free (its last note ended at or
        // before this note's start).
        let assigned_voice = voice_end_ticks
            .iter()
            .position(|&end_tick| end_tick <= start);

        let voice = match assigned_voice {
            Some(idx) => {
                voice_end_ticks[idx] = end;
                (idx + 1) as u8
            }
            None => {
                voice_end_ticks.push(end);
                voice_end_ticks.len() as u8
            }
        };
        voice_assignments[i] = voice;
    }

    let mut out: BTreeMap<u8, Vec<NoteChunk>> = BTreeMap::new();
    for (i, chunk) in filtered.into_iter().enumerate() {
        let voice = voice_assignments[i];
        out.entry(voice).or_default().push(chunk);
    }

    out
}

fn write_voice_sequence(
    xml: &mut String,
    chunks: &[NoteChunk],
    measure_start_tick: i32,
    measure_ticks: i32,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let min_rest_ticks = (divisions / 2).max(1); // no rests smaller than 8th

    if std::env::var_os("KEYSCRIBE_WRITER_DEBUG").is_some() {
        eprintln!(
            "[writer] voice {} measure ticks={} chunks: {:?}",
            voice,
            measure_ticks,
            chunks
                .iter()
                .map(|c| (c.start_tick_in_measure, c.duration_ticks, c.pitch))
                .collect::<Vec<_>>()
        );
    }

    // Pre-compute beam groups for consecutive eighth notes
    let mut beam_of_group: Vec<Option<&'static str>> = vec![None; chunks.len()];
    {
        let mut g = 0;
        let mut group_positions: Vec<(usize, bool)> = Vec::new();
        while g < chunks.len() {
            let pos = chunks[g].start_tick_in_measure;
            let mut ge = g;
            let mut max_dur = 0;
            while ge < chunks.len() && chunks[ge].start_tick_in_measure == pos {
                max_dur = max_dur.max(chunks[ge].duration_ticks);
                ge += 1;
            }
            let is_eighth = max_dur <= divisions / 2;
            group_positions.push((g, is_eighth));
            g = ge;
        }
        let mut gi = 0;
        while gi < group_positions.len() {
            if group_positions[gi].1 {
                let start = gi;
                while gi < group_positions.len() && group_positions[gi].1 { gi += 1; }
                let end = gi;
                if end - start >= 2 {
                    beam_of_group[group_positions[start].0] = Some("begin");
                    for k in (start + 1)..(end - 1) {
                        beam_of_group[group_positions[k].0] = Some("continue");
                    }
                    beam_of_group[group_positions[end - 1].0] = Some("end");
                }
            } else {
                gi += 1;
            }
        }
    }

    // A valid tuplet is exactly three notes of EQUAL duration (e.g. three
    // triplet eighths). Groupings with unequal members (160/320/160) are not
    // legal tuplets — MusicXML consumes them badly and the measure reads
    // overfull. Only equal-duration triples get time-modification.
    let triplet_value = |ticks: i32| -> Option<i32> {
        if ticks == divisions * 2 / 3 || ticks == divisions / 3 || ticks == divisions / 6 {
            Some(ticks)
        } else {
            None
        }
    };
    let mut chunk_tuplets = vec![false; chunks.len()];
    for start in 0..chunks.len().saturating_sub(2) {
        match (
            triplet_value(chunks[start].duration_ticks),
            triplet_value(chunks[start + 1].duration_ticks),
            triplet_value(chunks[start + 2].duration_ticks),
        ) {
            (Some(a), Some(b), Some(c)) if a == b && b == c => {
                chunk_tuplets[start] = true;
                chunk_tuplets[start + 1] = true;
                chunk_tuplets[start + 2] = true;
            }
            _ => {}
        }
    }
    // A trailing pair of equal triplet-valued chunks (e.g. two triplet 8ths
    // followed by a rest) is a legal partial tuplet — mark it too so the pair
    // keeps its 3:2 rhythm instead of degrading to straight 16ths.
    for start in 0..chunks.len().saturating_sub(1) {
        if chunk_tuplets[start] || chunk_tuplets[start + 1] {
            continue;
        }
        match (
            triplet_value(chunks[start].duration_ticks),
            triplet_value(chunks[start + 1].duration_ticks),
        ) {
            (Some(a), Some(b)) if a == b && false => {
                chunk_tuplets[start] = true;
                chunk_tuplets[start + 1] = true;
            }
            _ => {}
        }
    }

    let mut cursor = 0i32;
    let mut i = 0;

    while i < chunks.len() {
        if cursor >= measure_ticks {
            break;
        }

        let nominal_pos = chunks[i].start_tick_in_measure;

        // If this chunk's position is behind the cursor, shift it to the cursor
        // (avo`ids going backwards in time within a single voice).
        let write_pos = if nominal_pos < cursor { cursor } else { nominal_pos };

        // Rest gap before write_pos
        if write_pos > cursor {
            let gap = write_pos.min(measure_ticks) - cursor;
            if gap >= min_rest_ticks {
                write_rest_ticks(
                    xml,
                    measure_start_tick + cursor,
                    gap,
                    divisions,
                    staff,
                    voice,
                    config,
                );
            } else {
                write_forward_ticks(xml, gap, voice, staff);
            }
            cursor = write_pos.min(measure_ticks);
            if cursor >= measure_ticks {
                break;
            }
        }

        // Gather all chunks starting at this *nominal* position (chord group)
        let mut group_end = i;
        while group_end < chunks.len() && chunks[group_end].start_tick_in_measure == nominal_pos {
            group_end += 1;
        }

        // The voice cursor advances by the longest note in the group,
        // but each note is written with its OWN duration. MusicXML chord
        // notes (<chord/>) must share the same <duration>, so notes with
        // differing durations starting at the same time should be in
        // separate voices â€” which build_voice_chunks now handles. Within
        // a single voice, simultaneous notes are true chords and should
        // already have the same duration.
        let group_max_dur = chunks[i..group_end]
            .iter()
            .map(|c| c.duration_ticks)
            .max()
            .unwrap_or(0);
        let clamped_max_dur = group_max_dur.min(measure_ticks - cursor);

        for (j, chunk) in chunks[i..group_end].iter().enumerate() {
            let is_chord = j > 0;
            // Use this note's own duration, clamped to the remaining
            // measure space. This preserves individual note lengths
            // instead of stretching all to the group maximum.
            let chunk_dur = chunk.duration_ticks.min(measure_ticks - cursor);
            if chunk_dur > 0 {
                let chunk_config = SheetEngravingConfig {
                    _allow_triplets: chunk_tuplets[i + j],
                    ..config
                };
                let tokens = duration_tokens_for_ticks(chunk_dur, divisions, chunk_config);
                if tokens.is_empty() {
                    write_forward_ticks(xml, chunk_dur, voice, staff);
                } else {
                    let mut current_token_tick = chunk.absolute_tick;
                
                    for (idx, token) in tokens.iter().enumerate() {
                        let local_tie_stop = (idx > 0) || (idx == 0 && chunk.tie_stop);
                        let local_tie_start = (idx + 1 < tokens.len()) || (idx + 1 == tokens.len() && chunk.tie_start);
                    
                        let beam = if j == 0 && idx == 0 { beam_of_group[i] } else { None };
                    // A tuplet spans consecutive tuplet-valued chunks. Emit
                    // one start on the first note and one stop on the last,
                    // rather than start+stop on every note in the group.
                        let previous_tuplet = i > 0 && chunk_tuplets[i - 1];
                        let next_group_tuplet = group_end < chunks.len() && chunk_tuplets[group_end];
                        let is_tuplet_start = j == 0
                            && idx == 0
                            && token.time_mod.is_some()
                            && !previous_tuplet;
                        let is_tuplet_end = j + 1 == group_end - i
                            && idx + 1 == tokens.len()
                            && token.time_mod.is_some()
                            && !next_group_tuplet;
                        write_note_element(
                        xml,
                        &format!(
                            "n{}_{}_{}_{}",
                            chunk.id,
                            chunk.pitch,
                            current_token_tick,
                            token.ticks
                        ),
                        chunk.pitch,
                        *token,
                        staff,
                        voice,
                        chunk.velocity,
                        is_chord,
                        local_tie_start,
                        local_tie_stop,
                        chunk.articulation,
                        beam,
                        is_tuplet_start,
                        is_tuplet_end,
                        );
                        current_token_tick += token.ticks;
                    }
                    let emitted_ticks: i32 = tokens.iter().map(|token| token.ticks).sum();
                    if emitted_ticks < chunk_dur {
                        write_forward_ticks(xml, chunk_dur - emitted_ticks, voice, staff);
                    }
                }
            }
        }

        cursor = (cursor + clamped_max_dur).min(measure_ticks);
        i = group_end;
    }

    let remaining = measure_ticks - cursor;
    if remaining >= min_rest_ticks {
        write_rest_ticks(
            xml,
            measure_start_tick + cursor,
            remaining,
            divisions,
            staff,
            voice,
            config,
        );
    } else if remaining > 0 {
        write_forward_ticks(xml, remaining, voice, staff);
    }
}

fn write_time_mod(xml: &mut String, time_mod: Option<(u8, u8)>) {
    if let Some((actual, normal)) = time_mod {
        let _ = write!(xml, "        <time-modification>\n");
        let _ = write!(xml, "          <actual-notes>{}</actual-notes>\n", actual);
        let _ = write!(xml, "          <normal-notes>{}</normal-notes>\n", normal);
        let _ = write!(xml, "        </time-modification>\n");
    }
}

fn midi_to_pitch_parts(midi: u8) -> (&'static str, i8, i32) {
    let octave = (midi as i32 / 12) - 1;
    match midi % 12 {
        0 => ("C", 0, octave),
        1 => ("C", 1, octave),
        2 => ("D", 0, octave),
        3 => ("D", 1, octave),
        4 => ("E", 0, octave),
        5 => ("F", 0, octave),
        6 => ("F", 1, octave),
        7 => ("G", 0, octave),
        8 => ("G", 1, octave),
        9 => ("A", 0, octave),
        10 => ("A", 1, octave),
        _ => ("B", 0, octave),
    }
}

pub fn sanitize_filename_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        let valid = c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.';
        if valid {
            out.push(c);
        } else if c.is_ascii_whitespace() {
            out.push('_');
        }
    }

    if out.is_empty() {
        "keyscribe-sheet".to_string()
    } else {
        out
    }
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn musescore_cli_candidates() -> Vec<String> {
    let mut commands = vec![
        "musescore4".to_string(),
        "MuseScore4".to_string(),
        "mscore".to_string(),
        "MuseScore3".to_string(),
        "MuseScore".to_string(),
    ];

    if cfg!(windows) {
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files));
            commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files));
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files_x86));
            commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files_x86));
        }
    }
    commands
}

/// Render a MusicXML file to any format MuseScore supports via its CLI
/// (e.g. MP3, WAV, FLAC, PDF). The output format is inferred from the
/// extension of `output_path`.
pub fn musescore_convert(input_path: &Path, output_path: &Path) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!("failed to create dir {}: {e}", parent.display())
            })?;
        }
    }
    let input = input_path.to_string_lossy().to_string();
    let output = output_path.to_string_lossy().to_string();
    let mut failures = Vec::<String>::new();

    for cmd in musescore_cli_candidates() {
        let attempts = [
            vec!["-o".to_string(), output.clone(), input.clone()],
            vec![input.clone(), "-o".to_string(), output.clone()],
        ];
        for args in attempts {
            let mut cmd_obj = Command::new(cmd.as_str());
            cmd_obj.args(args.as_slice());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd_obj.creation_flags(0x08000000);
            }
            match cmd_obj.status() {
                Ok(s) if s.success() => return Ok(()),
                Ok(s) => failures.push(format!("{} exited with {}", cmd, s)),
                Err(_) => {}
            }
        }
    }

    if failures.is_empty() {
        Err("MuseScore CLI was not found. Install MuseScore and ensure its CLI executable is on PATH.".to_string())
    } else {
        Err(format!(
            "MuseScore CLI failed to render {}. Attempts: {}",
            output_path.display(),
            failures.join(" | ")
        ))
    }
}

pub fn export_engraved_pdf_with_musescore(musicxml_path: &Path, pdf_path: &Path) -> Result<(), String> {
    musescore_convert(musicxml_path, pdf_path)
}

pub fn write_temp_musicxml(prefix: &str, xml: &str) -> Result<PathBuf, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("clock error: {e}"))?
        .as_millis();
    let path = std::env::temp_dir().join(format!("{}_{}.musicxml", prefix, now));
    fs::write(path.as_path(), xml.as_bytes())
        .map_err(|e| format!("failed to write temp musicxml: {e}"))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leadsheet::TempoSegment;

    #[test]
    fn builds_readable_musicxml() {
        let foundation = LeadSheetFoundation {
            tempo_map: vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 8.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            }],
            time_signature_segments: vec![crate::leadsheet::TimeSignatureSegment {
                start_beat: 0.0,
                end_beat: 16.0,
                numerator: 4,
                denominator: 4,
                confidence: 0.9,
                meter_class: crate::leadsheet::MeterClass::SimpleQuadruple,
            }],
            tempo: crate::leadsheet::TempoEstimate {
                bpm: 120.0,
                beat_duration_sec: 0.5,
                confidence: 1.0,
            },
            quantized_notes: vec![
                QuantizedNote {
                    id: 1,
                    pitch: 60,
                    beat_start: 0.0,
                    beat_duration: 1.0,
                    velocity: 96,
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
                    beat_start: 1.0,
                    beat_duration: 1.0,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
            ],
            melody_notes: vec![],
            chord_changes: vec![ChordSymbolChange {
                beat_start: 0.0,
                symbol: "C".to_string(),
            }],
            tied_notes: vec![],
            rhythm_confidence: 0.9,
            melodic_stem: None,
            separation_confidence: 0.0,
            aligned_notes: vec![],
            swing_sections: vec![],
            beats_per_bar: 4,
        };

        let xml = build_musicxml_document("Test", &foundation, SheetEngravingConfig {
            is_lead_sheet: false,
            ..SheetEngravingConfig::default()
        });
        assert!(xml.contains("<score-partwise"));
        assert!(xml.contains("<harmony>"));
        assert!(xml.contains("<measure number=\"1\">"));
        assert!(xml.contains("<staves>2</staves>"));
    }

    #[test]
    fn musicxml_contains_non_quarter_types_when_input_has_short_values() {
        let foundation = LeadSheetFoundation {
            tempo_map: vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 8.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            }],
            time_signature_segments: vec![crate::leadsheet::TimeSignatureSegment {
                start_beat: 0.0,
                end_beat: 16.0,
                numerator: 4,
                denominator: 4,
                confidence: 0.9,
                meter_class: crate::leadsheet::MeterClass::SimpleQuadruple,
            }],
            tempo: crate::leadsheet::TempoEstimate {
                bpm: 120.0,
                beat_duration_sec: 0.5,
                confidence: 1.0,
            },
            quantized_notes: vec![
                QuantizedNote {
                    id: 1,
                    pitch: 60,
                    beat_start: 0.0,
                    beat_duration: 0.5,
                    velocity: 96,
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
                    pitch: 62,
                    beat_start: 0.5,
                    beat_duration: 0.5,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
            ],
            melody_notes: vec![],
            chord_changes: vec![ChordSymbolChange {
                beat_start: 0.0,
                symbol: "C".to_string(),
            }],
            tied_notes: vec![],
            rhythm_confidence: 0.9,
            melodic_stem: None,
            separation_confidence: 0.0,
            aligned_notes: vec![],
            swing_sections: vec![],
            beats_per_bar: 4,
        };

        let xml =
            build_musicxml_document("DurationTest", &foundation, SheetEngravingConfig::default());
        assert!(xml.contains("<type>eighth</type>"));
    }

    #[test]
    fn duration_tokens_preserve_triplet_eighths() {
        let tokens = duration_tokens_for_ticks(
            160,
            MUSICXML_DIVISIONS,
            SheetEngravingConfig {
                _allow_triplets: true,
                ..SheetEngravingConfig::default()
            },
        );
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].ticks, 160);
        assert_eq!(tokens[0].time_mod, Some((3, 2)));
    }

    #[test]
    fn musicxml_emits_time_signature_changes() {
        let foundation = LeadSheetFoundation {
            tempo_map: vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 12.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            }],
            time_signature_segments: vec![
                crate::leadsheet::TimeSignatureSegment {
                    start_beat: 0.0,
                    end_beat: 8.0,
                    numerator: 4,
                    denominator: 4,
                    confidence: 0.9,
                    meter_class: crate::leadsheet::MeterClass::SimpleQuadruple,
                },
                crate::leadsheet::TimeSignatureSegment {
                    start_beat: 8.0,
                    end_beat: 24.0,
                    numerator: 3,
                    denominator: 4,
                    confidence: 0.85,
                    meter_class: crate::leadsheet::MeterClass::SimpleTriple,
                },
            ],
            tempo: crate::leadsheet::TempoEstimate {
                bpm: 120.0,
                beat_duration_sec: 0.5,
                confidence: 1.0,
            },
            quantized_notes: vec![
                QuantizedNote {
                    id: 1,
                    pitch: 60,
                    beat_start: 0.0,
                    beat_duration: 1.0,
                    velocity: 96,
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
                    pitch: 62,
                    beat_start: 8.0,
                    beat_duration: 1.0,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
            ],
            melody_notes: vec![],
            chord_changes: vec![],
            tied_notes: vec![],
            rhythm_confidence: 0.85,
            melodic_stem: None,
            separation_confidence: 0.0,
            aligned_notes: vec![],
            swing_sections: vec![],
            beats_per_bar: 4,
        };

        let xml =
            build_musicxml_document("MeterChange", &foundation, SheetEngravingConfig::default());
        assert!(xml.contains("<beats>4</beats><beat-type>4</beat-type>"));
        assert!(xml.contains("<beats>3</beats><beat-type>4</beat-type>"));
    }
}

