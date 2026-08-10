//! Headless round-trip tests that need no audio or MuseScore:
//! MIDI write/parse fidelity, MusicXML note extraction, and compare metrics.

use keyscribe_lib::leadsheet::NoteEvent;
use keyscribe_lib::midi::{parse_midi_notes, write_midi};
use keyscribe_lib::sheet_compare::{compare_note_lists, parse_musicxml_notes};

fn temp_path(ext: &str) -> std::path::PathBuf {
    let name = format!(
        "keyscribe_test_{}_{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        ext
    );
    std::env::temp_dir().join(name)
}

#[test]
fn midi_write_parse_roundtrip_preserves_notes() {
    let notes = vec![
        NoteEvent { id: 0, pitch: 60, start_time: 0.0, end_time: 0.5, velocity: 100, channel: None },
        NoteEvent { id: 1, pitch: 62, start_time: 0.6, end_time: 1.2, velocity: 90, channel: None },
        NoteEvent { id: 2, pitch: 64, start_time: 1.2, end_time: 1.6, velocity: 80, channel: None },
        NoteEvent { id: 3, pitch: 67, start_time: 2.0, end_time: 2.5, velocity: 110, channel: None },
    ];

    let path = temp_path("mid");
    write_midi(&notes, &path, 120.0).expect("write_midi");
    let parsed = parse_midi_notes(&path).expect("parse_midi_notes");
    std::fs::remove_file(&path).ok();

    assert_eq!(parsed.len(), notes.len());

    // At 120 BPM, 1 beat = 0.5s; PPQ = 480 -> ticks/sec = 960.
    let expected_onsets: Vec<f32> = notes.iter().map(|n| n.start_time * 2.0).collect();
    for (got, want) in parsed.iter().zip(expected_onsets.iter()) {
        assert!((got.onset_beats - want).abs() < 1e-2, "onset {got:?} != {want}");
    }
    assert_eq!(parsed[0].pitch, 60);
    assert_eq!(parsed[3].pitch, 67);
}

#[test]
fn musicxml_note_extraction_reads_pitches_and_onsets() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part id="P1">
    <measure number="1">
      <attributes><divisions>480</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note id="a"><pitch><step>C</step><octave>4</octave></pitch><duration>480</duration><type>quarter</type></note>
      <note id="b"><pitch><step>E</step><alter>1</alter><octave>4</octave></pitch><duration>480</duration><type>quarter</type></note>
      <note id="c"><pitch><step>G</step><octave>4</octave></pitch><duration>960</duration><type>half</type></note>
      <note id="d"><rest/><duration>480</duration></note>
      <note id="e"><pitch><step>C</step><octave>5</octave></pitch><duration>480</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>"#;

    let notes = parse_musicxml_notes(xml).expect("parse_musicxml_notes");

    // C4, E#4, G4(half), rest skipped, C5.
    assert_eq!(notes.len(), 4);
    assert_eq!(notes[0].pitch, 60); // C4
    assert_eq!(notes[0].onset_beats, 0.0);
    assert_eq!(notes[0].dur_beats, 1.0);
    assert_eq!(notes[1].pitch, 65); // E#4 (alter +1)
    assert_eq!(notes[1].onset_beats, 1.0);
    assert_eq!(notes[2].pitch, 67); // G4
    assert_eq!(notes[2].onset_beats, 2.0);
    assert_eq!(notes[2].dur_beats, 2.0);
    assert_eq!(notes[3].pitch, 72); // C5 (after the rest)
    assert_eq!(notes[3].onset_beats, 5.0);
}

#[test]
fn musicxml_note_extraction_honors_voice_backups() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part id="P1">
    <measure number="1">
      <attributes><divisions>480</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>480</duration></note>
      <backup><duration>480</duration></backup>
      <note><pitch><step>G</step><octave>3</octave></pitch><duration>480</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>480</duration></note>
    </measure>
  </part>
</score-partwise>"#;

    let notes = parse_musicxml_notes(xml).expect("parse_musicxml_notes");

    assert_eq!(notes.len(), 3);
    assert_eq!(notes[0].pitch, 55);
    assert_eq!(notes[0].onset_beats, 0.0);
    assert_eq!(notes[1].pitch, 60);
    assert_eq!(notes[1].onset_beats, 0.0);
    assert_eq!(notes[2].pitch, 62);
    assert_eq!(notes[2].onset_beats, 1.0);
}

#[test]
fn compare_identical_sheets_scores_perfect() {
    let a = vec![
        keyscribe_lib::midi::MidiNote { pitch: 60, onset_beats: 0.0, dur_beats: 1.0 },
        keyscribe_lib::midi::MidiNote { pitch: 64, onset_beats: 1.0, dur_beats: 1.0 },
        keyscribe_lib::midi::MidiNote { pitch: 67, onset_beats: 2.0, dur_beats: 2.0 },
    ];
    let report = compare_note_lists(&a, &a, 0.25);
    assert_eq!(report.note_accuracy, 1.0);
    assert_eq!(report.recall, 1.0);
    assert_eq!(report.pitch_accuracy, 1.0);
}

#[test]
fn compare_counts_pitch_errors() {
    let reference = vec![
        keyscribe_lib::midi::MidiNote { pitch: 60, onset_beats: 0.0, dur_beats: 1.0 },
        keyscribe_lib::midi::MidiNote { pitch: 64, onset_beats: 1.0, dur_beats: 1.0 },
    ];
    // One correct note, one wrong pitch.
    let transcribed = vec![
        keyscribe_lib::midi::MidiNote { pitch: 60, onset_beats: 0.0, dur_beats: 1.0 },
        keyscribe_lib::midi::MidiNote { pitch: 69, onset_beats: 1.0, dur_beats: 1.0 },
    ];
    let report = compare_note_lists(&reference, &transcribed, 0.25);
    assert!((report.note_accuracy - 0.5).abs() < 1e-4);
    assert!((report.pitch_accuracy - 0.5).abs() < 1e-4);
}
