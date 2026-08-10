//! MIDI writing and parsing helpers shared between the desktop app and the CLI.
//! UI-independent: usable from the GUI exporter and from the headless test loop.

use std::collections::HashMap;
use std::path::Path;

use midly::{
    Header, Format, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind,
};

use crate::leadsheet::NoteEvent;

/// Write `notes` (times in seconds) to a single-track MIDI file.
///
/// A `SetTempo` meta message at the start makes the seconds-based note times
/// map onto the MIDI timeline at `bpm` (so rendering the file with MuseScore
/// produces audio with the intended tempo).
pub fn write_midi(notes: &[NoteEvent], path: &Path, bpm: f32) -> anyhow::Result<()> {
    const PPQ: u32 = 480;

    let mut smf = Smf::new(Header::new(Format::SingleTrack, Timing::Metrical((PPQ as u16).into())));
    let mut track = Track::new();

    let micros_per_qn = (60_000_000.0 / bpm.clamp(20.0, 400.0)).round() as u32;
    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(micros_per_qn.into())),
    });

    #[derive(Clone, Copy)]
    struct Evt {
        time_ticks: u32,
        is_note_on: bool,
        pitch: u8,
        velocity: u8,
    }

    let ticks_per_sec = PPQ as f32 * bpm.clamp(20.0, 400.0) / 60.0;
    let mut events = Vec::new();
    for note in notes {
        let start = (note.start_time.max(0.0) * ticks_per_sec) as u32;
        let mut end = (note.end_time.max(0.0) * ticks_per_sec) as u32;
        if end <= start {
            end = start + 1;
        }
        events.push(Evt {
            time_ticks: start,
            is_note_on: true,
            pitch: note.pitch,
            velocity: note.velocity.max(1).min(127),
        });
        events.push(Evt {
            time_ticks: end,
            is_note_on: false,
            pitch: note.pitch,
            velocity: 0,
        });
    }

    events.sort_by_key(|e| e.time_ticks);
    events.dedup_by_key(|e| (e.time_ticks, e.is_note_on, e.pitch));

    let mut last_tick = 0u32;
    for e in events {
        let delta = e.time_ticks.saturating_sub(last_tick);
        last_tick = e.time_ticks;
        let message = if e.is_note_on {
            TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOn {
                    key: e.pitch.into(),
                    vel: e.velocity.into(),
                },
            }
        } else {
            TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOff {
                    key: e.pitch.into(),
                    vel: e.velocity.into(),
                },
            }
        };
        let delta_u28 = midly::num::u28::try_from(delta).unwrap_or(midly::num::u28::max_value());
        track.push(TrackEvent {
            delta: delta_u28,
            kind: message,
        });
    }

    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    smf.tracks.push(track);

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("failed to create dir {}: {e}", parent.display()))?;
        }
    }
    smf.save(path)
        .map_err(|e| anyhow::anyhow!("failed to write MIDI {}: {e}", path.display()))?;
    Ok(())
}

/// A note parsed from a MIDI or MusicXML file, with onset/duration expressed in
/// beats so the two representations can be compared format-independently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidiNote {
    pub pitch: u8,
    pub onset_beats: f32,
    pub dur_beats: f32,
}

/// Parse all notes from a MIDI file, merging every track. Onsets/durations are
/// normalized to beats using the file's pulses-per-quarter-note.
pub fn parse_midi_notes(path: &Path) -> anyhow::Result<Vec<MidiNote>> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read MIDI {}: {e}", path.display()))?;
    let smf = Smf::parse(&data)
        .map_err(|e| anyhow::anyhow!("failed to parse MIDI {}: {e}", path.display()))?;

    let ppq = match smf.header.timing {
        Timing::Metrical(t) => t.as_int().max(1) as f32,
        _ => 480.0,
    };

    // Track pending note-ons keyed by (channel, pitch).
    let mut pending: HashMap<(u8, u8), u64> = HashMap::new();
    let mut notes: Vec<MidiNote> = Vec::new();

    for track in &smf.tracks {
        let mut tick = 0u64;
        pending.clear();
        for evt in track {
            tick += evt.delta.as_int() as u64;
            match evt.kind {
                TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOn { key, vel },
                } if vel.as_int() > 0 => {
                    pending.insert((channel.as_int(), key.as_int()), tick);
                }
                TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOn { key, vel: _ },
                } => {
                    if let Some(start) = pending.remove(&(channel.as_int(), key.as_int())) {
                        notes.push(MidiNote {
                            pitch: key.as_int(),
                            onset_beats: start as f32 / ppq,
                            dur_beats: (tick.saturating_sub(start)) as f32 / ppq,
                        });
                    }
                }
                TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOff { key, vel: _ },
                } => {
                    if let Some(start) = pending.remove(&(channel.as_int(), key.as_int())) {
                        notes.push(MidiNote {
                            pitch: key.as_int(),
                            onset_beats: start as f32 / ppq,
                            dur_beats: (tick.saturating_sub(start)) as f32 / ppq,
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
