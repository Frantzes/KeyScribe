//! In-process MusicXML → audio synthesizer.
//!
//! Parses single-part MusicXML (the Charlie Parker Omnibook layout: one melody
//! voice plus `<harmony>` chord symbols and a `<metronome>` tempo) into melody
//! notes + chord progression + tempo, then renders melody-with-comping to
//! stereo WAV / MP3 without any external tool (no MuseScore, no soundfont).
//!
//! The generated audio is the round-trip training corpus: ground truth is the
//! source `.xml`, so `sheet` → `compare`/`tune` measure how much of the melody
//! and chords survive transcription.

use std::path::Path;

use anyhow::{Context, Result};

/// A melody note with seconds-based timing (synthesis input).
#[derive(Debug, Clone)]
pub struct SynthNote {
    pub pitch: u8,
    pub start_sec: f32,
    pub dur_sec: f32,
    pub velocity: f32,
}

/// A chord with seconds-based onset, ready for voicing.
#[derive(Debug, Clone)]
pub struct SynthChord {
    pub onset_sec: f32,
    pub dur_sec: f32,
    /// Root pitch class (0 = C ... 11 = B).
    pub root_pc: u8,
    /// Chord tones as pitch classes relative to C (0-11), includes root.
    pub pcs: Vec<u8>,
}

/// A parsed song ready for rendering.
#[derive(Debug, Clone)]
pub struct Song {
    pub title: String,
    pub bpm: f32,
    pub beats_per_bar: u32,
    pub melody: Vec<SynthNote>,
    pub chords: Vec<SynthChord>,
    pub duration_sec: f32,
}

fn parse_doc(xml: &str) -> Result<roxmltree::Document> {
    roxmltree::Document::parse_with_options(
        xml,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("failed to parse MusicXML: {e}"))
}

fn step_pc(step: &str) -> Option<u8> {
    match step {
        "C" => Some(0),
        "D" => Some(2),
        "E" => Some(4),
        "F" => Some(5),
        "G" => Some(7),
        "A" => Some(9),
        "B" => Some(11),
        _ => None,
    }
}

fn midi_from_step(step: &str, alter: i8, octave: u8) -> Option<u8> {
    let pc = step_pc(step)?;
    let oct = octave.checked_add(1)?;
    let midi = oct as i16 * 12 + pc as i16 + alter as i16;
    Some(midi.clamp(0, 127) as u8)
}

/// Map a MusicXML `<kind>` value to chord-tone pitch classes (relative to C,
/// includes the root). Falls back to a major triad for unknown kinds.
pub fn kind_to_pcs(kind: &str) -> Vec<u8> {
    let set: &[u8] = match kind {
        "major" => &[0, 4, 7],
        "major-seventh" | "maj7" => &[0, 4, 7, 11],
        "major-sixth" | "major-6th" => &[0, 4, 7, 9],
        "major-ninth" => &[0, 4, 7, 11, 2],
        "major-11th" => &[0, 4, 7, 11, 2, 5],
        "major-13th" | "major-thirteenth" => &[0, 4, 7, 11, 2, 9],
        "minor" => &[0, 3, 7],
        "minor-seventh" | "min7" => &[0, 3, 7, 10],
        "minor-sixth" | "minor-6th" => &[0, 3, 7, 9],
        "minor-major" | "minor-major-seventh" => &[0, 3, 7, 11],
        "minor-ninth" => &[0, 3, 7, 10, 2],
        "minor-11th" => &[0, 3, 7, 10, 2, 5],
        "minor-13th" => &[0, 3, 7, 10, 2, 9],
        "dominant" => &[0, 4, 7, 10],
        "dominant-seventh" => &[0, 4, 7, 10],
        "dominant-ninth" => &[0, 4, 7, 10, 2],
        "dominant-11th" => &[0, 4, 7, 10, 2, 5],
        "dominant-13th" | "dominant-thirteenth" => &[0, 4, 7, 10, 2, 9],
        "half-diminished" | "half-diminished-seventh" => &[0, 3, 6, 10],
        "diminished" | "diminished-seventh" => &[0, 3, 6, 9],
        "augmented" | "augmented-seventh" => &[0, 4, 8],
        "suspended-fourth" | "sus4" => &[0, 5, 7],
        "suspended-second" | "sus2" => &[0, 2, 7],
        "power" => &[0, 7],
        _ => &[0, 4, 7],
    };
    set.to_vec()
}

/// Parse a single-part MusicXML string into a [`Song`].
///
/// Tolerates DOCTYPE and any divisions/measure layout. The melody is voice
/// `1`; `<harmony>` elements become chord onsets at the beat position where
/// they appear; rests advance the position but produce no notes.
pub fn parse_musicxml(xml: &str) -> Result<Song> {
    let doc = parse_doc(xml)?;

    let root = doc.root_element();
    let title = root
        .descendants()
        .find(|n| n.tag_name().name() == "movement-title")
        .or_else(|| doc.descendants().find(|n| n.tag_name().name() == "work-title"))
        .and_then(|n| n.text())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Untitled".to_string());

    // First <divisions> anywhere (single-part Omnibook keeps it constant).
    let divisions: f32 = doc
        .descendants()
        .find(|n| n.tag_name().name() == "divisions")
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<f32>().ok())
        .context("missing <divisions>")?;

    // Tempo: first <metronome>/<per-minute>.
    let mut bpm: f32 = doc
        .descendants()
        .find(|n| n.tag_name().name() == "per-minute")
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|&b| b >= 20.0 && b <= 400.0)
        .unwrap_or(120.0);

    // Time signature from the first <time>.
    let beats_per_bar: u32 = doc
        .descendants()
        .find(|n| n.tag_name().name() == "time")
        .and_then(|t| t.children().find(|c| c.is_element() && c.tag_name().name() == "beats"))
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&b| b >= 1 && b <= 16)
        .unwrap_or(4);

    let spq = 60.0 / bpm; // seconds per quarter

    let mut melody: Vec<SynthNote> = Vec::new();
    let mut chords: Vec<SynthChord> = Vec::new();
    let mut beat_pos: f32 = 0.0;

    let mut cur_divs = divisions;

    for measure in doc.descendants().filter(|n| n.tag_name().name() == "measure") {
        for child in measure.children().filter(|c| c.is_element()) {
            match child.tag_name().name() {
                "attributes" => {
                    if let Some(d) = child
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "divisions")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        cur_divs = d;
                    }
                    // Tempo can change mid-song in the Omnibook (ritardandos).
                    if let Some(t) = child
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "time")
                        .and_then(|t| t.children().find(|c| c.is_element() && c.tag_name().name() == "beats"))
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        if (1..=16).contains(&t) {
                            let _ = t; // beats_per_bar only read from first measure
                        }
                    }
                }
                "direction" => {
                    if let Some(m) = child.descendants().find(|n| n.tag_name().name() == "metronome") {
                        if let Some(pm) = m
                            .children()
                            .find(|c| c.is_element() && c.tag_name().name() == "per-minute")
                            .and_then(|c| c.text())
                            .and_then(|s| s.parse::<f32>().ok())
                        {
                            if (20.0..=400.0).contains(&pm) {
                                bpm = pm;
                            }
                        }
                    }
                }
                "harmony" => {
                    let mut root_pc = 0u8;
                    let mut alter = 0i8;
                    let mut kind = String::new();
                    for hc in child.children().filter(|c| c.is_element()) {
                        match hc.tag_name().name() {
                            "root" => {
                                for rc in hc.children().filter(|c| c.is_element()) {
                                    match rc.tag_name().name() {
                                        "root-step" => {
                                            if let Some(pc) = rc.text().and_then(step_pc) {
                                                root_pc = pc;
                                            }
                                        }
                                        "root-alter" => {
                                            alter = rc
                                                .text()
                                                .and_then(|s| s.parse::<i8>().ok())
                                                .unwrap_or(0);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "kind" => {
                                kind = hc.text().unwrap_or("").trim().to_string();
                            }
                            _ => {}
                        }
                    }
                    root_pc = (root_pc as i16 + alter as i16).rem_euclid(12) as u8;
                    let pcs = kind_to_pcs(&kind);
                    let onset_sec = beat_pos * spq;
                    chords.push(SynthChord {
                        onset_sec,
                        dur_sec: 0.0, // filled in after the walk
                        root_pc,
                        pcs,
                    });
                }
                "backup" => {
                    if let Some(d) = child
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "duration")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        beat_pos = (beat_pos - d / cur_divs).max(0.0);
                    }
                }
                "forward" => {
                    if let Some(d) = child
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "duration")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        beat_pos += d / cur_divs;
                    }
                }
                "note" => {
                    let is_rest = child
                        .children()
                        .any(|c| c.is_element() && c.tag_name().name() == "rest");
                    let is_grace = child
                        .children()
                        .any(|c| c.is_element() && c.tag_name().name() == "grace");
                    let is_chord = child
                        .children()
                        .any(|c| c.is_element() && c.tag_name().name() == "chord");

                    let duration_quarters = child
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "duration")
                        .and_then(|c| c.text())
                        .and_then(|s| s.parse::<f32>().ok())
                        .map(|d| d / cur_divs)
                        .unwrap_or(0.0);

                    if is_chord {
                        // A simultaneous note sharing the previous onset; the
                        // Omnibook is monophonic so this is just a guard.
                        if !is_rest {
                            if let Some((pitch, _)) = parse_pitch(&child) {
                                let (prev_start, prev_dur) = melody
                                    .last()
                                    .map(|n| (n.start_sec, n.dur_sec))
                                    .unwrap_or((beat_pos * spq, 0.25));
                                melody.push(SynthNote {
                                    pitch,
                                    start_sec: prev_start,
                                    dur_sec: prev_dur,
                                    velocity: 0.85,
                                });
                            }
                        }
                        continue;
                    }

                    if is_grace {
                        if !is_rest {
                            if let Some((pitch, _)) = parse_pitch(&child) {
                                melody.push(SynthNote {
                                    pitch,
                                    start_sec: beat_pos * spq,
                                    dur_sec: (0.08 * spq).max(0.03),
                                    velocity: 0.7,
                                });
                            }
                        }
                        continue; // no position advance
                    }

                    if is_rest {
                        beat_pos += duration_quarters;
                        continue;
                    }

                    let Some((pitch, _)) = parse_pitch(&child) else {
                        beat_pos += duration_quarters;
                        continue;
                    };

                    let start = beat_pos * spq;
                    let dur = (duration_quarters * spq).max(0.02);
                    let velocity = if duration_quarters >= 1.0 { 0.95 } else { 0.85 };
                    melody.push(SynthNote {
                        pitch,
                        start_sec: start,
                        dur_sec: dur,
                        velocity,
                    });
                    beat_pos += duration_quarters;
                }
                _ => {}
            }
        }
    }

    let duration_sec = beat_pos * spq;

    // Fill chord durations from onset to next onset (clamped to the bar length).
    let bar_sec = beats_per_bar as f32 * spq;
    for i in 0..chords.len() {
        let end = chords
            .get(i + 1)
            .map(|c| c.onset_sec)
            .unwrap_or(duration_sec);
        chords[i].dur_sec = (end - chords[i].onset_sec).clamp(0.15, bar_sec.max(0.15));
    }

    Ok(Song {
        title,
        bpm,
        beats_per_bar,
        melody,
        chords,
        duration_sec,
    })
}

/// Extract `(midi pitch, is_rest)` from a `<note>` element.
fn parse_pitch(note: &roxmltree::Node) -> Option<(u8, bool)> {
    let pitch = note
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "pitch")?;
    let step = pitch
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "step")?
        .text()?;
    let alter = pitch
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "alter")
        .and_then(|c| c.text())
        .and_then(|s| s.parse::<i8>().ok())
        .unwrap_or(0);
    let octave = pitch
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "octave")?
        .text()?
        .parse::<u8>()
        .ok()?;
    midi_from_step(step, alter, octave).map(|m| (m, false))
}

// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------

/// Rendering parameters.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub sample_rate: u32,
    /// Melody gain (linear amplitude).
    pub melody_gain: f32,
    /// Chord-comping gain.
    pub chord_gain: f32,
    /// Seconds of silence appended after the last note.
    pub end_pad_secs: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            melody_gain: 0.62,
            chord_gain: 0.42,
            end_pad_secs: 0.6,
        }
    }
}

const LEAD_HARMONICS: &[(f32, f32)] = &[(1.0, 1.0), (2.0, 0.45), (3.0, 0.28), (4.0, 0.18), (5.0, 0.10)];
const PIANO_HARMONICS: &[(f32, f32)] = &[
    (1.0, 1.0),
    (2.0, 0.55),
    (3.0, 0.32),
    (4.0, 0.20),
    (5.0, 0.13),
    (6.0, 0.09),
    (7.0, 0.06),
];

fn freq_from_midi(pitch: u8) -> f32 {
    440.0 * 2.0_f32.powf((pitch as f32 - 69.0) / 12.0)
}

/// Build a jazz-ish voicing for a chord: root in octave 3, chord tones
/// clustered around middle C (upper-3rds-style spread). The root is voiced
/// prominently so the harmony detector can identify it.
pub fn voice_chord(root_pc: u8, pcs: &[u8]) -> Vec<u8> {
    let mut out = vec![48 + root_pc];
    let mut upper: Vec<u8> = Vec::new();
    for &pc in pcs {
        let pc = pc % 12;
        if pc == root_pc {
            continue; // root already voiced
        }
        let lo = 48 + pc;
        let hi = lo + 12;
        // Prefer the lower placement so the voicing stays below the melody.
        let p = if (lo as i16 - 60).abs() <= (hi as i16 - 60).abs() {
            lo
        } else {
            hi
        };
        upper.push(p);
    }
    upper.sort_unstable();
    upper.dedup();
    upper.truncate(4);
    out.extend(upper);
    out
}

/// Render a song to interleaved stereo f32 samples (L, R, L, R, ...).
pub fn render_song(song: &Song, opts: &RenderOptions) -> Vec<f32> {
    render_song_partial(song, opts, true, true)
}

/// Render a song with melody/chords independently toggled.
pub fn render_song_partial(
    song: &Song,
    opts: &RenderOptions,
    include_melody: bool,
    include_chords: bool,
) -> Vec<f32> {
    let sr = opts.sample_rate as f32;
    let total_secs = song.duration_sec + opts.end_pad_secs;
    let total = (total_secs * sr) as usize;
    let mut buf = vec![0.0f32; total * 2];

    if include_chords {
        // Chords first (underneath), then melody on top.
        for chord in &song.chords {
            for midi in voice_chord(chord.root_pc, &chord.pcs) {
                add_tone(
                    &mut buf,
                    sr,
                    chord.onset_sec,
                    chord.dur_sec,
                    freq_from_midi(midi),
                    opts.chord_gain * 0.85,
                    PIANO_HARMONICS,
                    true,
                    -0.18,
                    0.18,
                );
            }
        }
    }
    if include_melody {
        for note in &song.melody {
            add_tone(
                &mut buf,
                sr,
                note.start_sec,
                note.dur_sec,
                freq_from_midi(note.pitch),
                opts.melody_gain * note.velocity,
                LEAD_HARMONICS,
                false,
                0.12,
                0.12,
            );
        }
    }

    // Normalize to peak 0.9.
    let mut peak = 0.0f32;
    for &s in &buf {
        peak = peak.max(s.abs());
    }
    if peak > 0.9 {
        let g = 0.9 / peak;
        for s in &mut buf {
            *s *= g;
        }
    }
    buf
}

/// Add one note to an interleaved stereo buffer.
#[allow(clippy::too_many_arguments)]
fn add_tone(
    buf: &mut [f32],
    sr: f32,
    start: f32,
    dur: f32,
    freq: f32,
    gain: f32,
    harmonics: &[(f32, f32)],
    piano: bool,
    pan_l: f32,
    pan_r: f32,
) {
    let i_start = (start * sr) as usize;
    let n = ((start + dur) * sr) as usize;
    let total = buf.len() / 2;
    if i_start >= total {
        return;
    }
    let n = n.min(total);

    let att_s = (0.006 * sr) as usize;
    let rel_s = if piano { 0.035 } else { 0.05 } * sr;
    let tau = if piano { 0.35 * sr } else { 0.06 * sr };
    let sustain = if piano { 0.30 } else { 0.72 };
    let two_pi = std::f32::consts::PI * 2.0;

    for i in i_start..n {
        let t = (i - i_start) as f32 / sr;
        let mut env = if i - i_start < att_s {
            (i - i_start) as f32 / att_s.max(1) as f32
        } else {
            1.0
        };
        if piano {
            let after = ((i - i_start) as f32 - att_s as f32).max(0.0);
            env *= sustain + (1.0 - sustain) * (-after / tau).exp();
        } else {
            let after = ((i - i_start) as f32 - att_s as f32).max(0.0);
            env *= sustain + (1.0 - sustain) * (-after / tau).exp();
        }
        let rem = (n - i) as f32;
        if rem < rel_s {
            env *= rem / rel_s;
        }

        let phase = two_pi * freq * t;
        let mut s = 0.0f32;
        for &(mult, amp) in harmonics {
            s += amp * (phase * mult).sin();
        }
        let v = s * env * gain;
        buf[i * 2] += v * pan_l;
        buf[i * 2 + 1] += v * pan_r;
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Rescale all timings in a song to a new BPM (times scale by old/new).
pub fn rescale_tempo(song: &mut Song, new_bpm: f32) {
    if song.bpm <= 0.0 || new_bpm <= 0.0 {
        return;
    }
    let ratio = song.bpm / new_bpm;
    for n in &mut song.melody {
        n.start_sec *= ratio;
        n.dur_sec *= ratio;
    }
    for c in &mut song.chords {
        c.onset_sec *= ratio;
        c.dur_sec *= ratio;
    }
    song.duration_sec *= ratio;
    song.bpm = new_bpm;
}

/// Write interleaved stereo f32 samples to a 16-bit PCM WAV.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Write interleaved stereo f32 samples to an MP3 via LAME (compiled from
/// source by `mp3lame-encoder`, no external encoder binary needed).
pub fn write_mp3(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut pcm = vec![0i16; samples.len()];
    for (i, &s) in samples.iter().enumerate() {
        pcm[i] = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
    }

    let mut encoder = mp3lame_encoder::Builder::new()
        .ok_or_else(|| anyhow::anyhow!("lame init failed"))?
        .with_sample_rate(sample_rate)
        .map_err(|e| anyhow::anyhow!("lame: {e}"))?
        .with_num_channels(2)
        .map_err(|e| anyhow::anyhow!("lame: {e}"))?
        .with_brate(mp3lame_encoder::Bitrate::Kbps192)
        .map_err(|e| anyhow::anyhow!("lame: {e}"))?
        .with_quality(mp3lame_encoder::Quality::NearBest)
        .map_err(|e| anyhow::anyhow!("lame: {e}"))?
        .build()
        .map_err(|e| anyhow::anyhow!("lame: {e}"))?;

    let mut out: Vec<u8> = Vec::new();
    out.reserve(mp3lame_encoder::max_required_buffer_size(pcm.len() / 2) + 8192);
    encoder
        .encode_to_vec(mp3lame_encoder::InterleavedPcm(&pcm), &mut out)
        .map_err(|e| anyhow::anyhow!("lame encode: {e}"))?;
    out.reserve(7200);
    encoder
        .flush_to_vec::<mp3lame_encoder::FlushNoGap>(&mut out)
        .map_err(|e| anyhow::anyhow!("lame flush: {e}"))?;

    std::fs::write(path, &out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWINKLE: &str = r#"
<score-partwise version="3.1"><work><work-title>Twinkle</work-title></work>
<part-list><score-part id="P1"><part-name>Melody</part-name></score-part></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>60</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
<direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome></direction-type><sound tempo="120"/></direction>
<harmony><root><root-step>C</root-step></root><kind>major</kind></harmony>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>60</duration><type>quarter</type></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>60</duration><type>quarter</type></note>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>60</duration><type>quarter</type></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>60</duration><type>quarter</type></note>
</measure></part></score-partwise>
"#;

    #[test]
    fn parses_melody_chords_tempo() {
        let song = parse_musicxml(TWINKLE).unwrap();
        assert_eq!(song.title, "Twinkle");
        assert_eq!(song.bpm, 120.0);
        assert_eq!(song.beats_per_bar, 4);
        assert_eq!(song.melody.len(), 4);
        assert_eq!(song.melody[0].pitch, 60);
        assert_eq!(song.melody[3].pitch, 72);
        assert!((song.melody[1].start_sec - 0.5).abs() < 1e-3);
        assert_eq!(song.chords.len(), 1);
        assert_eq!(song.chords[0].root_pc, 0);
        assert!(song.chords[0].pcs.contains(&4));
        assert!((song.duration_sec - 2.0).abs() < 1e-3);
    }

    #[test]
    fn renders_audio_with_chords() {
        let song = parse_musicxml(TWINKLE).unwrap();
        let opts = RenderOptions {
            sample_rate: 8000,
            ..Default::default()
        };
        let samples = render_song(&song, &opts);
        let expect = ((song.duration_sec + opts.end_pad_secs) * 8000.0) as usize * 2;
        assert_eq!(samples.len(), expect);
        assert!(samples.iter().any(|&s| s.abs() > 0.1), "audio not silent");
    }

    #[test]
    fn voicing_keeps_chord_tones() {
        let v = voice_chord(0, &[0, 4, 7, 10]); // C7
        assert!(v.contains(&48)); // octave-3 root C
        assert!(v.iter().any(|&p| p % 12 == 4));
        assert!(v.iter().any(|&p| p % 12 == 10));
    }
}
