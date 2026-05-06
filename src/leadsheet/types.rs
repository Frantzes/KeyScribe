use serde::{Deserialize, Serialize};

/// Beat-note association: how a note relates to its surrounding beats.
#[derive(Debug, Clone, PartialEq)]
pub struct BeatAlignedNote {
    pub pitch: u8,
    pub velocity: u8,
    pub channel: Option<u8>,
    pub original_start_time: f32,
    pub original_end_time: f32,
    pub beat_index: u32,
    pub bar_index: u32,
    pub intra_beat_pos: f32,
    pub prev_beat_time: f32,
    pub next_beat_time: f32,
    pub beat_duration_sec: f32,
}

impl BeatAlignedNote {
    pub fn onset_ratio_in_bar(&self, beats_per_bar: u32) -> f32 {
        (self.beat_index % beats_per_bar) as f32 + self.intra_beat_pos
    }
}

/// Swing style classification for a section of music.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwingStyle {
    Straight,
    Swing,
    Triplet,
}

/// Section-level swing information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwingSection {
    pub bar_start: u32,
    pub bar_end: u32,
    pub style: SwingStyle,
    pub confidence: f32,
    pub swing_ratio: Option<f32>,
}

impl SwingSection {
    pub fn contains_bar(&self, bar: u32) -> bool {
        bar >= self.bar_start && bar < self.bar_end
    }
}

/// Articulation marking for a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Articulation {
    Normal,
    Staccato,
    Tenuto,
    Accent,
    Grace,
}

impl Default for Articulation {
    fn default() -> Self {
        Self::Normal
    }
}

/// Confidence-weighted downbeat inference result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Downbeat {
    pub beat_position: f32,
    pub confidence: f32,
    pub meter_class: MeterClass,
}

/// Unified rhythm map: tempo segments + time-signature segments + downbeats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RhythmMap {
    pub tempo_segments: Vec<TempoSegment>,
    pub time_signature_segments: Vec<TimeSignatureSegment>,
    pub downbeats: Vec<Downbeat>,
}

impl RhythmMap {
    pub fn is_empty(&self) -> bool {
        self.tempo_segments.is_empty()
            && self.time_signature_segments.is_empty()
            && self.downbeats.is_empty()
    }

    pub fn tempo_at_beat(&self, beat: f32) -> Option<&TempoSegment> {
        for seg in &self.tempo_segments {
            if seg.beat_offset <= beat && seg.end_time_sec > seg.start_time_sec {
                return Some(seg);
            }
        }
        self.tempo_segments.last()
    }

    pub fn time_signature_at_beat(&self, beat: f32) -> Option<&TimeSignatureSegment> {
        for seg in &self.time_signature_segments {
            if seg.contains_beat(beat) {
                return Some(seg);
            }
        }
        self.time_signature_segments.last()
    }
}

/// Core note-event contract from inference/output layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteEvent {
    /// MIDI note number (0-127).
    pub pitch: u8,
    /// Note start time in seconds from track start.
    pub start_time: f32,
    /// Note end time in seconds from track start.
    pub end_time: f32,
    /// MIDI velocity (0-127).
    pub velocity: u8,
    /// Optional source channel for multi-part tracks.
    pub channel: Option<u8>,
}

impl NoteEvent {
    pub fn duration_seconds(&self) -> f32 {
        (self.end_time - self.start_time).max(0.0)
    }
}

/// Stage-2 output after beat-grid quantization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedNote {
    pub pitch: u8,
    pub beat_start: f32,
    pub beat_duration: f32,
    pub velocity: u8,
    pub channel: Option<u8>,
    pub confidence: f32,
    /// Bar index this note falls in.
    pub bar_index: u32,
    /// Beat index within the bar (0-based).
    pub beat_index: u32,
    /// Original intra-beat position before quantization (0.0-1.0).
    pub intra_beat_pos: f32,
    /// Articulation marking.
    pub articulation: Articulation,
    /// Swing style active for this note's section.
    pub swing_style: SwingStyle,
    /// If true, render as straight 8ths with swing feel indicator.
    pub swing_feel: bool,
}

impl Default for QuantizedNote {
    fn default() -> Self {
        Self {
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TempoSegment {
    pub start_time_sec: f32,
    pub end_time_sec: f32,
    pub bpm: f32,
    pub beat_duration_sec: f32,
    pub beat_offset: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChordSymbolChange {
    pub beat_start: f32,
    pub symbol: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeterClass {
    SimpleDuple,
    SimpleTriple,
    SimpleQuadruple,
    CompoundDuple,
    CompoundQuadruple,
    Other,
}

impl MeterClass {
    pub fn from_signature(numerator: u8, denominator: u8) -> Self {
        match (numerator, denominator) {
            (2, 4) => Self::SimpleDuple,
            (3, 4) => Self::SimpleTriple,
            (4, 4) => Self::SimpleQuadruple,
            (6, 8) => Self::CompoundDuple,
            (12, 8) => Self::CompoundQuadruple,
            _ => Self::Other,
        }
    }

    pub fn is_compound(self) -> bool {
        matches!(self, Self::CompoundDuple | Self::CompoundQuadruple)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSignatureSegment {
    pub start_beat: f32,
    pub end_beat: f32,
    pub numerator: u8,
    pub denominator: u8,
    pub confidence: f32,
    pub meter_class: MeterClass,
}

impl TimeSignatureSegment {
    pub fn contains_beat(&self, beat: f32) -> bool {
        beat >= self.start_beat && beat < self.end_beat
    }

    pub fn beats_per_measure(&self) -> f32 {
        self.numerator as f32 * (4.0 / self.denominator as f32)
    }

    pub fn meter_class(&self) -> MeterClass {
        self.meter_class
    }
}

impl Default for TimeSignatureSegment {
    fn default() -> Self {
        Self {
            start_beat: 0.0,
            end_beat: f32::MAX,
            numerator: 4,
            denominator: 4,
            confidence: 1.0,
            meter_class: MeterClass::SimpleQuadruple,
        }
    }
}
