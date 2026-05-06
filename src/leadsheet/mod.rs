pub mod beat_association;
pub mod beat_tracking;
pub mod bpm;
pub mod chord;
pub mod instrument_separation;
pub mod joint_tracker;
pub mod preset;
pub mod quantize;
pub mod tempo_map;
pub mod types;

pub use beat_association::{
    associate_note_events, associate_notes_to_beat_grid, beats_per_bar_from_downbeats,
    BeatAssociationConfig,
};
pub use beat_tracking::{run_beat_this, BeatTrackConfig, BeatTrackDevice, BeatTrackResult};
pub use bpm::{detect_bpm, BpmDetectionConfig, TempoEstimate};
pub use chord::{detect_chord_changes, ChordAnalysisConfig};
pub use instrument_separation::{
    blend_for_chords, blend_interleaved_stems, extract_melodic_audio, InstrumentSeparator,
    SeparatedStem, SeparationConfig, StemType,
};
pub use joint_tracker::{
    collapse_to_tempo_segments, collapse_to_time_signature_segments, extract_downbeats_from_path,
    JointRhythmConfig, JointRhythmTracker, RhythmState,
};
pub use preset::{
    generate_lead_sheet_enhanced, generate_lead_sheet_foundation,
    generate_lead_sheet_with_tempo_map, LeadSheetFoundation, LeadSheetPresetConfig,
};
pub use quantize::{
    detect_articulation, detect_grace_notes, detect_swing, quantize_aligned_notes, quantize_notes,
    quantize_notes_with_rhythm_map, quantize_notes_with_tempo_map, quantize_notes_with_ties,
    QuantizationConfig, SwingDetectionConfig, TiedNote,
};
pub use tempo_map::{
    beat_at_time, detect_tempo_map, detect_time_signature_segments, tempo_map_from_beats,
    TempoMapConfig, TimeSignatureConfig,
};
pub use types::{
    Articulation, BeatAlignedNote, ChordSymbolChange, Downbeat, MeterClass, NoteEvent,
    QuantizedNote, RhythmMap, SwingSection, SwingStyle, TempoSegment, TimeSignatureSegment,
};
