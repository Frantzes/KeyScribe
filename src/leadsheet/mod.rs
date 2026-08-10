pub mod beat_association;
pub mod beat_tracking;
pub mod bpm;
pub mod chord;
pub mod harmony;
pub mod instrument_separation;
pub mod joint_tracker;
pub mod preset;
pub mod quantize;
pub mod tempo_map;
pub mod transition;
pub mod types;

pub use beat_association::{
    associate_note_events, associate_notes_to_beat_grid, beats_per_bar_from_downbeats,
    BeatAssociationConfig,
};
pub use beat_tracking::{
    cross_validate_beat_sources, detect_beats_from_notes, detect_beats_from_stems, run_beat_this,
    run_beat_this_combined, run_beat_this_multi, BeatTrackConfig, BeatTrackDevice,
    refine_beat_phase, BeatTrackResult, CrossValidatedBeats,
};
pub use bpm::{detect_bpm, BpmDetectionConfig, TempoEstimate};
pub use chord::{debug_chord_notes_to_json, detect_chord_changes, detect_chord_changes_per_bar, ChordAnalysisConfig};
pub use harmony::{compute_bar_profiles, detect_chords_from_timeline, estimate_key, TimelineChordInput};
pub use instrument_separation::{
    blend_for_chords, blend_interleaved_stems, extract_melodic_audio, InstrumentSeparator,
    SeparatedStem, SeparationConfig, StemType,
};
pub use joint_tracker::{
    collapse_to_tempo_segments, collapse_to_time_signature_segments, extract_downbeats_from_path,
    JointRhythmConfig, JointRhythmTracker, RhythmState,
};
pub use preset::{
    generate_lead_sheet_enhanced, generate_lead_sheet_enhanced_with_timeline,
    generate_lead_sheet_foundation, generate_lead_sheet_with_tempo_map, LeadSheetFoundation,
    LeadSheetPresetConfig,
};
pub use quantize::{
    detect_articulation, detect_grace_notes, detect_swing, learned_note_features,
    quantize_aligned_notes, quantize_aligned_notes_learned, quantize_notes,
    quantize_notes_with_rhythm_map, quantize_notes_with_tempo_map, quantize_notes_with_ties,
    resolve_quantizer_model_path, QuantizationConfig, QuantizerEngine, QuantizerToken,
    SwingDetectionConfig, TiedNote, LEARNED_TOKEN_TABLE,
};
pub use tempo_map::{
    beat_at_time, detect_tempo_map, detect_time_signature_segments, tempo_map_from_beats,
    TempoMapConfig, TimeSignatureConfig,
};
pub use transition::{
    fit_transitions_from_musicxml_dir, load_learned_transitions, LearnedTransitions,
};
pub use types::{
    Articulation, BeatAlignedNote, ChordSymbolChange, Downbeat, MeterClass, NoteEvent,
    QuantizedNote, RhythmMap, SwingSection, SwingStyle, TempoSegment, TimeSignatureSegment,
};
