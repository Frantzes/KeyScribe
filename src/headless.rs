//! Headless transcription & sheet-generation pipeline for the CLI.
//!
//! Mirrors the desktop app's core flow (Basic Pitch -> note events -> beat
//! tracking -> enhanced lead sheet -> MusicXML) without any egui dependency, so
//! an agent can drive the whole pipeline from the command line.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::leadsheet::{
    cross_validate_beat_sources, generate_lead_sheet_enhanced_with_timeline, refine_beat_phase,
    BeatTrackConfig, CrossValidatedBeats, LeadSheetFoundation, LeadSheetPresetConfig, NoteEvent,
};
use crate::musicxml::{
    build_musicxml_document, extract_melody_heuristic, extract_melody_skyline,
    merge_adjacent_notes_with_gap, SheetEngravingConfig, SHEET_SWING_BIAS,
};
use crate::pipeline::{AudioPipeline, PipelineConfig};

pub const DEFAULT_NOTE_THRESHOLD: f32 = 0.10;
/// Mirror of the GUI's `NOTE_HIGHLIGHT_ACTIVATION_THRESHOLD` (src/app.rs).
const KEY_SENSITIVITY_ACTIVATION_THRESHOLD: f32 = 0.12;
/// Default "Key Color Sensitivity" slider value, matching the GUI default
/// (internal `key_color_sensitivity` of 0.46 shown as 0.46 * 0.5 = 0.23).
pub const DEFAULT_KEY_SENSITIVITY: f32 = 0.23;
const PIANO_LOW_MIDI: u8 = 21;
const PIANO_HIGH_MIDI: u8 = 108;
const MIN_SHEET_NOTE_FRAMES: usize = 2;

/// Convert the GUI "Key Color Sensitivity" slider value (0.0-1.0) to the note
/// probability threshold, mirroring the app's internal mapping:
/// internal sensitivity = slider * 2.0, threshold = 0.12 / internal (clamped).
/// This keeps CLI note density identical to what the desktop app produces for
/// the same slider setting.
pub fn key_sensitivity_to_threshold(slider_value: f32) -> f32 {
    let internal = slider_value.clamp(0.0, 1.0) * 2.0;
    if internal > 0.0 {
        (KEY_SENSITIVITY_ACTIVATION_THRESHOLD / internal).clamp(0.05, 0.95)
    } else {
        0.95
    }
}

fn default_melody_quantizer() -> String {
    "legacy".to_string()
}

/// Learned pipeline parameters written by the `tune` subcommand and applied by
/// `sheet`/`midi` via `--config`. Explicit CLI flags override these defaults.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunedConfig {
    pub key_sensitivity: f32,
    pub chord_beat: f32,
    pub chord_cleanest: bool,
    pub chord_strike: bool,
    pub melody: String,
    pub stems: bool,
    /// Stem-first melody default (use the identified melodic stem as the melody
    /// source). Missing from legacy config files → deserializes to `false`, so
    /// an existing `keyscribe.tuned.json` keeps the behavior it was tuned with.
    #[serde(default)]
    pub melody_stems: bool,
    /// Rhythm-quantization engine: "legacy" | "learned". A fresh run (no
    /// config) defaults to "learned" (auto-falls back to the rule grid when
    /// `melody_quantizer.onnx` is absent); missing from legacy config files →
    /// "legacy" (back-compat).
    #[serde(default = "default_melody_quantizer")]
    pub melody_quantizer: String,
    /// Fixed tempo grid override learned by `tune` (applies when the track
    /// has no explicit `--bpm`).
    pub bpm: Option<f32>,
}

impl Default for TunedConfig {
    fn default() -> Self {
        Self {
            key_sensitivity: DEFAULT_KEY_SENSITIVITY,
            chord_beat: -1.0,
            chord_cleanest: false,
            chord_strike: false,
            melody: "poly".to_string(),
            stems: false,
            melody_stems: true,
            melody_quantizer: "learned".to_string(),
            bpm: None,
        }
    }
}

impl TunedConfig {
    /// Load a tuned-parameters config written by the `tune` subcommand.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("invalid config {}", path.display()))
    }

    /// Load the default tuned config (`keyscribe.tuned.json` in the working
    /// directory) if it exists.
    pub fn load_default() -> Result<Option<Self>> {
        let path = Path::new("keyscribe.tuned.json");
        if path.is_file() {
            Ok(Some(Self::load(path)?))
        } else {
            Ok(None)
        }
    }

    /// Persist the tuned parameters to a JSON config file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let raw = serde_json::to_string_pretty(self)?;
        std::fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum MelodyMode {
    Polyphonic,
    Skyline,
    Heuristic,
}

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub threshold: f32,
    pub melody_mode: MelodyMode,
    pub melody_outlier_semitones: u8,
    /// Directory containing `basic-pitch.onnx`. Falls back to the default
    /// relative `models/` path when `None`.
    pub model_dir: Option<PathBuf>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_NOTE_THRESHOLD,
            melody_mode: MelodyMode::Polyphonic,
            melody_outlier_semitones: 12,
            model_dir: None,
        }
    }
}

#[derive(Debug)]
pub struct Transcription {
    pub notes: Vec<NoteEvent>,
    pub beats: CrossValidatedBeats,
    pub sample_rate: u32,
    pub duration_sec: f32,
    pub channel_count: u16,
}

/// Run Basic Pitch on a mono sample buffer and return the note-probability
/// timeline plus the per-frame step size (seconds), plus the onset-probability
/// timeline when the model provides one. This is the expensive inference
/// step; downstream parameters (threshold, melody mode) are applied later
/// from the timeline so they can be swept cheaply.
fn run_basic_pitch(
    samples: &[f32],
    sample_rate: u32,
    model_dir: Option<&Path>,
) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>, f32)> {
    if samples.is_empty() {
        return Ok((Vec::new(), None, 0.0));
    }
    let sample_rate = sample_rate.max(1);
    let duration_sec = samples.len() as f32 / sample_rate as f32;
    eprintln!(
        "[keyscribe] Basic Pitch on {:.1}s of audio...",
        duration_sec
    );
    let t0 = std::time::Instant::now();

    let model_path = model_dir
        .map(|d| d.join("basic-pitch.onnx").to_string_lossy().into_owned())
        .unwrap_or_else(|| "models/basic-pitch.onnx".to_string());

    let config = PipelineConfig {
        sample_rate,
        chunk_size: sample_rate as usize / 10,
        lookahead_frames: 5,
        model_path,
        ..Default::default()
    };
    let pipeline = AudioPipeline::new(config).context("failed to init Basic Pitch pipeline")?;
    let result = pipeline
        .process_audio(samples)
        .context("Basic Pitch analysis failed")?;
    let probs = result.note_probs_sequence;
    let onsets = result.onset_probs_sequence;

    let step_sec = if probs.is_empty() {
        0.0
    } else {
        (duration_sec / probs.len() as f32).max(1e-3)
    };
    eprintln!(
        "[keyscribe] Basic Pitch done in {:.1}s (onset head: {})",
        t0.elapsed().as_secs_f32(),
        if onsets.is_some() { "yes" } else { "no" }
    );
    Ok((probs, onsets, step_sec))
}

/// Convert one or more probability timelines into note events, applying the
/// threshold and (optional) melody reduction per source, mirroring the app's
/// `extract_events_from_timeline_data` logic so CLI and GUI agree.
fn notes_from_timeline(
    timelines: &[(Vec<Vec<f32>>, f32)],
    onset_timelines: &[(Vec<Vec<f32>>, f32)],
    threshold: f32,
    melody_mode: MelodyMode,
    outlier_semitones: u8,
) -> Vec<NoteEvent> {
    let mut all = Vec::new();
    let mut next_id: u32 = 0;
    for (i, (timeline, step_sec)) in timelines.iter().enumerate() {
        let onset_tl = onset_timelines
            .get(i)
            .map(|(tl, _)| tl.as_slice());
        let mut notes = extract_notes_from_timeline(
            timeline,
            onset_tl,
            *step_sec,
            threshold,
            &mut next_id,
        );
        notes = match melody_mode {
            MelodyMode::Polyphonic => notes,
            MelodyMode::Skyline => extract_melody_skyline(&notes, outlier_semitones),
            MelodyMode::Heuristic => extract_melody_heuristic(&notes, outlier_semitones),
        };
        all.extend(notes);
    }
    all
}

/// Result of the expensive analysis step (Basic Pitch + beat tracking),
/// reusable across cheap parameter sweeps so the `tune` command can train the
/// pipeline's thresholds/sampling without re-running inference.
pub struct AnalyzedAudio {
    /// Full-mix Basic Pitch timeline (note-probability, step_sec). Always one
    /// entry; feeds chord detection (per the chord plan, the chord head sees
    /// the full mix rather than per-stem evidence).
    pub timelines: Vec<(Vec<Vec<f32>>, f32)>,
    /// Parallel to `timelines`: (onset-probability timeline, step_sec) for the
    /// full mix, when the model exposes an onset head.
    pub onset_timelines: Vec<(Vec<Vec<f32>>, f32)>,
    /// Basic Pitch timelines of the identified melodic stem — the melody source
    /// when stems are used (one entry, or empty when stems aren't run).
    pub stem_timelines: Vec<(Vec<Vec<f32>>, f32)>,
    /// Parallel to `stem_timelines`: (onset-probability timeline, step_sec).
    pub stem_onset_timelines: Vec<(Vec<Vec<f32>>, f32)>,
    /// Index into `stem_timelines` of the stem identified as carrying the melody
    /// (when stems are used and a melodic stem is identified).
    pub melodic_stem_timeline: Option<usize>,
    pub beats: CrossValidatedBeats,
    pub sample_rate: u32,
    pub duration_sec: f32,
    pub channel_count: u16,
}

/// Run the expensive analysis once: load audio, run Basic Pitch on the full mix
/// (always; chord evidence), run Basic Pitch on the identified melodic stem when
/// `use_stems` is set (melody evidence), and cross-validate beat tracking using
/// the drums/bass stems when available. Returns cached timelines that
/// downstream steps consume.
pub fn analyze_audio(
    audio_path: &Path,
    use_stems: bool,
    model_dir: Option<&Path>,
) -> Result<AnalyzedAudio> {
    let audio = crate::audio_io::load_audio_file(audio_path)
        .with_context(|| format!("failed to load audio {}", audio_path.display()))?;
    let sample_rate = audio.sample_rate.max(1);
    let duration_sec = audio.samples_mono.len() as f32 / sample_rate as f32;

    let mut timelines = Vec::new();
    let mut onset_timelines: Vec<(Vec<Vec<f32>>, f32)> = Vec::new();
    let mut stem_timelines: Vec<(Vec<Vec<f32>>, f32)> = Vec::new();
    let mut stem_onset_timelines: Vec<(Vec<Vec<f32>>, f32)> = Vec::new();
    let mut bass: Option<Vec<f32>> = None;
    let mut drums: Option<Vec<f32>> = None;
    let mut melodic_stem_timeline: Option<usize> = None;

    // Full-mix timeline: always produced, feeds the chord head (full mix).
    {
        let (tl, ons, step) = run_basic_pitch(&audio.samples_mono, sample_rate, model_dir)?;
        timelines.push((tl, step));
        onset_timelines.push((ons.unwrap_or_default(), step));
    }

    if use_stems {
        let mut separator = crate::demucs::DemucsSeparator::new("htdemucs_6s")
            .context("failed to load demucs model")?;
        let stems = separator
            .separate_file(audio_path, None)
            .with_context(|| format!("demucs separation failed on {}", audio_path.display()))?;
        let melodic_stem = crate::leadsheet::preset::identify_melodic_stem_from_stems(&stems);
        let mut melodic_timeline_idx: Option<usize> = None;
        for stem in &stems {
            if stem.stem_type.is_melodic() {
                if Some(&stem.stem_type) == melodic_stem.as_ref() {
                    eprintln!(
                        "[keyscribe] analyzing {} stem...",
                        stem.stem_type.display_name()
                    );
                    let (tl, ons, step) =
                        run_basic_pitch(&stem.samples_mono, stem.sample_rate, model_dir)?;
                    melodic_timeline_idx = Some(stem_timelines.len());
                    stem_timelines.push((tl, step));
                    stem_onset_timelines.push((ons.unwrap_or_default(), step));
                }
            } else {
                match stem.stem_type {
                    crate::leadsheet::StemType::Bass => bass = Some(stem.samples_mono.to_vec()),
                    crate::leadsheet::StemType::Drums => drums = Some(stem.samples_mono.to_vec()),
                    _ => {}
                }
            }
        }
        eprintln!(
            "[keyscribe] melodic stem: {} (timeline {})",
            melodic_stem
                .as_ref()
                .map(|s| s.display_name().to_string())
                .unwrap_or_else(|| "none".to_string()),
            melodic_timeline_idx
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".to_string())
        );
        melodic_stem_timeline = melodic_timeline_idx;
    }

    eprintln!("[keyscribe] beat tracking...");
    let t_bt = std::time::Instant::now();
    let beats = cross_validate_beat_sources(
        bass.as_deref(),
        drums.as_deref(),
        Some(&audio.samples_mono),
        sample_rate,
        &BeatTrackConfig::default(),
    )
    .context("beat tracking failed")?;
    eprintln!(
        "[keyscribe] beat tracking done in {:.1}s: {:.0} bpm, {} beats/bar",
        t_bt.elapsed().as_secs_f32(),
        beats.bpm,
        beats.beats_per_bar
    );

    Ok(AnalyzedAudio {
        timelines,
        onset_timelines,
        stem_timelines,
        stem_onset_timelines,
        melodic_stem_timeline,
        beats,
        sample_rate,
        duration_sec,
        channel_count: audio.channels,
    })
}

/// Extract note events from an analysis at a given threshold / melody mode.
/// Cheap enough to call repeatedly during parameter tuning.
pub fn notes_from_analysis(
    analysis: &AnalyzedAudio,
    threshold: f32,
    melody_mode: MelodyMode,
    outlier_semitones: u8,
) -> Vec<NoteEvent> {
    // For melody reduction, transcribe only the identified melodic stem when
    // available (stem separation): the accompaniment registers would otherwise
    // be picked as the "highest line" and destroy the melody.
    let notes = if melody_mode != MelodyMode::Polyphonic {
        if let Some(idx) = analysis.melodic_stem_timeline {
            if let Some(tl) = analysis.stem_timelines.get(idx) {
                let mut nid: u32 = 0;
                let onset_tl = analysis
                    .stem_onset_timelines
                    .get(idx)
                    .map(|(o, _)| o.as_slice());
                let mut notes = extract_notes_from_timeline(
                    &tl.0,
                    onset_tl,
                    tl.1,
                    threshold,
                    &mut nid,
                );
                notes = match melody_mode {
                    MelodyMode::Polyphonic => notes,
                    MelodyMode::Skyline => extract_melody_skyline(&notes, outlier_semitones),
                    MelodyMode::Heuristic => extract_melody_heuristic(&notes, outlier_semitones),
                };
                eprintln!(
                    "[keyscribe] melody from identified stem (timeline {}): {} note events",
                    idx,
                    notes.len()
                );
                notes
            } else {
                notes_from_timeline(
                    &analysis.timelines,
                    &analysis.onset_timelines,
                    threshold,
                    melody_mode,
                    outlier_semitones,
                )
            }
        } else {
            notes_from_timeline(
                &analysis.timelines,
                &analysis.onset_timelines,
                threshold,
                melody_mode,
                outlier_semitones,
            )
        }
    } else {
        notes_from_timeline(
            &analysis.timelines,
            &analysis.onset_timelines,
            threshold,
            melody_mode,
            outlier_semitones,
        )
    };
    eprintln!("[keyscribe] {} note events", notes.len());
    if std::env::var_os("KEYSCRIBE_RAW_NOTES_DEBUG").is_some() {
        for n in notes.iter().take(40) {
            eprintln!(
                "[keyscribe-raw] {:.3}s pitch={} vel={} end={:.3}s",
                n.start_time, n.pitch, n.velocity, n.end_time
            );
        }
    }
    notes
}

/// Run the full-mix transcription pipeline on an audio file: load audio, run
/// Basic Pitch, convert the note-probability timeline into note events, apply
/// optional melody reduction, and cross-validate beat tracking.
pub fn transcribe_notes(audio_path: &Path, opts: &TranscribeOptions) -> Result<Transcription> {
    let analysis = analyze_audio(audio_path, false, opts.model_dir.as_deref())?;
    let notes = notes_from_analysis(
        &analysis,
        opts.threshold,
        opts.melody_mode,
        opts.melody_outlier_semitones,
    );
    Ok(Transcription {
        notes,
        beats: analysis.beats,
        sample_rate: analysis.sample_rate,
        duration_sec: analysis.duration_sec,
        channel_count: analysis.channel_count,
    })
}

/// Transcribe an audio file using htdemucs stem separation: only the melodic
/// stems (vocals, piano, guitar, other) are analysed with Basic Pitch, which
/// dramatically improves note/chord accuracy on full-band recordings. Beat
/// tracking uses the drums+bass stems when available.
pub fn transcribe_notes_with_stems(audio_path: &Path, opts: &TranscribeOptions) -> Result<Transcription> {
    let analysis = analyze_audio(audio_path, true, opts.model_dir.as_deref())?;
    let notes = notes_from_analysis(
        &analysis,
        opts.threshold,
        opts.melody_mode,
        opts.melody_outlier_semitones,
    );
    Ok(Transcription {
        notes,
        beats: analysis.beats,
        sample_rate: analysis.sample_rate,
        duration_sec: analysis.duration_sec,
        channel_count: analysis.channel_count,
    })
}

#[derive(Debug, Clone)]
pub struct SheetOptions {
    pub title: String,
    pub is_lead_sheet: bool,
    pub single_staff: bool,
    /// Override beat tracking with a fixed tempo grid (in BPM, 4 beats/bar).
    /// Mirrors the app's manual-BPM path; useful when the ML beat tracker gets
    /// the tempo wrong (e.g. half-time) for a given track.
    pub manual_bpm: Option<f32>,
    /// Separate the audio into stems and transcribe only the melodic stems.
    /// Much better for full-band recordings; requires models/htdemucs_6s.onnx.
    pub use_stems: bool,
    /// Stem-first melody default: use the identified melodic stem as the melody
    /// source (via Demucs separation) even when `use_stems` is false. Falls back
    /// to the full mix when `htdemucs_6s.onnx` is unavailable. Set false to keep
    /// the cheaper full-mix melody path.
    pub melody_stems: bool,
    /// Beat offset within the bar (0.0 = downbeat) at which to sample notes for
    /// the primary chord. Negative disables the fixed bias and uses the legacy
    /// max-simultaneous-notes scan.
    pub chord_sample_beat: f32,
    /// When true and `chord_sample_beat < 0`, scan the first half of the bar but
    /// pick the position with the fewest pitch classes (>= the simultaneous
    /// threshold) instead of the most.
    pub chord_sample_cleanest: bool,
    /// When true and `chord_sample_beat < 0`, scan the first half of the bar but
    /// pick the position where the most notes onset together (the strike moment).
    pub chord_sample_strike: bool,
    /// Rhythm-quantization engine for the melody. `LearnedOnnx` (default) falls
    /// back to the rule grid when `melody_quantizer.onnx` is absent.
    pub quantizer: crate::leadsheet::QuantizerEngine,
    /// Optional explicit path to `melody_quantizer.onnx`.
    pub quantizer_model_path: Option<PathBuf>,
}

impl Default for SheetOptions {
    fn default() -> Self {
        Self {
            title: "Untitled".to_string(),
            is_lead_sheet: true,
            single_staff: false,
            manual_bpm: None,
            use_stems: false,
            melody_stems: true,
            chord_sample_beat: -1.0,
            chord_sample_cleanest: false,
            chord_sample_strike: false,
            quantizer: crate::leadsheet::QuantizerEngine::default(),
            quantizer_model_path: None,
        }
    }
}

/// Build a synthetic beat grid at a fixed BPM (4/4), matching the app's
/// manual-BPM override path.
pub fn synthetic_beat_grid(bpm: f32, duration_sec: f32) -> CrossValidatedBeats {
    let bpm = bpm.clamp(30.0, 400.0);
    let beat_duration = 60.0 / bpm;
    let total_sec = duration_sec.max(10.0) + 2.0;
    let mut beats: Vec<f32> = Vec::new();
    let mut t = 0.0f32;
    while t <= total_sec + 1e-3 {
        beats.push(t);
        t += beat_duration;
    }
    let downbeats: Vec<f32> = beats.iter().step_by(4).copied().collect();
    CrossValidatedBeats {
        beats,
        downbeats,
        beats_per_bar: 4,
        bpm,
        confidence: 1.0,
        source_count: 1,
    }
}

#[derive(Debug)]
pub struct SheetResult {
    pub foundation: LeadSheetFoundation,
    pub musicxml: String,
    pub note_count: usize,
    pub bpm: f32,
    pub beats_per_bar: u32,
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

fn generate_sheet_inner(
    analysis: &AnalyzedAudio,
    transcribe_opts: &TranscribeOptions,
    sheet_opts: &SheetOptions,
) -> Result<SheetResult> {
    let notes = notes_from_analysis(
        analysis,
        transcribe_opts.threshold,
        transcribe_opts.melody_mode,
        transcribe_opts.melody_outlier_semitones,
    );

    let beats = match sheet_opts.manual_bpm {
        Some(bpm) => synthetic_beat_grid(bpm, analysis.duration_sec),
        None => refine_beat_phase(&notes, &analysis.beats),
    };

    let mut config = LeadSheetPresetConfig::default();
    // Keep 16th notes and dotted/compound durations (matches the app's export).
    config.quantization.min_duration_beats = 0.25;
    config.chord_analysis.chord_sample_beat = sheet_opts.chord_sample_beat;
    config.chord_analysis.chord_sample_cleanest = sheet_opts.chord_sample_cleanest;
    config.chord_analysis.chord_sample_strike = sheet_opts.chord_sample_strike;
    config.quantizer = sheet_opts.quantizer;
    config.quantizer_model_path = sheet_opts.quantizer_model_path.clone();

    let note_count = notes.len();
    let beat_count = beats.beats.len();
    let timeline_input = crate::leadsheet::TimelineChordInput {
        timelines: analysis.timelines.clone(),
        onset_timelines: analysis.onset_timelines.clone(),
    };
    let foundation = generate_lead_sheet_enhanced_with_timeline(
        &notes,
        beats.beats.as_slice(),
        beats.downbeats.as_slice(),
        beats.beats_per_bar,
        &config,
        Some(&timeline_input),
    )
    .ok_or_else(|| {
        anyhow!(
            "lead-sheet generation failed: not enough notes/beats ({} notes, {} beats)",
            note_count,
            beat_count
        )
    })?;

    let engraving = SheetEngravingConfig {
        _allow_triplets: !SHEET_SWING_BIAS,
        is_lead_sheet: sheet_opts.is_lead_sheet,
        single_staff: sheet_opts.single_staff,
    };
    let musicxml = build_musicxml_document(&sheet_opts.title, &foundation, engraving);

    Ok(SheetResult {
        foundation,
        musicxml,
        note_count,
        bpm: beats.bpm,
        beats_per_bar: beats.beats_per_bar,
        beats: beats.beats,
        downbeats: beats.downbeats,
    })
}

/// Generate a MusicXML lead sheet from an audio file.
pub fn generate_sheet(
    audio_path: &Path,
    transcribe_opts: &TranscribeOptions,
    sheet_opts: &SheetOptions,
) -> Result<SheetResult> {
    let analysis = analyze_audio_for_sheet(audio_path, sheet_opts, transcribe_opts.model_dir.as_deref())?;
    generate_sheet_inner(&analysis, transcribe_opts, sheet_opts)
}

/// Choose and run the analysis for the sheet path. Explicit `use_stems` (the
/// `--stems` flag) hard-fails if the Demucs model is missing; the stem-first
/// melody default (`melody_stems`, no explicit flag) degrades gracefully to the
/// full-mix melody when `htdemucs_6s.onnx` isn't available.
fn analyze_audio_for_sheet(
    audio_path: &Path,
    sheet_opts: &SheetOptions,
    model_dir: Option<&Path>,
) -> Result<AnalyzedAudio> {
    if sheet_opts.use_stems {
        return analyze_audio(audio_path, true, model_dir);
    }
    if sheet_opts.melody_stems {
        if crate::demucs::resolve_model_path("htdemucs_6s.onnx").is_some() {
            return analyze_audio(audio_path, true, model_dir);
        }
        eprintln!(
            "[keyscribe] htdemucs_6s.onnx not found — falling back to full-mix melody"
        );
    }
    analyze_audio(audio_path, false, model_dir)
}

/// Generate a MusicXML lead sheet from a precomputed analysis. Cheap; used by
/// the `tune` command to sweep parameters without re-running inference.
pub fn generate_sheet_from_analysis(
    analysis: &AnalyzedAudio,
    transcribe_opts: &TranscribeOptions,
    sheet_opts: &SheetOptions,
) -> Result<SheetResult> {
    generate_sheet_inner(analysis, transcribe_opts, sheet_opts)
}

/// Run htdemucs_6s stem separation and write each stem to `out_dir` as a WAV.
/// Returns the list of written files.
pub fn separate_stems(audio_path: &Path, out_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut separator = crate::demucs::DemucsSeparator::new("htdemucs_6s")
        .context("failed to load demucs model")?;
    let stems = separator
        .separate_file(audio_path, None)
        .with_context(|| format!("demucs separation failed on {}", audio_path.display()))?;

    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;

    let mut written = Vec::new();
    for stem in stems {
        let name = format!("{}.wav", stem.stem_type.display_name());
        let path = out_dir.join(name);
        let spec = hound::WavSpec {
            channels: stem.channels.max(1),
            sample_rate: stem.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec)
            .with_context(|| format!("failed to create {}", path.display()))?;
        for &s in stem.samples_interleaved.iter() {
            let _ = writer.write_sample(s);
        }
        writer
            .finalize()
            .with_context(|| format!("failed to finalize {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// Convert a probability timeline into `NoteEvent`s, mirroring the app's
/// `extract_events_from_timeline_data` logic so CLI and GUI agree. Notes are
/// split at re-articulations (staccato repeats) via an adaptive release
/// threshold and the model's onset head, and onsets are attack-adjusted so
/// timing is accurate.
fn extract_notes_from_timeline(
    timeline: &[Vec<f32>],
    onset_timeline: Option<&[Vec<f32>]>,
    step_sec: f32,
    threshold: f32,
    next_id: &mut u32,
) -> Vec<NoteEvent> {
    if timeline.is_empty() || step_sec <= 0.0 {
        return Vec::new();
    }

    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    let mut out = Vec::new();
    let min_duration_sec = (step_sec * MIN_SHEET_NOTE_FRAMES as f32).max(0.05);
    // Adaptive release: a note ends when its probability falls below this
    // fraction of its own peak (or the absolute floor). A fixed threshold
    // keeps staccato re-articulations merged into one long note; the
    // peak-relative level splits them.
    let release_ratio = 0.55f32;
    let release_floor = 0.05f32;
    // When a note starts, search back up to this many frames for the low
    // point where its probability began the rise, and start there instead of
    // at the threshold crossing (which lags the true onset).
    let attack_lookback = 5usize;
    // A frame is a re-articulation when the model's onset head fires while
    // the note is still sounding — this splits staccato repeats even when the
    // note probability never dips below the release level.
    let onset_split_threshold = 0.35f32;

    let prob_at = |note_idx: usize, frame_idx: usize| -> f32 {
        timeline
            .get(frame_idx)
            .and_then(|f| f.get(note_idx))
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    };
    let onset_at = |note_idx: usize, frame_idx: usize| -> f32 {
        onset_timeline
            .and_then(|tl| tl.get(frame_idx))
            .and_then(|f| f.get(note_idx))
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    };

    for note_idx in 0..note_count {
        let mut run_start: Option<usize> = None;
        let mut max_prob: f32 = 0.0;

        for frame_idx in 0..timeline.len() {
            let prob = prob_at(note_idx, frame_idx);
            let active = prob >= threshold;

            if active {
                if run_start.is_none() {
                    let mut onset = frame_idx;
                    let mut k = frame_idx;
                    let mut p_k = prob;
                    while k > 0 && frame_idx - k < attack_lookback {
                        let p_prev = prob_at(note_idx, k - 1);
                        if p_prev < release_floor && p_k > p_prev {
                            onset = k - 1;
                            break;
                        }
                        if p_prev >= p_k {
                            break;
                        }
                        k -= 1;
                        p_k = p_prev;
                    }
                    run_start = Some(onset);
                    max_prob = prob;
                } else {
                    max_prob = max_prob.max(prob);
                    // Re-articulation: the onset head fired on a sounding
                    // note (staccato repeat) — split here.
                    if onset_at(note_idx, frame_idx) >= onset_split_threshold {
                        let start_time = run_start.unwrap() as f32 * step_sec;
                        let mut end_time = frame_idx as f32 * step_sec;
                        if end_time <= start_time {
                            end_time = start_time + step_sec;
                        }
                        let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                        out.push(NoteEvent {
                            id: *next_id,
                            pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                            start_time,
                            end_time,
                            velocity,
                            channel: None,
                        });
                        *next_id = next_id.saturating_add(1);
                        run_start = Some(frame_idx);
                        max_prob = prob;
                    }
                }
            } else if let Some(start_idx) = run_start {
                let release_thr = (max_prob * release_ratio).max(release_floor);
                if prob < release_thr {
                    let start_time = start_idx as f32 * step_sec;
                    let mut end_time = frame_idx as f32 * step_sec;
                    if end_time <= start_time {
                        end_time = start_time + step_sec;
                    }
                    let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                    out.push(NoteEvent {
                        id: *next_id,
                        pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                        start_time,
                        end_time,
                        velocity,
                        channel: None,
                    });
                    *next_id = next_id.saturating_add(1);
                    run_start = None;
                    max_prob = 0.0;
                }
            }
        }

        if let Some(start_idx) = run_start {
            let start_time = start_idx as f32 * step_sec;
            let end_time = timeline.len() as f32 * step_sec;
            let end_time = end_time.max(start_time + step_sec);
            let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
            out.push(NoteEvent {
                id: *next_id,
                pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                start_time,
                end_time,
                velocity,
                channel: None,
            });
            *next_id = next_id.saturating_add(1);
        }
    }

    // Merge only single-frame jitter so genuine re-articulations survive.
    merge_adjacent_notes_with_gap(&mut out, step_sec);
    out.retain(|n| n.end_time - n.start_time >= min_duration_sec);
    out
}
