//! Batch evaluation harness for the whole labeled corpus.
//!
//! `keyscribe-cli eval-corpus -i <dir>` iterates every `(audio, reference
//! .musicxml)` pair in a directory (the Omnibook set), runs the full
//! `sheet` pipeline against each, and aggregates the per-track
//! melody/chord metrics into a JSON report plus a printed dashboard table.
//!
//! A per-track BPM override file (`--bpm-file <track>:<bpm>` JSON object or
//! one `<track> <bpm>` per line) makes runs reproducible when the ML beat
//! tracker misfires on a track's meter (e.g. returns half-time).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::headless::{
    analyze_audio, generate_sheet_from_analysis, key_sensitivity_to_threshold, MelodyMode,
    SheetOptions, TranscribeOptions,
};
use crate::leadsheet::QuantizerEngine;
use crate::sheet_compare::{
    compare_bar_placement, compare_harmonies, compare_note_lists, parse_beats_per_bar,
    parse_musicxml_harmonies, parse_musicxml_notes,
};
use crate::tune::Objective;

/// Configuration for one `eval-corpus` run.
#[derive(Debug, Clone)]
pub struct EvalCorpusConfig {
    /// Objective to score the aggregate dashboard (drives `score` only).
    pub objective: Objective,
    pub key_sensitivity: f32,
    pub melody: MelodyMode,
    /// Rhythm-quantization engine for the melody.
    pub quantizer: QuantizerEngine,
    /// Optional explicit path to `melody_quantizer.onnx`.
    pub quantizer_model_path: Option<PathBuf>,
    pub use_stems: bool,
    /// Chord onset tolerance in beats.
    pub onset_tolerance_beats: f32,
    /// Note onset tolerance in beats.
    pub note_tolerance_beats: f32,
    /// Per-track fixed BPM overrides (track stem -> bpm).
    pub bpm_overrides: HashMap<String, f32>,
    pub model_dir: Option<PathBuf>,
}

impl Default for EvalCorpusConfig {
    fn default() -> Self {
        Self {
            objective: Objective::Balanced,
            key_sensitivity: 0.23,
            melody: MelodyMode::Heuristic,
            quantizer: QuantizerEngine::default(),
            quantizer_model_path: None,
            use_stems: false,
            onset_tolerance_beats: 0.5,
            note_tolerance_beats: 0.5,
            bpm_overrides: HashMap::new(),
            model_dir: None,
        }
    }
}

/// One track's evaluation results.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackResult {
    pub track: String,
    pub audio: String,
    pub reference: String,
    pub bpm: Option<f32>,
    pub beats_per_bar: u32,
    pub transcribed_notes: usize,
    pub transcribed_chords: usize,
    pub reference_chords: usize,
    pub pitch_accuracy: f32,
    pub note_accuracy: f32,
    pub recall: f32,
    pub mean_onset_error_beats: Option<f32>,
    pub mean_duration_error: Option<f32>,
    pub root_match_rate: f32,
    pub exact_match_rate: f32,
    pub root_coverage: f32,
    pub in_bar_rate: f32,
    pub bar_count_score: f32,
    pub score: f32,
    pub error: Option<String>,
}

/// Aggregated corpus report (mean across tracks, error tracks flagged).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CorpusReport {
    pub track_count: usize,
    pub ok_tracks: usize,
    pub mean_pitch_accuracy: f32,
    pub mean_note_accuracy: f32,
    pub mean_recall: f32,
    pub mean_onset_error_beats: Option<f32>,
    pub mean_duration_error: Option<f32>,
    pub mean_root_match_rate: f32,
    pub mean_exact_match_rate: f32,
    pub mean_root_coverage: f32,
    pub mean_in_bar_rate: f32,
    pub mean_bar_count_score: f32,
    pub mean_score: f32,
    pub tracks: Vec<TrackResult>,
}

/// Run the full pipeline + compare over every labeled pair in `dir`.
pub fn eval_corpus(dir: &Path, cfg: &EvalCorpusConfig) -> Result<CorpusReport> {
    let pairs = crate::tune::discover_pairs(dir)?;
    if pairs.is_empty() {
        return Err(anyhow!(
            "no (audio, reference.musicxml) pairs found in {}",
            dir.display()
        ));
    }
    println!(
        "eval-corpus: {} track(s), objective={:?}, melody={:?}, quantizer={}, stems={}",
        pairs.len(),
        cfg.objective,
        cfg.melody,
        cfg.quantizer.as_str(),
        cfg.use_stems
    );

    let mut tracks: Vec<TrackResult> = Vec::new();
    for (audio, reference) in &pairs {
        let stem = audio
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let bpm = cfg.bpm_overrides.get(&stem).copied();
        match eval_one(audio, reference, cfg, bpm) {
            Ok(mut tr) => {
                let err = tr.error.clone();
                tr.track = stem.clone();
                if err.is_some() {
                    println!("  [error] {}: {:?}", stem, err);
                } else {
                    println!(
                        "  {:<28} root={:.3} exact={:.3} pitch={:.3} note={:.3} recall={:.3} onset={:?}",
                        stem,
                        tr.root_match_rate,
                        tr.exact_match_rate,
                        tr.pitch_accuracy,
                        tr.note_accuracy,
                        tr.recall,
                        tr.mean_onset_error_beats
                            .map(|e| format!("{e:.3}"))
                    );
                }
                tracks.push(tr);
            }
            Err(e) => {
                eprintln!("[eval-corpus] {stem}: {e:#}");
                tracks.push(TrackResult {
                    track: stem,
                    audio: audio.to_string_lossy().into_owned(),
                    reference: reference.to_string_lossy().into_owned(),
                    bpm,
                    beats_per_bar: 0,
                    transcribed_notes: 0,
                    transcribed_chords: 0,
                    reference_chords: 0,
                    pitch_accuracy: 0.0,
                    note_accuracy: 0.0,
                    recall: 0.0,
                    mean_onset_error_beats: None,
                    mean_duration_error: None,
                    root_match_rate: 0.0,
                    exact_match_rate: 0.0,
                    root_coverage: 0.0,
                    in_bar_rate: 0.0,
                    bar_count_score: 0.0,
                    score: 0.0,
                    error: Some(format!("{e:#}")),
                });
            }
        }
    }

    let ok: Vec<&TrackResult> = tracks.iter().filter(|t| t.error.is_none()).collect();
    let n = ok.len().max(1) as f32;
    let mean = |f: &dyn Fn(&TrackResult) -> f32| {
        ok.iter().map(|t| f(t)).sum::<f32>() / n
    };
    let mean_opt = |f: &dyn Fn(&TrackResult) -> Option<f32>| -> Option<f32> {
        let vals: Vec<f32> = ok.iter().filter_map(|t| f(t)).collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f32>() / vals.len() as f32)
        }
    };

    let report = CorpusReport {
        track_count: tracks.len(),
        ok_tracks: ok.len(),
        mean_pitch_accuracy: mean(&|t| t.pitch_accuracy),
        mean_note_accuracy: mean(&|t| t.note_accuracy),
        mean_recall: mean(&|t| t.recall),
        mean_onset_error_beats: mean_opt(&|t| t.mean_onset_error_beats),
        mean_duration_error: mean_opt(&|t| t.mean_duration_error),
        mean_root_match_rate: mean(&|t| t.root_match_rate),
        mean_exact_match_rate: mean(&|t| t.exact_match_rate),
        mean_root_coverage: mean(&|t| t.root_coverage),
        mean_in_bar_rate: mean(&|t| t.in_bar_rate),
        mean_bar_count_score: mean(&|t| t.bar_count_score),
        mean_score: mean(&|t| t.score),
        tracks,
    };
    Ok(report)
}

/// Evaluate a single (audio, reference) pair and return the metrics.
fn eval_one(
    audio: &Path,
    reference: &Path,
    cfg: &EvalCorpusConfig,
    bpm: Option<f32>,
) -> Result<TrackResult> {
    let ref_xml = std::fs::read_to_string(reference)
        .with_context(|| format!("failed to read reference {}", reference.display()))?;
    let ref_notes = parse_musicxml_notes(&ref_xml)?;
    let ref_harmonies = parse_musicxml_harmonies(&ref_xml)?;
    let ref_bpb = parse_beats_per_bar(&ref_xml)?;

    eprintln!("[eval-corpus] analyzing {}...", audio.display());
    let analysis = analyze_audio(audio, cfg.use_stems, cfg.model_dir.as_deref())?;

    let t_opts = TranscribeOptions {
        threshold: key_sensitivity_to_threshold(cfg.key_sensitivity),
        melody_mode: cfg.melody,
        melody_outlier_semitones: 12,
        model_dir: cfg.model_dir.clone(),
    };
    let sheet_opts = SheetOptions {
        title: "eval".to_string(),
        use_stems: cfg.use_stems,
        melody_stems: cfg.use_stems,
        manual_bpm: bpm,
        quantizer: cfg.quantizer,
        quantizer_model_path: cfg.quantizer_model_path.clone(),
        ..Default::default()
    };
    let sheet = generate_sheet_from_analysis(&analysis, &t_opts, &sheet_opts)?;

    let trans_notes = parse_musicxml_notes(&sheet.musicxml)?;
    let note_report = compare_note_lists(&ref_notes, &trans_notes, cfg.note_tolerance_beats);
    let bar = compare_bar_placement(
        &ref_notes,
        &trans_notes,
        ref_bpb,
        sheet.beats_per_bar,
        cfg.note_tolerance_beats,
        0.5,
    );

    let (root_rate, exact_rate, root_coverage, exact_coverage) =
        if ref_harmonies.is_empty() {
            (0.0f32, 0.0f32, 0.0f32, 0.0f32)
        } else {
            let trans_harmonies = parse_musicxml_harmonies(&sheet.musicxml).unwrap_or_default();
            let rep = compare_harmonies(
                &ref_harmonies,
                &trans_harmonies,
                cfg.onset_tolerance_beats,
            );
            let tc = rep.transcription_count.max(1) as f32;
            let rc = rep.reference_count.max(1) as f32;
            (
                rep.root_match_rate,
                rep.exact_match_rate,
                rep.root_match_rate * tc / rc,
                rep.exact_match_rate * tc / rc,
            )
        };

    let melody_sim = crate::tune::melody_similarity_public(&note_report);
    let score = cfg.objective.score(
        melody_sim,
        note_report.pitch_accuracy,
        rate_to_f1(root_rate, root_coverage),
        bar.in_bar_rate,
        rate_to_f1(exact_rate, exact_coverage),
    );

    Ok(TrackResult {
        track: String::new(),
        audio: audio.to_string_lossy().into_owned(),
        reference: reference.to_string_lossy().into_owned(),
        bpm,
        beats_per_bar: sheet.beats_per_bar,
        transcribed_notes: trans_notes.len(),
        transcribed_chords: {
            let h = parse_musicxml_harmonies(&sheet.musicxml).unwrap_or_default();
            h.len()
        },
        reference_chords: ref_harmonies.len(),
        pitch_accuracy: note_report.pitch_accuracy,
        note_accuracy: note_report.note_accuracy,
        recall: note_report.recall,
        mean_onset_error_beats: if note_report.mean_onset_error_beats.is_nan() {
            None
        } else {
            Some(note_report.mean_onset_error_beats)
        },
        mean_duration_error: if note_report.mean_duration_error.is_nan() {
            None
        } else {
            Some(note_report.mean_duration_error)
        },
        root_match_rate: root_rate,
        exact_match_rate: exact_rate,
        root_coverage,
        in_bar_rate: bar.in_bar_rate,
        bar_count_score: bar.bar_count_score,
        score,
        error: None,
    })
}

fn rate_to_f1(rate: f32, coverage: f32) -> f32 {
    if rate > 0.0 && coverage > 0.0 {
        2.0 * rate * coverage / (rate + coverage)
    } else {
        0.0
    }
}

/// Persist the aggregate report as JSON.
pub fn write_report(path: &Path, report: &CorpusReport) -> Result<()> {
    let raw = serde_json::to_string_pretty(report)?;
    std::fs::write(path, raw).with_context(|| format!("failed to write {}", path.display()))
}

/// Print the dashboard table + aggregate row.
pub fn print_report(report: &CorpusReport) {
    println!();
    println!("{:<28} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}", "track", "root", "exact", "pitch", "note", "recall", "onset", "inbar", "bars");
    for t in &report.tracks {
        if t.error.is_some() {
            println!("{:<28} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}", t.track, "-", "-", "-", "-", "-", "-", "-", "-");
            continue;
        }
        println!(
            "{:<28} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>7.3} {:>6.3} {:>6.3}",
            t.track,
            t.root_match_rate,
            t.exact_match_rate,
            t.pitch_accuracy,
            t.note_accuracy,
            t.recall,
            t.mean_onset_error_beats.unwrap_or(f32::NAN),
            t.in_bar_rate,
            t.bar_count_score
        );
    }
    println!("{}", "-".repeat(95));
    println!(
        "{:<28} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>7.3} {:>6.3} {:>6.3}",
        "MEAN",
        report.mean_root_match_rate,
        report.mean_exact_match_rate,
        report.mean_pitch_accuracy,
        report.mean_note_accuracy,
        report.mean_recall,
        report.mean_onset_error_beats.unwrap_or(f32::NAN),
        report.mean_in_bar_rate,
        report.mean_bar_count_score
    );
    println!(
        "({} / {} tracks OK; mean score {:.3})",
        report.ok_tracks, report.track_count, report.mean_score
    );
}

/// Parse a per-track BPM override file. Two formats:
/// - JSON object: `{"Track_Name": 208.0, ...}`
/// - text lines: `<track stem> <bpm>` (space separated)
pub fn parse_bpm_overrides(path: &Path) -> Result<HashMap<String, f32>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        let map: HashMap<String, f32> =
            serde_json::from_str(trimmed).context("invalid JSON bpm override file")?;
        return Ok(map);
    }
    let mut out = HashMap::new();
    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let name = parts.next().unwrap_or_default();
        let bpm = parts.next().and_then(|b| b.parse::<f32>().ok());
        if !name.is_empty() {
            if let Some(bpm) = bpm {
                out.insert(name.to_string(), bpm);
            }
        }
    }
    Ok(out)
}
