//! Dataset-driven parameter tuning for the transcription pipeline.
//!
//! Given a folder of `(audio, reference.musicxml)` pairs, this sweeps the
//! pipeline's cheap parameters (note threshold via key sensitivity, chord
//! sampling bias, melody reduction, stem usage) against the ground truth and
//! writes the best combination to a config file that `sheet`/`midi` load via
//! `--config`. Basic Pitch + beat tracking run once per track; every
//! evaluation after that only re-runs note extraction and lead-sheet building.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::headless::{
    analyze_audio, generate_sheet_from_analysis, key_sensitivity_to_threshold, MelodyMode,
    SheetOptions, TranscribeOptions, TunedConfig,
};
use crate::sheet_compare::{
    compare_bar_placement, compare_harmonies, compare_note_lists, parse_beats_per_bar,
    parse_musicxml_harmonies, parse_musicxml_notes,
};

/// How the primary chord is sampled within each bar.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub enum ChordSampling {
    /// Legacy max-simultaneous-notes scan.
    Legacy,
    /// Fewest-pitch-classes scan (cleanest moment).
    Cleanest,
    /// Max simultaneous onsets (strike moment).
    Strike,
    /// Fixed beat offset within the bar (0.0 = downbeat).
    Beat(f32),
}

/// Objective to maximize across the dataset. Scores are F1-style so a solution
/// that matches few notes/chords on few detections can't game the metric.
/// Evaluation dimensions: melody similarity (note values + durations +
/// timing), melody pitch accuracy (a reliable signal the rhythm-weighted
/// similarity compresses near zero), chord root notes, downbeat/bar
/// placement, and chord quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Melody similarity only (note values + durations + timing), with a
    /// pitch-accuracy vote.
    Melody,
    /// Chord root-match F1 only.
    Root,
    /// 70% chord root F1 + 30% chord quality (exact) F1.
    Chord,
    /// Melody (30% similarity + 10% pitch + 10% bar placement) and chords
    /// (35% root + 15% quality), weighted equally.
    Balanced,
}

impl Objective {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "melody" => Ok(Objective::Melody),
            "root" => Ok(Objective::Root),
            "chord" => Ok(Objective::Chord),
            "balanced" => Ok(Objective::Balanced),
            _ => Err(anyhow!(
                "unknown objective '{s}' (expected melody|root|chord|balanced)"
            )),
        }
    }

    fn score(&self, melody: f32, pitch: f32, root: f32, bar: f32, quality: f32) -> f32 {
        match self {
            // The timing-weighted melody_sim is compressed near zero by the
            // rhythm gap, so the reliable pitch_accuracy gets a direct vote
            // — otherwise the search trades a readable melody for chords.
            Objective::Melody => 0.7 * melody + 0.3 * pitch,
            Objective::Root => root,
            Objective::Chord => 0.7 * root + 0.3 * quality,
            // Melody and chords weighted equally (50/50): melody + pitch +
            // bar placement on one side, chord roots + quality on the other.
            Objective::Balanced => {
                0.30 * melody + 0.10 * pitch + 0.10 * bar + 0.35 * root + 0.15 * quality
            }
        }
    }
}

/// F1 combining detection precision with coverage against the reference, so
/// trivial single-note/chord detections are heavily penalised.
fn f1(precision: f32, recall: f32) -> f32 {
    if precision > 0.0 && recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Melody similarity: note values (pitch+onset F1 with coverage) plus onset
/// and duration timing accuracy. Timing credits scale WITH note F1 — a few
/// stray matches can't inflate the score via perfect timing.
fn melody_similarity(note_report: &crate::sheet_compare::CompareReport) -> f32 {
    let note_f1 = f1(note_report.note_accuracy, note_report.recall);
    if note_f1 <= 0.0 {
        return 0.0;
    }
    let onset_score = if note_report.mean_onset_error_beats.is_nan() {
        0.0
    } else {
        (1.0 - (note_report.mean_onset_error_beats / 0.5).min(1.0)).max(0.0)
    };
    let dur_score = if note_report.mean_duration_error.is_nan() {
        0.0
    } else {
        (1.0 - (note_report.mean_duration_error / 1.0).min(1.0)).max(0.0)
    };
    note_f1 * (0.7 + 0.15 * onset_score + 0.15 * dur_score)
}

#[derive(Debug, Clone, Copy)]
struct Combo {
    key_sensitivity: f32,
    chord: ChordSampling,
    melody: MelodyMode,
}

impl Combo {
    fn label(&self) -> String {
        format!(
            "sens={:.2}/{}/{}",
            self.key_sensitivity,
            chord_label(self.chord),
            melody_label(self.melody)
        )
    }
}

/// One evaluated configuration on one (track, stem-mode). These are logged in
/// full so the training data can be inspected later to find where the pipeline
/// is weak (low coverage, low precision, wrong grid, etc.).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Evaluation {
    pub track: String,
    pub reference: String,
    pub stems: bool,
    pub bpm: Option<f32>,
    pub key_sensitivity: f32,
    pub threshold: f32,
    pub chord_sampling: String,
    pub melody: String,
    pub transcribed_notes: usize,
    pub transcribed_chords: usize,
    pub reference_chords: usize,
    /// Melody similarity (note values + durations + timing), 0..1.
    pub melody_sim: f32,
    /// Chord root-match F1 (precision vs coverage).
    pub root_f1: f32,
    /// Chord exact-match F1 (root+quality).
    pub exact_f1: f32,
    /// Downbeat / bar-placement: in-bar position agreement among matched notes.
    pub bar_placement: f32,
    /// Downbeat density agreement (1 - |bars| mismatch, catches half/double tempo).
    pub bar_count_score: f32,
    /// Raw chord root-match rate (matched / detected).
    pub root_rate: f32,
    /// Raw chord exact-match rate.
    pub exact_rate: f32,
    /// Root-match coverage: fraction of reference chords detected.
    pub root_coverage: f32,
    pub pitch_accuracy: f32,
    pub note_accuracy: f32,
    /// Objective score (drives ranking).
    pub score: f32,
}

/// Aggregated score of one combo across the whole dataset.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComboScore {
    pub label: String,
    pub key_sensitivity: f32,
    pub chord: ChordSampling,
    pub melody: MelodyMode,
    pub stems: bool,
    pub bpm: Option<f32>,
    pub root_match: f32,
    pub exact_match: f32,
    /// Root-match coverage: fraction of the reference chords whose root was
    /// detected at all.
    pub root_coverage: f32,
    /// Melody similarity (note values + durations + timing).
    pub melody_sim: f32,
    /// Downbeat / bar-placement agreement.
    pub bar_placement: f32,
    pub pitch_accuracy: f32,
    pub note_accuracy: f32,
    pub score: f32,
    pub transcribed_notes: usize,
    pub chord_count: usize,
}

#[derive(Debug, Clone)]
pub struct TuneConfig {
    pub objective: Objective,
    pub fast: bool,
    pub use_stems: bool,
    pub onset_tolerance_beats: f32,
    /// Fixed tempo override applied to every track (asserts a uniform tempo
    /// across the dataset). The search also tries half/double of each track's
    /// inferred BPM, plus this value when given.
    pub bpm: Option<f32>,
    pub model_dir: Option<PathBuf>,
}

#[derive(Debug, serde::Serialize)]
pub struct TuneReport {
    pub ranked: Vec<ComboScore>,
    pub best: ComboScore,
    pub track_count: usize,
    pub evaluation_count: usize,
    pub config: TunedConfig,
    /// Raw per-evaluation log, for post-hoc analysis.
    pub evaluations: Vec<Evaluation>,
}

fn sensitivities(fast: bool) -> Vec<f32> {
    if fast {
        vec![0.1, 0.2, 0.23, 0.3, 0.4]
    } else {
        vec![0.1, 0.15, 0.2, 0.23, 0.25, 0.3, 0.4, 0.5]
    }
}

fn chord_samplings(fast: bool) -> Vec<ChordSampling> {
    if fast {
        vec![
            ChordSampling::Legacy,
            ChordSampling::Cleanest,
            ChordSampling::Strike,
            ChordSampling::Beat(0.0),
        ]
    } else {
        vec![
            ChordSampling::Legacy,
            ChordSampling::Cleanest,
            ChordSampling::Strike,
            ChordSampling::Beat(0.0),
            ChordSampling::Beat(0.25),
            ChordSampling::Beat(0.5),
        ]
    }
}

fn melodies() -> Vec<MelodyMode> {
    vec![
        MelodyMode::Polyphonic,
        MelodyMode::Skyline,
        MelodyMode::Heuristic,
    ]
}

fn chord_label(c: ChordSampling) -> String {
    match c {
        ChordSampling::Legacy => "legacy".to_string(),
        ChordSampling::Cleanest => "cleanest".to_string(),
        ChordSampling::Strike => "strike".to_string(),
        ChordSampling::Beat(b) => format!("beat={b:.2}"),
    }
}

fn melody_label(m: MelodyMode) -> &'static str {
    match m {
        MelodyMode::Polyphonic => "poly",
        MelodyMode::Skyline => "skyline",
        MelodyMode::Heuristic => "heuristic",
    }
}

fn bpm_label(b: Option<f32>) -> String {
    match b {
        Some(v) => format!("bpm={v:.0}"),
        None => "bpm=auto".to_string(),
    }
}

/// Candidate tempo grids for a track: the inferred BPM (None), the doubled
/// tracker BPM (the tracker commonly returns half-time) and the user's
/// `--bpm` override when given, each with a fine ±1.2% refinement sweep —
/// the tracker's estimate is coarse (e.g. 111 bpm for a real 225), and a
/// fixed grid a few percent off accumulates a beat of drift over a long
/// track, wrecking bar-aligned chord sampling. The half-time candidate is
/// deliberately NOT searched: a slower grid coarsens quantization and can
/// game beat-space metrics.
fn bpm_candidates(beats: &crate::leadsheet::CrossValidatedBeats, override_bpm: Option<f32>) -> Vec<Option<f32>> {
    let mut opts: Vec<Option<f32>> = vec![None];
    let mut bases: Vec<f32> = Vec::new();
    if let Some(b) = override_bpm {
        bases.push(b.clamp(30.0, 400.0));
    }
    bases.push((beats.bpm * 2.0).clamp(30.0, 400.0));
    for base in bases {
        for &f in &[1.0, 0.988, 1.012] {
            let cand = ((base * f) * 2.0).round() / 2.0;
            if !opts.iter().any(|o| o.is_some_and(|c| (c - cand).abs() < 0.5)) {
                opts.push(Some(cand));
            }
        }
    }
    opts
}

fn build_combos(fast: bool) -> Vec<Combo> {
    let mut out = Vec::new();
    for &key_sensitivity in &sensitivities(fast) {
        for &chord in &chord_samplings(fast) {
            for &melody in &melodies() {
                out.push(Combo {
                    key_sensitivity,
                    chord,
                    melody,
                });
            }
        }
    }
    out
}

/// Find `(audio, reference.musicxml)` pairs in a directory: every audio file
/// that has a sibling with the same stem and a `.musicxml` extension.
pub fn discover_pairs(dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut pairs = Vec::new();
    for entry in entries {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
        else {
            continue;
        };
        if !matches!(ext.as_str(), "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aiff" | "aif") {
            continue;
        }
        let reference = path.with_extension("musicxml");
        if reference.is_file() {
            pairs.push((path, reference));
        } else {
            eprintln!(
                "[tune] skipping {} (no sibling .musicxml reference)",
                path.display()
            );
        }
    }
    Ok(pairs)
}

fn sheet_opts_for(combo: &Combo, title: &str, use_stems: bool) -> SheetOptions {
    let mut s = SheetOptions {
        title: title.to_string(),
        use_stems,
        // Keep the tune sweep binary: either stems entirely (melody from the
        // identified stem + beat tracking from drums/bass) or not at all.
        // This mirrors what the CLI does with `--stems` / the tuned config.
        melody_stems: use_stems,
        ..Default::default()
    };
    match combo.chord {
        ChordSampling::Legacy => {}
        ChordSampling::Cleanest => s.chord_sample_cleanest = true,
        ChordSampling::Strike => s.chord_sample_strike = true,
        ChordSampling::Beat(b) => s.chord_sample_beat = b,
    }
    s
}

/// Train the pipeline parameters on the labeled dataset and return the ranked
/// results plus the config for the best combination.
pub fn tune(dir: &Path, cfg: &TuneConfig) -> Result<TuneReport> {
    let pairs = discover_pairs(dir)?;
    if pairs.is_empty() {
        return Err(anyhow!(
            "no (audio, reference.musicxml) pairs found in {}",
            dir.display()
        ));
    }
    println!("tuning on {} track(s):", pairs.len());
    for (audio, reference) in &pairs {
        println!("  {}  <->  {}", audio.display(), reference.display());
    }

    let combos = build_combos(cfg.fast);
    // Candidate tempo grids per track: auto + the doubled-tracker base refined
    // at x1.00/x0.988/x1.012, plus the same for the --bpm override when given.
    let bpm_candidates_count = if cfg.bpm.is_some() { 7 } else { 4 };
    let stems_variants: Vec<bool> = if cfg.use_stems {
        vec![false, true]
    } else {
        vec![false]
    };
    println!(
        "search space: {} sensitivity x {} sampling x {} melody x {} bpm x {} stem-mode = {} combos per track",
        sensitivities(cfg.fast).len(),
        chord_samplings(cfg.fast).len(),
        melodies().len(),
        bpm_candidates_count,
        stems_variants.len(),
        combos.len() * stems_variants.len() * bpm_candidates_count
    );

    let mut totals: HashMap<String, ComboScore> = HashMap::new();
    let mut evaluations: Vec<Evaluation> = Vec::new();
    let mut evaluation_count = 0usize;

    for (audio, reference) in &pairs {
        let ref_xml = std::fs::read_to_string(reference)
            .with_context(|| format!("failed to read reference {}", reference.display()))?;
        let ref_notes = parse_musicxml_notes(&ref_xml)?;
        let ref_harmonies = parse_musicxml_harmonies(&ref_xml)?;
        let ref_bpb = parse_beats_per_bar(&ref_xml)?;

        for use_stems in &stems_variants {
            eprintln!(
                "[tune] analyzing {} (stems={})...",
                audio.display(),
                use_stems
            );
            let analysis = analyze_audio(audio, *use_stems, cfg.model_dir.as_deref())?;

            for bpm in bpm_candidates(&analysis.beats, cfg.bpm) {
                for combo in &combos {
                    let t_opts = TranscribeOptions {
                        threshold: key_sensitivity_to_threshold(combo.key_sensitivity),
                        melody_mode: combo.melody,
                        melody_outlier_semitones: 12,
                        model_dir: cfg.model_dir.clone(),
                    };
                    let mut sheet_opts = sheet_opts_for(combo, "tune", *use_stems);
                    sheet_opts.manual_bpm = bpm;
                    let Ok(sheet) =
                        generate_sheet_from_analysis(&analysis, &t_opts, &sheet_opts)
                    else {
                        continue;
                    };
                    let Ok(trans_notes) = parse_musicxml_notes(&sheet.musicxml) else {
                        continue;
                    };
                    let note_report = compare_note_lists(&ref_notes, &trans_notes, 0.5);
                    let melody_sim = melody_similarity(&note_report);
                    let bar = compare_bar_placement(
                        &ref_notes,
                        &trans_notes,
                        ref_bpb,
                        sheet.beats_per_bar,
                        0.5,
                        0.5,
                    );
                    let (root_f1, exact_f1, root_rate, exact_rate, root_cov, chord_count) =
                        if ref_harmonies.is_empty() {
                            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize)
                        } else {
                            let trans_harmonies =
                                parse_musicxml_harmonies(&sheet.musicxml).unwrap_or_default();
                            let rep = compare_harmonies(
                                &ref_harmonies,
                                &trans_harmonies,
                                cfg.onset_tolerance_beats,
                            );
                            let matched_root =
                                rep.root_match_rate * rep.transcription_count.max(1) as f32;
                            let matched_exact =
                                rep.exact_match_rate * rep.transcription_count.max(1) as f32;
                            let root_recall = matched_root / rep.reference_count.max(1) as f32;
                            let exact_recall = matched_exact / rep.reference_count.max(1) as f32;
                            (
                                f1(rep.root_match_rate, root_recall),
                                f1(rep.exact_match_rate, exact_recall),
                                rep.root_match_rate,
                                rep.exact_match_rate,
                                root_recall,
                                rep.transcription_count,
                            )
                        };
                    let score = cfg.objective.score(
                        melody_sim,
                        note_report.pitch_accuracy,
                        root_f1,
                        bar.in_bar_rate,
                        exact_f1,
                    );

                    let key = if *use_stems {
                        format!("stems+{}/{}", combo.label(), bpm_label(bpm))
                    } else {
                        format!("{}/{}", combo.label(), bpm_label(bpm))
                    };
                    let entry = totals.entry(key.clone()).or_insert_with(|| ComboScore {
                        label: key.clone(),
                        key_sensitivity: combo.key_sensitivity,
                        chord: combo.chord,
                        melody: combo.melody,
                        stems: *use_stems,
                        bpm,
                        root_match: 0.0,
                        exact_match: 0.0,
                        root_coverage: 0.0,
                        melody_sim: 0.0,
                        bar_placement: 0.0,
                        pitch_accuracy: 0.0,
                        note_accuracy: 0.0,
                        score: 0.0,
                        transcribed_notes: 0,
                        chord_count: 0,
                    });
                    entry.root_match += root_f1;
                    entry.exact_match += exact_f1;
                    entry.root_coverage += root_cov;
                    entry.melody_sim += melody_sim;
                    entry.bar_placement += bar.in_bar_rate;
                    entry.pitch_accuracy += note_report.pitch_accuracy;
                    entry.note_accuracy += note_report.note_accuracy;
                    entry.transcribed_notes += note_report.transcription_note_count;
                    entry.chord_count += chord_count;
                    entry.score += score;
                    evaluation_count += 1;

                    evaluations.push(Evaluation {
                        track: audio.to_string_lossy().into_owned(),
                        reference: reference.to_string_lossy().into_owned(),
                        stems: *use_stems,
                        bpm,
                        key_sensitivity: combo.key_sensitivity,
                        threshold: key_sensitivity_to_threshold(combo.key_sensitivity),
                        chord_sampling: chord_label(combo.chord),
                        melody: melody_label(combo.melody).to_string(),
                        transcribed_notes: trans_notes.len(),
                        transcribed_chords: chord_count,
                        reference_chords: ref_harmonies.len(),
                        melody_sim,
                        root_f1,
                        exact_f1,
                        bar_placement: bar.in_bar_rate,
                        bar_count_score: bar.bar_count_score,
                        root_rate,
                        exact_rate,
                        root_coverage: root_cov,
                        pitch_accuracy: note_report.pitch_accuracy,
                        note_accuracy: note_report.note_accuracy,
                        score,
                    });
                }
            }
        }
    }

    let track_samples = (pairs.len() * stems_variants.len()) as f32;
    let mut ranked: Vec<ComboScore> = totals.into_values().collect();
    for s in ranked.iter_mut() {
        s.root_match /= track_samples;
        s.exact_match /= track_samples;
        s.root_coverage /= track_samples;
        s.melody_sim /= track_samples;
        s.bar_placement /= track_samples;
        s.pitch_accuracy /= track_samples;
        s.note_accuracy /= track_samples;
        s.score /= track_samples;
    }
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = ranked
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("no valid evaluations produced a score"))?;
    let config = TunedConfig {
        key_sensitivity: best.key_sensitivity,
        chord_beat: match best.chord {
            ChordSampling::Beat(b) => b,
            _ => -1.0,
        },
        chord_cleanest: best.chord == ChordSampling::Cleanest,
        chord_strike: best.chord == ChordSampling::Strike,
        melody: melody_label(best.melody).to_string(),
        stems: best.stems,
        melody_stems: best.stems,
        melody_quantizer: "learned".to_string(),
        bpm: best.bpm,
    };

    Ok(TuneReport {
        ranked,
        best,
        track_count: pairs.len(),
        evaluation_count,
        config,
        evaluations,
    })
}

/// Persist the full evaluation log to a JSON file for post-hoc analysis.
pub fn write_report(path: &Path, report: &TuneReport) -> Result<()> {
    let raw = serde_json::to_string_pretty(report)?;
    std::fs::write(path, raw).with_context(|| format!("failed to write report {}", path.display()))
}

/// Print the ranked table to stdout.
pub fn print_report(report: &TuneReport) {
    println!();
    println!("rank  score   melody  bar     rootF1  quality cov     notes   chords  combo");
    for (i, s) in report.ranked.iter().enumerate().take(20) {
        println!(
            "{:>4} {:.3}   {:.3}   {:.3}   {:.3}   {:.3}   {:.3}   {:>6}   {:>6}   {}",
            i + 1,
            s.score,
            s.melody_sim,
            s.bar_placement,
            s.root_match,
            s.exact_match,
            s.root_coverage,
            s.transcribed_notes,
            s.chord_count,
            s.label
        );
    }
}
