//! Headless CLI for the KeyScribe transcription pipeline.
//!
//! Currently Windows-only (used for local debugging / the agentic test loop).
//! On other platforms this binary builds as a no-op stub so CI keeps working.

#[cfg(target_os = "windows")]
mod cli_impl {
    use std::path::PathBuf;

    use anyhow::Context;
    use clap::{Parser, Subcommand};
    use keyscribe_lib::headless::{self, MelodyMode, SheetOptions, TranscribeOptions, TunedConfig};
    use keyscribe_lib::leadsheet::{fit_transitions_from_musicxml_dir, load_learned_transitions};
    use keyscribe_lib::midi;
    use keyscribe_lib::musicxml::musescore_convert;
    use keyscribe_lib::sheet_compare;
    use keyscribe_lib::tune::{self, Objective};

    #[derive(Parser)]
    #[command(
        name = "keyscribe-cli",
        version,
        about = "Headless transcription & sheet-music tool (Windows debug build)."
    )]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }

    #[derive(Subcommand)]
    enum Command {
        /// Transcribe audio to a MusicXML sheet.
        #[command(alias = "transcribe")]
        Sheet {
            /// Input audio file (wav/mp3/flac/ogg/m4a...).
            input: PathBuf,
            /// Output .musicxml path (defaults to <input>.musicxml).
            #[arg(short, long)]
            output: Option<PathBuf>,
            /// Sheet title.
            #[arg(long)]
            title: Option<String>,
            /// Key color sensitivity (0.0-1.0, GUI slider value). Converted
            /// internally to the note probability threshold so the CLI matches
            /// the desktop app for the same setting.
            #[arg(long)]
            key_sensitivity: Option<f32>,
            /// Melody reduction mode: poly, skyline, heuristic.
            #[arg(long, value_enum)]
            melody: Option<MelodyArg>,
            /// Outlier filter in semitones for skyline/heuristic modes.
            #[arg(long, default_value_t = 12)]
            outlier_semitones: u8,
            /// Use grand-staff (piano) engraving instead of a lead sheet.
            #[arg(long)]
            no_leadsheet: bool,
            /// Force single staff.
            #[arg(long)]
            single_staff: bool,
            /// Directory containing basic-pitch.onnx.
            #[arg(long)]
            model_dir: Option<PathBuf>,
            /// Override beat tracking with a fixed BPM grid (4 beats/bar).
            #[arg(long)]
            bpm: Option<f32>,
            /// Separate into stems and transcribe only the melodic stems
            /// (requires models/htdemucs_6s.onnx).
            #[arg(long, num_args = 0..=1, default_missing_value = "true")]
            stems: Option<bool>,
            /// Disable the stem-first melody default: use the full mix as the
            /// melody source instead of the identified melodic stem (which
            /// needs models/htdemucs_6s.onnx).
            #[arg(long)]
            no_stems_melody: bool,
            /// Rhythm-quantization engine for the melody: legacy (rule grid) or
            /// learned (needs models/melody_quantizer.onnx; falls back to the
            /// rule grid when absent).
            #[arg(long)]
            quantizer: Option<String>,
            /// Explicit path to melody_quantizer.onnx (defaults to the model
            /// resolver next to the executable / in the working directory).
            #[arg(long)]
            quantizer_model: Option<PathBuf>,
            /// Disable the post-quantization rhythm merge & coarsening pass
            /// (Tier A1). Coarsening is on by default.
            #[arg(long)]
            no_rhythm_coarsen: bool,
            /// Disable the jazz-extension→7th-chord quality collapse (Tier A2).
            /// Collapse is on by default.
            #[arg(long)]
            no_chord_collapse: bool,
            /// Half-bar chord split threshold 0.0-1.0 (Tier A2 Part 2): bars
            /// whose two whitened halves differ by more than this are emitted
            /// as two chords (one per half). 0 = off (one chord per bar).
            #[arg(long, default_value_t = 0.0)]
            chord_split: f32,
            /// Beat offset within the bar (0.0 = downbeat) at which to sample
            /// notes for the primary chord. Negative (default) uses the legacy
            /// max-simultaneous-notes scan.
            #[arg(long)]
            chord_beat: Option<f32>,
            /// Sample the primary chord at the position in the first half of the
            /// bar with the fewest pitch classes instead of the most (cleanest
            /// moment). Only applies when --chord-beat is negative.
            #[arg(long, num_args = 0..=1, default_missing_value = "true")]
            chord_cleanest: Option<bool>,
            /// Sample the primary chord at the strike moment: the position in
            /// the first half of the bar where the most notes onset together.
            /// Only applies when --chord-beat is negative.
            #[arg(long, num_args = 0..=1, default_missing_value = "true")]
            chord_strike: Option<bool>,
            /// Apply defaults from a config file written by `tune`. Explicit
            /// flags on this command override the config's values.
            #[arg(long)]
            config: Option<PathBuf>,
            /// Also render the MusicXML to audio/pdf via MuseScore CLI.
            /// Value is the output extension: mp3, wav, flac, ogg, pdf.
            #[arg(long)]
            render: Option<String>,
            /// Path to a learned (root, quality) chord-transition matrix
            /// fitted by `fit-transition-matrix`. When absent, it is resolved
            /// automatically from models/transition_matrix.json next to the
            /// executable or in the working directory.
            #[arg(long)]
            transition_matrix: Option<PathBuf>,
        },
        /// Transcribe audio to a MIDI file.
        #[command(alias = "tomidi")]
        Midi {
            input: PathBuf,
            #[arg(short, long)]
            output: Option<PathBuf>,
            /// Key color sensitivity (0.0-1.0, GUI slider value). Converted
            /// internally to the note probability threshold so the CLI matches
            /// the desktop app for the same setting.
            #[arg(long)]
            key_sensitivity: Option<f32>,
            #[arg(long, value_enum)]
            melody: Option<MelodyArg>,
            #[arg(long, default_value_t = 12)]
            outlier_semitones: u8,
            #[arg(long)]
            model_dir: Option<PathBuf>,
            /// Override tempo in BPM (default: inferred from beat tracking).
            #[arg(long)]
            bpm: Option<f32>,
            /// Apply defaults from a config file written by `tune`. Explicit
            /// flags on this command override the config's values.
            #[arg(long)]
            config: Option<PathBuf>,
        },
        /// Render MusicXML/MIDI to audio or PDF via the MuseScore CLI.
        #[command(alias = "synth")]
        Render {
            input: PathBuf,
            #[arg(short, long)]
            output: PathBuf,
        },
        /// Compare two sheets (MusicXML or MIDI) and print accuracy metrics.
        Compare {
            /// Ground-truth / reference sheet.
            reference: PathBuf,
            /// Transcribed sheet.
            transcribed: PathBuf,
            /// Onset tolerance in beats for note matching.
            #[arg(long, default_value_t = 0.25)]
            tolerance: f32,
            /// Emit the report as JSON.
            #[arg(long)]
            json: bool,
        },
        /// Separate an audio file into instrument stems (htdemucs_6s) as WAVs.
        Stems {
            input: PathBuf,
            #[arg(short, long)]
            output: PathBuf,
        },
        /// Train pipeline parameters on labeled audio+musicxml pairs and write
        /// a config file that `sheet`/`midi` load with --config.
        #[command(alias = "train")]
        Tune {
            /// Folder containing <name>.mp3 + <name>.musicxml pairs.
            input: PathBuf,
            /// Output config file (default: keyscribe.tuned.json).
            #[arg(short, long)]
            output: Option<PathBuf>,
            /// Write the full per-evaluation training log here as JSON
            /// (default: <output>.report.json).
            #[arg(long)]
            report: Option<PathBuf>,
            /// Objective to maximize: melody, root, chord, balanced.
            /// Balanced = melody and chords weighted equally (30% melody
            /// similarity + 10% pitch accuracy + 10% bar placement + 35%
            /// chord roots + 15% chord quality).
            #[arg(long, default_value = "balanced")]
            objective: String,
            /// Reduced search grid for quick runs.
            #[arg(long)]
            fast: bool,
            /// Also evaluate stem-separated transcription (needs demucs model).
            #[arg(long)]
            stems: bool,
            /// Chord onset tolerance in beats.
            #[arg(long, default_value_t = 0.5)]
            tolerance: f32,
            /// Fixed tempo override applied to every track (asserts a uniform
            /// tempo across the dataset).
            #[arg(long)]
            bpm: Option<f32>,
            /// Directory containing basic-pitch.onnx.
            #[arg(long)]
            model_dir: Option<PathBuf>,
        },
        /// Generate a built-in test melody MIDI for round-trip testing.
        Maketest {
            #[arg(short, long)]
            output: PathBuf,
            #[arg(long, default_value_t = 100.0)]
            bpm: f32,
        },
        /// Synthesize audio (melody + chord voicings) from an Omnibook
        /// MusicXML, or batch-render every .xml in a directory. The generated
        /// mp3 + the source .xml form a labeled round-trip training pair.
        Omnibook {
            /// Input .xml file, or a directory of .xml files (batch mode).
            input: PathBuf,
            /// Output path (.mp3 or .wav) in single-file mode, or the output
            /// directory in batch mode.
            #[arg(short, long)]
            output: Option<PathBuf>,
            /// Also write the lossless .wav next to each .mp3.
            #[arg(long)]
            keep_wav: bool,
            /// Override the tempo read from the XML (BPM).
            #[arg(long)]
            bpm: Option<f32>,
            /// Melody only, no chord accompaniment.
            #[arg(long)]
            melody_only: bool,
            /// Chords only, no melody.
            #[arg(long)]
            chords_only: bool,
        },
        /// Fit a learned (root, quality) chord-transition matrix from a corpus
        /// of MusicXML lead sheets (e.g. the Omnibook set) and write it for use
        /// by `sheet --transition-matrix`.
        FitTransitionMatrix {
            /// Directory of MusicXML lead sheets (the corpus).
            #[arg(short, long)]
            input: PathBuf,
            /// Output JSON path (default: transition_matrix.json).
            #[arg(short, long)]
            output: Option<PathBuf>,
            /// Laplace smoothing alpha.
            #[arg(long, default_value_t = 0.5)]
            alpha: f32,
            /// Blend weight of the learned cost vs the hand-tuned Viterbi
            /// costs (0.0 = pure hand-tuned, 1.0 = pure learned).
            #[arg(long, default_value_t = 0.65)]
            blend: f32,
        },
        /// Batch evaluation harness: run the full pipeline + compare over every
        /// (audio, reference.musicxml) pair in a directory and print an
        /// aggregated dashboard (plus optional JSON report).
        EvalCorpus {
            /// Directory containing (audio, reference.musicxml) pairs.
            #[arg(short, long)]
            input: PathBuf,
            /// Write the aggregate report here as JSON.
            #[arg(short, long)]
            output: Option<PathBuf>,
            /// Per-track BPM override file: a JSON object {"Track": bpm, ...}
            /// or one `<track> <bpm>` per line. Makes runs reproducible when
            /// the ML beat tracker misfires on a track's meter.
            #[arg(long)]
            bpm_file: Option<PathBuf>,
            /// Key color sensitivity (0.0-1.0).
            #[arg(long)]
            key_sensitivity: Option<f32>,
            /// Melody reduction mode: poly, skyline, heuristic.
            #[arg(long, value_enum)]
            melody: Option<MelodyArg>,
            /// Rhythm-quantization engine: legacy | learned.
            #[arg(long)]
            quantizer: Option<String>,
            /// Explicit path to melody_quantizer.onnx.
            #[arg(long)]
            quantizer_model: Option<PathBuf>,
            /// Separate into stems (requires models/htdemucs_6s.onnx).
            #[arg(long, num_args = 0..=1, default_missing_value = "true")]
            stems: Option<bool>,
            /// Chord onset tolerance in beats.
            #[arg(long, default_value_t = 0.5)]
            tolerance: f32,
            /// Directory containing basic-pitch.onnx.
            #[arg(long)]
            model_dir: Option<PathBuf>,
            /// Objective to score the aggregate: melody, root, chord, balanced.
            #[arg(long, default_value = "balanced")]
            objective: String,
        },
    }

    #[derive(Clone, Copy, clap::ValueEnum)]
    enum MelodyArg {
        Poly,
        Skyline,
        Heuristic,
    }

    impl From<MelodyArg> for MelodyMode {
        fn from(v: MelodyArg) -> Self {
            match v {
                MelodyArg::Poly => MelodyMode::Polyphonic,
                MelodyArg::Skyline => MelodyMode::Skyline,
                MelodyArg::Heuristic => MelodyMode::Heuristic,
            }
        }
    }

    fn parse_melody(s: &str) -> anyhow::Result<MelodyMode> {
        match s.to_ascii_lowercase().as_str() {
            "poly" | "polyphonic" => Ok(MelodyMode::Polyphonic),
            "skyline" => Ok(MelodyMode::Skyline),
            "heuristic" => Ok(MelodyMode::Heuristic),
            _ => anyhow::bail!("unknown melody mode '{s}' (expected poly|skyline|heuristic)"),
        }
    }

    pub fn main() {
        let cli = Cli::parse();
        if let Err(e) = run(cli) {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
    }

    fn run(cli: Cli) -> anyhow::Result<()> {
        match cli.command {
            Command::Sheet {
                input,
                output,
                title,
                key_sensitivity,
                melody,
                outlier_semitones,
                no_leadsheet,
                single_staff,
                model_dir,
                bpm,
                stems,
                chord_beat,
                chord_cleanest,
                chord_strike,
                no_stems_melody,
                quantizer,
                quantizer_model,
                no_rhythm_coarsen,
                no_chord_collapse,
                chord_split,
                config,
                render,
                transition_matrix,
            } => {
                let tuned = config.as_deref().map(TunedConfig::load).transpose()?.unwrap_or_default();
                let key_sensitivity = key_sensitivity.unwrap_or(tuned.key_sensitivity);
                let melody = match melody {
                    Some(m) => m.into(),
                    None => parse_melody(&tuned.melody)?,
                };
                let stems = stems.unwrap_or(tuned.stems);
                let melody_stems = if no_stems_melody {
                    false
                } else {
                    tuned.melody_stems
                };
                let chord_beat = chord_beat.unwrap_or(tuned.chord_beat);
                let chord_cleanest = chord_cleanest.unwrap_or(tuned.chord_cleanest);
                let chord_strike = chord_strike.unwrap_or(tuned.chord_strike);
                let quantizer_engine = match quantizer {
                    Some(q) => keyscribe_lib::leadsheet::QuantizerEngine::parse(&q)
                        .ok_or_else(|| {
                            anyhow::anyhow!("unknown --quantizer '{q}' (expected legacy|learned)")
                        })?,
                    None => keyscribe_lib::leadsheet::QuantizerEngine::parse(&tuned.melody_quantizer)
                        .unwrap_or_default(),
                };
                load_learned_transitions(resolve_transition_matrix(transition_matrix.as_ref()).as_deref());

                let output = output.unwrap_or_else(|| append_ext(&input, "musicxml"));
                let sheet_opts = SheetOptions {
                    title: title
                        .or_else(|| input.file_stem().map(|s| s.to_string_lossy().into_owned()))
                        .unwrap_or_else(|| "Untitled".to_string()),
                    is_lead_sheet: !no_leadsheet,
                    single_staff,
                    manual_bpm: bpm.or(tuned.bpm),
                    use_stems: stems,
                    melody_stems,
                    chord_sample_beat: chord_beat,
                    chord_sample_cleanest: chord_cleanest,
                    chord_sample_strike: chord_strike,
                    quantizer: quantizer_engine,
                    quantizer_model_path: quantizer_model,
                    rhythm_coarsen: !no_rhythm_coarsen,
                    chord_collapse: !no_chord_collapse,
                    chord_split,
                };
                let t_opts = TranscribeOptions {
                    threshold: headless::key_sensitivity_to_threshold(key_sensitivity),
                    melody_mode: melody,
                    melody_outlier_semitones: outlier_semitones,
                    model_dir,
                };
                let result = headless::generate_sheet(&input, &t_opts, &sheet_opts)?;
                write_text(&output, &result.musicxml)?;
                println!(
                    "wrote {} ({} notes, {:.0} bpm, {} beats/bar)",
                    output.display(),
                    result.note_count,
                    result.bpm,
                    result.beats_per_bar
                );
                if let Some(ext) = render {
                    let render_path = append_ext(&input, &ext);
                    musescore_convert(&output, &render_path).map_err(anyhow::Error::msg)?;
                    println!("rendered {}", render_path.display());
                }
            }
            Command::Midi {
                input,
                output,
                key_sensitivity,
                melody,
                outlier_semitones,
                model_dir,
                bpm,
                config,
            } => {
                let tuned = config.as_deref().map(TunedConfig::load).transpose()?.unwrap_or_default();
                let key_sensitivity = key_sensitivity.unwrap_or(tuned.key_sensitivity);
                let melody = match melody {
                    Some(m) => m.into(),
                    None => parse_melody(&tuned.melody)?,
                };
                let t_opts = TranscribeOptions {
                    threshold: headless::key_sensitivity_to_threshold(key_sensitivity),
                    melody_mode: melody,
                    melody_outlier_semitones: outlier_semitones,
                    model_dir,
                };
                let tr = headless::transcribe_notes(&input, &t_opts)?;
                let bpm = bpm.or(tuned.bpm).unwrap_or(tr.beats.bpm.max(30.0));
                let output = output.unwrap_or_else(|| append_ext(&input, "mid"));
                midi::write_midi(&tr.notes, &output, bpm)?;
                println!(
                    "wrote {} ({} notes, {:.0} bpm)",
                    output.display(),
                    tr.notes.len(),
                    bpm
                );
            }
            Command::Tune {
                input,
                output,
                report,
                objective,
                fast,
                stems,
                tolerance,
                bpm,
                model_dir,
            } => {
                let objective = Objective::parse(&objective)?;
                let tune_cfg = tune::TuneConfig {
                    objective,
                    fast,
                    use_stems: stems,
                    onset_tolerance_beats: tolerance,
                    bpm,
                    model_dir,
                };
                let report_obj = tune::tune(&input, &tune_cfg)?;
                tune::print_report(&report_obj);
                let out = output.unwrap_or_else(|| PathBuf::from("keyscribe.tuned.json"));
                report_obj.config.save(&out)?;
                let report_path = report.unwrap_or_else(|| {
                    PathBuf::from(format!("{}.report.json", out.display()))
                });
                tune::write_report(&report_path, &report_obj)?;
                println!();
                println!(
                    "best: {}  ->  wrote {} ({} evaluations logged to {})",
                    report_obj.best.label,
                    out.display(),
                    report_obj.evaluation_count,
                    report_path.display()
                );
                println!(
                    "  key_sensitivity={} chord_beat={} chord_cleanest={} chord_strike={} melody={} stems={} bpm={}",
                    report_obj.config.key_sensitivity,
                    report_obj.config.chord_beat,
                    report_obj.config.chord_cleanest,
                    report_obj.config.chord_strike,
                    report_obj.config.melody,
                    report_obj.config.stems,
                    report_obj
                        .config
                        .bpm
                        .map(|b| format!("{b:.0}"))
                        .unwrap_or_else(|| "auto".to_string())
                );
            }
            Command::Render { input, output } => {
                musescore_convert(&input, &output).map_err(anyhow::Error::msg)?;
                println!("rendered {}", output.display());
            }
            Command::Compare {
                reference,
                transcribed,
                tolerance,
                json,
            } => {
                let ref_notes = sheet_compare::load_notes(&reference)?;
                let trans_notes = sheet_compare::load_notes(&transcribed)?;
                let report = sheet_compare::compare_note_lists(&ref_notes, &trans_notes, tolerance);

                let ref_chords = sheet_compare::load_harmonies(&reference)?;
                let trans_chords = sheet_compare::load_harmonies(&transcribed)?;
                let chord_report = if ref_chords.is_empty() {
                    None
                } else {
                    Some(sheet_compare::compare_harmonies(
                        &ref_chords,
                        &trans_chords,
                        tolerance,
                    ))
                };

                if json {
                    let mut obj = serde_json::to_value(&report)?;
                    if let Some(chords) = &chord_report {
                        obj["chords"] = serde_json::to_value(chords)?;
                    }
                    println!("{}", serde_json::to_string_pretty(&obj)?);
                } else {
                    println!("reference notes: {}", report.reference_note_count);
                    println!("transcribed notes: {}", report.transcription_note_count);
                    println!("pitch accuracy:   {:.3}", report.pitch_accuracy);
                    println!("note accuracy:    {:.3}", report.note_accuracy);
                    println!("recall:           {:.3}", report.recall);
                    if let Some(e) = non_nan(report.mean_onset_error_beats) {
                        println!("mean onset error: {:.3} beats", e);
                    }
                    if let Some(e) = non_nan(report.mean_duration_error) {
                        println!("mean dur error:   {:.3}", e);
                    }
                    if let Some(chords) = &chord_report {
                        println!("");
                        println!("reference chords: {}", chords.reference_count);
                        println!("transcribed chords: {}", chords.transcription_count);
                        println!("chord exact match: {:.3}", chords.exact_match_rate);
                        println!("chord root match:  {:.3}", chords.root_match_rate);
                        if let Some(e) = non_nan(chords.mean_onset_error_beats) {
                            println!("chord onset error: {:.3} beats", e);
                        }
                        if !chords.reference_symbols.is_empty() {
                            println!(
                                "truth chords:      {}",
                                chords.reference_symbols.join(" ")
                            );
                        }
                        if !chords.transcription_symbols.is_empty() {
                            println!(
                                "ours chords:       {}",
                                chords.transcription_symbols.join(" ")
                            );
                        }
                    }
                }
            }
            Command::Stems { input, output } => {
                let files = headless::separate_stems(&input, &output)?;
                for f in files {
                    println!("wrote {}", f.display());
                }
            }
            Command::Maketest { output, bpm } => {
                let notes = test_melody();
                midi::write_midi(&notes, &output, bpm)?;
                println!(
                    "wrote {} ({} notes, {:.0} bpm)",
                    output.display(),
                    notes.len(),
                    bpm
                );
            }
            Command::Omnibook {
                input,
                output,
                keep_wav,
                bpm,
                melody_only,
                chords_only,
            } => {
                use keyscribe_lib::synth::{self, RenderOptions};

                let files: Vec<PathBuf> = if input.is_dir() {
                    let mut v: Vec<PathBuf> = std::fs::read_dir(&input)?
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            p.extension()
                                .map(|x| x.to_string_lossy().eq_ignore_ascii_case("xml"))
                                .unwrap_or(false)
                        })
                        .collect();
                    v.sort();
                    v
                } else {
                    vec![input.clone()]
                };
                if files.is_empty() {
                    anyhow::bail!("no .xml files found in {}", input.display());
                }
                let batch = input.is_dir();

                for f in &files {
                    let xml = std::fs::read_to_string(f)
                        .with_context(|| format!("failed to read {}", f.display()))?;
                    let mut song = synth::parse_musicxml(&xml)?;
                    if let Some(b) = bpm {
                        synth::rescale_tempo(&mut song, b);
                    }

                    let out_path = match &output {
                        Some(o) => {
                            if batch {
                                o.join(format!(
                                    "{}.mp3",
                                    f.file_stem().unwrap_or_default().to_string_lossy()
                                ))
                            } else {
                                let mut p = o.clone();
                                if p.extension().is_none() {
                                    p.set_extension("mp3");
                                }
                                p
                            }
                        }
                        None => {
                            let mut p = f.clone();
                            p.set_extension("mp3");
                            p
                        }
                    };

                    let samples = synth::render_song_partial(
                        &song,
                        &RenderOptions::default(),
                        !chords_only,
                        !melody_only,
                    );
                    let ext = out_path
                        .extension()
                        .map(|e| e.to_string_lossy().to_ascii_lowercase())
                        .unwrap_or_default();
                    if ext == "wav" {
                        synth::write_wav(&out_path, &samples, 44_100)?;
                    } else {
                        synth::write_mp3(&out_path, &samples, 44_100)?;
                    }
                    if keep_wav {
                        synth::write_wav(&out_path.with_extension("wav"), &samples, 44_100)?;
                    }
                    if batch {
                        // Copy the ground-truth sheet next to the audio so
                        // `tune`/`compare` can consume the labeled pairs.
                        std::fs::copy(f, out_path.with_extension("musicxml"))?;
                    }
                    println!(
                        "wrote {} ({} notes, {} chords, {:.0} bpm)",
                        out_path.display(),
                        song.melody.len(),
                        song.chords.len(),
                        song.bpm
                    );
                }
            }
            Command::FitTransitionMatrix { input, output, alpha, blend } => {
                let mut lt = fit_transitions_from_musicxml_dir(&input, alpha)?;
                lt.blend_w = blend.clamp(0.0, 1.0);
                let json = serde_json::to_string_pretty(&lt)?;
                let out = output.unwrap_or_else(|| PathBuf::from("transition_matrix.json"));
                write_text(&out, &json)?;
                println!(
                    "wrote {} ({} states, {} transitions, alpha={}, blend={})",
                    out.display(),
                    lt.states.len(),
                    lt.n_transitions,
                    lt.alpha,
                    lt.blend_w
                );
            }
            Command::EvalCorpus {
                input,
                output,
                bpm_file,
                key_sensitivity,
                melody,
                quantizer,
                quantizer_model,
                stems,
                tolerance,
                model_dir,
                objective,
            } => {
                use keyscribe_lib::eval_corpus::{self, EvalCorpusConfig};
                use keyscribe_lib::tune::Objective;

                let objective = Objective::parse(&objective)?;
                let bpm_overrides = bpm_file
                    .as_deref()
                    .map(eval_corpus::parse_bpm_overrides)
                    .transpose()?
                    .unwrap_or_default();
                let cfg = EvalCorpusConfig {
                    objective,
                    key_sensitivity: key_sensitivity.unwrap_or(0.23),
                    melody: melody.map(Into::into).unwrap_or(MelodyArg::Heuristic.into()),
                    quantizer: match quantizer {
                        Some(q) => keyscribe_lib::leadsheet::QuantizerEngine::parse(&q)
                            .ok_or_else(|| anyhow::anyhow!("unknown --quantizer '{q}' (expected legacy|learned)"))?,
                        None => keyscribe_lib::leadsheet::QuantizerEngine::default(),
                    },
                    quantizer_model_path: quantizer_model,
                    use_stems: stems.unwrap_or(false),
                    onset_tolerance_beats: tolerance,
                    note_tolerance_beats: tolerance,
                    bpm_overrides,
                    model_dir,
                };
                let report = eval_corpus::eval_corpus(&input, &cfg)?;
                eval_corpus::print_report(&report);
                if let Some(out) = output {
                    eval_corpus::write_report(&out, &report)?;
                    println!("\nwrote {}", out.display());
                }
            }
        }
        Ok(())
    }

    /// Resolve the learned transition matrix path: an explicit flag wins, else
    /// models/transition_matrix.json (or the bare file) next to the exe or in
    /// the working directory.
    fn resolve_transition_matrix(explicit: Option<&PathBuf>) -> Option<PathBuf> {
        if let Some(p) = explicit {
            return Some(p.clone());
        }
        let mut candidates = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("models").join("transition_matrix.json"));
                candidates.push(dir.join("transition_matrix.json"));
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("models").join("transition_matrix.json"));
            candidates.push(cwd.join("transition_matrix.json"));
        }
        candidates.into_iter().find(|p| p.is_file())
    }

    fn non_nan(v: f32) -> Option<f32> {
        if v.is_nan() {
            None
        } else {
            Some(v)
        }
    }

    fn write_text(path: &PathBuf, text: &str) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, text)?;
        Ok(())
    }

    fn append_ext(path: &PathBuf, ext: &str) -> PathBuf {
        let mut os = path.clone();
        os.set_extension(ext);
        os
    }

    /// Built-in "Twinkle Twinkle Little Star" melody for round-trip tests.
    fn test_melody() -> Vec<keyscribe_lib::leadsheet::NoteEvent> {
        // C4 G4 A4 F4 E4 D4 note numbers.
        let phrase = [60u8, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60];
        let mut notes = Vec::new();
        let mut id: u32 = 0;
        for rep in 0..2 {
            let base = rep as f32 * phrase.len() as f32 * 0.6;
            for (i, &pitch) in phrase.iter().enumerate() {
                let start = base + i as f32 * 0.6;
                notes.push(keyscribe_lib::leadsheet::NoteEvent {
                    id,
                    pitch,
                    start_time: start,
                    end_time: start + 0.5,
                    velocity: 100,
                    channel: None,
                    is_rearticulation: false,
                });
                id += 1;
            }
        }
        notes
    }
}

#[cfg(target_os = "windows")]
fn main() {
    cli_impl::main();
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("keyscribe-cli is currently Windows-only (used for local debugging).");
    std::process::exit(1);
}
