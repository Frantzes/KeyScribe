# Chord Symbol Detection Plan — Learned 251-Class Decoder + Learned Viterbi

Status: **partial — the learned Viterbi transition matrix is implemented and
live in the legacy engine; the 251-class ONNX chord head is still pending.**
Companion to `HEADLESS.md`.

## What is implemented (2026-08)

The genuinely-novel piece — **fitting the Viterbi transition matrix to a chord
corpus** — shipped ahead of the model head, scoped to the legacy engine's
`(root, quality)` emission space:

- `src/leadsheet/transition.rs` — `fit_transitions_from_musicxml_dir` counts
  chord adjacencies from a directory of MusicXML lead sheets, Laplace-smooths
  them, and writes `transition_matrix.json`. `learned_transitions()` /
  `load_learned_transitions` keep a process-wide cache used by the Viterbi.
- `keyscribe-cli fit-transition-matrix -i <corpus> -o <out>` (with `--alpha`,
  `--blend`); `sheet --transition-matrix <path>` or auto-resolution of
  `models/transition_matrix.json`.
- `viterbi_best_path` (`harmony.rs:647`) blends the hand-tuned root/quality
  costs with the learned joint probabilities (default blend 0.65) for every
  (source, target) pair both observed in the corpus; unobserved states fall
  back to the hand-tuned table, so a sparse corpus can't dominate.
- Fitted on the 58-track Omnibox corpus (50 songs parsed, 4464 chords, 4414
  transitions, 45 observed (root, quality) states — all Real-Book triads/7ths).
  Top learned moves are textbook jazz: `ii-7→V7` (D-→G7 P=0.78), `V7→i`
  (A7→D- P=0.55), `ii∅→V7` (E-7b5→A7 P=0.45), `V7→IV` back-cycling.
  Measured: Confirmation chord root 0.18 → 0.30 (emission/bass/key work) →
  **0.34** (learned transitions); exact 0.02→0.03; neutral (no regression) on
  Ornithology/Donna Lee.

Still pending: the **251-class ONNX chord head** (emission) that replaces the
27-template `soft_contrast` matcher — the plan below (Phases 0–8) remains
accurate for that.

## Objective

Replace the hand-crafted chord detector in `src/leadsheet/harmony.rs`
(`detect_chords_from_timeline:490`) with a **pre-trained PyTorch chord
recognizer (BTC/ChordFormer-class, exported to ONNX)** that emits the standard
**251-class maj/min/7/bass vocabulary**, decoded by a **Viterbi whose
transition matrix is fit from a real chord corpus** of jazz-standard backing
tracks (iRealPro-style) plus their MusicXML. Trains in Python, runs in Rust via
the existing `ort` ONNX runtime. Everything else (BeatThis!, Demucs, Basic
Pitch AMT, MusicXML writer, `compare`, `tune`) stays unchanged.

### Decisions locked

1. **Chord model input** — raw **full-mix mono audio** (no Demucs pre-separation
   for the chord head). Simpler, faster, matches the BTC/ChordFormer regime.
2. **Vocabulary** — the standard **251-class Harte maj/min/7/bass** universe.
   Jazz-extended qualities the 251 set cannot represent
   (maj7Δ13, -13, -11, dim7, etc.) are **collapsed to the nearest 7th-class**
   (Δ13 → maj7, -13 → min7, G13 → dom7, dim7 → dim, …). The exact-match rate on
   jazz-extended reference chords will *correctly* fall (a measurable gap); root
   F1 should rise. No second-stage extension classifier in this plan.
3. **Transition-matrix corpus** — **jazz-standard backing tracks + MusicXML**
   (iRealPro-style). Acquisition is an explicit sub-task (Phase 1.3); the model
   weights and corpus counts are the licensed assets, the code is corpus-agnostic.
4. **Engine default** — **`learned` becomes the default when the chord ONNX and
   the transition-matrix file are both present on the model path**; falls back to
   `legacy` (the current 27-template detector) so the test suite, the round-trip
   script, and zero-model builds keep working unchanged.

---

## Background — where KeyScribe sits vs SOTA (2022–2026)

The hand-crafted detector is the weakest stage of the pipeline. Measured on
`tests/data/Pretty standard.mp3` after the recent contrast/Viterbi work:
chord root F1 **0.200** (was 0.062), exact F1 **0.200** (was 0.000). Public
learned chord recognizers on the standard Beatles benchmark report maj/min
accuracy **~0.83–0.85** (BTC, ISMIR 2019) and the large-vocabulary
ChordFormer (2025) reports **+2% frame-wise / +6% class-wise** over prior SOTA.
Headline relevant SOTA:

- **BTC** (`arXiv:1907.02698`, 2019) — bi-directional Transformer for chords;
  still the most-cited **open teacher** (used as the teacher in the 2026
  pseudo-labelling paper `arXiv:2602.19778`).
- **ChordFormer** (`arXiv:2502.11840`, 2025) — Conformer for large-vocabulary
  ACE (triads + bass + sevenths); +6% class-wise, long-tail-aware.
- **"From Discord to Harmony"** (`arXiv:2509.01588`, ISMIR 2025) — Conformer ACE
  with consonance-based label smoothing.
- **BMACE** (`arXiv:2601.02101`, ISMIR LBD 2024) — bidirectional Mamba (SSM),
  SOTA-class with fewer params.
- **LLM-CoT ACE** (`arXiv:2509.18700`, 2025) — GPT-4o over a
  separation+beats+chord cascade; **validates our cascade architecture**, +1–2.77%
  MIREX over the underlying ACE tools.
- **Pseudo-labeling + KD** (`arXiv:2602.19778`, DAFx 2026) — distils BTC over
  1000+ hrs of unlabeled audio; **Stage-2 student beats both the supervised
  baseline and the teacher**, with the biggest gains on **rare chord qualities**
  — our regime. Confirms open-weight pretrained models are more accessible than
  their training data.
- **End-to-end A2S** — SheetSage2Kern (`arXiv:2608.06165`, 2026, 4.98% SER
  classical / 20.92% SER pop) and Rubato/Dual-Eval (`arXiv:2608.04511`, 2026).
  **The latter directly measured that the MIDI→score stage, not the AMT
  front-end, dominates notation quality** — so the chord decoder is the right
  place to invest, ahead of upgrading Basic Pitch.
- **BeatThis!** (`arXiv:2407.21658`, ISMIR 2024, already in KeyScribe) —
  confirmed SOTA-class; MIT-licensed weights loadable via `torch.hub`.
- **SSL front-end upgrade option** — MuQ (`arXiv:2501.01108`, 2025, 0.9K hrs
  open data) and MERT (`arXiv:2306.00107`, ICLR 2024). Frozen MuQ is what
  SheetSage2Kern uses. **Not in this plan**; flagged as Tier 2 follow-up.

### Genuinely novel piece

No 2023–2026 paper I found **explicitly fits the Viterbi transition matrix to a
chord corpus and plugs it back into a Viterbi decoder.** ChordFormer/BMACE learn
the transition function *implicitly* via attention/SSM context; the explicit
"learn the transition matrix from data" approach (replacing our hand-tuned
12×12 root-interval table at `harmony.rs:663-679`) is open. This plan makes
both the **emission (BTC/ChordFormer ONNX)** and the **transition matrix**
learned-from-data, while keeping the **decoder shell + jazz-standard 7th-class
mapping** — a hybrid the literature hasn't published.

---

## Architecture (target state)

```
audio mono → SpectrogramFrontend (ONNX, frozen) → ChordHead (ONNX, frozen) → per-frame [251] logits
                                                              │
             BeatThis! beats/downbeats → bin frames → bar-level evidence  (bar-level Viterbi, 251 states)
                                                              │
                    Viterbi decode over 251 states  ←  learned 251×251 transition matrix (file)
                                                              │
                            argmax → Harte chord string → collapse-to-7th-class → ChordSymbolChange
```

The current `detect_chords_from_timeline` becomes a **thin wrapper** that:
1. Builds the spectrogram from the raw mono audio (`AnalyzedAudio` already has
   `samples_mono` and `sample_rate`).
2. Runs the two ONNX models via a new `ort` wrapper beside
   `BasicPitchInference` (`inference.rs:34`).
3. Reads the loaded `transition_matrix_251.json` (loaded once at startup).
4. Bins frame logits to bars using the existing `bar_for_frame` helper
   (`harmony.rs:94`).
5. Runs the learned Viterbi (generalise `viterbi_best_path:647` to N classes).
6. Maps the 251-class argmax → `ChordSymbolChange { beat_start, symbol }`.

The old `compute_bar_profiles`/`soft_contrast`/`estimate_key` modules stay in
the codebase behind a `chord_engine = legacy` switch (zero rip risk; supports
A/B and the test suite, and gives us a fallback if the learned engine
underperforms on our measured metrics).

---

## Integration seams in the current code (read-only findings)

These are the load-bearing file:line references used by the plan.

- **`generate_lead_sheet_enhanced_with_timeline`** — `src/leadsheet/preset.rs:292`.
  Orchestrator; calls `detect_chords_from_timeline` at `preset.rs:401` when a
  timeline is supplied, falls back to `detect_chord_changes_per_bar` at `:405`.
- **`detect_chords_from_timeline`** — `src/leadsheet/harmony.rs:490`. The hand-crafted
  layer to replace: `compute_bar_profiles` (`:109`), `estimate_key` (`:335`),
  emission scoring over `12 * 27` root×quality states (`:519`),
  hand-tuned `viterbi_best_path` (`:647`).
- **`TimelineChordInput`** — `src/leadsheet/harmony.rs:20`. Per-frame tensors are
  `Vec<Vec<f32>>` (88 floats per frame for MIDI 21..108), with `step_sec`.
  The timeline path does **not** carry raw audio samples today — that needs to
  be threaded from `analyze_audio`.
- **`analyze_audio`** — `src/headless.rs:244`. Already has `samples_mono` (from
  `crate::audio_io::load_audio_file`) and `sample_rate`; the chord head reuses
  these.
- **`generate_sheet_inner`** — `src/headless.rs:525`. Builds `TimelineChordInput`
  at `:551`; would need to pass raw audio through to the lead-sheet function
  (or extend `TimelineChordInput`/a sibling struct).
- **`BasicPitchInference`** — `src/inference.rs:34`. ONNX Runtime (`ort`)
  wrapper; `new` at `:43`, `infer_audio_window` at `:71` returns
  `(note_probs, onset_probs)` shaped `(output_frames=172, 88)`. Head-name-based
  dispatch at `:128-176`. The pattern to mirror for a new `ChordInference`.
- **`AudioPipeline::process_audio`** — `src/pipeline.rs:63`. Returns
  `PipelineResult { note_probs_sequence, onset_probs_sequence, smoothed_notes }`
  (`:42`) assembled from 50%-hop windows.
- **`CrossValidatedBeats`** — `src/leadsheet/beat_tracking.rs:159`:
  `downbeats, beats, beats_per_bar (2..8), bpm, confidence, source_count`.
  Bar math in `harmony.rs:143`:
  `num_bars = ceil((len(beat_times)-1) / bpb)`; one chord per bar emitted at
  `beat_start = bar * bpb` (`harmony.rs:633`).
- **`quantize_aligned_notes`** — `src/leadsheet/quantize.rs:695` (rule grid-snap).
  **Not changed by this plan**; flagged as the next-highest-ROI follow-up
  (per the Dual-Eval finding that MIDI→score dominates notation quality).
- **`tune::tune`** — `src/tune.rs:379`. Per-track `analyze_audio` once
  (`:428`), then cheap sweeps (`bpm_candidates` × `combos`); writes
  `keyscribe.tuned.json` via `TunedConfig` (`headless.rs:47`). The learned
  engine slots in transparently — `key_sensitivity`/`melody`/`stems`/`bpm`
  still apply (they affect AMT/melody/beat-grid, not the chord head directly).
- **`compare_harmonies`** — `src/sheet_compare.rs:281`, returns
  `ChordReport { exact_match_rate, root_match_rate, mean_onset_error_beats,
  reference_symbols, transcription_symbols, … }` (`:262`). Works unchanged once
  the learned engine emits `ChordSymbolChange`s with conventional symbols.
- **`parse_root_pc`** — `src/musicxml.rs:1042`. Handles `C/C#/Cb/Db/…`; Harte
  syntax uses the same letters (with `:kind` and `/bass`) so the mapping table
  is mechanical.
- **`ShedStemType::identify_melodic_stem_from_stems`** —
  `src/leadsheet/preset.rs:478`. Unchanged; only relevant if a future
  `--chord-stems-demixed` option is added (not in this plan).
- **`models/` inventory** — `basic-pitch.onnx` (230 KB),
  `mel_spectrogram.onnx` (271 KB), `beat_this_small.onnx` (10.6 MB),
  `htdemucs_6s.onnx` (285 MB). The chord ONNX(es) join this directory.
  `--model-dir` only applies to Basic Pitch (`headless.rs:158`); Demucs and
  BeatThis! use a separate `resolve_model_path` (`demucs.rs:776`,
  `beat_this.rs:32`). The chord head will need its own path resolution.

---

## Phased work

### Phase 0 — Lock decisions (no code) — ~0.5 day

1. **Pick the chord backbone.** Candidates, ranked by pragmatics:
   - **BTC** (bidirectional Transformer, 2019, mature, used as the 2026 teacher).
   - **ChordFormer** (Conformer, 2025, +6% class-wise).
   - **madmom-Chordino** (DBN-CRN, simpler, weaker but proven).
   Decision criterion: weights available for non-commercial research + runnable
   with `ort ≥ 2.x` + supports (or convertible to) the 251-class output.
2. **Lock the spectrogram frontend.** Most chord models want a **CQT or mel
   spectrogram** at 44.1 kHz with parameters stored in the checkpoint. Two
   integration options:
   - **(A)** reuse the existing BeatThis! `mel_spectrogram.onnx`
     (`beat_this.rs:32-54`) if the chord model tolerates the same settings;
   - **(B)** export the chord model's own frontend (`torchaudio` CQT/mel via
     `torch.onnx.export`). Default: ship the model's own frontend ONNX so we
     aren't coupled to BeatThis!'s mel params.
3. **Lock the 251-class index** — confirm the exact class-index map is part of
   the checkpoint (or fix one and re-train the head). Used both for emission
   labels and for the learned transition matrix.
4. **Confirm licences** — Isophonics/Beatles + Billboard annotations are
   research-non-commercial; the *transition counts* we ship are derived data.
   For model weights, prefer MIT/Apache-Chord releases. Document licence
   constraints in `HEADLESS.md`.

### Phase 1 — Acquire jazz-standard corpus + build labels (Python) — ~1.5 days

**Phase 1.1 — iRealPro-style corpus search (manual / sources TBD).** Find a
source of **jazz-standard backing tracks paired with MusicXML lead sheets**
(iRealPro-format exports, Band-in-a-Back tracks, or a custom render of a
lead-sheet MusicXML corpus). This is the binding licensing question — the plan
needs a concrete acquisition route before Phase 2 can run on real data; if no
usable source is found we fall back to **synthetic generation** (Phase 1.3).

**Phase 1.2 — Real corpus ingestion.** For each `(audio, musicxml)` pair:
- parse the MusicXML chord symbols (`parse_musicxml_harmonies` in
  `sheet_compare.rs:157` already does this);
- run **BeatThis! on the audio** (mirroring the Rust pipeline) to get
  beats/downbeats and bar boundaries;
- emit a standardised JSON per song:
  `{path, sr, beats:[...], bars:[...], segments:[{start,end,chord_idx_251}]}`.

**Phase 1.3 — Synthetic fallback / scale-up.** If real corpus acquisition is
blocked, generate one:
- harvest lead-sheet MusicXML from iRealPro/WikiFonia-style sources;
- `musescore --export` each to MP3 (`keyscribe-cli render` already exists at
  `bin/keyscribe_cli.rs:118`);
- pair them; drop into `data/labeled/`.
- The existing `scripts/roundtrip.ps1` already does this for one seed; a new
  driver iterates a corpus.
This mirrors the existing `Pretty standard` test-pair creation (the audio was
MuseScore-rendered from the reference MusicXML) and the `out/gt_render.mp3`
fact noted in `HEADLESS.md`.

Cache the corpus to `data/chord_corpus/*.json`. Add to `.gitignore`. Ship only
the **transition-matrix counts** (derived data) and the **model weights**.

### Phase 2 — Fit the learned Viterbi transition matrix (Python) — ~1 day

This is the genuinely novel piece.

1. From the labelled corpus, count chord **adjacency** counts **at bar
   resolution** (one chord per bar = one transition per adjacent bar pair).
   Build `N[i→j]`.
2. Fit `T[i,j] = P(j | i) = (N[i,j] + α) / (sum_j N[i,j] + α·251)` (Laplace
   smoothing; `α` tuned on held-out).
3. **Cap the support** to actually observed chords (~80–120 distinct in a
   standards corpus) — keeps the matrix lean and avoids over-smoothing the
   long tail.
4. **Optional higher-order (2-step)** — `P(j | i, k)` for ii-V-I chains. Ship
   only if it improves the held-out likelihood; otherwise stay 1-step.
5. Persist `transition_matrix_251.json` (or `.bin`) into `models/`. Small
   (~250 KB JSON). Loaded once at startup by the Rust side.

### Phase 3 — Chord head model export (Python) — ~1 day

1. Load the chosen backbone checkpoint (BTC/ChordFormer) into PyTorch.
2. **Decode granularity** — use **per-frame** logits (more flexible; can collapse
   to bar or per-beat labels). For ChordFormer, collapse its segment output to
   per-frame via the segment starts.
3. Export the (frozen) chord head as a single ONNX with `torch.onnx.export`,
   dynamic axis over time. Output: `logits [1, n_frames, 251]`
   (and optional `boundary_logits`).
4. Export the spectrogram frontend **separately** as `chord_mel.onnx`.
5. Write `tools/chord_corpus/export_chord_model.py` — checkpoint → ONNX, so the
   build is reproducible.
6. **Calibration test** (Python) — feed a held-out Beatles song; confirm we
   reproduce the published F1 (≈ BTC Beatles maj/min ≈ 0.83–0.85). Lock this as
   the benchmark.

### Phase 4 — Rust inference layer — ~3–4 days (the bulk)

1. **Generalise the ONNX loader.** Refactor `BasicPitchInference`
   (`inference.rs:34`) cleanly; add a sibling **`ChordInference`** holding the
   spectrogram + chord ONNX sessions and an `InferenceConfig` for time hop /
   frame size. Mirror the existing head-name dispatch pattern (`:128-176`).
2. **New `ChordEngine` enum** at `headless.rs`:
   ```rust
   pub enum ChordEngine { LegacyTemplates, LearnedOnnx { model_dir: PathBuf } }
   ```
   Default (per Decision 4): **`LearnedOnnx` when the chord ONNX and
   `transition_matrix_251.*` are both present on the model path**; else fall
   back to `LegacyTemplates` so the test suite, round-trip script, and zero-model
   builds continue working. A `--chord-engine legacy|learned` CLI flag
   (`keyscribe_cli.rs`) and a `TunedConfig.chord_engine: String` field
   (back-compat default `"legacy"`/absent) let the user force one or the other.

3. **Wire raw audio through** to the chord decoder. Today
   `generate_lead_sheet_enhanced_with_timeline` receives only `NoteEvent`s +
   `TimelineChordInput`. Change: extend `TimelineChordInput` (or add a sibling)
   to carry `samples_mono: Arc<[f32]>` and `sample_rate: u32`, threaded from
   `analyze_audio` → `generate_sheet_inner` (`headless.rs:525`) →
   `generate_lead_sheet_enhanced_with_timeline` (`preset.rs:292`). `Arc` keeps
   the cheap-sweep cost model (the `tune` doc-bargain at `tune.rs:1-8`).
4. **`detect_chords_from_timeline_learned(...)`** at `harmony.rs`:
   - spectrogram = `ChordInference::spectrogram(samples, sr)`;
   - logits = `ChordInference::infer(spec)` → `[n_frames, 251]`;
   - **bar-level evidence** by `bar_for_frame` (`harmony.rs:94`):
     default = **softmax-mean over the bar**;
     `--chord-beat`/`--chord-cleanest`/`--chord-strike` fall back to the
     sub-window assignment from `compute_bar_profiles` (`harmony.rs:204-261`)
     so the live knobs still differentiate runs;
   - **Viterbi over 251 states** with the loaded
     `transition_matrix_251.json` — generalise `viterbi_best_path` (`harmony.rs:647`)
     to take `&[f32]` matrix + `N` states; complexity O(bars × 251²) — for a
     200-bar track that's ~12.6 M ops, fine;
   - emit one chord per bar at `beat_start = bar * bpb`
     (existing convention, `harmony.rs:633`) — keeps the MusicXML writer
     (`musicxml.rs:882`) and `compare_harmonies` (`sheet_compare.rs:281`)
     unchanged.
5. **251 → `ChordSymbolChange` mapping.** Write
   `harte_index_to_keyscribe_symbol(idx: u16) -> String` — a small table
   (251 entries) producing strings in the existing convention
   (`Cmaj7`, `B-7`, `C/G`, …). **Collapse jazz extensions to nearest 7th-class**
   per Decision 2 (maj7Δ13 → maj7, -13 → min7, G13 → dom7, Gdim → empty/dim
   inversions, …). Unit tests against the Isophonics labels + our reference.
6. **Model path resolution.** Add a `resolve_chord_model_path` mirroring the
   Demucs/BeatThis! resolver (`demucs.rs:776`), searching `<exe_dir>/models/`,
   `<exe_dir>/`, `cwd/models/`, `cwd/`. Optionally extend `--model-dir` to cover
   the chord head for parity with Basic Pitch.
7. **GUI plumbing.** `app/sheet_music.rs:838-845` builds `TimelineChordInput`;
   the learned path branches on `chord_engine`. No new GUI control is required
   initially — the GUI just gets better chords through the same
   `ChordSymbolChange` stream.

### Phase 5 — `tune` & `compare` adapt — ~1 day

1. **`compare_harmonies`** (`sheet_compare.rs:281`) needs no change — verify the
   Harte-mapped symbols parse in `parse_root_pc` (`musicxml.rs:1042`). Add tests
   for Harte-specific syntax not yet handled (`:maj7`, `/G`, inversions).
2. **`tune.rs` simplifications** when `chord_engine = learned`:
   - `--chord-beat`/`chord_cleanest`/`chord_strike` still apply (they steer
     sub-window sampling of logits) — keep them;
   - `key_sensitivity` no longer affects chords (only AMT/melody) — keep it;
   - `bpm_candidates` (`tune.rs:291`) already refined ±1.2%; widen to ±2% if
     bar-alignment proves more sensitive under the learned head;
   - `--objective chord` (`tune.rs:65`) becomes the calibration driver for the
     learned engine.
3. **`TunedConfig` schema** (`headless.rs:47`) gets a back-compatible
   `chord_engine: String` field; default `"legacy"` if absent.
4. Update the `keyscribe_cli.rs` `sheet`/`midi` handlers to resolve
   `chord_engine` from CLI flag → config → presence-of-files (per Decision 4).

### Phase 6 — Dataset expansion for evaluation & future fine-tune — ~2–3 days

1. Run the synthetic renderer (Phase 1.3) at scale — produce a labeled
   `(audio, musicxml)` corpus large enough to (a) measure the learned engine on
   jazz standards, and (b) leave room for a future head fine-tune.
2. **Domain-gap awareness** — `arXiv:2408.04737` shows ~20-point F1 drops across
   sound shifts. Log source-domain feature distributions; if the Beatles-trained
   head underperforms on our jazz-standard backing tracks, plan a small head
   fine-tune (out of scope for this plan; flagged in `HEADLESS.md`).
3. Keep the test corpus distinct from any future training split.

### Phase 7 — Evaluation & rollback gate — ~1 day

1. `cargo build --release --no-default-features --bin keyscribe-cli` on
   `tests/data/Pretty standard`. Compare **legacy** vs **learned** end-to-end
   with `keyscribe-cli compare`. Expected (Decision 2):
   - root F1 likely rises (cleaner learning; Cmaj7 / B-7 / D-7 are in the
     251-vocab);
   - exact F1 on the *reference* (Cmaj7 / B-7 / G13 / G-13 / A-7 / D7 / ...)
     will fall on the jazz-extended bars (G-13 → min7, etc.) — a **correct,
     measurable** gap, not a regression in the detection logic.
2. Integration test in `tests/`: runs the learned engine on a small fixture, or
   **skips with a warning when the chord ONNX is absent** (so the existing 50
   tests + `roundtrip.ps1` keep passing in zero-model builds).
3. Patch `scripts/roundtrip.ps1` to optionally use `--chord-engine learned`
   (default legacy; the round-trip's `twinkle.mid` doesn't exercise chords
   anyway).
4. **Rollback path** — if learned underperforms legacy on our measured Harmony
   similarity on the test track, ship it as **opt-in** (`--chord-engine learned`
   only) until Phase 6 fine-tune data closes the gap. Document the trade in
   `HEADLESS.md`.

### Phase 8 — Documentation — ~0.5 day

Update `HEADLESS.md`:
- new section **"## Chord detection: learned 251-class decoder"** describing the
  engine, the transition-matrix file, and the default-when-present rule;
- "by default" caveat — `legacy` until the chord ONNX + transition file are
  present, then `learned` automatically;
- update "### Training log / where to improve" with the new boundary
  ("low root & exact rate → chord head fine-tune on jazz corpus; not just
  thresholds");
- document the licence caveats for the corpus and the model weights;
- update the `Measured on …` line with the new (legacy vs learned) figures.

---

## Files touched (summary)

**Rust (existing):**
- `src/leadsheet/harmony.rs` — add `detect_chords_from_timeline_learned`,
  generalise `viterbi_best_path` to N classes, add
  `harte_index_to_keyscribe_symbol`. Keep legacy behind the switch.
- `src/leadsheet/preset.rs` — `generate_lead_sheet_enhanced_with_timeline`
  accepts raw audio; selects engine.
- `src/headless.rs` — `ChordEngine` enum, `TunedConfig.chord_engine` field,
  thread raw audio through `generate_sheet_inner`.
- `src/inference.rs` — add `ChordInference` beside `BasicPitchInference`.
- `src/bin/keyscribe_cli.rs` — `--chord-engine` flag, model-path resolution.
- `src/tune.rs` — minor (knob set already correct).
- `tests/` — new integration test (skip-on-missing-model).
- `scripts/roundtrip.ps1` — optional `--chord-engine learned`.
- `models/` — new `chord_head.onnx`, `chord_mel.onnx`, `transition_matrix_251.json`.

**Python (new, under `tools/chord_corpus/`):**
- `build_labels.py` — corpus → standardised JSON labels.
- `fit_transition_matrix.py` — corpus labels → `transition_matrix_251.json`.
- `export_chord_model.py` — checkpoint → ONNX.
- `eval_chord_model.py` — held-out F1 calibration vs published numbers.

**Docs:**
- `HEADLESS.md` — Phase 8 updates.
- `CHORD_DETECTION_PLAN.md` — this file (status → implemented once shipped).

---

## Risks & open questions

- **Backbone availability in `ort`** — BTC/ChordFormer were built on the full
  PyTorch stack; some use custom ops. Phase 0 must confirm ORT exportability
  *before* writing the Rust loader. Fallback: re-train a
  Conformer/Transformer-chord from scratch on our synthetic corpus (~1 wk PyTorch).
- **Corpus acquisition (Phase 1.1)** — the single biggest schedule risk. The
  plan name-checks iRealPro-style backing tracks; a concrete acquisition route
  must be confirmed before Phase 2. Synthetic fallback (Phase 1.3) de-risks the
  schedule but lessens the transition-matrix's jazz fidelity.
- **Domain shift** — Beatles/Billboard-trained heads may not transfer to jazz
  backing tracks. Phase 6's fine-tune is the long-term fix; the collapse-to-7th
  mapping (Decision 2) keeps the measurable gap honest in the meantime.
- **Inference cost** — the chord head is ~80 MB; ~1–2 s/track of extra CPU. The
  `tune` cost model (`tune.rs:1-8`) is preserved because the chord model runs
  once per `analyze_audio`, then the Viterbi is cheap to sweep.
- **Viterbi complexity** — 251² × bars is fine on CPU; the matrix-file is loaded
  once at process start.

---

## Follow-ups (not in this plan, queued)

- **Tier 2** — frozen SSL encoder (MuQ / MERT) as the chord/AMT front-end.
- **Tier 3** — BeatThis! `small` → `final` weights; masked-diffusion tracker
  (`arXiv:2608.04624`) for the half-tempo blind-spot.
- **Tier 1.3** — replace the rule-based quantizer
  (`quantize.rs:695`) with a symbolic beat tracker (`arXiv:2507.00466`),
  targeting the rhythm gap the Dual-Eval paper (`2608.04511`) identified as the
  dominant notation-quality driver.
- **AMT front-end** — upgrade Basic Pitch to YourMT3+ (`github.com/mimbres/YourMT3`)
  or MuScriptor (`arXiv:2607.08168`) for polyphonic multi-instrument strength.