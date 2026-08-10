# Melody Transcription Plan — Learned Rhythm Quantizer + Stem-First Melody

Status: **in progress** — Phase 2 (stem-first melody default) implemented and
merged into `src/headless.rs`; Phases 0/1/3/6 still pending (need corpus + GPU
training). Companion to `HEADLESS.md` and `CHORD_DETECTION_PLAN.md`.

## Progress log

- **Phase 2 (done)** — melody stem-first default implemented:
  - `AnalyzedAudio` split into `timelines` (full-mix, chord evidence) +
    `stem_timelines`/`stem_onset_timelines` (identified melodic stem, melody
    source); `melodic_stem_timeline` now indexes `stem_timelines`.
  - `analyze_audio` always produces the full-mix timeline; when `use_stems` it
    additionally transcribes only the identified melodic stem and uses the
    drums/bass stems for beat cross-validation.
  - `SheetOptions.melody_stems` (default `true`) + `TunedConfig.melody_stems`
    (serde-default `false` so legacy configs keep their tuned behavior) +
    `--no-stems-melody` CLI flag.
  - `generate_sheet` → `analyze_audio_for_sheet`: explicit `--stems` hard-fails
    without the Demucs model; the stem-first *default* falls back to full-mix
    melody when `htdemucs_6s.onnx` is absent.
  - Phase 2.3: `identify_melodic_stem_from_stems` now adds an onset-density
    term (`0.4 pitch-variance + 0.3 spectral-energy + 0.3 onset-density`).
  - Fixed a latent out-of-bounds in `compute_bar_profiles` (`harmony.rs:172`)
    exposed by pairing the full-mix timeline with the shorter stem-based beat
    grid (last frame's `bar` could equal `num_bars`).
  - **Stem-identification bug (root-caused + fixed):** on the piano-only test
    track the heuristic picked the near-silent **Vocals** stem (69-artifact-note
    transcription, misleadingly high pitch accuracy). Cause:
    `compute_pitch_variance` is actually a zero-crossing *rate-change* measure
    (spectral instability) that scores **high** for sparse noise bursts in
    near-silent stems (Vocals pv=0.609, rms=0.0003) and **low** for sustained
    content (Piano pv=0.040, rms=0.0498) — inverted for our purpose. Scoring
    rebalanced to `se*0.5 + od*0.4 + pv*se*0.1` so energy+onsets dominate and
    `pv` is energy-gated (near-silent leakage stems can never win). Debug log
    via `KEYSCRIBE_STEM_DEBUG=1`. After fix: Piano correctly picked
    (0.184 vs 0.009 next), 172 notes.
  - Verified: 50 lib tests pass; `cargo check --bin keyscribe` clean; release
    CLI smoke-tested end-to-end on `tests/data/Pretty standard.mp3`.
- **Phase 2 interim measurement** (single track, `--melody heuristic`, bpm=222,
  vs `Pretty standard.musicxml`):

  | path | pitch | note | recall | notes |
  |---|---|---|---|---|
  | full-mix (baseline, `--no-stems-melody`) | 0.521 | 0.026 | 0.039 | 182 |
  | stem-first (pre-fix: mis-picked Vocals) | 0.650 | 0.019 | 0.016 | 69 |
  | stem-first (post-fix: Piano) | 0.532 | 0.027 | 0.039 | 188 |

  Post-fix stem-first ≈ full-mix on this piano-only track (expected: the piano
  stem ≈ the mix). The stem-first default pays off when there is a genuinely
  separate melodic source (vocals over comp, lead over rhythm section); it's
  neutral on solo-piano. The misleading 0.650-pitch row was an artifact of
  transcribing the near-silent vocal-leakage stem.

## Objective

Replace the rule-based rhythm quantizer (`quantize_aligned_notes` at
`src/leadsheet/quantize.rs:695`) with a **learned MIDI-to-score converter**
(encoder-decoder over the note sequence, emitting per-note rhythmic-value
tokens), trained on (performed-MIDI, score) pairs from **A-MAPS, ASAP,
GuitarSet, Leduc**. In parallel, flip the melody branch to **Demucs
stem-first by default** so the quantizer receives cleaner monophonic input.
Trains in Python, runs in Rust via the existing `ort` ONNX runtime. BeatThis!,
Basic Pitch, the MusicXML writer, `compare`, and `tune` stay unchanged as far as
possible.

### Decisions locked

1. **Primary target — learned rhythm quantizer.** The Dual-Eval finding
   (`arXiv:2608.04511`, ACM MM 2026) directly measured that, across 24
   pipelines (8 audio-to-MIDI × 3 MIDI-to-score converters), "the latter
   component systematically determines the favored objective" — i.e. the
   quantizer, not the AMT front-end, dominates notation quality. KeyScribe's
   note-accuracy **0.07** (vs pitch accuracy 0.565) is exactly this gap.
2. **Melody source — Demucs stem-first default.** SOTA-consistent: the
   Mel-RoFormer melody-transcription work (`arXiv:2409.04702`, 2024) and the
   Charlie-Parker Omnibook jazz A2S pipeline (`arXiv:2405.16687`, 2024) both
   separate first, then transcribe the melodic stem. KeyScribe's Demucs
   `htdemucs_6s` model is already shipped; making it the melody default gives
   the quantizer cleaner input without new infra. Mix-first is retained as
   `--no-stems-melody` fallback and for zero-model builds.
3. **Strict monophonic melody.** The learned quantizer sees one note at a time
   (matches the existing skyline monophonic-segment output at
   `musicxml.rs:591`). Chordal pickups / grace notes are out of scope —
   consistent with the leadsheet reference, which has sparse monophonic
   melody notes.
4. **Toolchain — Python training + ONNX in Rust.** Consistent with the chord
   plan: PyTorch training on GPU, export to ONNX, run via the existing
   `ort`-based CLI. Keeps the two plans symmetrical and reuses the
   `BasicPitchInference` (`inference.rs:34`) ONNX pattern.

---

## Background — current state vs SOTA

Measured on `tests/data/Pretty standard.mp3` (`HEADLESS.md`):

| Metric | Value | SOTA gap |
|---|---|---|
| pitch accuracy | **0.565** (was 0.139 before onset-density skyline) | moderate |
| note accuracy | **0.07** | **dominant gap** |
| recall | 0.10 | moderate |
| mean onset error | **0.19 beats** | large |
| mean duration error | **0.37 beats** | large |

So the melody notes are mostly the right *pitches* but wrong *times/durations* —
the rhythm/quantization stage is the gap, exactly the stage the Dual-Eval
paper measured as dominant.

### Relevant SOTA (2022–2026)

- **Dual-Eval / Rubato** (`arXiv:2608.04511`, ACM MM 2026) — across 24
  pipelines (8 audio-to-MIDI models × 3 MIDI-to-score converters), "the latter
  component [MIDI-to-score] systematically determines the favored objective."
  **Invest in the quantizer, not the AMT front-end.**
- **Symbolic beat tracker** (`arXiv:2507.00466`, SMC 2025) — encoder-decoder
  Transformer, "sequence-to-sequence translation of MIDI input to beat
  annotations", trained on **A-MAPS / ASAP / GuitarSet / Leduc**. Confirms
  these corpora are the canonical training material for the
  MIDI→score / MIDI→beat task family.
- **SheetSage2Kern** (`arXiv:2608.06165`, ACM MM 2026) — audio-to-score A2S
  benchmark, 4.98% SER classical / 20.92% SER pop, uses MuQ features. The
  full A2S line bundles AMT+quantize; our targeted fix keeps Basic Pitch and
  swaps only the quantizer.
- **Mel-RoFormer / Charlie-Parker Omnibook** — separation-first melody
  pipelines (`arXiv:2409.04702`, 2024; `arXiv:2405.16687`, 2024 jazz). Confirms
  Demucs-then-transcribe as the SOTA melody recipe.
- **TONet** (`arXiv:2202.00951`, ICASSP 2022, tone-octave) — learned salience
  baseline; Tier-2 follow-up if pitch accuracy becomes the bottleneck after
  the quantizer lands.
- **Classic HMM/GMM rhythm transcription** (Nakamura, Yoshii et al.) —
  principled baseline; a small Transformer/GRU over the same features is the
  modern successor.
- **Corpus bias** (`arXiv:2408.04737`) — ~20-pt F1 drops across *sound* shifts
  and ~14-pt across *genre* shifts. Piano-trained quantizers can over-snap
  plain 8ths where jazz phrasing wants swung or behind-the-beat 8ths; the
  `swing_style` input feature (Phase 1.2) is the lever, and a jazz-phrasing
  fine-tune (Phase 6) is the long-term fix.

### Novelty / position

No 2024–2026 melody-quantizer paper integrates (a) a **stem-separated
front-end**, (b) a **learned tokenizer of the existing engraver's
`DurationToken` set**, (c) a **beat-aligned (not grid-aligned) evidence
featurizer**. The combination is KeyScribe-specific.

---

## Architecture (target state)

```
audio mono → Demucs htdemucs_6s → identify_melodic_stem → Basic Pitch (melodic stem timeline)
                                              │
        extract_notes_from_timeline (mono) → NoteEvent → extract_melody_skyline (monophonic segments)
                                              │
        associate_note_events + BeatThis! beats → BeatAlignedNote (intra_beat_pos, raw_dur, beat_index, swing)
                                              │
        MelodyQuantizer (ONNX) → per-note (rhythmic_value, dots, time_mod) ∈ DurationToken set
                                              │
        QuantizedNote → build_musicxml_document (unchanged engraver)
```

The current `quantize_aligned_notes:695` becomes a **thin wrapper** that:

1. Featurizes each `BeatAlignedNote` into a per-note feature vector
   (`intra_beat_pos`, `raw_duration_beats`, `tempo`, `beat_index mod bpb`,
   `swing_style`, `pitch`, `velocity`).
2. Runs the learned `MelodyQuantizer` ONNX → per-note token
   `(note_type, dots, time_mod)`.
3. Validates the token against the existing `DurationToken` set
   (`musicxml.rs:1210`) and assembles a `QuantizedNote` exactly as the rule
   path does (`quantize.rs:737`).

The rule path stays in the codebase behind a `quantizer_engine =
legacy | learned` switch (zero rip risk; A/B support; the existing test suite +
`roundtrip.ps1` keep passing on zero-model builds).

---

## Integration seams in the current code (read-only findings)

These are the load-bearing file:line references used by the plan.

- **`quantize_aligned_notes`** — `src/leadsheet/quantize.rs:695`. The function
  to replace. Inputs: `&[BeatAlignedNote]`, `&[SwingSection]`, `beats_per_bar:
  u32`. Output: `Vec<QuantizedNote>` (`types.rs:133`). Per-note snap at `:722`
  (`snap_intra_beat_pos` over `subdivision_grid`), duration at `:733`
  (`quantize_duration` over `compute_next_onset_duration`).
- **`BeatAlignedNote`** — `src/leadsheet/types.rs:5`:
  `id, pitch, velocity, original_start/end_sec, beat_index, bar_index,
  intra_beat_pos (0..1), surrounding beat times, beat_duration_sec`. The
  *exact* feature sequence the learned quantizer consumes.
- **`QuantizedNote`** — `src/leadsheet/types.rs:133`: `beat_start,
  beat_duration, bar_index, beat_index, intra_beat_pos, swing_style,
  swing_feel, articulation, confidence`. The learned model fills these fields
  directly.
- **`QuantizationConfig`** — `quantize.rs:19`: current grid definitions
  (`grids = [1.0, 0.5, 0.25]`, `duration_grids`, `min_duration_beats = 0.25`).
  **Becomes** the *fallback* config used only by `legacy`; the `learned`
  engine ignores it (the model emits dots/tuplets directly).
- **`subdivision_grid`** — `quantize.rs:625`; the hand-coded `[0, 0.5, 1]` /
  swing `[0, 2/3, 1]` / triplet `[0, 1/3, 2/3, 1]` grids the learned model
  replaces.
- **`DurationToken` set** — `src/musicxml.rs:1210`:
  `{whole, half., half, quarter., quarter, eighth., eighth, 16th., 16th}` +
  triplet `time_mod=(2,3)` variants. **This is the learned model's output
  vocabulary** (~12 tokens) — the engraver (`duration_tokens_for_ticks:1191`)
  already consumes it.
- **`generate_lead_sheet_enhanced_with_timeline`** —
  `src/leadsheet/preset.rs:292`. Calls
  `quantize_aligned_notes(&aligned, &swing_sections, beats_per_bar)` at
  `preset.rs:350`. `aligned` comes from `associate_note_events` at `:304`. The
  learned engine slots in right here — same call site, swapped impl.
- **`extract_melody_skyline`** — `src/musicxml.rs:591`. The monophonic-segment
  producer. Unchanged by the quantizer plan, but **fed by the stem-first
  change** (cleaner upstream `NoteEvent`s).
- **`identify_melodic_stem_from_stems`** — `src/leadsheet/preset.rs:478`. The
  stem-identity heuristic (`score = pitch_variance*0.5 + spectral_energy*0.5`).
  Phase 2.4 improves this with an onset-density term.
- **`analyze_audio`** — `src/headless.rs:244`. Already supports
  `use_stems=true`; returns `AnalyzedAudio { timelines, onset_timelines,
  melodic_stem_timeline, beats, … }`. Phase 2 flips the melody branch's
  default to call with `use_stems=true`.
- **`notes_from_analysis`** — `src/headless.rs:347`. Routes melody reduction
  through `analysis.melodic_stem_timeline` (`:356-378`) when present — already
  stem-aware; Phase 2 makes it the only path for melody.
- **`BasicPitchInference`** — `src/inference.rs:34`. The ONNX wrapper pattern
  to mirror for `MelodyQuantizer`. Head-name-based dispatch at `:128-176`.
- **`detect_swing`** — `quantize.rs:490`. **Kept** — the swing section becomes
  an *input feature* to the learned quantizer (`swing_style`), so the model
  picks the right subdivision (straight vs triplet) given the section label,
  validated against the section.
- **`compare_note_lists`** — `src/sheet_compare.rs:417`. `CompareReport
  { note_accuracy, mean_onset_error_beats, mean_duration_error, … }` (`:391`).
  **The metric the plan optimizes** — `note_accuracy` and onset/duration
  error are the primary calibration numbers during training and eval.
- **`tune::tune`** — `src/tune.rs:379`. The cost model (analyze once, sweep
  cheap) is preserved; the quantizer ONNX runs <50 ms / 200 notes per
  evaluation — negligible vs `generate_sheet_from_analysis`.
- **`keyscribe.tuned.json` schema** — `src/headless.rs:47`. Gets a back-compat
  `melody_quantizer: String` field (default `"legacy"`).

---

## Phased work

### Phase 0 — Lock decisions (no code) — ~0.5 day

1. **Confirm the quantizer corpus.** Download A-MAPS, ASAP, GuitarSet, Leduc
   (licences: research / academic). Confirm each provides (performed MIDI,
   score with rhythmic values, beats).
2. **Lock the output vocabulary** at the existing `DurationToken` set
   (`musicxml.rs:1210`) — ~12 tokens including triplet `time_mod`. This
   *constrains* the learned model to outputs the engraver already supports,
   eliminating post-hoc rounding.
3. **Lock the architecture** at a small encoder-decoder Transformer or 2-layer
   GRU seq2seq over per-note features (~50 K–200 K params — small enough that
   inference is trivial in `ort`). Decide attention vs plain GRU based on the
   length of the longest melody phrase (~200 notes; attention helps
   phrase-level context).
4. **Confirm `ort` exportability** of the chosen architecture
   (`torch.onnx.export` with dynamic time axis on the note dimension).
5. **Decide swing handling** — model receives `swing_style` as an input
   feature AND has a "triplet time_mod" token in its output vocabulary → it
   learns to emit triplet grids in swing sections. Keep `detect_swing`
  (`quantize.rs:490`) upstream as the section-labeller.

### Phase 1 — Corpus + feature pipeline (Python) — ~1.5 days

1. `tools/melody_corpus/build_features.py`:
   - load each `(performed MIDI, score)` pair from A-MAPS / ASAP / GuitarSet
     / Leduc;
   - run BeatThis! on the performed MIDI (or use the dataset's own beats if
     present) to get `intra_beat_pos`, `raw_duration_beats`, `beat_index`,
     `bar_index`;
   - encode the score's per-note rhythmic value as the target token index in
     the `DurationToken` set;
   - emit `{note_features: [...], target_token: int, song_id: ...}` per song,
     plus a held-out split.
2. **Feature vector per note** (locked): `(intra_beat_pos,
   raw_duration_beats, tempo, beat_index mod bpb, swing_style_one_hot[3],
   pitch, velocity)` — 10 floats. Compact, deterministic, contains everything
   the rule grid used plus phrasing cues.
3. **Data hygiene** — remove notes with `raw_duration < 0`, cap `tempo` to
   [40, 260], normalize `intra_beat_pos ∈ [0, 1)`. Log the per-dataset token
   distribution — the long tail (whole / 16th. / triplets) is the hard case;
   consider focal loss or class-balanced sampling.
4. Cache to `data/melody_corpus/*.npz` (gitignored).

### Phase 2 — Demucs stem-first default for melody (Rust) — ~1 day

1. **Flip the default in the melody branch** — `generate_sheet_inner`
   (`headless.rs:525`) calls `analyze_audio(use_stems=true)` when the Demucs
   model is present and no `--no-stems-melody` is set. Mix-first remains the
   zero-model fallback (`analyze_audio` already errors cleanly when
   `htdemucs_6s.onnx` is absent).
2. **Wire the melodic stem through to the melody branch** —
   `notes_from_analysis` (`headless.rs:347`) already routes through
   `analysis.melodic_stem_timeline` when present; confirm it's *always*
   populated when Demucs runs, else fall back to the first melodic-stem
   timeline.
3. **Improve `identify_melodic_stem_from_stems`** (`preset.rs:478`) — the
   `pitch_variance*0.5 + spectral_energy*0.5` heuristic. Phase 2 light fix:
   add **onset-density** as a third term (the melody re-articulates more than
   the comp — mirrors the skyline logic). A learned instrument classifier is
   left as a Tier-2 follow-up.
4. **Expose a `--no-stems-melody` CLI flag** (`keyscribe_cli.rs`) and a
   corresponding `TunedConfig` field for users who want the cheaper mix-first
   melody.
5. Update `tune.rs` to sweep `stem-mode` over `[false, true]` (already does
   when `--stems`) — but now the *melody* stem is on by default; the chord
   branch keeps its own `--stems` toggle (per the chord plan, the chord head
   sees the full mix).

### Phase 3 — Train the MelodyQuantizer (Python) — ~2–3 days

1. `tools/melody_corpus/train_quantizer.py`:
   - build the seq2seq model (Transformer or GRU), vocab = ~12
     `DurationToken`s + `<bos>/<eos>/<pad>`;
   - train on the Phase-1 features; objective = per-note cross-entropy + a
     **sequence-level onset-position consistency** auxiliary loss (so adjacent
     notes snap to a coherent grid, not per-note argmax independently);
   - include the **measure-position constraint**: the sum of quantized
     durations within a measure must equal `beats_per_bar` — enforce with a
     differentiable approximate penalty or a constrained beam search at
     decode.
2. **Calibration** — hold out A-MAPS / ASAP subsets; report per-token
   accuracy, onset error (beats), duration error (beats). Target: beat the
   rule grid-snap baseline (`quantize_aligned_notes`) on all three. Lock the
   checkpoint if it ties or beats on all three.
3. **Export** the trained model to ONNX (`torch.onnx.export`, dynamic time
   axis on the note dimension). Ship as `models/melody_quantizer.onnx` (<1 MB
   expected).
4. `tools/melody_corpus/eval_quantizer.py` — runs the ONNX on a held-out pair
   and prints the same metrics `compare` produces (so Rust and Python report
   the same numbers).

### Phase 4 — Rust inference layer — ~2 days

1. **New `MelodyQuantizerInference`** in `src/inference.rs` beside
   `BasicPitchInference`. Loads `melody_quantizer.onnx`; takes a sequence of
   10-feature vectors; returns a sequence of `DurationToken`-ish
   `(note_type, dots, time_mod)`.
2. **New `QuantizerEngine` enum** at `headless.rs`:
   ```rust
   pub enum QuantizerEngine { LegacyGrid, LearnedOnnx { model_dir: PathBuf } }
   ```
   Default (per Decision 4): `LearnedOnnx` when `melody_quantizer.onnx` is
   present; else `LegacyGrid` so zero-model builds and the test suite keep
   working. A `--quantizer legacy|learned` CLI flag and
   `TunedConfig.melody_quantizer: String` field force one or the other.
3. **`quantize_aligned_notes_learned(...)`** at `quantize.rs`:
   - featurize each `BeatAlignedNote` (same vector as Phase 1.2);
   - run `MelodyQuantizerInference::infer(features) ->
     Vec<DurationTokenOut>`;
   - validate each output token against the existing `DurationToken` set and
     assemble a `QuantizedNote` exactly as the rule path does at
     `quantize.rs:737` (so the engraver and `compare` see identical types);
   - **measure-fill validation**: if the per-measure quantized durations don't
     sum to `beats_per_bar`, fall back to the rule grid for that measure (so
     we never emit an invalid measure — mirrors the engraver's existing
     `duration_tokens_for_ticks:1191` measure-fill logic).
4. **`generate_lead_sheet_enhanced_with_timeline`** (`preset.rs:350`) — swap
   the call to `quantize_aligned_notes_learned` when the learned engine is
   selected; otherwise unchanged.
5. Model-path resolution — add a `resolve_melody_model_path` mirroring the
   Demucs/BeatThis! resolver (`demucs.rs:776`), searching `<exe_dir>/models/`,
   `<exe_dir>/`, `cwd/models/`, `cwd/`. Optionally extend `--model-dir` to
   cover the quantizer for parity with Basic Pitch.

### Phase 5 — `tune`, `compare`, and metrics — ~0.5 day

1. **`compare_note_lists`** (`sheet_compare.rs:417`) is unchanged — verify the
   learned quantizer's output parses identically. The metrics
   (`note_accuracy`, `mean_onset_error_beats`, `mean_duration_error`) become
   the *training/eval calibration numbers* — log them per-evaluation in
   `tune`'s report (`tune.rs:510`).
2. **`tune.rs`** — add `melody_quantizer` to the `TunedConfig` it writes
   (`tune.rs:578`); the `--objective melody` mode (`tune.rs:65`) becomes the
   calibration driver for the learned quantizer (sweeps `key_sensitivity`,
   `bpm`, `stem-mode` against the *learned* quantizer's note-accuracy output).
   Keep the cheap-sweep cost model — the quantizer ONNX runs <50 ms per
   evaluation.
3. **`TunedConfig` schema** (`headless.rs:47`) gets the back-compat
   `melody_quantizer: String` field; default `"legacy"` if absent.

### Phase 6 — Dataset expansion for jazz-statistics (optional but recommended) — ~1–2 days

1. A-MAPS / ASAP / GuitarSet are piano/guitar; jazz is under-trained. Generate
   **jazz-standard backing tracks with their reference MusicXML** via the
   iRealPro-style corpus route (mirrors the chord plan's Phase 1.3),
   batch-`render` to audio, transcribe with Basic Pitch + BeatThis!, then
   **train a jazz-specific quantizer head** (or fine-tune the Phase-3 model).
2. Domain-gap log — `arXiv:2408.04737` shows ~20-pt F1 drops across sound
   shifts. Piano-trained quantizers can over-snap plain 8ths where jazz
   phrasing wants swung or behind-the-beat 8ths. The `swing_style` input
   feature is the lever — confirm it generalises on held-out jazz.

### Phase 7 — Evaluation & rollback gate — ~1 day

1. `cargo build --release --no-default-features --bin keyscribe-cli` on
   `tests/data/Pretty standard`. Compare **legacy** vs **learned** end-to-end
   with `compare`. **Expected** (Decision 1 + 2 combine):
   - pitch accuracy stable or slightly improved (cleaner stem);
   - note accuracy materially lifted from 0.07 (the model beats the rule
     grid on the calibration corpora);
   - mean onset/duration errors drop.
2. Integration test in `tests/`: runs the learned quantizer on a small
   fixture, or **skips with a warning when `melody_quantizer.onnx` is absent**
   (existing 50 tests + `roundtrip.ps1` keep passing in zero-model builds).
3. Patch `scripts/roundtrip.ps1` to optionally use `--quantizer learned`
   (default `legacy` so the round-trip doesn't need the new model).
4. **Rollback path** — if learned underperforms legacy on note-accuracy on the
   test track, ship as opt-in (`--quantizer learned` only) until Phase 6's
   jazz fine-tune closes the gap. Document the trade in `HEADLESS.md`.

### Phase 8 — Documentation — ~0.5 day

Update `HEADLESS.md`:
- new section **"## Rhythm quantization: learned MIDI-to-score converter"**
  describing the engine, `models/melody_quantizer.onnx`, the per-note feature
  vector, and the default-when-present rule;
- "by default" caveat — `legacy` until the quantizer ONNX is present, then
  `learned` automatically;
- update **"## Melody extraction (musicxml.rs)"** to note Demucs stem-first
  is now the melody default;
- update **"### Training log / where to improve"** with the new boundary
  ("low note-accuracy → quantizer fine-tune on jazz corpus; not just
  `min_duration_beats`.");
- update the `Measured on …` lines with legacy vs learned figures.

---

## Files touched (summary)

**Rust (existing):**
- `src/leadsheet/quantize.rs` — add `quantize_aligned_notes_learned`; keep
  legacy behind the switch.
- `src/leadsheet/preset.rs` — `generate_lead_sheet_enhanced_with_timeline`
  selects the quantizer engine; light improvement to
  `identify_melodic_stem_from_stems` (`:478`, add onset-density term).
- `src/headless.rs` — `QuantizerEngine` enum, `TunedConfig.melody_quantizer`
  field, melody branch defaults to `analyze_audio(use_stems=true)`.
- `src/inference.rs` — add `MelodyQuantizerInference` beside
  `BasicPitchInference`.
- `src/bin/keyscribe_cli.rs` — `--quantizer legacy|learned` flag +
  `--no-stems-melody` flag + model-path resolution.
- `tests/` — new integration test (skip-on-missing-model).
- `scripts/roundtrip.ps1` — optional `--quantizer learned`.
- `models/` — new `melody_quantizer.onnx` (<1 MB).

**Python (new, under `tools/melody_corpus/`):**
- `build_features.py` — (performed-MIDI, score) → featurized note sequences +
  token targets.
- `train_quantizer.py` — train the seq2seq model on the featurized corpus.
- `eval_quantizer.py` — held-out per-token accuracy + onset/duration errors
  (matches `compare`'s metrics).
- `export_quantizer_onnx.py` — checkpoint → ONNX.

**Docs:**
- `HEADLESS.md` — Phase 8 updates.
- `MELODY_TRANSCRIPTION_PLAN.md` — this file (status → implemented once
  shipped).

---

## Risks & open questions

- **Quantizer-corpus domain shift** — A-MAPS / ASAP / GuitarSet are piano /
  guitar; jazz phrasing (behind-the-beat 8ths, swung triplet feels) is
  under-represented. Phase 6's jazz fine-tune is the long-term fix; the
  `swing_style` input feature is the short-term lever.
- **Measure-fill correctness** — a learned per-note argmax can produce
  durations that don't sum to a full measure. Mitigated by (a) a
  sequence-level additive penalty at training, and (b) a per-measure
  fallback to the rule grid in Rust (`quantize.rs:1191`'s measure-fill logic
  already absorbs the slack). Keep the fallback; never emit an invalid
  measure.
- **Inference cost / sweep model** — the quantizer ONNX is tiny (<1 MB, <50 ms
  / 200 notes). The `tune` cheap-sweep cost model (`tune.rs:1-8`) is preserved.
- **Stem-first inference cost** — Demucs is the expensive step (~seconds /
  track). Making it the *melody* default doubles inference cost vs mix-first
  on tracks where the chord branch doesn't already run Demucs. Mitigation:
  when `--stems` is already set (chord plan's option), reuse the same stem
  separation for both branches. When `--no-stems` is explicit, fall back to
  mix-first melody too.
- **Co-existence with the chord plan** — both this and the chord plan touch
  `generate_sheet_inner` (`headless.rs:525`) and
  `generate_lead_sheet_enhanced_with_timeline` (`preset.rs:292`).
  They compose cleanly: the melody branch uses stems + the learned quantizer;
  the chord branch uses full-mix + the learned chord head. The two engine
  enums (`ChordEngine`, `QuantizerEngine`) are independent switches — neither
  forces the other.
- **`extract_melody_skyline` is unchanged** — pitch accuracy 0.565 is *not*
  the dominant gap per Dual-Eval, so this plan deliberately doesn't touch it.
  A learned salience model (Deep-Salience / TONet) is queued as a Tier-2
  follow-up if pitch accuracy becomes the bottleneck after the quantizer
  lands.

---

## Follow-ups (not in this plan, queued)

- **Tier 2 — learned melody salience model** (Deep-Salience
  `arXiv:1612.05065` / TONet `arXiv:2202.00951`) replacing
  `extract_melody_skyline`, lifting pitch accuracy past 0.565 when it becomes
  the bottleneck.
- **Tier 2 — frozen SSL encoder** (MuQ `arXiv:2501.01108` / MERT
  `arXiv:2306.00107`) feeding the quantizer, AMT, and chord heads.
- **Tier 3 — Basic Pitch → YourMT3+** (`github.com/mimbres/YourMT3`; open,
  handles vocals, stronger on polyphonic mixes). Note corpus-bias
  (`arXiv:2408.04737`) — ~20-pt F1 drop across sound shifts — re-validate on
  jazz before committing.
- **Tier 1 (parallel) — learned chord head + transition matrix** (the
  companion `CHORD_DETECTION_PLAN.md`); the two plans are orthogonal and
  composable. Combining them would unblock tight parallel engineering since
  they share only a thin seam at `preset.rs:292` and `headless.rs:525`.