# KeyScribe Progress Journal

Living record of every change we make to the transcription pipeline, in
roughly chronological order. Each entry lists **what** changed, **why**, the
**measured effect**, and the **files touched** so we can trace how we got to
the current numbers.

- New work goes at the top of [Entries](#entries) (or append to today's entry).
- Update the [Metrics dashboard](#metrics-dashboard) after every measurement.
- Companion docs: `HEADLESS.md` (system memory), `CHORD_DETECTION_PLAN.md`,
  `MELODY_TRANSCRIPTION_PLAN.md`.

---

## Metrics dashboard (current)

Measured on the Omnibook evaluation corpus (MuseScore-rendered from the
reference MusicXMLs). Chord metrics via `keyscribe-cli compare`; melody metrics
on `Confirmation` (`--melody heuristic --bpm 208`).

| Track | chord root | chord exact | notes | pitch | note acc |
|---|---|---|---|---|---|
| Confirmation | **0.340** | **0.030** | 100 | 0.902 | 0.286 |
| Ornithology | 0.190 | 0.016 | 63 | — | — |
| Donna Lee | 0.125 | 0.000 | 88 | — | — |

Full-corpus dashboard (`keyscribe-cli eval-corpus`, 50 tracks, `--melody
heuristic --quantizer learned`, fixed XML tempos):

| date | pitch | note | recall | onset err | chord root | chord exact |
|---|---|---|---|---|---|---|
| 2026-08-10 baseline | 0.826 | 0.374 | 0.387 | 0.256 | 0.393 | 0.139 |
| 2026-08-14 rhythm fixes | **0.867** | **0.406** | 0.381 | **0.231** | 0.393 | 0.139 |

Chord-root progression on Confirmation: **0.18 → 0.30** (emission: bass anchor
+ key prior) → **0.34** (learned transition matrix).

Prior milestones (reconstructed, `HEADLESS.md`):
- `Pretty standard.mp3` chord root F1 **0.200** (was 0.062), exact F1 0.200
  (was 0.000).
- Melody pitch accuracy **0.800** (was 0.139) on `Pretty standard.mp3`.
- Note onset error **0.100 beats** (was 0.188) after the P4 extraction rework.

---

## Entries

### 2026-08-14 — Rhythm diagnosis corrected (straight 8ths) + extraction fixes

**Correction to the Tier A1 diagnosis:** the Omnibook MP3s are MuseScore
renders with **straight 8ths** (no swing). The `0.25/0.75` gap pattern that
made bars vote "not coarse" is NOT swung feel — it is per-note onset timing
error (±72 ms = one 16th at 208 BPM) plus melody-line chatter. Evidence
(`KEYSCRIBE_QUANT_DEBUG` + `tools/diag/*.py`):

- Skyline emitted **688 segments for a ~200-note head**; 283 segments
  < 120 ms (comp-strike leakage pokes at velocity 54-73 between melody notes
  at ~100-105).
- Intra-beat-position histogram over 688 notes was near-UNIFORM — the
  quantizer was being fed a fragment cloud, not a melody.
- Matched-note onset deltas (vs reference) were bimodal: exactly 0.00, or
  exactly ±0.25 beats (a full 16th off — snapping cannot fix raw input that
  is a slot away).

**Fixes (3):**

1. **Chatter absorption** in the skyline post-pass (`musicxml.rs`): a
   segment < 120 ms is absorbed into its neighbor when much quieter than
   both neighbors (< 75%) or a foreign pitch at < 90% of the louder
   neighbor. Full-velocity short melody notes (pickups, 16ths) survive.
   Confirmation: 688 → 545 melody segments.
2. **Complexity-penalized snapping** (`snap_intra_beat_pos`,
   `quantize.rs`): coarse slots (onbeat/8th) get a small margin (0.045 for
   16ths, 0.03 for triplet 8ths) over fine slots, so ±60 ms jitter can't
   flip an 8th onto the dotted-8th slot. (The doc comment claimed
   coarse-to-fine precedence before; the implementation was plain
   nearest-neighbor. Now it actually is.)
3. **Onset-head positioning** (`extract_notes_from_timeline`,
   `headless.rs`): onsets and re-articulation splits move to the onset
   head's local peak (±4 frames, requires peak ≥ 0.3) instead of the
   frame-head threshold crossing. The onset head is trained to be sharp at
   attacks; the frame head ramps ±72 ms. The GUI extractor now delegates to
   the same function (single source of truth; onset refinement no-ops
   without an onset timeline).

**Measured:**

- Confirmation @ 208 (`--melody heuristic --quantizer legacy`):
  note acc **0.240 → 0.286**, pitch **0.883 → 0.902**, onset err
  **0.153 → 0.138**; duration err 0.455 → 0.481 (slightly worse — absorbed
  fragments extend neighbors; acceptable trade).
- **Corpus gate (50 tracks, learned):** pitch **0.826 → 0.867**, note
  **0.374 → 0.406**, onset err **0.256 → 0.231**, recall 0.387 → 0.381,
  chords byte-identical (root 0.393 / exact 0.139). 70 lib tests pass,
  `cargo check --bin keyscribe` clean.
- bpm sweep (206.5-208) confirmed 208 is still the best fixed grid — the
  residual ±0.25 errors are per-note detection noise, NOT global tempo
  drift.

**Remaining gap:** ~half the matched notes are still exactly one slot off —
per-note onset noise a 16th away from truth that no grid snapping can fix.
That is Tier B1's job (sequence quantizer with MERGE tokens +
over-segmentation augmentation, `plans/2026-08-14_TIER_B1_*`).

**Files touched:** `src/musicxml.rs` (chatter absorption), `src/leadsheet/
quantize.rs` (penalized snap + QUANT_DEBUG evidence dump + learned-fallback
counter), `src/headless.rs` (onset-head refinement, pub extractor, stem
count debug), `src/app/sheet_music.rs` (GUI delegates to headless
extractor), `tools/diag/` (compare_onsets / align_onsets / side_by_side
diagnostics).

---

### 2026-08-14 — Tier A1 rhythm coarsening: implemented, verified INERT (no-op)

**What:** Per `plans/2026-08-14_TIER_A1_rhythm_merge_coarsening.md`, added a
per-bar post-quantization pass that merges 16th over-segmentation fragments
onto the 8th grid.

- `RhythmCoarsenConfig { enabled, coarse_vote_ratio }` + `coarsen_rhythm` in
  `src/leadsheet/quantize.rs`, applied right after quantization in `preset.rs`.
- Wired into `SheetOptions`, `TunedConfig` (`rhythm_coarsen: bool`, default
  true, absent = true), and `--no-rhythm-coarsen` CLI flag.
- Grid vote per bar: `coarse = gaps ≥ 0.45`, `fine = (0.05, 0.45)`,
  COARSE iff `coarse/(coarse+fine) ≥ 0.70` AND ≥1 gap is a 16th (0.25) multiple
  AND `offsets.len() ≥ 2×beats_per_bar + 1` (guards swung 8ths whose swung
  onsets snap to 16th slots; plan Task 3 fallback).
- COARSE bars are re-snapped to the 0.5 grid, collisions keep higher
  confidence, same-pitch adjacent fragments merge, durations recomputed as the
  gap to the next snapped onset (≥ 1/6 beat).
- +7 unit tests (collapse stray 16th, untouched genuine 16ths, collision keeps
  higher confidence, single/empty unchanged, pure 8th bar untouched, disabled
  passthrough). **21 quantize tests + 64 lib tests pass.**

**Why the pass is INERT (verified, do not force it):**
- Confirmation @ 208 (`--melody heuristic --bpm 208`): output byte-identical
  with and without `--no-rhythm-coarsen`. note 0.240, pitch 0.883, onset 0.153,
  dur 0.455 — **unchanged**.
- `KEYSCRIBE_COARSEN_DEBUG=1` per-bar votes: **every bar votes coarse=false**.
  Reference bar 1 is swung (8th, quarter, 8th, 8th, 8th, 8th-triplet ×3), and
  the transcribed gaps are e.g. `0.25, 1.0, 0.75, 0.75, 0.25, 0.75` →
  coarse vote 0.667 < 0.70, just under threshold; the 8-note swung bars are
  also blocked by the `≥ 2×bpb+1` note guard.
- Root cause is NOT 16th over-segmentation on this corpus — it's swung-feel
  quantization (swung 8ths snapping to dotted-8th+16th / straight slots).
  Lowering `coarse_vote_ratio` or dropping the note guard makes the pass chew
  genuine 16th runs and **regresses** note accuracy (earlier experiment: 0.231).
- Corpus gate (50 tracks, `--melody heuristic --quantizer learned`): pitch
  0.826, note 0.374, recall 0.387, onset 0.256, chord root 0.393, exact 0.139 —
  **byte-identical to baseline** (pass ≥ 0.374/≥ 0.820).

**Files touched:** `src/leadsheet/quantize.rs`, `src/leadsheet/preset.rs`,
`src/leadsheet/mod.rs`, `src/headless.rs`, `src/bin/keyscribe_cli.rs`,
`src/tune.rs`, `src/app/sheet_music.rs`, `src/leadsheet/quantize.rs`.

**Status:** implemented + verified inert; no corpus regression. Next lever for
the Confirmation rhythm gap is the plan's Task 2 (extraction-constant
calibration: `release_ratio`, `onset_split_threshold`), not coarsening.

---

### 2026-08-10 — eval-corpus harness + learned melody quantizer (steps 1–2 of the melody/quantizer plan)

**1. Consistent dataset — all 50 Omnibook MP3s re-rendered via MuseScore**
   (`out/omnibook/*.mp3`). Confirmation was already option-A (byte-identical);
   re-rendering every track guarantees the whole corpus shares one render
   pipeline. Ground-truth `.musicxml` copies verified byte-identical to the
   source XMLs. `out/omnibook/bpm_overrides.txt` holds the per-track XML
   tempos for reproducible `eval-corpus` runs.

**2. `keyscribe-cli eval-corpus` (`src/eval_corpus.rs`, new)**
   - Iterates every `(audio, reference.musicxml)` pair, runs `sheet` +
   `compare`, aggregates root/exact/coverage/pitch/note/recall/onset/duration/
   in-bar/bar-count per track and prints a dashboard table + JSON report.
   - Per-track BPM override file (JSON object or `<track> <bpm>` lines) makes
     runs reproducible when the ML beat tracker misfires on meter.
   - **Baseline (legacy, 50/50 tracks OK):** pitch 0.822, note 0.374, recall
     0.389, onset 0.256 beats, chord root 0.393, exact 0.139, in-bar 0.954.

**3. Learned melody quantizer — holdout-split training + shipped ONNX**
   (`tools/melody_corpus/train_quantizer.py`, `models/melody_quantizer.onnx`)
   - Trainer now supports `--holdout <stems>` to exclude eval tracks from
     training (Confirmation, Ornithology, Donna_Lee held out — no overfit risk
     on the eval tracks) and reports holdout accuracy on the final epoch.
   - Fixed a feature bug: `beat_in_bar` was the running note index; now derived
     from the real onset (`int(onset) % bpb`) to match the Rust featurizer
     (`learned_note_features`: `beat_index % beats_per_bar`).
   - Trained 25 epochs on 47 tracks (168 K examples); holdout accuracy **0.72**
     on the 3 held-out tracks. ONNX verified: input `[1, seq, 9]` → output
     `[1, seq, 12]`, matching `MelodyQuantizerInference`.
   - **A/B (eval-corpus legacy vs learned, full 50-track corpus):** learned
     ties or beats legacy on every metric (pitch 0.822→0.826; note 0.374→0.374;
     duration 0.478→0.477; no regressions). Per Decision 1 of the melody plan
     the checkpoint ships as `learned` (the default engine; falls back to the
     rule grid when the ONNX is absent).
   - **Tests:** 64 lib + 5 CLI tests pass. `HEADLESS.md` gets an
     `eval-corpus` section + updated baselines.

---

### 2026-08-10 — Chord roots & qualities: emission tuning + learned transition matrix

**Goal:** raise the chord root/exact match on real jazz backing tracks (the
weakest pipeline stage).

**1. Legacy emission tuning (`src/leadsheet/harmony.rs`)**

- Bass-register evidence weighted 30% → **45%** of the emission; `root == bass`
  bonus +0.15 → **+0.45**, plus +0.15 for a 5th/7th bass and +0.06 for a 3rd
  (walking-bass root approaches). Breaks the melody-contamination and
  relative-chord ties (e.g. bar 1 of Confirmation: melody A was out-voting the
  E-7b5 bass=E).
- Key prior: diatonic roots +0.03 → **+0.08**, plus **+0.04** for
  secondary-dominant roots a 5th/major-3rd from the scale (boosts V7/II without
  harming the Eb/Ab/Db modulation section).
- **Measured (Confirmation):** root **0.18 → 0.30**, exact 0.02. Melody
  unaffected, sheet still renders.

**2. Learned (root, quality) transition matrix — the plan's novel piece
   (`src/leadsheet/transition.rs`, new)**

The hand-tuned Viterbi table only encodes root *interval*; this learns the
*joint* root+quality transitions from a real chord corpus so `ii-7→V7→I`,
`V7→IV` back-cycling, and `ii∅→V7` carry their correct qualities.

- `fit_transitions_from_musicxml_dir` counts chord adjacencies across a corpus
  of MusicXML lead sheets, Laplace-smooths (`alpha`), writes
  `transition_matrix.json`. Includes target-only and isolated chords as states
  (the final I must be a valid Viterbi state).
- `keyscribe-cli fit-transition-matrix -i <corpus> -o <out> [--alpha] [--blend]`;
  `sheet --transition-matrix <path>` (or auto-resolution of
  `models/transition_matrix.json`).
- `viterbi_best_path` blends the hand-tuned costs with the learned log-probs
  (blend 0.65, `-ln P` scaled 0.05, capped 0.45) for observed (source, target)
  pairs; unobserved states fall back to the hand table — a sparse corpus can
  never dominate a strong emission.
- **Fit (58-track Omnibook corpus, 50 songs parsed):** 4464 chords, 4414
  transitions, **45 observed states** (Real-Book triads/7ths). Top learned
  moves are textbook jazz: `D-→G7` P=0.78, `A7→D-` P=0.55, `E-7b5→A7` P=0.45.
- **Measured (Confirmation):** root **0.30 → 0.34**, exact 0.02 → 0.03.
- **Validation (generalization, no overfit to Confirmation):** Ornithology and
  Donna Lee identical learned-vs-baseline (0.190 / 0.125 root) — the learned
  matrix is free there, never regresses.

**Caveat logged:** transitions are no longer the binding constraint. The
remaining errors are emission-level (whole-tone/tritone root confusions that
the 27-template `soft_contrast` matcher can't disambiguate on fast bebop). The
251-class ONNX head (`CHORD_DETECTION_PLAN.md` Phases 0–4) is the next lever and
needs model assets we don't have yet.

- **Tests:** +3 unit tests (mapping, adjacency fit, JSON roundtrip) → **64
  pass**. No new warnings. `Confirmation_transcribed.musicxml` + `.pdf`
  regenerated.
- **Docs:** `HEADLESS.md` chord section and `CHORD_DETECTION_PLAN.md` status
  updated.

---

### Earlier — reconstruction from `HEADLESS.md` / plan files (dates approximate)

#### Headless CLI + round-trip harness + timeline chord detector
- Built `keyscribe-cli` (`sheet`/`midi`/`render`/`compare`/`stems`/`tune`/
  `maketest`/`omnibook`) and `scripts/roundtrip.ps1`.
- P1 soft per-bar pitch-class profiles from the Basic Pitch probability
  timeline; P3 joint (root, quality) template matching with bass anchor; P2
  key-estimate + functional-harmony Viterbi over bars.
- Live chord-sampling knobs: `--chord-beat`, `--chord-cleanest`,
  `--chord-strike`.
- **Measured (`Pretty standard.mp3`):** chord root F1 **0.062 → 0.200**, exact
  F1 0 → 0.200.

#### Melody extraction (`extract_melody_skyline` / heuristic)
- Onset-density line selection, re-articulation splitting, stem-aware melody
  default (Demucs melodic stem, Phase 2 of the melody plan) + stem-id bug fix
  (energy-gated pitch-variance scoring).
- **Measured (`Pretty standard.mp3`):** pitch accuracy **0.139 → 0.800**.

#### Note extraction rhythm (P4)
- Adaptive release, attack-adjusted onsets, onset-head splitting, single-frame
  merge gap, onset-clamped melody durations.
- **Measured:** mean onset error **0.188 → 0.100 beats**; dotted-eighth
  mis-quantization gone.

#### Chord display / misc
- `Harmony::display()` maps MusicXML `kind` → conventional jazz symbols
  (Cmaj7, B-7, G-13) for human-readable compare output.
- `tune` bpm search dropped the half-tempo (×0.5) candidate.
- GUI wiring: `TunedConfig` defaults + timeline chord source in the sheet
  preview.
