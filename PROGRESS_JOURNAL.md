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
| Confirmation | **0.340** | **0.030** | 100 | 0.851 | 0.248 |
| Ornithology | 0.190 | 0.016 | 63 | — | — |
| Donna Lee | 0.125 | 0.000 | 88 | — | — |

Chord-root progression on Confirmation: **0.18 → 0.30** (emission: bass anchor
+ key prior) → **0.34** (learned transition matrix).

Prior milestones (reconstructed, `HEADLESS.md`):
- `Pretty standard.mp3` chord root F1 **0.200** (was 0.062), exact F1 0.200
  (was 0.000).
- Melody pitch accuracy **0.800** (was 0.139) on `Pretty standard.mp3`.
- Note onset error **0.100 beats** (was 0.188) after the P4 extraction rework.

---

## Entries

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
