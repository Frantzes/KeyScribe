# Implementation Plan Index — 2026-08-14

Audio-to-sheet-music pipeline improvement program. Four dated, self-contained
plan files written to be executable by an AI implementer without extra
context. Read the target plan file COMPLETELY before touching code.

## Diagnosis (why these four plans)

Measured on Confirmation @ 208 bpm (pitch 0.883, note 0.240, root 0.340,
exact 0.030) and the 50-track Omnibook corpus (pitch 0.826, note 0.374,
root 0.393, exact 0.139):

1. **Rhythm** — the extractor (`src/headless.rs:719` `extract_notes_from_
   timeline`) over-splits notes (onset-head ≥ 0.35 splits, 0.55×peak
   release), the straight grid admits 16th slots (`src/leadsheet/quantize.rs:
   664`), and NOTHING downstream merges fragments back. The learned
   quantizer is a per-note MLP with no sequence context, trained only on
   jittered clean notes — it cannot merge what it has never seen.
2. **Chords** — emission-bound (transitions are already learned). The
   27-template matcher hallucinates extensions (minor-11th/major-13th) where
   references have plain 7ths, and segmentation is hard-coded to one chord
   per bar while bebop changes twice per bar.

## The four plans (execute in this order)

| # | File | Front | Effort | Depends on |
|---|---|---|---|---|
| A1 | `2026-08-14_TIER_A1_rhythm_merge_coarsening.md` | rhythm | 1-2 days | — |
| A2 | `2026-08-14_TIER_A2_chord_collapse_sevenths_halfbar.md` | chords | 1-2 days | — |
| B1 | `2026-08-14_TIER_B1_sequence_quantizer_onnx.md` | rhythm | 2-3 days | A1 helps; not required |
| B2 | `2026-08-14_TIER_B2_chord_head_crnn_onnx.md` | chords | 4-6 days | A2 Part 1 required |

A1 and A2 are independent — may run in parallel. B1 and B2 are independent
of each other.

## Locked decisions (apply everywhere)

- **User decision 1**: Python/PyTorch allowed for TRAINING; the shipped
  artifact must always be an ONNX model run via the existing `ort` runtime.
  No Python at inference.
- **User decision 2**: jazz chord extensions COLLAPSE to the nearest
  7th-chord class at output (mirrors `CHORD_DETECTION_PLAN.md` Decision 2).
- Never train on the holdout tracks: `Confirmation`, `Ornithology`,
  `Donna_Lee`.
- Never regress the 50-track corpus baseline
  (`eval-corpus`, `--melody heuristic`, fixed bpm file):
  pitch 0.822/0.826, note 0.374, recall 0.389, onset 0.256, root 0.393,
  exact 0.139.
- Every plan ends by updating `HEADLESS.md` + `PROGRESS_JOURNAL.md` and
  flipping its own Status line to DONE with final numbers.
- Do not commit unless the user explicitly asks.

## Metric targets

| Metric | Now | After A1+B1 | After A2+B2 |
|---|---|---|---|
| Confirmation note acc | 0.240 | ≥ 0.35 | — |
| Confirmation duration err | 0.455 | ≤ 0.35 | — |
| Confirmation chord root | 0.340 | — | ≥ 0.40 |
| Confirmation chord exact | 0.030 | — | ≥ 0.10 |
| Corpus note acc | 0.374 | ≥ 0.42 | — |
| Corpus chord exact | 0.139 | — | ≥ 0.139 (floor) |

## Shared verification commands

```powershell
cargo test --no-default-features --lib
cargo check --bin keyscribe
cargo build --release --no-default-features --bin keyscribe-cli
$cli = "target\release\keyscribe-cli.exe"
& $cli sheet out\omnibook\Confirmation.mp3 -o $env:TEMP\opencode\conf.musicxml --bpm 208 --melody heuristic
& $cli compare out\omnibook\Confirmation.musicxml $env:TEMP\opencode\conf.musicxml
& $cli eval-corpus out\omnibook --bpm-file out\omnibook\bpm_overrides.txt --melody heuristic -o $env:TEMP\opencode\corpus.json
```

## Relationship to the existing plan docs

- `CHORD_DETECTION_PLAN.md` — B2 is a pragmatic intermediate step (features
  the pipeline already computes, corpus we can already render) toward its
  251-class audio-domain head, which remains the Tier-2 successor.
- `MELODY_TRANSCRIPTION_PLAN.md` — B1 IS its Phase 3 done properly (sequence
  model + over-segmentation augmentation instead of the per-note MLP
  bootstrap), plus A1 adds the rule-based merge pass that plan never had.
