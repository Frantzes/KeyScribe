# Tier B1 — Sequence-model melody quantizer with merge tokens (ONNX)

Date: 2026-08-14
Status: NOT STARTED
Owner: implementer (any AI model — follow steps exactly, in order)
Prerequisite: Tier A1 (`2026-08-14_TIER_A1_rhythm_merge_coarsening.md`)
helps but is not strictly required; do A1 first if possible.
Hard constraint from the user: **training may be Python/PyTorch, but the
shipped artifact MUST be an ONNX model run through the existing `ort` runtime
in Rust. No Python at inference time.**

## Goal

Replace the per-note MLP quantizer (which cannot fix over-segmentation because
it sees one note at a time) with a **sequence model that can merge detection
fragments back into the logical note** and pick the correct duration token
with neighbor context.

Current state (verified):

- Model: `MelodyQuantizerInference` (`src/inference.rs:308-408`), input
  `[1, seq, 9]`, output `[1, seq, 12]`. Architecture is a per-note MLP
  9→64→64→12 (`tools/melody_corpus/train_quantizer.py:110-119`) — **zero
  sequence context**.
- Tokens: `LEARNED_TOKEN_TABLE` (`src/leadsheet/quantize.rs:960-973`), 12
  duration values (whole … 16th + 3 triplets).
- Rust consumer: `quantize_aligned_notes_learned`
  (`src/leadsheet/quantize.rs:1035-1196`): onsets still snapped by the rule
  grid (quantize.rs:1120-1122), duration = token beats (:1127), per-bar
  legacy fallback (:1147-1186).
- Training data: Omnibook MusicXML with jitter augmentation ONLY
  (`train_quantizer.py`, onset σ=0.035 beat) — the model has never seen a
  split note, so it labels fragments as 16ths.

Targets (50-track `eval-corpus`, `--quantizer learned`):

- note accuracy 0.374 → **≥ 0.42**
- mean duration error 0.477 → **≤ 0.40**
- pitch accuracy ≥ 0.826 maintained; onset error ≤ 0.256 beats maintained

## Design (locked — implement exactly)

### Vocabulary: 13 classes

Indexes 0-11 = the existing `LEARNED_TOKEN_TABLE` entries, UNCHANGED.
Index **12 = MERGE_INTO_PREVIOUS**: "this detection is a spurious fragment of
the previous detection — do not emit a note; the previous note covers it."

### Semantics

- For the FIRST detection of a logical note, the target label is the logical
  note's duration token (even though the detection's raw duration is only a
  fragment of it). The model infers this from sequence context: the fragment
  is followed immediately by further fragments with near-zero gaps.
- For every SUBSEQUENT fragment of the same logical note, the target label is
  12 (MERGE_INTO_PREVIOUS).

### Architecture

- Input `[batch, seq, 9]` — the SAME 9 features as today
  (`learned_note_features`, quantize.rs:980-1005; do not change the
  featurizer).
- Model: `nn.GRU(input=9, hidden=64, num_layers=2, bidirectional=True,
  batch_first=True)` → `nn.Linear(128, 13)`. (~100 K params, exports to ONNX
  trivially, well under 1 MB.)
- Output `[batch, seq, 13]` logits, per-detection argmax at inference.

## Step-by-step

### Step 1 — Retrain in Python with split augmentation

File: `tools/melody_corpus/train_quantizer.py` (extend, don't rewrite).

1. Keep the existing feature builder and Omnibook MusicXML supervision.
2. Add augmentation to the training loop (keep the existing jitter aug):
   For each note in a training sequence, with probability `p_split = 0.35`:
   - Choose `k ∈ {2, 3}` fragments uniformly.
   - Split the note's duration into `k` near-equal parts (±20% jitter).
   - Emit `k` feature rows: the first keeps the note's true onset-derived
     features (intra-beat pos, beat index, bar); each continuation gets its
     own (jittered) onset position and its fragment's raw duration.
   - Labels: first row = the note's true token; continuation rows = 12.
3. Model class per the Architecture section; loss = cross-entropy with class
   weights (MERGE is frequent after augmentation — weight it 0.5×, not higher,
   so the model doesn't over-merge).
4. Holdout: keep `--holdout Confirmation Ornithology Donna_Lee` EXACTLY as-is
   (never train on the eval tracks). Report: token accuracy AND merge
   precision/recall on the holdout.
5. Export ONNX (`torch.onnx.export`, `dynamic_axes` for the sequence axis):
   - File name: `models/melody_quantizer.onnx` (replaces v1).
   - FIRST back up the old model: copy it to
     `models/melody_quantizer_v1_mlp.onnx` (this is the rollback artifact).
   - Input `[1, seq, 9]` float32, output `[1, seq, 13]` float32.
   - Verify with `onnxruntime` in Python: random input `[1, 50, 9]` →
     output shape `[1, 50, 13]`.

Acceptance for the checkpoint: holdout token accuracy ≥ 0.70 (v1 was 0.72)
AND merge recall ≥ 0.6 AND merge precision ≥ 0.8. If not met, train longer /
tune `p_split` ∈ {0.25, 0.5} before touching Rust.

### Step 2 — Rust inference layer

1. `src/inference.rs` `MelodyQuantizerInference`: the output-dimension
   handling must accept ANY vocab size ≥ 12 (it currently assumes 12 —
   generalize the argmax loop; log a warning if vocab < 13 and treat every
   index as a duration token, i.e. v1 models keep working).
2. `src/leadsheet/quantize.rs` `quantize_aligned_notes_learned`:
   after `infer.infer(&features)` returns `tokens` (quantize.rs:1105-1111):
   - Walk the token list IN ORDER (features were built time-sorted,
     quantize.rs:1083-1088 — keep that order).
   - If `tokens[i] == 12` (MERGE): do NOT create a QuantizedNote for
     detection `i`. Instead find the previously-emitted note for the SAME
     pitch (or simply the previous emitted note — melody is monophonic) and
     extend its coverage: set its `beat_duration` to
     `max(beat_duration, next_kept_snapped_onset - beat_start)` where
     `next_kept_snapped_onset` is the snapped onset of the next NON-merged
     detection (or the bar end if none). Clamp ≥ 1/6 beat.
   - Do not merge across a bar boundary if the previous note is in a
     different bar — instead DROP the fragment silently (a fragment that
     spills over a bar line is extraction noise; dropping is safe).
3. Leave the per-bar legacy fallback (quantize.rs:1147-1186) exactly as-is;
   it operates on the merged output and stays the safety net.

### Step 3 — Tests

- Unit test in `quantize.rs`: fabricate a feature sequence whose tokens are
  `[4, 12, 6, 12, 12]` (quarter, merge, eighth, merge, merge) via a mock or a
  test-only injection path — verify 2 notes are emitted and durations span
  the merged span. (If `MelodyQuantizerInference` can't be mocked cheaply,
  factor the merge walk into a pure fn `apply_merge_tokens(tokens: &[u8],
  ...) -> ...` and unit-test THAT; the ONNX path calls it.)
- The existing quantizer tests must pass unchanged.

### Step 4 — A/B + regression gate

```
cargo test --no-default-features --lib
cargo build --release --no-default-features --bin keyscribe-cli
target\release\keyscribe-cli.exe eval-corpus out\omnibook --bpm-file out\omnibook\bpm_overrides.txt --melody heuristic --quantizer learned -o $env:TEMP\opencode\corpus_b1_learned.json
target\release\keyscribe-cli.exe eval-corpus out\omnibook --bpm-file out\omnibook\bpm_overrides.txt --melody heuristic --quantizer legacy  -o $env:TEMP\opencode\corpus_b1_legacy.json
```

- PASS: learned note ≥ 0.42, duration error ≤ 0.40, pitch ≥ 0.826, onset
  error ≤ 0.256.
- FAIL/rollback: restore `models/melody_quantizer_v1_mlp.onnx` over
  `models/melody_quantizer.onnx`, keep the Rust changes (they are
  backward-compatible with the 12-token model), and record the numbers in
  this file. Re-attempt only after Step 1's acceptance numbers improve.

### Step 5 — Docs

- `PROGRESS_JOURNAL.md` dated entry + dashboard rows (learned column).
- `HEADLESS.md`: update the learned-quantizer description (sequence model,
  MERGE token, v1 backup path, new interface `[1,seq,9]→[1,seq,13]`).
- `MELODY_TRANSCRIPTION_PLAN.md`: Phase 3 progress log entry.
- This file: Status → DONE + final numbers.

## Guardrails (do NOT)

- Do NOT change `learned_note_features` (the 9-dim featurizer) — the whole
  point is train/inference feature parity.
- Do NOT train on Confirmation / Ornithology / Donna_Lee (holdout stays).
- Do NOT remove the legacy quantizer or the per-bar fallback.
- Do NOT ship a PyTorch dependency into the runtime — ONNX only.
- Do NOT commit unless the user explicitly asks.
