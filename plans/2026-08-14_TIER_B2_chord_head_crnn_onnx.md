# Tier B2 — Learned chord head (BiGRU over bar profiles) exported to ONNX

Date: 2026-08-14
Status: NOT STARTED
Owner: implementer (any AI model — follow steps exactly, in order)
Prerequisite: Tier A2 Part 1 (`collapse_quality_to_seventh`) should be merged
first — this plan reuses it.
Hard constraint from the user: **training may be Python/PyTorch, but the
shipped artifact MUST be an ONNX model run through the existing `ort` runtime
in Rust. No Python at inference time.**

## Goal

Replace the hand-crafted 27-template `soft_contrast` EMISSION (the proven
binding constraint — the journal says so explicitly) with a learned per-bar
chord classifier, while KEEPING the existing Viterbi + learned transition
matrix + bass/key priors infrastructure.

This is a deliberate simplification of `CHORD_DETECTION_PLAN.md`'s 251-class
BTC/ChordFormer route (which is blocked on external model assets): we train
on the synthetic corpus we can already generate (MuseScore renders of
lead-sheet MusicXML), using features the Rust pipeline ALREADY computes
(`compute_bar_profiles`), so there is no feature-parity risk and no raw-audio
plumbing needed.

Current: Confirmation root **0.340**, exact **0.030**; corpus root 0.393,
exact 0.139.

Targets:

- Confirmation: root ≥ **0.40**, exact ≥ **0.10**
- 50-track corpus: root ≥ **0.393**, exact ≥ **0.139** (no regression)

## Locked design decisions (do not deviate)

1. **Features** = per-bar `(comp_pcp[12], bass_pcp[12], energy[1])` — 25
   floats — exactly what `compute_bar_profiles` (`src/leadsheet/harmony.rs:
   109-301`) already produces on `BarProfile` (fields `pcp`, `bass_pcp`,
   `energy`) BEFORE whitening. Training obtains them by RUNNING THE RUST CLI
   (new `chord-features` command, Step 1) so train and inference features are
   byte-identical by construction.
2. **Vocabulary = 85 classes**: 12 roots × 7 qualities
   `{"", "-", "7", "Δ7", "-7", "-7b5", "dim7"}` (the 7th-class keep-set from
   Tier A2) + index 84 = `N.C.` (no chord / low-energy bar).
3. **Model**: `nn.GRU(25, 64, num_layers=2, bidirectional=True, batch_first=
   True)` → `nn.Linear(128, 85)`. Sequence dimension = BARS.
4. **Output artifact**: `models/chord_head.onnx`, input `[1, n_bars, 25]`
   float32, output `[1, n_bars, 85]` float32 (logits).
5. **Decoding in Rust**: learned logits are BLENDED into the existing
   324-state emission grid (add `blend_weight × log_softmax` at the matching
   `(root, quality)` states), then the EXISTING Viterbi + learned transition
   matrix runs unchanged. Emission replacement, not pipeline replacement.

## Step-by-step

### Step 1 — Rust: `chord-features` CLI command (feature dumper)

Add to `src/bin/keyscribe_cli.rs` (mirror the `compare` command's argument
parsing style) and a handler in `src/headless.rs`:

```
keyscribe-cli chord-features <audio> [--bpm <f32>] [--stems] -o features.json
```

Behavior:
1. Run the same `analyze_audio` path `sheet` uses (bpm override supported —
   training corpus uses the XML tempos from `out/omnibook/bpm_overrides.txt`).
2. Call `compute_bar_profiles` with the default `ChordAnalysisConfig`.
3. Write JSON: `{"step": <bpb>, "bars": [{"pcp": [12 f32], "bass_pcp": [12
   f32], "energy": f32}, ...]}` (round to 6 decimals).

Unit test: run on a synthetic timeline, assert bar count and that each pcp
sums into [0, 12]. Manual check: dump `out/omnibook/Confirmation.mp3` with
`--bpm 208` and confirm ~57 bars (Confirmation is 57 bars incl. pickup —
verify against the reference MusicXML bar count and note the real number
here when done).

### Step 2 — Python: corpus builder + labels

New file `tools/chord_corpus/build_dataset.py`:

1. Inputs: a folder of `(audio, reference.musicxml)` pairs (start with
   `out/omnibook/`, 50 tracks; expand per Step 5).
2. For each pair:
   - run `keyscribe-cli chord-features <audio> --bpm <xml tempo> -o f.json`
     (subprocess; tempos from `out/omnibook/bpm_overrides.txt`);
   - parse the reference MusicXML harmonies with `xml.etree.ElementTree`:
     each `<harmony>` gives `<root><root-step>`/`<root-alter>` and `<kind>`.
     Map MusicXML `kind` strings to the 7-class suffix with EXACTLY this
     table (add rows only if you hit an unmapped kind — then record it):
     | MusicXML kind | class suffix |
     |---|---|
     | major | "" |
     | minor | "-" |
     | dominant | "7" |
     | major-seventh | "Δ7" |
     | minor-seventh | "-7" |
     | half-diminished | "-7b5" |
     | diminished-seventh | "dim7" |
     | diminished | "dim7" |
     | augmented, sus, sus2, sus4, 6, major-6, minor-6, and ALL "ninth"/
       "11th"/"13th"/"altered" variants | collapse per
       `collapse_quality_to_seventh` semantics (Tier A2 table) |
   - map root step+alter → pitch class 0-11 (C=0 … B=11, `alter` is a
     child element with text like `-1`/`1`).
   - align harmonies to bars: k-th `<haronomy>`… careful: k-th `<harmony>`
     element lands in the bar of the note that follows it in document order —
     walk the `<part>` XML in order, tracking the current `<measure>` number;
     if a measure has several harmonies, keep the FIRST (labels are
     bar-level; note the limitation).
   - emit one row per bar: `{"features": [25 f32], "label": 0..84, "nc":
     bool}`; bars before the first harmony / with no harmony → label 84
     (N.C.).
3. **Alignment sanity check (mandatory)**: for 3 random tracks, print the
   label chord roots per bar next to the top-3 pcs of the bar profile; the
   root must appear in the top-4 pcs in ≥ 80% of bars. If not, the bar grid
   is misaligned (bpm/offset) — fix the `--bpm` value for that track before
   proceeding.
4. Split by song: train = all tracks EXCEPT `Confirmation`, `Ornithology`,
   `Donna_Lee` (holdout, never trained on).
5. **Augmentation (cheap, big win)**: for each training song, emit 11
   transposed copies (rotate both pcp arrays by +1…+11 semitones; bass_pcp
   too; labels' root rotated the same amount; energy unchanged).

### Step 3 — Python: train + export ONNX

New file `tools/chord_corpus/train_chord_head.py`:

1. Model per Locked design §3. LayerNorm on the input features helps (add
   `nn.LayerNorm(25)` before the GRU — then ALSO apply the identical
   normalization inside the exported graph, i.e. make it part of the module;
   do NOT normalize only in Python).
2. Loss: cross-entropy, class-weighted inversely by frequency (N.C. is
   frequent; cap its weight at 1.0 so the model can't win by predicting
   N.C.).
3. Train up to 100 epochs, early stop on holdout song-level accuracy.
   Metrics to report: overall bar accuracy, root-only accuracy, exact
   (root+quality) accuracy on the 3 holdout tracks.
4. Export: `torch.onnx.export(model, dummy [1, 40, 25], "models/
   chord_head.onnx", dynamic_axes={...time...})`. Verify in Python with
   `onnxruntime`: input `[1, 7, 25]` → `[1, 7, 85]`.
   Acceptance for the checkpoint: holdout root accuracy ≥ 0.55 AND exact ≥
   0.30 at the bar level (bar-level on clean synthetic renders; end-to-end
   numbers will be lower).

### Step 4 — Rust: learned chord engine + blending

1. `src/inference.rs`: new `ChordHeadInference` beside
   `MelodyQuantizerInference` (copy its structure): loads
   `models/chord_head.onnx` (resolve like `resolve_quantizer_model_path`,
   new fn `resolve_chord_head_path`), method
   `infer(&self, features: &[Vec<f32>]) -> Result<Vec<[f32; 85]>>`.
2. `src/headless.rs`: 
   ```rust
   pub enum ChordEngine { LegacyTemplates, LearnedOnnx }
   ```
   Resolution: `LearnedOnnx` when `models/chord_head.onnx` resolves, else
   `LegacyTemplates`. CLI flag `--chord-engine legacy|learned` overrides;
   `TunedConfig.chord_engine: String` (serde default `"legacy"`) overrides;
   flag > config > presence-of-file.
3. `src/leadsheet/harmony.rs`: extend `detect_chords_from_timeline` — after
   the emission grid is built (harmony.rs:509-558) and when the learned
   engine is active:
   - build `[n_bars, 25]` from the profiles;
   - `infer` → per-bar `[85]` logits → `log_softmax` (write a tiny local
     softmax, no dependency);
   - for each class c < 84: root = c / 7, quality index = position of the
     class suffix in `ALL_TEMPLATES` (build a small lookup
     `SEVENTH_CLASS_TO_TEMPLATE_INDEX: [(suffix, usize); 7]` by scanning
     `ALL_TEMPLATES` at init); add `blend_weight * logprob[c]` to
     `emissions[bar][root * 27 + q]`. `blend_weight = 1.5` to start.
   - N.C. (class 84): subtract `blend_weight * logprob[84]` from ALL states
     of that bar (pulls silent bars toward "skip" via the energy gate).
   - The Viterbi, learned transitions, energy gate, dedupe, and the Tier A2
     collapse all run unchanged afterwards.
4. `SheetOptions` / `generate_lead_sheet_enhanced_with_timeline`
   (`src/leadsheet/preset.rs:292`): thread the engine through (grep how
   `chord_beat` flows today and copy the pattern).

Tests: a mock 85-logit fixture (write a tiny ONNX by hand in Python — a
`Linear(25→85)` with zero weights plus a fixed bias — commit it under
`tests/data/chord_head_test.onnx`); assert that with the engine forced
learned, argmax classes map to the expected symbols. Skip gracefully when
neither ONNX exists.

### Step 5 — Corpus expansion (do after Step 4 first works on 50 tracks)

1. Find more lead-sheet MusicXML (the repo's transition-matrix corpus used
   58 Omnibook tracks / 50 songs — locate that folder via
   `keyscribe-cli fit-transition-matrix` usage in `PROGRESS_JOURNAL.md`;
   same source). Render audio with `keyscribe-cli render` (MP3 only — WAV
   export fails, see HEADLESS.md gotchas). Target: **≥ 150 songs**.
2. Re-run Steps 2-3, re-export the ONNX, re-run the Step 6 gate. Expect the
   biggest jump here (50 songs is thin for 85 classes).

### Step 6 — A/B + regression gate (mandatory)

```
cargo test --no-default-features --lib
cargo build --release --no-default-features --bin keyscribe-cli
target\release\keyscribe-cli.exe sheet out\omnibook\Confirmation.mp3 -o $env:TEMP\opencode\conf_b2l.musicxml --bpm 208 --chord-engine legacy
target\release\keyscribe-cli.exe sheet out\omnibook\Confirmation.mp3 -o $env:TEMP\opencode\conf_b2n.musicxml --bpm 208 --chord-engine learned
target\release\keyscribe-cli.exe compare out\omnibook\Confirmation.musicxml $env:TEMP\opencode\conf_b2l.musicxml
target\release\keyscribe-cli.exe compare out\omnibook\Confirmation.musicxml $env:TEMP\opencode\conf_b2n.musicxml
target\release\keyscribe-cli.exe eval-corpus out\omnibook --bpm-file out\omnibook\bpm_overrides.txt --melody heuristic --chord-engine learned -o $env:TEMP\opencode\corpus_b2.json
```
Note: if `eval-corpus` lacks a `--chord-engine` flag, add it (copy the
`--quantizer` plumbing in `src/eval_corpus.rs`).

- PASS: targets above met.
- FAIL/rollback: `chord_head.onnx` stays in `models/` but the DEFAULT flips
  back to `LegacyTemplates` (presence-of-file rule inverted until green);
  document the gap in this file.

### Step 7 — Docs

- `PROGRESS_JOURNAL.md` dated entry + dashboard.
- `HEADLESS.md`: new section "## Chord detection: learned bar-profile head"
  (engine enum, blending, model file, flags, measured numbers).
- `CHORD_DETECTION_PLAN.md`: status note — the bar-profile BiGRU shipped as
  an intermediate step; the 251-class audio-domain head remains the Tier-2
  successor.
- This file: Status → DONE + final numbers.

## Guardrails (do NOT)

- Do NOT remove or mutate `soft_contrast` / `compute_bar_profiles` math —
  the learned engine only ADDS to the emission grid.
- Do NOT train on the 3 holdout tracks (ever).
- Do NOT hardcode the 27-template indices — derive the lookup from
  `ALL_TEMPLATES` at runtime.
- Do NOT ship Python in the runtime — ONNX only.
- Do NOT commit unless the user explicitly asks.
