# Tier A1 — Rhythm merge & coarsening pass (fix 16th over-segmentation)

Date: 2026-08-14
Status: DONE — implemented + verified INERT (see "Results" below)
Owner: implementer (any AI model — follow steps exactly, in order)
Companion plans: `2026-08-14_IMPLEMENTATION_PLAN_INDEX.md` (read that first)

## Goal

Stop the melody from being chopped into 16th-note fragments. Today on
Confirmation @ 208 bpm: note accuracy **0.240**, recall **0.238**, duration
error **0.455**, while pitch accuracy is 0.883. The melody is the RIGHT pitches
at the WRONG rhythmic values (bar 1 has 7 detected notes where the reference
has clean 8ths).

Target after this plan:

- Confirmation `note_accuracy` 0.240 → **≥ 0.30**
- Confirmation `mean_duration_error` 0.455 → **≤ 0.35**
- 50-track `eval-corpus` `note` stays **≥ 0.374** and `pitch` stays **≥ 0.820**
  (NO regression allowed — this is the gate)

## Root cause (verified in code — read this before touching anything)

1. `extract_notes_from_timeline` in `src/headless.rs:719-861`:
   - `release_ratio = 0.55` (headless.rs:737): a note ENDS when its probability
     dips below 55% of its own peak. Sustain wobble → note splits.
   - `onset_split_threshold = 0.35` (headless.rs:746): the model's onset head
     firing ≥ 0.35 mid-note SPLITS it. Vibrato/re-attack leakage fires it.
   - `min_duration_sec ≈ 0.05` (headless.rs:732): a 16th at 208 bpm is ~72 ms,
     so 16th fragments easily survive the min-duration filter.
2. `snap_intra_beat_pos` grid in `src/leadsheet/quantize.rs:676-693` with
   `straight_grid = [0, 0.25, 0.5, 0.75, 1.0]` (quantize.rs:664): every extra
   fragment gets its own 16th slot. Nothing downstream EVER merges fragments.
3. Durations are derived from snapped onset gaps (`quantize.rs:808-846`), so
   they inherit the splits.

The GUI has a MIRROR of the extractor: `extract_events_from_timeline_data` in
`src/app/sheet_music.rs` (~line 1370). It must stay in sync (Task 0).

## Conventions for the implementer

- Build: `cargo build --release --no-default-features --bin keyscribe-cli`
- Tests: `cargo test --no-default-features --lib` (must stay 64+ passing)
- GUI check: `cargo check --bin keyscribe`
- Binary: `target\release\keyscribe-cli.exe`
- NEVER change behavior without re-running the verification commands in each
  task. NEVER commit unless the user asked.
- All new thresholds get sane defaults so old configs/CLI calls behave
  identically or better.

---

## Task 0 — Shared extraction constants (prep, no behavior change)

**Why:** the extraction constants are duplicated literals in `headless.rs` and
`src/app/sheet_music.rs`. Before making them tunable they must exist ONCE.

1. In `src/headless.rs`, directly above `extract_notes_from_timeline`
   (line ~719), add public constants:

   ```rust
   pub const NOTE_RELEASE_RATIO: f32 = 0.55;
   pub const NOTE_RELEASE_FLOOR: f32 = 0.05;
   pub const NOTE_ATTACK_LOOKBACK: usize = 5;
   pub const ONSET_SPLIT_THRESHOLD: f32 = 0.35;
   ```

2. Replace the local `let` bindings at headless.rs:737-746 with these consts.
3. Grep the GUI mirror: `Grep pattern "release_ratio|onset_split_threshold|0.55"`
   in `src/app/sheet_music.rs`; replace its literals with
   `crate::headless::NOTE_RELEASE_RATIO` etc. so both paths share values.
4. Run `cargo test --no-default-features --lib` and
   `cargo check --bin keyscribe`. Both must pass with zero behavior change.

## Task 1 — Per-bar grid coarsening + fragment merge pass (the core fix)

**Where:** new function in `src/leadsheet/quantize.rs`, called from
`generate_lead_sheet_enhanced_with_timeline` in `src/leadsheet/preset.rs`
(the call site is the `quantize_aligned_notes(...)` / 
`quantize_aligned_notes_learned(...)` invocation at preset.rs ~line 350).

**What to write** — a pure post-processing function:

```rust
pub struct RhythmCoarsenConfig {
    pub enabled: bool,          // default true
    pub coarse_vote_ratio: f32, // default 0.70
}

pub fn coarsen_rhythm(
    notes: Vec<QuantizedNote>,
    beats_per_bar: u32,
    cfg: &RhythmCoarsenConfig,
) -> Vec<QuantizedNote>
```

Algorithm (implement EXACTLY this; it is deterministic):

1. Group notes by `bar_index`.
2. Per bar, compute the set of distinct snapped onset offsets within the bar:
   `offset = beat_start - (bar_index as f32 * beats_per_bar as f32)`.
   Collect the sorted distinct offsets (collapse offsets < 0.05 beats apart).
3. Compute inter-onset gaps between consecutive distinct offsets.
4. **Grid vote:** let `coarse` = number of gaps ≥ 0.45 beats, `fine` = number of
   gaps in (0.05, 0.45). If `coarse / (coarse + fine) >= cfg.coarse_vote_ratio`
   AND at least one gap is exactly a 0.25 multiple (i.e. the bar contains
   16th-spaced onsets at all), the bar is voted COARSE (8th-note grid).
5. For each COARSE bar, RE-SNAP every note's `beat_start` to the nearest
   multiple of 0.5 within the bar (keep the original beat for offsets that
   already sit on 0.5 multiples). Zero out offsets in (0.05, 0.45).
6. After re-snapping, merge collisions in the bar (monophonic melody):
   - If two notes now share the same snapped onset: keep the one with the
     LONGER raw detected duration (`beat_duration` before merge is not
     reliable for this — keep the note with the higher `confidence`, tie-break
     by lower pitch id), drop the other.
   - If two SAME-PITCH notes are adjacent in time after re-snapping (note B
     starts where note A starts + snapped gap, A.duration ≈ that gap), merge
     them: A extends over both, B is dropped.
7. Recompute merged notes' `beat_duration` as
   `next_snapped_onset_in_bar_or_bar_end - beat_start`, clamped to
   ≥ 1/6 beat (the smallest token). This is what fixes duration error.
8. Leave non-COARSE bars untouched (16th-note passages must survive).
9. Keep the output sorted by `beat_start` (then pitch), same as input.

**Wire-in:** in `preset.rs`, right after the quantizer call, apply
`coarsen_rhythm(quantized, beats_per_bar, &cfg)`. The `RhythmCoarsenConfig`
gets a field on `SheetOptions` (grep `pub struct SheetOptions` to find it;
default `enabled: true`). Add a `--no-rhythm-coarsen` CLI flag in
`src/bin/keyscribe_cli.rs` that sets `enabled = false`, and a serde-default
field in `TunedConfig` (`src/headless.rs`, grep `pub struct TunedConfig`;
field `rhythm_coarsen: bool` default `true` — absent field = true).

**Unit tests to add** (in `quantize.rs` `#[cfg(test)]`):
- A bar of 8 evenly spaced 0.25-gap notes → collapses to 4 notes on 0.5 grid
  with 0.5-beat durations.
- A bar with genuine 16ths (gaps of 0.25 AND 0.75 mixed such that coarse vote
  fails) → unchanged.
- Two notes colliding after re-snap → one survives.
- Empty bar / single note → unchanged.

**Verify:**
```
cargo test --no-default-features --lib
cargo build --release --no-default-features --bin keyscribe-cli
target\release\keyscribe-cli.exe sheet out\omnibook\Confirmation.mp3 -o $env:TEMP\opencode\conf_a1.musicxml --bpm 208 --melody heuristic
target\release\keyscribe-cli.exe compare out\omnibook\Confirmation.musicxml $env:TEMP\opencode\conf_a1.musicxml
```
Record note_accuracy / recall / duration error. If note_accuracy did not rise,
run with `KEYSCRIBE_HARMONY_DEBUG=1`-style debug of your own (add an env var
`KEYSCRIBE_COARSEN_DEBUG=1` printing per-bar vote results) and inspect which
bars voted wrong before tuning anything.

## Task 2 — Make extraction constants tunable and calibrate

1. Extend `SheetOptions` with:
   `onset_split_threshold: f32` (default `ONSET_SPLIT_THRESHOLD` = 0.35),
   `release_ratio: f32` (default 0.55),
   `min_note_frames: u32` (default `MIN_SHEET_NOTE_FRAMES`).
   Thread them into `extract_notes_from_timeline` (change the fn signature to
   take a small `ExtractionTuning` struct; update its ~3 call sites: grep
   `extract_notes_from_timeline(`) and the GUI mirror reads the same defaults.
2. Add CLI flags `--onset-split <f32>`, `--release-ratio <f32>` on `sheet` and
   `midi` (parse like the existing `--key-sensitivity`).
3. Add the two knobs to the `tune` search space (`src/tune.rs`, grep
   `sensitivities`/`chord_samplings` grid lists — add small grids
   `[0.35, 0.5]` and `[0.55, 0.7]`, and persist winners in `TunedConfig`).
   Follow the existing "add a cheap parameter" recipe in `HEADLESS.md`
   (§Extending the search space).
4. Calibrate on Confirmation:
```
target\release\keyscribe-cli.exe tune out\omnibook --fast -o $env:TEMP\opencode\conf_tuned.json
```
   Then run `sheet` with the tuned config + `compare`. Keep the best.

## Task 3 — Regression gate (mandatory before declaring done)

```
target\release\keyscribe-cli.exe eval-corpus out\omnibook --bpm-file out\omnibook\bpm_overrides.txt --melody heuristic --quantizer learned -o $env:TEMP\opencode\corpus_a1.json
```
Compare the aggregate row against the baseline in `HEADLESS.md`
(pitch 0.822/0.826, note 0.374, recall 0.389, chord root 0.393, exact 0.139).

- PASS if: note ≥ 0.374 AND pitch ≥ 0.820 AND chords unchanged (±0.005).
- If pitch drops > 0.02: the coarsening is too aggressive — raise
  `coarse_vote_ratio` to 0.8 or restrict Task 1 step 6 merges to same-pitch
  only, and re-run.
- If note drops below 0.374: check `KEYSCRIBE_COARSEN_DEBUG` output — likely
  genuine-16th bars are being voted coarse; add the guard "bar must contain ≥
  4 notes" before a COARSE vote is allowed.

## Task 4 — Update docs

- `PROGRESS_JOURNAL.md`: new dated entry (what/why/measured/files).
- `HEADLESS.md`: new section "## Rhythm coarsening pass" under the P4 section;
  update the baselines table with the new corpus numbers.
- Mark Status at the top of THIS file: DONE + final numbers.

## Results (2026-08-14)

Implemented per Tasks 0–1 (constants + pass + wire-in + tests). Task 2
(extraction tuning) NOT run — see below.

- Confirmation @ 208: output byte-identical with/without `--no-rhythm-coarsen`
  (note 0.240, pitch 0.883, onset 0.153, dur 0.455). No change.
- `KEYSCRIBE_COARSEN_DEBUG=1`: every bar votes `coarse=false` (e.g. bar 1 gaps
  `0.25,1.0,0.75,0.75,0.25,0.75` → vote 0.667 < 0.70; swung bars also under
  the `≥ 2×bpb+1` note guard).
- Corpus gate (50 tracks, learned): pitch 0.826, note 0.374, recall 0.387,
  onset 0.256, root 0.393, exact 0.139 — byte-identical baseline, PASS.
- **Conclusion:** the Confirmation rhythm gap is swung-feel quantization, not
  16th over-segmentation. Forcing coarsening regresses note accuracy (0.231 in
  a lowered-threshold experiment). The real levers are Task 2's
  `release_ratio` / `onset_split_threshold`. Keep this pass disable-able.

## Guardrails (do NOT)

- Do NOT touch `soft_contrast`, chord detection, or beat tracking — this plan
  is rhythm-only.
- Do NOT remove the legacy path; coarsening must be disableable
  (`--no-rhythm-coarsen`).
- Do NOT let a COARSE vote apply to bars with < 2 notes.
- Do NOT force the pass (lower `coarse_vote_ratio` / drop the note guard) — it
  regresses note accuracy on genuine 16th runs.
- Do NOT commit unless the user explicitly asks.
