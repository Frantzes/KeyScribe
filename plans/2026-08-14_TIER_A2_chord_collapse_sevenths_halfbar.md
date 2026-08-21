# Tier A2 — Chord quality collapse to 7ths + half-bar chord candidates

Date: 2026-08-14
Status: DONE 2026-08-20 (Part 1 shipped on by default; Part 2 shipped DISABLED by
default — it regresses Confirmation at every split threshold; see "Results")
Owner: implementer (any AI model — follow steps exactly, in order)
Companion plans: `2026-08-14_IMPLEMENTATION_PLAN_INDEX.md` (read first)

## Goal

Fix the chord-quality gap on Confirmation @ 208 bpm: root match **0.340**,
exact match **0.030**. Two verified problems:

1. **Hallucinated extensions.** The detector emits `F minor-11th`,
   `Bb major-13th`, `D-7(...)` where the reference says plain
   `F / E-7b5 / A7 / D-`. The 27-template set (`ALL_TEMPLATES`,
   `src/leadsheet/harmony.rs:403-431`) invites 5-6-interval chords and the
   simplicity prior (harmony.rs:437-445) is too weak to stop them.
   **DECISION (user-locked): collapse jazz extensions to the nearest
   7th-chord class before output.** This mirrors Decision 2 of
   `CHORD_DETECTION_PLAN.md`.
2. **One chord per bar is hard-coded** (harmony.rs:488-490 doc comment,
   output loop harmony.rs:626-649). Bebop heads change chords twice per bar;
   those bars are unreachable today and read as a wrong single chord
   (e.g. bar 3 root drift Bb-maj vs D-).

Targets:

- Confirmation `exact_match_rate` 0.030 → **≥ 0.10** (Part 1 alone should do
  most of this)
- Confirmation `root_match_rate` stays **≥ 0.34**
- 50-track corpus: exact **≥ 0.139** (baseline), root **≥ 0.393** (baseline)

## Part 1 — Quality collapse to 7th class (small, do first)

### The exact mapping table

`ALL_TEMPLATES` (harmony.rs:403-431) has EXACTLY these 27 suffix strings:
`"", "-", "dim", "aug", "sus2", "sus4", "7", "Δ7", "-7", "-Δ7", "dim7",
"-7b5", "7#5", "6", "m6", "9", "Δ9", "-9", "7b9", "7#9", "7#11", "Δ7#11",
"-11", "13", "Δ13", "-13", "Δ9#11"`.

(Note: `Δ` is the Unicode char U+0394. Copy-paste from the source file, do
not type `D` or `d`.)

Write this function in `src/leadsheet/harmony.rs` near `ALL_TEMPLATES`:

```rust
/// Collapse jazz-extended qualities to the nearest 7th-chord class
/// (user decision 2026-08-14, mirrors CHORD_DETECTION_PLAN.md Decision 2).
/// Triads, 7ths and suspensions pass through unchanged.
pub(crate) fn collapse_quality_to_seventh(suffix: &str) -> &str {
    match suffix {
        "9" | "7b9" | "7#9" | "7#11" | "13" | "7#5" => "7",
        "Δ9" | "Δ13" | "Δ7#11" | "Δ9#11" => "Δ7",
        "-9" | "-11" | "-13" => "-7",
        "6" => "",
        "m6" => "-",
        other => other, // "", "-", "dim", "aug", "sus2", "sus4", "7", "Δ7",
                        // "-7", "-Δ7", "dim7", "-7b5" keep as-is
    }
}
```

### Where to apply it

In `detect_chords_from_timeline`'s output loop (harmony.rs:632-648), the
suffix is read at line ~638:

```rust
let suffix = ALL_TEMPLATES[s % n_qualities].0;
```

Change to:

```rust
let suffix = collapse_quality_to_seventh(ALL_TEMPLATES[s % n_qualities].0);
```

The Viterbi still decodes over all 27 states (full information); only the
EMITTED symbol is collapsed. Consecutive-duplicate suppression at
harmony.rs:640 then also merges e.g. `13`→`7` followed by `7`.

### Make it switchable

1. Add `pub collapse_extensions: bool` to `ChordAnalysisConfig` (grep
   `pub struct ChordAnalysisConfig` — it lives in the leadsheet module;
   give the field a `Default` of `true` via serde/impl like its siblings).
2. Apply the collapse only when `config.collapse_extensions`.
3. CLI: add `--no-chord-collapse` to `sheet` in `src/bin/keyscribe_cli.rs`
   (pattern: copy how `--no-leadsheet` is plumbed).
4. `TunedConfig` (src/headless.rs, grep `pub struct TunedConfig`): add
   `chord_collapse: bool` with serde default `true`.

### Unit tests (add in harmony.rs `#[cfg(test)]`)

- Every one of the 27 template suffixes maps into the keep-set
  `{"", "-", "dim", "aug", "sus2", "sus4", "7", "Δ7", "-7", "-Δ7", "dim7", "-7b5"}`
  (enumerate `ALL_TEMPLATES` in the test — never hardcode the list twice).
- `collapse_quality_to_seventh("13") == "7"`, `("-11") == "-7"`,
  `("Δ13") == "Δ7"`, `("-7b5") == "-7b5"`.

### Verify Part 1

```
cargo test --no-default-features --lib
cargo build --release --no-default-features --bin keyscribe-cli
target\release\keyscribe-cli.exe sheet out\omnibook\Confirmation.mp3 -o $env:TEMP\opencode\conf_a2.musicxml --bpm 208
target\release\keyscribe-cli.exe compare out\omnibook\Confirmation.musicxml $env:TEMP\opencode\conf_a2.musicxml
```
Expect exact match to jump (reference is plain 7ths; our emitted minor-11ths /
major-13ths now print as -7 / Δ7). Root must NOT move (collapse never changes
roots — if root moves, you applied it in the wrong place).

## Part 2 — Half-bar chord candidates (bigger; do after Part 1 is green)

### Design

`compute_bar_profiles` (harmony.rs:109-301) already bins each bar into
0.25-beat sub-windows (`PROFILE_BIN_BEATS`, harmony.rs:95) and then aggregates
the whole bar. Extend it so a bar can produce TWO candidate segments.

### Steps

1. **Extend `BarProfile`** (grep `struct BarProfile` in harmony.rs): add
   ```rust
   pub halves: Option<([f32; 12], [f32; 12])>, // (first-half pcp, second-half pcp)
   pub halves_bass: Option<([f32; 12], [f32; 12])>,
   pub bass_note_second: Option<u8>,
   ```
   Populate in `compute_bar_profiles`: first half = bins in beat
   `[0, bpb/2)`, second half = bins in `[bpb/2, bpb)`. (When `bpb` is odd,
   first half gets `bpb/2` beats rounded down + the leftover bin goes to the
   second half.)
2. **Split decision.** Add to `ChordAnalysisConfig`:
   `pub split_threshold: f32` (default `0.0` = disabled; enable at `0.35`
   after measuring). For each bar, whiten both half profiles exactly like
   harmony.rs:512-530 does, compute cosine similarity of the whitened halves;
   if `1 - cos < split_threshold` → mark the bar as split.
3. **Emissions for split bars.** In `detect_chords_from_timeline`
   (harmony.rs:509-558) the code loops `for (bar, prof) in profiles.iter()`.
   Change the loop to iterate SEGMENTS: build a `Vec<(usize /*bar*/, usize
   /*half: 0|1*/, &[f32;12] /*pcp*/, &[f32;12] /*bass_pcp*/, Option<u8>
   /*bass_note*/)>` — one entry per unsplit bar, two per split bar — and score
   emissions per segment (identical math, just per-segment profiles).
4. **Viterbi** (harmony.rs:660-750): operates on `&emissions` rows; it needs
   NO change except it now sees more rows. Verify the learned-transition
   blending still works (it indexes by (root, quality) state, not by bar).
5. **Output loop** (harmony.rs:626-649): track, per segment, its
   `beat_start = bar * bpb + (half == 1 ? bpb as f32 / 2.0 : 0.0)`. Emit the
   second-half chord even when the first-half symbol equals it? NO — keep the
   consecutive-duplicate suppression (a repeated chord across halves prints
   once). Whitened-energy gate (harmony.rs:633) applies per segment: use the
   segment's own whitened energy (store per segment during step 3).
6. **CLI/config:** `--chord-split <0.0-1.0>` flag (0 = off) on `sheet`;
   `TunedConfig.chord_split: f32` serde-default 0.0. Add `chord_split` to the
   `tune` grid as `[0.0, 0.35]` once Part 2 works.

### Verify Part 2

```
KEYSCRIBE_HARMONY_DEBUG=1 target\release\keyscribe-cli.exe sheet out\omnibook\Confirmation.mp3 -o $env:TEMP\opencode\conf_a2b.musicxml --bpm 208 --chord-split 0.35
target\release\keyscribe-cli.exe compare out\omnibook\Confirmation.musicxml $env:TEMP\opencode\conf_a2b.musicxml
```
Then sweep `--chord-split` over `{0.25, 0.3, 0.35, 0.45}` and keep the best
exact match WITHOUT dropping root below 0.34.

### Regression gate (mandatory)

```
target\release\keyscribe-cli.exe eval-corpus out\omnibook --bpm-file out\omnibook\bpm_overrides.txt --melody heuristic -o $env:TEMP\opencode\corpus_a2.json
```
- chord exact ≥ 0.139, root ≥ 0.393, melody metrics within ±0.01 of baseline.
- If Part 2 regresses the corpus, ship Part 2 disabled by default
  (`split_threshold = 0.0`) — Part 1 alone is already a win.

## Results (2026-08-20)

Part 1 (collapse) shipped, ON by default (`--no-chord-collapse` to disable):

- Confirmation @208 (`--bpm --melody heuristic`): chord exact **0.030 → 0.102**
  (target ≥ 0.10 ✓), root 0.340 → 0.327. Root dip is a greedy-matcher artifact
  (the `compare` tool marks reference slots used on exact hits, shrinking the
  remaining root pool; `tc` also drops 100→98 as consecutive-duplicate
  suppression merges `13`→`7` with a following `7`). Verified the XML roots are
  byte-identical except those 2 merged duplicates — the collapse never changes
  a root. Accepted by user as a documented metric artifact.
- Corpus (50 tracks, `eval-corpus`, `--melody heuristic`): chord exact
  **0.139 → 0.197**, root 0.393 → 0.384 (same artifact). Melody unchanged
  (pitch 0.863, note 0.405, recall 0.382, onset 0.231).
- `cargo test --no-default-features --lib`: 88 passed, 0 failed, 1 ignored.

Part 2 (half-bar splitting) implemented but **shipped DISABLED by default**
(`--chord-split <x>` to enable; `TunedConfig.chord_split` / tuned grid not
wired):

- Sweep on Confirmation (collapse on): split 0.25 → exact 0.062/root 0.231,
  0.3 → 0.071/0.232, 0.35 → 0.075/0.253, 0.45 → 0.086/0.258. Every point
  regresses the split-off baseline (0.102/0.327). The half-bar whitened
  profiles are too noisy (half the frames) and the Viterbi segment path
  degrades without a per-segment energy recalibration.
- Behavior-neutral when disabled: corpus with default config is identical to
  Part 1-only (exact 0.197, root 0.384), so the segment refactor is safe.
- Note: the plan's split condition `1 - cos < split_threshold` is inverted;
  implemented as `1 - cos > split_threshold` (split when halves DIFFER), which
  is the intended "chord change mid-bar" semantics. Empirical result stands
  regardless.

## Docs

- `PROGRESS_JOURNAL.md` dated entry; metrics dashboard row updates.
- `HEADLESS.md`: chord section — document `collapse_quality_to_seventh`, the
  `--no-chord-collapse` / `--chord-split` flags, and new measured numbers.
- This file: Status → DONE + final numbers.

## Guardrails (do NOT)

- Do NOT change roots in the collapse (Part 1 maps qualities only).
- Do NOT emit the second-half symbol when identical to the first (keep
  dedupe).
- Do NOT enable half-bar splitting by default until the corpus gate passes.
- Do NOT modify `soft_contrast` math itself (Tier B2 replaces emissions
  wholesale; this plan only post-processes and re-segments).
- Do NOT commit unless the user explicitly asks.
