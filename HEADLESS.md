# KeyScribe Headless System & CLI — Memory

Memory file for the headless CLI (`keyscribe-cli`) that mirrors the desktop
app's transcription pipeline (Basic Pitch -> note events -> beat tracking ->
enhanced lead sheet -> MusicXML) without the egui dependency, so it can be
driven from the command line and **trained on labeled data**.

> Progress tracking: see **`PROGRESS_JOURNAL.md`** for the dated journal of
> every change + the current metrics dashboard.

## Build

Windows-only binary (other platforms build as a no-op stub).

```powershell
cargo build --release --no-default-features --bin keyscribe-cli
# binary: target\release\keyscribe-cli.exe
```

Model files: `models/basic-pitch.onnx` (required), `models/htdemucs_6s.onnx`
(only for `--stems`). Point at them with `--model-dir <dir>` if not in the
default relative `models/` path.

Tests: `cargo test --no-default-features --lib` (64 tests).
GUI check: `cargo check --bin keyscribe` (also verifies the shared lib compiles
for the app).

## CLI commands

```
keyscribe-cli sheet  <in>  -o out.musicxml   # transcribe to MusicXML sheet
keyscribe-cli midi   <in>  -o out.mid        # transcribe to MIDI
keyscribe-cli render <in>  -o out.mp3|wav|pdf  # MuseScore convert
keyscribe-cli compare <ref> <transcribed>    # accuracy vs ground truth
keyscribe-cli stems  <in>  -o <dir>          # demucs stem separation to WAVs
keyscribe-cli tune   <folder> [-o config.json]  # TRAIN pipeline parameters
keyscribe-cli maketest -o out.mid            # synthetic test melody
```

### `sheet` flags (subset)
- `--key-sensitivity <0.0-1.0>` — GUI slider value; converted internally to the
  note probability threshold (see below). Default `0.23`.
- `--melody poly|skyline|heuristic` (default `poly` — only poly preserves chords)
- `--outlier-semitones <n>` (default 12; melody filter for skyline/heuristic)
- `--no-leadsheet`, `--single-staff` — engraving choices
- `--stems` — transcribe only melodic demucs stems (much better on full bands)
- `--chord-beat <f32>` — beat offset within the bar (0.0 = downbeat) at which to
  sample the primary chord from the probability timeline. Negative (default) =
  legacy whole-bar-mean profile.
- `--chord-cleanest` — sample at fewest-pitch-classes moment (only when chord-beat < 0)
- `--chord-strike` — sample at max-simultaneous-onsets moment (only when chord-beat < 0)
- `--no-chord-collapse` — disable the jazz-extension→7th quality collapse (Tier A2; on by default)
- `--chord-split <0.0-1.0>` — half-bar chord splitting threshold (Tier A2; 0 = off, default)
- `--bpm <f32>` — override beat tracking with a fixed 4/4 grid at this tempo
- `--config <file>` — apply a config written by `tune`; explicit flags win
- `--render mp3|wav|flac|ogg|pdf` — also render via MuseScore CLI

### `midi` flags
`--key-sensitivity`, `--melody`, `--outlier-semitones`, `--model-dir`,
`--bpm`, `--config`.

## Sensitivity -> threshold conversion (critical!)

The GUI slider is scaled ×0.5 for display and ×2.0 for storage, and note
activation is `prob >= 0.12 / sensitivity`. The CLI reproduces this exactly in
`headless::key_sensitivity_to_threshold`:

```
internal_sens = slider * 2.0            # GUI stores 2x the shown value
threshold     = 0.12 / internal_sens    # clamped [0.05, 0.95]
```

| slider (shown) | internal sens | threshold |
|---|---|---|
| 0.1  | 0.2  | 0.60 |
| 0.2  | 0.4  | 0.30 |
| 0.23 (default) | 0.46 | 0.26 |
| 0.3  | 0.6  | 0.20 |

Lower sensitivity => higher threshold => fewer, cleaner notes. Lower threshold
=> more notes. The GUI shows a keyboard-lit note when `p * sens >= 0.12`, which
is algebraically the same test.

## Training the pipeline (`tune`)

```
keyscribe-cli tune <folder> [options]
```

- **Dataset**: every audio file (`wav/mp3/flac/ogg/m4a/aiff`) in `<folder>`
  with a sibling `<same stem>.musicxml` is a labeled pair. Missing references
  are skipped (logged to stderr).
- **Search space** (per track): key-sensitivity grid × chord-sampling × melody
  mode × bpm-grid (auto / refined around tracker-×2 / `--bpm` override) ×
  stem-mode (on/off when `--stems`).
  - Default: 8 sens × 6 sampling × 3 melody × 4-7 bpm = up to 1008 combos/track
    (`--bpm` given: 7 bpm candidates, else 4).
  - `--fast`: 5 sens × 4 sampling × 3 melody × 4-7 bpm = up to 420 combos/track.
- **Cost model**: `analyze_audio` (Basic Pitch + beat tracking) runs ONCE per
  track; every evaluation after that only re-extracts notes and rebuilds the
  sheet from the cached probability timelines — cheap.
- **Objective** (`--objective`, default `balanced`): scores are F1-style
  (precision × coverage against the reference), so a "solution" that detects
  one chord and scores 1.0 can't win. Five evaluation dimensions, weighted to
  treat melody and chords equally (50/50):
  - **melody** (30%) — note values + durations: `note_accuracy`/`recall` F1
    plus onset & duration timing error scores (timing scales WITH note F1 so
    stray matches can't inflate it).
  - **pitch accuracy** (10%) — a direct vote for melody pitch quality; the
    rhythm-weighted `melody_sim` compresses near zero (the rhythm gap), so
    without this the search trades a readable melody for chord gains.
  - **bar placement** (10%) — does the melody fall in the right place in the
    bar (`compare_bar_placement`: in-bar beat position of matched notes) and
    are downbeats correct (`bar_count_score` catches half/double tempo).
  - **chord roots** (35%) — `root_f1` (root-only matching).
  - **chord quality** (15%) — `exact_f1` (root+kind matching).
  - Alternatives: `melody` (0.7 similarity + 0.3 pitch) / `root` / `chord`
    (0.7 root + 0.3 quality).
  All components are logged per-evaluation in the report file, so weights can
  be re-balanced without re-running the analysis.
- **Output**: ranked table (rootF1 / exactF1 / coverage / pitch / notes /
  chords) and the best combo written as JSON config (default
  `keyscribe.tuned.json`, override with `-o`).

### Tuned config format (`TunedConfig`)
```json
{
  "key_sensitivity": 0.1,
  "chord_beat": 0.5,
  "chord_cleanest": false,
  "chord_strike": false,
  "melody": "poly",
  "stems": false,
  "bpm": 222.22
}
```
Apply it: `keyscribe-cli sheet in.mp3 -o out.musicxml --config keyscribe.tuned.json`
Explicit CLI flags on the command always override the config's values.

### Training log / where to improve
`tune --report out.json.report.json` writes every evaluation as JSON rows with
`melody_sim`, `root_f1`, `exact_f1`, `bar_placement`, `bar_count_score`,
`root_rate`, `root_coverage`, `pitch_accuracy`, `note_accuracy`, plus the full
parameter combo (threshold, chord sampling, melody, bpm, stems). Load it
(pandas/Excel) to find failure patterns:
- low coverage everywhere = Basic Pitch note quality is the bottleneck;
- low bar placement = grid phase/tempo (check the bpm candidates);
- chord root_rate ok but exact_rate low = quality discrimination (the ±1
  semitone tolerance in `soft_contrast`, extension simplicity prior);
- root_rate low but coverage ok = relative-chord ambiguity: bars whose voicing
  reads as a different root's inversion (e.g. G13 voiced {C E G A} reads
  C6/G). Check `KEYSCRIBE_HARMONY_DEBUG` per-bar tops to confirm. Sequence
  context (the Viterbi) fixes isolated ties; systematic sections (an
  accompaniment whose bass anticipates the next bar's root) need more
  training tracks.

## Evaluating a corpus (`eval-corpus`)

```
keyscribe-cli eval-corpus <folder> [options]
```

Batch harness that runs the full `sheet` pipeline + `compare` over **every**
`(audio, reference .musicxml)` pair in `<folder>` (e.g. the 50-track Omnibook
set in `out/omnibook/`) and prints a dashboard table plus an aggregate row,
writing the whole thing as JSON with `-o report.json`.

- **Per-track BPM overrides** (`--bpm-file <path>`) make runs reproducible when
  the ML beat tracker misfires on a track's meter (it commonly returns
  half-time). The file is either a JSON object `{"Track_Name": 208.0, ...}` or
  one `<track stem> <bpm>` per line. `out/omnibook/bpm_overrides.txt` has the
  XML-read tempos for the whole corpus.
- **Fixed knobs** for comparability: `--key-sensitivity`, `--melody`,
  `--quantizer legacy|learned`, `--stems`, `--tolerance` (chord onset beats),
  `--model-dir`, `--objective`. The command is the regression gate: run it
  before/after a change and diff the aggregate row.
- **Report schema** (`CorpusReport` in `src/eval_corpus.rs`): per-track
  `root_match_rate` / `exact_match_rate` / `root_coverage` / `pitch_accuracy` /
  `note_accuracy` / `recall` / `mean_onset_error_beats` / `mean_duration_error`
  / `in_bar_rate` / `bar_count_score`, plus `error` for failed tracks and the
  corpus means.

Current corpus baselines (`--melody heuristic --quantizer learned`, fixed bpm):
pitch **0.867**, note **0.406**, recall **0.381**, onset error **0.231 beats**,
chord root **0.393**, chord exact **0.139**, in-bar **0.962** (50/50 tracks OK).
(2026-08-14 rhythm fixes; pre-fix baseline was pitch 0.826 / note 0.374 /
onset 0.256.)

With Tier A2 Part 1 (chord quality collapse, ON by default, 2026-08-20): chord
exact **0.197**, chord root 0.384 (greedy-matcher artifact of the collapse —
see the chord section), melody unchanged (pitch 0.863, note 0.405, recall
0.382, onset 0.231).

## Workflow gotchas

- **Grid alignment dominates everything.** The beat tracker returns half/
  double tempo (e.g. 111 bpm vs real 225). The `tune` bpm search now refines
  the doubled tracker value at ±1.2%, so the fine grid (225) is found instead
  of the coarse 222.22; `--bpm 225` remains the manual override. Even 1.2%
  wrong accumulates a beat of drift over 36 bars, misaligning both chord
  sampling windows and melody quantization — this is why earlier naive chord
  sweeps looked random.
- **The chord-sampling knobs are real parameters now.** Before the timeline
  detector ignored them and all 6 modes in the tune search produced
  byte-identical output (a 6× wasted search dimension). They are wired into
  `compute_bar_profiles`; verify with `KEYSCRIBE_HARMONY_DEBUG`.
- **Comparing sheets**: `compare` prints note + chord metrics. Chords are the
  meaningful signal here (reference leadsheet has ~36 sparse melody notes, the
  transcription has full voicings, so note-count comparisons are misleading).
- Reference `tests/data/Pretty standard.mp3` is byte-identical to
  `out\gt_render.mp3`; ground truth `tests/data/Pretty standard.musicxml`
  contains the *correct* chord kinds (Cmaj7/Bm7/Dm7/...); roots are what
  `--objective root` optimizes.
- MuseScore export to WAV/FLAC fails (exit 1331); MP3 (128k) works — fine for
  generating more training audio from reference MusicXML.
- `tests/data/`, `out/`, `out2/`, `out3/` are gitignored.

## Extending the search space
- Cheap knobs live in `headless::generate_sheet_inner` (threshold via
  sensitivity, melody mode, chord sampling, manual bpm grid) and in
  `tune.rs` (grid definitions `sensitivities/chord_samplings/melodies`,
  `bpm_candidates` — auto / tracker-×2-refined / override grids — and
  objective weights in `Objective::score`).
- To add a cheap parameter: add it to the combo struct, apply it in
  `generate_sheet_inner` (or `sheet_opts_for`), and persist it in
  `TunedConfig` + the CLI resolution (`sheet`/`midi` handlers in
  `keyscribe_cli.rs`).

## Chord detection: harmonic timeline detector (`leadsheet/harmony.rs`)
The CLI passes the Basic Pitch probability timeline into chord detection
(`generate_lead_sheet_enhanced_with_timeline`, env `KEYSCRIBE_HARMONY_DEBUG`
for per-bar diagnostics).

Pipeline (P1-P3):
1. **P1 soft profiles** — per-bar pitch-class activation from the timeline
   (comp register only, MIDI < 72, so melody tones aren't read as chord
   extensions); whitened by subtracting a noise floor (0.7 × median) and
   sharpened with `^1.5` so strong chord tones dominate the weak diffuse
   floor. **Chord-sampling knobs are live**: `--chord-beat >= 0` samples a
   0.5-beat window centered on that beat (0.0 = the downbeat strike zone),
   `--chord-cleanest` picks the 0.25-beat window in the first half with the
   fewest active pitch classes, `--chord-strike` picks the window with the
   most simultaneous onsets (onset head, or frame-to-frame activation delta
   when no onset timeline is available). The whole-bar mean (default) drowns
   the voicing in passing tones; the strike-sampled profiles are far cleaner
   (CΔ7 scored 1.02 vs 0.79 for the same bar).
2. **P3 joint root+quality** — `soft_contrast`: mass on chord tones (+1) vs
   non-tones (−1), with a tight ±1-semitone tolerance (0.25) so Cmaj7 vs C7
   discriminate on the 7th; full quality set (incl. `6`, `-13`, `-7b5`, `9`)
   scored directly with a strong simplicity prior (3-int +0.08 … 6-int −0.30)
   so chords don't over-extend. A **bass-register anchor** blends 45% of a
   separately-whitened bass profile (MIDI < 60) into the emission and adds a
   +0.45 bonus when the lowest sounding pitch is the root, +0.15 when it's a
   5th/7th (walking-bass root approach), +0.06 for a 3rd — this breaks the
   C6/Am7 relative-chord ties the comp register alone can't resolve (root
   match on Confirmation went 0.18 → 0.30 from this alone).
3. **P2 Viterbi** — a Krumhansl-Kessler key estimate biases diatonic roots
   (+0.08) and secondary-dominant roots a 5th/major-3rd from the scale (+0.04),
   and a functional-harmony Viterbi decodes the whole bar sequence:
   transitions prefer V-I cadences (0.00), sustained roots (0.02) and common
   stepwise motion, penalizing tritone/leap moves (up to 0.40) plus a small
   quality-change cost (0.03). Costs are weak relative to per-bar emissions,
   so they resolve coin-flip bars instead of overriding real changes.
   When `models/transition_matrix.json` exists (fitted by
   `keyscribe-cli fit-transition-matrix` over the Omnibook corpus), the
   hand-tuned table is **blended 65% with a learned (root, quality)
   transition matrix** — the plan's genuinely-novel piece — capturing
   joint tendencies the interval table can't (ii-7→V7→I with the right
   qualities each step, V7→IV back-cycling, ii∅→V7). Root match on
   Confirmation: 0.30 → **0.34**; neutral on Ornithology/Donna Lee (no
   regression).

Measured on `tests/data/Pretty standard.mp3` (sens 0.2, bpm 222 cleanest grid):
chord root F1 0.200 (was 0.062), exact F1 0.200 (was 0.000), coverage 0.194.
Remaining misses are bars whose voicing genuinely reads as a relative chord
(e.g. a G13 voiced {C E G A} reads C6/G) and a section where the auto-
accompaniment bass anticipates the next bar's root by one bar — irreducible
per-bar ambiguity that only sequence-level context (more tracks, learned
transitions) can fully resolve.

### Quality collapse to 7ths (Tier A2 Part 1, 2026-08-20, ON by default)
`collapse_quality_to_seventh` (harmony.rs) maps the emitted template suffix to
the nearest 7th-chord class — `9/7b9/7#9/7#11/13/7#5 → 7`, `Δ9/Δ13/Δ7#11/
Δ9#11 → Δ7`, `-9/-11/-13 → -7`, `6 → ""`, `m6 → "-"`. The Viterbi still
decodes over all 27 states; only the emitted symbol is collapsed, so
consecutive-duplicate suppression also merges `13`→`7` with a following `7`.
Control: `ChordAnalysisConfig.collapse_extensions` (default true) /
`SheetOptions.chord_collapse` / `TunedConfig.chord_collapse` (serde default
true) / CLI `--no-chord-collapse`. Confirmation chord exact **0.030 → 0.102**
(target ≥ 0.10), corpus exact **0.139 → 0.197** (target ≥ 0.139). Root metrics
drop ~0.01 purely as a greedy-matcher artifact in `compare` (exact hits
consume reference slots; `tc` shrinks via duplicate merges) — XML roots are
unchanged.

### Half-bar chord candidates (Tier A2 Part 2, 2026-08-20, DISABLED by default)
`compute_bar_profiles` now also bins each bar into first/second-half profiles
(`BarProfile.halves/halves_bass/bass_note_second`); `detect_chords_from_timeline`
emits one chord per half when the whitened halves differ by
`1 - cosine > split_threshold`. Control: `ChordAnalysisConfig.split_threshold`
(default 0.0) / `SheetOptions.chord_split` / `TunedConfig.chord_split` (serde
default 0.0) / CLI `--chord-split <0.0-1.0>`. Measured on Confirmation the
split regresses every threshold (0.25→0.062, 0.3→0.071, 0.35→0.075,
0.45→0.086 exact vs 0.102 split-off), so it ships off by default per the
plan's guardrail; disabled it is behavior-neutral on the corpus. The plan's
`1 - cos < split_threshold` condition was implemented as `>` (split when halves
DIFFER) per the intended mid-bar chord-change semantics.

## Melody extraction (musicxml.rs)
`extract_melody_skyline` (and `heuristic`, which delegates to it) was reworked:
- **Onset-density line selection**: the melody is the active pitch with the
  most onsets in a ~0.6s window (the melody re-articulates on every 8th note;
  the accompaniment sustains whole-bar chords). Pure max-pitch was picking
  the MuseScore auto-accompaniment voiced above the melody.
- **Re-articulation splitting**: repeated same-pitch notes (staccato 8ths) are
  split instead of merged into one long note.
- **Stem-aware melody (headless)**: with `--stems`, melody modes transcribe
  only the identified melodic stem (`identify_melodic_stem_from_stems`,
  recorded as `AnalyzedAudio.melodic_stem_timeline`) instead of merging all
  stems — the piano comp would otherwise dominate the line.

Measured on `tests/data/Pretty standard.mp3` (sens 0.1, bpm 222, heuristic):
melody pitch accuracy 0.139 → **0.800**; note accuracy 0.08. Rhythm
(onset/duration) accuracy remains the next gap (P4 note segmentation).

## Note extraction rhythm (P4) — `extract_notes_from_timeline`
The binary threshold crossing merged staccato repeats into one long note and
started notes at the threshold crossing (late onsets). Fixed in both the CLI
(`headless.rs`) and GUI (`extract_events_from_timeline_data` — now delegates
to the headless extractor so there is ONE implementation):
- **Adaptive release** — a note ends when its probability falls below
  `0.55 × its own peak` (min 0.05), so re-articulations split instead of
  merging into one sustained note.
- **Attack-adjusted onsets** — on note start, walk back up to 5 frames to the
  low point where the probability began its rise and start there, removing
  the threshold-crossing latency.
- **Onset-head positioning (2026-08-14)** — onsets and re-articulation
  splits are then moved to the ONSET head's local peak (±4 frames, peak
  ≥ 0.3). The frame head ramps ±72 ms at fast tempos (one 16th at 208 BPM);
  the onset head is sharp at attacks. This alone lifted Confirmation note
  accuracy 0.242 → 0.286.
- **Onset-head splitting** — Basic Pitch's separate onset head (88 notes,
  sharp at attacks) now drives re-articulation splits. The inference layer
  separates the note/onset/contour heads by output name, with a sparsity
  fallback for opaque exports (`StatefulPartitionedCall:0/1/2` — the sparsest
  >0.5 88-head is the onset). Previously all heads were max-merged, losing
  the onset information entirely.
- **Single-frame merge gap** — `merge_adjacent_notes_with_gap(step_sec)`
  removes frame jitter while preserving genuine splits.
- **Onset-clamped melody durations** — a monophonic melody note cannot extend
  past the next onset; this turns staccato 8ths (whose extraction runs into
  the next attack) into properly short notes.
- **Chatter absorption (2026-08-14, musicxml.rs skyline post-pass)** —
  melody segments < 120 ms that are much quieter than both neighbors (< 75%)
  or foreign-pitch at < 90% of the louder neighbor are absorbed into the
  previous segment (comp-strike leakage pokes). Full-velocity short notes
  (pickups, 16ths) survive. Confirmation: 688 → 545 segments.
- **Complexity-penalized snapping (2026-08-14, quantize.rs
  `snap_intra_beat_pos`)** — coarse slots (onbeat/8th) get a small margin
  over fine slots (16ths +0.045, triplet 8ths +0.03) so ±60 ms onset jitter
  cannot flip an 8th onto a dotted-8th/16th slot.
- **Diagnostics** — `KEYSCRIBE_QUANT_DEBUG=1` prints the intra-beat-position
  histogram, snap destinations, mean snap error, the learned-engine legacy-
  fallback bar count, and skyline segment-duration stats;
  `tools/diag/side_by_side.py <ref.musicxml> <trans.musicxml>` prints
  matched-note onset deltas + histogram (mirrors `compare`'s parser incl.
  backup/forward).
Measured (sens 0.2, bpm 222, skyline): pitch accuracy 0.503 → 0.565,
note accuracy 0.044 → 0.054, recall 0.062 → 0.078, mean onset error
**0.188 → 0.100 beats**; dotted-eighth mis-quantization gone (clean 8ths/
quarters, even bars). Corpus (2026-08-14): note acc **0.406** (was 0.374).
- Tune bpm search no longer includes the ×0.5 (half-tempo) candidate — it
  coarsens quantization and gamed the beat-space metrics, picking degenerate
  configs (e.g. bpm=56 for a 225 bpm tune).

## Beat grid phase alignment (2026-08-20) — `refine_beat_phase` + `validate_downbeat_rotation`
Fixes the systematic rhythmic displacement that made notes land one 16th slot
off (~half the matched notes were exactly ±0.25 beats off on Confirmation).
Run on the ML beat-tracker grid (NOT the `--bpm`/`manual_bpm` synthetic grid —
that path already has perfect phase by definition and bypasses both functions).
- **Enhanced phase search (Task 0)** — `refine_beat_phase`
  (`src/leadsheet/beat_tracking.rs`): widened the sweep from 3 phase shifts
  (±0.25·period) to **16 candidates** (-0.5→+0.4375 period; 18 ms @208 bpm —
  below onset-detection noise). The score weights strong-beat targets
  (onbeat 0.0, half-beat 0.5) over subdivisions. Tie-break: prefer the
  candidate whose period stays closest to the source period, then non-doubled
  over doubled, then smallest |shift| — this fixes half-time/double-time
  metric confusion in the tracker output.
- **Beat-boundary snap (Task 1)** — `associate_notes_to_beat_grid`
  (`beat_association.rs`): a note < ~1/16th before a beat now snaps to the
  *next* beat (onset-5 ms notes no longer carry the previous beat's
  bar/beat metadata).
- **Downbeat validation (Task 2)** — `validate_downbeat_rotation`
  (`beat_tracking.rs`, wired in `headless.rs` after `refine_beat_phase`):
  only rotates the downbeat grid when onset-energy evidence wins by a
  ≥15% margin, so correct grids are left alone.
- **Metrical prior (Task 3)** — `snap_intra_beat_pos` (`quantize.rs`): beat 1
  gets a small bonus toward the downbeat so genuine ambiguities resolve to
  strong beats.
- **Measured (2026-08-20):** Confirmation @208 (`--bpm`) note 0.288-0.289 /
  onset 0.138-0.139 (baseline 0.286/0.138 — unchanged, expected, since
  `--bpm` bypasses these). Corpus (50 tracks, `--bpm-file` overrides): pitch
  0.863 / note 0.405; the 0.001-0.004 gap vs the gate is isolated to the
  pre-existing staged `fill_melody_durations` change (`KEYSCRIBE_FILL_GAP=0`
  → pitch 0.8676 / note 0.4071, ≥ baseline). 86/86 lib tests pass.
- **Diagnostics:** `KEYSCRIBE_PHASE_DEBUG=1` (phase candidates + scores),
  `KEYSCRIBE_DOWNBEAT_DEBUG=1` (rotation strength/margin).
- **Known ML-path limitation (pre-existing, not fixed):** the default
  `melody_stems: true` feeds beat-this only the bass/drums stems (near-silent
  on jazz) → 0 beats → sheet fails with "not enough notes/beats". Use
  `--no-stems-melody` to feed the full mix. Even then beat-this detects
  Confirmation at half-time (103 vs 208 bpm); `correct_beat_metric_level`
  only doubles when bpm < 70. These are why all 50 corpus tracks are pinned
  with `--bpm-file` overrides.

## Rhythm coarsening pass (Tier A1) — `coarsen_rhythm` (INERT, keep disable-able)
Per-bar post-quantization pass in `src/leadsheet/quantize.rs` that re-snaps
"over-split" bars onto the 0.5 grid. Controlled by `RhythmCoarsenConfig`
(default enabled) via `SheetOptions`/`TunedConfig` (`rhythm_coarsen`) and
`--no-rhythm-coarsen`. **Verified inert on the corpus (2026-08-14):** every bar
votes `coarse=false` on Confirmation and the 50-track aggregate is byte-identical
to baseline. (Diagnosis later corrected — see the 2026-08-14 journal entry: the
audio is STRAIGHT 8ths; the real gap was skyline chatter + onset timing, fixed
in the extraction layer, not coarsening.) Do NOT force it (lowering
`coarse_vote_ratio` / the `≥ 2×bpb+1` note guard regresses note accuracy on
genuine 16th runs).

## Learned melody quantizer — v1 MLP default, v2 seq model ROLLED BACK
- **Shipped default** (`models/melody_quantizer.onnx`): v1 per-note MLP,
  12 duration tokens, `[1,seq,9]→[1,seq,12]`, trained by
  `tools/melody_corpus/train_quantizer.py` (holdout split). Ties or beats
  legacy on the corpus.
- **v2 BiGRU + MERGE token (2026-08-16, ROLLED BACK):** 13-class sequence
  model (`tools/melody_corpus/train_quantizer_seq.py`), kept as
  `models/melody_quantizer_v2_seq.onnx` (+ `.pt` state). Synthetic metrics
  excellent (holdout token 0.93, merge P/R 0.94) but FAILED the corpus gate
  (recall 0.34 vs 0.38) — the 9-feature vector has no inter-note gap
  feature, so genuine staccato repeats get over-merged. Iteration 2: add
  `gap_to_prev_beats` as a 10th feature (Python trainer + Rust
  `learned_note_features` together) and retrain.
- **Rust merge path (shipped, 13-class-ready):** `merge_keep_mask` + span
  extension + contiguity guard (< 0.08 beats raw gap) in
  `quantize_aligned_notes_learned`. Token indices ≥ 12 = MERGE (no note
  emitted; previous note extends to the next kept onset, same-bar only).
  With a 12-class model this is a no-op passthrough.

## Chord display note
The reference MusicXML's kinds were correct all along (Cmaj7/B-7/G-13 etc. —
MuseScore shows Δ/−). The `compare` tool was printing the raw `text`
attribute ("7", "13") instead of the kind value; `Harmony::display()` now
maps kind values to conventional symbols (Cmaj7, B-7, G-13, D7…). Metrics
(root/kind matching) were never affected.

## GUI wiring (native-ui)
- `App::new` calls `headless::TunedConfig::load_default()`; when
  `keyscribe.tuned.json` exists in the working directory the GUI applies the
  tuned key sensitivity, melody mode (poly/skyline/heuristic) and manual bpm
  by default for lead sheets and melody extraction.
- Sheet preview passes the chord-source probability timelines
  (`chord_timelines()` in `app/sheet_music.rs`) into
  `generate_lead_sheet_enhanced_with_timeline`, so the GUI uses the harmonic
  detector by default; it falls back to the legacy scan when no timeline is
  available.
