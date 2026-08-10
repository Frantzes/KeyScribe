//! Harmonic chord detection from the Basic Pitch probability timeline.
//!
//! Three improvements over the legacy binary-note scan:
//!   P1: chords are derived from soft pitch-class profiles (mean activation
//!       per pitch class per bar) computed directly from the probability
//!       timeline, so a note that sits just above/below the extraction
//!       threshold no longer makes or breaks a chord.
//!   P2: a key estimate (Krumhansl-Kessler profiles) plus a Viterbi pass over
//!       the bar sequence with functional-harmony transitions (circle-of-
//!       fifths, cadences, diatonic preference) smooths single-bar errors.
//!   P3: (root, quality) templates are scored jointly against the soft profile
//!       with a bass-register anchor, so the root follows the actual bass
//!       instead of a coin flip among equally-scored templates.

use crate::leadsheet::chord::{chord_symbol_from_root, ChordAnalysisConfig};
use crate::leadsheet::transition::learned_transitions;
use crate::leadsheet::types::ChordSymbolChange;

/// Basic Pitch probability timelines plus per-frame step sizes, passed in so
/// chord detection can use soft evidence instead of binary note events.
#[derive(Debug, Clone)]
pub struct TimelineChordInput {
    /// (frame probabilities for MIDI pitches 21..108, seconds per frame) per
    /// source (full mix, or one entry per melodic stem).
    pub timelines: Vec<(Vec<Vec<f32>>, f32)>,
    /// Parallel to `timelines`: (onset-head probabilities, step_sec) per
    /// source, when the model exposes an onset head. Used to find the chord
    /// "strike" moment; may be empty (then the frame-to-frame activation
    /// delta is used instead).
    pub onset_timelines: Vec<(Vec<Vec<f32>>, f32)>,
}

fn pitch_class_char(pc: u8) -> &'static str {
    ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"][(pc % 12) as usize]
}

/// Soft per-bar pitch-class profile derived from the probability timeline.
#[derive(Debug, Clone)]
pub struct BarProfile {
    pub bar_index: usize,
    /// Mean frame probability per pitch class (0..1).
    pub pcp: [f32; 12],
    /// Mean probability per pitch class from pitches below middle C only
    /// (bass-register evidence).
    pub bass_pcp: [f32; 12],
    /// Lowest pitch (MIDI) with strong activation in the bar, if any.
    pub bass_note: Option<u8>,
    /// Total activation energy (sum of `pcp`) — low-energy bars are skipped.
    pub energy: f32,
}

/// Estimated key: tonic pitch class + major/minor + diatonic scale mask.
#[derive(Debug, Clone)]
pub struct KeyInfo {
    pub tonic: u8,
    pub minor: bool,
    pub scale: [bool; 12],
}

fn scale_pcs(tonic: u8, minor: bool) -> [bool; 12] {    let intervals: &[u8] = if minor {
        &[0, 2, 3, 5, 7, 8, 10]
    } else {
        &[0, 2, 4, 5, 7, 9, 11]
    };
    let mut s = [false; 12];
    for &iv in intervals {
        s[((tonic + iv) % 12) as usize] = true;
    }
    s
}

/// Map each frame's time to the index of the last beat at or before it.
fn bar_for_frame(
    frame_time_sec: f32,
    beat_times: &[f32],
    beats_per_bar: usize,
    beat_ptr: &mut usize,
) -> usize {
    while *beat_ptr + 1 < beat_times.len() && beat_times[*beat_ptr + 1] <= frame_time_sec {
        *beat_ptr += 1;
    }
    *beat_ptr / beats_per_bar
}

/// Highest MIDI pitch included in the chord profile. The melody register
/// (above B4) is excluded because sustained melody tones would otherwise be
/// read as chord extensions (9ths/13ths) and every chord would over-extend.
const CHORD_MAX_PITCH: usize = 71;

/// Profile sub-window width in beats. Chord voicings are struck at the beat;
/// sub-window sampling lets the harmonic detector look at the strike moment
/// instead of averaging the whole bar (which includes melody movement, note
/// releases and the walking bass, drowning the voicing in the model's
/// diffuse probability floor).
const PROFILE_BIN_BEATS: f32 = 0.25;

/// Build one soft pitch-class profile per bar from the probability timelines.
///
/// Frames are aggregated over a sub-window chosen from the chord-sampling
/// knobs (`config.chord_sample_beat` / `chord_sample_cleanest` /
/// `chord_sample_strike`; the default whole-bar mean is the legacy behavior):
/// - `chord_sample_beat >= 0`: the 0.5-beat window centered on that beat
///   offset (0.0 = the downbeat strike zone).
/// - `chord_sample_cleanest`: scan the first half of the bar in 0.25-beat
///   windows and pick the one with the fewest active pitch classes (still
///   `>= chord_min_simultaneous`).
/// - `chord_sample_strike`: scan the first half and pick the window where
///   the most notes onset together (onset head, or frame-to-frame activation
///   delta when no onset timeline is available).
pub fn compute_bar_profiles(
    input: &TimelineChordInput,
    beat_times: &[f32],
    beats_per_bar: u32,
    config: &ChordAnalysisConfig,
) -> Vec<BarProfile> {
    if beat_times.len() < 2 || input.timelines.is_empty() {
        return Vec::new();
    }
    if std::env::var_os("KEYSCRIBE_HARMONY_DEBUG").is_some() {
        for (ti, (timeline, step_sec)) in input.timelines.iter().enumerate().take(4) {
            let n_frames = timeline.len();
            let n_active = timeline
                .iter()
                .flatten()
                .filter(|&&p| p > 0.5)
                .count();
            let n_cells = n_frames * 88;
            let over05 = timeline
                .iter()
                .filter(|f| f.iter().filter(|&&p| p > 0.5).count() >= 3)
                .count();
            let (mut mn, mut mx) = (f32::MAX, f32::MIN);
            for f in timeline.iter().take(200) {
                for &p in f.iter().take(88) {
                    mn = mn.min(p);
                    mx = mx.max(p);
                }
            }
            eprintln!(
                "[harmony] timeline {ti}: frames={n_frames} step={step_sec:.4}s cells>0.5={n_active}/{n_cells} frames_with_3pcs>0.5={over05}/{n_frames} min={mn:.2} max={mx:.2}"
            );
        }
    }
    let bpb = beats_per_bar.max(2) as usize;
    let num_bars = ((beat_times.len().saturating_sub(1) as f32 / bpb as f32).ceil() as usize).max(1);
    let bins_per_bar = ((bpb as f32 / PROFILE_BIN_BEATS).round() as usize).max(1);
    let bins_total = num_bars * bins_per_bar;
    let has_onset_timelines = input.onset_timelines.len() == input.timelines.len()
        && input
            .onset_timelines
            .iter()
            .all(|(o, _)| !o.is_empty());

    // Per-window accumulators: pitch-class sums (chord + bass registers),
    // per-pitch maxima, frame counts, onset-head energy and frame-to-frame
    // activation delta (strike fallback).
    let mut bin_pcp = vec![[0.0f32; 12]; bins_total];
    let mut bin_bass = vec![[0.0f32; 12]; bins_total];
    let mut bin_max = vec![[0.0f32; 88]; bins_total];
    let mut bin_frames = vec![0usize; bins_total];
    let mut bin_onset = vec![0.0f32; bins_total];
    let mut bin_delta = vec![0.0f32; bins_total];

    for (ti, (timeline, step_sec)) in input.timelines.iter().enumerate() {
        if timeline.is_empty() || *step_sec <= 0.0 {
            continue;
        }
        let onset_tl = input.onset_timelines.get(ti).map(|(o, _)| o.as_slice());
        let mut beat_ptr = 0usize;
        let mut prev_total = 0.0f32;
        for (f, frame) in timeline.iter().enumerate() {
            let t = f as f32 * step_sec;
            let bar = bar_for_frame(t, beat_times, bpb, &mut beat_ptr).min(num_bars - 1);
            let bar_start = beat_times[(bar * bpb).min(beat_times.len().saturating_sub(1))];
            let beat_idx = beat_ptr.min(beat_times.len().saturating_sub(1));
            let beat_dur =
                (beat_times[(beat_idx + 1).min(beat_times.len().saturating_sub(1))]
                    - beat_times[beat_idx])
                    .max(1e-3);
            let within = ((t - bar_start).max(0.0) / beat_dur).min(bpb as f32);
            let bin = bar * bins_per_bar
                + ((within / PROFILE_BIN_BEATS) as usize).min(bins_per_bar - 1);
            bin_frames[bin] += 1;

            let mut total = 0.0f32;
            for (i, &p) in frame.iter().enumerate().take(CHORD_MAX_PITCH - 20) {
                if p <= 0.0 {
                    continue;
                }
                let pc = ((21 + i) % 12) as usize;
                bin_pcp[bin][pc] += p;
                if 21 + i < 60 {
                    bin_bass[bin][pc] += p;
                }
                if p > bin_max[bin][i] {
                    bin_max[bin][i] = p;
                }
                total += p;
            }
            if total > prev_total {
                bin_delta[bin] += total - prev_total;
            }
            prev_total = total;
            if let Some(otl) = onset_tl {
                if let Some(of) = otl.get(f) {
                    bin_onset[bin] += of.iter().take(CHORD_MAX_PITCH - 20).sum::<f32>();
                }
            }
        }
    }

    let mut profiles = Vec::with_capacity(num_bars);
    for bar in 0..num_bars {
        let base = bar * bins_per_bar;
        let bins: Vec<usize> = if config.chord_sample_beat >= 0.0 {
            let center = ((config.chord_sample_beat / PROFILE_BIN_BEATS).round() as usize)
                .min(bins_per_bar.saturating_sub(1));
            let hi = (center + 1).min(bins_per_bar);
            (center..hi).collect()
        } else if config.chord_sample_cleanest || config.chord_sample_strike {
            let half = (bins_per_bar / 2).max(1);
            let mut best_clean: Option<(usize, usize)> = None; // (bin, active count)
            let mut best_mass: Option<(usize, usize)> = None; // fallback: most active
            let mut best_strike: Option<(usize, f32)> = None;
            for b in 0..half {
                if bin_frames[base + b] == 0 {
                    continue;
                }
                let active = whitened_active_count(&bin_pcp[base + b], bin_frames[base + b]);
                if config.chord_sample_cleanest {
                    if active >= config.chord_min_simultaneous {
                        if best_clean.is_none_or(|(_, c)| active < c) {
                            best_clean = Some((b, active));
                        }
                    } else if best_mass.is_none_or(|(_, c)| active > c) {
                        best_mass = Some((b, active));
                    }
                } else {
                    let e = if has_onset_timelines {
                        bin_onset[base + b]
                    } else {
                        bin_delta[base + b]
                    };
                    if best_strike.is_none_or(|(_, s)| e > s) {
                        best_strike = Some((b, e));
                    }
                }
            }
            if config.chord_sample_cleanest {
                if let Some((b, _)) = best_clean.or(best_mass) {
                    vec![b]
                } else {
                    (0..bins_per_bar).collect()
                }
            } else if let Some((b, _)) = best_strike {
                vec![b]
            } else {
                (0..bins_per_bar).collect()
            }
        } else {
            (0..bins_per_bar).collect()
        };

        let mut pcp = [0.0f32; 12];
        let mut bass = [0.0f32; 12];
        let mut mx = [0.0f32; 88];
        let mut n = 0usize;
        for &b in &bins {
            if bin_frames[base + b] == 0 {
                continue;
            }
            n += bin_frames[base + b];
            for pc in 0..12 {
                pcp[pc] += bin_pcp[base + b][pc];
                bass[pc] += bin_bass[base + b][pc];
            }
            for i in 0..88 {
                if bin_max[base + b][i] > mx[i] {
                    mx[i] = bin_max[base + b][i];
                }
            }
        }
        let n = n.max(1) as f32;
        let mut prof = BarProfile {
            bar_index: bar,
            pcp: [0.0; 12],
            bass_pcp: [0.0; 12],
            bass_note: None,
            energy: 0.0,
        };
        for pc in 0..12 {
            prof.pcp[pc] = pcp[pc] / n;
            prof.bass_pcp[pc] = bass[pc] / n;
            prof.energy += prof.pcp[pc];
        }
        for i in 0..88 {
            if mx[i] >= 0.25 {
                prof.bass_note = Some((21 + i) as u8);
                break;
            }
        }
        profiles.push(prof);
    }
    profiles
}

/// Number of pitch classes clearly above the noise floor of a sub-window's
/// mean activation profile (used by the `cleanest` sampling scan).
fn whitened_active_count(pcp_sum: &[f32; 12], frame_count: usize) -> usize {
    let mut mean = [0.0f32; 12];
    let f = frame_count.max(1) as f32;
    for pc in 0..12 {
        mean[pc] = pcp_sum[pc] / f;
    }
    let mut sorted = mean.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = 0.7 * sorted[sorted.len() / 2];
    let maxv = mean.iter().map(|v| (v - floor).max(0.0)).fold(0.0f32, f32::max);
    if maxv <= 0.01 {
        return 0;
    }
    mean.iter()
        .filter(|&&v| (v - floor).max(0.0) >= 0.25 * maxv)
        .count()
}

/// Krumhansl-Kessler profiles for major and minor keys.
const KK_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KK_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Estimate the key from the aggregate pitch-class profile (P2).
pub fn estimate_key(profiles: &[BarProfile]) -> KeyInfo {
    let mut agg = [0.0f32; 12];
    for p in profiles {
        for pc in 0..12 {
            agg[pc] += p.pcp[pc];
        }
    }
    let sum: f32 = agg.iter().sum();
    if sum <= 0.0 {
        return KeyInfo {
            tonic: 0,
            minor: false,
            scale: scale_pcs(0, false),
        };
    }
    let mean = sum / 12.0;
    let sd = (agg.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 12.0)
        .sqrt()
        .max(1e-6);
    let normalized: [f32; 12] = agg.map(|v| (v - mean) / sd);

    let mut best = (0u8, false, f32::MIN);
    for tonic in 0..12u8 {
        for (minor, prof) in [(false, KK_MAJOR), (true, KK_MINOR)] {
            let mut corr = 0.0f32;
            for i in 0..12 {
                corr += normalized[i]
                    * prof[((i as i32 - tonic as i32).rem_euclid(12)) as usize];
            }
            if corr > best.2 {
                best = (tonic, minor, corr);
            }
        }
    }
    KeyInfo {
        tonic: best.0,
        minor: best.1,
        scale: scale_pcs(best.0, best.1),
    }
}

/// Extract the chord-tone pitch classes of a bar from its whitened profile:
/// pitch classes whose activation sits clearly above the noise floor, capped
/// at the strongest 6 (P1).
fn pcs_from_profile(whitened: &[f32; 12]) -> Vec<u8> {
    let maxv = whitened.iter().cloned().fold(0.0f32, f32::max);
    if maxv <= 0.01 {
        return Vec::new();
    }
    let thresh = 0.30 * maxv;
    let mut pcs: Vec<u8> = (0..12u8)
        .filter(|&pc| whitened[pc as usize] >= thresh)
        .collect();
    pcs.sort_by(|a, b| {
        whitened[*b as usize]
            .partial_cmp(&whitened[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pcs.truncate(6);
    pcs.sort();
    pcs
}

/// Chord qualities used in the Viterbi emission. The full set is scored
/// directly (extended qualities included) because dense voicings like Gm13
/// have their 9th and 11th sounding in the profile — a basic-only set makes
/// the root score collapse and the extended chord is never reachable.
pub(crate) const ALL_TEMPLATES: [(&str, &[u8]); 27] = [
    ("", &[0, 4, 7]),
    ("-", &[0, 3, 7]),
    ("dim", &[0, 3, 6]),
    ("aug", &[0, 4, 8]),
    ("sus2", &[0, 2, 7]),
    ("sus4", &[0, 5, 7]),
    ("7", &[0, 4, 7, 10]),
    ("\u{0394}7", &[0, 4, 7, 11]),
    ("-7", &[0, 3, 7, 10]),
    ("-\u{0394}7", &[0, 3, 7, 11]),
    ("dim7", &[0, 3, 6, 9]),
    ("-7b5", &[0, 3, 6, 10]),
    ("7#5", &[0, 4, 8, 10]),
    ("6", &[0, 4, 7, 9]),
    ("m6", &[0, 3, 7, 9]),
    ("9", &[0, 4, 7, 10, 2]),
    ("\u{0394}9", &[0, 4, 7, 11, 2]),
    ("-9", &[0, 3, 7, 10, 2]),
    ("7b9", &[0, 4, 7, 10, 1]),
    ("7#9", &[0, 4, 7, 10, 3]),
    ("7#11", &[0, 4, 7, 10, 6]),
    ("\u{0394}7#11", &[0, 4, 7, 11, 6]),
    ("-11", &[0, 3, 7, 10, 2, 5]),
    ("13", &[0, 4, 7, 10, 2, 9]),
    ("\u{0394}13", &[0, 4, 7, 11, 2, 9]),
    ("-13", &[0, 3, 7, 10, 2, 9]),
    ("\u{0394}9#11", &[0, 4, 7, 11, 2, 6]),
];

/// Simplicity bonus so a plain triad/7th wins unless the extension tones are
/// clearly present in the profile. Each extra template interval adds ~0.15 of
/// spurious mass on a noisy floor, so the penalty must match that to keep
/// chords like Cmaj7 instead of inflating everything to Cmaj13.
fn simplicity_bonus(intervals: &[u8]) -> f32 {
    match intervals.len() {
        3 => 0.08,
        4 => 0.05,
        5 => -0.15,
        6 => -0.30,
        _ => -0.40,
    }
}

/// Soft contrast score: mass on chord tones contributes positively, mass on
/// non-chord tones negatively, weighted by activation strength. Unlike a
/// plain "explained fraction", a strong non-chord tone (e.g. C under a G13)
/// is punished, giving real dynamic range on noisy profiles (P3).
fn soft_contrast(whitened: &[f32; 12], root: u8, intervals: &[u8]) -> f32 {
    let mut target = [0.0f32; 12];
    for &iv in intervals {
        target[((root + iv) % 12) as usize] = 1.0;
    }
    // ±1 semitone tolerance: neighbors count at a quarter weight. Keeping this
    // small is essential for quality discrimination: a natural 7th (B) vs a
    // flat 7th (Bb) are a semitone apart, and a large neighbor credit would
    // let the wrong quality win (C7 scoring above Cmaj7 when the audio has a
    // strong B).
    let mut blurred = [0.0f32; 12];
    for i in 0..12 {
        blurred[i] = (target[i] + 0.25 * target[(i + 1) % 12] + 0.25 * target[(i + 11) % 12])
            .min(1.0);
    }
    let total: f32 = whitened.iter().sum();
    if total <= 1e-4 {
        return 0.0;
    }
    let mut s = 0.0f32;
    for pc in 0..12 {
        s += (whitened[pc] / total) * (2.0 * blurred[pc] - 1.0);
    }
    s
}

/// Nonlinear sharpening applied to the whitened pitch-class profile before
/// template matching. Strong chord tones are emphasized and the weak diffuse
/// tail (the model's probability floor, melody leakage) is de-emphasized, so
/// a barely-audible extension tone can't extend every chord to a 13th. A
/// squared profile also gives the root more separation from equally-sparse
/// slash-chord alternatives.
const WHITEN_SHARPEN_POWER: f32 = 1.5;

/// Detect chords per bar from the soft probability timeline: whitened per-bar
/// pitch-class profiles (P1), joint (root, quality) template matching with a
/// bass anchor (P3), a key-estimate prior, and a functional-harmony Viterbi
/// pass over the bar sequence (P2: circle-of-fifths cadences, stepwise motion,
/// sustained roots). One chord per bar, placed at the downbeat; extended
/// qualities are applied only when the extension tone is clearly present.
pub fn detect_chords_from_timeline(
    input: &TimelineChordInput,
    beat_times: &[f32],
    beats_per_bar: u32,
    config: ChordAnalysisConfig,
) -> Vec<ChordSymbolChange> {
    if config.skip || input.timelines.is_empty() || beat_times.len() < 2 {
        return Vec::new();
    }
    let profiles = compute_bar_profiles(input, beat_times, beats_per_bar, &config);
    if profiles.is_empty() {
        return Vec::new();
    }
    let key = estimate_key(&profiles);
    let profile_step = beats_per_bar.max(2) as f32;

    let n_qualities = ALL_TEMPLATES.len();
    let n_states = 12 * n_qualities;
    let mut emissions = vec![vec![0.0f32; n_states]; profiles.len()];
    let mut whitened_energy = vec![0.0f32; profiles.len()];
    for (bar, prof) in profiles.iter().enumerate() {
        let mut sorted: Vec<f32> = prof.pcp.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let floor = 0.7 * sorted[sorted.len() / 2];
        let mut whitened = [0.0f32; 12];
        for pc in 0..12 {
            whitened[pc] = (prof.pcp[pc] - floor).max(0.0).powf(WHITEN_SHARPEN_POWER);
            whitened_energy[bar] += whitened[pc];
        }
        // Bass-register profile (P3): the walking bass is nearly a delta
        // function on one pitch class (root or a neighbor), so templates are
        // scored against both the full profile and the bass profile, and the
        // root additionally gets a direct bass-pitch bonus. This breaks the
        // C6/Am7-style relative-chord ties the comp register can't resolve.
        let mut bass_sorted: Vec<f32> = prof.bass_pcp.to_vec();
        bass_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bass_floor = 0.7 * bass_sorted[bass_sorted.len() / 2];
        let bass_whitened: [f32; 12] = prof
            .bass_pcp
            .map(|v| (v - bass_floor).max(0.0).powf(WHITEN_SHARPEN_POWER));
        for root in 0..12u8 {
            for (q, (_, intervals)) in ALL_TEMPLATES.iter().enumerate() {
                let mut e = 0.55 * soft_contrast(&whitened, root, intervals)
                    + 0.45 * soft_contrast(&bass_whitened, root, intervals)
                    + simplicity_bonus(intervals);
                if let Some(b) = prof.bass_note {
                    let b = b % 12;
                    if b == root {
                        e += 0.45;
                    } else if (root + 7) % 12 == b || (root + 5) % 12 == b {
                        e += 0.15;
                    } else if (root + 3) % 12 == b || (root + 4) % 12 == b {
                        e += 0.06;
                    }
                }
                if key.scale[root as usize] {
                    e += 0.08;
                } else if key.scale[((root as i32 + 7).rem_euclid(12)) as usize]
                    || key.scale[((root as i32 + 4).rem_euclid(12)) as usize]
                {
                    // Secondary dominants: roots a 5th or major-3rd (V of scale)
                    // are common even when not the tonic key's scale.
                    e += 0.04;
                }
                emissions[bar][root as usize * n_qualities + q] = e;
            }
        }
    }

    if std::env::var_os("KEYSCRIBE_HARMONY_DEBUG").is_some() {
        for (bar, prof) in profiles.iter().enumerate().take(40) {
            let mut sorted: Vec<f32> = prof.pcp.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let floor = 0.7 * sorted[sorted.len() / 2];
            let whitened: [f32; 12] = prof
                .pcp
                .map(|v| (v - floor).max(0.0).powf(WHITEN_SHARPEN_POWER));
            let pcs = pcs_from_profile(&whitened);
            let mut ranked: Vec<(u8, usize, f32)> = Vec::new();
            for root in 0..12u8 {
                for q in 0..n_qualities {
                    ranked.push((root, q, emissions[bar][root as usize * n_qualities + q]));
                }
            }
            ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = ranked
                .iter()
                .take(3)
                .map(|(r, q, e)| {
                    format!(
                        "{}={:.2}",
                        chord_symbol_from_root(*r, ALL_TEMPLATES[*q].0, prof.bass_note.map(|b| b % 12)),
                        e
                    )
                })
                .collect();
            let pcp_str: Vec<String> = prof
                .pcp
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > 0.02)
                .map(|(pc, v)| format!("{}:{:.2}", pc, v))
                .collect();
            let bass_str: Vec<String> = prof
                .bass_pcp
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > 0.02)
                .map(|(pc, v)| format!("{}:{:.2}", pc, v))
                .collect();
            eprintln!(
                "[harmony] bar {:>2} bass={:?} basspcp[{}] key={}{} pcs=[{}] pcp[{}] top[{}]",
                bar,
                prof.bass_note,
                bass_str.join(" "),
                pitch_class_char(key.tonic),
                if key.minor { "m" } else { "" },
                pcs.iter()
                    .map(|pc| pitch_class_char(*pc))
                    .collect::<Vec<_>>()
                    .join(" "),
                pcp_str.join(" "),
                top.join(" ")
            );
        }
    }

    // P2: functional-harmony Viterbi over the bar sequence. The per-bar
    // argmax is often a coin flip between equally-plausible templates; the
    // transition prior (cadential V-I roots, sustained roots, common stepwise
    // motion) resolves those ties while staying weak enough that a strong
    // emission still wins. When a learned (root, quality) transition matrix
    // was loaded from a corpus, it is blended in as the data-driven prior.
    let path = viterbi_best_path(&emissions, 12, n_qualities, learned_transitions().as_deref());

    // Emit one chord per bar at the downbeat, following the Viterbi path.
    // The key estimate still influences the choice through the diatonic prior
    // in the emission.
    let min_energy = (config.chord_min_simultaneous.max(1) as f32) * 0.2;
    let mut out = Vec::new();
    let mut last_symbol = String::new();
    for (bar, prof) in profiles.iter().enumerate() {
        if whitened_energy[bar] < min_energy {
            continue;
        }
        let s = path[bar];
        let root = (s / n_qualities) as u8;
        let suffix = ALL_TEMPLATES[s % n_qualities].0;
        let symbol = chord_symbol_from_root(root, suffix, prof.bass_note.map(|b| b % 12));
        if symbol == last_symbol {
            continue;
        }
        out.push(ChordSymbolChange {
        beat_start: bar as f32 * profile_step,
            symbol: symbol.clone(),
        });
        last_symbol = symbol;
    }
    out
}

/// Viterbi decode over the bar sequence. States are (root, quality); the
/// transition prior encodes functional-harmony root movement (V-I cadences,
/// sustained roots, common stepwise motion) plus a mild quality-continuity
/// preference. Costs are small relative to per-bar emission differences so
/// the prior only breaks ties and corrects isolated errors. When a learned
/// `(root, quality)` transition matrix is supplied, its blended cost replaces
/// the hand-tuned one for every (source, target) pair both observed in the
/// corpus.
fn viterbi_best_path(
    emissions: &[Vec<f32>],
    n_roots: usize,
    n_qualities: usize,
    transitions: Option<&crate::leadsheet::transition::LearnedTransitions>,
) -> Vec<usize> {
    let n_bars = emissions.len();
    let n_states = n_roots * n_qualities;
    if n_bars == 0 || n_states == 0 {
        return Vec::new();
    }
    if n_bars == 1 {
        let (s, _) = emissions[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return vec![s];
    }

    // Root-to-root transition cost by directed interval d = (r2 - r1) mod 12.
    // d=5 (V -> I) is the strongest move; large leaps and the tritone cost the
    // most. Values stay in the 0.02..0.40 range so they never swamp a genuine
    // per-bar emission difference.
    let mut root_cost = [[0.0f32; 12]; 12];
    for r1 in 0..12 {
        for r2 in 0..12 {
            root_cost[r1][r2] = match ((r2 + 12 - r1) % 12) as usize {
                5 => 0.00,
                0 => 0.02,
                7 => 0.06,
                10 => 0.08,
                2 => 0.10,
                9 => 0.14,
                3 => 0.16,
                11 => 0.20,
                1 => 0.22,
                8 => 0.24,
                4 => 0.26,
                _ => 0.40,
            };
        }
    }
    const QUALITY_CHANGE_COST: f32 = 0.03;

    let mut dp = vec![f32::NEG_INFINITY; n_states];
    let mut back = vec![0usize; n_bars * n_states];
    for s in 0..n_states {
        dp[s] = emissions[0][s];
    }
    for bar in 1..n_bars {
        let prev = dp.clone();
        for s2 in 0..n_states {
            let r2 = s2 / n_qualities;
            let q2 = s2 % n_qualities;
            let mut best = f32::NEG_INFINITY;
            let mut best_s1 = 0usize;
            for s1 in 0..n_states {
                let r1 = (s1 / n_qualities) as u8;
                let q1 = s1 % n_qualities;
                let mut c = root_cost[s1 / n_qualities][r2];
                if s1 % n_qualities != q2 {
                    c += QUALITY_CHANGE_COST;
                }
                if let Some(lt) = transitions {
                    if let Some(learned) = lt.cost((r1, q1), (r2 as u8, q2)) {
                        c = learned;
                    }
                }
                let v = prev[s1] + c;
                if v > best {
                    best = v;
                    best_s1 = s1;
                }
            }
            dp[s2] = best + emissions[bar][s2];
            back[bar * n_states + s2] = best_s1;
        }
    }

    let mut path = vec![0usize; n_bars];
    let mut s = (0..n_states)
        .max_by(|&a, &b| dp[a].partial_cmp(&dp[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    path[n_bars - 1] = s;
    for bar in (1..n_bars).rev() {
        s = back[bar * n_states + s];
        path[bar - 1] = s;
    }
    path
}
