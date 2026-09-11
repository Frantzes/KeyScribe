//! Learned (root, quality) transition matrix fitted from a chord-symbol corpus.
//!
//! The hand-tuned Viterbi root-cost table in `harmony.rs` encodes only root
//! *interval* movement. This module learns the **joint** root+quality
//! transition probabilities from a corpus of jazz-standard lead sheets
//! (Omnibook MusicXML), which captures tendencies a root-interval table cannot:
//! `ii-7 -> V7 -> I` with the right qualities each step, `V7 -> VI-7`,
//! blues changes, and the long functional tail. The fitted matrix plugs into
//! the same Viterbi (`harmony::viterbi_best_path`) as a data-driven
//! transition prior, blended against the hand-tuned costs so a sparse
//! corpus can never dominate a strong per-bar emission.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::Context;

use crate::leadsheet::harmony::ALL_TEMPLATES;
use crate::sheet_compare::{parse_musicxml_harmonies, Harmony};

/// Map a MusicXML `<kind>` value to an index into [`ALL_TEMPLATES`].
fn quality_idx_for_kind(kind: &str) -> Option<usize> {
    let suffix = match kind {
        "" | "major" => "",
        "minor" => "-",
        "dominant" => "7",
        "major-seventh" => "\u{0394}7",
        "minor-seventh" => "-7",
        "half-diminished" => "-7b5",
        "diminished" => "dim",
        "augmented" => "aug",
        "diminished-seventh" => "dim7",
        "minor-major-seventh" => "-\u{0394}7",
        "major-sixth" => "6",
        "minor-sixth" => "m6",
        "suspended-second" => "sus2",
        "suspended-fourth" => "sus4",
        "major-ninth" => "\u{0394}9",
        "minor-ninth" => "-9",
        "dominant-ninth" => "9",
        "dominant-13th" => "13",
        "minor-13th" => "-13",
        _ => return None,
    };
    ALL_TEMPLATES.iter().position(|(s, _)| *s == suffix)
}

/// Parse a root name ("C", "Bb", "F#") to a pitch class.
fn root_pc_from_name(root: &str) -> Option<u8> {
    let base = match root.chars().next()? {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    match root.chars().nth(1) {
        Some('#') => Some((base + 1) % 12),
        Some('b') => Some((base + 11) % 12),
        _ => Some(base),
    }
}

/// Map a corpus `Harmony` onto the (root, quality) emission space used by the
/// timeline Viterbi. Returns `None` for chords the 27-template vocabulary
/// cannot represent.
fn harmony_to_state(h: &Harmony) -> Option<(u8, usize)> {
    let root = root_pc_from_name(&h.root)?;
    let q = quality_idx_for_kind(&h.kind)?;
    Some((root, q))
}

/// Data-driven (root, quality) transition model, fitted by counting chord
/// adjacencies at bar/harmony resolution across a corpus and Laplace-smoothing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LearnedTransitions {
    /// Laplace smoothing alpha used at fit time.
    pub alpha: f32,
    /// Corpus adjacency transitions counted.
    pub n_transitions: usize,
    /// Mapped (root, quality) states observed in the corpus, densely indexed.
    pub states: Vec<(u8, usize)>,
    /// Row-normalized log P(target | source), row-major M x M.
    pub log_probs: Vec<Vec<f32>>,
    /// Weight given to the learned cost when blended against the hand-tuned
    /// Viterbi costs (0.0 = pure hand-tuned, 1.0 = pure learned).
    pub blend_w: f32,
    /// Multiplier applied to `-ln P` so learned costs land in the same
    /// 0.02..0.45 range as the hand-tuned root/quality costs.
    pub scale: f32,
    /// Maximum learned cost after scaling (keeps rare-but-valid moves
    /// possible instead of letting a sparse row veto them).
    pub cap: f32,

    #[serde(skip)]
    state_index: HashMap<(u8, usize), usize>,
}

impl LearnedTransitions {
    /// Hand-tuned transition cost for (r1,q1) -> (r2,q2), mirroring
    /// `harmony::viterbi_best_path` so the learned table blends against the
    /// same baseline (root interval cost + quality-change cost).
    fn hand_cost(&self, r1: u8, q1: usize, r2: u8, q2: usize) -> f32 {
        let d = ((r2 + 12 - r1) % 12) as usize;
        let root_cost = match d {
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
        root_cost + if q1 != q2 { 0.03 } else { 0.0 }
    }

    /// Blended transition cost from `from` to `to`, or `None` when either
    /// state was never observed in the corpus (caller falls back to the
    /// hand-tuned table alone).
    pub fn cost(&self, from: (u8, usize), to: (u8, usize)) -> Option<f32> {
        let i = *self.state_index.get(&from)?;
        let j = *self.state_index.get(&to)?;
        let learned = (-self.log_probs[i][j] * self.scale).clamp(0.0, self.cap);
        let hand = self.hand_cost(from.0, from.1, to.0, to.1);
        Some(self.blend_w * learned + (1.0 - self.blend_w) * hand)
    }

    fn rebuild_index(&mut self) {
        self.state_index = self
            .states
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();
    }
}

/// Fit a transition matrix from every MusicXML file in `dir` (document-order
/// chord adjacencies per song). `alpha` is the Laplace smoothing parameter.
pub fn fit_transitions_from_musicxml_dir(
    dir: &Path,
    alpha: f32,
) -> anyhow::Result<LearnedTransitions> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("open corpus dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| x.to_string_lossy().eq_ignore_ascii_case("xml"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    let mut songs = 0usize;
    let mut chords = 0usize;
    let mut transitions = 0usize;
    let mut all_states: std::collections::BTreeSet<(u8, usize)> = std::collections::BTreeSet::new();
    let mut counts: HashMap<(u8, usize), HashMap<(u8, usize), usize>> = HashMap::new();

    for f in &files {
        let xml =
            std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
        let harms = parse_musicxml_harmonies(&xml)?;
        let mut prev: Option<(u8, usize)> = None;
        for h in &harms {
            if let Some(state) = harmony_to_state(h) {
                chords += 1;
                all_states.insert(state);
                if let Some(p) = prev {
                    *counts.entry(p).or_default().entry(state).or_insert(0) += 1;
                    transitions += 1;
                }
                prev = Some(state);
            }
        }
        songs += 1;
    }

    let states: Vec<(u8, usize)> = all_states.into_iter().collect();
    let m = states.len();
    if m == 0 {
        anyhow::bail!(
            "no mappable chords found in {} ({} songs, {} chords)",
            dir.display(),
            songs,
            chords
        );
    }

    let mut log_probs = vec![vec![0.0f32; m]; m];
    for (i, &src) in states.iter().enumerate() {
        let row = counts.get(&src);
        let row_total: usize = row.map(|r| r.values().sum()).unwrap_or(0);
        let denom = row_total as f32 + alpha * m as f32;
        for (j, &dst) in states.iter().enumerate() {
            let n = row.and_then(|r| r.get(&dst)).copied().unwrap_or(0) as f32;
            log_probs[i][j] = ((n + alpha) / denom).ln();
        }
    }

    let mut lt = LearnedTransitions {
        alpha,
        n_transitions: transitions,
        states,
        log_probs,
        blend_w: 0.65,
        scale: 0.05,
        cap: 0.45,
        state_index: HashMap::new(),
    };
    lt.rebuild_index();
    eprintln!(
        "[transition] fitted {} transitions from {songs} songs ({chords} chords, {m} states, alpha={alpha})",
        transitions
    );
    Ok(lt)
}

impl LearnedTransitions {
    /// Deserialize from a JSON string, rebuilding the lookup index.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let mut lt: LearnedTransitions = serde_json::from_str(json)
            .context("failed to parse transition matrix JSON")?;
        lt.rebuild_index();
        Ok(lt)
    }
}

/// Process-wide cache of the learned transition matrix. Loaded explicitly by
/// the CLI (or any embedder) via [`load_learned_transitions`]; until then the
/// chord detector silently uses the hand-tuned Viterbi table.
static TRANSITIONS: OnceLock<Option<Arc<LearnedTransitions>>> = OnceLock::new();

/// Load the learned transition matrix from `path` once per process. Returns
/// the loaded matrix, or `None` if the file is missing/invalid (the chord
/// detector keeps its hand-tuned fallback). Subsequent calls return the cached
/// result regardless of `path`.
pub fn load_learned_transitions(path: Option<&Path>) -> Option<Arc<LearnedTransitions>> {
    TRANSITIONS
        .get_or_init(|| {
            let path = path?;
            let bytes = std::fs::read(path).ok()?;
            let text = String::from_utf8(bytes).ok()?;
            match LearnedTransitions::from_json(&text) {
                Ok(lt) => {
                    eprintln!(
                        "[transition] loaded {} states, {} transitions from {}",
                        lt.states.len(),
                        lt.n_transitions,
                        path.display()
                    );
                    Some(Arc::new(lt))
                }
                Err(e) => {
                    eprintln!("[transition] ignoring {}: {e:#}", path.display());
                    None
                }
            }
        })
        .clone()
}

/// Reference to the process-wide learned transitions, if loaded.
pub(crate) fn learned_transitions() -> Option<Arc<LearnedTransitions>> {
    TRANSITIONS.get().and_then(|t| t.as_ref()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leadsheet::harmony::ALL_TEMPLATES;

    fn quality_suffix_idx(suffix: &str) -> usize {
        ALL_TEMPLATES
            .iter()
            .position(|(s, _)| *s == suffix)
            .unwrap()
    }

    #[test]
    fn maps_musicxml_kinds_onto_template_space() {
        let h = Harmony {
            onset_beats: 0.0,
            root: "Bb".to_string(),
            kind: "dominant".to_string(),
            suffix: String::new(),
        };
        assert_eq!(harmony_to_state(&h), Some((10, quality_suffix_idx("7"))));

        let h = Harmony {
            onset_beats: 0.0,
            root: "E".to_string(),
            kind: "half-diminished".to_string(),
            suffix: String::new(),
        };
        assert_eq!(harmony_to_state(&h), Some((4, quality_suffix_idx("-7b5"))));

        let h = Harmony {
            onset_beats: 0.0,
            root: "G".to_string(),
            kind: "major".to_string(),
            suffix: String::new(),
        };
        assert_eq!(harmony_to_state(&h), Some((7, quality_suffix_idx(""))));

        // A quality outside the 27-template vocabulary is skipped.
        let h = Harmony {
            onset_beats: 0.0,
            root: "C".to_string(),
            kind: "sus-whatever".to_string(),
            suffix: String::new(),
        };
        assert_eq!(harmony_to_state(&h), None);
    }

    fn mini_musicxml(harmonies: &[(&str, &str)]) -> String {
        // (root step, kind) pairs rendered as whole-note measures.
        let mut measures = String::new();
        for (m, (step, kind)) in harmonies.iter().enumerate() {
            measures.push_str(&format!(
                r#"<measure number="{}"><note><pitch><step>{}</step><octave>4</octave></pitch><duration>4</duration></note><harmony><root><root-step>{}</root-step></root><kind>{}</kind></harmony></measure>"#,
                m + 1,
                step,
                step,
                kind
            ));
        }
        format!(
            r#"<?xml version="1.0"?><score-partwise><part><attributes><divisions>4</divisions></attributes>{}</part></score-partwise>"#,
            measures
        )
    }

    #[test]
    fn fits_adjacency_counts_from_corpus() {
        let dir = std::env::temp_dir().join(format!("ks_transition_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Song 1: Dm7 G7 Dm7 G7 Cmaj7 — Dm7->G7 (ii->V) occurs twice.
        std::fs::write(
            dir.join("a.xml"),
            mini_musicxml(&[
                ("D", "minor"),
                ("G", "dominant"),
                ("D", "minor"),
                ("G", "dominant"),
                ("C", "major"),
            ]),
        )
        .unwrap();
        // Song 2: Am7 D7 Gmaj7 (ii-V-I in G).
        std::fs::write(
            dir.join("b.xml"),
            mini_musicxml(&[
                ("A", "minor"),
                ("D", "dominant"),
                ("G", "major"),
            ]),
        )
        .unwrap();

        let lt = fit_transitions_from_musicxml_dir(&dir, 0.5).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(lt.n_transitions, 6);
        // Corpus qualities are Real-Book triads: "minor" -> "-", "major" -> "".
        let dm = (2, quality_suffix_idx("-"));
        let g7 = (7, quality_suffix_idx("7"));
        let c = (0, quality_suffix_idx(""));

        // Target-only chords (the final I) are still matrix states.
        assert!(lt.states.contains(&c));

        // The repeated ii->V must cost less than the single V->I (both are
        // d=5 in the hand table, so the learned counts decide).
        assert!(
            lt.cost(dm, g7).unwrap() < lt.cost(g7, c).unwrap(),
            "repeated ii->V should be cheaper than single V->I: {} vs {}",
            lt.cost(dm, g7).unwrap(),
            lt.cost(g7, c).unwrap()
        );
        // A never-observed (root, quality) state has no learned cost.
        assert!(lt.cost((5, quality_suffix_idx("7")), c).is_none());
    }

    #[test]
    fn json_roundtrip_rebuilds_index() {
        let dir = std::env::temp_dir().join(format!("ks_transition_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.xml"),
            mini_musicxml(&[("D", "minor"), ("G", "dominant")]),
        )
        .unwrap();
        let lt = fit_transitions_from_musicxml_dir(&dir, 0.5).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        let json = serde_json::to_string(&lt).unwrap();
        let reloaded = LearnedTransitions::from_json(&json).unwrap();
        let dm = (2, quality_suffix_idx("-"));
        let g7 = (7, quality_suffix_idx("7"));
        assert_eq!(lt.cost(dm, g7).unwrap(), reloaded.cost(dm, g7).unwrap());
    }
}
