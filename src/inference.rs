#![allow(dead_code)]

use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, Once};

/// Process-wide lock around ONNX Runtime *session construction*
/// (`Session::builder()`, `commit_from_file`, `ort::init_from`).
///
/// ort's global init (`G_ORT_LIB` / `G_ORT_API`) is not safe against
/// concurrent first-use from multiple threads: when the runtime library
/// fails to load, the error-formatting path re-enters the half-initialized
/// globals while another thread is still initializing them, and both threads
/// park forever. The analysis worker, stem-analysis worker and separation
/// worker can all build sessions at once, so without this lock a missing
/// `libonnxruntime` (or a slow first init) wedges every transcription at
/// "Analyzing..." with `is_processing` stuck forever instead of failing
/// fast with a readable error. Construction is rare (once per job/model),
/// so serializing it costs nothing; `session.run` inference itself stays
/// concurrent (basic-pitch runs are additionally serialized by
/// `BASIC_PITCH_ENGINE`, as before).
pub(crate) static ORT_SESSION_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn ort_session_guard() -> MutexGuard<'static, ()> {
    ORT_SESSION_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Verify the ONNX Runtime shared library can actually be loaded, *before*
/// touching any ort API.
///
/// Background: with `load-dynamic`, ort resolves `G_ORT_LIB` lazily, and a
/// failed load formats its error via `Error::new` → `ort::api()` →
/// `G_ORT_API` init → `setup_api` → back into `G_ORT_LIB` init, which is
/// still in progress on the same thread. `std::sync::Once` parks forever on
/// re-entrant init, so the FIRST session construction on a machine without
/// a loadable runtime deadlocks single-threadedly instead of returning an
/// error — every transcription wedges at "Analyzing..." with no message.
/// Probing with the real OS loader first turns that into a fast, actionable
/// error. The probe mirrors ort's own search order (explicit
/// `ORT_DYLIB_PATH`, exe-adjacent, system search), so anything the probe
/// accepts, ort will load too.
pub(crate) fn ensure_onnxruntime_loadable() -> Result<()> {
    fn probe(path: &Path) -> bool {
        // SAFETY: immediately dropped; only tests OS loader resolvability.
        unsafe { libloading::Library::new(path).is_ok() }
    }

    let mut tried: Vec<PathBuf> = Vec::new();
    if let Ok(s) = std::env::var("ORT_DYLIB_PATH") {
        if !s.trim().is_empty() {
            tried.push(PathBuf::from(s.trim()));
        }
    }
    let lib_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "linux") {
        "libonnxruntime.so"
    } else {
        "libonnxruntime.dylib"
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            tried.push(parent.join(lib_name));
        }
    }
    tried.push(PathBuf::from(lib_name));

    if tried.iter().any(|p| probe(p)) {
        return Ok(());
    }

    let searched = tried
        .iter()
        .map(|p| format!("'{}'", p.display()))
        .collect::<Vec<_>>()
        .join(", ");
    Err(anyhow!(
        "ONNX Runtime library ({lib_name}) could not be loaded (searched {searched}). \
         Transcription and stem separation need it: place {lib_name} next to the \
         executable or install it system-wide."
    ))
}

/// Initialize the ONNX Runtime environment from a bundled
/// `onnxruntime` shared library.
///
/// With the `load-dynamic` feature, ort loads the runtime library at
/// runtime instead of linking it at build time. We search for it next to
/// the executable (portable bundle) and in the working directory (dev
/// layout), then call `ort::init_from()` to load it.
///
/// This must run before any `Session::builder()` call — callers hold
/// [`ORT_SESSION_LOCK`] across init + construction so concurrent first-use
/// from several worker threads cannot interleave. It is safe to call
/// multiple times; the `Once` makes repeats no-ops.
pub(crate) fn init_ort_environment() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let lib_name = if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "linux") {
            "libonnxruntime.so"
        } else {
            "libonnxruntime.dylib"
        };

        // Search for the library next to the executable, then in the
        // working directory.
        let mut lib_path: Option<PathBuf> = None;
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let p = parent.join(lib_name);
                if p.exists() {
                    lib_path = Some(p);
                }
            }
        }
        if lib_path.is_none() {
            let p = PathBuf::from(lib_name);
            if p.exists() {
                lib_path = Some(p);
            }
        }

        match &lib_path {
            Some(path) => {
                eprintln!("[ORT] Loading ONNX Runtime from {}", path.display());
                if let Err(e) = ort::init_from(path) {
                    eprintln!("[ORT] Failed to load ONNX Runtime: {e}");
                }
            }
            None => {
                eprintln!(
                    "[ORT] {lib_name} not found next to executable; \
                     ort will use the system default"
                );
                // Don't call init_from — ort will try to load from PATH.
            }
        }
    });
}

/// Configuration for Spotify Basic Pitch ONNX inference.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to ONNX model file.
    pub model_path: String,
    /// Model input size in mono samples.
    pub input_samples: usize,
    /// Number of MIDI notes (A0..C8).
    pub num_notes: usize,
    /// Number of frame steps produced by the model per window.
    pub output_frames: usize,
    /// Model operating sample rate.
    pub model_sample_rate: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: "models/basic-pitch.onnx".to_string(),
            input_samples: 43_844,
            num_notes: 88,
            output_frames: 172,
            model_sample_rate: 22_050,
        }
    }
}

/// Basic Pitch ONNX inference engine.
pub struct BasicPitchInference {
    config: InferenceConfig,
    session: Session,
    input_name: String,
}

impl BasicPitchInference {
    /// Create a new Basic Pitch inference engine.
    ///
    /// `config.model_path` may be an exact path or a bare filename; when it
    /// does not exist as given, it is resolved next to the running
    /// executable (`<exe>/models/<file>`, portable bundle/AppImage/Flatpak
    /// layout) before falling back to the working directory. This keeps the
    /// app working regardless of the launch CWD (e.g. desktop entries).
    pub fn new(config: InferenceConfig) -> Result<Self> {
        // Serialize with every other ORT session construction in the
        // process (see ORT_SESSION_LOCK) and make sure the runtime library
        // is initialized first — the transcription path previously skipped
        // init entirely, so exe-bundled libonnxruntime was never found here.
        let _ort_guard = ort_session_guard();
        init_ort_environment();
        // Fail fast when unloadable: entering ort without a loadable runtime
        // deadlocks inside its global init (see ensure_onnxruntime_loadable).
        ensure_onnxruntime_loadable()?;

        let given = Path::new(&config.model_path);
        let model_path: PathBuf = if given.exists() {
            given.to_path_buf()
        } else {
            let filename = given
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&config.model_path);
            crate::demucs::resolve_model_path(filename).ok_or_else(|| {
                anyhow!(
                    "Basic Pitch ONNX model '{}' not found. Searched <exe>/models/, \
                     <exe>/, ./models/ and ./ — place it next to the executable \
                     or run from the project directory.",
                    filename
                )
            })?
        };

        let session = Session::builder()?.commit_from_file(&model_path)?;
        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("ONNX model has no inputs"))?
            .name()
            .to_string();

        Ok(Self {
            config,
            session,
            input_name,
        })
    }

    /// Infer note probabilities for a single Basic Pitch window.
    ///
    /// Returns `(note_probs, onset_probs)` both shaped (output_frames, 88).
    /// `onset_probs` is `None` when the model export doesn't expose a
    /// separate onset head (in that case the legacy merged behavior is used).
    pub fn infer_audio_window(
        &mut self,
        audio_window: &[f32],
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>)> {
        let prepared = Self::prepare_audio_window(audio_window, self.config.input_samples);

        let input_tensor = Tensor::from_array((
            [1usize, self.config.input_samples, 1usize],
            prepared.into_boxed_slice(),
        ))?;

        let outputs = self
            .session
            .run(ort::inputs! { self.input_name.as_str() => input_tensor })?;

        if std::env::var_os("KEYSCRIBE_INFERENCE_DEBUG").is_some() {
            eprintln!(
                "[inference] output names: {:?}",
                outputs.keys().collect::<Vec<_>>()
            );
            for (name, output) in &outputs {
                if let Ok(arr) = output.try_extract_array::<f32>() {
                    let shape = arr.shape();
                    let mut sum = 0.0f64;
                    let mut mx = 0.0f32;
                    let mut cnt = 0usize;
                    let mut over005 = 0usize;
                    let mut over05 = 0usize;
                    for v in arr.iter() {
                        sum += *v as f64;
                        mx = mx.max(*v);
                        if *v > 0.05 {
                            over005 += 1;
                        }
                        if *v > 0.5 {
                            over05 += 1;
                        }
                        cnt += 1;
                    }
                    eprintln!(
                        "[inference]   {} shape={:?} mean={:.4} max={:.2} >0.05={}/{} >0.5={}/{}",
                        name,
                        shape,
                        if cnt > 0 { sum / cnt as f64 } else { 0.0 },
                        mx,
                        over005,
                        cnt,
                        over05,
                        cnt
                    );
                }
            }
        }

        let frames = self.config.output_frames;
        let notes = self.config.num_notes;

        // Basic Pitch exports two 88-note heads (note/onset) and one 264-bin
        // contour head. Try to separate them by output name first; when the
        // names are opaque, fall back to sparsity (the onset head fires
        // sharply and is sparse at high values, the note head sustains).
        let mut heads_88: Vec<(usize, Vec<f32>)> = Vec::new(); // (high-count, data)
        let mut named_onset: Option<Vec<f32>> = None;
        let mut named_note: Vec<Vec<f32>> = Vec::new();
        let mut named = false;
        for (name, output) in &outputs {
            let arr = output.try_extract_array::<f32>()?;
            let shape = arr.shape();
            if shape.len() == 3 && shape[0] == 1 && shape[1] == frames && shape[2] == notes {
                let mut flat = vec![0.0f32; frames * notes];
                for t in 0..frames {
                    for n in 0..notes {
                        flat[t * notes + n] = arr[[0, t, n]];
                    }
                }
                let high = flat.iter().filter(|&&v| v > 0.5).count();
                heads_88.push((high, flat.clone()));
                let lname = name.to_lowercase();
                if lname.contains("onset") {
                    named_onset = Some(flat);
                    named = true;
                } else if lname.contains("contour") || lname.contains("mel") {
                    continue;
                } else if lname.contains("note")
                    || lname.contains("prob")
                    || lname.contains("head")
                {
                    named_note.push(flat);
                    named = true;
                }
            }
        }

        let (note_heads, onset_head) = if named {
            (named_note, named_onset)
        } else if heads_88.len() >= 2 {
            // Sparsest 88-head is the onset head; the rest are note heads.
            heads_88.sort_by_key(|(high, _)| *high);
            let (_, onset_flat) = heads_88.remove(0);
            (
                heads_88.into_iter().map(|(_, f)| f).collect(),
                Some(onset_flat),
            )
        } else {
            (heads_88.into_iter().map(|(_, f)| f).collect(), None)
        };

        if note_heads.is_empty() {
            let output_names: Vec<String> = outputs.keys().map(|name| name.to_string()).collect();
            return Err(anyhow!(
                "Basic Pitch outputs did not include any (1, {}, {}) tensors. Outputs: {:?}",
                self.config.output_frames,
                self.config.num_notes,
                output_names
            ));
        }

        let mut note_probs = vec![vec![0.0f32; notes]; frames];
        for t in 0..frames {
            for n in 0..notes {
                let idx = t * notes + n;
                let mut v = 0.0f32;
                for head in &note_heads {
                    v = v.max(head[idx]);
                }
                note_probs[t][n] = v.clamp(0.0, 1.0);
            }
        }

        let onset_probs = onset_head.map(|head| {
            let mut out = vec![vec![0.0f32; notes]; frames];
            for t in 0..frames {
                for n in 0..notes {
                    out[t][n] = head[t * notes + n].clamp(0.0, 1.0);
                }
            }
            out
        });

        Ok((note_probs, onset_probs))
    }

    /// Resample with linear interpolation.
    pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if samples.is_empty() || src_rate == 0 || dst_rate == 0 {
            return Vec::new();
        }
        if src_rate == dst_rate {
            return samples.to_vec();
        }

        let ratio = dst_rate as f64 / src_rate as f64;
        let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
        let mut out = Vec::with_capacity(out_len);

        let downsample_factor = if ratio < 1.0 { (1.0 / ratio).ceil() as usize } else { 1 };

        for i in 0..out_len {
            let src_pos = (i as f64) / ratio;
            let idx0 = (src_pos.floor() as usize).min(samples.len().saturating_sub(1));
            let idx1 = (idx0 + 1).min(samples.len().saturating_sub(1));
            let frac = (src_pos - idx0 as f64) as f32;

            if ratio < 1.0 && downsample_factor > 1 {
                let start_idx = idx0.saturating_sub(downsample_factor / 2);
                let end_idx = (start_idx + downsample_factor).min(samples.len());
                let mut sum = 0.0;
                for j in start_idx..end_idx {
                    sum += samples[j];
                }
                out.push(sum / (end_idx - start_idx).max(1) as f32);
            } else {
                let s0 = samples[idx0];
                let s1 = samples[idx1];
                out.push(s0 + (s1 - s0) * frac);
            }
        }

        out
    }

    /// Pad or trim an audio window to the exact model input length.
    pub fn prepare_audio_window(samples: &[f32], target_len: usize) -> Vec<f32> {
        if samples.len() >= target_len {
            samples[..target_len].to_vec()
        } else {
            let mut out = Vec::with_capacity(target_len);
            out.extend_from_slice(samples);
            out.resize(target_len, 0.0);
            out
        }
    }

    /// Apply confidence threshold to predictions.
    pub fn threshold_predictions(note_probs: &[Vec<f32>], threshold: f32) -> Vec<Vec<bool>> {
        note_probs
            .iter()
            .map(|frame| frame.iter().map(|&p| p >= threshold).collect())
            .collect()
    }

    /// Get confidence-weighted note indices for each frame.
    pub fn get_active_notes(note_probs: &[Vec<f32>], threshold: f32) -> Vec<Vec<(usize, f32)>> {
        note_probs
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &prob)| {
                        if prob >= threshold {
                            Some((idx, prob))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// ONNX inference for the learned melody quantizer (a MIDI-to-score tokenizer).
///
/// Interface (defined by `tools/melody_corpus/export_quantizer_onnx.py`):
/// - input: `[1, seq, feature_dim]` f32 — per-note feature vectors
///   (see `quantize::learned_note_features`);
/// - output: `[1, seq, vocab]` f32 logits over the 12-token `LEARNED_TOKEN_TABLE`
///   vocabulary (or `[seq, vocab]`).
///
/// When `melody_quantizer.onnx` is absent, `quantize_aligned_notes_learned`
/// never constructs this type and falls back to the rule grid, so the learned
/// path is a strict improvement when it works and never a regression.
pub struct MelodyQuantizerInference {
    session: Session,
    input_name: String,
    feature_dim: usize,
}

impl MelodyQuantizerInference {
    pub fn new(model_path: &Path) -> Result<Self> {
        // Same ORT construction discipline as BasicPitchInference::new.
        let _ort_guard = ort_session_guard();
        init_ort_environment();
        ensure_onnxruntime_loadable()?;

        if !model_path.exists() {
            return Err(anyhow!(
                "melody quantizer ONNX model not found at {}",
                model_path.display()
            ));
        }

        let session = Session::builder()?.commit_from_file(model_path)?;
        let input = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("melody quantizer model has no inputs"))?;
        let input_name = input.name().to_string();
        let feature_dim = input
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.iter().last().copied())
            .filter(|&d| d > 0)
            .map(|d| d as usize)
            .unwrap_or(9);

        Ok(Self {
            session,
            input_name,
            feature_dim,
        })
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Run the model over a sequence of per-note feature vectors, returning one
    /// vocabulary index per note (argmax over the 12-token vocabulary).
    pub fn infer(&mut self, features: &[Vec<f32>]) -> Result<Vec<u32>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        let seq = features.len();
        let dim = self.feature_dim;
        let mut flat = Vec::with_capacity(seq * dim);
        for f in features {
            if f.len() != dim {
                return Err(anyhow!(
                    "melody quantizer feature vector length {} != model feature_dim {}",
                    f.len(),
                    dim
                ));
            }
            flat.extend_from_slice(f);
        }

        let input_tensor = Tensor::from_array(([1usize, seq, dim], flat.into_boxed_slice()))?;
        let outputs = self
            .session
            .run(ort::inputs! { self.input_name.as_str() => input_tensor })?;
        let (_, output) = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow!("melody quantizer produced no outputs"))?;
        let arr = output.try_extract_array::<f32>()?;
        let shape = arr.shape();
        let (t_seq, vocab) = if shape.len() == 3 && shape[0] == 1 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return Err(anyhow!(
                "unexpected melody quantizer output shape {:?} (expected [1, seq, vocab])",
                shape
            ));
        };

        let mut out = Vec::with_capacity(t_seq);
        for t in 0..t_seq {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for v in 0..vocab {
                let val = if shape.len() == 3 {
                    arr[[0, t, v]]
                } else {
                    arr[[t, v]]
                };
                if val > best_v {
                    best_v = val;
                    best = v;
                }
            }
            out.push(best as u32);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_audio_window() {
        let src = vec![1.0f32, 2.0, 3.0];
        let padded = BasicPitchInference::prepare_audio_window(&src, 6);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);

        let trimmed = BasicPitchInference::prepare_audio_window(&src, 2);
        assert_eq!(trimmed, vec![1.0, 2.0]);
    }

    #[test]
    fn test_threshold_predictions() {
        let probs = vec![vec![0.1, 0.7, 0.2], vec![0.5, 0.3, 0.8]];

        let predictions = BasicPitchInference::threshold_predictions(&probs, 0.5);
        assert_eq!(predictions[0], vec![false, true, false]);
        assert_eq!(predictions[1], vec![true, false, true]);
    }

    #[test]
    fn test_get_active_notes() {
        let probs = vec![vec![0.1, 0.7, 0.2], vec![0.5, 0.3, 0.8]];

        let active = BasicPitchInference::get_active_notes(&probs, 0.5);
        assert_eq!(active[0].len(), 1);
        assert_eq!(active[0][0].0, 1); // Note index
        assert!((active[0][0].1 - 0.7).abs() < 1e-5);
    }
}
