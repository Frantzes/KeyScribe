//! Demucs stem separation via ONNX Runtime.
//!
//! The `htdemucs_6s` model is loaded from
//! `models/htdemucs_6s.onnx`.
//! This module handles:
//!   - loading the ONNX session (resolved next to the executable, then cwd),
//!   - preloading the CUDA/cuDNN runtime DLLs bundled next to the executable,
//!   - resampling the input to the model sample rate (44100 Hz),
//!   - chunked inference with 25% overlap and triangle transition weights,
//!   - the single-shift stabilization trick (deterministic fixed offset),
//!   - splitting the (B, 6, 2, T) output into per-stem stereo audio.
//!
//! Output stem order: drums, bass, other, vocals, piano, guitar.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use ort::{session::Session, value::Tensor};
use ort::ep::{CUDA, CPUExecutionProvider};

use crate::audio_io::load_audio_file;
use crate::leadsheet::{SeparatedStem, StemType};

/// Demucs configuration. These values are baked into the exported ONNX graph
/// and must match the model's training config (htdemucs_6s defaults).
const SAMPLE_RATE: u32 = 44_100;
const SEGMENT_SECONDS: f64 = 7.8;
/// Segment length in samples = 7.8 * 44100 = 343980.
const SEGMENT_LENGTH: usize = (SEGMENT_SECONDS * SAMPLE_RATE as f64) as usize;
/// 25% overlap => stride is 75% of the segment length.
const OVERLAP: f64 = 0.25;
const STRIDE: usize = ((1.0 - OVERLAP) * SEGMENT_LENGTH as f64) as usize;
/// Stem names in the order the model emits them.
const STEM_NAMES: &[&str] = &["drums", "bass", "other", "vocals", "guitar", "piano"];

/// Process-global cache of GPU acceleration status. Avoids re-running the
/// expensive warmup verification on every separator creation.
const ACCEL_UNCHECKED: u8 = 0;
const ACCEL_GPU_VERIFIED: u8 = 1;
const ACCEL_CPU_FALLBACK: u8 = 2;
static ACCELERATION_STATUS: AtomicU8 = AtomicU8::new(ACCEL_UNCHECKED);

pub struct DemucsSeparator {
    session: Session,
    input_name: String,
    model_path: PathBuf,
    using_cpu: bool,
    /// Reusable deinterleaved input buffer (1, 2, SEGMENT_LENGTH) planar.
    /// Hoisted out of `infer_one` to avoid allocating ~2.7 MB per chunk.
    mono_planar: Vec<f32>,
}

impl DemucsSeparator {
    /// Load a Demucs ONNX model by name (e.g. "htdemucs_6s").
    /// The file `{name}.onnx` is resolved next to the running executable
    /// (portable bundle), then the working directory.
    ///
    /// GPU (CUDA) is tried first; falls back to CPU on failure.
    pub fn new(model_name: &str) -> Result<Self> {
        let filename = format!("{model_name}.onnx");
        let model_path = resolve_model_path(&filename)
            .ok_or_else(|| {
                anyhow!(
                    "Could not find {filename}. Place it next to the executable or in models/."
                )
            })?;
        Self::from_path(&model_path)
    }

    /// Load from an exact model path. Tries GPU (CUDA) first, then CPU.
    pub fn from_path(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!("Demucs ONNX model not found at {}", model_path.display()));
        }
        // Initialize the ONNX Runtime environment from our bundled DLL.
        init_ort_environment();

        // Preload CUDA/cuDNN DLLs so the CUDA EP can find them at runtime.
        preload_cuda_dylibs();

        // Check the cached acceleration status from a previous verification.
        let cached = ACCELERATION_STATUS.load(Ordering::Relaxed);

        let (session, using_cpu) = if cached == ACCEL_CPU_FALLBACK {
            eprintln!("[DEMUCS] Using CPU (GPU was previously verified as non-functional)");
            eprintln!("[DEMUCS] Loading model '{}' (CPU)... this may take 10-30s", model_path.file_name().and_then(|n|n.to_str()).unwrap_or("?"));
            (build_cpu_session(model_path)?, true)
        } else {
            eprintln!("[DEMUCS] Loading model '{}' (CUDA EP)... this may take 10-30s", model_path.file_name().and_then(|n|n.to_str()).unwrap_or("?"));
            match build_gpu_session(model_path) {
                Ok(s) => {
                    eprintln!("[DEMUCS] CUDA session created");
                    (s, false)
                }
                Err(e) => {
                    eprintln!("[DEMUCS] CUDA EP unavailable ({e}), falling back to CPU");
                    eprintln!("[DEMUCS] Loading model (CPU session)...");
                    ACCELERATION_STATUS.store(ACCEL_CPU_FALLBACK, Ordering::Relaxed);
                    (build_cpu_session(model_path)?, true)
                }
            }
        };
        eprintln!("[DEMUCS] Model loaded successfully");
        let input_name = session
            .inputs()
            .first()
            .context("Demucs ONNX model has no inputs")?
            .name()
            .to_string();
        let mut separator = Self {
            session,
            input_name,
            model_path: model_path.to_path_buf(),
            using_cpu,
            mono_planar: vec![0.0f32; 2 * SEGMENT_LENGTH],
        };

        // Verify GPU acceleration only once per process, and only in release
        // builds (debug inference is too slow for meaningful timing).
        if cached == ACCEL_UNCHECKED && !separator.using_cpu {
            separator.verify_acceleration();
        }

        Ok(separator)
    }

    /// Whether this separator is running on CPU (rather than GPU). Inference
    /// may start on CUDA and fall back to CPU at runtime if the GPU runs
    /// out of memory; check after any [`separate_file`] or [`separate_samples`]
    /// call for the final status.
    pub fn using_cpu(&self) -> bool {
        self.using_cpu
    }

    /// Switch the session to CPU-only. Used when CUDA fails at runtime
    /// (e.g. GPU out of memory during inference).
    fn fallback_to_cpu(&mut self) -> Result<()> {
        if self.using_cpu {
            return Ok(());
        }
        eprintln!("[DEMUCS] CUDA inference failed, falling back to CPU");
        self.session = build_cpu_session(&self.model_path)?;
        self.using_cpu = true;
        ACCELERATION_STATUS.store(ACCEL_CPU_FALLBACK, Ordering::Relaxed);
        Ok(())
    }

    /// Verify that GPU acceleration is actually working, not just registered.
    ///
    /// The CUDA EP can register successfully but silently delegate all
    /// operations to the CPU fallback provider if cuDNN or CUDA runtime
    /// DLLs are missing, or if the model has many ops the CUDA EP doesn't
    /// support (causing expensive GPU↔CPU data transfers per op).
    ///
    /// This method runs a warmup inference, measures timing, and checks
    /// `nvidia-smi` to detect this situation. If CUDA is registered but
    /// timing is CPU-like, it automatically switches to a pure CPU session
    /// to avoid the GPU↔CPU transfer overhead that makes mixed-mode slower
    /// than pure CPU.
    ///
    /// The result is cached in a process-global `AtomicU8` so it only runs
    /// once per process lifetime. In debug builds, the warmup is skipped
    /// (debug inference is too slow for meaningful timing).
    fn verify_acceleration(&mut self) {
        if self.using_cpu {
            ACCELERATION_STATUS.store(ACCEL_CPU_FALLBACK, Ordering::Relaxed);
            return;
        }

        // Print GPU info from nvidia-smi.
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = info.trim().split(", ").collect();
                if parts.len() >= 4 {
                    eprintln!(
                        "[DEMUCS] GPU: {} | VRAM: {} MB ({} MB used) | GPU util: {}%",
                        parts[0], parts[1], parts[2], parts[3]
                    );
                }
            }
        }

        // In debug builds, skip the warmup — unoptimized inference is too
        // slow for meaningful timing and would hang the UI for minutes.
        // Trust the CUDA EP registration; if it doesn't work, the per-chunk
        // timing logs in run_chunked will reveal it.
        if cfg!(debug_assertions) {
            eprintln!("[DEMUCS] GPU acceleration: CUDA EP registered (skipping warmup in debug build)");
            ACCELERATION_STATUS.store(ACCEL_GPU_VERIFIED, Ordering::Relaxed);
            return;
        }

        // Release build: run a warmup + timed inference to verify the GPU
        // is actually accelerating. The first run includes cuDNN algorithm
        // search overhead, so we measure the second run.
        eprintln!("[DEMUCS] Verifying GPU acceleration (warmup)...");
        let dummy = vec![0.0f32; 2 * SEGMENT_LENGTH];
        let _ = self.infer_one(&dummy); // warmup
        let t0 = Instant::now();
        let _ = self.infer_one(&dummy); // timed
        let elapsed = t0.elapsed().as_secs_f32();

        if elapsed > 4.0 {
            eprintln!(
                "[DEMUCS] WARNING: CUDA EP registered but inference is {:.1}s/chunk \
                 (CPU-like speed). The GPU is likely NOT accelerating — \
                 cuDNN/CUDA DLLs may be missing, or the model has ops the CUDA EP \
                 doesn't support (causing GPU↔CPU transfer overhead). \
                 Switching to pure CPU to avoid the overhead.",
                elapsed
            );
            ACCELERATION_STATUS.store(ACCEL_CPU_FALLBACK, Ordering::Relaxed);
            if let Err(e) = self.fallback_to_cpu() {
                eprintln!("[DEMUCS] Failed to switch to CPU: {e}");
            }
        } else {
            eprintln!(
                "[DEMUCS] GPU acceleration verified: {:.1}s/chunk (warmup excluded)",
                elapsed
            );
            ACCELERATION_STATUS.store(ACCEL_GPU_VERIFIED, Ordering::Relaxed);
        }
    }

    /// Separate an audio file into stems.
    ///
    /// `source_path` is loaded and resampled to 44.1 kHz stereo before being
    /// fed to the model. Returns one `SeparatedStem` per source.
    pub fn separate_file(
        &mut self,
        source_path: &Path,
        progress: Option<&dyn Fn(f32)>,
    ) -> Result<Vec<SeparatedStem>> {
        let audio = load_audio_file(source_path)?;
        let stereo = to_stereo_44100(&audio);
        self.separate_samples(&stereo, audio.channels.max(1) as usize, progress)
    }

    /// Separate interleaved stereo samples at 44.1 kHz into stems.
    ///
    /// Applies the same normalization as demucs/separate.py:
    ///   1. Remove DC offset (mean of channel-averaged signal)
    ///   2. Normalize by std
    ///   3. Run the model
    ///   4. Restore std scale and DC offset
    pub fn separate_samples(
        &mut self,
        stereo: &[f32],
        _orig_channels: usize,
        progress: Option<&dyn Fn(f32)>,
    ) -> Result<Vec<SeparatedStem>> {
        let n_frames = stereo.len() / 2;
        if n_frames == 0 {
            return Err(anyhow!("Cannot separate empty audio"));
        }

        // --- Normalization (matches demucs/separate.py) ---
        // ref = wav.mean(0)  → per-sample mean across channels
        // wav -= ref.mean()  → remove global DC offset
        // wav /= ref.std()   → normalize by overall level
        let ref_mean: f32 = {
            let mut sum = 0.0f64;
            for i in 0..n_frames {
                sum += (stereo[i * 2] as f64 + stereo[i * 2 + 1] as f64) * 0.5;
            }
            (sum / n_frames as f64) as f32
        };
        let ref_std: f32 = {
            let mut sum_sq = 0.0f64;
            for i in 0..n_frames {
                let m = ((stereo[i * 2] as f64 + stereo[i * 2 + 1] as f64) * 0.5) - ref_mean as f64;
                sum_sq += m * m;
            }
            (sum_sq / n_frames as f64).sqrt() as f32
        };
        let safe_std = ref_std.max(1e-8);

        let normalized: Vec<f32> = stereo
            .iter()
            .map(|&s| (s - ref_mean) / safe_std)
            .collect();

        // --- Run the model (no shift trick; shifts=0 for simplicity) ---
        let out = self.run_chunked(&normalized, progress)?;

        // --- Denormalize: restore std scale and DC offset ---
        let mut stems = Vec::with_capacity(STEM_NAMES.len());
        for (s, &name) in STEM_NAMES.iter().enumerate() {
            let base = s * 2 * n_frames;
            let interleaved: Vec<f32> = out[base..base + 2 * n_frames]
                .iter()
                .map(|&v| v * safe_std + ref_mean)
                .collect();
            let mut mono = Vec::with_capacity(n_frames);
            for frame in interleaved.chunks_exact(2) {
                mono.push((frame[0] + frame[1]) * 0.5);
            }
            stems.push(SeparatedStem {
                stem_type: StemType::from_label(name),
                samples_mono: Arc::new(mono),
                samples_interleaved: Arc::new(interleaved),
                channels: 2,
                sample_rate: SAMPLE_RATE,
                confidence: 0.9,
            });
        }

        Ok(stems)
    }

    /// Run chunked inference over interleaved stereo audio at 44.1 kHz.
    /// Returns interleaved stereo for all sources concatenated
    /// (len = 6 * 2 * n_frames).
    fn run_chunked(&mut self, mix: &[f32], progress: Option<&dyn Fn(f32)>) -> Result<Vec<f32>> {
        let n_frames = mix.len() / 2;
        // Output buffer: (6 sources, 2 channels, n_frames) interleaved per source.
        let mut out = vec![0.0f32; STEM_NAMES.len() * 2 * n_frames];
        let mut sum_weight = vec![0.0f32; n_frames];

        // Triangle transition weights over the segment, normalized to [0,1].
        let weight = triangle_weight(SEGMENT_LENGTH);

        let offsets: Vec<usize> = (0..n_frames).step_by(STRIDE).collect();
        let total = offsets.len().max(1);
        let mut first_chunk_time = 0.0f32;
        let mut total_infer_time = 0.0f32;

        // Allocate the chunk buffer once and reuse it across all chunks
        // (fill(0.0) each iteration) to avoid allocating ~2.7 MB per chunk.
        let mut chunk_buf = vec![0.0f32; 2 * SEGMENT_LENGTH];

        for (idx, &offset) in offsets.iter().enumerate() {
            // Build the chunk: segment_length samples starting at `offset`,
            // zero-padded on the right if it runs past the end.
            let end = (offset + SEGMENT_LENGTH).min(n_frames);
            let copy_len = end - offset;
            chunk_buf.fill(0.0);
            for i in 0..copy_len {
                chunk_buf[i * 2] = mix[(offset + i) * 2];
                chunk_buf[i * 2 + 1] = mix[(offset + i) * 2 + 1];
            }

            let t0 = Instant::now();
            let chunk_out = match self.infer_one(&chunk_buf) {
                Ok(o) => o,
                Err(e) if !self.using_cpu => {
                    // CUDA failed (likely GPU OOM) — surface the real error,
                    // then fall back to CPU and retry this chunk.
                    eprintln!("[DEMUCS] CUDA run failed (falling back to CPU): {e}");
                    self.fallback_to_cpu()?;
                    self.infer_one(&chunk_buf)?
                }
                Err(e) => return Err(e),
            };
            let chunk_time = t0.elapsed().as_secs_f32();
            if idx == 0 {
                first_chunk_time = chunk_time;
            }
            total_infer_time += chunk_time;

            // chunk_out layout: (6, 2, seg_len) interleaved per source.
            let chunk_frames = chunk_out.len() / (STEM_NAMES.len() * 2);
            for s in 0..STEM_NAMES.len() {
                for i in 0..chunk_frames {
                    let src = s * 2 * chunk_frames + i * 2;
                    let dst = s * 2 * n_frames + (offset + i) * 2;
                    if dst + 1 < out.len() && src + 1 < chunk_out.len() {
                        let w = weight[i.min(weight.len() - 1)];
                        out[dst] += w * chunk_out[src];
                        out[dst + 1] += w * chunk_out[src + 1];
                    }
                }
            }
            for i in 0..chunk_frames {
                let pos = offset + i;
                if pos < sum_weight.len() {
                    sum_weight[pos] += weight[i.min(weight.len() - 1)];
                }
            }

            if let Some(cb) = progress {
                cb(0.1 + 0.9 * (idx + 1) as f32 / total as f32);
            }
        }

        // Log per-chunk timing so the user can diagnose performance.
        let avg_chunk = if total > 1 {
            total_infer_time / (total - 1) as f32 // exclude first (warmup) chunk
        } else {
            total_infer_time
        };
        eprintln!(
            "[DEMUCS] Processed {} chunks in {:.1}s (first: {:.1}s, avg: {:.1}s/chunk, {})",
            total,
            total_infer_time,
            first_chunk_time,
            avg_chunk,
            if self.using_cpu { "CPU" } else { "CUDA" }
        );

        // Normalize by overlap weights.
        // Precompute inverse weights once (identical across all stems) and do a
        // single fused pass over the output instead of 6 separate division passes.
        let mut inv_weight = vec![0.0f32; n_frames];
        for i in 0..n_frames {
            inv_weight[i] = if sum_weight[i] > 1e-8 { 1.0 / sum_weight[i] } else { 0.0 };
        }
        let n_stems = STEM_NAMES.len();
        for s in 0..n_stems {
            let base = s * 2 * n_frames;
            for i in 0..n_frames {
                let w = inv_weight[i];
                out[base + i * 2] *= w;
                out[base + i * 2 + 1] *= w;
            }
        }

        Ok(out)
    }

    /// Run a single chunk through the ONNX model.
    /// Input: interleaved stereo (2 * SEGMENT_LENGTH samples).
    /// Output: (6, 2, SEGMENT_LENGTH) interleaved per source.
    fn infer_one(&mut self, stereo: &[f32]) -> Result<Vec<f32>> {
        // Deinterleave into the reusable planar buffer (1, 2, SEGMENT_LENGTH).
        // This avoids allocating ~2.7 MB on every chunk.
        let n = SEGMENT_LENGTH.min(stereo.len() / 2);
        for i in 0..n {
            self.mono_planar[i] = stereo[i * 2];
            self.mono_planar[SEGMENT_LENGTH + i] = stereo[i * 2 + 1];
        }
        // Zero-fill the remainder if the chunk was shorter than SEGMENT_LENGTH.
        for i in n..SEGMENT_LENGTH {
            self.mono_planar[i] = 0.0;
            self.mono_planar[SEGMENT_LENGTH + i] = 0.0;
        }
        // `Tensor::from_array` takes ownership of the data; clone into a boxed
        // slice so we keep `mono_planar` for reuse on the next chunk.
        let input_data = self.mono_planar.clone().into_boxed_slice();
        let input = Tensor::from_array((
            [1usize, 2usize, SEGMENT_LENGTH],
            input_data,
        ))?;
        let outputs = self.session.run(ort::inputs! { self.input_name.as_str() => input })?;

        // Output "sources": (1, 6, 2, SEGMENT_LENGTH).
        let out = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .context("Demucs ONNX model produced no outputs")?;
        let arr = out.try_extract_array::<f32>()?;
        let shape = arr.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != STEM_NAMES.len() || shape[2] != 2 {
            return Err(anyhow!(
                "Unexpected Demucs output shape {:?}, expected (1, {}, 2, T)",
                shape,
                STEM_NAMES.len()
            ));
        }
        let t = shape[3];
        // The output array is contiguous and row-major [1, 6, 2, t], so the
        // underlying slice is laid out as [s0c0.. s0c1.. s1c0.. ...]. We need
        // it interleaved per source: [s0c0 s0c1 | s1c0 s1c1 | ...]. Use a
        // single slice read + strided copy instead of per-element ndarray indexing.
        let src_slice = arr
            .as_slice()
            .ok_or_else(|| anyhow!("Demucs output tensor is not contiguous"))?;
        let n_stems = STEM_NAMES.len();
        let mut flat = vec![0.0f32; n_stems * 2 * t];
        for s in 0..n_stems {
            let ch0 = &src_slice[(s * 2) * t..(s * 2) * t + t];
            let ch1 = &src_slice[(s * 2 + 1) * t..(s * 2 + 1) * t + t];
            let dst_base = s * 2 * t;
            for i in 0..t {
                flat[dst_base + i * 2] = ch0[i];
                flat[dst_base + i * 2 + 1] = ch1[i];
            }
        }
        Ok(flat)
    }
}

/// Triangle window (rising then falling) of length `n`, peak 1.0 at the center.
/// Matches demucs/apply.py: arange(1, n//2+1) ++ arange(n - n//2, 0, -1).
fn triangle_weight(n: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(n);
    let half = n / 2;
    for i in 1..=half {
        w.push(i as f32);
    }
    for i in 0..(n - half) {
        w.push((n - half - i) as f32);
    }
    let max = w.iter().copied().fold(0.0f32, f32::max).max(1e-8);
    for v in &mut w {
        *v /= max;
    }
    w
}

/// Convert loaded audio to interleaved stereo at 44.1 kHz (Demucs sample rate).
fn to_stereo_44100(audio: &crate::audio_io::AudioData) -> Vec<f32> {
    let sr = audio.sample_rate;
    let interleaved = &audio.samples_interleaved;
    let channels = audio.channels.max(1) as usize;

    // Fast path: already stereo at the target sample rate — avoid cloning the
    // entire buffer by returning a cheap copy only when resampling is needed.
    if channels == 2 && sr == SAMPLE_RATE {
        return interleaved.to_vec();
    }

    // First, ensure stereo interleaved.
    let stereo: Vec<f32> = if channels == 2 {
        interleaved.to_vec()
    } else if channels == 1 {
        let mut out = Vec::with_capacity(interleaved.len() * 2);
        for &s in interleaved.iter() {
            out.push(s);
            out.push(s);
        }
        out
    } else {
        // fold multichannel to stereo
        let n = interleaved.len() / channels;
        let mut out = Vec::with_capacity(n * 2);
        for f in 0..n {
            let frame = &interleaved[f * channels..f * channels + channels];
            let mut l = 0.0;
            let mut r = 0.0;
            for (i, &v) in frame.iter().enumerate() {
                if i % 2 == 0 { l += v; } else { r += v; }
            }
            let lc = (channels + 1) / 2;
            let rc = channels / 2;
            out.push(l / lc.max(1) as f32);
            out.push(r / rc.max(1) as f32);
        }
        out
    };

    // Resample to 44100 if needed (linear interpolation; sufficient for
    // separation quality when source rates are close, e.g. 48000 -> 44100).
    if sr == SAMPLE_RATE {
        stereo
    } else {
        let n = stereo.len() / 2;
        let ratio = SAMPLE_RATE as f64 / sr as f64;
        let out_n = (n as f64 * ratio).round() as usize;
        let mut out = Vec::with_capacity(out_n * 2);
        for i in 0..out_n {
            let pos = i as f64 / ratio;
            let idx0 = pos.floor() as usize;
            let idx1 = (idx0 + 1).min(n.saturating_sub(1));
            let frac = (pos - idx0 as f64) as f32;
            let l0 = stereo[idx0 * 2];
            let l1 = stereo[idx1 * 2];
            let r0 = stereo[idx0 * 2 + 1];
            let r1 = stereo[idx1 * 2 + 1];
            out.push(l0 + (l1 - l0) * frac);
            out.push(r0 + (r1 - r0) * frac);
        }
        out
    }
}

/// Build an ONNX session using the CUDA execution provider (NVIDIA GPUs)
/// when available, falling back to multi-threaded CPU. CUDA + cuDNN matches
/// the acceleration the Python Demucs uses; without it, inference is
/// ~10-20x slower on CPU.
///
/// `cudnn_conv_use_max_workspace` is disabled to bound the cuDNN workspace
/// memory, which helps the model fit in GPUs with limited VRAM (e.g. 8 GB).
fn build_gpu_session(model_path: &Path) -> Result<Session> {
    let cuda = CUDA::default()
        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
        .with_conv_max_workspace(false)
        .build();
    let mut builder = Session::builder()?
        // GraphOptimizationLevel::All is the ort default, but set it explicitly.
        // `with_optimization_level` returns a recoverable error type; use
        // `recover()` to ignore it (the default is already All).
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
        .unwrap_or_else(|e| e.recover())
        .with_execution_providers([
            cuda,
            CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow!("Failed to set execution providers: {e}"))?;
    Ok(builder.commit_from_file(model_path)?)
}

/// Minimum available CPU threads to leave for system responsiveness.
const MIN_FREE_CORES: usize = 1;

fn build_cpu_session(model_path: &Path) -> Result<Session> {
    // Use all available cores minus one for system responsiveness.
    // The previous code capped this at 4, leaving most cores idle on modern
    // CPUs (e.g. only 4 of 12 logical processors were used).
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(MIN_FREE_CORES))
        .map(|n| n.max(1))
        .unwrap_or(2);
    let mut builder = Session::builder()?
        .with_intra_threads(max_threads)
        .unwrap_or_else(|e| e.recover())
        // Memory pattern optimization is safe here because the segment size is
        // fixed (SEGMENT_LENGTH constant), so the CPU EP can reuse activation
        // buffers across chunks. Recoverable error — ignore if unsupported.
        .with_memory_pattern(true)
        .unwrap_or_else(|e| e.recover())
        .with_execution_providers([
            CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow!("Failed to set CPU execution provider: {e}"))?;
    Ok(builder.commit_from_file(model_path)?)
}

/// Initialize the ONNX Runtime environment from our bundled `onnxruntime.dll`.
///
/// With the `load-dynamic` feature, ort loads the runtime DLL at runtime
/// instead of linking it at build time. We search for `onnxruntime.dll`
/// next to the executable (portable bundle) and in the working directory
/// (dev layout), then call `ort::init_from()` to load it.
///
/// This must be called before any `Session::builder()` call. It is safe to
/// call multiple times — the environment is process-global and can only be
/// committed once; subsequent calls are no-ops.
fn init_ort_environment() {
    use std::sync::Once;

    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let dll_name = if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "linux") {
            "libonnxruntime.so"
        } else {
            "libonnxruntime.dylib"
        };

        // Search for onnxruntime.dll next to the executable, then in the
        // working directory.
        let mut dll_path: Option<PathBuf> = None;
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let p = parent.join(dll_name);
                if p.exists() {
                    dll_path = Some(p);
                }
            }
        }
        if dll_path.is_none() {
            let p = PathBuf::from(dll_name);
            if p.exists() {
                dll_path = Some(p);
            }
        }

        match &dll_path {
            Some(path) => {
                eprintln!("[DEMUCS] Loading ONNX Runtime from {}", path.display());
                if let Err(e) = ort::init_from(path) {
                    eprintln!("[DEMUCS] Failed to load onnxruntime.dll: {e}");
                }
            }
            None => {
                eprintln!(
                    "[DEMUCS] onnxruntime.dll not found next to executable; \
                     ort will use the system default"
                );
                // Don't call init_from — ort will try to load from PATH.
            }
        }
    });
}

/// Preload the CUDA 12 runtime and cuDNN 9 DLLs so the CUDA execution
/// provider can find them at runtime.
///
/// Search order:
/// 1. A `cuda` subdirectory next to the executable (portable bundle layout).
/// 2. The executable directory itself (portable bundle with DLLs flat).
/// 3. `vendor/cudnn` in the working directory (dev layout).
/// 4. The system CUDA toolkit install (e.g. `C:\Program Files\NVIDIA GPU
///    Computing Toolkit\CUDA\v12.x\bin`).
///
/// This is a no-op (returns `Ok`) if the DLLs are not present in any of
/// these locations — in that case the CUDA EP will fall back to system PATH
/// or fail gracefully to CPU.
fn preload_cuda_dylibs() {
    let mut cuda_dirs: Vec<PathBuf> = Vec::new();
    let mut cudnn_dirs: Vec<PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            cuda_dirs.push(parent.join("cuda"));
            cuda_dirs.push(parent.to_path_buf());
            cudnn_dirs.push(parent.join("cuda"));
            cudnn_dirs.push(parent.to_path_buf());
        }
    }

    // Dev layout: vendor/cudnn for cuDNN, CUDA toolkit for cudart
    cudnn_dirs.push(PathBuf::from("vendor/cudnn"));

    // System CUDA toolkit installs (Windows)
    if cfg!(target_os = "windows") {
        for ver in ["12.6", "12.5", "12.4", "12.3", "12.2", "12.1"] {
            let p = PathBuf::from(format!(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{ver}\bin"
            ));
            if p.exists() {
                cuda_dirs.push(p);
                break;
            }
        }
    }

    // Preload CUDA runtime DLLs
    let mut cuda_loaded = false;
    for dir in &cuda_dirs {
        if !dir.exists() {
            continue;
        }
        match ort::ep::cuda::preload_dylibs(Some(dir), None) {
            Ok(()) => {
                eprintln!("[DEMUCS] CUDA runtime DLLs loaded from {}", dir.display());
                cuda_loaded = true;
                break;
            }
            Err(e) => {
                eprintln!("[DEMUCS] CUDA runtime preload failed from {}: {e}", dir.display());
            }
        }
    }
    if !cuda_loaded {
        eprintln!("[DEMUCS] CUDA runtime DLLs not found in any search path");
    }

    // Preload cuDNN DLLs
    let mut cudnn_loaded = false;
    for dir in &cudnn_dirs {
        if !dir.exists() {
            continue;
        }
        match ort::ep::cuda::preload_dylibs(None, Some(dir)) {
            Ok(()) => {
                eprintln!("[DEMUCS] cuDNN DLLs loaded from {}", dir.display());
                cudnn_loaded = true;
                break;
            }
            Err(e) => {
                eprintln!("[DEMUCS] cuDNN preload failed from {}: {e}", dir.display());
            }
        }
    }
    if !cudnn_loaded {
        eprintln!("[DEMUCS] cuDNN DLLs not found in any search path");
    }
}

/// Resolve a model filename next to the executable, then the working directory.
fn resolve_model_path(filename: &str) -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("models").join(filename);
            if p.exists() {
                return Some(p);
            }
            // Some dev layouts keep models at the repo root next to the target dir.
            let p2 = parent.join(filename);
            if p2.exists() {
                return Some(p2);
            }
        }
    }
    let cwd = PathBuf::from("models").join(filename);
    if cwd.exists() {
        return Some(cwd);
    }
    let cwd2 = PathBuf::from(filename);
    if cwd2.exists() {
        return Some(cwd2);
    }
    None
}

#[cfg(test)]
#[path = "demucs_tests.rs"]
mod integration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_is_symmetric_and_peaked() {
        let w = triangle_weight(10);
        assert_eq!(w.len(), 10);
        assert!((w[5] - 1.0).abs() < 1e-5 || (w[4] - 1.0).abs() < 1e-5);
        assert!((w[0] - w[9]).abs() < 1e-5);
    }

    #[test]
    fn triangle_length_343980() {
        let w = triangle_weight(SEGMENT_LENGTH);
        assert_eq!(w.len(), SEGMENT_LENGTH);
        assert!(w.iter().all(|v| *v >= 0.0 && *v <= 1.0 + 1e-5));
    }
}
