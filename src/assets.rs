//! On-demand downloads for optional runtime assets.
//!
//! The portable bundles ship only the ~60 MB core (binary, small models,
//! ONNX Runtime). The large optional pieces — the Demucs stem-separation
//! model, FFmpeg (audio fallback decode + video), and the Windows GPU pack
//! (CUDA 12 + cuDNN 9 + ONNX Runtime CUDA provider) — are fetched the first
//! time the feature that needs them is actually used:
//!
//! * `htdemucs_6s.onnx` — only when the user picks a *local* Demucs model for
//!   separation. The default separation path is MVSep cloud, which never
//!   touches this model.
//! * FFmpeg — only when Symphonia cannot decode a file (mkv/webm/wma/...) or
//!   a video file is opened.
//! * GPU pack — Windows only, only on first local Demucs use, and only when
//!   `nvidia-smi` reports an NVIDIA GPU. Failure is non-fatal: Demucs falls
//!   back to CPU.
//!
//! Every download is pinned: HTTPS only, exact expected byte size and
//! SHA-256 checked while streaming, written to a `.part` file and atomically
//! renamed only after verification. A mismatch aborts and deletes the file.
//! The pinned URLs and hashes are listed in README.md ("Runtime downloads").

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Maximum size accepted for a single download (sanity cap, 1.5 GiB).
const MAX_DOWNLOAD_BYTES: u64 = 1536 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Pinned assets
// ---------------------------------------------------------------------------

/// Demucs v4 `htdemucs_6s` ONNX model (source: Demucs, MIT).
const HTDEMUCS_URL: &str = "https://github.com/Frantzes/KeyScribe/releases/download/assets-v1/htdemucs_6s.onnx";
const HTDEMUCS_SHA256: &str =
    "b268327119b709141d43ba4103e7bb5f54bbf33311d6f1af5054ef663bf83322";
const HTDEMUCS_SIZE: u64 = 284_749_531;

/// Mirrored BtbN FFmpeg-Builds `ffmpeg-n9.0-latest-win64-gpl-9.0` (GPLv3),
/// repacked to `bin/ffmpeg.exe` + `bin/ffprobe.exe`. Upstream zip verified
/// against BtbN's own `checksums.sha256` before mirroring.
const FFMPEG_WIN_URL: &str = "https://github.com/Frantzes/KeyScribe/releases/download/assets-v1/keyscribe-ffmpeg-win64-v1.zip";
const FFMPEG_WIN_SHA256: &str =
    "502c913aca7e637314088b5d6c5d92359a65ac8ee5e8b91daa9b2fe4c1edc520";
const FFMPEG_WIN_SIZE: u64 = 126_991_730;

/// Microsoft ONNX Runtime GPU 1.24.4 (MIT), Windows x64 wheel from PyPI.
/// Wheel path pinned for cp313; the native DLLs are the same across the
/// cp311/cp312/cp313 builds, but the file itself is pinned to one hash.
const ORT_WHEEL_URL: &str = "https://files.pythonhosted.org/packages/fa/bc/35f3a37226d7a28c84b8b456f52237ccd39eb7111114bcf9ac340178e1ec/onnxruntime_gpu-1.24.4-cp313-cp313-win_amd64.whl";
const ORT_WHEEL_SHA256: &str =
    "6be8bf2048777c517fca33eb61e114969fa326619feaa789d8c75f24337ea762";
const ORT_WHEEL_SIZE: u64 = 207_198_775;

/// NVIDIA cuDNN 9.3.0.75 for CUDA 12, Windows x86_64 official redist archive.
/// NVIDIA does not publish a checksum for this archive; the hash below was
/// computed once from the official HTTPS URL and is pinned here.
const CUDNN_WIN_URL: &str = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.3.0.75_cuda12-archive.zip";
const CUDNN_WIN_SHA256: &str =
    "864a85dc67c7f92b9a8639f323acb4af63ad65de2ca82dccdf2c0b6a701c27c0";
const CUDNN_WIN_SIZE: u64 = 566_118_754;

/// CUDA 12.4 runtime components (NVIDIA official redist). Hashes are the ones
/// NVIDIA publishes in `redistrib_12.4.1.json`.
const CUDA_REDIST_BASE: &str = "https://developer.download.nvidia.com/compute/cuda/redist";
const CUDA_COMPONENTS: &[(&str, &str, u64)] = &[
    (
        "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip",
        "6a1c32e68ee1a95ca17334691ff9ad1ffe7f352c24a083d55e4c96b8063b2bcb",
        2_474_721,
    ),
    (
        "cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.4.127-archive.zip",
        "f140545e06d0d10780c1382a577db2e2c242db7a2d94970f0e6026b2d01aeb1b",
        101_865_624,
    ),
    (
        "libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip",
        "698140f12da055a3709eee2e022fcfe7bc8edf31f30115e3f7a5c877a9491de5",
        391_538_487,
    ),
    (
        "libcufft/windows-x86_64/libcufft-windows-x86_64-11.2.1.3-archive.zip",
        "df1594afd3de4e23779511eb8eccc1f77faacd9fa64e6154828b9bdd68d9a785",
        209_047_054,
    ),
    (
        "libcurand/windows-x86_64/libcurand-windows-x86_64-10.3.5.147-archive.zip",
        "24c3b0fb7063e49ccc0ac1bff387c5f4fd9617b72aab3fa3b642f08607770ad3",
        55_087_990,
    ),
];

// ---------------------------------------------------------------------------
// Global download status (polled by the UI)
// ---------------------------------------------------------------------------

static ACTIVE_DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
static DOWNLOAD_PERMILLE: AtomicU32 = AtomicU32::new(0);
static DOWNLOAD_LABEL: Mutex<String> = Mutex::new(String::new());

#[cfg(feature = "native-ui")]
pub(crate) fn is_downloading() -> bool {
    ACTIVE_DOWNLOADS.load(Ordering::Acquire) > 0
}

/// Current download label and progress (0.0..=1.0), or `None` when idle.
#[cfg(feature = "native-ui")]
pub(crate) fn download_status() -> Option<(String, f32)> {
    if !is_downloading() {
        return None;
    }
    let label = DOWNLOAD_LABEL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let progress = DOWNLOAD_PERMILLE.load(Ordering::Relaxed) as f32 / 1000.0;
    Some((label, progress.clamp(0.0, 1.0)))
}

struct DownloadSession;

impl DownloadSession {
    fn new(label: &str) -> Self {
        ACTIVE_DOWNLOADS.fetch_add(1, Ordering::AcqRel);
        set_status(label, 0.0);
        Self
    }

    fn set(&self, label: &str, progress: f32) {
        set_status(label, progress);
    }
}

impl Drop for DownloadSession {
    fn drop(&mut self) {
        ACTIVE_DOWNLOADS.fetch_sub(1, Ordering::AcqRel);
    }
}

fn set_status(label: &str, progress: f32) {
    *DOWNLOAD_LABEL.lock().unwrap_or_else(|e| e.into_inner()) = label.to_string();
    DOWNLOAD_PERMILLE.store((progress.clamp(0.0, 1.0) * 1000.0) as u32, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Writable locations
// ---------------------------------------------------------------------------

fn portable_base_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn is_dir_writable(dir: &Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".keyscribe-write-test");
    match fs::write(&probe, b"w") {
        Ok(()) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

fn user_data_dir() -> PathBuf {
    directories::ProjectDirs::from("com", "Frantzes", "KeyScribe")
        .map(|p| p.data_local_dir().to_path_buf())
        .unwrap_or_else(portable_base_dir)
}

/// Read/write directory for downloaded models. Prefers `<exe>/models` (so a
/// portable bundle behaves exactly like a bundled install); falls back to the
/// user data dir when the install location is read-only (Program Files,
/// AppImage, Flatpak).
pub(crate) fn models_dir() -> PathBuf {
    let exe_dir = portable_base_dir();
    if is_dir_writable(&exe_dir) {
        exe_dir.join("models")
    } else {
        user_data_dir().join("models")
    }
}

/// Directory for downloaded executables (ffmpeg/ffprobe).
pub(crate) fn bin_dir() -> PathBuf {
    let exe_dir = portable_base_dir();
    if is_dir_writable(&exe_dir) {
        exe_dir
    } else {
        user_data_dir().join("bin")
    }
}

/// Directory for the downloaded Windows GPU pack (CUDA/cuDNN/providers).
pub(crate) fn gpu_dir() -> PathBuf {
    let exe_dir = portable_base_dir();
    if is_dir_writable(&exe_dir) {
        exe_dir.join("cuda")
    } else {
        user_data_dir().join("cuda")
    }
}

fn temp_download_dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("keyscribe-downloads");
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    Ok(dir)
}

// ---------------------------------------------------------------------------
// Download + verify
// ---------------------------------------------------------------------------

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Stream `url` into `dest`, verifying exact size and SHA-256 before rename.
/// Never leaves a partial file at `dest`.
fn download_verified(
    url: &str,
    sha256_hex: &str,
    expected_size: u64,
    dest: &Path,
    label: &str,
) -> Result<()> {
    if expected_size > MAX_DOWNLOAD_BYTES {
        bail!("refusing to download {label}: size exceeds the safety cap");
    }

    // The blocking client has no separate read timeout; a generous total cap
    // keeps a stalled connection from hanging forever while still allowing
    // slow links to finish multi-hundred-MB components.
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(2 * 60 * 60))
        .build()
        .context("failed to build HTTP client")?;

    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("failed to start download of {label}"))?
        .error_for_status()
        .with_context(|| format!("download of {label} failed"))?;

    if let Some(len) = response.content_length() {
        if len != expected_size {
            bail!(
                "refusing {label}: server reports {len} bytes but {expected_size} were expected"
            );
        }
    }

    let file_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "download.bin".to_string());
    let part = dest.with_file_name(format!("{file_name}.part"));

    let mut hasher = Sha256::new();
    let mut file = File::create(&part)
        .with_context(|| format!("failed to create {}", part.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut total: u64 = 0;

    let result = (|| -> Result<()> {
        loop {
            let n = response
                .read(&mut buf)
                .with_context(|| format!("network error while downloading {label}"))?;
            if n == 0 {
                break;
            }
            total += n as u64;
            if total > expected_size {
                bail!(
                    "refusing {label}: download exceeded the expected {expected_size} bytes"
                );
            }
            file.write_all(&buf[..n])
                .with_context(|| format!("failed to write {}", part.display()))?;
            hasher.update(&buf[..n]);
            set_status(label, total as f32 / expected_size.max(1) as f32);
        }
        file.flush()?;
        Ok(())
    })();

    if let Err(err) = result {
        drop(file);
        let _ = fs::remove_file(&part);
        return Err(err);
    }
    drop(file);

    if total != expected_size {
        let _ = fs::remove_file(&part);
        bail!("download of {label} was truncated ({total} of {expected_size} bytes)");
    }

    let digest = to_hex(&hasher.finalize());
    if !digest.eq_ignore_ascii_case(sha256_hex) {
        let _ = fs::remove_file(&part);
        bail!(
            "checksum mismatch for {label}: expected {sha256_hex}, got {digest}. \
             The download was discarded."
        );
    }

    fs::rename(&part, dest)
        .with_context(|| format!("failed to move download into {}", dest.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// ZIP extraction
// ---------------------------------------------------------------------------

fn extract_selected<F: Fn(&str) -> bool>(
    zip_path: &Path,
    dest_dir: &Path,
    select: F,
    label: &str,
) -> Result<usize> {
    let file = File::open(zip_path)
        .with_context(|| format!("failed to open {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("invalid archive {}", zip_path.display()))?;
    fs::create_dir_all(dest_dir)?;

    let mut extracted = 0usize;
    for i in 0..archive.len() {
        let mut entry = match archive.by_index(i) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("[ASSETS] skipping unreadable entry in {label}: {err}");
                continue;
            }
        };
        if entry.is_dir() {
            continue;
        }
        // Only ever write the base name, never an archive-controlled path.
        let base = Path::new(entry.name())
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.to_string())
            .unwrap_or_default();
        if base.is_empty() || !select(&base) {
            continue;
        }
        let out_path = dest_dir.join(&base);
        let mut out = File::create(&out_path)
            .with_context(|| format!("failed to create {}", out_path.display()))?;
        std::io::copy(&mut entry, &mut out)
            .with_context(|| format!("failed to extract {base} from {label}"))?;
        extracted += 1;
    }
    Ok(extracted)
}

fn extract_named(zip_path: &Path, dest_dir: &Path, names: &[&str], label: &str) -> Result<usize> {
    extract_selected(
        zip_path,
        dest_dir,
        |base| names.iter().any(|n| n.eq_ignore_ascii_case(base)),
        label,
    )
}

fn extract_dlls(zip_path: &Path, dest_dir: &Path, label: &str) -> Result<usize> {
    extract_selected(
        zip_path,
        dest_dir,
        |base| base.to_ascii_lowercase().ends_with(".dll"),
        label,
    )
}

// ---------------------------------------------------------------------------
// Demucs model
// ---------------------------------------------------------------------------

/// Ensure `htdemucs_6s.onnx` exists, downloading it if needed. Returns its path.
pub(crate) fn ensure_htdemucs() -> Result<PathBuf> {
    let dir = models_dir();
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    let dest = dir.join("htdemucs_6s.onnx");
    if dest.exists() {
        return Ok(dest);
    }

    let session = DownloadSession::new("Demucs stem-separation model");
    session.set("Downloading Demucs stem-separation model (stem separation)", 0.0);
    download_verified(
        HTDEMUCS_URL,
        HTDEMUCS_SHA256,
        HTDEMUCS_SIZE,
        &dest,
        "Demucs stem-separation model",
    )?;
    Ok(dest)
}

/// Prepare everything a local Demucs separation needs: the model, and (on
/// Windows with an NVIDIA GPU) the GPU acceleration pack. GPU pack failures
/// are logged and ignored — Demucs then runs on CPU.
pub(crate) fn ensure_local_demucs_assets(model_name: &str) -> Result<()> {
    let filename = format!("{model_name}.onnx");
    if crate::demucs::resolve_model_path(&filename).is_none() {
        if model_name == "htdemucs_6s" {
            ensure_htdemucs()?;
        } else {
            bail!(
                "Could not find {filename}. Only the built-in htdemucs_6s model can be \
                 downloaded automatically; place {filename} next to the executable or in models/."
            );
        }
    }

    #[cfg(target_os = "windows")]
    {
        if has_nvidia_gpu() {
            match ensure_windows_gpu_pack() {
                Ok(dir) => eprintln!("[ASSETS] GPU acceleration pack ready at {}", dir.display()),
                Err(err) => {
                    eprintln!("[ASSETS] GPU acceleration unavailable ({err}); running Demucs on CPU")
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Windows GPU pack
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
pub(crate) fn has_nvidia_gpu() -> bool {
    use std::os::windows::process::CommandExt;
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .creation_flags(0x08000000)
        .output();
    matches!(output, Ok(out) if out.status.success() && !out.stdout.is_empty())
}

#[cfg(target_os = "windows")]
const GPU_PACK_REQUIRED_DLLS: &[&str] = &[
    "onnxruntime_providers_cuda.dll",
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "curand64_10.dll",
    "nvrtc64_120_0.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
];

#[cfg(target_os = "windows")]
fn gpu_pack_complete(dir: &Path) -> bool {
    GPU_PACK_REQUIRED_DLLS.iter().all(|dll| dir.join(dll).exists())
}

/// Download and extract the Windows GPU pack (ONNX Runtime CUDA provider +
/// CUDA 12 runtime + cuDNN 9) into [`gpu_dir`]. All components are official
/// vendor archives with pinned hashes.
#[cfg(target_os = "windows")]
pub(crate) fn ensure_windows_gpu_pack() -> Result<PathBuf> {
    let dir = gpu_dir();
    if gpu_pack_complete(&dir) {
        return Ok(dir);
    }
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;

    let session = DownloadSession::new("GPU acceleration pack");
    let tmp = temp_download_dir()?;

    // 1) ONNX Runtime CUDA provider (from the official GPU wheel).
    let wheel = tmp.join("onnxruntime_gpu-1.24.4-cp313-cp313-win_amd64.whl");
    if !wheel.exists() {
        download_verified(
            ORT_WHEEL_URL,
            ORT_WHEEL_SHA256,
            ORT_WHEEL_SIZE,
            &wheel,
            "ONNX Runtime CUDA provider",
        )?;
    }
    session.set("Extracting ONNX Runtime CUDA provider", 1.0);
    let n = extract_named(
        &wheel,
        &dir,
        &[
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_shared.dll",
        ],
        "ONNX Runtime wheel",
    )?;
    if n == 0 {
        bail!("ONNX Runtime wheel did not contain the CUDA provider DLL");
    }
    let _ = fs::remove_file(&wheel);

    // 2) cuDNN 9.
    let cudnn = tmp.join("cudnn-windows-x86_64-9.3.0.75_cuda12-archive.zip");
    if !cudnn.exists() {
        download_verified(
            CUDNN_WIN_URL,
            CUDNN_WIN_SHA256,
            CUDNN_WIN_SIZE,
            &cudnn,
            "cuDNN 9",
        )?;
    }
    session.set("Extracting cuDNN 9", 1.0);
    extract_dlls(&cudnn, &dir, "cuDNN archive")?;
    let _ = fs::remove_file(&cudnn);

    // 3) CUDA 12 runtime components.
    for (relative, sha, size) in CUDA_COMPONENTS {
        let archive_name = Path::new(relative)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "cuda-component.zip".to_string());
        let archive = tmp.join(&archive_name);
        let label = format!("CUDA runtime ({archive_name})");
        if !archive.exists() {
            download_verified(
                &format!("{CUDA_REDIST_BASE}/{relative}"),
                sha,
                *size,
                &archive,
                &label,
            )?;
        }
        session.set(&format!("Extracting {label}"), 1.0);
        extract_dlls(&archive, &dir, &label)?;
        let _ = fs::remove_file(&archive);
    }

    if !gpu_pack_complete(&dir) {
        let missing: Vec<&str> = GPU_PACK_REQUIRED_DLLS
            .iter()
            .filter(|dll| !dir.join(dll).exists())
            .copied()
            .collect();
        bail!("GPU pack incomplete, missing: {}", missing.join(", "));
    }
    Ok(dir)
}

// ---------------------------------------------------------------------------
// FFmpeg
// ---------------------------------------------------------------------------

/// Ensure an FFmpeg binary is available, downloading the mirrored Windows
/// build on first use. Returns the path to `ffmpeg`.
///
/// Only called by [`crate::dsp::get_ffmpeg_command`] after its local search
/// (exe dir, cwd, bin dir, system PATH) came up empty.
pub(crate) fn ensure_ffmpeg() -> Result<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let dir = bin_dir();
        let ffmpeg = dir.join("ffmpeg.exe");
        let ffprobe = dir.join("ffprobe.exe");
        if ffmpeg.exists() && ffprobe.exists() {
            return Ok(ffmpeg);
        }
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;

        let session = DownloadSession::new("FFmpeg media tools");
        let archive = temp_download_dir()?.join("keyscribe-ffmpeg-win64-v1.zip");
        if !archive.exists() {
            download_verified(
                FFMPEG_WIN_URL,
                FFMPEG_WIN_SHA256,
                FFMPEG_WIN_SIZE,
                &archive,
                "FFmpeg (audio fallback decoding and video)",
            )?;
        }
        session.set("Extracting FFmpeg", 1.0);
        extract_named(
            &archive,
            &dir,
            &["ffmpeg.exe", "ffprobe.exe"],
            "FFmpeg archive",
        )?;
        let _ = fs::remove_file(&archive);

        if ffmpeg.exists() {
            return Ok(ffmpeg);
        }
        bail!("FFmpeg became available but ffmpeg.exe was not extracted");
    }

    #[cfg(not(target_os = "windows"))]
    {
        bail!(
            "FFmpeg was not found. Install it with your package manager \
             (e.g. `sudo apt install ffmpeg` or `brew install ffmpeg`) to decode \
             this file format."
        )
    }
}
