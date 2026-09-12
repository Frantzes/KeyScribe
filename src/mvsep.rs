use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// Base API endpoint for MVSep
pub const MVSEP_API_BASE_URL: &str = "https://mvsep.com/api";

/// Default MVSep model: BS Roformer SW (6 stems: vocals, bass, drums, guitar, piano, other)
pub const DEFAULT_MVSEP_MODEL_ID: u32 = 63;
pub const DEFAULT_MVSEP_MODEL_NAME: &str = "mvsep_63_bs_roformer_sw";

/// Global mutual exclusion lock to guarantee strictly sequential, non-concurrent
/// requests to MVSep API to prevent rate limit violations or account bans.
static MVSEP_CONCURRENCY_LOCK: Mutex<()> = Mutex::new(());

/// Metadata describing an available separation model on MVSep
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvsepModel {
    pub id: u32,
    pub technical_name: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    pub stems: &'static [&'static str],
}

/// Curated list of top separation models supported by MVSep
pub const MVSEP_MODELS: &[MvsepModel] = &[
    MvsepModel {
        id: 63,
        technical_name: "mvsep_63_bs_roformer_sw",
        display_name: "BS Roformer SW (vocals, bass, drums, guitar, piano, other)",
        description: "6 stems with superior SDR quality (Default)",
        stems: &["vocals", "bass", "drums", "guitar", "piano", "other"],
    },
    MvsepModel {
        id: 30,
        technical_name: "mvsep_30_ensemble_all_in",
        display_name: "Ensemble All-In (vocals, bass, drums, piano, guitar, lead/back vocals, other)",
        description: "7 stems ensemble separation",
        stems: &[
            "vocals",
            "bass",
            "drums",
            "piano",
            "guitar",
            "lead_back_vocals",
            "other",
        ],
    },
    MvsepModel {
        id: 28,
        technical_name: "mvsep_28_ensemble_5s",
        display_name: "Ensemble (vocals, instrum, bass, drums, other)",
        description: "5 stems ensemble separation",
        stems: &["vocals", "instrum", "bass", "drums", "other"],
    },
    MvsepModel {
        id: 20,
        technical_name: "mvsep_20_demucs4_ht",
        display_name: "Demucs4 HT (vocals, drums, bass, other)",
        description: "4 stems high quality separation",
        stems: &["vocals", "drums", "bass", "other"],
    },
    MvsepModel {
        id: 40,
        technical_name: "mvsep_40_bs_roformer_2s",
        display_name: "BS Roformer (vocals, instrumental)",
        description: "2 stems vocal and instrumental isolation",
        stems: &["vocals", "instrumental"],
    },
    MvsepModel {
        id: 48,
        technical_name: "mvsep_48_melband_roformer_2s",
        display_name: "MelBand Roformer (vocals, instrumental)",
        description: "2 stems vocal and instrumental isolation",
        stems: &["vocals", "instrumental"],
    },
    MvsepModel {
        id: 29,
        technical_name: "mvsep_29_piano",
        display_name: "MVSep Piano (piano, other)",
        description: "Dedicated piano isolation",
        stems: &["piano", "other"],
    },
    MvsepModel {
        id: 44,
        technical_name: "mvsep_44_drums",
        display_name: "MVSep Drums (drums, other)",
        description: "Dedicated drums isolation",
        stems: &["drums", "other"],
    },
    MvsepModel {
        id: 41,
        technical_name: "mvsep_41_bass",
        display_name: "MVSep Bass (bass, other)",
        description: "Dedicated bass isolation",
        stems: &["bass", "other"],
    },
    MvsepModel {
        id: 31,
        technical_name: "mvsep_31_guitar",
        display_name: "MVSep Guitar (guitar, other)",
        description: "Dedicated guitar isolation",
        stems: &["guitar", "other"],
    },
];

pub fn get_mvsep_model_by_id(id: u32) -> Option<&'static MvsepModel> {
    MVSEP_MODELS.iter().find(|m| m.id == id)
}

pub fn get_mvsep_model_by_name(name: &str) -> Option<&'static MvsepModel> {
    MVSEP_MODELS.iter().find(|m| m.technical_name == name)
}

/// Helper to parse model ID from model string (e.g. "mvsep_63_bs_roformer_sw" -> 63)
pub fn parse_mvsep_model_id(name_or_id: &str) -> Option<u32> {
    if let Ok(id) = name_or_id.parse::<u32>() {
        return Some(id);
    }
    if let Some(model) = get_mvsep_model_by_name(name_or_id) {
        return Some(model.id);
    }
    if let Some(stripped) = name_or_id.strip_prefix("mvsep_") {
        if let Some((id_str, _)) = stripped.split_once('_') {
            if let Ok(id) = id_str.parse::<u32>() {
                return Some(id);
            }
        }
    }
    None
}

/// Map an input audio file's extension to MVSep API's `output_format` parameter and expected extension:
/// - "0": MP3 (320kbps)
/// - "1": WAV (uncompressed 16-bit)
/// - "2": FLAC (lossless 16-bit)
/// - "3": M4A (lossy AAC)
pub fn detect_output_format(file_path: &Path) -> (&'static str, &'static str) {
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "mp3" => ("0", "mp3"),
        "flac" => ("2", "flac"),
        "m4a" | "aac" => ("3", "m4a"),
        _ => ("1", "wav"),
    }
}

/// Retrieve the MVSep API key from environment variables or local .env files.
/// Never logs or displays the raw key to avoid security leaks.
///
/// Lookup order:
/// 1. `MVSEP_API_KEY` process environment variable.
/// 2. User-local `.env` managed by the app (see [`user_dotenv_path`]) — the
///    primary store for keys entered in Settings; lives outside the repo.
/// 3. `.env` in the current working directory (dev convenience).
/// 4. `.env` next to the executable (portable installs).
pub fn find_mvsep_api_key() -> Option<String> {
    // 1. Process environment variable (explicit override).
    if let Ok(key) = std::env::var("MVSEP_API_KEY") {
        let trimmed = key.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }

    // 2. OS/device keystore (Android Keystore / Windows DPAPI).
    if let Some(key) = crate::secrets::load() {
        return Some(key);
    }

    // 3-5. Legacy plaintext `.env` files. Migrate into the keystore on hit.
    let mut search_dirs: Vec<PathBuf> = Vec::new();
    if let Some(user_env) = user_dotenv_path() {
        if let Some(parent) = user_env.parent() {
            search_dirs.push(parent.to_path_buf());
        }
    }
    search_dirs.push(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if let Some(exe_parent) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
    {
        search_dirs.push(exe_parent);
    }

    for dir in search_dirs {
        let env_path = dir.join(".env");
        if env_path.is_file() {
            if let Some(key) = parse_dotenv_api_key(&env_path) {
                // One-time migration away from plaintext, when a keystore exists.
                if crate::secrets::store(&key).is_ok() {
                    remove_legacy_dotenv_key();
                }
                return Some(key);
            }
        }
    }

    None
}

/// Persist the API key using the platform keystore, falling back to the
/// legacy user `.env` file only when no OS keystore is available. An empty
/// key clears both stores.
pub fn save_mvsep_api_key(key: &str) -> Result<()> {
    let key = key.trim();
    if key.is_empty() {
        let _ = crate::secrets::clear();
        remove_legacy_dotenv_key();
        return Ok(());
    }
    match crate::secrets::store(key) {
        Ok(()) => {
            // Remove any legacy plaintext copy.
            remove_legacy_dotenv_key();
            Ok(())
        }
        Err(_) => save_mvsep_api_key_to_dotenv(key),
    }
}

/// Delete a legacy plaintext `.env` entry, but only if the file exists (so we
/// never create an empty `.env`).
fn remove_legacy_dotenv_key() {
    if let Some(path) = user_dotenv_path() {
        if path.is_file() {
            let _ = save_mvsep_api_key_to_dotenv("");
        }
    }
}

/// Parse `MVSEP_API_KEY` out of a dotenv file. Returns `None` when absent.
fn parse_dotenv_api_key(env_path: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(env_path).ok()?;
    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "MVSEP_API_KEY" {
                let key = v
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();
                if !key.is_empty() {
                    return Some(key);
                }
            }
        }
    }
    None
}

/// Path of the user-local `.env` file managed by the app.
///
/// Always under the OS user-data dir (`~/.local/share/KeyScribe/.env` on
/// Linux), never next to the executable (read-only in AppImage/Flatpak) and
/// never inside the repo, so keys cannot be committed by accident.
pub fn user_dotenv_path() -> Option<PathBuf> {
    // Android has no home/XDG dirs, so `directories` either returns None or an
    // unusable path; use the app-private files dir instead.
    if let Some(dir) = crate::platform::data_dir() {
        return Some(dir.join(".env"));
    }
    directories::ProjectDirs::from("com", "Frantzes", "KeyScribe")
        .map(|d| d.data_local_dir().join(".env"))
}

/// Persist the MVSep API key to the user-local `.env` file, preserving any
/// other variables already present. The file is created with `0600`
/// permissions on Unix. An empty key removes our entry instead.
pub fn save_mvsep_api_key_to_dotenv(key: &str) -> Result<()> {
    let path = user_dotenv_path().ok_or_else(|| anyhow!("Could not locate user data directory"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }

    let key = key.trim();
    let mut lines: Vec<String> = Vec::new();
    if path.is_file() {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        for line in contents.lines() {
            let trimmed = line.trim();
            let is_ours = !trimmed.starts_with('#')
                && trimmed.split_once('=').map(|(k, _)| k.trim()) == Some("MVSEP_API_KEY");
            if !is_ours {
                lines.push(line.to_string());
            }
        }
    }
    if !key.is_empty() {
        lines.push(format!("MVSEP_API_KEY={key}"));
    }

    std::fs::write(&path, lines.join("\n") + if lines.is_empty() { "" } else { "\n" })
        .with_context(|| format!("Failed to write {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// User profile info returned by MVSep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvsepUserInfo {
 pub name: Option<String>,
 pub email: Option<String>,
 pub premium_enabled: Option<i32>,
 pub current_queue: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct MvsepUserApiResponse {
 success: bool,
 data: Option<MvsepUserInfo>,
 error: Option<String>,
 message: Option<String>,
}

/// Verify an API key with MVSep by querying /api/app/user.
pub fn verify_api_key(api_key: &str) -> Result<MvsepUserInfo> {
 let key = api_key.trim();
 if key.is_empty() {
 return Err(anyhow!("MVSep API key is empty"));
 }

 let client = reqwest::blocking::Client::builder()
 .timeout(Duration::from_secs(15))
 .build()
 .context("Failed to build HTTP client")?;

 let url = format!("{MVSEP_API_BASE_URL}/app/user?api_token={key}");
 let resp = client
 .get(&url)
 .send()
 .context("Failed to send verification request to MVSep")?;

 if !resp.status().is_success() {
 return Err(anyhow!(
 "MVSep server responded with HTTP status {}",
 resp.status()
 ));
 }

 let parsed: MvsepUserApiResponse = resp
 .json()
 .context("Failed to parse MVSep user profile response")?;

 if !parsed.success {
 let msg = parsed
 .message
 .or(parsed.error)
 .unwrap_or_else(|| "Invalid or unauthorized API key".to_string());
 return Err(anyhow!("{msg}"));
 }

 parsed
 .data
 .ok_or_else(|| anyhow!("No user data returned by MVSep"))
}

#[derive(Deserialize, Default)]
struct MvsepCreateData {
  #[serde(default)]
  hash: Option<String>,
  #[serde(default)]
  message: Option<String>,
}

#[derive(Deserialize, Default)]
struct MvsepCreateResponse {
  #[serde(default)]
  success: bool,
  #[serde(default)]
  data: Option<MvsepCreateData>,
  #[serde(default)]
  error: Option<String>,
  #[serde(default)]
  message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MvsepFileEntry {
    #[serde(rename = "type")]
    pub stem_type: String,
    pub url: String,
    pub download: Option<String>,
    pub size: Option<String>,
}

#[derive(Deserialize)]
pub struct MvsepStatusData {
    pub queue_count: Option<u32>,
    pub current_order: Option<u32>,
    pub message: Option<String>,
    pub files: Option<Vec<MvsepFileEntry>>,
}

#[derive(Deserialize)]
pub struct MvsepStatusResponse {
    pub success: bool,
    #[serde(default)]
    pub status: Option<String>,
    pub data: Option<MvsepStatusData>,
    pub error: Option<String>,
    pub message: Option<String>,
}

/// Maximum bytes accepted for a single downloaded stem (matches the 1000 MB
/// client-side upload cap). Guards against disk-fill from a misbehaving or
/// compromised server.
pub const MAX_STEM_DOWNLOAD_BYTES: u64 = 1000 * 1024 * 1024;

/// Hosts stem files may be downloaded from. A compromised server could
/// otherwise redirect downloads (and the resulting file writes) anywhere.
fn stem_url_host_allowed(url: &reqwest::Url) -> bool {
    if url.scheme() != "https" {
        return false;
    }
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    if host == "mvsep.com" || host.ends_with(".mvsep.com") {
        return true;
    }
    // Escape hatch for legit CDN moves (documented in the rejection error).
    if let Ok(extra) = std::env::var("MVSEP_EXTRA_DOWNLOAD_HOSTS") {
        for h in extra.split(',') {
            let h = h.trim().trim_start_matches('.').to_ascii_lowercase();
            if !h.is_empty() && (host == h || host.ends_with(&format!(".{h}"))) {
                return true;
            }
        }
    }
    false
}

/// Build a safe stem filename from a server-provided stem type.
///
/// Only `[a-z0-9_-]` survive; anything else (including `/`, `\` and `..`
/// path-traversal sequences) is stripped, with a positional fallback when
/// nothing remains. The extension must come from the fixed set chosen by the
/// caller, never from the server.
fn sanitize_stem_filename(stem_type: &str, ext: &str, index: usize) -> String {
    let mut clean = stem_type.trim().to_lowercase().replace(' ', "_");
    for ext_to_strip in &[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"] {
        if clean.ends_with(ext_to_strip) {
            clean.truncate(clean.len() - ext_to_strip.len());
        }
    }
    let safe: String = clean
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    let stem = if safe.is_empty() {
        format!("stem_{index}")
    } else {
        safe
    };
    format!("{stem}.{ext}")
}

/// Check if an error message indicates an invalid token, missing token, or unauthorized error
pub fn is_invalid_token_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("invalid token")
        || lower.contains("invalid api key")
        || lower.contains("bad api token")
        || lower.contains("api key is required")
        || lower.contains("api key required")
        || lower.contains("unauthorized")
}

/// Compact a raw server response body into a short single-line snippet for
/// error messages (whitespace-collapsed, truncated). Never includes the API
/// key: MVSep responses echo job data, never credentials.
fn body_text_snippet(body: &str) -> String {
    let compact: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX_SNIPPET_CHARS: usize = 300;
    if compact.chars().count() <= MAX_SNIPPET_CHARS {
        compact
    } else {
        let truncated: String = compact.chars().take(MAX_SNIPPET_CHARS).collect();
        format!("{truncated}…")
    }
}

/// Format an error message returned by MVSep into user-friendly explanations.
pub fn format_mvsep_error(raw_err: &str) -> String {
    let lower = raw_err.to_ascii_lowercase();
    if lower.contains("invalid token")
        || lower.contains("invalid api key")
        || lower.contains("bad api token")
        || lower.contains("unauthorized")
    {
        "invalid token".to_string()
    } else if lower.contains("too long") || lower.contains("duration") || lower.contains("length exceeds") {
        format!("{raw_err} (Free accounts are limited to 10 minutes; Premium accounts allow up to 100 minutes)")
    } else if lower.contains("too large") || lower.contains("file size") || lower.contains("payload too large") || lower.contains("filesize") {
        format!("{raw_err} (Free accounts are limited to 100 MB; Premium accounts allow up to 1000 MB)")
    } else if lower.contains("concurrent") || lower.contains("queue limit") {
        format!("{raw_err} (Free accounts allow only 1 concurrent separation at a time)")
    } else if lower.contains("credit") || lower.contains("balance") || lower.contains("funds") {
        format!("{raw_err} (Check your credit balance at https://mvsep.com)")
    } else {
        raw_err.to_string()
    }
}

/// Separate an audio file using MVSep Cloud.
///
/// Ensures strictly sequential requests via MVSEP_CONCURRENCY_LOCK.
/// Reports progress via progress_cb(ratio, message).
/// Downloads output stems as WAV files into output_dir.
pub fn separate_audio_file(
    api_key: &str,
    file_path: &Path,
    sep_type: u32,
    output_dir: &Path,
    progress_cb: Option<&dyn Fn(f32, &str)>,
) -> Result<Vec<PathBuf>> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err(anyhow!(
            "MVSep API key is required. Please provide an API key (https://mvsep.com/en/full_api)."
        ));
    }

    if !file_path.exists() {
        return Err(anyhow!("Audio source file not found: {:?}", file_path));
    }

    // Pre-check file size against MVSep absolute hard limit (1000 MB)
    if let Ok(meta) = file_path.metadata() {
        if meta.len() > 1000 * 1024 * 1024 {
            return Err(anyhow!(
                "Audio file is too large ({:.1} MB). MVSep maximum file size limit is 1000 MB (100 MB for free accounts).",
                meta.len() as f64 / (1024.0 * 1024.0)
            ));
        }
    }

    // Acquire lock to guarantee no concurrent separation jobs run against MVSep
    let _concurrency_guard = MVSEP_CONCURRENCY_LOCK
        .lock()
        .map_err(|_| anyhow!("Failed to acquire MVSep concurrency guard"))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300)) // 5 minute timeout for uploads
        .build()
        .context("Failed to build HTTP client")?;

    if let Some(cb) = progress_cb {
        cb(0.02, "Uploading audio to MVSep cloud...");
    }

    let (output_format_code, expected_ext) = detect_output_format(file_path);

    // 1. Submit separation job via multipart/form-data
    // Note: is_demo = "0" guarantees the separation is private and not published to demo page
    let form = reqwest::blocking::multipart::Form::new()
        .text("api_token", key.to_string())
        .text("sep_type", sep_type.to_string())
        .text("output_format", output_format_code)
        .text("is_demo", "0")
        .file("audiofile", file_path)
        .context("Failed to attach audio file to multipart form")?;

    let create_url = format!("{MVSEP_API_BASE_URL}/separation/create");
    let create_resp = client
        .post(&create_url)
        .multipart(form)
        .send()
        .context("Failed to upload audio to MVSep")?;

    let create_status = create_resp.status();
    let body_text = create_resp.text().unwrap_or_default();
    let create_body: Option<MvsepCreateResponse> = serde_json::from_str(&body_text).ok();

    let job_hash = if let Some(body) = create_body {
        if !body.success {
            // Log the full body for diagnosis; server rejections without a
            // message (e.g. concurrent-job or rate limits on free accounts)
            // previously surfaced as an opaque "Unknown MVSep error".
            eprintln!("[mvsep] job creation rejected (HTTP {create_status}): {body_text}");
            let err_msg = body
                .data
                .and_then(|d| d.message)
                .or(body.message)
                .or(body.error)
                .filter(|m| !m.trim().is_empty())
                .map(|m| {
                    let snippet = body_text_snippet(&body_text);
                    if snippet.is_empty() {
                        m
                    } else {
                        format!("{m} (server response: {snippet})")
                    }
                })
                .unwrap_or_else(|| {
                    let snippet = body_text_snippet(&body_text);
                    let mut msg = format!(
                        "MVSep rejected the job without an explanation (HTTP {create_status})"
                    );
                    if !snippet.is_empty() {
                        msg.push_str(&format!(" — server response: {snippet}"));
                    }
                    msg.push_str(
                        ". If you recently ran a separation, MVSep free accounts allow only \
                         1 concurrent job: wait for the previous job to finish and retry.",
                    );
                    msg
                });
            let formatted = format_mvsep_error(&err_msg);
            return Err(anyhow!("MVSep job creation failed: {formatted}"));
        }

        body.data
            .and_then(|d| d.hash)
            .ok_or_else(|| anyhow!("MVSep response did not return a job hash"))?
    } else {
        if create_status == reqwest::StatusCode::PAYLOAD_TOO_LARGE || body_text.to_ascii_lowercase().contains("too large") {
            return Err(anyhow!(
                "MVSep job creation failed: Audio file size or duration exceeds MVSep limit (HTTP 413). Free accounts have a 10-minute / 100 MB limit (Premium allows up to 100 minutes / 1000 MB)."
            ));
        }
        if !create_status.is_success() {
            eprintln!("[mvsep] job creation HTTP {create_status}: {body_text}");
            let first_line = body_text.lines().next().unwrap_or("").trim();
            let summary = if !first_line.is_empty() {
                format_mvsep_error(first_line)
            } else {
                format!("HTTP error {create_status}")
            };
            return Err(anyhow!("MVSep job creation failed: {summary}"));
        }
        return Err(anyhow!("Failed to parse MVSep create response (status {create_status})"));
    };

    if let Some(cb) = progress_cb {
        cb(0.15, "Separation queued on MVSep cloud...");
    }

    // 2. Poll job status until done or failed
    let poll_url = format!("{MVSEP_API_BASE_URL}/separation/get?hash={job_hash}");
    let poll_client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP poll client")?;

    let mut poll_attempts = 0;
    let max_poll_attempts = 300; // ~15-20 minutes max wait
    let mut consecutive_errors = 0;

    let stem_files = loop {
        poll_attempts += 1;
        if poll_attempts > max_poll_attempts {
            return Err(anyhow!("MVSep separation timed out after 15 minutes"));
        }

        std::thread::sleep(Duration::from_secs(3));

        let poll_resp = match poll_client.get(&poll_url).send() {
            Ok(r) => {
                consecutive_errors = 0;
                r
            }
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors >= 5 {
                    return Err(anyhow!("MVSep polling network error: {e}"));
                }
                continue;
            }
        };

        let status_body: MvsepStatusResponse = match poll_resp.json() {
            Ok(b) => b,
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors >= 5 {
                    return Err(anyhow!("Failed to parse MVSep status response: {e}"));
                }
                continue;
            }
        };

        let current_status = status_body
            .status
            .as_deref()
            .unwrap_or("unknown")
            .to_ascii_lowercase();

        match current_status.as_str() {
            "waiting" => {
                let order = status_body
                    .data
                    .as_ref()
                    .and_then(|d| d.current_order)
                    .unwrap_or(1);
                let count = status_body
                    .data
                    .as_ref()
                    .and_then(|d| d.queue_count)
                    .unwrap_or(1);
                let msg = format!("MVSep in queue: position {order} of {count}");
                let frac = if count > 0 {
                    (1.0 - (order as f32 / count as f32).clamp(0.0, 1.0)) * 0.25
                } else {
                    0.0
                };
                if let Some(cb) = progress_cb {
                    cb(0.15 + frac, &msg);
                }
            }
            "processing" => {
                if let Some(cb) = progress_cb {
                    cb(0.50, "MVSep processing audio with GPU...");
                }
            }
            "distributing" | "merging" => {
                if let Some(cb) = progress_cb {
                    cb(0.65, "MVSep merging stem chunks...");
                }
            }
            "done" => {
                let files = status_body
                    .data
                    .and_then(|d| d.files)
                    .unwrap_or_default();
                if files.is_empty() {
                    return Err(anyhow!("MVSep completed but returned no output stem files"));
                }
                break files;
            }
            "failed" => {
                let err_msg = status_body
                    .data
                    .and_then(|d| d.message)
                    .or(status_body.message)
                    .or(status_body.error)
                    .unwrap_or_else(|| "Separation failed on MVSep server".to_string());
                let formatted = format_mvsep_error(&err_msg);
                return Err(anyhow!("MVSep separation failed: {formatted}"));
            }
            other => {
                if let Some(cb) = progress_cb {
                    cb(0.40, &format!("MVSep status: {other}..."));
                }
            }
        }
    };

    // 3. Download separated stem files into output_dir
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

    let has_other = stem_files.iter().any(|f| {
        let t = f.stem_type.trim().to_ascii_lowercase();
        t == "other" || t == "other.wav"
    });
    let has_individual_stems = stem_files.iter().any(|f| {
        let t = f.stem_type.trim().to_ascii_lowercase();
        matches!(t.as_str(), "bass" | "drums" | "piano" | "guitar")
    });

    let filtered_stem_files: Vec<_> = stem_files
        .into_iter()
        .filter(|f| {
            let t = f.stem_type.trim().to_ascii_lowercase();
            // If individual stems (like other, bass, drums, etc.) exist,
            // "instrum" is a composite backing track generated by MVSep for karaoke,
            // NOT an independent stem.
            if (t == "instrum" || t == "instrumental") && (has_other || has_individual_stems) {
                return false;
            }
            true
        })
        .collect();

    let total_stems = filtered_stem_files.len();
    let mut downloaded_paths = Vec::new();

    for (idx, file) in filtered_stem_files.iter().enumerate() {
        // Validate the download URL before touching the network: HTTPS only,
        // MVSep hosts only. A compromised server must not redirect downloads
        // (or the resulting file writes) to arbitrary hosts.
        let stem_url = reqwest::Url::parse(&file.url).with_context(|| {
            format!("MVSep returned an invalid stem URL for '{}'", file.stem_type)
        })?;
        if !stem_url_host_allowed(&stem_url) {
            return Err(anyhow!(
                "MVSep returned a stem URL outside the trusted hosts for '{}': {} \
                 (expected https://mvsep.com/…; allow more via MVSEP_EXTRA_DOWNLOAD_HOSTS)",
                file.stem_type,
                stem_url.host_str().unwrap_or("?"),
            ));
        }

        let url_path = file.url.split('?').next().unwrap_or(&file.url);
        let ext = if url_path.ends_with(".mp3") {
            "mp3"
        } else if url_path.ends_with(".flac") {
            "flac"
        } else if url_path.ends_with(".m4a") || url_path.ends_with(".aac") {
            "m4a"
        } else if url_path.ends_with(".wav") {
            "wav"
        } else {
            expected_ext
        };
        // Server-provided stem types are untrusted: sanitize so the filename
        // cannot escape output_dir (no `/`, `\`, `..` survive).
        let stem_filename = sanitize_stem_filename(&file.stem_type, ext, idx);
        let stem_path = output_dir.join(&stem_filename);
        if !stem_path.starts_with(output_dir) {
            return Err(anyhow!(
                "Refusing to write stem '{}' outside the output directory",
                file.stem_type
            ));
        }

        if let Some(cb) = progress_cb {
            let progress = 0.70 + 0.28 * (idx as f32 / total_stems.max(1) as f32);
            let msg = format!(
                "Downloading stem {}/{}: {}...",
                idx + 1,
                total_stems,
                file.stem_type
            );
            cb(progress, &msg);
        }

        let mut file_resp = client
            .get(stem_url)
            .send()
            .with_context(|| format!("Failed to download stem file from {}", file.url))?;

        if !file_resp.status().is_success() {
            return Err(anyhow!(
                "Failed to download stem '{}' (HTTP {})",
                file.stem_type,
                file_resp.status()
            ));
        }

        // Stream to disk with a hard cap instead of buffering the whole
        // response in RAM: a misbehaving server must not fill memory/disk.
        if let Some(len) = file_resp.content_length() {
            if len > MAX_STEM_DOWNLOAD_BYTES {
                return Err(anyhow!(
                    "Stem '{}' exceeds the {} MB download limit (server claims {} bytes)",
                    file.stem_type,
                    MAX_STEM_DOWNLOAD_BYTES / (1024 * 1024),
                    len,
                ));
            }
        }
        let mut out_file = std::fs::File::create(&stem_path)
            .with_context(|| format!("Failed to create stem file at {:?}", stem_path))?;
        // Stream to disk through a byte-counting writer instead of
        // buffering the whole response in RAM: a misbehaving server must
        // not exhaust memory or disk.
        struct CappedWriter<W: std::io::Write> {
            inner: W,
            written: u64,
            cap: u64,
        }
        impl<W: std::io::Write> std::io::Write for CappedWriter<W> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let n = self.inner.write(buf)?;
                self.written += n as u64;
                if self.written > self.cap {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::QuotaExceeded,
                        "download size cap exceeded",
                    ));
                }
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.inner.flush()
            }
        }
        let mut capped = CappedWriter {
            inner: &mut out_file,
            written: 0,
            cap: MAX_STEM_DOWNLOAD_BYTES,
        };
        if let Err(e) = file_resp.copy_to(&mut capped) {
            drop(out_file);
            let _ = std::fs::remove_file(&stem_path);
            return Err(anyhow!(
                "Failed to download stem '{}' ({}; limit {} MB)",
                file.stem_type,
                e,
                MAX_STEM_DOWNLOAD_BYTES / (1024 * 1024),
            ));
        }

        downloaded_paths.push(stem_path);
    }

    if let Some(cb) = progress_cb {
        cb(1.0, "MVSep stem separation complete!");
    }

    Ok(downloaded_paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_mvsep_api_key_from_env() {
        let key = find_mvsep_api_key();
        assert!(
            key.is_some(),
            "Expected to find MVSEP_API_KEY from .env or environment"
        );
        let key_str = key.unwrap();
        assert!(!key_str.is_empty());
    }

    #[test]
    fn test_default_model_is_bs_roformer_sw() {
        let default_model = get_mvsep_model_by_id(DEFAULT_MVSEP_MODEL_ID);
        assert!(default_model.is_some());
        let m = default_model.unwrap();
        assert_eq!(m.id, 63);
        assert_eq!(m.technical_name, DEFAULT_MVSEP_MODEL_NAME);
        assert!(m.display_name.contains("BS Roformer SW"));
        assert_eq!(m.stems.len(), 6);
        assert_eq!(
            m.stems,
            &["vocals", "bass", "drums", "guitar", "piano", "other"]
        );
    }

    #[test]
    fn test_parse_mvsep_model_id() {
        assert_eq!(parse_mvsep_model_id("63"), Some(63));
        assert_eq!(parse_mvsep_model_id("mvsep_63_bs_roformer_sw"), Some(63));
        assert_eq!(parse_mvsep_model_id("mvsep_20_demucs4_ht"), Some(20));
        assert_eq!(parse_mvsep_model_id("invalid"), None);
    }

    #[test]
    fn test_parse_mvsep_status_response_waiting() {
        let json_waiting = r#"{
            "success": true,
            "status": "waiting",
            "data": {
                "queue_count": 5,
                "current_order": 2
            }
        }"#;

        let res: MvsepStatusResponse = serde_json::from_str(json_waiting).unwrap();
        assert!(res.success);
        assert_eq!(res.status.as_deref(), Some("waiting"));
        let data = res.data.unwrap();
        assert_eq!(data.queue_count, Some(5));
        assert_eq!(data.current_order, Some(2));
    }

    #[test]
    fn test_parse_mvsep_status_response_done() {
        let json_done = r#"{
            "success": true,
            "status": "done",
            "data": {
                "files": [
                    {"type": "Vocals", "url": "https://mvsep.com/storage/vocals.wav", "download": "vocals.wav"},
                    {"type": "Bass", "url": "https://mvsep.com/storage/bass.wav", "download": "bass.wav"},
                    {"type": "Drums", "url": "https://mvsep.com/storage/drums.wav", "download": "drums.wav"},
                    {"type": "Guitar", "url": "https://mvsep.com/storage/guitar.wav", "download": "guitar.wav"},
                    {"type": "Piano", "url": "https://mvsep.com/storage/piano.wav", "download": "piano.wav"},
                    {"type": "Other", "url": "https://mvsep.com/storage/other.wav", "download": "other.wav"}
                ]
            }
        }"#;

        let res: MvsepStatusResponse = serde_json::from_str(json_done).unwrap();
        assert!(res.success);
        assert_eq!(res.status.as_deref(), Some("done"));
        let files = res.data.unwrap().files.unwrap();
        assert_eq!(files.len(), 6);
        assert_eq!(files[0].stem_type, "Vocals");
        assert_eq!(files[1].stem_type, "Bass");
        assert_eq!(files[2].stem_type, "Drums");
        assert_eq!(files[3].stem_type, "Guitar");
        assert_eq!(files[4].stem_type, "Piano");
        assert_eq!(files[5].stem_type, "Other");
    }

    #[test]
    fn test_mvsep_user_verification_live() {
        let Some(key) = find_mvsep_api_key() else {
            eprintln!("Skipping live verification: MVSEP_API_KEY not found");
            return;
        };

        if key.len() < 20 {
            eprintln!("Skipping live verification: MVSEP_API_KEY looks like a dummy/test token");
            return;
        }

        match verify_api_key(&key) {
            Ok(user_info) => {
                assert!(user_info.name.is_some() || user_info.email.is_some());
            }
            Err(e) => {
                eprintln!("Live verification skipped or failed: {e}");
            }
        }
    }

    #[test]
    fn test_detect_output_format() {
        assert_eq!(detect_output_format(Path::new("song.mp3")), ("0", "mp3"));
        assert_eq!(detect_output_format(Path::new("track.MP3")), ("0", "mp3"));
        assert_eq!(detect_output_format(Path::new("audio.flac")), ("2", "flac"));
        assert_eq!(detect_output_format(Path::new("recording.m4a")), ("3", "m4a"));
        assert_eq!(detect_output_format(Path::new("clip.aac")), ("3", "m4a"));
        assert_eq!(detect_output_format(Path::new("master.wav")), ("1", "wav"));
        assert_eq!(detect_output_format(Path::new("voice.ogg")), ("1", "wav"));
    }

    #[test]
    fn test_is_invalid_token_error() {
        assert!(is_invalid_token_error("Separation failed: MVSep job creation failed: invalid token"));
        assert!(is_invalid_token_error("invalid token"));
        assert!(is_invalid_token_error("MVSep API key is required"));
        assert!(is_invalid_token_error("unauthorized"));
        assert!(!is_invalid_token_error("Audio file is too long"));
        assert!(!is_invalid_token_error("File size exceeds limit"));
    }

    #[test]
    fn test_format_mvsep_error() {
        assert_eq!(format_mvsep_error("invalid token"), "invalid token");
        assert!(format_mvsep_error("Audio file is too long").contains("Free accounts are limited to 10 minutes"));
        assert!(format_mvsep_error("Duration exceeds limit").contains("Free accounts are limited to 10 minutes"));
        assert!(format_mvsep_error("File size is too large").contains("Free accounts are limited to 100 MB"));
        assert!(format_mvsep_error("Too many concurrent jobs").contains("Free accounts allow only 1 concurrent"));
    }

    #[test]
    fn test_sanitize_stem_filename() {
        assert_eq!(sanitize_stem_filename("Vocals", "wav", 0), "vocals.wav");
        assert_eq!(sanitize_stem_filename("Lead Back Vocals", "mp3", 1), "lead_back_vocals.mp3");
        assert_eq!(sanitize_stem_filename("other.wav", "wav", 2), "other.wav");
        // Path traversal attempts are neutralized.
        assert_eq!(sanitize_stem_filename("../../evil", "wav", 0), "evil.wav");
        assert_eq!(sanitize_stem_filename("a/b\\c", "wav", 1), "abc.wav");
        assert_eq!(sanitize_stem_filename("...", "wav", 2), "stem_2.wav");
        assert_eq!(sanitize_stem_filename("", "wav", 3), "stem_3.wav");
        assert_eq!(sanitize_stem_filename("!!!", "flac", 4), "stem_4.flac");
    }

    #[test]
    fn test_stem_url_host_allowed() {
        let ok = reqwest::Url::parse("https://mvsep.com/storage/vocals.wav").unwrap();
        assert!(stem_url_host_allowed(&ok));
        let sub = reqwest::Url::parse("https://cdn.mvsep.com/x.wav").unwrap();
        assert!(stem_url_host_allowed(&sub));
        // Wrong scheme, lookalike hosts, and internal targets are rejected.
        for bad in [
            "http://mvsep.com/x.wav",
            "https://mvsep.com.evil.com/x.wav",
            "https://evil-mvsep.com/x.wav",
            "https://169.254.169.254/latest/meta-data/",
            "https://example.com/x.wav",
        ] {
            let url = reqwest::Url::parse(bad).unwrap();
            assert!(!stem_url_host_allowed(&url), "{bad} should be rejected");
        }
    }
}

