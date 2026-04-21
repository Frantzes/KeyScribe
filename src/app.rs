use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bincode::Options;
use directories::ProjectDirs;
use eframe::egui;
#[cfg(not(feature = "desktop-ui"))]
use egui_phosphor::regular::GEAR;
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Polygon, VLine};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "desktop-ui")]
use rfd::FileDialog;

use crate::analysis::{
    analyze_with_full_pipeline, detect_note_probabilities, detect_note_probabilities_cqt_preview,
    PIANO_HIGH_MIDI, PIANO_LOW_MIDI,
};
use crate::audio_io::{
    load_audio_file_streaming, load_audio_preview_chunk, AudioData, AudioPreviewChunk,
    StreamingAudioEvent,
};
use crate::core::processing::build_waveform_for_processed;
use crate::dsp::{apply_speed_and_pitch, apply_speed_and_pitch_interleaved};
use crate::playback::{available_output_devices, AudioEngine, AudioOutputDeviceOption};
use crate::theme::{apply_brand_theme, ACCENT_PURPLE, ERROR_RED};
use crate::ui::keyboard::{
    draw_piano_view, draw_probability_pane, keyboard_white_key_width, MIN_PIANO_KEY_HEIGHT,
    MIN_PROBABILITY_STRIP_HEIGHT, PIANO_ZOOM_MAX, PIANO_ZOOM_MIN, WHITE_KEY_LENGTH_TO_WIDTH,
};
use crate::ui::utils::{accent_soft, color_to_hex, parse_hex_color, push_recent_color};
#[cfg(not(feature = "desktop-ui"))]
use crate::ui::widgets::icon_button;

mod cache;
mod loading;
mod media_controls;
mod playback;
mod processing;
mod runtime;
mod top_controls;
mod update;
use media_controls::{draw_media_controls, media_controls_height_for_width, setting_toggle_row};

const STATE_FILE_NAME: &str = ".keyscribe_state.json";
const LEGACY_STATE_FILE_NAME: &str = ".transcriber_state.json";
const MAX_STATE_FILE_BYTES: u64 = 256 * 1024;
const PROBABILITY_UPDATE_INTERVAL: Duration = Duration::from_millis(16);
const FFT_TIMELINE_STEP_SEC: f32 = 0.05;
const FFT_WINDOW_SIZE: usize = 4096;
const LOOP_MIN_DURATION_SEC: f32 = 0.01;
const SEEK_STEP_SEC: f32 = 5.0;
const PARAM_UPDATE_PREVIEW_SEC: f32 = 8.0;
const PARAM_UPDATE_LIVE_DEBOUNCE: Duration = Duration::from_millis(120);
const ANALYSIS_CACHE_DIR_NAME: &str = ".keyscribe_cache";
const LEGACY_ANALYSIS_CACHE_DIR_NAME: &str = ".transcriber_cache";
const ANALYSIS_CACHE_LIBRARY_DIR_NAME: &str = "library";
const ANALYSIS_CACHE_VERSION: u32 = 4;
const ANALYSIS_CACHE_ZSTD_LEVEL: i32 = 9;
const ANALYSIS_CACHE_MAX_COMPRESSED_BYTES: usize = 128 * 1024 * 1024;
const ANALYSIS_CACHE_MAX_DECOMPRESSED_BYTES: usize = 1024 * 1024 * 1024;
const ANALYSIS_CACHE_MAX_DECOMPRESS_RATIO: usize = 64;
const ANALYSIS_CACHE_MAX_TIMELINE_FRAMES: usize = 120_000;
const ANALYSIS_CACHE_MAX_WAVEFORM_POINTS: usize = 64_000;
const STARTUP_MAX_AUDIO_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const ALBUM_ART_MAX_BYTES: usize = 32 * 1024 * 1024;
const ALBUM_ART_MAX_DIMENSION: usize = 4096;
const AUDIO_STREAM_CHUNK_SAMPLES: usize = 44_100;
const AUDIO_LOADING_MAX_EVENTS_PER_FRAME: usize = 2;
const STREAMING_WAVEFORM_REBUILD_INTERVAL: Duration = Duration::from_millis(220);
const STREAMING_WAVEFORM_REBUILD_SAMPLE_DELTA: usize = AUDIO_STREAM_CHUNK_SAMPLES * 2;
const LOADING_PREVIEW_CACHE_CHUNK_SEC: f32 = 16.0;
const LOADING_PREVIEW_CACHE_STRIDE_SEC: f32 = 8.0;
const LOADING_PREVIEW_CACHE_MAX_ENTRIES: usize = 8;
const ACTIVE_REPAINT_INTERVAL: Duration = Duration::from_millis(16);
const IDLE_REPAINT_INTERVAL: Duration = Duration::from_millis(80);
const TRANSCRIBE_PROGRESS_LOADING_WEIGHT: f32 = 0.35;
const TRANSCRIBE_PROGRESS_MAX_BEFORE_DONE: f32 = 0.99;
const MAX_RECENT_FILES: usize = 10;
const UI_VSPACE_TIGHT: f32 = 2.0;
const UI_VSPACE_COMPACT: f32 = 4.0;
const UI_VSPACE_MEDIUM: f32 = 8.0;
const UI_STACK_VSPACE: f32 = 4.0;
const UI_SEPARATOR_STROKE_WIDTH: f32 = 1.0;
const PRESET_HIGHLIGHT_COLORS: [(&str, egui::Color32); 8] = [
    ("Purple", egui::Color32::from_rgb(148, 106, 255)),
    ("Sky", egui::Color32::from_rgb(72, 162, 255)),
    ("Mint", egui::Color32::from_rgb(56, 204, 142)),
    ("Rose", egui::Color32::from_rgb(248, 112, 134)),
    ("Gold", egui::Color32::from_rgb(238, 190, 73)),
    ("Lime", egui::Color32::from_rgb(162, 216, 58)),
    ("Cyan", egui::Color32::from_rgb(57, 205, 217)),
    ("Violet", egui::Color32::from_rgb(170, 134, 255)),
];

fn default_preprocess_audio() -> bool {
    true
}

fn ui_separator_stroke(ui: &egui::Ui) -> egui::Stroke {
    egui::Stroke::new(
        UI_SEPARATOR_STROKE_WIDTH,
        ui.visuals().widgets.noninteractive.bg_stroke.color,
    )
}

fn draw_horizontal_separator(ui: &mut egui::Ui, horizontal_bleed: f32) {
    let width = ui.available_width().max(0.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
    let bleed = horizontal_bleed.max(0.0);
    let clip = if bleed > 0.0 {
        ui.ctx().input(|i| i.screen_rect())
    } else {
        ui.clip_rect()
    };
    let painter = if bleed > 0.0 {
        ui.painter().with_clip_rect(clip)
    } else {
        ui.painter().clone()
    };
    let x_range = if bleed > 0.0 {
        egui::Rangef::new(clip.left(), clip.right())
    } else {
        egui::Rangef::new(
            (rect.left() - bleed).max(clip.left()),
            (rect.right() + bleed).min(clip.right()),
        )
    };
    // Snap to full pixel to avoid anti-alias variance between panels.
    let y = rect.center().y.round();
    painter.hline(x_range, y, ui_separator_stroke(ui));
}

fn default_key_color_sensitivity() -> f32 {
    0.46
}

fn default_playback_volume() -> f32 {
    0.8
}

fn default_audio_quality_mode() -> AudioQualityMode {
    AudioQualityMode::Balanced
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AudioQualityMode {
    Draft,
    Balanced,
    Studio,
}

impl AudioQualityMode {
    fn label(self) -> &'static str {
        match self {
            Self::Draft => "Draft (fastest)",
            Self::Balanced => "Balanced",
            Self::Studio => "Studio (highest detail)",
        }
    }

    fn fft_window_size(self) -> usize {
        match self {
            Self::Draft => 2048,
            Self::Balanced => FFT_WINDOW_SIZE,
            Self::Studio => 8192,
        }
    }

    fn waveform_points(self) -> usize {
        match self {
            Self::Draft => 3000,
            Self::Balanced => 6000,
            Self::Studio => 12000,
        }
    }

    fn cache_code(self) -> u8 {
        match self {
            Self::Draft => 0,
            Self::Balanced => 1,
            Self::Studio => 2,
        }
    }
}

fn default_dark_mode() -> bool {
    true
}

fn default_use_cqt_analysis() -> bool {
    true
}

fn default_highlight_hex() -> String {
    "#946AFF".to_string()
}

fn normalize_recent_file_key(path: &Path) -> String {
    path.to_string_lossy().to_lowercase()
}

fn push_recent_file_path(recent_paths: &mut Vec<PathBuf>, path: &Path) {
    let key = normalize_recent_file_key(path);
    recent_paths.retain(|existing| normalize_recent_file_key(existing.as_path()) != key);
    recent_paths.insert(0, path.to_path_buf());
    recent_paths.truncate(MAX_RECENT_FILES);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    last_file: Option<PathBuf>,
    #[serde(default)]
    recent_files: Vec<PathBuf>,
    selected_time_sec: f32,
    speed: f32,
    pitch_semitones: f32,
    #[serde(default = "default_key_color_sensitivity")]
    key_color_sensitivity: f32,
    piano_zoom: f32,
    piano_key_height: f32,
    waveform_panel_height: f32,
    probability_panel_height: f32,
    piano_panel_height: f32,
    show_note_hist_window: bool,
    #[serde(default = "default_use_cqt_analysis")]
    use_cqt_analysis: bool,
    #[serde(default = "default_preprocess_audio")]
    preprocess_audio: bool,
    #[serde(default = "default_playback_volume")]
    playback_volume: f32,
    #[serde(default = "default_audio_quality_mode")]
    audio_quality_mode: AudioQualityMode,
    #[serde(default)]
    audio_output_device_id: Option<String>,
    #[serde(default)]
    loop_enabled: bool,
    #[serde(default = "default_dark_mode")]
    dark_mode: bool,
    #[serde(default = "default_highlight_hex")]
    highlight_hex: String,
    #[serde(default)]
    recent_highlight_hex: Vec<String>,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            last_file: None,
            recent_files: Vec::new(),
            selected_time_sec: 0.0,
            speed: 1.0,
            pitch_semitones: 0.0,
            key_color_sensitivity: default_key_color_sensitivity(),
            piano_zoom: 1.0,
            piano_key_height: 72.0,
            waveform_panel_height: 320.0,
            probability_panel_height: 130.0,
            piano_panel_height: 170.0,
            show_note_hist_window: true,
            use_cqt_analysis: default_use_cqt_analysis(),
            preprocess_audio: true,
            playback_volume: 0.8,
            audio_quality_mode: AudioQualityMode::Balanced,
            audio_output_device_id: None,
            loop_enabled: false,
            dark_mode: true,
            highlight_hex: default_highlight_hex(),
            recent_highlight_hex: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct AnalysisCacheBlob {
    cache_version: u32,
    sample_rate: u32,
    raw_sample_len: usize,
    audio_quality_mode_code: u8,
    speed_bits: u32,
    pitch_bits: u32,
    use_cqt_analysis: bool,
    preprocess_audio: bool,
    processed_samples_len: usize,
    processed_samples_shuffled_bytes: Option<Vec<u8>>,
    #[serde(default)]
    waveform_points: Option<Vec<[f32; 2]>>,
    base_note_timeline: Vec<Vec<f32>>,
    base_note_timeline_step_sec: f32,
}

#[derive(Serialize)]
struct AnalysisCacheSnapshot<'a> {
    cache_version: u32,
    sample_rate: u32,
    raw_sample_len: usize,
    audio_quality_mode_code: u8,
    speed_bits: u32,
    pitch_bits: u32,
    use_cqt_analysis: bool,
    preprocess_audio: bool,
    processed_samples_len: usize,
    processed_samples_shuffled_bytes: Option<&'a [u8]>,
    waveform_points: Option<&'a [[f32; 2]]>,
    base_note_timeline: &'a [Vec<f32>],
    base_note_timeline_step_sec: f32,
}

#[derive(Clone, Copy, Debug)]
enum CacheBlobDecodeFailure {
    Decompress,
    Deserialize,
}

fn app_portable_base_dir() -> PathBuf {
    // Portable build behavior: persist all runtime data beside the executable.
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.to_path_buf()))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn app_project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("com", "Frantzes", "KeyScribe")
}

#[cfg(target_os = "macos")]
fn is_running_from_macos_app_bundle() -> bool {
    std::env::current_exe()
        .ok()
        .map(|exe| exe.to_string_lossy().contains(".app/Contents/MacOS/"))
        .unwrap_or(false)
}

#[cfg(not(target_os = "macos"))]
fn is_running_from_macos_app_bundle() -> bool {
    false
}

fn app_data_dir() -> PathBuf {
    if is_running_from_macos_app_bundle() {
        if let Some(project_dirs) = app_project_dirs() {
            return project_dirs.data_local_dir().to_path_buf();
        }
    }

    app_portable_base_dir()
}

fn app_cache_base_dir() -> PathBuf {
    if is_running_from_macos_app_bundle() {
        if let Some(project_dirs) = app_project_dirs() {
            return project_dirs.cache_dir().to_path_buf();
        }
    }

    app_portable_base_dir()
}

fn ensure_parent_dir(path: &Path) -> bool {
    path.parent()
        .map(|parent| fs::create_dir_all(parent).is_ok())
        .unwrap_or(true)
}

fn is_supported_audio_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };

    matches!(
        ext.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac"
    )
}

#[cfg(windows)]
fn is_windows_network_path(path: &Path) -> bool {
    use std::path::{Component, Prefix};

    matches!(
        path.components().next(),
        Some(Component::Prefix(prefix))
            if matches!(
                prefix.kind(),
                Prefix::UNC(_, _) | Prefix::VerbatimUNC(_, _) | Prefix::DeviceNS(_)
            )
    )
}

#[cfg(not(windows))]
fn is_windows_network_path(_path: &Path) -> bool {
    false
}

fn is_safe_startup_audio_path(path: &Path) -> bool {
    if !path.is_absolute() || is_windows_network_path(path) {
        return false;
    }
    if !is_supported_audio_extension(path) || !path.is_file() {
        return false;
    }

    match fs::metadata(path) {
        Ok(meta) => meta.len() <= STARTUP_MAX_AUDIO_FILE_BYTES,
        Err(_) => false,
    }
}

fn validate_cached_note_timeline(note_timeline: &[Vec<f32>]) -> bool {
    if note_timeline.len() > ANALYSIS_CACHE_MAX_TIMELINE_FRAMES {
        return false;
    }

    let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
    note_timeline
        .iter()
        .all(|frame| frame.len() == note_count && frame.iter().all(|value| value.is_finite()))
}

fn validate_cached_waveform_points(points: &[[f32; 2]]) -> bool {
    if points.is_empty() {
        return false;
    }
    if points.len() > ANALYSIS_CACHE_MAX_WAVEFORM_POINTS {
        return false;
    }

    let mut prev_x = f64::NEG_INFINITY;
    for &[x, y] in points {
        if !x.is_finite() || !y.is_finite() {
            return false;
        }

        let x64 = x as f64;
        if x64 < prev_x {
            return false;
        }
        prev_x = x64;
    }

    true
}

fn pack_waveform_points(points: &[[f64; 2]]) -> Vec<[f32; 2]> {
    points
        .iter()
        .filter_map(|&[x, y]| {
            if !x.is_finite() || !y.is_finite() {
                return None;
            }

            Some([x as f32, y as f32])
        })
        .take(ANALYSIS_CACHE_MAX_WAVEFORM_POINTS)
        .collect()
}

fn unpack_waveform_points(points: &[[f32; 2]]) -> Vec<[f64; 2]> {
    points.iter().map(|&[x, y]| [x as f64, y as f64]).collect()
}

fn analysis_cache_dir() -> PathBuf {
    app_cache_base_dir().join(ANALYSIS_CACHE_DIR_NAME)
}

fn analysis_cache_library_dir() -> PathBuf {
    analysis_cache_dir().join(ANALYSIS_CACHE_LIBRARY_DIR_NAME)
}

fn analysis_cache_library_dirs() -> Vec<PathBuf> {
    fn push_unique_dir(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
        if !dirs.iter().any(|existing| existing == &dir) {
            dirs.push(dir);
        }
    }

    let primary = analysis_cache_library_dir();
    let legacy_primary = app_cache_base_dir()
        .join(LEGACY_ANALYSIS_CACHE_DIR_NAME)
        .join(ANALYSIS_CACHE_LIBRARY_DIR_NAME);

    let mut dirs = Vec::new();
    push_unique_dir(&mut dirs, primary);
    push_unique_dir(&mut dirs, legacy_primary);

    // Fallback to workspace-relative legacy cache location so entries created
    // from a different runtime context remain reusable.
    if let Ok(cwd) = std::env::current_dir() {
        let workspace_current = cwd
            .join(ANALYSIS_CACHE_DIR_NAME)
            .join(ANALYSIS_CACHE_LIBRARY_DIR_NAME);

        let workspace_legacy = cwd
            .join(LEGACY_ANALYSIS_CACHE_DIR_NAME)
            .join(ANALYSIS_CACHE_LIBRARY_DIR_NAME);

        push_unique_dir(&mut dirs, workspace_current);
        push_unique_dir(&mut dirs, workspace_legacy);
    }

    dirs
}

fn analysis_cache_primary_file_path(song_hash: &str, variant_key: &str) -> PathBuf {
    analysis_cache_library_dir()
        .join(song_hash)
        .join(format!("{variant_key}.bin.zst"))
}

fn analysis_cache_candidate_file_paths(song_hash: &str, variant_key: &str) -> Vec<PathBuf> {
    analysis_cache_library_dirs()
        .into_iter()
        .map(|dir| dir.join(song_hash).join(format!("{variant_key}.bin.zst")))
        .collect()
}

fn analysis_cache_song_file_paths(song_hash: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();

    for dir in analysis_cache_library_dirs() {
        let song_dir = dir.join(song_hash);
        let Ok(entries) = fs::read_dir(song_dir) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let is_cache_blob = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("zst"))
                .unwrap_or(false);
            if is_cache_blob && !out.iter().any(|existing| existing == &path) {
                out.push(path);
            }
        }
    }

    out
}

fn compute_file_hash(path: &Path) -> Option<String> {
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = blake3::Hasher::new();
    let mut chunk = vec![0u8; 1024 * 1024];

    loop {
        let read = reader.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }

    Some(hasher.finalize().to_hex().to_string())
}

fn compute_audio_content_hash(sample_rate: u32, samples: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&sample_rate.to_le_bytes());
    hasher.update(&(samples.len() as u64).to_le_bytes());
    for sample in samples {
        hasher.update(&sample.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn analysis_cache_variant_key(
    sample_rate: u32,
    raw_sample_len: usize,
    audio_quality_mode: AudioQualityMode,
    speed: f32,
    pitch_semitones: f32,
    use_cqt_analysis: bool,
    preprocess_audio: bool,
) -> String {
    let mut hash = 14695981039346656037u64;
    hash = fnv1a64_update(hash, &sample_rate.to_le_bytes());
    hash = fnv1a64_update(hash, &(raw_sample_len as u64).to_le_bytes());
    hash = fnv1a64_update(hash, &[audio_quality_mode.cache_code()]);
    hash = fnv1a64_update(hash, &speed.to_bits().to_le_bytes());
    hash = fnv1a64_update(hash, &pitch_semitones.to_bits().to_le_bytes());
    hash = fnv1a64_update(hash, &[if use_cqt_analysis { 1 } else { 0 }]);
    hash = fnv1a64_update(hash, &[if preprocess_audio { 1 } else { 0 }]);
    hash = fnv1a64_update(hash, &ANALYSIS_CACHE_VERSION.to_le_bytes());

    format!("{hash:016x}")
}

fn analysis_cache_decompress_budget(compressed_len: usize) -> usize {
    compressed_len
        .saturating_mul(ANALYSIS_CACHE_MAX_DECOMPRESS_RATIO)
        .clamp(
            ANALYSIS_CACHE_MAX_COMPRESSED_BYTES,
            ANALYSIS_CACHE_MAX_DECOMPRESSED_BYTES,
        )
}

fn deserialize_analysis_cache_blob(payload: &[u8]) -> Option<AnalysisCacheBlob> {
    let decode_limit = (payload.len().min(ANALYSIS_CACHE_MAX_DECOMPRESSED_BYTES)) as u64;

    bincode::DefaultOptions::new()
        .with_limit(decode_limit)
        .deserialize::<AnalysisCacheBlob>(payload)
        .ok()
        .or_else(|| {
            bincode::DefaultOptions::new()
                .with_limit(decode_limit)
                .allow_trailing_bytes()
                .deserialize::<AnalysisCacheBlob>(payload)
                .ok()
        })
        .or_else(|| {
            bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .with_limit(decode_limit)
                .deserialize::<AnalysisCacheBlob>(payload)
                .ok()
        })
        .or_else(|| {
            bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .with_limit(decode_limit)
                .allow_trailing_bytes()
                .deserialize::<AnalysisCacheBlob>(payload)
                .ok()
        })
        .or_else(|| {
            bincode::DefaultOptions::new()
                .with_varint_encoding()
                .with_limit(decode_limit)
                .deserialize::<AnalysisCacheBlob>(payload)
                .ok()
        })
        .or_else(|| {
            bincode::DefaultOptions::new()
                .with_varint_encoding()
                .with_limit(decode_limit)
                .allow_trailing_bytes()
                .deserialize::<AnalysisCacheBlob>(payload)
                .ok()
        })
}

fn decode_analysis_cache_blob(bytes: &[u8]) -> Result<AnalysisCacheBlob, CacheBlobDecodeFailure> {
    if bytes.is_empty() || bytes.len() > ANALYSIS_CACHE_MAX_COMPRESSED_BYTES {
        return Err(CacheBlobDecodeFailure::Decompress);
    }

    let decompress_budget = analysis_cache_decompress_budget(bytes.len());
    if let Ok(payload) = zstd::bulk::decompress(bytes, decompress_budget) {
        return deserialize_analysis_cache_blob(payload.as_slice())
            .ok_or(CacheBlobDecodeFailure::Deserialize);
    }

    // Backward compatibility: some older builds may have written raw bincode
    // bytes with a .zst extension when compression failed.
    deserialize_analysis_cache_blob(bytes).ok_or(CacheBlobDecodeFailure::Decompress)
}

fn speed_pitch_is_identity(speed: f32, pitch_semitones: f32) -> bool {
    (speed.clamp(0.25, 4.0) - 1.0).abs() < 1.0e-4 && pitch_semitones.abs() < 1.0e-4
}

fn shuffle_f32_bytes(samples: &[f32]) -> Vec<u8> {
    let len = samples.len();
    let mut shuffled = vec![0u8; len * 4];
    for (i, sample) in samples.iter().enumerate() {
        let bytes = sample.to_le_bytes();
        shuffled[i] = bytes[0];
        shuffled[len + i] = bytes[1];
        shuffled[len * 2 + i] = bytes[2];
        shuffled[len * 3 + i] = bytes[3];
    }
    shuffled
}

fn unshuffle_f32_bytes(shuffled: &[u8], sample_len: usize) -> Option<Vec<f32>> {
    if shuffled.len() != sample_len.saturating_mul(4) {
        return None;
    }

    let mut samples = Vec::with_capacity(sample_len);
    for i in 0..sample_len {
        let bytes = [
            shuffled[i],
            shuffled[sample_len + i],
            shuffled[sample_len * 2 + i],
            shuffled[sample_len * 3 + i],
        ];
        samples.push(f32::from_le_bytes(bytes));
    }
    Some(samples)
}

fn state_file_path() -> PathBuf {
    app_data_dir().join(STATE_FILE_NAME)
}

fn legacy_state_file_path() -> PathBuf {
    app_data_dir().join(LEGACY_STATE_FILE_NAME)
}

fn load_persisted_state() -> PersistedState {
    for path in [state_file_path(), legacy_state_file_path()] {
        let Ok(meta) = fs::metadata(&path) else {
            continue;
        };
        if meta.len() > MAX_STATE_FILE_BYTES {
            continue;
        }

        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };

        if let Ok(state) = serde_json::from_str::<PersistedState>(&raw) {
            return state;
        }
    }

    PersistedState::default()
}

pub struct KeyScribeApp {
    loaded_path: Option<PathBuf>,
    loaded_audio_hash: Option<String>,
    audio_raw: Option<AudioData>,
    processed_samples: Vec<f32>,
    processed_playback_samples: Vec<f32>,
    processed_playback_channels: u16,
    waveform: Vec<[f64; 2]>,
    waveform_version: u64,
    loop_waveform_cache_version: u64,
    loop_waveform_cache_selection: Option<(f32, f32)>,
    loop_waveform_cache_pre: Vec<[f64; 2]>,
    loop_waveform_cache_mid: Vec<[f64; 2]>,
    loop_waveform_cache_post: Vec<[f64; 2]>,
    note_timeline: Arc<Vec<Vec<f32>>>,
    note_timeline_step_sec: f32,
    base_note_timeline: Arc<Vec<Vec<f32>>>,
    base_note_timeline_step_sec: f32,
    note_probs: Vec<f32>,
    note_probs_smoothed: Vec<f32>,
    selected_time_sec: f32,
    speed: f32,
    pitch_semitones: f32,
    key_color_sensitivity: f32,
    piano_zoom: f32,
    piano_key_height: f32,
    waveform_panel_height: f32,
    probability_panel_height: f32,
    piano_panel_height: f32,
    piano_panel_height_needs_init: bool,
    piano_scroll_px: f32,
    piano_has_focus: bool,
    piano_drag_last_x: Option<f32>,
    last_error: Option<String>,
    engine: Option<AudioEngine>,
    processing_rx: Option<Receiver<ProcessingResult>>,
    processing_epoch: Arc<AtomicU64>,
    active_rebuild_mode: RebuildMode,
    active_job_id: Option<u64>,
    next_job_id: u64,
    is_processing: bool,
    processing_started_at: Option<Instant>,
    processing_estimated_total_sec: f32,
    processing_audio_duration_sec: f32,
    analysis_seconds_per_audio_second_ema: Option<f32>,
    cache_status_message: Option<String>,
    cache_status_message_at: Option<Instant>,
    cache_precheck_done: bool,
    loading_cache_timeline_preloaded: bool,
    loading_cache_waveform_preloaded: bool,
    pending_param_change: bool,
    last_param_change_at: Option<Instant>,
    queued_param_update: bool,
    restart_playback_after_processing: bool,
    last_prob_update: Instant,
    show_note_hist_window: bool,
    playback_volume: f32,
    audio_quality_mode: AudioQualityMode,
    audio_output_device_id: Option<String>,
    audio_output_devices: Vec<AudioOutputDeviceOption>,
    loop_enabled: bool,
    dark_mode: bool,
    highlight_color: egui::Color32,
    custom_rgb: [u8; 3],
    highlight_hex_input: String,
    recent_file_paths: Vec<PathBuf>,
    recent_highlight_hex: Vec<String>,
    last_state_save_at: Instant,
    waveform_reset_view: bool,
    loop_selection: Option<(f32, f32)>,
    drag_select_anchor_sec: Option<f32>,
    loop_playback_enabled: bool,
    playing_preview_buffer: bool,
    live_stream_playback: bool,
    use_cqt_analysis: bool,
    preprocess_audio: bool,
    album_art_texture: Option<egui::TextureHandle>,
    startup_min_window_size_locked: bool,
    audio_loading_rx: Option<Receiver<StreamingAudioEvent>>,
    audio_loading_cancel: Option<Arc<AtomicBool>>,
    is_audio_loading: bool,
    loading_sample_rate: u32,
    loading_total_samples: Option<usize>,
    loading_decoded_samples: usize,
    loading_last_waveform_rebuild_at: Option<Instant>,
    loading_last_waveform_rebuild_samples: usize,
    loading_raw_samples: Vec<f32>,
    loading_raw_samples_interleaved: Vec<f32>,
    loading_source_channels: u16,
    loading_provisional_timeline: Vec<Vec<f32>>,
    loading_next_transcribe_time_sec: f32,
    loading_timeline_frames_pending_sync: usize,
    loading_preview_cache: Vec<LoadingPreviewCacheEntry>,
    touch_loop_select_mode: bool,
    manual_import_path: String,
    mobile_ui_tweaks_applied: bool,
    show_shortcuts_help_modal: bool,
}

struct ProcessingResult {
    job_id: u64,
    mode: RebuildMode,
    cache_lookup_hit: Option<bool>,
    source_hash: Option<String>,
    processed_samples: Vec<f32>,
    processed_playback_samples: Vec<f32>,
    processed_playback_channels: u16,
    waveform: Vec<[f64; 2]>,
    note_timeline: Arc<Vec<Vec<f32>>>,
    note_timeline_step_sec: f32,
    base_note_timeline: Arc<Vec<Vec<f32>>>,
    base_note_timeline_step_sec: f32,
    analysis_error: Option<String>,
    preview_playback: Option<PreviewPlayback>,
}

struct PreviewPlayback {
    samples: Vec<f32>,
    channels: u16,
    timeline_start_sec: f32,
}

struct LoadingPreviewCacheEntry {
    source_key: String,
    chunk_start_sec: f32,
    sample_rate: u32,
    channels: u16,
    samples_interleaved: Arc<Vec<f32>>,
    last_used_at: Instant,
}

#[derive(Default)]
struct CachePrecheckDiagnostics {
    total_candidates: usize,
    existing_files: usize,
    parsed_blobs: usize,
    read_failures: usize,
    decompress_failures: usize,
    deserialize_failures: usize,
    shared_param_mismatches: usize,
    strict_len_mismatches: usize,
    invalid_timeline_blobs: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RebuildMode {
    Full,
    ParametersOnly,
    ParametersPreview,
}

impl KeyScribeApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let persisted = load_persisted_state();
        let startup_path = persisted.last_file.clone();
        let mut recent_file_paths = persisted.recent_files.clone();
        if recent_file_paths.is_empty() {
            if let Some(path) = startup_path.as_ref() {
                push_recent_file_path(&mut recent_file_paths, path.as_path());
            }
        }
        let highlight_color = parse_hex_color(&persisted.highlight_hex).unwrap_or(ACCENT_PURPLE);
        apply_brand_theme(&_cc.egui_ctx, persisted.dark_mode, highlight_color);

        let mut app = Self {
            loaded_path: None,
            loaded_audio_hash: None,
            audio_raw: None,
            processed_samples: Vec::new(),
            processed_playback_samples: Vec::new(),
            processed_playback_channels: 1,
            waveform: Vec::new(),
            waveform_version: 0,
            loop_waveform_cache_version: u64::MAX,
            loop_waveform_cache_selection: None,
            loop_waveform_cache_pre: Vec::new(),
            loop_waveform_cache_mid: Vec::new(),
            loop_waveform_cache_post: Vec::new(),
            note_timeline: Arc::new(Vec::new()),
            note_timeline_step_sec: 0.0,
            base_note_timeline: Arc::new(Vec::new()),
            base_note_timeline_step_sec: 0.0,
            note_probs: vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize],
            note_probs_smoothed: vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize],
            selected_time_sec: persisted.selected_time_sec.max(0.0),
            speed: persisted.speed.clamp(0.5, 2.0),
            pitch_semitones: persisted.pitch_semitones.clamp(-12.0, 12.0),
            key_color_sensitivity: persisted.key_color_sensitivity.clamp(0.0, 2.0),
            piano_zoom: persisted.piano_zoom.clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX),
            piano_key_height: persisted
                .piano_key_height
                .clamp(MIN_PIANO_KEY_HEIGHT, 220.0),
            waveform_panel_height: persisted.waveform_panel_height.clamp(120.0, 5000.0),
            probability_panel_height: persisted.probability_panel_height.clamp(0.0, 5000.0),
            piano_panel_height: persisted.piano_panel_height.clamp(80.0, 5000.0),
            piano_panel_height_needs_init: true,
            piano_scroll_px: 0.0,
            piano_has_focus: false,
            piano_drag_last_x: None,
            last_error: None,
            engine: AudioEngine::new_with_output_device(
                persisted.audio_output_device_id.as_deref(),
            )
            .ok(),
            processing_rx: None,
            processing_epoch: Arc::new(AtomicU64::new(0)),
            active_rebuild_mode: RebuildMode::Full,
            active_job_id: None,
            next_job_id: 1,
            is_processing: false,
            processing_started_at: None,
            processing_estimated_total_sec: 0.0,
            processing_audio_duration_sec: 0.0,
            analysis_seconds_per_audio_second_ema: None,
            cache_status_message: None,
            cache_status_message_at: None,
            cache_precheck_done: false,
            loading_cache_timeline_preloaded: false,
            loading_cache_waveform_preloaded: false,
            pending_param_change: false,
            last_param_change_at: None,
            queued_param_update: false,
            restart_playback_after_processing: false,
            last_prob_update: Instant::now(),
            show_note_hist_window: persisted.show_note_hist_window,
            playback_volume: persisted.playback_volume.clamp(0.0, 1.5),
            audio_quality_mode: persisted.audio_quality_mode,
            audio_output_device_id: persisted.audio_output_device_id.clone(),
            audio_output_devices: Vec::new(),
            loop_enabled: persisted.loop_enabled,
            dark_mode: persisted.dark_mode,
            highlight_color,
            custom_rgb: [
                highlight_color.r(),
                highlight_color.g(),
                highlight_color.b(),
            ],
            highlight_hex_input: color_to_hex(highlight_color),
            recent_file_paths,
            recent_highlight_hex: persisted.recent_highlight_hex,
            last_state_save_at: Instant::now(),
            waveform_reset_view: true,
            loop_selection: None,
            drag_select_anchor_sec: None,
            loop_playback_enabled: false,
            playing_preview_buffer: false,
            live_stream_playback: false,
            use_cqt_analysis: persisted.use_cqt_analysis,
            preprocess_audio: persisted.preprocess_audio,
            album_art_texture: None,
            startup_min_window_size_locked: false,
            audio_loading_rx: None,
            audio_loading_cancel: None,
            is_audio_loading: false,
            loading_sample_rate: 0,
            loading_total_samples: None,
            loading_decoded_samples: 0,
            loading_last_waveform_rebuild_at: None,
            loading_last_waveform_rebuild_samples: 0,
            loading_raw_samples: Vec::new(),
            loading_raw_samples_interleaved: Vec::new(),
            loading_source_channels: 1,
            loading_provisional_timeline: Vec::new(),
            loading_next_transcribe_time_sec: 0.0,
            loading_timeline_frames_pending_sync: 0,
            loading_preview_cache: Vec::new(),
            touch_loop_select_mode: false,
            manual_import_path: startup_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string())
                .unwrap_or_default(),
            mobile_ui_tweaks_applied: false,
            show_shortcuts_help_modal: false,
        };

        app.refresh_audio_output_devices();
        if let Some(selected) = app.audio_output_device_id.clone() {
            let exists = app.audio_output_devices.iter().any(|d| d.id == selected);
            if !exists {
                app.audio_output_device_id = None;
                app.engine = AudioEngine::new().ok();
            }
        }

        if let Some(engine) = &mut app.engine {
            engine.set_volume(app.playback_volume);
        }

        if let Some(path) = startup_path {
            if is_safe_startup_audio_path(path.as_path()) {
                app.start_audio_loading(path, &_cc.egui_ctx);
            }
        }

        app
    }
}
