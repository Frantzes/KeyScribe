use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bincode::Options;
use eframe::egui;
use egui_phosphor::regular::{DOWNLOAD_SIMPLE, GEAR};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Polygon, VLine};
use rayon::prelude::*;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    analyze_with_full_pipeline, detect_note_probabilities, detect_note_probabilities_cqt,
    waveform_points, PIANO_HIGH_MIDI, PIANO_LOW_MIDI,
};
use crate::audio_io::{load_audio_file, AudioData};
use crate::dsp::apply_speed_and_pitch;
use crate::playback::{available_output_devices, AudioEngine, AudioOutputDeviceOption};
use crate::theme::{apply_brand_theme, ACCENT_ORANGE, ERROR_RED};
use crate::ui::keyboard::{
    draw_piano_view, draw_probability_pane, keyboard_white_key_width, MIN_PIANO_KEY_HEIGHT,
    MIN_PROBABILITY_STRIP_HEIGHT, PIANO_ZOOM_MAX, PIANO_ZOOM_MIN, WHITE_KEY_LENGTH_TO_WIDTH,
};
use crate::ui::utils::{accent_soft, color_to_hex, parse_hex_color, push_recent_color};
use crate::ui::widgets::icon_button;

mod media_controls;
use media_controls::{draw_media_controls, setting_toggle_row};

const STATE_FILE_NAME: &str = ".transcriber_state.json";
const MAX_STATE_FILE_BYTES: u64 = 256 * 1024;
const PROBABILITY_UPDATE_INTERVAL: Duration = Duration::from_millis(16);
const FFT_TIMELINE_STEP_SEC: f32 = 0.05;
const FFT_WINDOW_SIZE: usize = 4096;
const LOOP_MIN_DURATION_SEC: f32 = 0.01;
const SEEK_STEP_SEC: f32 = 5.0;
const PARAM_UPDATE_PREVIEW_SEC: f32 = 8.0;
const PARAM_UPDATE_LIVE_DEBOUNCE: Duration = Duration::from_millis(120);
const ANALYSIS_CACHE_DIR_NAME: &str = ".transcriber_cache";
const ANALYSIS_CACHE_LIBRARY_DIR_NAME: &str = "library";
const ANALYSIS_CACHE_VERSION: u32 = 4;
const ANALYSIS_CACHE_ZSTD_LEVEL: i32 = 9;
const ANALYSIS_CACHE_MAX_COMPRESSED_BYTES: usize = 128 * 1024 * 1024;
const ANALYSIS_CACHE_MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;
const ANALYSIS_CACHE_MAX_TIMELINE_FRAMES: usize = 120_000;
const STARTUP_MAX_AUDIO_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const ALBUM_ART_MAX_BYTES: usize = 32 * 1024 * 1024;
const ALBUM_ART_MAX_DIMENSION: usize = 4096;
const PRESET_HIGHLIGHT_COLORS: [(&str, egui::Color32); 8] = [
    ("Orange", egui::Color32::from_rgb(255, 140, 45)),
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

fn default_key_color_sensitivity() -> f32 {
    0.4
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
    "#FF8C2D".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    last_file: Option<PathBuf>,
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
    base_note_timeline: &'a [Vec<f32>],
    base_note_timeline_step_sec: f32,
}

fn app_portable_base_dir() -> PathBuf {
    // Portable build behavior: persist all runtime data beside the executable.
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.to_path_buf()))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn app_data_dir() -> PathBuf {
    app_portable_base_dir()
}

fn app_cache_base_dir() -> PathBuf {
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

fn analysis_cache_dir() -> PathBuf {
    app_cache_base_dir().join(ANALYSIS_CACHE_DIR_NAME)
}

fn analysis_cache_library_dir() -> PathBuf {
    analysis_cache_dir().join(ANALYSIS_CACHE_LIBRARY_DIR_NAME)
}

fn analysis_cache_file_path(song_hash: &str, variant_key: &str) -> PathBuf {
    analysis_cache_library_dir()
        .join(song_hash)
        .join(format!("{variant_key}.bin.zst"))
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

fn load_persisted_state() -> PersistedState {
    let path = state_file_path();
    if let Ok(meta) = fs::metadata(&path) {
        if meta.len() > MAX_STATE_FILE_BYTES {
            return PersistedState::default();
        }
    }

    let Ok(raw) = fs::read_to_string(path) else {
        return PersistedState::default();
    };

    serde_json::from_str::<PersistedState>(&raw).unwrap_or_default()
}

pub struct TranscriberApp {
    loaded_path: Option<PathBuf>,
    loaded_audio_hash: Option<String>,
    audio_raw: Option<AudioData>,
    processed_samples: Vec<f32>,
    waveform: Vec<[f64; 2]>,
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
    last_error: Option<String>,
    engine: Option<AudioEngine>,
    processing_rx: Option<Receiver<ProcessingResult>>,
    processing_epoch: Arc<AtomicU64>,
    active_rebuild_mode: RebuildMode,
    active_job_id: Option<u64>,
    next_job_id: u64,
    is_processing: bool,
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
    recent_highlight_hex: Vec<String>,
    last_state_save_at: Instant,
    waveform_reset_view: bool,
    loop_selection: Option<(f32, f32)>,
    drag_select_anchor_sec: Option<f32>,
    loop_playback_enabled: bool,
    playing_preview_buffer: bool,
    use_cqt_analysis: bool,
    preprocess_audio: bool,
    album_art_texture: Option<egui::TextureHandle>,
    startup_min_window_size_locked: bool,
}

struct ProcessingResult {
    job_id: u64,
    mode: RebuildMode,
    processed_samples: Vec<f32>,
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
    timeline_start_sec: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RebuildMode {
    Full,
    ParametersOnly,
    ParametersPreview,
}

impl TranscriberApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let persisted = load_persisted_state();
        let highlight_color = parse_hex_color(&persisted.highlight_hex).unwrap_or(ACCENT_ORANGE);
        apply_brand_theme(&_cc.egui_ctx, persisted.dark_mode, highlight_color);

        let mut app = Self {
            loaded_path: None,
            loaded_audio_hash: None,
            audio_raw: None,
            processed_samples: Vec::new(),
            waveform: Vec::new(),
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
            recent_highlight_hex: persisted.recent_highlight_hex,
            last_state_save_at: Instant::now(),
            waveform_reset_view: true,
            loop_selection: None,
            drag_select_anchor_sec: None,
            loop_playback_enabled: false,
            playing_preview_buffer: false,
            use_cqt_analysis: persisted.use_cqt_analysis,
            preprocess_audio: persisted.preprocess_audio,
            album_art_texture: None,
            startup_min_window_size_locked: false,
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

        if let Some(path) = persisted.last_file {
            if is_safe_startup_audio_path(path.as_path()) {
                match load_audio_file(&path) {
                    Ok(audio) => {
                        app.apply_loaded_audio(path.clone(), audio, &_cc.egui_ctx);
                        if !app.try_restore_analysis_cache(&path) {
                            let restore_mode = if app.preprocess_audio {
                                RebuildMode::Full
                            } else {
                                RebuildMode::ParametersOnly
                            };
                            app.request_rebuild(false, restore_mode);
                        }
                    }
                    Err(err) => {
                        app.last_error =
                            Some(format!("Failed to restore previous audio file: {err}"));
                    }
                }
            }
        }

        app
    }

    fn lock_startup_min_window_size_once(&mut self, ctx: &egui::Context) {
        if self.startup_min_window_size_locked {
            return;
        }

        let viewport = ctx.input(|i| i.viewport().clone());
        let Some(inner_rect) = viewport.inner_rect else {
            return;
        };

        let size = inner_rect.size();
        if size.x <= 1.0 || size.y <= 1.0 {
            return;
        }

        let sized_like_fullscreen = viewport
            .monitor_size
            .map(|monitor| size.x >= monitor.x * 0.9 && size.y >= monitor.y * 0.9)
            .unwrap_or(false);

        let should_lock = viewport.maximized.unwrap_or(false)
            || viewport.fullscreen.unwrap_or(false)
            || sized_like_fullscreen;

        if should_lock {
            ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(size));
            self.startup_min_window_size_locked = true;
        }
    }

    fn is_playing(&self) -> bool {
        self.engine
            .as_ref()
            .map(|e| e.is_playing())
            .unwrap_or(false)
    }

    fn current_position_sec(&self) -> f32 {
        self.engine
            .as_ref()
            .map(|e| e.current_position())
            .unwrap_or(0.0)
    }

    fn stop_if_playing(&mut self) -> bool {
        let was_playing = self.is_playing();
        if was_playing {
            self.stop();
        }
        was_playing
    }

    fn request_rebuild_preserving_playback(&mut self) {
        let was_playing = self.stop_if_playing();
        self.request_rebuild(was_playing, RebuildMode::Full);
    }

    fn cancel_active_processing(&mut self) {
        let cancel_epoch = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1);
        self.processing_epoch.store(cancel_epoch, Ordering::Release);
        self.clear_processing_job();
        self.pending_param_change = false;
        self.last_param_change_at = None;
        self.queued_param_update = false;
        self.restart_playback_after_processing = false;
    }

    fn refresh_audio_output_devices(&mut self) {
        self.audio_output_devices = available_output_devices();
        if let Some(selected) = self.audio_output_device_id.as_deref() {
            let exists = self.audio_output_devices.iter().any(|d| d.id == selected);
            if !exists {
                self.audio_output_device_id = None;
            }
        }
    }

    fn apply_audio_output_device_change(&mut self, device_id: Option<String>) {
        if self.audio_output_device_id == device_id {
            return;
        }

        let was_playing = self.is_playing();
        let resume_pos = self.current_position_sec().min(self.source_duration());
        self.stop();

        match AudioEngine::new_with_output_device(device_id.as_deref()) {
            Ok(mut engine) => {
                engine.set_volume(self.playback_volume);
                self.engine = Some(engine);
                self.audio_output_device_id = device_id;
                self.last_error = None;

                if was_playing && !self.processed_samples.is_empty() {
                    self.selected_time_sec = resume_pos;
                    self.play_from_selected();
                }
            }
            Err(err) => {
                self.last_error = Some(format!("Audio device error: {err}"));
                self.engine = AudioEngine::new().ok();
                if let Some(engine) = &mut self.engine {
                    engine.set_volume(self.playback_volume);
                }
                self.audio_output_device_id = None;
            }
        }
    }

    fn try_restore_analysis_cache(&mut self, source_path: &Path) -> bool {
        let Some(audio) = &self.audio_raw else {
            return false;
        };
        let sample_rate = audio.sample_rate;
        let raw_sample_len = audio.samples_mono.len();
        let raw_samples = Arc::clone(&audio.samples_mono);

        let song_hash = if let Some(hash) = self.loaded_audio_hash.clone() {
            hash
        } else {
            let Some(hash) = compute_file_hash(source_path) else {
                return false;
            };
            self.loaded_audio_hash = Some(hash.clone());
            hash
        };

        let variant_key = analysis_cache_variant_key(
            sample_rate,
            raw_sample_len,
            self.audio_quality_mode,
            self.speed,
            self.pitch_semitones,
            self.use_cqt_analysis,
            self.preprocess_audio,
        );
        let cache_path = analysis_cache_file_path(&song_hash, &variant_key);

        let Ok(bytes) = fs::read(cache_path) else {
            return false;
        };
        if bytes.is_empty() || bytes.len() > ANALYSIS_CACHE_MAX_COMPRESSED_BYTES {
            return false;
        }

        let Ok(payload) = zstd::bulk::decompress(&bytes, ANALYSIS_CACHE_MAX_DECOMPRESSED_BYTES)
        else {
            return false;
        };
        let options = bincode::options().with_limit(ANALYSIS_CACHE_MAX_DECOMPRESSED_BYTES as u64);
        let Ok(cache) = options.deserialize::<AnalysisCacheBlob>(&payload) else {
            return false;
        };

        if !cache.base_note_timeline_step_sec.is_finite()
            || !validate_cached_note_timeline(cache.base_note_timeline.as_slice())
        {
            return false;
        }

        let expected_speed_bits = self.speed.to_bits();
        let expected_pitch_bits = self.pitch_semitones.to_bits();
        let cache_matches = cache.cache_version == ANALYSIS_CACHE_VERSION
            && cache.sample_rate == sample_rate
            && cache.raw_sample_len == raw_sample_len
            && cache.audio_quality_mode_code == self.audio_quality_mode.cache_code()
            && cache.speed_bits == expected_speed_bits
            && cache.pitch_bits == expected_pitch_bits
            && cache.use_cqt_analysis == self.use_cqt_analysis
            && cache.preprocess_audio == self.preprocess_audio;

        if !cache_matches {
            return false;
        }

        let processed_samples = if let Some(shuffled) = cache.processed_samples_shuffled_bytes {
            let max_processed_len = raw_sample_len.saturating_mul(8);
            if cache.processed_samples_len > max_processed_len {
                return false;
            }
            if cache.processed_samples_len == 0 {
                return false;
            }
            let Some(samples) = unshuffle_f32_bytes(&shuffled, cache.processed_samples_len) else {
                return false;
            };
            if samples.is_empty() {
                return false;
            }
            samples
        } else if speed_pitch_is_identity(self.speed, self.pitch_semitones) {
            raw_samples.as_slice().to_vec()
        } else {
            return false;
        };

        let base_note_timeline = Arc::new(cache.base_note_timeline);
        let base_note_timeline_step_sec = cache.base_note_timeline_step_sec;

        if self.preprocess_audio
            && (base_note_timeline.is_empty() || base_note_timeline_step_sec <= 0.0)
        {
            return false;
        }

        let (note_timeline, note_timeline_step_sec) = if self.preprocess_audio {
            Self::transform_note_timeline(
                Arc::clone(&base_note_timeline),
                base_note_timeline_step_sec,
                self.speed,
                self.pitch_semitones,
            )
        } else {
            (Arc::new(Vec::new()), 0.0)
        };

        self.processed_samples = processed_samples;
        self.waveform = Self::build_waveform_for_processed(
            self.processed_samples.as_slice(),
            sample_rate,
            self.audio_quality_mode,
            self.speed,
        );
        self.note_timeline = note_timeline;
        self.note_timeline_step_sec = note_timeline_step_sec;
        self.base_note_timeline = base_note_timeline;
        self.base_note_timeline_step_sec = base_note_timeline_step_sec;
        self.waveform_reset_view = true;
        self.selected_time_sec = self.selected_time_sec.min(self.source_duration());
        self.playing_preview_buffer = false;
        self.update_note_probabilities(true);

        true
    }

    #[allow(clippy::too_many_arguments)]
    fn persist_analysis_cache(
        song_hash: &str,
        sample_rate: u32,
        raw_sample_len: usize,
        audio_quality_mode: AudioQualityMode,
        speed: f32,
        pitch_semitones: f32,
        use_cqt_analysis: bool,
        preprocess_audio: bool,
        processed_samples: &[f32],
        base_note_timeline: &[Vec<f32>],
        base_note_timeline_step_sec: f32,
    ) {
        if processed_samples.is_empty() {
            return;
        }

        if preprocess_audio && (base_note_timeline.is_empty() || base_note_timeline_step_sec <= 0.0)
        {
            return;
        }

        let variant_key = analysis_cache_variant_key(
            sample_rate,
            raw_sample_len,
            audio_quality_mode,
            speed,
            pitch_semitones,
            use_cqt_analysis,
            preprocess_audio,
        );
        let song_dir = analysis_cache_library_dir().join(song_hash);
        if fs::create_dir_all(&song_dir).is_err() {
            return;
        }

        let store_processed = !speed_pitch_is_identity(speed, pitch_semitones);
        let (processed_samples_len, processed_samples_shuffled_bytes) = if store_processed {
            (
                processed_samples.len(),
                Some(shuffle_f32_bytes(processed_samples)),
            )
        } else {
            (0usize, None)
        };

        let snapshot = AnalysisCacheSnapshot {
            cache_version: ANALYSIS_CACHE_VERSION,
            sample_rate,
            raw_sample_len,
            audio_quality_mode_code: audio_quality_mode.cache_code(),
            speed_bits: speed.to_bits(),
            pitch_bits: pitch_semitones.to_bits(),
            use_cqt_analysis,
            preprocess_audio,
            processed_samples_len,
            processed_samples_shuffled_bytes: processed_samples_shuffled_bytes.as_deref(),
            base_note_timeline,
            base_note_timeline_step_sec,
        };

        let Ok(serialized) = bincode::serialize(&snapshot) else {
            return;
        };

        let compressed =
            zstd::bulk::compress(&serialized, ANALYSIS_CACHE_ZSTD_LEVEL).unwrap_or(serialized);
        if compressed.len() > ANALYSIS_CACHE_MAX_COMPRESSED_BYTES {
            return;
        }

        let cache_path = analysis_cache_file_path(song_hash, &variant_key);
        if ensure_parent_dir(cache_path.as_path()) {
            let _ = fs::write(cache_path, compressed);
        }
    }

    fn build_waveform_for_processed(
        processed_samples: &[f32],
        sample_rate: u32,
        audio_quality_mode: AudioQualityMode,
        speed: f32,
    ) -> Vec<[f64; 2]> {
        let mut waveform = waveform_points(
            processed_samples,
            sample_rate,
            audio_quality_mode.waveform_points(),
        );
        let speed_for_waveform = speed.clamp(0.25, 4.0) as f64;
        if (speed_for_waveform - 1.0).abs() > f64::EPSILON {
            for pt in &mut waveform {
                pt[0] *= speed_for_waveform;
            }
        }
        waveform
    }

    fn request_param_update_preserving_playback(&mut self) {
        self.refresh_timeline_for_current_params();

        // Parameter-only rebuilds rely on an existing analyzed base timeline.
        // If analysis is not ready yet, force a full rebuild to avoid ending up
        // with an empty timeline state that blocks playback controls.
        let needs_full_rebuild = self.preprocess_audio
            && (self.base_note_timeline.is_empty() || self.base_note_timeline_step_sec <= 0.0);
        if needs_full_rebuild {
            let was_playing = self.stop_if_playing();
            self.request_rebuild(was_playing, RebuildMode::Full);
            return;
        }

        let was_playing = self.stop_if_playing();
        if was_playing {
            self.request_rebuild(true, RebuildMode::ParametersPreview);
        } else {
            self.request_rebuild(false, RebuildMode::ParametersOnly);
        }
    }

    fn refresh_timeline_for_current_params(&mut self) {
        if !self.preprocess_audio {
            return;
        }
        if self.base_note_timeline.is_empty() || self.base_note_timeline_step_sec <= 0.0 {
            return;
        }

        let idx = (self.selected_time_sec.max(0.0) / self.base_note_timeline_step_sec) as usize;
        let idx = idx.min(self.base_note_timeline.len().saturating_sub(1));
        let frame = &self.base_note_timeline[idx];

        self.note_probs = if self.pitch_semitones.abs() < 1.0e-6 {
            frame.clone()
        } else {
            Self::transpose_frame(frame, self.pitch_semitones)
        };

        for (smoothed, current) in self
            .note_probs_smoothed
            .iter_mut()
            .zip(self.note_probs.iter())
        {
            *smoothed = *smoothed * 0.78 + *current * 0.22;
        }

        self.last_prob_update = Instant::now();
    }

    fn clear_processing_job(&mut self) {
        self.is_processing = false;
        self.processing_rx = None;
        self.active_job_id = None;
        self.active_rebuild_mode = RebuildMode::Full;
    }

    fn apply_processing_result(&mut self, result: ProcessingResult) {
        if result.mode == RebuildMode::ParametersPreview {
            if self.queued_param_update {
                self.queued_param_update = false;
                self.clear_processing_job();
                self.request_param_update_preserving_playback();
                return;
            }

            self.clear_processing_job();

            if self.restart_playback_after_processing {
                self.restart_playback_after_processing = false;
                if let Some(preview) = result.preview_playback {
                    let playback_rate = self.playback_rate();
                    if let Some(raw) = &self.audio_raw {
                        if let Some(engine) = &mut self.engine {
                            if let Err(err) = engine.play_chunk_at_timeline(
                                &preview.samples,
                                raw.sample_rate,
                                preview.timeline_start_sec,
                                playback_rate,
                            ) {
                                self.last_error = Some(format!("Playback error: {err}"));
                            } else {
                                self.playing_preview_buffer = true;
                            }
                        }
                    }
                }
            }

            // Continue with full render in the background so seeking and waveform stay accurate.
            self.request_rebuild(false, RebuildMode::ParametersOnly);
            return;
        }

        if result.mode == RebuildMode::ParametersOnly && self.queued_param_update {
            self.queued_param_update = false;
            self.clear_processing_job();
            self.request_param_update_preserving_playback();
            return;
        }

        let handoff_pos = if result.mode == RebuildMode::ParametersOnly
            && self.playing_preview_buffer
            && self.is_playing()
        {
            Some(self.current_position_sec())
        } else {
            None
        };
        let handoff_loop_end = if self.loop_enabled && self.loop_playback_enabled {
            self.loop_selection.map(|(a, b)| a.max(b))
        } else {
            None
        };

        self.processed_samples = result.processed_samples;
        self.waveform = result.waveform;
        self.note_timeline = result.note_timeline;
        self.note_timeline_step_sec = result.note_timeline_step_sec;
        self.base_note_timeline = result.base_note_timeline;
        self.base_note_timeline_step_sec = result.base_note_timeline_step_sec;
        self.waveform_reset_view = true;
        self.clear_processing_job();
        self.selected_time_sec = self.selected_time_sec.min(self.source_duration());

        if let Some(err) = result.analysis_error {
            self.last_error = Some(err);
        }
        self.update_note_probabilities(true);

        if self.restart_playback_after_processing {
            self.restart_playback_after_processing = false;
            self.play_from_selected();
        } else if let Some(source_pos) = handoff_pos {
            if let Some(loop_end) = handoff_loop_end {
                if loop_end - source_pos > LOOP_MIN_DURATION_SEC {
                    self.play_range(source_pos, Some(loop_end));
                } else {
                    self.play_from_selected();
                }
            } else {
                self.play_range(source_pos, None);
            }
            self.selected_time_sec = source_pos.min(self.source_duration());
            self.playing_preview_buffer = false;
        } else {
            self.playing_preview_buffer = false;
        }
    }

    fn maybe_commit_pending_param_change(&mut self, pointer_down: bool) {
        if !self.pending_param_change {
            return;
        }

        let debounce_elapsed = self
            .last_param_change_at
            .map(|at| at.elapsed() >= PARAM_UPDATE_LIVE_DEBOUNCE)
            .unwrap_or(!pointer_down);

        if pointer_down && !debounce_elapsed {
            return;
        }

        self.refresh_timeline_for_current_params();

        if self.is_param_render_in_progress() {
            self.queued_param_update = true;
        } else {
            self.request_param_update_preserving_playback();
        }

        self.pending_param_change = false;
        self.last_param_change_at = None;
    }

    fn is_blocking_processing(&self) -> bool {
        self.is_processing
            && self.active_rebuild_mode == RebuildMode::Full
            && self.processed_samples.is_empty()
    }

    fn is_param_render_in_progress(&self) -> bool {
        self.is_processing
            && matches!(
                self.active_rebuild_mode,
                RebuildMode::ParametersOnly | RebuildMode::ParametersPreview
            )
    }

    fn playback_rate(&self) -> f32 {
        self.speed.clamp(0.25, 4.0)
    }

    fn source_duration(&self) -> f32 {
        if let Some(audio) = &self.audio_raw {
            if audio.sample_rate > 0 {
                return audio.samples_mono.len() as f32 / audio.sample_rate as f32;
            }
        }
        0.0
    }

    fn source_to_output_time(&self, source_sec: f32) -> f32 {
        source_sec / self.playback_rate()
    }

    fn play_preview_at(&mut self, start_sec: f32, end_sec: Option<f32>) -> bool {
        if !self.is_param_render_in_progress() {
            return false;
        }

        let _ = end_sec;
        self.selected_time_sec = start_sec.max(0.0);

        if self.active_rebuild_mode == RebuildMode::ParametersPreview {
            self.restart_playback_after_processing = true;
        } else {
            self.request_rebuild(true, RebuildMode::ParametersPreview);
        }

        true
    }

    fn import_audio_with_ctx(&mut self, ctx: &egui::Context) {
        let picked = FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "aac"])
            .pick_file();

        if let Some(path) = picked {
            match load_audio_file(&path) {
                Ok(audio) => {
                    let source_path = path.to_path_buf();
                    self.apply_loaded_audio(source_path.clone(), audio, ctx);
                    if !self.try_restore_analysis_cache(&source_path) {
                        self.request_rebuild(false, RebuildMode::Full);
                    }
                    self.last_error = None;
                }
                Err(err) => {
                    self.last_error = Some(format!("Failed to load audio: {err}"));
                }
            }
        }
    }

    fn apply_loaded_audio(&mut self, path: PathBuf, audio: AudioData, ctx: &egui::Context) {
        self.cancel_active_processing();
        self.loaded_audio_hash = compute_file_hash(path.as_path());
        self.loaded_path = Some(path);
        self.selected_time_sec = 0.0;
        self.audio_raw = Some(audio);
        self.note_timeline = Arc::new(Vec::new());
        self.note_timeline_step_sec = 0.0;
        self.base_note_timeline = Arc::new(Vec::new());
        self.base_note_timeline_step_sec = 0.0;
        self.album_art_texture = self.create_album_art_texture(ctx);
    }

    fn create_album_art_texture(&self, ctx: &egui::Context) -> Option<egui::TextureHandle> {
        let bytes = self
            .audio_raw
            .as_ref()
            .and_then(|a| a.metadata.artwork_bytes.as_deref())?;
        if bytes.len() > ALBUM_ART_MAX_BYTES {
            return None;
        }

        let image = image::load_from_memory(bytes).ok()?.to_rgba8();
        let width = image.width() as usize;
        let height = image.height() as usize;
        if width == 0
            || height == 0
            || width > ALBUM_ART_MAX_DIMENSION
            || height > ALBUM_ART_MAX_DIMENSION
        {
            return None;
        }

        let size = [width, height];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        Some(ctx.load_texture("album-art", color_image, egui::TextureOptions::LINEAR))
    }
    fn request_rebuild(&mut self, restart_playback: bool, mode: RebuildMode) {
        let Some(raw) = &self.audio_raw else {
            return;
        };

        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let sample_rate = raw.sample_rate;
        let raw_samples: Arc<Vec<f32>> = Arc::clone(&raw.samples_mono);
        let speed = self.speed;
        let pitch_semitones = self.pitch_semitones;
        let audio_quality_mode = self.audio_quality_mode;
        let use_cqt = self.use_cqt_analysis;
        let preprocess_audio = self.preprocess_audio;
        let base_timeline = Arc::clone(&self.base_note_timeline);
        let base_step = self.base_note_timeline_step_sec;
        let selected_time_sec = self.selected_time_sec;
        let source_hash = self.loaded_audio_hash.clone();
        let processing_epoch = Arc::clone(&self.processing_epoch);

        let (tx, rx) = mpsc::channel::<ProcessingResult>();
        self.processing_rx = Some(rx);
        self.active_rebuild_mode = mode;
        self.active_job_id = Some(job_id);
        self.is_processing = true;
        self.restart_playback_after_processing |= restart_playback;
        self.processing_epoch.store(job_id, Ordering::Release);

        thread::spawn(move || {
            if mode == RebuildMode::ParametersPreview {
                if processing_epoch.load(Ordering::Acquire) != job_id {
                    return;
                }

                let preview_start_idx = (selected_time_sec.max(0.0) * sample_rate as f32) as usize;
                let preview_len = (PARAM_UPDATE_PREVIEW_SEC * sample_rate as f32) as usize;
                let preview_end_idx =
                    (preview_start_idx.saturating_add(preview_len)).min(raw_samples.len());

                let preview_samples = if preview_start_idx < preview_end_idx {
                    apply_speed_and_pitch(
                        &raw_samples[preview_start_idx..preview_end_idx],
                        sample_rate,
                        speed,
                        pitch_semitones,
                    )
                } else {
                    Vec::new()
                };

                if processing_epoch.load(Ordering::Acquire) != job_id {
                    return;
                }

                let _ = tx.send(ProcessingResult {
                    job_id,
                    mode,
                    processed_samples: Vec::new(),
                    waveform: Vec::new(),
                    note_timeline: Arc::new(Vec::new()),
                    note_timeline_step_sec: 0.0,
                    base_note_timeline: Arc::new(Vec::new()),
                    base_note_timeline_step_sec: 0.0,
                    analysis_error: None,
                    preview_playback: Some(PreviewPlayback {
                        samples: preview_samples,
                        timeline_start_sec: selected_time_sec.max(0.0),
                    }),
                });
                return;
            }

            let processed_samples = if processing_epoch.load(Ordering::Acquire) != job_id {
                None
            } else {
                Some(apply_speed_and_pitch(
                    raw_samples.as_slice(),
                    sample_rate,
                    speed,
                    pitch_semitones,
                ))
            };
            let Some(processed_samples) = processed_samples else {
                return;
            };

            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            let waveform = Self::build_waveform_for_processed(
                &processed_samples,
                sample_rate,
                audio_quality_mode,
                speed,
            );

            let (base_note_timeline, base_note_timeline_step_sec, analysis_error) = match mode {
                RebuildMode::Full => {
                    let (timeline, step, err) = Self::build_note_timeline(
                        raw_samples.as_slice(),
                        sample_rate,
                        audio_quality_mode.fft_window_size(),
                        use_cqt,
                        preprocess_audio,
                    );
                    (Arc::new(timeline), step, err)
                }
                RebuildMode::ParametersOnly => (base_timeline, base_step, None),
                RebuildMode::ParametersPreview => unreachable!("preview mode returns early"),
            };

            let (note_timeline, note_timeline_step_sec) = Self::transform_note_timeline(
                Arc::clone(&base_note_timeline),
                base_note_timeline_step_sec,
                speed,
                pitch_semitones,
            );

            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            if let Some(song_hash) = source_hash.as_ref() {
                Self::persist_analysis_cache(
                    song_hash,
                    sample_rate,
                    raw_samples.len(),
                    audio_quality_mode,
                    speed,
                    pitch_semitones,
                    use_cqt,
                    preprocess_audio,
                    processed_samples.as_slice(),
                    base_note_timeline.as_ref(),
                    base_note_timeline_step_sec,
                );
            }

            let _ = tx.send(ProcessingResult {
                job_id,
                mode,
                processed_samples,
                waveform,
                note_timeline,
                note_timeline_step_sec,
                base_note_timeline,
                base_note_timeline_step_sec,
                analysis_error,
                preview_playback: None,
            });
        });
    }

    fn poll_processing_result(&mut self) {
        let Some(rx) = &self.processing_rx else {
            return;
        };

        match rx.try_recv() {
            Ok(result) => {
                if Some(result.job_id) == self.active_job_id {
                    self.apply_processing_result(result);
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.clear_processing_job();
            }
        }
    }

    fn save_state_to_disk(&self) {
        let state = PersistedState {
            last_file: self.loaded_path.clone(),
            selected_time_sec: self.selected_time_sec,
            speed: self.speed,
            pitch_semitones: self.pitch_semitones,
            key_color_sensitivity: self.key_color_sensitivity,
            piano_zoom: self.piano_zoom,
            piano_key_height: self.piano_key_height,
            waveform_panel_height: self.waveform_panel_height,
            probability_panel_height: self.probability_panel_height,
            piano_panel_height: self.piano_panel_height,
            show_note_hist_window: self.show_note_hist_window,
            use_cqt_analysis: self.use_cqt_analysis,
            preprocess_audio: self.preprocess_audio,
            playback_volume: self.playback_volume,
            audio_quality_mode: self.audio_quality_mode,
            audio_output_device_id: self.audio_output_device_id.clone(),
            loop_enabled: self.loop_enabled,
            dark_mode: self.dark_mode,
            highlight_hex: color_to_hex(self.highlight_color),
            recent_highlight_hex: self.recent_highlight_hex.clone(),
        };

        if let Ok(raw) = serde_json::to_string_pretty(&state) {
            let path = state_file_path();
            if ensure_parent_dir(path.as_path()) {
                let _ = fs::write(path, raw);
            }
        }
    }

    fn update_note_probabilities(&mut self, force: bool) {
        if !force && self.last_prob_update.elapsed() < PROBABILITY_UPDATE_INTERVAL {
            return;
        }

        if self.preprocess_audio {
            if self.note_timeline.is_empty() || self.note_timeline_step_sec <= 0.0 {
                return;
            }

            let idx = (self.selected_time_sec.max(0.0) / self.note_timeline_step_sec) as usize;
            let idx = idx.min(self.note_timeline.len().saturating_sub(1));
            self.note_probs = self.note_timeline[idx].clone();
        } else {
            let Some(raw) = &self.audio_raw else {
                return;
            };
            if self.processed_samples.is_empty() {
                return;
            }

            let output_time_sec = self.source_to_output_time(self.selected_time_sec.max(0.0));
            let center = (output_time_sec * raw.sample_rate as f32) as usize;
            let fft_window_size = self.audio_quality_mode.fft_window_size();
            self.note_probs = if self.use_cqt_analysis {
                detect_note_probabilities_cqt(
                    &self.processed_samples,
                    raw.sample_rate,
                    center,
                    fft_window_size,
                )
            } else {
                detect_note_probabilities(
                    &self.processed_samples,
                    raw.sample_rate,
                    center,
                    fft_window_size,
                )
            };
        }

        // Smooth the visual state to reduce rapid flicker between adjacent notes.
        for (smoothed, current) in self
            .note_probs_smoothed
            .iter_mut()
            .zip(self.note_probs.iter())
        {
            *smoothed = *smoothed * 0.78 + *current * 0.22;
        }

        self.last_prob_update = Instant::now();
    }

    fn compute_fft_timeline(
        samples: &[f32],
        sample_rate: u32,
        step_sec: f32,
        fft_window_size: usize,
    ) -> Vec<Vec<f32>> {
        if samples.is_empty() || sample_rate == 0 || step_sec <= 0.0 || fft_window_size < 64 {
            return Vec::new();
        }

        let mut timeline = Vec::new();
        let total_sec = samples.len() as f32 / sample_rate as f32;
        let mut t = 0.0f32;

        while t <= total_sec {
            let center = (t * sample_rate as f32) as usize;
            timeline.push(detect_note_probabilities(
                samples,
                sample_rate,
                center,
                fft_window_size,
            ));
            t += step_sec;
        }

        if timeline.is_empty() {
            timeline.push(vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize]);
        }

        timeline
    }

    fn build_note_timeline(
        source_samples: &[f32],
        sample_rate: u32,
        fft_window_size: usize,
        use_cqt: bool,
        preprocess_audio: bool,
    ) -> (Vec<Vec<f32>>, f32, Option<String>) {
        if !preprocess_audio {
            return (Vec::new(), 0.0, None);
        }

        if use_cqt {
            match analyze_with_full_pipeline(source_samples, sample_rate) {
                Ok((_smoothed, probs)) => {
                    let duration_sec = source_samples.len() as f32 / sample_rate.max(1) as f32;
                    let step_sec = if probs.is_empty() {
                        0.0
                    } else {
                        (duration_sec / probs.len() as f32).max(1e-3)
                    };
                    (probs, step_sec, None)
                }
                Err(err) => {
                    let fallback = Self::compute_fft_timeline(
                        source_samples,
                        sample_rate,
                        FFT_TIMELINE_STEP_SEC,
                        fft_window_size,
                    );
                    (
                        fallback,
                        FFT_TIMELINE_STEP_SEC,
                        Some(format!("Pro analysis failed, using FFT fallback: {err}")),
                    )
                }
            }
        } else {
            (
                Self::compute_fft_timeline(
                    source_samples,
                    sample_rate,
                    FFT_TIMELINE_STEP_SEC,
                    fft_window_size,
                ),
                FFT_TIMELINE_STEP_SEC,
                None,
            )
        }
    }

    fn transpose_frame(frame: &[f32], semitones: f32) -> Vec<f32> {
        if frame.is_empty() {
            return Vec::new();
        }

        if semitones.abs() < 1.0e-6 {
            return frame.to_vec();
        }

        let n = frame.len() as f32;
        let mut out = vec![0.0f32; frame.len()];
        for (dst_idx, dst) in out.iter_mut().enumerate() {
            let src_idx = dst_idx as f32 - semitones;
            if src_idx < 0.0 || src_idx >= n - 1.0 {
                continue;
            }

            let i0 = src_idx.floor() as usize;
            let i1 = (i0 + 1).min(frame.len() - 1);
            let frac = src_idx - i0 as f32;
            *dst = frame[i0] * (1.0 - frac) + frame[i1] * frac;
        }

        out
    }

    fn transform_note_timeline(
        base_timeline: Arc<Vec<Vec<f32>>>,
        base_step_sec: f32,
        speed: f32,
        pitch_semitones: f32,
    ) -> (Arc<Vec<Vec<f32>>>, f32) {
        if base_timeline.is_empty() || base_step_sec <= 0.0 {
            return (Arc::new(Vec::new()), 0.0);
        }

        let transformed = if pitch_semitones.abs() < 1.0e-6 {
            base_timeline
        } else {
            let transformed = if base_timeline.len() >= 256 {
                base_timeline
                    .par_iter()
                    .map(|frame| Self::transpose_frame(frame, pitch_semitones))
                    .collect()
            } else {
                base_timeline
                    .iter()
                    .map(|frame| Self::transpose_frame(frame, pitch_semitones))
                    .collect()
            };
            Arc::new(transformed)
        };

        let _ = speed;
        let step_sec = base_step_sec;
        (transformed, step_sec)
    }

    fn play_from_selected(&mut self) {
        if self.play_preview_at(self.selected_time_sec, None) {
            return;
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };
        let playback_rate = self.playback_rate();

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_from(
                &self.processed_samples,
                raw.sample_rate,
                self.selected_time_sec,
                playback_rate,
            ) {
                self.last_error = Some(format!("Playback error: {err}"));
                self.playing_preview_buffer = false;
            } else {
                self.playing_preview_buffer = false;
            }
        } else {
            self.last_error = Some("Audio engine unavailable on this machine".to_string());
        }
    }

    fn skip_by_seconds(&mut self, delta_sec: f32) {
        if self.audio_raw.is_none() || self.processed_samples.is_empty() {
            return;
        }

        let duration = self.source_duration();
        let target = self.selected_time_sec + delta_sec;
        self.selected_time_sec = if self.loop_enabled {
            if let Some((a, b)) = self.loop_selection {
                let start = a.min(b);
                let end = a.max(b);
                if end - start > LOOP_MIN_DURATION_SEC {
                    target.clamp(start, end)
                } else {
                    target.clamp(0.0, duration)
                }
            } else {
                target.clamp(0.0, duration)
            }
        } else {
            target.clamp(0.0, duration)
        };
        self.update_note_probabilities(true);

        if self.is_playing() {
            if self.loop_enabled {
                if let Some((a, b)) = self.loop_selection {
                    let start = a.min(b);
                    let end = a.max(b);
                    if end - start > LOOP_MIN_DURATION_SEC {
                        self.loop_playback_enabled = true;
                        self.play_range(self.selected_time_sec, Some(end));
                        return;
                    }
                }
            }
            self.play_from_selected();
        }
    }

    fn play_range(&mut self, start_sec: f32, end_sec: Option<f32>) {
        if self.play_preview_at(start_sec, end_sec) {
            return;
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };
        let playback_rate = self.playback_rate();

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_range(
                &self.processed_samples,
                raw.sample_rate,
                start_sec,
                end_sec,
                playback_rate,
            ) {
                self.last_error = Some(format!("Playback error: {err}"));
                self.playing_preview_buffer = false;
            } else {
                self.playing_preview_buffer = false;
            }
        }
    }

    fn handle_space_replay(&mut self) {
        if self.audio_raw.is_none() || self.processed_samples.is_empty() {
            return;
        }

        if self.loop_enabled {
            if let Some((a, b)) = self.loop_selection {
                let start = a.min(b);
                let end = a.max(b);
                if end - start > LOOP_MIN_DURATION_SEC {
                    self.loop_playback_enabled = true;
                    self.selected_time_sec = start;
                    self.play_range(start, Some(end));
                    return;
                }
            }
        }

        self.loop_playback_enabled = false;
        let is_playing = self.is_playing();

        if is_playing {
            if let Some(engine) = &mut self.engine {
                engine.pause();
            }
            return;
        }

        let current_pos = self.current_position_sec();

        if current_pos <= 0.0 {
            self.play_from_selected();
        } else if let Some(engine) = &mut self.engine {
            engine.resume();
        }
    }

    fn stop(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.stop();
        }
        self.loop_playback_enabled = false;
        self.playing_preview_buffer = false;
    }

    fn sync_playhead_from_engine(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.sync_finished();
            if engine.is_playing() {
                self.selected_time_sec = engine.current_position().min(self.source_duration());
                self.update_note_probabilities(false);
            } else if self.loop_enabled && self.loop_playback_enabled {
                if let Some((a, b)) = self.loop_selection {
                    let start = a.min(b);
                    let end = a.max(b);
                    if end - start > LOOP_MIN_DURATION_SEC {
                        self.selected_time_sec = start;
                        self.play_range(start, Some(end));
                    }
                }
            } else {
                self.playing_preview_buffer = false;
            }
        }
    }

    fn draw_settings_menu(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(360.0);

        setting_toggle_row(ui, &mut self.dark_mode, "Dark Mode");
        ui.separator();

        ui.label("Highlight Presets");
        ui.horizontal_wrapped(|ui| {
            for (name, color) in PRESET_HIGHLIGHT_COLORS {
                let swatch = egui::RichText::new("   ").background_color(color);
                if ui
                    .add(egui::Button::new(swatch))
                    .on_hover_text(name)
                    .clicked()
                {
                    self.highlight_color = color;
                    self.custom_rgb = [color.r(), color.g(), color.b()];
                    push_recent_color(&mut self.recent_highlight_hex, color);
                }
            }
        });

        ui.separator();
        ui.label("Custom RGB");
        let mut rgb_changed = false;
        rgb_changed |= ui
            .add(egui::Slider::new(&mut self.custom_rgb[0], 0..=255).text("R"))
            .changed();
        rgb_changed |= ui
            .add(egui::Slider::new(&mut self.custom_rgb[1], 0..=255).text("G"))
            .changed();
        rgb_changed |= ui
            .add(egui::Slider::new(&mut self.custom_rgb[2], 0..=255).text("B"))
            .changed();

        if rgb_changed {
            self.highlight_color =
                egui::Color32::from_rgb(self.custom_rgb[0], self.custom_rgb[1], self.custom_rgb[2]);
        }

        ui.horizontal(|ui| {
            ui.label(color_to_hex(self.highlight_color));
            if ui.button("Save Color").clicked() {
                push_recent_color(&mut self.recent_highlight_hex, self.highlight_color);
            }
        });

        if !self.recent_highlight_hex.is_empty() {
            ui.label("Recent Colors");
            ui.horizontal_wrapped(|ui| {
                for hex in self.recent_highlight_hex.clone() {
                    if let Some(color) = parse_hex_color(&hex) {
                        let swatch = egui::RichText::new("   ").background_color(color);
                        if ui
                            .add(egui::Button::new(swatch))
                            .on_hover_text(hex.clone())
                            .clicked()
                        {
                            self.highlight_color = color;
                            self.custom_rgb = [color.r(), color.g(), color.b()];
                        }
                    }
                }
            });
        }

        ui.separator();

        ui.label("Audio Config");
        let mut quality_changed = false;
        egui::ComboBox::from_id_source("audio_quality_mode")
            .selected_text(self.audio_quality_mode.label())
            .show_ui(ui, |ui| {
                quality_changed |= ui
                    .selectable_value(
                        &mut self.audio_quality_mode,
                        AudioQualityMode::Draft,
                        AudioQualityMode::Draft.label(),
                    )
                    .changed();
                quality_changed |= ui
                    .selectable_value(
                        &mut self.audio_quality_mode,
                        AudioQualityMode::Balanced,
                        AudioQualityMode::Balanced.label(),
                    )
                    .changed();
                quality_changed |= ui
                    .selectable_value(
                        &mut self.audio_quality_mode,
                        AudioQualityMode::Studio,
                        AudioQualityMode::Studio.label(),
                    )
                    .changed();
            });

        if quality_changed {
            self.request_rebuild_preserving_playback();
        }

        ui.horizontal(|ui| {
            ui.label("Output Device");
            if ui.button("Refresh").clicked() {
                self.refresh_audio_output_devices();
            }
        });

        let selected_device_text = self
            .audio_output_device_id
            .as_deref()
            .and_then(|id| self.audio_output_devices.iter().find(|d| d.id == id))
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "System Default".to_string());

        let mut pending_device_change: Option<Option<String>> = None;
        egui::ComboBox::from_id_source("audio_output_device")
            .selected_text(selected_device_text)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(self.audio_output_device_id.is_none(), "System Default")
                    .clicked()
                {
                    pending_device_change = Some(None);
                }

                for option in self.audio_output_devices.clone() {
                    let label = if option.is_default {
                        format!("{} (OS Default)", option.name)
                    } else {
                        option.name.clone()
                    };
                    let selected =
                        self.audio_output_device_id.as_deref() == Some(option.id.as_str());
                    if ui.selectable_label(selected, label).clicked() {
                        pending_device_change = Some(Some(option.id.clone()));
                    }
                }
            });

        if let Some(device_change) = pending_device_change {
            self.apply_audio_output_device_change(device_change);
        }

        ui.separator();

        let preprocess_changed = setting_toggle_row(
            ui,
            &mut self.preprocess_audio,
            "Preprocess Audio (recommended)",
        );

        let cqt_changed = setting_toggle_row(
            ui,
            &mut self.use_cqt_analysis,
            "Use CQT Analysis (Pro Mode)",
        );

        let _ = setting_toggle_row(ui, &mut self.show_note_hist_window, "Show Probability Pane");

        if preprocess_changed || cqt_changed {
            self.request_rebuild_preserving_playback();
        }
    }

    fn top_bar_slider_with_input(
        ui: &mut egui::Ui,
        label: &str,
        value: &mut f32,
        min: f32,
        max: f32,
        suffix: &str,
        drag_speed: f64,
        max_decimals: usize,
    ) -> bool {
        let mut changed = false;

        let dark = ui.visuals().dark_mode;
        let row_fill = if dark {
            egui::Color32::from_rgb(28, 34, 43)
        } else {
            egui::Color32::from_rgb(234, 238, 244)
        };
        let row_stroke = if dark {
            egui::Color32::from_rgb(82, 93, 108)
        } else {
            egui::Color32::from_rgb(166, 176, 191)
        };
        let rail_fill = if dark {
            egui::Color32::from_rgb(78, 89, 105)
        } else {
            egui::Color32::from_rgb(184, 194, 210)
        };
        let rail_fill_hover = if dark {
            egui::Color32::from_rgb(95, 108, 126)
        } else {
            egui::Color32::from_rgb(170, 182, 199)
        };
        let rail_fill_active = if dark {
            egui::Color32::from_rgb(108, 124, 145)
        } else {
            egui::Color32::from_rgb(156, 170, 190)
        };

        egui::Frame::none()
            .fill(row_fill)
            .rounding(egui::Rounding::same(8.0))
            .stroke(egui::Stroke::new(1.0, row_stroke))
            .outer_margin(egui::Margin::symmetric(1.0, 0.0))
            .inner_margin(egui::Margin::symmetric(9.0, 6.0))
            .show(ui, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;

                    let label_color = ui.visuals().text_color();
                    let label_font = egui::TextStyle::Body.resolve(ui.style());
                    let label_width = ui
                        .fonts(|fonts| {
                            fonts
                                .layout_no_wrap(label.to_owned(), label_font.clone(), label_color)
                                .size()
                                .x
                        })
                        .max(56.0);
                    let (label_rect, _) =
                        ui.allocate_exact_size(egui::vec2(label_width, 22.0), egui::Sense::hover());
                    ui.painter().text(
                        label_rect.left_center(),
                        egui::Align2::LEFT_CENTER,
                        label,
                        label_font,
                        label_color,
                    );

                    ui.scope(|ui| {
                        let visuals = ui.visuals_mut();
                        visuals.slider_trailing_fill = true;
                        visuals.widgets.inactive.weak_bg_fill = rail_fill;
                        visuals.widgets.hovered.weak_bg_fill = rail_fill_hover;
                        visuals.widgets.active.weak_bg_fill = rail_fill_active;
                        visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, row_stroke);
                        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, row_stroke);
                        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, row_stroke);

                        changed |= ui
                            .add_sized(
                                [142.0, 22.0],
                                egui::Slider::new(value, min..=max)
                                    .show_value(false)
                                    .suffix(suffix),
                            )
                            .changed();

                        changed |= ui
                            .add_sized(
                                [74.0, 22.0],
                                egui::DragValue::new(value)
                                    .clamp_range(min..=max)
                                    .speed(drag_speed)
                                    .max_decimals(max_decimals)
                                    .suffix(suffix),
                            )
                            .changed();
                    });
                });
            });

        changed
    }

    fn draw_top_controls_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            egui::Frame::none()
                .inner_margin(egui::Margin::symmetric(12.0, 10.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 12.0;

                        if icon_button(ui, DOWNLOAD_SIMPLE, "Import Audio", true).clicked() {
                            self.import_audio_with_ctx(ctx);
                        }

                        let speed_changed = Self::top_bar_slider_with_input(
                            ui,
                            "Speed",
                            &mut self.speed,
                            0.5,
                            2.0,
                            "x",
                            0.01,
                            2,
                        );

                        let pitch_changed = Self::top_bar_slider_with_input(
                            ui,
                            "Pitch",
                            &mut self.pitch_semitones,
                            -12.0,
                            12.0,
                            " st",
                            0.1,
                            1,
                        );

                        if speed_changed || pitch_changed {
                            self.pending_param_change = true;
                            self.last_param_change_at = Some(Instant::now());
                        }

                        let settings_popup_id = ui.make_persistent_id("settings_popup_menu");
                        let settings_response = icon_button(ui, GEAR, "Settings", true);
                        if settings_response.clicked() {
                            ui.memory_mut(|mem| mem.toggle_popup(settings_popup_id));
                        }
                        egui::popup::popup_below_widget(
                            ui,
                            settings_popup_id,
                            &settings_response,
                            |ui| {
                                self.draw_settings_menu(ui);
                            },
                        );

                        let pointer_down = ui.input(|i| i.pointer.primary_down());
                        self.maybe_commit_pending_param_change(pointer_down);
                    });

                    if let Some(err) = &self.last_error {
                        ui.colored_label(ERROR_RED, err);
                    }

                    if self.is_processing {
                        let msg = match self.active_rebuild_mode {
                            RebuildMode::Full if self.preprocess_audio => {
                                "Analyzing track... controls unlock when note extraction finishes."
                            }
                            RebuildMode::ParametersPreview => "Buffering speed/pitch preview...",
                            _ => "Rendering full speed/pitch update...",
                        };
                        let processing_color = egui::Color32::from_rgb(
                            self.highlight_color.r().saturating_add(12),
                            self.highlight_color.g().saturating_add(12),
                            self.highlight_color.b().saturating_add(12),
                        );
                        ui.colored_label(processing_color, msg);
                    }
                });
        });
    }
}

impl eframe::App for TranscriberApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_brand_theme(ctx, self.dark_mode, self.highlight_color);
        self.lock_startup_min_window_size_once(ctx);

        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.handle_space_replay();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            self.skip_by_seconds(-SEEK_STEP_SEC);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            self.skip_by_seconds(SEEK_STEP_SEC);
        }

        self.poll_processing_result();
        self.sync_playhead_from_engine();

        self.draw_top_controls_panel(ctx);

        let mut piano_panel_builder = egui::TopBottomPanel::bottom("piano_panel")
            .resizable(true)
            .min_height(120.0);
        if self.piano_panel_height_needs_init {
            piano_panel_builder = piano_panel_builder.default_height(self.piano_panel_height);
            self.piano_panel_height_needs_init = false;
        }
        let piano_panel = piano_panel_builder.show(ctx, |ui| {
            if self.audio_raw.is_none() {
                return;
            }

            let pane_rect = ui.max_rect();
            let pane_hovered = ui.rect_contains_pointer(pane_rect);
            if pane_hovered && ui.input(|i| i.pointer.primary_clicked()) {
                self.piano_has_focus = true;
            }

            let panel_available_h = pane_rect.height();
            let white_w_for_zoom = keyboard_white_key_width(ui.available_width(), self.piano_zoom);
            let max_allowed_key_h =
                (white_w_for_zoom * WHITE_KEY_LENGTH_TO_WIDTH).clamp(MIN_PIANO_KEY_HEIGHT, 220.0);
            let key_h_for_frame = max_allowed_key_h;

            let prob_strip_height = if self.show_note_hist_window {
                (key_h_for_frame * 0.9).clamp(MIN_PROBABILITY_STRIP_HEIGHT, 120.0)
            } else {
                0.0
            };

            let keyboard_stack_h = key_h_for_frame
                + if self.show_note_hist_window {
                    prob_strip_height + 4.0
                } else {
                    0.0
                };
            let controls_reserved_h = 74.0;
            let extra_vertical =
                (panel_available_h - controls_reserved_h - keyboard_stack_h).max(0.0);
            if extra_vertical > 0.0 {
                ui.add_space(extra_vertical * 0.5);
            }

            let mut max_scroll_px: f32 = 0.0;
            if self.show_note_hist_window {
                let prob_draw = draw_probability_pane(
                    ui,
                    &self.note_probs_smoothed,
                    self.note_probs.as_slice(),
                    self.piano_zoom,
                    self.piano_scroll_px,
                    prob_strip_height,
                    self.highlight_color,
                );
                max_scroll_px = max_scroll_px.max(prob_draw.max_scroll_px);
                if prob_draw.clicked {
                    self.piano_has_focus = true;
                }
                ui.add_space(4.0);
            }

            let piano_draw = draw_piano_view(
                ui,
                &self.note_probs_smoothed,
                self.key_color_sensitivity,
                self.piano_zoom,
                key_h_for_frame,
                self.piano_scroll_px,
                self.highlight_color,
            );
            max_scroll_px = max_scroll_px.max(piano_draw.max_scroll_px);

            if piano_draw.clicked {
                self.piano_has_focus = true;
            }

            let (raw, smooth, shift, ctrl) = ui.ctx().input(|i| {
                (
                    i.raw_scroll_delta,
                    i.smooth_scroll_delta,
                    i.modifiers.shift,
                    i.modifiers.ctrl,
                )
            });
            let wheel_y = if raw.y.abs() > f32::EPSILON {
                raw.y
            } else {
                smooth.y
            };

            if self.piano_has_focus && pane_hovered {
                if ctrl && wheel_y.abs() > f32::EPSILON {
                    let z = if wheel_y > 0.0 { 1.08 } else { 0.92 };
                    self.piano_zoom = (self.piano_zoom * z).clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX);
                } else if shift && wheel_y.abs() > f32::EPSILON {
                    self.piano_scroll_px =
                        (self.piano_scroll_px - wheel_y * 0.7).clamp(0.0, max_scroll_px);
                }
            }

            ui.separator();
            egui::Frame::none()
                .inner_margin(egui::Margin::symmetric(12.0, 10.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 12.0;

                        let _ = Self::top_bar_slider_with_input(
                            ui,
                            "Key Color Sensitivity",
                            &mut self.key_color_sensitivity,
                            0.0,
                            2.0,
                            "",
                            0.01,
                            2,
                        );

                        let _ = Self::top_bar_slider_with_input(
                            ui,
                            "Piano Zoom",
                            &mut self.piano_zoom,
                            PIANO_ZOOM_MIN,
                            PIANO_ZOOM_MAX,
                            "x",
                            0.01,
                            2,
                        );
                    });
                });
        });
        self.piano_panel_height = piano_panel.response.rect.height().max(80.0);

        // Always use the tallest proportion-correct key height.
        let white_w_for_zoom =
            keyboard_white_key_width(piano_panel.response.rect.width(), self.piano_zoom);
        self.piano_key_height =
            (white_w_for_zoom * WHITE_KEY_LENGTH_TO_WIDTH).clamp(MIN_PIANO_KEY_HEIGHT, 220.0);

        self.probability_panel_height = if self.show_note_hist_window {
            (self.piano_key_height * 0.9).clamp(MIN_PROBABILITY_STRIP_HEIGHT, 120.0)
        } else {
            0.0
        };

        let waveform_central = egui::CentralPanel::default().show(ctx, |ui| {
            if self.audio_raw.is_none() {
                ui.label("Import an audio file to begin.");
                return;
            }

            let source_duration = self.source_duration().max(0.01);
            let waveform_height = (ui.available_height() - 112.0).max(40.0);
            let analysis_ready = !self.is_blocking_processing()
                && !self.processed_samples.is_empty()
                && (!self.preprocess_audio || !self.note_timeline.is_empty());

            Plot::new("waveform_plot")
                .height(waveform_height)
                .allow_scroll(false)
                .allow_zoom(false)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .show_grid(false)
                .show_x(false)
                .show_y(false)
                .show_axes([false, false])
                .include_y(-1.05)
                .include_y(1.05)
                .show(ui, |plot_ui| {
                    let highlight = self.highlight_color;
                    let loop_bg = egui::Color32::from_rgba_unmultiplied(
                        highlight.r(),
                        highlight.g(),
                        highlight.b(),
                        32,
                    );
                    let loop_wave_active = egui::Color32::from_rgb(
                        highlight.r().saturating_add(24),
                        highlight.g().saturating_add(24),
                        highlight.b().saturating_add(24),
                    );
                    let loop_wave_dim = egui::Color32::from_rgb(
                        highlight.r().saturating_sub(42),
                        highlight.g().saturating_sub(42),
                        highlight.b().saturating_sub(42),
                    );
                    let loop_edge = egui::Color32::from_rgb(
                        highlight.r().saturating_add(18),
                        highlight.g().saturating_add(18),
                        highlight.b().saturating_add(18),
                    );

                    if let Some((a, b)) = self.loop_selection {
                        let start = a.min(b) as f64;
                        let end = a.max(b) as f64;

                        let highlight = Polygon::new(PlotPoints::from(vec![
                            [start, -1.05],
                            [end, -1.05],
                            [end, 1.05],
                            [start, 1.05],
                        ]))
                        .fill_color(loop_bg)
                        .stroke(egui::Stroke::new(1.0, loop_edge));
                        plot_ui.polygon(highlight);
                    }

                    let mut wave_pre = Vec::<[f64; 2]>::new();
                    let mut wave_loop = Vec::<[f64; 2]>::new();
                    let mut wave_post = Vec::<[f64; 2]>::new();

                    if let Some((a, b)) = self.loop_selection {
                        let start = a.min(b) as f64;
                        let end = a.max(b) as f64;
                        for &pt in &self.waveform {
                            if pt[0] < start {
                                wave_pre.push(pt);
                            } else if pt[0] <= end {
                                wave_loop.push(pt);
                            } else {
                                wave_post.push(pt);
                            }
                        }

                        if !wave_pre.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(wave_pre.into_iter()))
                                    .color(loop_wave_dim),
                            );
                        }
                        if !wave_loop.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(wave_loop.into_iter()))
                                    .color(loop_wave_active),
                            );
                        }
                        if !wave_post.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(wave_post.into_iter()))
                                    .color(loop_wave_dim),
                            );
                        }
                    } else {
                        let line = Line::new(PlotPoints::from_iter(self.waveform.iter().copied()));
                        plot_ui.line(line.color(self.highlight_color));
                    }

                    plot_ui.vline(
                        VLine::new(self.selected_time_sec as f64)
                            .color(accent_soft(self.highlight_color)),
                    );

                    if let Some((a, b)) = self.loop_selection {
                        let start = a.min(b);
                        let end = a.max(b);
                        plot_ui.vline(VLine::new(start as f64).color(loop_edge));
                        plot_ui.vline(VLine::new(end as f64).color(loop_edge));
                    }

                    // Keep Y scale fixed and clamp X so navigation stays within audio bounds.
                    // On a fresh load, force full-track bounds first and clamp from those values.
                    let mut b = if self.waveform_reset_view {
                        self.waveform_reset_view = false;
                        PlotBounds::from_min_max([0.0, -1.05], [source_duration as f64, 1.05])
                    } else {
                        plot_ui.plot_bounds()
                    };

                    let pointer = plot_ui.pointer_coordinate();
                    let hovered = plot_ui.response().hovered();
                    let drag_started = plot_ui.response().drag_started();
                    let dragged = plot_ui.response().dragged();
                    let drag_stopped = plot_ui.response().drag_stopped();
                    let clicked = plot_ui.response().clicked();
                    let (raw_scroll, smooth_scroll, shift_held, ctrl_held, zoom_delta) =
                        plot_ui.ctx().input(|i| {
                            (
                                i.raw_scroll_delta,
                                i.smooth_scroll_delta,
                                i.modifiers.shift,
                                i.modifiers.ctrl,
                                i.zoom_delta_2d(),
                            )
                        });

                    let wheel_y = if raw_scroll.y.abs() > f32::EPSILON {
                        raw_scroll.y
                    } else if smooth_scroll.y.abs() > f32::EPSILON {
                        smooth_scroll.y
                    } else {
                        0.0
                    };

                    let wheel_x = if raw_scroll.x.abs() > f32::EPSILON {
                        raw_scroll.x
                    } else if smooth_scroll.x.abs() > f32::EPSILON {
                        smooth_scroll.x
                    } else {
                        0.0
                    };

                    if hovered {
                        let span = (b.max()[0] - b.min()[0]).max(0.001);

                        if shift_held
                            && (wheel_y.abs() > f32::EPSILON || wheel_x.abs() > f32::EPSILON)
                        {
                            let dominant_wheel = if wheel_x.abs() > wheel_y.abs() {
                                wheel_x
                            } else {
                                wheel_y
                            };
                            let shift_amount = -(dominant_wheel as f64) * 0.0015 * span;
                            b = PlotBounds::from_min_max(
                                [b.min()[0] + shift_amount, b.min()[1]],
                                [b.max()[0] + shift_amount, b.max()[1]],
                            );
                        } else if ctrl_held {
                            let zoom_from_wheel = if wheel_y.abs() > f32::EPSILON {
                                if wheel_y > 0.0 {
                                    0.88
                                } else {
                                    1.14
                                }
                            } else {
                                1.0
                            };

                            let zoom_from_input = if (zoom_delta.y - 1.0).abs() > f32::EPSILON {
                                (1.0 / zoom_delta.y as f64).clamp(0.7, 1.4)
                            } else {
                                1.0
                            };

                            let zoom = zoom_from_wheel * zoom_from_input;

                            if (zoom - 1.0).abs() > f64::EPSILON {
                                let min_span = (source_duration as f64 / 400.0).max(0.02);
                                let max_span = source_duration as f64;
                                let new_span = (span * zoom).clamp(min_span, max_span);

                                let center_x = pointer
                                    .map(|p| p.x)
                                    .unwrap_or((b.min()[0] + b.max()[0]) * 0.5)
                                    .clamp(0.0, source_duration as f64);

                                let left_ratio = ((center_x - b.min()[0]) / span).clamp(0.0, 1.0);
                                let new_min = center_x - left_ratio * new_span;
                                let new_max = new_min + new_span;
                                b = PlotBounds::from_min_max(
                                    [new_min, b.min()[1]],
                                    [new_max, b.max()[1]],
                                );
                            }
                        }
                    }

                    let mut x_span = (b.max()[0] - b.min()[0]).max(0.001);
                    let max_span = source_duration as f64;
                    if x_span > max_span {
                        x_span = max_span;
                    }

                    let min_x = if x_span >= max_span {
                        0.0
                    } else {
                        b.min()[0].clamp(0.0, max_span - x_span)
                    };
                    let max_x = (min_x + x_span).min(max_span);

                    plot_ui
                        .set_plot_bounds(PlotBounds::from_min_max([min_x, -1.05], [max_x, 1.05]));

                    if analysis_ready && drag_started {
                        self.drag_select_anchor_sec = pointer
                            .map(|p| p.x.clamp(0.0, source_duration as f64) as f32)
                            .or(Some(self.selected_time_sec));
                    }

                    if analysis_ready && dragged {
                        if let (Some(anchor), Some(p)) = (
                            self.drag_select_anchor_sec,
                            pointer.map(|p| p.x.clamp(0.0, source_duration as f64) as f32),
                        ) {
                            self.loop_selection = Some((anchor, p));
                        }
                    }

                    if analysis_ready && drag_stopped {
                        if let Some((a, b)) = self.loop_selection {
                            if (a - b).abs() < LOOP_MIN_DURATION_SEC {
                                self.loop_selection = None;
                                self.loop_playback_enabled = false;
                            } else {
                                let start = a.min(b);
                                let end = a.max(b);
                                self.selected_time_sec = start;
                                self.loop_enabled = true;
                                self.loop_playback_enabled = true;
                                self.play_range(start, Some(end));
                            }
                        }
                        self.drag_select_anchor_sec = None;
                    }

                    if analysis_ready && clicked {
                        if let Some(pointer) = pointer {
                            self.selected_time_sec =
                                pointer.x.clamp(0.0, source_duration as f64) as f32;
                            self.loop_selection = None;
                            self.loop_playback_enabled = false;
                            self.update_note_probabilities(true);
                            if self.is_playing() {
                                self.play_from_selected();
                            }
                        }
                    }
                });

            ui.add_space(8.0);
            let remaining_h = ui.available_height();
            let media_height = 96.0;
            let top_pad = ((remaining_h - media_height) * 0.5).max(0.0);
            if top_pad > 0.0 {
                ui.add_space(top_pad);
            }

            let available_w = ui.available_width();
            ui.allocate_ui(egui::vec2(available_w, media_height), |ui| {
                draw_media_controls(self, ui, analysis_ready, source_duration);
            });
        });
        self.waveform_panel_height = waveform_central.response.rect.height().clamp(120.0, 5000.0);

        // Keep UI responsive while playing.
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        if self.last_state_save_at.elapsed() >= Duration::from_secs(2) {
            self.save_state_to_disk();
            self.last_state_save_at = Instant::now();
        }
    }
}

impl Drop for TranscriberApp {
    fn drop(&mut self) {
        self.save_state_to_disk();
    }
}
