use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use egui_phosphor::regular::{DOWNLOAD_SIMPLE, GEAR};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Polygon, VLine};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    analyze_with_full_pipeline, detect_note_probabilities, detect_note_probabilities_cqt,
    waveform_points, PIANO_HIGH_MIDI, PIANO_LOW_MIDI,
};
use crate::audio_io::{load_audio_file, AudioData};
use crate::dsp::apply_speed_and_pitch;
use crate::playback::AudioEngine;
use crate::theme::{apply_brand_theme, ACCENT_ORANGE, ERROR_RED};
use crate::ui::keyboard::{
    draw_piano_view, draw_probability_pane, keyboard_white_key_width, MIN_PIANO_KEY_HEIGHT,
    MIN_PROBABILITY_STRIP_HEIGHT, PIANO_ZOOM_MAX, PIANO_ZOOM_MIN, WHITE_KEY_LENGTH_TO_WIDTH,
};
use crate::ui::utils::{accent_soft, color_to_hex, parse_hex_color, push_recent_color};
use crate::ui::widgets::{icon_button, icon_font_id};

mod media_controls;
use media_controls::{draw_media_controls, setting_toggle_row};

const STATE_FILE_NAME: &str = ".transcriber_state.json";
const PROBABILITY_UPDATE_INTERVAL: Duration = Duration::from_millis(16);
const FFT_TIMELINE_STEP_SEC: f32 = 0.05;
const FFT_WINDOW_SIZE: usize = 4096;
const LOOP_MIN_DURATION_SEC: f32 = 0.01;
const SEEK_STEP_SEC: f32 = 5.0;
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

fn default_playback_volume() -> f32 {
    0.8
}

fn default_dark_mode() -> bool {
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
    key_color_sensitivity: f32,
    piano_zoom: f32,
    piano_key_height: f32,
    waveform_panel_height: f32,
    probability_panel_height: f32,
    piano_panel_height: f32,
    show_note_hist_window: bool,
    #[serde(default)]
    use_cqt_analysis: bool,
    #[serde(default = "default_preprocess_audio")]
    preprocess_audio: bool,
    #[serde(default = "default_playback_volume")]
    playback_volume: f32,
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
            key_color_sensitivity: 0.75,
            piano_zoom: 1.0,
            piano_key_height: 72.0,
            waveform_panel_height: 320.0,
            probability_panel_height: 130.0,
            piano_panel_height: 170.0,
            show_note_hist_window: true,
            use_cqt_analysis: false,
            preprocess_audio: true,
            playback_volume: 0.8,
            loop_enabled: false,
            dark_mode: true,
            highlight_hex: default_highlight_hex(),
            recent_highlight_hex: Vec::new(),
        }
    }
}

fn state_file_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(STATE_FILE_NAME)
}

fn load_persisted_state() -> PersistedState {
    let path = state_file_path();
    let Ok(raw) = fs::read_to_string(path) else {
        return PersistedState::default();
    };

    serde_json::from_str::<PersistedState>(&raw).unwrap_or_default()
}

pub struct TranscriberApp {
    loaded_path: Option<PathBuf>,
    audio_raw: Option<AudioData>,
    processed_samples: Vec<f32>,
    waveform: Vec<[f64; 2]>,
    note_timeline: Vec<Vec<f32>>,
    note_timeline_step_sec: f32,
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
    active_job_id: Option<u64>,
    next_job_id: u64,
    is_processing: bool,
    pending_param_change: bool,
    restart_playback_after_processing: bool,
    last_prob_update: Instant,
    show_note_hist_window: bool,
    playback_volume: f32,
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
    use_cqt_analysis: bool,
    preprocess_audio: bool,
    album_art_texture: Option<egui::TextureHandle>,
}

struct ProcessingResult {
    job_id: u64,
    processed_samples: Vec<f32>,
    waveform: Vec<[f64; 2]>,
    note_timeline: Vec<Vec<f32>>,
    note_timeline_step_sec: f32,
    analysis_error: Option<String>,
}

impl TranscriberApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let persisted = load_persisted_state();
        let highlight_color = parse_hex_color(&persisted.highlight_hex).unwrap_or(ACCENT_ORANGE);
        apply_brand_theme(&_cc.egui_ctx, persisted.dark_mode, highlight_color);

        let mut app = Self {
            loaded_path: None,
            audio_raw: None,
            processed_samples: Vec::new(),
            waveform: Vec::new(),
            note_timeline: Vec::new(),
            note_timeline_step_sec: 0.0,
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
            engine: AudioEngine::new().ok(),
            processing_rx: None,
            active_job_id: None,
            next_job_id: 1,
            is_processing: false,
            pending_param_change: false,
            restart_playback_after_processing: false,
            last_prob_update: Instant::now(),
            show_note_hist_window: persisted.show_note_hist_window,
            playback_volume: persisted.playback_volume.clamp(0.0, 1.5),
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
            use_cqt_analysis: persisted.use_cqt_analysis,
            preprocess_audio: persisted.preprocess_audio,
            album_art_texture: None,
        };

        if let Some(engine) = &mut app.engine {
            engine.set_volume(app.playback_volume);
        }

        if let Some(path) = persisted.last_file {
            if path.exists() {
                match load_audio_file(&path) {
                    Ok(audio) => {
                        app.apply_loaded_audio(path, audio, &_cc.egui_ctx);
                        app.request_rebuild(false);
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
        self.request_rebuild(was_playing);
    }

    fn clear_processing_job(&mut self) {
        self.is_processing = false;
        self.processing_rx = None;
        self.active_job_id = None;
    }

    fn apply_processing_result(&mut self, result: ProcessingResult) {
        self.processed_samples = result.processed_samples;
        self.waveform = result.waveform;
        self.note_timeline = result.note_timeline;
        self.note_timeline_step_sec = result.note_timeline_step_sec;
        self.waveform_reset_view = true;
        self.clear_processing_job();
        self.selected_time_sec = self.selected_time_sec.min(self.duration());

        if let Some(err) = result.analysis_error {
            self.last_error = Some(err);
        }
        self.update_note_probabilities(true);

        if self.restart_playback_after_processing {
            self.restart_playback_after_processing = false;
            self.play_from_selected();
        }
    }

    fn maybe_commit_pending_param_change(&mut self, pointer_down: bool) {
        if self.pending_param_change && !pointer_down {
            self.request_rebuild_preserving_playback();
            self.pending_param_change = false;
        }
    }

    fn duration(&self) -> f32 {
        if let Some(audio) = &self.audio_raw {
            if audio.sample_rate > 0 {
                return self.processed_samples.len() as f32 / audio.sample_rate as f32;
            }
        }
        0.0
    }

    fn import_audio_with_ctx(&mut self, ctx: &egui::Context) {
        let picked = FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "aac"])
            .pick_file();

        if let Some(path) = picked {
            match load_audio_file(&path) {
                Ok(audio) => {
                    self.apply_loaded_audio(path.to_path_buf(), audio, ctx);
                    self.request_rebuild(false);
                    self.last_error = None;
                }
                Err(err) => {
                    self.last_error = Some(format!("Failed to load audio: {err}"));
                }
            }
        }
    }

    fn apply_loaded_audio(&mut self, path: PathBuf, audio: AudioData, ctx: &egui::Context) {
        self.loaded_path = Some(path);
        self.selected_time_sec = 0.0;
        self.audio_raw = Some(audio);
        self.album_art_texture = self.create_album_art_texture(ctx);
    }

    fn create_album_art_texture(&self, ctx: &egui::Context) -> Option<egui::TextureHandle> {
        let bytes = self
            .audio_raw
            .as_ref()
            .and_then(|a| a.metadata.artwork_bytes.as_deref())?;

        let image = image::load_from_memory(bytes).ok()?.to_rgba8();
        let size = [image.width() as usize, image.height() as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        Some(ctx.load_texture("album-art", color_image, egui::TextureOptions::LINEAR))
    }
    fn request_rebuild(&mut self, restart_playback: bool) {
        let Some(raw) = &self.audio_raw else {
            return;
        };

        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let sample_rate = raw.sample_rate;
        let raw_samples: Arc<Vec<f32>> = Arc::clone(&raw.samples_mono);
        let speed = self.speed;
        let pitch_semitones = self.pitch_semitones;
        let use_cqt = self.use_cqt_analysis;
        let preprocess_audio = self.preprocess_audio;

        let (tx, rx) = mpsc::channel::<ProcessingResult>();
        self.processing_rx = Some(rx);
        self.active_job_id = Some(job_id);
        self.is_processing = true;
        self.restart_playback_after_processing |= restart_playback;

        thread::spawn(move || {
            let processed_samples =
                apply_speed_and_pitch(raw_samples.as_slice(), speed, pitch_semitones);
            let waveform = waveform_points(&processed_samples, sample_rate, 6000);

            let (note_timeline, note_timeline_step_sec, analysis_error) = Self::build_note_timeline(
                &processed_samples,
                sample_rate,
                use_cqt,
                preprocess_audio,
            );

            let _ = tx.send(ProcessingResult {
                job_id,
                processed_samples,
                waveform,
                note_timeline,
                note_timeline_step_sec,
                analysis_error,
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
            loop_enabled: self.loop_enabled,
            dark_mode: self.dark_mode,
            highlight_hex: color_to_hex(self.highlight_color),
            recent_highlight_hex: self.recent_highlight_hex.clone(),
        };

        if let Ok(raw) = serde_json::to_string_pretty(&state) {
            let _ = fs::write(state_file_path(), raw);
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

            let center = (self.selected_time_sec.max(0.0) * raw.sample_rate as f32) as usize;
            self.note_probs = if self.use_cqt_analysis {
                detect_note_probabilities_cqt(
                    &self.processed_samples,
                    raw.sample_rate,
                    center,
                    FFT_WINDOW_SIZE,
                )
            } else {
                detect_note_probabilities(
                    &self.processed_samples,
                    raw.sample_rate,
                    center,
                    FFT_WINDOW_SIZE,
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

    fn compute_fft_timeline(samples: &[f32], sample_rate: u32, step_sec: f32) -> Vec<Vec<f32>> {
        if samples.is_empty() || sample_rate == 0 || step_sec <= 0.0 {
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
                FFT_WINDOW_SIZE,
            ));
            t += step_sec;
        }

        if timeline.is_empty() {
            timeline.push(vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize]);
        }

        timeline
    }

    fn build_note_timeline(
        processed_samples: &[f32],
        sample_rate: u32,
        use_cqt: bool,
        preprocess_audio: bool,
    ) -> (Vec<Vec<f32>>, f32, Option<String>) {
        if !preprocess_audio {
            return (Vec::new(), 0.0, None);
        }

        if use_cqt {
            match analyze_with_full_pipeline(processed_samples, sample_rate) {
                Ok((_smoothed, probs)) => {
                    let duration_sec = processed_samples.len() as f32 / sample_rate.max(1) as f32;
                    let step_sec = if probs.is_empty() {
                        0.0
                    } else {
                        (duration_sec / probs.len() as f32).max(1e-3)
                    };
                    (probs, step_sec, None)
                }
                Err(err) => {
                    let fallback = Self::compute_fft_timeline(
                        processed_samples,
                        sample_rate,
                        FFT_TIMELINE_STEP_SEC,
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
                Self::compute_fft_timeline(processed_samples, sample_rate, FFT_TIMELINE_STEP_SEC),
                FFT_TIMELINE_STEP_SEC,
                None,
            )
        }
    }

    fn play_from_selected(&mut self) {
        let Some(raw) = &self.audio_raw else {
            return;
        };

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_from(
                &self.processed_samples,
                raw.sample_rate,
                self.selected_time_sec,
            ) {
                self.last_error = Some(format!("Playback error: {err}"));
            }
        } else {
            self.last_error = Some("Audio engine unavailable on this machine".to_string());
        }
    }

    fn skip_by_seconds(&mut self, delta_sec: f32) {
        if self.audio_raw.is_none() || self.processed_samples.is_empty() {
            return;
        }

        let duration = self.duration();
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
        let Some(raw) = &self.audio_raw else {
            return;
        };

        if let Some(engine) = &mut self.engine {
            if let Err(err) =
                engine.play_range(&self.processed_samples, raw.sample_rate, start_sec, end_sec)
            {
                self.last_error = Some(format!("Playback error: {err}"));
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
    }

    fn sync_playhead_from_engine(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.sync_finished();
            if engine.is_playing() {
                self.selected_time_sec = engine.current_position().min(self.duration());
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

    fn draw_top_controls_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if icon_button(ui, DOWNLOAD_SIMPLE, "Import Audio", true).clicked() {
                    self.import_audio_with_ctx(ctx);
                }

                let speed_changed = ui
                    .add(egui::Slider::new(&mut self.speed, 0.5..=2.0).suffix("x"))
                    .changed();

                let pitch_changed = ui
                    .add(egui::Slider::new(&mut self.pitch_semitones, -12.0..=12.0).suffix(" st"))
                    .changed();

                if speed_changed || pitch_changed {
                    self.pending_param_change = true;
                }

                ui.menu_button(egui::RichText::new(GEAR).font(icon_font_id(18.0)), |ui| {
                    self.draw_settings_menu(ui);
                });

                let pointer_down = ui.input(|i| i.pointer.primary_down());
                self.maybe_commit_pending_param_change(pointer_down);
            });

            if let Some(err) = &self.last_error {
                ui.colored_label(ERROR_RED, err);
            }

            if self.is_processing {
                let msg = if self.preprocess_audio {
                    "Analyzing track... controls unlock when note extraction finishes."
                } else {
                    "Processing speed/pitch update..."
                };
                ui.colored_label(egui::Color32::from_rgb(240, 180, 30), msg);
            }
        });
    }
}

impl eframe::App for TranscriberApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_brand_theme(ctx, self.dark_mode, self.highlight_color);

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
            ui.horizontal_wrapped(|ui| {
                ui.add_sized(
                    [720.0, 0.0],
                    egui::Slider::new(&mut self.key_color_sensitivity, 0.0..=2.0)
                        .text("Key Color Sensitivity"),
                );
                ui.add(
                    egui::Slider::new(&mut self.piano_zoom, PIANO_ZOOM_MIN..=PIANO_ZOOM_MAX)
                        .text("Piano Zoom")
                        .suffix("x"),
                );
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

            let duration = self.duration().max(0.01);
            let waveform_height = (ui.available_height() - 112.0).max(40.0);
            let analysis_ready = !self.is_processing
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
                    if let Some((a, b)) = self.loop_selection {
                        let start = a.min(b) as f64;
                        let end = a.max(b) as f64;

                        let highlight = Polygon::new(PlotPoints::from(vec![
                            [start, -1.05],
                            [end, -1.05],
                            [end, 1.05],
                            [start, 1.05],
                        ]))
                        .fill_color(egui::Color32::from_rgba_unmultiplied(255, 120, 35, 32));
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
                                    .color(egui::Color32::from_rgb(214, 130, 74)),
                            );
                        }
                        if !wave_loop.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(wave_loop.into_iter()))
                                    .color(egui::Color32::from_rgb(255, 188, 72)),
                            );
                        }
                        if !wave_post.is_empty() {
                            plot_ui.line(
                                Line::new(PlotPoints::from_iter(wave_post.into_iter()))
                                    .color(egui::Color32::from_rgb(214, 130, 74)),
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
                        plot_ui.vline(
                            VLine::new(start as f64).color(egui::Color32::from_rgb(255, 190, 120)),
                        );
                        plot_ui.vline(
                            VLine::new(end as f64).color(egui::Color32::from_rgb(255, 190, 120)),
                        );
                    }

                    // Keep Y scale fixed and clamp X so navigation stays within audio bounds.
                    // On a fresh load, force full-track bounds first and clamp from those values.
                    let mut b = if self.waveform_reset_view {
                        self.waveform_reset_view = false;
                        PlotBounds::from_min_max([0.0, -1.05], [duration as f64, 1.05])
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
                                let min_span = (duration as f64 / 400.0).max(0.02);
                                let max_span = duration as f64;
                                let new_span = (span * zoom).clamp(min_span, max_span);

                                let center_x = pointer
                                    .map(|p| p.x)
                                    .unwrap_or((b.min()[0] + b.max()[0]) * 0.5)
                                    .clamp(0.0, duration as f64);

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
                    let max_span = duration as f64;
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
                            .map(|p| p.x.clamp(0.0, duration as f64) as f32)
                            .or(Some(self.selected_time_sec));
                    }

                    if analysis_ready && dragged {
                        if let (Some(anchor), Some(p)) = (
                            self.drag_select_anchor_sec,
                            pointer.map(|p| p.x.clamp(0.0, duration as f64) as f32),
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
                            self.selected_time_sec = pointer.x.clamp(0.0, duration as f64) as f32;
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
                draw_media_controls(self, ui, analysis_ready, duration);
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
