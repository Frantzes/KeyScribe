use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use egui_phosphor::regular::{
    DOWNLOAD_SIMPLE, GEAR, PAUSE, PLAY, REPEAT, SKIP_BACK, SKIP_FORWARD, SPEAKER_HIGH, SPEAKER_NONE,
};
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

const STATE_FILE_NAME: &str = ".transcriber_state.json";
const PIANO_ZOOM_MIN: f32 = 0.35;
const PIANO_ZOOM_MAX: f32 = 1.0;
const WHITE_KEY_LENGTH_TO_WIDTH: f32 = 6.3;
const MIN_PIANO_KEY_HEIGHT: f32 = 16.0;
const MIN_PROBABILITY_STRIP_HEIGHT: f32 = 20.0;
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

            let (note_timeline, note_timeline_step_sec, analysis_error) = if preprocess_audio {
                if use_cqt {
                    match analyze_with_full_pipeline(&processed_samples, sample_rate) {
                        Ok((_smoothed, probs)) => {
                            let duration_sec =
                                processed_samples.len() as f32 / sample_rate.max(1) as f32;
                            let step_sec = if probs.is_empty() {
                                0.0
                            } else {
                                (duration_sec / probs.len() as f32).max(1e-3)
                            };
                            (probs, step_sec, None)
                        }
                        Err(err) => {
                            let fallback =
                                Self::compute_fft_timeline(&processed_samples, sample_rate, 0.05);
                            (
                                fallback,
                                0.05,
                                Some(format!("Pro analysis failed, using FFT fallback: {err}")),
                            )
                        }
                    }
                } else {
                    let step_sec = 0.05;
                    (
                        Self::compute_fft_timeline(&processed_samples, sample_rate, step_sec),
                        step_sec,
                        None,
                    )
                }
            } else {
                (Vec::new(), 0.0, None)
            };

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
                    self.processed_samples = result.processed_samples;
                    self.waveform = result.waveform;
                    self.note_timeline = result.note_timeline;
                    self.note_timeline_step_sec = result.note_timeline_step_sec;
                    self.waveform_reset_view = true;
                    self.is_processing = false;
                    self.processing_rx = None;
                    self.active_job_id = None;
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
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.is_processing = false;
                self.processing_rx = None;
                self.active_job_id = None;
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
        if !force && self.last_prob_update.elapsed() < Duration::from_millis(80) {
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
                    4096,
                )
            } else {
                detect_note_probabilities(&self.processed_samples, raw.sample_rate, center, 4096)
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
                4096,
            ));
            t += step_sec;
        }

        if timeline.is_empty() {
            timeline.push(vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize]);
        }

        timeline
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
                if end - start > 0.01 {
                    self.loop_playback_enabled = true;
                    self.selected_time_sec = start;
                    self.play_range(start, Some(end));
                    return;
                }
            }
        }

        self.loop_playback_enabled = false;
        let is_playing = self
            .engine
            .as_ref()
            .map(|e| e.is_playing())
            .unwrap_or(false);

        if is_playing {
            if let Some(engine) = &mut self.engine {
                engine.pause();
            }
            return;
        }

        let current_pos = self
            .engine
            .as_ref()
            .map(|e| e.current_position())
            .unwrap_or(0.0);

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
                    if end - start > 0.01 {
                        self.selected_time_sec = start;
                        self.play_range(start, Some(end));
                    }
                }
            }
        }
    }
}

impl eframe::App for TranscriberApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_brand_theme(ctx, self.dark_mode, self.highlight_color);

        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.handle_space_replay();
        }

        self.poll_processing_result();
        self.sync_playhead_from_engine();

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
                        self.highlight_color = egui::Color32::from_rgb(
                            self.custom_rgb[0],
                            self.custom_rgb[1],
                            self.custom_rgb[2],
                        );
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

                    let _ = setting_toggle_row(
                        ui,
                        &mut self.show_note_hist_window,
                        "Show Probability Pane",
                    );

                    if preprocess_changed || cqt_changed {
                        let was_playing = self
                            .engine
                            .as_ref()
                            .map(|e| e.is_playing())
                            .unwrap_or(false);
                        if was_playing {
                            self.stop();
                        }
                        self.request_rebuild(was_playing);
                    }
                });

                let pointer_down = ui.input(|i| i.pointer.primary_down());
                if self.pending_param_change && !pointer_down {
                    let was_playing = self
                        .engine
                        .as_ref()
                        .map(|e| e.is_playing())
                        .unwrap_or(false);
                    if was_playing {
                        self.stop();
                    }
                    self.request_rebuild(was_playing);
                    self.pending_param_change = false;
                }
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
                .allow_drag(analysis_ready)
                .allow_boxed_zoom(false)
                .show_grid(false)
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
                            if (a - b).abs() < 0.01 {
                                self.loop_selection = None;
                                self.loop_playback_enabled = false;
                            } else {
                                let start = a.min(b);
                                let end = a.max(b);
                                self.selected_time_sec = start;
                                if self.loop_enabled {
                                    self.loop_playback_enabled = true;
                                    self.play_range(start, Some(end));
                                } else {
                                    self.loop_playback_enabled = false;
                                }
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
                            if self
                                .engine
                                .as_ref()
                                .map(|e| e.is_playing())
                                .unwrap_or(false)
                            {
                                self.play_from_selected();
                            }
                        }
                    }
                });

            ui.add_space(8.0);
            let available_w = ui.available_width();
            let media_width = available_w.min(980.0);
            let free_w = (available_w - media_width).max(0.0);
            let center_bias_px = 10.0;
            let base_left_gutter = (free_w * 0.5).floor();
            let left_gutter = (base_left_gutter + center_bias_px).min(free_w);
            let right_gutter = (free_w - left_gutter).max(0.0);
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.horizontal(|ui| {
                    if left_gutter > 0.0 {
                        ui.allocate_exact_size(egui::vec2(left_gutter, 0.0), egui::Sense::hover());
                    }
                    ui.allocate_ui_with_layout(
                        egui::vec2(media_width, 0.0),
                        egui::Layout::top_down(egui::Align::Center),
                        |ui| {
                            draw_media_controls(self, ui, analysis_ready, duration);
                        },
                    );
                    if right_gutter > 0.0 {
                        ui.allocate_exact_size(egui::vec2(right_gutter, 0.0), egui::Sense::hover());
                    }
                });
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

struct KeyboardDrawResult {
    clicked: bool,
    max_scroll_px: f32,
}

fn keyboard_white_key_width(viewport_width: f32, zoom: f32) -> f32 {
    let white_count = (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI)
        .filter(|midi| !is_black_key(*midi))
        .count() as f32;
    let fit_width = (viewport_width / white_count.max(1.0)).max(1.0);
    fit_width * zoom.clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX)
}

fn white_index_before_midi(midi: u8) -> usize {
    (PIANO_LOW_MIDI..midi).filter(|m| !is_black_key(*m)).count()
}

fn draw_piano_view(
    ui: &mut egui::Ui,
    probs: &[f32],
    sensitivity: f32,
    zoom: f32,
    key_height: f32,
    scroll_px: f32,
    highlight_color: egui::Color32,
) -> KeyboardDrawResult {
    let desired_size = egui::vec2(
        ui.available_width(),
        key_height.clamp(MIN_PIANO_KEY_HEIGHT, 220.0),
    );
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    let painter = ui.painter_at(rect);

    let white_count = (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI)
        .filter(|midi| !is_black_key(*midi))
        .count();
    let white_w = keyboard_white_key_width(rect.width(), zoom);
    let black_w = white_w * 0.62;
    let black_h = rect.height() * 0.62;
    let total_w = white_w * white_count as f32;
    let max_scroll_px = (total_w - rect.width()).max(0.0);
    let scroll_px = scroll_px.clamp(0.0, max_scroll_px);
    let x_start = rect.left() + ((rect.width() - total_w) * 0.5).max(0.0) - scroll_px;

    let mut white_index = 0usize;
    for midi in PIANO_LOW_MIDI..=PIANO_HIGH_MIDI {
        if is_black_key(midi) {
            continue;
        }

        let x0 = x_start + white_index as f32 * white_w;
        let x1 = x0 + white_w;
        if x1 < rect.left() || x0 > rect.right() {
            white_index += 1;
            continue;
        }
        let key_rect =
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom()));

        painter.rect_filled(key_rect, 0.0, egui::Color32::from_gray(238));
        painter.rect_stroke(
            key_rect,
            0.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(90)),
        );

        let idx = (midi - PIANO_LOW_MIDI) as usize;
        let p = probs.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let s = sensitivity.clamp(0.0, 2.0);
        let gain = s.powf(1.35);
        let adjusted = (p * gain).clamp(0.0, 1.0).powf(1.55);
        let activation_threshold = 0.12 - (s.min(1.5) / 1.5) * 0.06;
        if adjusted >= activation_threshold {
            painter.rect_filled(key_rect, 0.0, highlight_color);
            painter.rect_stroke(
                key_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(90)),
            );
        }

        white_index += 1;
    }

    let mut white_before = 0usize;
    for midi in PIANO_LOW_MIDI..=PIANO_HIGH_MIDI {
        if !is_black_key(midi) {
            white_before += 1;
            continue;
        }

        let center_x = x_start + white_before as f32 * white_w;
        let x0 = center_x - black_w * 0.5;
        let x1 = center_x + black_w * 0.5;
        if x1 < rect.left() || x0 > rect.right() {
            continue;
        }
        let key_rect = egui::Rect::from_min_max(
            egui::pos2(x0, rect.top()),
            egui::pos2(x1, rect.top() + black_h),
        );

        painter.rect_filled(key_rect, 2.0, egui::Color32::from_gray(55));
        painter.rect_stroke(
            key_rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(65)),
        );

        let idx = (midi - PIANO_LOW_MIDI) as usize;
        let p = probs.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let s = sensitivity.clamp(0.0, 2.0);
        let gain = s.powf(1.35);
        let adjusted = (p * gain).clamp(0.0, 1.0).powf(1.55);
        let activation_threshold = 0.12 - (s.min(1.5) / 1.5) * 0.06;
        if adjusted >= activation_threshold {
            painter.rect_filled(key_rect, 2.0, highlight_color);
            painter.rect_stroke(
                key_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(65)),
            );
        }
    }

    // Mark Middle C (MIDI 60).
    if (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI).contains(&60) {
        let c4_white_idx = white_index_before_midi(60);
        let cx = x_start + c4_white_idx as f32 * white_w + white_w * 0.5;
        if cx >= rect.left() && cx <= rect.right() {
            painter.circle_filled(
                egui::pos2(cx, rect.top() + 8.0),
                4.0,
                egui::Color32::from_gray(155),
            );
        }
    }

    KeyboardDrawResult {
        clicked: response.clicked(),
        max_scroll_px,
    }
}

fn draw_probability_pane(
    ui: &mut egui::Ui,
    probs_smoothed: &[f32],
    probs_raw: &[f32],
    zoom: f32,
    scroll_px: f32,
    strip_height: f32,
    highlight_color: egui::Color32,
) -> KeyboardDrawResult {
    let desired_size = egui::vec2(
        ui.available_width(),
        strip_height.max(MIN_PROBABILITY_STRIP_HEIGHT),
    );
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(33, 38, 46));

    let white_count = (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI)
        .filter(|midi| !is_black_key(*midi))
        .count();
    let white_w = keyboard_white_key_width(rect.width(), zoom);
    let black_w = white_w * 0.62;
    let total_w = white_w * white_count as f32;
    let max_scroll_px = (total_w - rect.width()).max(0.0);
    let scroll_px = scroll_px.clamp(0.0, max_scroll_px);
    let x_start = rect.left() + ((rect.width() - total_w) * 0.5).max(0.0) - scroll_px;

    let mut white_index = 0usize;
    for midi in PIANO_LOW_MIDI..=PIANO_HIGH_MIDI {
        if is_black_key(midi) {
            continue;
        }

        let x0 = x_start + white_index as f32 * white_w;
        let x1 = x0 + white_w;
        if x1 < rect.left() || x0 > rect.right() {
            white_index += 1;
            continue;
        }

        let key_rect =
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom()));
        painter.rect_stroke(
            key_rect,
            0.0,
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(120, 120, 120, 60),
            ),
        );

        let idx = (midi - PIANO_LOW_MIDI) as usize;
        let p_raw = probs_raw.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let p_smooth = probs_smoothed
            .get(idx)
            .copied()
            .unwrap_or(p_raw)
            .clamp(0.0, 1.0);

        // Raw probability controls bar height, while smoothed value controls a subtle glow.
        let h = p_raw * (rect.height() - 8.0);
        if h > 0.5 {
            let bar = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, rect.bottom() - h - 2.0),
                egui::pos2(x1 - 1.0, rect.bottom() - 2.0),
            );
            painter.rect_filled(bar, 1.0, highlight_color);
        }

        let glow_h = p_smooth * (rect.height() - 8.0);
        if glow_h > 0.5 {
            let glow = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, rect.bottom() - glow_h - 2.0),
                egui::pos2(x1 - 1.0, rect.bottom() - glow_h - 1.0),
            );
            painter.rect_filled(
                glow,
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 215, 140, 180),
            );
        }

        white_index += 1;
    }

    let black_h = rect.height() * 0.62;
    let mut white_before = 0usize;
    for midi in PIANO_LOW_MIDI..=PIANO_HIGH_MIDI {
        if !is_black_key(midi) {
            white_before += 1;
            continue;
        }

        let center_x = x_start + white_before as f32 * white_w;
        let x0 = center_x - black_w * 0.5;
        let x1 = center_x + black_w * 0.5;
        if x1 < rect.left() || x0 > rect.right() {
            continue;
        }

        let key_rect = egui::Rect::from_min_max(
            egui::pos2(x0, rect.top()),
            egui::pos2(x1, rect.top() + black_h),
        );
        painter.rect_filled(key_rect, 2.0, egui::Color32::from_rgb(28, 31, 38));
        painter.rect_stroke(
            key_rect,
            2.0,
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(160, 160, 170, 80),
            ),
        );

        let idx = (midi - PIANO_LOW_MIDI) as usize;
        let p_raw = probs_raw.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let p_smooth = probs_smoothed
            .get(idx)
            .copied()
            .unwrap_or(p_raw)
            .clamp(0.0, 1.0);

        let h = p_raw * (key_rect.height() - 4.0);
        if h > 0.5 {
            let bar = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, key_rect.bottom() - h - 1.0),
                egui::pos2(x1 - 1.0, key_rect.bottom() - 1.0),
            );
            painter.rect_filled(bar, 1.0, highlight_color);
        }

        let glow_h = p_smooth * (key_rect.height() - 4.0);
        if glow_h > 0.5 {
            let glow = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, key_rect.bottom() - glow_h - 1.0),
                egui::pos2(x1 - 1.0, key_rect.bottom() - glow_h),
            );
            painter.rect_filled(
                glow,
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 230, 170, 210),
            );
        }
    }

    KeyboardDrawResult {
        clicked: response.clicked(),
        max_scroll_px,
    }
}

fn is_black_key(midi: u8) -> bool {
    matches!(midi % 12, 1 | 3 | 6 | 8 | 10)
}

fn setting_toggle_row(ui: &mut egui::Ui, value: &mut bool, label: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.checkbox(value, "").changed();
        let response = ui.add(
            egui::Label::new(label)
                .wrap(false)
                .sense(egui::Sense::click()),
        );
        if response.clicked() {
            *value = !*value;
            changed = true;
        }
    });
    changed
}

fn draw_media_controls(
    app: &mut TranscriberApp,
    ui: &mut egui::Ui,
    analysis_ready: bool,
    duration: f32,
) {
    let art_size = 72.0;
    let panel_fill = if app.dark_mode {
        egui::Color32::from_rgb(19, 28, 38)
    } else {
        egui::Color32::from_rgb(232, 236, 243)
    };

    let target_w = ui.available_width();
    let target_h = art_size + 24.0;

    ui.allocate_ui_with_layout(
        egui::vec2(target_w, target_h),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            egui::Frame::none()
                .fill(panel_fill)
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                .show(ui, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            if let Some(texture) = &app.album_art_texture {
                                ui.add(
                                    egui::Image::new(texture)
                                        .fit_to_exact_size(egui::vec2(art_size, art_size)),
                                );
                            } else {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(art_size, art_size),
                                    egui::Sense::hover(),
                                );
                                let painter = ui.painter_at(rect);
                                painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(38, 49, 63));
                                painter.text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    PLAY,
                                    icon_font_id(20.0),
                                    egui::Color32::from_rgb(177, 192, 210),
                                );
                            }

                            ui.add_space(8.0);

                            ui.allocate_ui_with_layout(
                                egui::vec2(280.0, art_size),
                                egui::Layout::top_down(egui::Align::Min),
                                |ui| {
                                    let fallback_name = app
                                        .loaded_path
                                        .as_ref()
                                        .and_then(|p| p.file_name())
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("Untitled");

                                    let title = app
                                        .audio_raw
                                        .as_ref()
                                        .and_then(|a| a.metadata.title.as_deref())
                                        .unwrap_or(fallback_name);

                                    let artist = app
                                        .audio_raw
                                        .as_ref()
                                        .and_then(|a| a.metadata.artist.as_deref())
                                        .unwrap_or("Unknown Artist");

                                    let album = app
                                        .audio_raw
                                        .as_ref()
                                        .and_then(|a| a.metadata.album.as_deref())
                                        .unwrap_or("");

                                    let title_h = ui
                                        .fonts(|f| f.row_height(&egui::FontId::proportional(17.0)));
                                    let artist_h = ui
                                        .fonts(|f| f.row_height(&egui::FontId::proportional(14.0)));
                                    let block_h = title_h + artist_h + ui.spacing().item_spacing.y;
                                    let top_pad = ((art_size - block_h) * 0.5).max(0.0);
                                    if top_pad > 0.0 {
                                        ui.add_space(top_pad);
                                    }

                                    ui.label(egui::RichText::new(title).size(17.0));
                                    if album.is_empty() {
                                        ui.label(
                                            egui::RichText::new(artist)
                                                .color(egui::Color32::from_rgb(166, 182, 202)),
                                        );
                                    } else {
                                        ui.label(
                                            egui::RichText::new(format!("{artist} · {album}"))
                                                .color(egui::Color32::from_rgb(166, 182, 202)),
                                        );
                                    }
                                },
                            );

                            ui.add_space(14.0);

                            if icon_button(ui, SKIP_BACK, "Go To Start", analysis_ready).clicked() {
                                app.stop();
                                app.selected_time_sec = 0.0;
                                app.update_note_probabilities(true);
                            }

                            let is_playing =
                                app.engine.as_ref().map(|e| e.is_playing()).unwrap_or(false);
                            let play_icon = if is_playing { PAUSE } else { PLAY };

                            if icon_button(ui, play_icon, "Play / Pause", analysis_ready).clicked()
                            {
                                let current_pos = app
                                    .engine
                                    .as_ref()
                                    .map(|e| e.current_position())
                                    .unwrap_or(0.0);

                                if is_playing {
                                    if let Some(engine) = &mut app.engine {
                                        engine.pause();
                                    }
                                } else if app.audio_raw.is_some() {
                                    if app.processed_samples.is_empty() {
                                        app.request_rebuild(false);
                                    }

                                    if current_pos <= 0.0 || current_pos >= duration - 0.01 {
                                        app.play_from_selected();
                                    } else if let Some(engine) = &mut app.engine {
                                        engine.resume();
                                    }
                                }
                            }

                            if icon_button(ui, SKIP_FORWARD, "Go To End", analysis_ready).clicked()
                            {
                                app.stop();
                                app.selected_time_sec = duration.max(0.0);
                                app.update_note_probabilities(true);
                            }

                            ui.add_space(6.0);

                            if icon_toggle_button(
                                ui,
                                REPEAT,
                                "Loop Selection",
                                app.loop_enabled,
                                analysis_ready,
                                app.highlight_color,
                            )
                            .clicked()
                            {
                                app.loop_enabled = !app.loop_enabled;
                                if !app.loop_enabled {
                                    app.loop_playback_enabled = false;
                                }
                            }

                            ui.add_space(8.0);

                            let vol_icon = if app.playback_volume <= 0.01 {
                                SPEAKER_NONE
                            } else {
                                SPEAKER_HIGH
                            };
                            ui.label(egui::RichText::new(vol_icon).font(icon_font_id(17.0)));

                            let vol_changed = ui
                                .add_sized(
                                    [120.0, 20.0],
                                    egui::Slider::new(&mut app.playback_volume, 0.0..=1.5)
                                        .show_value(false),
                                )
                                .changed();
                            if vol_changed {
                                if let Some(engine) = &mut app.engine {
                                    engine.set_volume(app.playback_volume);
                                }
                            }

                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} / {}",
                                    format_time(app.selected_time_sec),
                                    format_time(duration)
                                ))
                                .color(egui::Color32::from_rgb(176, 188, 203)),
                            );
                        });
                    });
                });
        },
    );
}

fn icon_button(ui: &mut egui::Ui, icon: &str, tooltip: &str, enabled: bool) -> egui::Response {
    icon_button_with_fill(ui, icon, tooltip, enabled, None)
}

fn icon_toggle_button(
    ui: &mut egui::Ui,
    icon: &str,
    tooltip: &str,
    enabled_state: bool,
    enabled: bool,
    accent_color: egui::Color32,
) -> egui::Response {
    let fill = if enabled_state {
        accent_color
    } else {
        ui.visuals().widgets.inactive.bg_fill
    };

    icon_button_with_fill(ui, icon, tooltip, enabled, Some(fill))
}

fn icon_button_with_fill(
    ui: &mut egui::Ui,
    icon: &str,
    tooltip: &str,
    enabled: bool,
    fill_override: Option<egui::Color32>,
) -> egui::Response {
    let desired = egui::vec2(34.0, 34.0);
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(desired, sense);
    let response = response.on_hover_text(tooltip);
    let visuals = ui.style().interact(&response);

    let mut bg_fill = fill_override.unwrap_or(visuals.bg_fill);
    if !enabled {
        bg_fill = ui.visuals().widgets.inactive.bg_fill;
    }

    ui.painter()
        .rect(rect, visuals.rounding, bg_fill, visuals.bg_stroke);

    let text_color = if enabled {
        visuals.text_color()
    } else {
        ui.visuals().widgets.inactive.text_color()
    };

    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        icon_font_id(18.0),
        text_color,
    );

    response
}

fn icon_font_id(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name("icons".into()))
}

fn parse_hex_color(hex: &str) -> Option<egui::Color32> {
    let trimmed = hex.trim().trim_start_matches('#');
    if trimmed.len() != 6 {
        return None;
    }

    let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
    let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
    let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
    Some(egui::Color32::from_rgb(r, g, b))
}

fn color_to_hex(color: egui::Color32) -> String {
    format!("#{:02X}{:02X}{:02X}", color.r(), color.g(), color.b())
}

fn push_recent_color(recent: &mut Vec<String>, color: egui::Color32) {
    let hex = color_to_hex(color);
    recent.retain(|item| item != &hex);
    recent.insert(0, hex);
    if recent.len() > 10 {
        recent.truncate(10);
    }
}

fn accent_soft(color: egui::Color32) -> egui::Color32 {
    let r = ((color.r() as u16 + 255) / 2) as u8;
    let g = ((color.g() as u16 + 255) / 2) as u8;
    let b = ((color.b() as u16 + 255) / 2) as u8;
    egui::Color32::from_rgb(r, g, b)
}

fn format_time(sec: f32) -> String {
    let total = sec.max(0.0).floor() as u64;
    let m = total / 60;
    let s = total % 60;
    format!("{m:02}:{s:02}")
}
