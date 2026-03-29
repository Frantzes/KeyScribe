use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Polygon, VLine};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    detect_note_probabilities, pitch_track, waveform_points, PIANO_HIGH_MIDI, PIANO_LOW_MIDI,
};
use crate::audio_io::{load_audio_file, AudioData};
use crate::dsp::apply_speed_and_pitch;
use crate::playback::AudioEngine;
use crate::theme::{apply_brand_theme, ACCENT_ORANGE, ACCENT_ORANGE_SOFT, ERROR_RED};

const STATE_FILE_NAME: &str = ".transcriber_state.json";
const PIANO_ZOOM_MIN: f32 = 0.35;
const PIANO_ZOOM_MAX: f32 = 1.0;
const WHITE_KEY_LENGTH_TO_WIDTH: f32 = 6.3;
const MIN_PIANO_KEY_HEIGHT: f32 = 16.0;
const MIN_PROBABILITY_STRIP_HEIGHT: f32 = 20.0;

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
    show_pitch_track_window: bool,
    show_note_hist_window: bool,
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
            show_pitch_track_window: false,
            show_note_hist_window: true,
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
    pitch_line: Vec<[f64; 2]>,
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
    show_pitch_track_window: bool,
    show_note_hist_window: bool,
    last_state_save_at: Instant,
    waveform_reset_view: bool,
    loop_selection: Option<(f32, f32)>,
    drag_select_anchor_sec: Option<f32>,
    loop_playback_enabled: bool,
}

struct ProcessingResult {
    job_id: u64,
    processed_samples: Vec<f32>,
    waveform: Vec<[f64; 2]>,
    pitch_line: Vec<[f64; 2]>,
}

impl TranscriberApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        apply_brand_theme(&_cc.egui_ctx);
        let persisted = load_persisted_state();

        let mut app = Self {
            loaded_path: None,
            audio_raw: None,
            processed_samples: Vec::new(),
            waveform: Vec::new(),
            pitch_line: Vec::new(),
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
            show_pitch_track_window: persisted.show_pitch_track_window,
            show_note_hist_window: true,
            last_state_save_at: Instant::now(),
            waveform_reset_view: true,
            loop_selection: None,
            drag_select_anchor_sec: None,
            loop_playback_enabled: false,
        };

        if let Some(path) = persisted.last_file {
            if path.exists() {
                match load_audio_file(&path) {
                    Ok(audio) => {
                        app.loaded_path = Some(path);
                        app.audio_raw = Some(audio);
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

    fn import_audio(&mut self) {
        let picked = FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "aac"])
            .pick_file();

        if let Some(path) = picked {
            match load_audio_file(&path) {
                Ok(audio) => {
                    self.loaded_path = Some(path);
                    self.selected_time_sec = 0.0;
                    self.audio_raw = Some(audio);
                    self.request_rebuild(false);
                    self.last_error = None;
                }
                Err(err) => {
                    self.last_error = Some(format!("Failed to load audio: {err}"));
                }
            }
        }
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
        let compute_pitch_track = self.show_pitch_track_window;

        let (tx, rx) = mpsc::channel::<ProcessingResult>();
        self.processing_rx = Some(rx);
        self.active_job_id = Some(job_id);
        self.is_processing = true;
        self.restart_playback_after_processing |= restart_playback;

        thread::spawn(move || {
            let processed_samples =
                apply_speed_and_pitch(raw_samples.as_slice(), speed, pitch_semitones);
            let waveform = waveform_points(&processed_samples, sample_rate, 6000);
            let pitch_line = if compute_pitch_track {
                pitch_track(&processed_samples, sample_rate, 320)
            } else {
                Vec::new()
            };

            let _ = tx.send(ProcessingResult {
                job_id,
                processed_samples,
                waveform,
                pitch_line,
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
                    self.pitch_line = result.pitch_line;
                    self.waveform_reset_view = true;
                    self.is_processing = false;
                    self.processing_rx = None;
                    self.active_job_id = None;
                    self.selected_time_sec = self.selected_time_sec.min(self.duration());
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
            show_pitch_track_window: self.show_pitch_track_window,
            show_note_hist_window: self.show_note_hist_window,
        };

        if let Ok(raw) = serde_json::to_string_pretty(&state) {
            let _ = fs::write(state_file_path(), raw);
        }
    }

    fn update_note_probabilities(&mut self, force: bool) {
        if !force && self.last_prob_update.elapsed() < Duration::from_millis(80) {
            return;
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };

        if self.processed_samples.is_empty() {
            return;
        }

        let center = (self.selected_time_sec.max(0.0) * raw.sample_rate as f32) as usize;
        self.note_probs =
            detect_note_probabilities(&self.processed_samples, raw.sample_rate, center, 4096);

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
            } else if self.loop_playback_enabled {
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
        apply_brand_theme(ctx);

        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.handle_space_replay();
        }

        self.poll_processing_result();
        self.sync_playhead_from_engine();

        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Import Audio").clicked() {
                    self.import_audio();
                }

                let play_icon = if self
                    .engine
                    .as_ref()
                    .map(|e| e.is_playing())
                    .unwrap_or(false)
                {
                    "⏸"
                } else {
                    "▶"
                };

                if ui
                    .add(egui::Button::new(play_icon).min_size(egui::vec2(28.0, 24.0)))
                    .on_hover_text("Play / Pause")
                    .clicked()
                {
                    let is_playing = self
                        .engine
                        .as_ref()
                        .map(|e| e.is_playing())
                        .unwrap_or(false);
                    let current_pos = self
                        .engine
                        .as_ref()
                        .map(|e| e.current_position())
                        .unwrap_or(0.0);

                    if is_playing {
                        if let Some(engine) = &mut self.engine {
                            engine.pause();
                        }
                    } else if self.audio_raw.is_some() {
                        if self.processed_samples.is_empty() {
                            self.request_rebuild(false);
                        }

                        if current_pos <= 0.0 {
                            self.play_from_selected();
                        } else if let Some(engine) = &mut self.engine {
                            engine.resume();
                        }
                    }
                }

                if ui
                    .add(egui::Button::new("⏹").min_size(egui::vec2(28.0, 24.0)))
                    .on_hover_text("Stop")
                    .clicked()
                {
                    self.stop();
                    self.selected_time_sec = 0.0;
                    self.update_note_probabilities(true);
                }

                let speed_changed = ui
                    .add(
                        egui::Slider::new(&mut self.speed, 0.5..=2.0)
                            .text("Speed")
                            .suffix("x"),
                    )
                    .changed();

                let pitch_changed = ui
                    .add(
                        egui::Slider::new(&mut self.pitch_semitones, -12.0..=12.0)
                            .text("Pitch")
                            .suffix(" st"),
                    )
                    .changed();

                if speed_changed || pitch_changed {
                    self.pending_param_change = true;
                }

                if self.loop_selection.is_some() {
                    if ui.button("Clear Loop Selection").clicked() {
                        self.loop_selection = None;
                        self.loop_playback_enabled = false;
                    }
                }

                if ui
                    .checkbox(&mut self.show_pitch_track_window, "Pitch Track Window")
                    .changed()
                    && self.show_pitch_track_window
                {
                    self.request_rebuild(false);
                }

                ui.checkbox(&mut self.show_note_hist_window, "Show Probability Pane");

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

            if let Some(path) = &self.loaded_path {
                ui.label(format!("Loaded: {}", path.display()));
            }

            if let Some(err) = &self.last_error {
                ui.colored_label(ERROR_RED, err);
            }

            if self.is_processing {
                ui.colored_label(
                    egui::Color32::from_rgb(240, 180, 30),
                    "Processing speed/pitch update...",
                );
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

            let duration = self.duration().max(0.01);
            let current = self.selected_time_sec.min(duration);
            ui.label(format!("Cursor: {:.2}s / {:.2}s", current, duration));
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
                ui.label("Import an audio file to begin visualizing waveform and pitch.");
                return;
            }

            let duration = self.duration().max(0.01);
            ui.heading("Waveform");
            let waveform_height = (ui.available_height() - 22.0).max(40.0);
            Plot::new("waveform_plot")
                .height(waveform_height)
                .allow_scroll(false)
                .allow_zoom(false)
                .allow_drag(false)
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
                        plot_ui.line(line.color(ACCENT_ORANGE));
                    }

                    plot_ui
                        .vline(VLine::new(self.selected_time_sec as f64).color(ACCENT_ORANGE_SOFT));

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

                    if drag_started {
                        self.drag_select_anchor_sec = pointer
                            .map(|p| p.x.clamp(0.0, duration as f64) as f32)
                            .or(Some(self.selected_time_sec));
                    }

                    if dragged {
                        if let (Some(anchor), Some(p)) = (
                            self.drag_select_anchor_sec,
                            pointer.map(|p| p.x.clamp(0.0, duration as f64) as f32),
                        ) {
                            self.loop_selection = Some((anchor, p));
                        }
                    }

                    if drag_stopped {
                        if let Some((a, b)) = self.loop_selection {
                            if (a - b).abs() < 0.01 {
                                self.loop_selection = None;
                                self.loop_playback_enabled = false;
                            } else {
                                let start = a.min(b);
                                let end = a.max(b);
                                self.selected_time_sec = start;
                                self.loop_playback_enabled = true;
                                self.play_range(start, Some(end));
                            }
                        }
                        self.drag_select_anchor_sec = None;
                    }

                    if clicked {
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

            ui.label(
                "Use Ctrl + wheel to zoom waveform, Shift + wheel to scroll waveform or keyboard.",
            );
        });
        self.waveform_panel_height = waveform_central.response.rect.height().clamp(120.0, 5000.0);

        if self.show_pitch_track_window {
            let mut open = self.show_pitch_track_window;
            egui::Window::new("Detected Pitch Track")
                .open(&mut open)
                .resizable(true)
                .vscroll(true)
                .show(ctx, |ui| {
                    if self.pitch_line.is_empty() && !self.is_processing {
                        ui.label("Pitch track not computed yet. Adjust speed/pitch or re-import to compute it.");
                    }

                    let duration = self.duration().max(0.01);
                    Plot::new("pitch_track_plot")
                        .height(220.0)
                        .allow_scroll(false)
                        .allow_zoom(true)
                        .include_x(0.0)
                        .include_x(duration as f64)
                        .include_y(PIANO_LOW_MIDI as f64)
                        .include_y(PIANO_HIGH_MIDI as f64)
                        .show(ui, |plot_ui| {
                            let line = Line::new(PlotPoints::from_iter(self.pitch_line.iter().copied()));
                            plot_ui.line(line.color(egui::Color32::from_rgb(80, 140, 240)));
                            plot_ui.vline(VLine::new(self.selected_time_sec as f64).color(egui::Color32::RED));
                        });
                });
            self.show_pitch_track_window = open;
        }

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
            let alpha = (30.0 + adjusted * 180.0).clamp(30.0, 210.0) as u8;
            let overlay = egui::Color32::from_rgba_unmultiplied(255, 150, 50, alpha);
            painter.rect_filled(key_rect.shrink(1.0), 0.0, overlay);
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

        painter.rect_filled(key_rect, 2.0, egui::Color32::from_gray(25));
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
            let alpha = (25.0 + adjusted * 170.0).clamp(25.0, 200.0) as u8;
            let overlay = egui::Color32::from_rgba_unmultiplied(255, 175, 80, alpha);
            painter.rect_filled(key_rect.shrink(1.0), 2.0, overlay);
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
            painter.rect_filled(bar, 1.0, egui::Color32::from_rgb(255, 160, 60));
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
            painter.rect_filled(bar, 1.0, egui::Color32::from_rgb(255, 180, 75));
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
