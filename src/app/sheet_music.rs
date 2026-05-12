use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::TryRecvError;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::leadsheet::{
    cross_validate_beat_sources, debug_chord_notes_to_json, detect_chord_changes_per_bar, generate_lead_sheet_enhanced,
    generate_lead_sheet_foundation, generate_lead_sheet_with_tempo_map, quantize_notes_with_rhythm_map,
    tempo_map_from_beats, Articulation, BeatTrackConfig, ChordSymbolChange,
    CrossValidatedBeats, LeadSheetFoundation, LeadSheetPresetConfig, NoteEvent, QuantizedNote,
    SwingStyle, TimeSignatureSegment,
};

#[cfg(test)]
use crate::leadsheet::TempoSegment;

const SHEET_NOTE_TABLE_LIMIT: usize = 96;
#[allow(dead_code)]
const BEATS_PER_MEASURE: f32 = 4.0;
const MUSICXML_DIVISIONS: i32 = 480;
const GRAND_STAFF_SPLIT_MIDI: u8 = 60;
const MIN_SHEET_NOTE_FRAMES: usize = 2;
const SHEET_SWING_BIAS: bool = true;

impl KeyScribeApp {
    fn estimate_sheet_cursor_offset_sec(
        note_events: &[NoteEvent],
        foundation: &LeadSheetFoundation,
    ) -> f32 {
        if note_events.is_empty()
            || foundation.quantized_notes.is_empty()
            || foundation.tempo_map.is_empty()
        {
            return 0.0;
        }

        let mut by_id: HashMap<u32, f32> = HashMap::with_capacity(note_events.len());
        for note in note_events {
            if note.start_time.is_finite() {
                by_id.insert(note.id, note.start_time.max(0.0));
            }
        }

        let mut offsets: Vec<f32> = Vec::new();
        for note in &foundation.quantized_notes {
            let Some(start_time) = by_id.get(&note.id) else {
                continue;
            };
            let expected_time = crate::leadsheet::tempo_map::time_at_beat(
                note.beat_start,
                foundation.tempo_map.as_slice(),
            );
            if expected_time.is_finite() {
                let delta = *start_time - expected_time;
                if delta.abs() <= 2.0 {
                    offsets.push(delta);
                }
            }
        }

        if offsets.len() < 4 {
            return 0.0;
        }

        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = offsets.len() / 2;
        if offsets.len() % 2 == 0 {
            (offsets[mid - 1] + offsets[mid]) * 0.5
        } else {
            offsets[mid]
        }
    }

    fn active_note_id_for_time(note_events: &[NoteEvent], time_sec: f32) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for note in note_events {
            if time_sec >= note.start_time && time_sec < note.end_time {
                let start = note.start_time;
                if best.map_or(true, |(prev, _)| start < prev) {
                    best = Some((start, note.id));
                }
            }
        }
        best.map(|(_, id)| id)
    }

    pub(super) fn draw_main_content_tabs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.main_content_tab, MainContentTab::Waveform, "Waveform");
            ui.selectable_value(
                &mut self.main_content_tab,
                MainContentTab::SheetMusic,
                "Sheet Music (Experimental, WIP)",
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(stems) = self.separated_stems.clone() {
                    let is_analyzing = self.stem_analysis_rx.is_some();
                    let analysis_ready = !self.stem_analyses.is_empty();

                    if is_analyzing {
                        self.show_visualize_selector = false;
                        self.show_listen_selector = false;
                    }

                    // --- Visualization Selector ---
                    let visualize_btn_text = if is_analyzing {
                        "Analyzing stems...".to_string()
                    } else if self.enabled_stem_indices.is_empty() {
                        "Visualize: Original Mix".to_string()
                    } else {
                        format!(
                            "Visualize: {} / {}",
                            self.enabled_stem_indices.len(),
                            stems.len()
                        )
                    };
                    
                    let visualize_resp = ui.add_enabled(
                        !is_analyzing && analysis_ready,
                        egui::Button::new(visualize_btn_text).min_size(egui::vec2(220.0, 0.0))
                    );
                    if visualize_resp.clicked() {
                        self.show_visualize_selector = !self.show_visualize_selector;
                        if self.show_visualize_selector {
                            self.show_listen_selector = false;
                            self.pending_stem_indices = self.enabled_stem_indices.clone();
                        }
                    }

                    if self.show_visualize_selector {
                        let popup_id = ui.make_persistent_id("visualize_selector_area");
                        let mut pos = visualize_resp.rect.left_bottom();
                        pos.y += 4.0;

                        egui::Area::new(popup_id)
                            .order(egui::Order::Foreground)
                            .fixed_pos(pos)
                            .show(ui.ctx(), |ui| {
                                egui::Frame::popup(ui.style()).show(ui, |ui| {
                                    ui.set_min_width(220.0);
                                    ui.vertical(|ui| {
                                        ui.label("Toggle instrument visualization");
                                        ui.horizontal(|ui| {
                                            if ui.button("Original Mix").clicked() {
                                                self.pending_stem_indices.clear();
                                            }
                                            if ui.button("All").clicked() {
                                                self.pending_stem_indices = (0..stems.len()).collect();
                                            }
                                        });
                                        ui.add_space(UI_VSPACE_TIGHT);

                                        for (i, stem) in stems.iter().enumerate() {
                                            let mut enabled = self.pending_stem_indices.contains(&i);
                                            let label = stem.stem_type.display_name();
                                            let stem_color = self
                                                .stem_colors
                                                .get(i)
                                                .copied()
                                                .unwrap_or(self.highlight_color);
                                            let conf = stem.confidence;
                                            let conf_label = if conf < 0.03 {
                                                " (inactive)"
                                            } else if conf < 0.08 {
                                                " (low)"
                                            } else {
                                                ""
                                            };
                                            ui.horizontal(|ui| {
                                                let (dot_rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(10.0, 10.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter()
                                                    .circle_filled(dot_rect.center(), 4.0, stem_color);
                                                let cb_label = format!("{}{}", label, conf_label);
                                                let cb = ui.checkbox(&mut enabled, cb_label.as_str());
                                                if conf < 0.08 {
                                                    cb.clone().on_hover_text(
                                                        "Low stem energy — may not contain meaningful audio for visualization",
                                                    );
                                                }
                                                if cb.changed() {
                                                    if enabled {
                                                        self.pending_stem_indices.insert(i);
                                                    } else {
                                                        self.pending_stem_indices.remove(&i);
                                                    }
                                                }
                                            });
                                        }

                                        ui.add_space(UI_VSPACE_TIGHT);
                                        let changed = self.pending_stem_indices != self.enabled_stem_indices;
                                        ui.horizontal(|ui| {
                                            if ui.add_enabled(changed, egui::Button::new("Apply Changes")).clicked() {
                                                self.enabled_stem_indices = self.pending_stem_indices.clone();
                                                self.note_timeline = Arc::new(Vec::new());
                                                self.note_timeline_step_sec = 0.0;
                                                self.base_note_timeline = Arc::new(Vec::new());
                                                self.base_note_timeline_step_sec = 0.0;
                                                self.refresh_note_timeline_from_selected_stems_preserving();
                                                self.show_visualize_selector = false;
                                            }
                                            if ui.button("Cancel").clicked() {
                                                self.show_visualize_selector = false;
                                            }
                                        });
                                    });
                                });
                            });
                    }

                    ui.add_space(UI_VSPACE_COMPACT);

                    // --- Listening Selector ---
                    let listen_btn_text = if is_analyzing {
                        "Analyzing stems...".to_string()
                    } else {
                        format!(
                            "Listen: {}",
                            if self.enabled_listening_indices.is_empty() { 
                                "Original Mix".to_string() 
                            } else { 
                                format!("{}/{}", self.enabled_listening_indices.len(), stems.len()) 
                            },
                        )
                    };

                    let listen_resp = ui.add_enabled(
                        !is_analyzing && analysis_ready,
                        egui::Button::new(listen_btn_text).min_size(egui::vec2(180.0, 0.0))
                    );
                    if listen_resp.clicked() {
                        self.show_listen_selector = !self.show_listen_selector;
                        if self.show_listen_selector {
                            self.show_visualize_selector = false;
                            self.pending_listening_indices = self.enabled_listening_indices.clone();
                        }
                    }

                    if self.show_listen_selector {
                        let popup_id = ui.make_persistent_id("listen_selector_area");
                        let mut pos = listen_resp.rect.left_bottom();
                        pos.y += 4.0;

                        egui::Area::new(popup_id)
                            .order(egui::Order::Foreground)
                            .fixed_pos(pos)
                            .show(ui.ctx(), |ui| {
                                egui::Frame::popup(ui.style()).show(ui, |ui| {
                                    ui.set_min_width(180.0);
                                    ui.vertical(|ui| {
                                        ui.label("Toggle audio playback");
                                        ui.horizontal(|ui| {
                                            if ui.button("Original Mix").clicked() {
                                                self.pending_listening_indices.clear();
                                            }
                                            if ui.button("All").clicked() {
                                                self.pending_listening_indices = (0..stems.len()).collect();
                                            }
                                            if ui.button("None").clicked() {
                                                self.pending_listening_indices.clear();
                                            }
                                        });
                                        ui.add_space(UI_VSPACE_TIGHT);

                                        for (i, stem) in stems.iter().enumerate() {
                                            let mut enabled = self.pending_listening_indices.contains(&i);
                                            let label = stem.stem_type.display_name();
                                            let stem_color = self
                                                .stem_colors
                                                .get(i)
                                                .copied()
                                                .unwrap_or(self.highlight_color);
                                            let conf = stem.confidence;
                                            let conf_label = if conf < 0.03 {
                                                " (inactive)"
                                            } else if conf < 0.08 {
                                                " (low)"
                                            } else {
                                                ""
                                            };
                                            ui.horizontal(|ui| {
                                                let (dot_rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(10.0, 10.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter()
                                                    .circle_filled(dot_rect.center(), 4.0, stem_color);
                                                let cb_label = format!("{}{}", label, conf_label);
                                                let cb = ui.checkbox(&mut enabled, cb_label.as_str());
                                                if conf < 0.08 {
                                                    cb.clone().on_hover_text(
                                                        "Low stem energy — may not contain meaningful audio",
                                                    );
                                                }
                                                if cb.changed() {
                                                    if enabled {
                                                        self.pending_listening_indices.insert(i);
                                                    } else {
                                                        self.pending_listening_indices.remove(&i);
                                                    }
                                                }
                                            });
                                        }

                                        ui.add_space(UI_VSPACE_TIGHT);
                                        let changed = self.pending_listening_indices != self.enabled_listening_indices;
                                        ui.horizontal(|ui| {
                                            if ui.add_enabled(changed, egui::Button::new("Apply Changes")).clicked() {
                                                let normalize_to_original_mix =
                                                    self.pending_listening_indices.len() == stems.len();
                                                if normalize_to_original_mix {
                                                    self.enabled_listening_indices.clear();
                                                } else {
                                                    self.enabled_listening_indices =
                                                        self.pending_listening_indices.clone();
                                                }
                                                self.maybe_restart_playback_for_listen_sync();
                                                self.show_listen_selector = false;
                                            }
                                            if ui.button("Cancel").clicked() {
                                                self.show_listen_selector = false;
                                            }
                                        });
                                    });
                                });
                            });
                    }
                }
            });
        });
    }

    pub(super) fn draw_sheet_music_view(
        &mut self,
        ui: &mut egui::Ui,
        _interaction_ready: bool,
        _interaction_duration: f32,
        _default_stack_spacing_y: f32,
        _vertical_gap: f32,
        content_height: f32,
    ) {
        // Show config modal if open
        if self.sheet_config_modal_open {
            self.draw_sheet_config_modal(ui);
        }

        let has_engraving = !self.sheet_engraving_pages.is_empty()
            || self.sheet_preview_cache.is_some();

        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;

            if !has_engraving {
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), content_height),
                    egui::Sense::click(),
                );

                // Draw the background box
                ui.painter().rect(
                    rect,
                    egui::Rounding::same(8.0),
                    ui.visuals().extreme_bg_color,
                    egui::Stroke::new(2.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
                );

                if self.sheet_preview_result_rx.is_some() {
                    ui.allocate_ui_at_rect(rect, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(content_height * 0.4);
                            ui.add(egui::Spinner::new().size(32.0));
                            ui.add_space(10.0);
                            ui.label(
                                egui::RichText::new("Analyzing audio and generating sheet music...")
                                    .strong(),
                            );
                        });
                    });
                } else {
                    // ---- large clickable placeholder ----
                    let text = "Click to configure sheet music";
                    let text_font = egui::FontId::proportional(20.0);
                    let text_color = ui.visuals().weak_text_color();
                    let galley = ui.painter().layout_no_wrap(text.to_owned(), text_font, text_color);
                    let galley_pos =
                        rect.center() - egui::vec2(galley.size().x / 2.0, galley.size().y / 2.0);
                    ui.painter().galley(galley_pos, galley, text_color);
                    if resp.clicked() {
                        self.sheet_config_modal_open = true;
                    }
                }
            } else {
                let preview = self.sheet_preview_cache.clone();

                // Row 1: Status + Re-generate + Updating Spinner
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Sheet music generated").strong());
                    if ui.button("Generate again").clicked() {
                        self.sheet_config_modal_open = true;
                    }
                    
                    if self.sheet_preview_result_rx.is_some() {
                        ui.add_space(12.0);
                        ui.add(egui::Spinner::new().size(14.0));
                        ui.label(egui::RichText::new("Updating...").weak());
                    }
                });

                ui.add_space(4.0);

                // Row 2: Export actions
                ui.horizontal(|ui| {
                    let can_export = preview.is_some();
                    if ui.add_enabled(can_export, egui::Button::new("Export MusicXML")).clicked() {
                        self.export_sheet_musicxml(ui.ctx());
                    }
                    if ui.add_enabled(can_export, egui::Button::new("Export Engraved PDF")).clicked() {
                        self.export_sheet_pdf(ui.ctx());
                    }
                    if ui.add_enabled(can_export, egui::Button::new("Open in MuseScore")).clicked() {
                        self.open_in_musescore(ui.ctx());
                    }
                });

                ui.add_space(4.0);

                // Error reporting
                if let Some(err) = self.sheet_preview_error.as_deref() {
                    ui.colored_label(ERROR_RED, err);
                }
                if let Some(err) = self.sheet_engraving_error.as_deref() {
                    ui.colored_label(ERROR_RED, err);
                }

                // Score Area - Fills all remaining space
                if let Some(data) = preview.as_ref() {
                    let playback_time = (self.selected_time_sec
                        + self.visualization_timing_offset_ms / 1000.0)
                        .max(0.0);
                    let active_note_id =
                        Self::active_note_id_for_time(&data.melody_events, playback_time);
                    let cursor_time = (playback_time - data.cursor_offset_sec).max(0.0);
                    let current_beat = data.foundation.beat_at_time(cursor_time);
                    
                    if self.sheet_engraving_pages.is_empty() && self.sheet_engraving_error.is_none() {
                        ui.centered_and_justified(|ui| {
                            ui.label("Engraving is being prepared...");
                        });
                    } else {
                        draw_scrollable_engraved_preview(
                            ui,
                            self.sheet_engraving_pages.as_slice(),
                            current_beat,
                            active_note_id,
                            self.highlight_color,
                        );
                    }
                }
            }
        });
    }

    fn draw_sheet_config_modal(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::Window::new("Sheet Music Configuration")
            .id(egui::Id::new("sheet_config_modal"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(&ctx, |ui| {
                egui::Grid::new("sheet_config_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(12.0, 6.0))
                    .striped(true)
                    .show(ui, |ui| {
                        // Mode
                        ui.label("Mode").on_hover_text("Lead Sheet: single staff with melody + chords. Piano Grand Staff: both hands. Single Staff: one staff.");
                        ui.horizontal(|ui| {
                            for mode in &[SheetMusicMode::LeadSheet, SheetMusicMode::PianoGrandStaff, SheetMusicMode::SingleStaff] {
                                ui.selectable_value(&mut self.sheet_music_mode, *mode, mode.label());
                            }
                        });
                        ui.end_row();

                        // Melody source
                        ui.label("Melody source").on_hover_text("Which stems to use for melody extraction. Empty = Full Mix (all enabled stems combined).");
                        let mel_label = if self.melody_stem_indices.is_empty() {
                            "Full Mix".to_string()
                        } else if let Some(stems) = self.separated_stems.as_ref() {
                            self.melody_stem_indices.iter()
                                .filter_map(|i| stems.get(*i))
                                .map(|s| s.stem_type.display_name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        } else {
                            format!("{} stem(s)", self.melody_stem_indices.len())
                        };
                        let mel_btn = ui.button(mel_label);
                        if mel_btn.clicked() {
                            self.melody_stem_selector_open = !self.melody_stem_selector_open;
                            self.chord_stem_selector_open = false;
                        }
                        if self.melody_stem_selector_open {
                            let popup_id = ui.make_persistent_id("modal_melody_selector");
                            egui::Area::new(popup_id)
                                .order(egui::Order::Foreground)
                                .fixed_pos(mel_btn.rect.left_bottom() + egui::vec2(0.0, 4.0))
                                .show(ui.ctx(), |ui| {
                                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                                        ui.set_min_width(180.0);
                                        ui.horizontal(|ui| {
                                            if ui.button("Full Mix").clicked() { self.melody_stem_indices.clear(); }
                                            if let Some(stems) = self.separated_stems.as_ref() {
                                                if ui.button("All").clicked() { self.melody_stem_indices = (0..stems.len()).collect(); }
                                            }
                                            if ui.button("None").clicked() { self.melody_stem_indices.clear(); }
                                        });
                                        if let Some(stems) = self.separated_stems.as_ref() {
                                            for (i, stem) in stems.iter().enumerate() {
                                                let mut enabled = self.melody_stem_indices.contains(&i);
                                                if ui.checkbox(&mut enabled, stem.stem_type.display_name()).changed() {
                                                    if enabled { self.melody_stem_indices.insert(i); } else { self.melody_stem_indices.remove(&i); }
                                                }
                                            }
                                        }
                                        if ui.button("Done").clicked() { self.melody_stem_selector_open = false; }
                                    });
                                });
                        }
                        ui.end_row();

                        // Chord source
                        ui.label("Chord source").on_hover_text("Which stems to use for chord detection. Empty = Full Mix. 'Off' disables chord symbols.");
                        let chord_label = if self.chord_skip {
                            "Off".to_string()
                        } else if self.chord_stem_indices.is_empty() {
                            "Full Mix".to_string()
                        } else if let Some(stems) = self.separated_stems.as_ref() {
                            self.chord_stem_indices.iter()
                                .filter_map(|i| stems.get(*i))
                                .map(|s| s.stem_type.display_name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        } else {
                            format!("{} stem(s)", self.chord_stem_indices.len())
                        };
                        let chord_btn = ui.button(chord_label);
                        if chord_btn.clicked() {
                            self.chord_stem_selector_open = !self.chord_stem_selector_open;
                            self.melody_stem_selector_open = false;
                        }
                        if self.chord_stem_selector_open {
                            let popup_id = ui.make_persistent_id("modal_chord_selector");
                            egui::Area::new(popup_id)
                                .order(egui::Order::Foreground)
                                .fixed_pos(chord_btn.rect.left_bottom() + egui::vec2(0.0, 4.0))
                                .show(ui.ctx(), |ui| {
                                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                                        ui.set_min_width(180.0);
                                        ui.horizontal(|ui| {
                                            if ui.button("Off").clicked() { self.chord_skip = true; self.chord_stem_indices.clear(); }
                                            if ui.button("Full Mix").clicked() { self.chord_skip = false; self.chord_stem_indices.clear(); }
                                            if let Some(stems) = self.separated_stems.as_ref() {
                                                if ui.button("All").clicked() { self.chord_skip = false; self.chord_stem_indices = (0..stems.len()).collect(); }
                                            }
                                        });
                                        if let Some(stems) = self.separated_stems.as_ref() {
                                            for (i, stem) in stems.iter().enumerate() {
                                                let mut enabled = self.chord_stem_indices.contains(&i);
                                                if ui.checkbox(&mut enabled, stem.stem_type.display_name()).changed() {
                                                    self.chord_skip = false;
                                                    if enabled { self.chord_stem_indices.insert(i); } else { self.chord_stem_indices.remove(&i); }
                                                }
                                            }
                                        }
                                        if ui.button("Done").clicked() { self.chord_stem_selector_open = false; }
                                    });
                                });
                        }
                        ui.end_row();

                        // BPM
                        ui.label("Tempo (BPM)").on_hover_text("Override detected tempo. Empty = auto-detect from audio.");
                        ui.horizontal(|ui| {
                            let mut bpm_invalid = false;
                            if !self.bpm_input_str.trim().is_empty() && self.manual_bpm.is_none() {
                                bpm_invalid = true;
                            }
                            let bpm_resp = ui.add(
                                egui::TextEdit::singleline(&mut self.bpm_input_str)
                                    .desired_width(60.0)
                                    .hint_text("Auto")
                                    .text_color(if bpm_invalid { egui::Color32::RED } else { egui::Color32::WHITE }),
                            );
                            if bpm_resp.lost_focus() {
                                let trimmed = self.bpm_input_str.trim().to_string();
                                if trimmed.is_empty() {
                                    self.manual_bpm = None;
                                } else if let Ok(bpm) = trimmed.parse::<f32>() {
                                    let clamped = bpm.clamp(30.0, 400.0);
                                    self.manual_bpm = Some(clamped);
                                    self.bpm_input_str = format!("{:.0}", clamped);
                                } else {
                                    self.manual_bpm = None;
                                }
                            }
                            if bpm_invalid {
                                ui.label(egui::RichText::new("invalid").color(egui::Color32::RED).weak());
                            }
                            if self.manual_bpm.is_some() {
                                if ui.button("×2").clicked() {
                                    let base = self.manual_bpm.unwrap_or(120.0);
                                    let clamped = (base * 2.0).clamp(30.0, 400.0);
                                    self.manual_bpm = Some(clamped);
                                    self.bpm_input_str = format!("{:.0}", clamped);
                                }
                                if ui.button("÷2").clicked() {
                                    let base = self.manual_bpm.unwrap_or(120.0);
                                    let clamped = (base / 2.0).clamp(30.0, 400.0);
                                    self.manual_bpm = Some(clamped);
                                    self.bpm_input_str = format!("{:.0}", clamped);
                                }
                                if ui.button("Clear").clicked() {
                                    self.manual_bpm = None;
                                    self.bpm_input_str.clear();
                                }
                            }
                        });
                        ui.end_row();

                        // Feel
                        ui.label("Rhythmic feel").on_hover_text("Override swing detection. 'Auto' detects from audio. 'Straight' forces even 8ths. 'Swing' forces swung 8ths.");
                        ui.horizontal(|ui| {
                            let feels = [
                                (None, "Auto"),
                                (Some(crate::leadsheet::SwingStyle::Straight), "Straight"),
                                (Some(crate::leadsheet::SwingStyle::Swing), "Swing"),
                                (Some(crate::leadsheet::SwingStyle::Triplet), "Triplet"),
                            ];
                            for (val, label) in &feels {
                                let selected = self.manual_swing == *val;
                                if ui.selectable_label(selected, *label).clicked() {
                                    self.manual_swing = *val;
                                }
                            }
                        });
                        ui.end_row();

                        // Polyphony
                        ui.label("Polyphony").on_hover_text("Monophonic: single note line. Polyphonic: preserves chords. Heuristic applies skyline + near-note continuity with outlier suppression.");
                        ui.horizontal(|ui| {
                            let mono = self.melody_mode == MelodyMode::Monophonic;
                            if ui.selectable_label(mono, "Monophonic").clicked() {
                                self.melody_mode = MelodyMode::Monophonic;
                            }
                            if ui.selectable_label(!mono, "Polyphonic").clicked() {
                                self.melody_mode = MelodyMode::Polyphonic;
                            }
                            if mono {
                                let mut h = self.melody_heuristic;
                                if ui.checkbox(&mut h, "Heuristic").changed() {
                                    self.melody_heuristic = h;
                                }
                                if self.melody_heuristic {
                                    ui.add(
                                        egui::Slider::new(&mut self.melody_outlier_semitones, 3u8..=24u8)
                                            .text("σ"),
                                    ).on_hover_text("Outlier threshold: melody jumps larger than this many semitones from the rolling median are suppressed. Lower = smoother line, higher = allows more leaps");
                                }
                            }
                        });
                        ui.end_row();
                    });

                ui.add_space(UI_VSPACE_MEDIUM);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.sheet_config_modal_open = false;
                    }
                    if ui.button("Generate Sheet Music").clicked() {
                        self.sheet_config_modal_open = false;
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                        self.refresh_sheet_preview_if_needed(ui.ctx());
                    }
                });
            });
    }

    fn sheet_preview_threshold(&self) -> f32 {
        (NOTE_HIGHLIGHT_ACTIVATION_THRESHOLD / self.key_color_sensitivity.max(0.05)).clamp(0.05, 0.95)
    }

    fn stem_label(&self, idx: usize) -> String {
        self.separated_stems
            .as_ref()
            .and_then(|stems| stems.get(idx))
            .map(|s| s.stem_type.display_name().to_string())
            .unwrap_or_else(|| format!("Stem {}", idx))
    }

    fn current_sheet_preview_key(&self) -> Option<SheetPreviewCacheKey> {
        // Per-stem analyses take priority; fall back to blended note_timeline
        let has_timeline = !self.stem_analyses.is_empty()
            || (!self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0);
        if !has_timeline {
            return None;
        }

        let mut separation_bits = 0u64;
        for &idx in &self.enabled_stem_indices {
            if idx < 64 {
                separation_bits |= 1 << idx;
            }
        }

        // When per-stem analyses are active, encode their combined state into the key
        let stem_key: u64 = if !self.stem_analyses.is_empty() {
            let mut hash: u64 = 0;
            for a in &self.stem_analyses {
                hash = hash.wrapping_mul(31).wrapping_add(a.stem_index as u64);
                hash = hash.wrapping_mul(31).wrapping_add(
                    Arc::as_ptr(&a.timeline) as usize as u64,
                );
                hash = hash.wrapping_mul(31).wrapping_add(a.step_sec.to_bits() as u64);
            }
            hash
        } else {
            0
        };

        let note_min_bits = self.melody_min_note.map(|m| m as u32).unwrap_or(0);
        let note_max_bits = self.melody_max_note.map(|m| m as u32).unwrap_or(0);

        // Encode melody stem indices as bitmask
        let mut melody_stem_bits = 0u64;
        for &idx in &self.melody_stem_indices {
            if idx < 64 {
                melody_stem_bits |= 1 << idx;
            }
        }

        // Encode chord stem indices as bitmask
        let mut chord_stem_bits = 0u64;
        for &idx in &self.chord_stem_indices {
            if idx < 64 {
                chord_stem_bits |= 1 << idx;
            }
        }

        Some(SheetPreviewCacheKey {
            timeline_ptr: Arc::as_ptr(&self.note_timeline) as usize,
            timeline_len: self.note_timeline.len(),
            timeline_step_bits: self.note_timeline_step_sec.to_bits(),
            threshold_bits: self.sheet_preview_threshold().to_bits(),
            separation_selection_bits: separation_bits,
            mode_bits: self.sheet_music_mode as u8,
            melody_stem_bits,
            chord_stem_bits,
            swing_style_bit: self.manual_swing.map(|s| s as u8),
            stem_analysis_key: stem_key,
            melody_note_range_bits: (note_min_bits << 8) | note_max_bits,
            melody_mode_bits: (self.melody_mode as u8 as u32) << 24
                | ((self.melody_heuristic as u8 as u32) << 23)
                | (self.melody_outlier_semitones as u32) << 16,
            use_musescore: self.sheet_use_musescore,
        })
    }

    fn refresh_sheet_preview_if_needed(&mut self, ctx: &egui::Context) {
        let Some(key) = self.current_sheet_preview_key() else {
            self.sheet_preview_cache_key = None;
            self.sheet_preview_cache = None;
            self.sheet_preview_error = Some("No timeline is available yet. Run transcription first.".to_string());
            self.sheet_engraving_cache_key = None;
            self.sheet_engraving_pages.clear();
            self.sheet_engraving_error = None;
            return;
        };

        if self.sheet_preview_cache_key == Some(key) {
            return;
        }

        if self.sheet_preview_result_rx.is_some() {
            return;
        }

        self.start_sheet_preview_build(ctx, key);
    }

    fn start_sheet_preview_build(&mut self, ctx: &egui::Context, key: SheetPreviewCacheKey) {
        let threshold = self.sheet_preview_threshold();

        // Extract melody notes from the selected melody source(s) or combined timeline
        let melody_events = self.extract_notes_for_stems(&self.melody_stem_indices, threshold);
        if melody_events.is_empty() {
            if !self.melody_stem_indices.is_empty() {
                self.sheet_preview_error = Some("Selected melody stem analysis not yet ready. Please wait for per-stem analysis to complete.".to_string());
            } else {
                self.sheet_preview_error = Some("Not enough note events to infer tempo and sheet layout.".to_string());
            }
            self.sheet_preview_cache_key = Some(key);
            self.sheet_preview_cache = None;
            return;
        }
        if melody_events.len() < 4 {
            self.sheet_preview_error = Some("Not enough note events to infer tempo and sheet layout.".to_string());
            self.sheet_preview_cache_key = Some(key);
            self.sheet_preview_cache = None;
            return;
        }

        let stems = self.separated_stems.as_ref();
        let sample_rate = stems
            .and_then(|s| s.first().map(|st| st.sample_rate))
            .or_else(|| self.audio_raw.as_ref().map(|r| r.sample_rate))
            .unwrap_or(44100);

        let bass_audio: Option<Vec<f32>> = stems.and_then(|stems| {
            stems
                .iter()
                .find(|s| s.stem_type == StemType::Bass)
                .map(|s| s.samples_mono.to_vec())
        });
        let drum_audio: Option<Vec<f32>> = stems.and_then(|stems| {
            stems
                .iter()
                .find(|s| s.stem_type == StemType::Drums)
                .map(|s| s.samples_mono.to_vec())
        });
        let full_mix: Option<Vec<f32>> =
            self.audio_raw.as_ref().map(|r| r.samples_mono.to_vec());

        let chord_notes = if !self.chord_skip && !self.chord_stem_indices.is_empty() {
            Some(self.extract_notes_for_stems(&self.chord_stem_indices, threshold))
        } else {
            None
        };

        let job = SheetPreviewJob {
            key,
            threshold,
            melody_events,
            melody_mode: self.melody_mode,
            melody_outlier_semitones: self.melody_outlier_semitones,
            melody_heuristic: self.melody_heuristic,
            bass_audio,
            drum_audio,
            full_mix,
            sample_rate,
            manual_bpm: self.manual_bpm,
            chord_skip: self.chord_skip,
            chord_notes,
            source_duration: self.source_duration(),
        };

        let (tx, rx) = std::sync::mpsc::channel::<(SheetPreviewCacheKey, Result<SheetPreviewData, String>)>();
        self.sheet_preview_result_rx = Some(rx);

        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = Self::run_preview_background(job);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn run_preview_background(job: SheetPreviewJob) -> (SheetPreviewCacheKey, Result<SheetPreviewData, String>) {
        let threshold = job.threshold;
        let melody_events = job.melody_events;

        // Reduce melody based on selected mode
        let note_events = match job.melody_mode {
            MelodyMode::Polyphonic => melody_events,
            MelodyMode::Monophonic if !job.melody_heuristic => {
                extract_melody_skyline(&melody_events, job.melody_outlier_semitones)
            }
            MelodyMode::Monophonic => {
                extract_melody_heuristic(&melody_events, job.melody_outlier_semitones)
            }
        };

        if note_events.len() < 4 {
            return (job.key, Err("Not enough melody notes after reduction.".to_string()));
        }

        let (beat_track, bpm_source) = if job.sample_rate > 0 {
            let beat_config = BeatTrackConfig::default();
            match cross_validate_beat_sources(
                job.bass_audio.as_deref(),
                job.drum_audio.as_deref(),
                job.full_mix.as_deref(),
                job.sample_rate,
                &beat_config,
            ) {
                Ok(cv) => {
                    let src = if cv.source_count > 1 {
                        format!("BeatThis cross-validated ({} sources)", cv.source_count)
                    } else {
                        "BeatThis ML".to_string()
                    };
                    (Some(cv), src)
                }
                Err(e) => {
                    eprintln!("BeatThis Python execution failed: {:?}", e);
                    let duration = job.source_duration;
                    match crate::leadsheet::detect_beats_from_stems(
                        job.bass_audio.as_deref(),
                        job.drum_audio.as_deref(),
                        job.full_mix.as_deref(),
                        job.sample_rate,
                        duration,
                    ) {
                        Some((bt, src)) => (Some(bt.into()), src),
                        None => (None, "Note onsets (fallback)".to_string()),
                    }
                }
            }
        } else {
            (None, "Unknown".to_string())
        };

        let beat_track = beat_track.or_else(|| {
            crate::leadsheet::detect_beats_from_notes(&note_events).map(CrossValidatedBeats::from)
        });

        let beat_track = if let Some(manual_bpm) = job.manual_bpm {
            let beat_duration = 60.0 / manual_bpm.clamp(30.0, 400.0);
            let duration = job.source_duration;
            let total_sec = duration.max(10.0) + 2.0;
            let mut beats: Vec<f32> = Vec::new();
            let mut t = 0.0f32;
            while t <= total_sec + 1e-3 {
                beats.push(t);
                t += beat_duration;
            }
            let downbeats: Vec<f32> = beats.iter().step_by(4).copied().collect();
            Some(CrossValidatedBeats {
                beats,
                downbeats,
                beats_per_bar: 4,
                bpm: manual_bpm,
                confidence: 1.0,
                source_count: 1,
            })
        } else {
            beat_track
        };

        let mut config = LeadSheetPresetConfig::default();
        config.quantization.min_duration_beats = 0.5;
        config.quantization.grids = vec![1.0, 0.5];
        config.quantization.duration_grids = vec![1.0, 0.5];
        config.chord_analysis.skip = job.chord_skip;
        let mut foundation = None;

        if let Some(bt) = beat_track.as_ref() {
            foundation = generate_lead_sheet_enhanced(
                &note_events,
                bt.beats.as_slice(),
                bt.downbeats.as_slice(),
                bt.beats_per_bar,
                &config,
            );
        }

        let fallback_bt = beat_track.as_ref().map(|bt| bt.clone().into());
        let foundation_res = foundation
            .or_else(|| {
                fallback_bt.as_ref().and_then(|bt: &crate::leadsheet::BeatTrackResult| {
                    tempo_map_from_beats(bt.beats.as_slice()).and_then(|(tempo, tempo_map)| {
                        generate_lead_sheet_with_tempo_map(
                            &note_events,
                            tempo,
                            tempo_map,
                            &config,
                        )
                    })
                })
            })
            .or_else(|| generate_lead_sheet_foundation(&note_events, &config))
            .ok_or_else(|| {
                "Tempo-map detection/quantization failed for the current selection.".to_string()
            });

        let mut foundation = match foundation_res {
            Ok(f) => f,
            Err(e) => return (job.key, Err(e)),
        };

        if foundation.quantized_notes.is_empty() {
            return (job.key, Err("No quantized notes available for engraving.".to_string()));
        }

        if !job.chord_skip {
            if let Some(chord_notes) = job.chord_notes {
                if chord_notes.len() >= 4 {
                    let chord_quantized = quantize_notes_with_rhythm_map(
                        &chord_notes,
                        foundation.tempo_map.as_slice(),
                        foundation.time_signature_segments.as_slice(),
                        &config.quantization,
                    );
                    if !chord_quantized.is_empty() {
                        let mut chord_config = config.chord_analysis;
                        chord_config.skip = false;
                        foundation.chord_changes =
                            detect_chord_changes_per_bar(chord_quantized.as_slice(), foundation.beats_per_bar, chord_config);
                        debug_chord_notes_to_json(chord_quantized.as_slice(), foundation.beats_per_bar, chord_config);
                    }
                }
            }
        }

        let bpm_source = if job.manual_bpm.is_some() {
            "Manual".to_string()
        } else {
            bpm_source
        };

        let cursor_offset_sec = Self::estimate_sheet_cursor_offset_sec(&note_events, &foundation);
        let note_count = note_events.len();
        let melody_events = note_events;

        (job.key, Ok(SheetPreviewData {
            foundation,
            note_count,
            threshold,
            bpm_source,
            cursor_offset_sec,
            melody_events,
        }))
    }

    pub(super) fn poll_sheet_preview(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.sheet_preview_result_rx else { return };
        match rx.try_recv() {
            Ok((key, result)) => {
                self.sheet_preview_result_rx = None;
                match result {
                    Ok(preview) => {
                        self.sheet_preview_cache_key = Some(key);
                        self.sheet_preview_cache = Some(preview.clone());
                        self.sheet_preview_error = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                        self.sheet_engraving_error = None;
                        
                        // Automatically trigger engraving now that we have the foundation
                        self.refresh_engraved_preview_if_needed(ctx, &preview);
                    }
                    Err(err) => {
                        self.sheet_preview_cache_key = Some(key);
                        self.sheet_preview_cache = None;
                        self.sheet_preview_error = Some(err);
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                        self.sheet_engraving_error = None;
                    }
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.sheet_preview_result_rx = None;
            }
        }
    }

    fn refresh_engraved_preview_if_needed(&mut self, ctx: &egui::Context, preview: &SheetPreviewData) {
        let Some(key) = self.current_sheet_preview_key() else {
            self.sheet_engraving_cache_key = None;
            self.sheet_engraving_pages.clear();
            self.sheet_engraving_error = None;
            return;
        };

        if self.sheet_engraving_cache_key == Some(key)
            && (!self.sheet_engraving_pages.is_empty() || self.sheet_engraving_error.is_some())
        {
            return;
        }

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let musicxml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );
        let _total_beats = total_beats_for_foundation(&preview.foundation);

        // Submit engraving job to background thread
        self.start_sheet_render(ctx, &musicxml, key);
    }

    fn start_sheet_render(
        &mut self,
        ctx: &egui::Context,
        musicxml: &str,
        key: SheetPreviewCacheKey,
    ) {
        let dpi_scale = ctx.pixels_per_point().max(1.0);
        let job = SheetRenderJob {
            musicxml: musicxml.to_string(),
            key,
        };
        let (tx, rx) = std::sync::mpsc::channel::<SheetRenderResult>();
        self.sheet_render_result_rx = Some(rx);

        thread::spawn(move || {
            let result = Self::run_render_background(&job, dpi_scale);
            let _ = tx.send(result);
        });
    }

    pub(super) fn poll_sheet_rendering(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.sheet_render_result_rx else { return };
        match rx.try_recv() {
            Ok(result) => {
                self.sheet_render_result_rx = None;
                if let Some(err) = result.error {
                    self.sheet_engraving_cache_key = Some(result.key);
                    self.sheet_engraving_pages.clear();
                    self.sheet_engraving_error = Some(err);
                } else {
                    // Convert raw RGBA data to egui textures on the main thread
                    let mut pages = Vec::with_capacity(result.pages.len());
                    for (page_idx, raw) in result.pages.iter().enumerate() {
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(
                            [raw.width_px, raw.height_px],
                            &raw.rgba_data,
                        );
                        let texture = ctx.load_texture(
                            format!(
                                "sheet-engraved-{}-{}-{}-{}",
                                result.key.timeline_ptr,
                                result.key.timeline_len,
                                result.key.timeline_step_bits,
                                page_idx
                            ),
                            color_image,
                            egui::TextureOptions::LINEAR,
                        );
                        pages.push(EngravedSheetPage {
                            texture,
                            width_px: raw.width_px,
                            height_px: raw.height_px,
                            note_positions: raw.note_positions.clone(),
                            beat_start: 0.0,
                            beat_end: 0.0,
                        });
                    }
                    self.sheet_engraving_cache_key = Some(result.key);
                    self.sheet_engraving_pages = pages;
                    self.sheet_engraving_error = None;
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sheet_render_result_rx = None;
            }
        }
    }

    /// Runs in a background thread: MusicXML → verovioxide → raw RGBA pages + note positions
    fn run_render_background(
        job: &SheetRenderJob,
        dpi_scale: f32,
    ) -> SheetRenderResult {
        use verovioxide::{Options, Png, Svg, Toolkit};

        let key = job.key;
        let render_result = (|| -> Result<Vec<SheetRawPage>, String> {
            let mut toolkit = Toolkit::new().map_err(|err| format!("verovioxide init failed: {err}"))?;
            toolkit
                .load_data(&job.musicxml)
                .map_err(|err| format!("verovioxide could not parse MusicXML: {err}"))?;

            let opts = Options::builder()
                .svg_bounding_boxes(true)
                .build();
            toolkit.set_options(&opts)
                .map_err(|err| format!("verovioxide options failed: {err}"))?;

            let svg_pages: Vec<String> = toolkit
                .render(Svg::all_pages())
                .map_err(|err| format!("verovioxide SVG render failed: {err}"))?;

            let render_w = ((3000.0 * dpi_scale).ceil() as u32).min(4096);
            let png_pages: Vec<Vec<u8>> = toolkit
                .render(Png::all_pages().width(render_w).white_background())
                .map_err(|err| format!("verovioxide PNG render failed: {err}"))?;

            if png_pages.is_empty() {
                return Err("verovioxide returned zero rendered pages.".to_string());
            }

            let mut raw_pages = Vec::with_capacity(png_pages.len());
            for (page_idx, page_bytes) in png_pages.into_iter().enumerate() {
                let rgba = image::load_from_memory(&page_bytes)
                    .map_err(|err| format!("failed decoding rendered PNG page {}: {err}", page_idx + 1))?
                    .to_rgba8();

                let width_px = rgba.width() as usize;
                let height_px = rgba.height() as usize;
                if width_px == 0 || height_px == 0 {
                    continue;
                }

                let note_positions = if page_idx < svg_pages.len() {
                    parse_svg_note_positions(&svg_pages[page_idx])
                } else {
                    Vec::new()
                };

                raw_pages.push(SheetRawPage {
                    width_px,
                    height_px,
                    rgba_data: rgba.into_raw(),
                    note_positions,
                });
            }

            if raw_pages.is_empty() {
                Err("verovioxide returned pages but no valid image could be decoded.".to_string())
            } else {
                Ok(raw_pages)
            }
        })();

        match render_result {
            Ok(pages) => SheetRenderResult { key, pages, error: None },
            Err(err) => SheetRenderResult { key, pages: Vec::new(), error: Some(err) },
        }
    }

    /// Extract note events from selected stem timelines, or from the combined
    /// visualization timeline if no stems are selected.
    fn extract_notes_for_stems(
        &self,
        stem_indices: &std::collections::BTreeSet<usize>,
        threshold: f32,
    ) -> Vec<NoteEvent> {
        let mut next_id: u32 = 1;
        if !stem_indices.is_empty() {
            // When specific stems are selected, merge their analyses
            let mut all_events: Vec<NoteEvent> = Vec::new();
            for &idx in stem_indices {
                if let Some(analysis) = self.stem_analyses.iter().find(|a| a.stem_index == idx) {
                    if !analysis.timeline.is_empty() && analysis.step_sec > 0.0 {
                        all_events.extend(Self::extract_events_from_timeline_data(
                            &analysis.timeline,
                            analysis.step_sec,
                            threshold,
                            &mut next_id,
                        ));
                    }
                }
            }
            if !all_events.is_empty() {
                all_events.sort_by(|a, b| {
                    a.start_time
                        .partial_cmp(&b.start_time)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.pitch.cmp(&b.pitch))
                });
                return all_events;
            }
            // Analysis not ready yet — return empty instead of using wrong data
            return Vec::new();
        }
        // "Full Mix" mode: use combined timeline if available, else all enabled stems
        if !self.stem_analyses.is_empty() && self.note_timeline.is_empty() {
            // Combine ALL enabled stem analyses into one polyphonic view
            let mut all_events: Vec<NoteEvent> = Vec::new();
            for analysis in &self.stem_analyses {
                if !self.enabled_stem_indices.contains(&analysis.stem_index) {
                    continue;
                }
                if analysis.timeline.is_empty() || analysis.step_sec <= 0.0 {
                    continue;
                }
                all_events.extend(Self::extract_events_from_timeline_data(
                    &analysis.timeline,
                    analysis.step_sec,
                    threshold,
                    &mut next_id,
                ));
            }
            all_events.sort_by(|a, b| {
                a.start_time
                    .partial_cmp(&b.start_time)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.pitch.cmp(&b.pitch))
            });
            return all_events;
        }
        self.extract_note_events_from_timeline(threshold)
    }

    /// Static helper to extract note events from any timeline data.
    fn extract_events_from_timeline_data(
        timeline: &[Vec<f32>],
        step_sec: f32,
        threshold: f32,
        next_id: &mut u32,
    ) -> Vec<NoteEvent> {
        if timeline.is_empty() || step_sec <= 0.0 {
            return Vec::new();
        }

        let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
        let mut out = Vec::new();
        let min_duration_sec = (step_sec * MIN_SHEET_NOTE_FRAMES as f32).max(0.05);

        for note_idx in 0..note_count {
            let mut active_start: Option<usize> = None;
            let mut max_prob: f32 = 0.0;

            for (frame_idx, frame) in timeline.iter().enumerate() {
                let prob = frame.get(note_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let active = prob >= threshold;

                if active {
                    max_prob = max_prob.max(prob);
                    if active_start.is_none() {
                        active_start = Some(frame_idx);
                    }
                } else if let Some(start_idx) = active_start.take() {
                    let start_time = start_idx as f32 * step_sec;
                    let mut end_time = frame_idx as f32 * step_sec;
                    if end_time <= start_time {
                        end_time = start_time + step_sec;
                    }
                    let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                    out.push(NoteEvent {
                        id: *next_id,
                        pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                        start_time,
                        end_time,
                        velocity,
                        channel: None,
                    });
                    *next_id = next_id.saturating_add(1);
                    max_prob = 0.0;
                }
            }

            if let Some(start_idx) = active_start {
                let start_time = start_idx as f32 * step_sec;
                let end_time = timeline.len() as f32 * step_sec;
                let end_time = end_time.max(start_time + step_sec);
                let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                out.push(NoteEvent {
                    id: *next_id,
                    pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                    start_time,
                    end_time,
                    velocity,
                    channel: None,
                });
                *next_id = next_id.saturating_add(1);
            }
        }

        merge_adjacent_notes(&mut out, step_sec);
        out.retain(|n| n.end_time - n.start_time >= min_duration_sec);
        out
    }

    fn extract_note_events_from_timeline(&self, threshold: f32) -> Vec<NoteEvent> {
        if self.note_timeline.is_empty() || self.note_timeline_step_sec <= 0.0 {
            return Vec::new();
        }

        let note_count = (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize;
        let mut out = Vec::new();
        let step = self.note_timeline_step_sec;
        let min_duration_sec = (step * MIN_SHEET_NOTE_FRAMES as f32).max(0.05);

        for note_idx in 0..note_count {
            let mut active_start: Option<usize> = None;
            let mut max_prob: f32 = 0.0;

            for (frame_idx, frame) in self.note_timeline.iter().enumerate() {
                let prob = frame.get(note_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let active = prob >= threshold;

                if active {
                    max_prob = max_prob.max(prob);
                    if active_start.is_none() {
                        active_start = Some(frame_idx);
                    }
                } else if let Some(start_idx) = active_start.take() {
                    let start_time = start_idx as f32 * step;
                    let mut end_time = frame_idx as f32 * step;
                    if end_time <= start_time {
                        end_time = start_time + step;
                    }

                    if end_time - start_time < min_duration_sec {
                        max_prob = 0.0;
                        continue;
                    }

                    let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                    out.push(NoteEvent {
                        id: (out.len() + 1) as u32,
                        pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                        start_time,
                        end_time,
                        velocity,
                        channel: None,
                    });

                    max_prob = 0.0;
                }
            }

            if let Some(start_idx) = active_start {
                let start_time = start_idx as f32 * step;
                let end_time = self.note_timeline.len() as f32 * step;
                let end_time = end_time.max(start_time + step);
                if end_time - start_time >= min_duration_sec {
                    let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                    out.push(NoteEvent {
                        id: (out.len() + 1) as u32,
                        pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                        start_time,
                        end_time,
                        velocity,
                        channel: None,
                    });
                }
            }
        }

        out.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.pitch.cmp(&b.pitch))
        });

        merge_adjacent_notes(&mut out, step);
        out
    }

    fn export_sheet_musicxml(&mut self, ctx: &egui::Context) {
        self.refresh_sheet_preview_if_needed(ctx);
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let xml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        if let Some(path) = self.pick_musicxml_export_path(file_stem.as_str()) {
            if fs::write(path.as_path(), xml.as_bytes()).is_ok() {
                self.last_error = None;
            } else {
                self.last_error = Some("Failed to write MusicXML export.".to_string());
            }
        }
    }

    fn export_sheet_pdf(&mut self, ctx: &egui::Context) {
        self.refresh_sheet_preview_if_needed(ctx);
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let xml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        let Some(pdf_path) = self.pick_pdf_export_path(file_stem.as_str()) else {
            return;
        };

        let sibling_xml_path = pdf_path.with_extension("musicxml");
        if fs::write(sibling_xml_path.as_path(), xml.as_bytes()).is_err() {
            self.last_error = Some("Failed to write intermediary MusicXML for PDF engraving.".to_string());
            return;
        }

        match export_engraved_pdf_with_musescore(sibling_xml_path.as_path(), pdf_path.as_path()) {
            Ok(()) => {
                self.last_error = None;
            }
            Err(err) => {
                self.last_error = Some(format!(
                    "Engraved PDF export failed: {err}. MusicXML was still written to {}",
                    sibling_xml_path.display()
                ));
            }
        }
    }

    fn open_in_musescore(&mut self, ctx: &egui::Context) {
        self.refresh_sheet_preview_if_needed(ctx);
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to open.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let xml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        let temp_path = match write_temp_musicxml("keyscribe", &xml) {
            Ok(p) => p,
            Err(e) => {
                self.last_error = Some(format!("Failed to write temp MusicXML: {e}"));
                return;
            }
        };

        let mut commands = vec![
            "musescore4".to_string(),
            "MuseScore4".to_string(),
            "mscore".to_string(),
            "MuseScore3".to_string(),
            "MuseScore".to_string(),
        ];

        if cfg!(windows) {
            if let Ok(program_files) = std::env::var("ProgramFiles") {
                commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files));
                commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files));
            }
            if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
                commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files_x86));
                commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files_x86));
            }
        }

        for cmd in &commands {
            match Command::new(cmd).arg(temp_path.as_os_str()).spawn() {
                Ok(_) => {
                    self.last_error = None;
                    return;
                }
                Err(_) => {}
            }
        }

        self.last_error = Some(
            "MuseScore was not found. Install MuseScore and ensure its CLI executable is on PATH.".to_string(),
        );
    }

    fn export_file_stem(&self) -> String {
        let raw = self
            .loaded_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("keyscribe-sheet");

        sanitize_filename_component(raw)
    }

    fn pick_musicxml_export_path(&self, file_stem: &str) -> Option<PathBuf> {
        #[cfg(feature = "desktop-ui")]
        {
            return FileDialog::new()
                .add_filter("MusicXML", &["musicxml", "xml"])
                .set_file_name(&format!("{file_stem}.musicxml"))
                .save_file();
        }

        #[cfg(not(feature = "desktop-ui"))]
        {
            Some(app_data_dir().join(format!("{file_stem}.musicxml")))
        }
    }

    fn pick_pdf_export_path(&self, file_stem: &str) -> Option<PathBuf> {
        #[cfg(feature = "desktop-ui")]
        {
            return FileDialog::new()
                .add_filter("PDF", &["pdf"])
                .set_file_name(&format!("{file_stem}.pdf"))
                .save_file();
        }

        #[cfg(not(feature = "desktop-ui"))]
        {
            Some(app_data_dir().join(format!("{file_stem}.pdf")))
        }
    }
}

fn total_beats_for_foundation(foundation: &LeadSheetFoundation) -> f32 {
    foundation
        .melody_notes
        .iter()
        .map(|n| n.beat_start + n.beat_duration)
        .fold(0.0f32, f32::max)
        .max(1.0)
}

fn parse_svg_note_positions(svg_str: &str) -> Vec<NotePosition> {
    let doc = match roxmltree::Document::parse(svg_str) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let svg_node = match doc.root().descendants().find(|n| n.has_tag_name("svg") && n.attribute("viewBox").is_some()) {
        Some(n) => n,
        None => return Vec::new(),
    };
    let (svg_w, svg_h) = match svg_node.attribute("viewBox") {
        Some(vb) => {
            let parts: Vec<f32> = vb.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 4 { (parts[2], parts[3]) } else { return Vec::new() }
        }
        None => return Vec::new(),
    };
    if svg_w <= 0.0 || svg_h <= 0.0 {
        return Vec::new();
    }

    // Parse page-margin offset from the outermost <g class="page-margin"> transform
    let mut margin_ox = 0.0f32;
    let mut margin_oy = 0.0f32;
    for node in doc.descendants() {
        if node.has_tag_name("g")
            && node.attribute("class").map_or(false, |c| c == "page-margin")
        {
            if let Some(t) = node.attribute("transform") {
                let t = t.trim();
                if let Some(inner) = t.strip_prefix("translate(") {
                    if let Some(paren) = inner.find(')') {
                        let coords: Vec<f32> = inner[..paren]
                            .split(|c| c == ',' || c == ' ')
                            .filter_map(|s| {
                                let s = s.trim();
                                if s.is_empty() { None } else { s.parse().ok() }
                            })
                            .collect();
                        if coords.len() >= 2 {
                            margin_ox = coords[0];
                            margin_oy = coords[1];
                        }
                    }
                }
            }
            break;
        }
    }

    let mut positions = Vec::new();
    for node in doc.descendants() {
        if !node.has_tag_name("g") {
            continue;
        }
        let id = match node.attribute("id") {
            Some(id) => {
                if let Some(pos) = id.find('n') {
                    // Safety check to ensure it's our note ID (starts with n followed by digit)
                    if id[pos + 1..]
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false)
                    {
                        &id[pos..]
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let parts: Vec<&str> = id[1..].split('_').collect();
        let (note_id, pitch, tick, duration_ticks) = if parts.len() >= 4 {
            let note_id = match parts[0].parse::<u32>().ok() {
                Some(id) => id,
                None => continue,
            };
            let pitch = match parts[1].parse::<u8>().ok() {
                Some(p) if p >= 21 && p <= 108 => p,
                _ => continue,
            };
            let tick = parts[2].parse::<i32>().ok().unwrap_or(0);
            let duration_ticks = parts[3].parse::<i32>().ok().unwrap_or(0);
            (note_id, pitch, tick, duration_ticks)
        } else if parts.len() >= 3 {
            let pitch = match parts[0].parse::<u8>().ok() {
                Some(p) if p >= 21 && p <= 108 => p,
                _ => continue,
            };
            let tick = parts[1].parse::<i32>().ok().unwrap_or(0);
            let duration_ticks = parts[2].parse::<i32>().ok().unwrap_or(0);
            (0, pitch, tick, duration_ticks)
        } else {
            continue;
        };

        // Try data-bounding-box first (most accurate)
        if let Some(bbox) = node.attribute("data-bounding-box") {
            let parts: Vec<f32> = bbox.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if parts.len() >= 4 && parts[2] > 0.0 && parts[3] > 0.0 {
                positions.push(NotePosition {
                    note_id,
                    x: (parts[0] + margin_ox) / svg_w,
                    y: (parts[1] + margin_oy) / svg_h,
                    w: parts[2] / svg_w,
                    h: parts[3] / svg_h,
                    pitch,
                    tick,
                    duration_ticks,
                });
                continue;
            }
        }

        // Try bounding-box <rect> child (Verovio's svg-bounding-box option)
        if let Some(rect) = node.children().find(|n| n.has_tag_name("rect")) {
            let rx = rect.attribute("x").and_then(|s| s.parse::<f32>().ok());
            let ry = rect.attribute("y").and_then(|s| s.parse::<f32>().ok());
            let rw = rect.attribute("width").and_then(|s| s.parse::<f32>().ok());
            let rh = rect.attribute("height").and_then(|s| s.parse::<f32>().ok());
            if let (Some(rx), Some(ry), Some(rw), Some(rh)) = (rx, ry, rw, rh) {
                if rw > 0.0 && rh > 0.0 {
                    positions.push(NotePosition {
                        note_id,
                        x: (rx + margin_ox) / svg_w,
                        y: (ry + margin_oy) / svg_h,
                        w: rw / svg_w,
                        h: rh / svg_h,
                        pitch,
                        tick,
                        duration_ticks,
                    });
                    continue;
                }
            }
        }

        // Fallback: find the first <use> element (note head) for x,y
        // Skip if this element already has a bbox child (avoids duplicating positions)
        let has_bbox_child = node.children().any(|n| {
            n.has_tag_name("g") && n.attribute("id").map_or(false, |id| id.contains("bbox-"))
        });
        if !has_bbox_child {
        if let Some(use_node) = node.descendants().find(|n| n.has_tag_name("use")) {
            // Verovio uses transform="translate(x,y)" instead of x/y attributes
            let nx = use_node.attribute("x").and_then(|s| s.parse::<f32>().ok());
            let ny = use_node.attribute("y").and_then(|s| s.parse::<f32>().ok());
            let coords = match (nx, ny) {
                (Some(x), Some(y)) => Some((x, y)),
                _ => {
                    use_node.attribute("transform").and_then(|t| {
                        let t = t.trim();
                        if t.starts_with("translate(") {
                            let inner = t.trim_start_matches("translate(");
                            if let Some(paren) = inner.find(')') {
                                let coords: Vec<f32> = inner[..paren]
                                    .split(|c| c == ',' || c == ' ')
                                    .filter_map(|s| {
                                        let s = s.trim();
                                        if s.is_empty() { None } else { s.parse().ok() }
                                    })
                                    .collect();
                                if coords.len() >= 2 {
                                    Some((coords[0], coords[1]))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                }
            };
            if let Some((ncx, ncy)) = coords {
                if ncx > 0.0 && ncy > 0.0 {
                    let note_w = 14.0 / svg_w;
                    let note_h = 14.0 / svg_h;
                    positions.push(NotePosition {
                        note_id,
                        x: (ncx - 7.0) / svg_w,
                        y: (ncy + margin_oy - 7.0) / svg_h,
                        w: note_w,
                        h: note_h,
                        pitch,
                        tick,
                        duration_ticks,
                    });
                }
            }
        }
        }
    }
    positions
}

fn draw_scrollable_engraved_preview(
    ui: &mut egui::Ui,
    pages: &[EngravedSheetPage],
    current_beat: f32,
    active_note_id: Option<u32>,
    accent: egui::Color32,
) {
    if pages.is_empty() {
        ui.label("No engraved pages available for preview.");
        return;
    }

    let cursor_color = egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200);

    let current_tick = (current_beat * MUSICXML_DIVISIONS as f32).round() as i32;

    // First, find the global active note or bounding notes for rest interpolation
    let mut global_active_count = 0usize;
    let mut active_pages = Vec::new();
    
    // Also track the best predecessor and successor across all pages if no active note
    let mut best_prev: Option<(usize, &NotePosition)> = None;
    let mut best_next: Option<(usize, &NotePosition)> = None;

    for (p_idx, page) in pages.iter().enumerate() {
        let mut has_active = false;
        for np in &page.note_positions {
            let is_active = if let Some(id) = active_note_id {
                np.note_id == id
            } else {
                current_tick >= np.tick && current_tick < np.tick + np.duration_ticks
            };

            if is_active {
                has_active = true;
                global_active_count += 1;
            } else {
                if np.tick + np.duration_ticks <= current_tick {
                    if best_prev.map_or(true, |(_, p)| np.tick + np.duration_ticks > p.tick + p.duration_ticks) {
                        best_prev = Some((p_idx, np));
                    }
                } else if np.tick > current_tick {
                    if best_next.map_or(true, |(_, n)| np.tick < n.tick) {
                        best_next = Some((p_idx, np));
                    }
                }
            }
        }
        if has_active {
            active_pages.push(p_idx);
        }
    }

    egui::ScrollArea::vertical()
        .id_source("sheet_music_scroll")
        .max_height(ui.available_height().max(120.0))
        .show(ui, |ui| {
            for (idx, page) in pages.iter().enumerate() {
                let page_width = page.width_px.max(1) as f32;
                let page_height = page.height_px.max(1) as f32;
                let target_width = ui.available_width().max(260.0);
                let scale = target_width / page_width;
                let target_height = (page_height * scale).max(120.0);

                let image = egui::Image::new(&page.texture)
                    .fit_to_exact_size(egui::vec2(target_width, target_height));
                let response = ui.add(image);

                let mut cursor_x: Option<f32> = None;
                let mut active_center_y: f32 = 0.0;
                let mut active_count = 0usize;

                if global_active_count > 0 {
                    if active_pages.contains(&idx) {
                        for np in &page.note_positions {
                            let is_active = if let Some(id) = active_note_id {
                                np.note_id == id
                            } else {
                                current_tick >= np.tick && current_tick < np.tick + np.duration_ticks
                            };
                            
                            if is_active {
                                let cx = response.rect.left() + (np.x + np.w * 0.5) * target_width;
                                cursor_x = Some(cursor_x.map_or(cx, |prev| prev.max(cx)));
                                active_center_y += np.y + np.h * 0.5;
                                active_count += 1;
                            }
                        }
                    }
                } else {
                    // No active notes globally, this is a rest.
                    // Interpolate between best_prev and best_next.
                    let draw_on_this_page = match (best_prev, best_next) {
                        (Some((p_idx, _)), Some((n_idx, _))) => idx == p_idx || (idx == n_idx && p_idx != n_idx),
                        (Some((p_idx, _)), None) => idx == p_idx,
                        (None, Some((n_idx, _))) => idx == n_idx,
                        (None, None) => false,
                    };

                    if draw_on_this_page {
                        if let (Some((p_idx, prev)), Some((n_idx, next))) = (best_prev, best_next) {
                            if p_idx == n_idx && idx == p_idx {
                                // Both on this page
                                let y_diff = (prev.y - next.y).abs();
                                if y_diff < 0.025 { // same system
                                    let prev_end = prev.tick + prev.duration_ticks;
                                    let gap = (next.tick - prev_end).max(1);
                                    let t = (current_tick - prev_end).max(0) as f32 / gap as f32;
                                    let t = t.clamp(0.0, 1.0);
                                    
                                    let prev_cx = response.rect.left() + (prev.x + prev.w * 0.5) * target_width;
                                    let next_cx = response.rect.left() + (next.x + next.w * 0.5) * target_width;
                                    
                                    cursor_x = Some(prev_cx + t * (next_cx - prev_cx));
                                    active_center_y += prev.y + prev.h * 0.5;
                                    active_count += 1;
                                } else {
                                    // Different systems on same page
                                    let prev_end = prev.tick + prev.duration_ticks;
                                    let gap = (next.tick - prev_end).max(1);
                                    let t = (current_tick - prev_end).max(0) as f32 / gap as f32;
                                    if t < 0.5 {
                                        cursor_x = Some(response.rect.left() + (prev.x + prev.w * 0.5) * target_width);
                                        active_center_y += prev.y + prev.h * 0.5;
                                        active_count += 1;
                                    } else {
                                        cursor_x = Some(response.rect.left() + (next.x + next.w * 0.5) * target_width);
                                        active_center_y += next.y + next.h * 0.5;
                                        active_count += 1;
                                    }
                                }
                            } else {
                                // On different pages. 
                                let prev_end = prev.tick + prev.duration_ticks;
                                let gap = (next.tick - prev_end).max(1);
                                let t = (current_tick - prev_end).max(0) as f32 / gap as f32;
                                if t < 0.5 && idx == p_idx {
                                    cursor_x = Some(response.rect.left() + (prev.x + prev.w * 0.5) * target_width);
                                    active_center_y += prev.y + prev.h * 0.5;
                                    active_count += 1;
                                } else if t >= 0.5 && idx == n_idx {
                                    cursor_x = Some(response.rect.left() + (next.x + next.w * 0.5) * target_width);
                                    active_center_y += next.y + next.h * 0.5;
                                    active_count += 1;
                                }
                            }
                        } else if let Some((p_idx, prev)) = best_prev {
                            if idx == p_idx {
                                cursor_x = Some(response.rect.left() + (prev.x + prev.w * 0.5) * target_width);
                                active_center_y += prev.y + prev.h * 0.5;
                                active_count += 1;
                            }
                        } else if let Some((n_idx, next)) = best_next {
                            if idx == n_idx {
                                cursor_x = Some(response.rect.left() + (next.x + next.w * 0.5) * target_width);
                                active_center_y += next.y + next.h * 0.5;
                                active_count += 1;
                            }
                        }
                    }
                }

                // Draw vertical cursor line centered on the staff system
                if let Some(x) = cursor_x {
                    // Find the staff system containing the active notes by clustering
                    // all note Y-centers on this page. A gap > 0.025 norm = system boundary.
                    let mut y_centers: Vec<f32> = page.note_positions.iter()
                        .map(|np| np.y + np.h * 0.5).collect();
                    y_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    if active_count == 0 {
                        continue;
                    }

                    let active_y = active_center_y / active_count as f32;
                    let gap_thresh = 0.025;
                    let mut system_top = y_centers.first().copied().unwrap_or(0.0);
                    let mut system_bot = system_top;
                    let mut i = 1;
                    while i < y_centers.len() {
                        if y_centers[i] - y_centers[i - 1] > gap_thresh {
                            // Check if active note falls in the just-finished system
                            if active_y >= system_top && active_y <= system_bot {
                                break;
                            }
                            system_top = y_centers[i];
                            system_bot = y_centers[i];
                        } else {
                            system_bot = y_centers[i];
                        }
                        i += 1;
                    }

                    let pad = (system_bot - system_top) * 0.1;
                    let sy0 = (system_top - pad).max(0.0);
                    let sy1 = (system_bot + pad).min(1.0);

                    let y0 = response.rect.top() + sy0 * target_height;
                    let y1 = response.rect.top() + sy1 * target_height;
                    ui.painter().line_segment(
                        [egui::pos2(x, y0), egui::pos2(x, y1)],
                        egui::Stroke::new(2.0, cursor_color),
                    );
                }

                if idx + 1 < pages.len() {
                    ui.add_space(UI_VSPACE_COMPACT);
                }
            }
        });
}

#[derive(Clone)]
struct NoteSpan {
    id: u32,
    start_tick: i32,
    end_tick: i32,
    pitch: u8,
    velocity: u8,
    staff: u8,
    articulation: Articulation,
}

#[derive(Clone)]
struct NoteChunk {
    id: u32,
    start_tick_in_measure: i32,
    duration_ticks: i32,
    absolute_tick: i32,
    pitch: u8,
    velocity: u8,
    tie_start: bool,
    tie_stop: bool,
    staff: u8,
    articulation: Articulation,
}

#[derive(Clone, Copy)]
struct DurationToken {
    ticks: i32,
    note_type: &'static str,
    dots: u8,
    time_mod: Option<(u8, u8)>,
}

#[derive(Clone, Copy)]
struct SheetEngravingConfig {
    allow_triplets: bool,
    is_lead_sheet: bool,
    single_staff: bool,
}

impl Default for SheetEngravingConfig {
    fn default() -> Self {
        Self {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: true,
            single_staff: false,
        }
    }
}

fn build_measure_boundaries(
    time_sigs: &[TimeSignatureSegment],
    default_beats_per_bar: u32,
    max_tick: i32,
) -> Vec<i32> {
    let mut boundaries: Vec<i32> = vec![0];

    if time_sigs.is_empty() {
        let mw = (default_beats_per_bar.max(1) as i32) * MUSICXML_DIVISIONS;
        let mut t = mw;
        while t < max_tick {
            boundaries.push(t);
            t += mw;
        }
        if boundaries.last().copied().unwrap_or(0) < max_tick {
            boundaries.push(max_tick);
        }
        return boundaries;
    }

    for (i, seg) in time_sigs.iter().enumerate() {
        let seg_start_tick = (seg.start_beat * MUSICXML_DIVISIONS as f32).round() as i32;
        let seg_end_tick = if i + 1 < time_sigs.len() {
            (time_sigs[i + 1].start_beat * MUSICXML_DIVISIONS as f32).round() as i32
        } else {
            (seg.end_beat * MUSICXML_DIVISIONS as f32).round() as i32
        };
        let mw = (seg.numerator.max(1) as i32) * MUSICXML_DIVISIONS;

        let seg_limit = seg_end_tick.min(max_tick);
        let mut cursor = boundaries.last().copied().unwrap_or(0).max(seg_start_tick);
        while cursor < seg_limit {
            let next = (cursor + mw).min(seg_limit);
            boundaries.push(next);
            cursor = next;
        }
    }

    let last = boundaries.last().copied().unwrap_or(0);
    if last < max_tick {
        boundaries.push(max_tick);
    }
    boundaries
}

fn split_span_into_measures(
    span: NoteSpan,
    boundaries: &[i32],
    target: &mut BTreeMap<i32, Vec<NoteChunk>>,
) {
    let mut cursor = span.start_tick;
    let mut first = true;

    while cursor < span.end_tick {
        let mi = match boundaries.binary_search(&cursor) {
            Ok(i) => i.min(boundaries.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(boundaries.len().saturating_sub(2)),
        };
        let ms = boundaries[mi];
        let me = boundaries[mi + 1];
        let chunk_end = span.end_tick.min(me);
        let dur = (chunk_end - cursor).max(1);

        target.entry(mi as i32).or_default().push(NoteChunk {
            id: span.id,
            start_tick_in_measure: cursor - ms,
            duration_ticks: dur,
            absolute_tick: cursor,
            pitch: span.pitch,
            velocity: span.velocity,
            tie_start: chunk_end < span.end_tick,
            tie_stop: !first,
            staff: span.staff,
            articulation: span.articulation,
        });

        first = false;
        cursor = chunk_end;
    }
}

fn build_musicxml_document(
    title: &str,
    foundation: &LeadSheetFoundation,
    config: SheetEngravingConfig,
) -> String {
    let note_spans = notes_to_spans(foundation.quantized_notes.as_slice(), config);

    let mut max_tick = 0i32;
    for span in &note_spans {
        max_tick = max_tick.max(span.end_tick);
    }

    let boundaries = build_measure_boundaries(
        &foundation.time_signature_segments,
        foundation.beats_per_bar,
        max_tick,
    );

    let mut chunks_by_measure: BTreeMap<i32, Vec<NoteChunk>> = BTreeMap::new();
    for span in note_spans {
        split_span_into_measures(span, &boundaries, &mut chunks_by_measure);
    }

    // Single average tempo for whole sheet
    let avg_bpm = if foundation.tempo_map.is_empty() {
        foundation.tempo.bpm
    } else {
        let total_weight: f32 = foundation
            .tempo_map
            .iter()
            .map(|s| (s.end_time_sec - s.start_time_sec).max(0.0))
            .sum();
        if total_weight > 0.0 {
            foundation
                .tempo_map
                .iter()
                .map(|s| s.bpm * (s.end_time_sec - s.start_time_sec).max(0.0))
                .sum::<f32>()
                / total_weight
        } else {
            foundation.tempo.bpm
        }
    };

    // Single tempo mark at the beginning
    let mut tempo_marks_by_measure: BTreeMap<i32, Vec<(i32, f32)>> = BTreeMap::new();
    tempo_marks_by_measure.entry(0).or_default().push((0, avg_bpm));

    let mut chord_by_measure: BTreeMap<i32, Vec<(i32, ChordSymbolChange)>> = BTreeMap::new();
    for chord in &foundation.chord_changes {
        let abs_tick = (chord.beat_start * MUSICXML_DIVISIONS as f32).round() as i32;
        let mi = match boundaries.binary_search(&abs_tick) {
            Ok(i) => i.min(boundaries.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(boundaries.len().saturating_sub(2)),
        };
        let offset = abs_tick - boundaries[mi];
        chord_by_measure
            .entry(mi as i32)
            .or_default()
            .push((offset, chord.clone()));
        max_tick = max_tick.max(abs_tick);
    }

    let total_measures = boundaries.len().saturating_sub(1).max(1);

    let mut time_signature_change_by_measure: BTreeMap<i32, (u8, u8)> = BTreeMap::new();
    for seg in &foundation.time_signature_segments {
        let abs_tick = (seg.start_beat * MUSICXML_DIVISIONS as f32).round() as i32;
        let mi = match boundaries.binary_search(&abs_tick) {
            Ok(i) => i.min(boundaries.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1).min(boundaries.len().saturating_sub(2)),
        };
        time_signature_change_by_measure
            .entry(mi.max(0) as i32)
            .or_insert((seg.numerator, seg.denominator));
    }

    if !time_signature_change_by_measure.contains_key(&0) {
        let default_num = foundation.beats_per_bar as u8;
        let default_ts = foundation
            .time_signature_segments
            .first()
            .map(|s| (s.numerator, s.denominator))
            .unwrap_or((default_num.max(1), 4));
        time_signature_change_by_measure.insert(0, default_ts);
    }

    let mut swing_by_measure: BTreeMap<i32, SwingStyle> = BTreeMap::new();
    for section in &foundation.swing_sections {
        if section.style != SwingStyle::Straight {
            let start_measure = section.bar_start as i32;
            let end_measure = (section.bar_end as i32).min(total_measures as i32);
            for m in start_measure..end_measure {
                swing_by_measure.entry(m).or_insert(section.style);
            }
        }
    }

    // Trim trailing empty measures
    let mut last_content = -1i32;
    for mi in 0..total_measures as i32 {
        let has_content = chunks_by_measure.contains_key(&mi)
            || chord_by_measure.contains_key(&mi)
            || tempo_marks_by_measure.contains_key(&mi)
            || swing_by_measure.contains_key(&mi);
        if has_content {
            last_content = mi;
        }
    }
    let total_measures = (last_content + 1).max(1).min(total_measures as i32) as usize;

    let mut xml = String::new();
    let _ = write!(
        xml,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<score-partwise version=\"3.1\">\n"
    );
    let _ = write!(
        xml,
        "  <work><work-title>{}</work-title></work>\n",
        xml_escape(title)
    );
    let _ = write!(xml, "  <part-list>\n");
    let _ = write!(
        xml,
        "    <score-part id=\"P1\"><part-name>Lead Sheet</part-name></score-part>\n"
    );
    let _ = write!(xml, "  </part-list>\n");
    let _ = write!(xml, "  <part id=\"P1\">\n");

    let mut current_time_sig = time_signature_change_by_measure
        .get(&0)
        .copied()
        .unwrap_or((4, 4));
    let mut prev_swing_style: Option<SwingStyle> = None;

    // Determine per-measure clef for single-staff modes based on average pitch
    let measure_clefs: Vec<&'static str> = if config.is_lead_sheet || config.single_staff {
        let mut clefs = Vec::with_capacity(total_measures);
        for measure_idx in 0..total_measures {
            if let Some(chunks) = chunks_by_measure.get(&(measure_idx as i32)) {
                let avg_pitch = chunks
                    .iter()
                    .map(|c| c.pitch as f32)
                    .sum::<f32>()
                    / chunks.len().max(1) as f32;
                clefs.push(if avg_pitch < 46.0 { "F" } else { "G" });
            } else {
                clefs.push(clefs.last().copied().unwrap_or("G"));
            }
        }
        clefs
    } else {
        Vec::new()
    };
    let mut current_clef: &'static str = "G";

    for measure_idx in 0..total_measures {
        let measure_ticks = boundaries[measure_idx + 1] - boundaries[measure_idx];
        let _ = write!(xml, "    <measure number=\"{}\">\n", measure_idx + 1);
        if let Some(&(num, den)) = time_signature_change_by_measure.get(&(measure_idx as i32)) {
            current_time_sig = (num, den);
        }

        let mut clef_to_write = None;
        if config.is_lead_sheet || config.single_staff {
            if let Some(&clef) = measure_clefs.get(measure_idx) {
                if clef != current_clef && measure_idx > 0 {
                    clef_to_write = Some(clef);
                }
            }
        }

        let has_time_sig_change = time_signature_change_by_measure.contains_key(&(measure_idx as i32));
        if measure_idx == 0 || has_time_sig_change || clef_to_write.is_some() {
            let _ = write!(xml, "      <attributes>\n");
            if measure_idx == 0 {
                let _ = write!(xml, "        <divisions>{}</divisions>\n", MUSICXML_DIVISIONS);
                let _ = write!(xml, "        <key><fifths>0</fifths></key>\n");
                if config.is_lead_sheet {
                    let _ = write!(xml, "        <clef><sign>G</sign><line>2</line></clef>\n");
                } else if config.single_staff {
                    let first_clef = measure_clefs.first().copied().unwrap_or("G");
                    let (sign, line) = if first_clef == "F" { ("F", 4) } else { ("G", 2) };
                    let _ = write!(xml, "        <clef><sign>{}</sign><line>{}</line></clef>\n", sign, line);
                } else {
                    let _ = write!(xml, "        <staves>2</staves>\n");
                    let _ = write!(xml, "        <clef number=\"1\"><sign>G</sign><line>2</line></clef>\n");
                    let _ = write!(xml, "        <clef number=\"2\"><sign>F</sign><line>4</line></clef>\n");
                }
            }
            if measure_idx == 0 || has_time_sig_change {
                let _ = write!(
                    xml,
                    "        <time><beats>{}</beats><beat-type>{}</beat-type></time>\n",
                    current_time_sig.0,
                    current_time_sig.1
                );
            }

            // Dynamic clef for single-staff modes: switch per-measure based on avg pitch
            if let Some(clef) = clef_to_write {
                current_clef = clef;
                let (sign, line) = if clef == "F" { ("F", 4) } else { ("G", 2) };
                let _ = write!(
                    xml,
                    "        <clef><sign>{}</sign><line>{}</line></clef>\n",
                    sign, line
                );
            }

            let _ = write!(xml, "      </attributes>\n");
        }

        // Swing direction element
        let current_swing = swing_by_measure.get(&(measure_idx as i32)).copied();
        if current_swing != prev_swing_style {
            if let Some(style) = current_swing {
                let _ = write!(xml, "      <direction placement=\"above\">\n");
                let _ = write!(xml, "        <direction-type>\n");
                match style {
                    SwingStyle::Swing => {
                        let _ = write!(xml, "          <words>Swing</words>\n");
                    }
                    SwingStyle::Triplet => {
                        let _ = write!(xml, "          <words>Triplet feel</words>\n");
                    }
                    _ => {}
                }
                let _ = write!(xml, "        </direction-type>\n");
                let _ = write!(xml, "        <sound type=\"dotted-quarter=quarter+dotted-quarter\"/>\n");
                let _ = write!(xml, "      </direction>\n");
            } else if prev_swing_style == Some(SwingStyle::Swing)
                || prev_swing_style == Some(SwingStyle::Triplet)
            {
                let _ = write!(xml, "      <direction placement=\"above\">\n");
                let _ = write!(xml, "        <direction-type>\n");
                let _ = write!(xml, "          <words>Straight</words>\n");
                let _ = write!(xml, "        </direction-type>\n");
                let _ = write!(xml, "      </direction>\n");
            }
            prev_swing_style = current_swing;
        }

        if let Some(tempo_marks) = tempo_marks_by_measure.get(&(measure_idx as i32)) {
            let mut sorted = tempo_marks.clone();
            sorted.sort_by_key(|(offset, _)| *offset);
            for (offset, bpm) in sorted {
                let _ = write!(xml, "      <direction placement=\"above\">\n");
                if offset > 0 {
                    let _ = write!(xml, "        <offset>{offset}</offset>\n");
                }
                let _ = write!(xml, "        <direction-type>\n");
                let _ = write!(xml, "          <metronome>\n");
                let _ = write!(xml, "            <beat-unit>quarter</beat-unit>\n");
                let _ = write!(xml, "            <per-minute>{:.2}</per-minute>\n", bpm);
                let _ = write!(xml, "          </metronome>\n");
                let _ = write!(xml, "        </direction-type>\n");
                let _ = write!(xml, "        <sound tempo=\"{:.2}\"/>\n", bpm);
                let _ = write!(xml, "      </direction>\n");
            }
        }

        if let Some(chords) = chord_by_measure.get(&(measure_idx as i32)) {
            let mut sorted = chords.clone();
            sorted.sort_by_key(|(offset, _)| *offset);
            for (offset, chord) in sorted {
                write_harmony(&mut xml, offset, &chord.symbol);
            }
        }

        let mut chunks = chunks_by_measure.remove(&(measure_idx as i32)).unwrap_or_default();
        chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);

        // If the measure has no notes, chords, or directions, fill it with a rest
        let has_content = !chunks.is_empty()
            || chord_by_measure.contains_key(&(measure_idx as i32))
            || tempo_marks_by_measure.contains_key(&(measure_idx as i32))
            || swing_by_measure.contains_key(&(measure_idx as i32));
        if !has_content {
            write_rest_ticks(
                &mut xml,
                boundaries[measure_idx],
                measure_ticks,
                MUSICXML_DIVISIONS,
                1,
                1,
                config,
            );
            let _ = write!(xml, "    </measure>\n");
            continue;
        }

        if config.is_lead_sheet {
            let voice_map = build_voice_chunks(chunks.as_slice(), 1);
            let voice_count = voice_map.len().max(1);
            let mut rendered = 0usize;
            for (voice, mut voice_chunks) in voice_map {
                voice_chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);
                write_voice_sequence(
                    &mut xml,
                    voice_chunks.as_slice(),
                    boundaries[measure_idx],
                    measure_ticks,
                    MUSICXML_DIVISIONS,
                    1,
                    voice,
                    config,
                );
                rendered += 1;
                if rendered < voice_count {
                    let _ = write!(xml, "      <backup>\n");
                    let _ = write!(xml, "        <duration>{}</duration>\n", measure_ticks.max(1));
                    let _ = write!(xml, "      </backup>\n");
                }
            }
        } else {
            for staff in [1u8, 2u8] {
                let voice_map = build_voice_chunks(chunks.as_slice(), staff);
                let voice_count = voice_map.len().max(1);
                let mut rendered = 0usize;

                for (voice, mut voice_chunks) in voice_map {
                    voice_chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);
                    write_voice_sequence(
                        &mut xml,
                        voice_chunks.as_slice(),
                        boundaries[measure_idx],
                        measure_ticks,
                        MUSICXML_DIVISIONS,
                        staff,
                        voice,
                        config,
                    );

                    rendered += 1;
                    if rendered < voice_count {
                        let _ = write!(xml, "      <backup>\n");
                        let _ = write!(xml, "        <duration>{}</duration>\n", measure_ticks.max(1));
                        let _ = write!(xml, "      </backup>\n");
                    }
                }

                if staff == 1 {
                    let _ = write!(xml, "      <backup>\n");
                    let _ = write!(xml, "        <duration>{}</duration>\n", measure_ticks.max(1));
                    let _ = write!(xml, "      </backup>\n");
                }
            }
        }

        let _ = write!(xml, "    </measure>\n");
    }

    let _ = write!(xml, "  </part>\n");
    let _ = write!(xml, "</score-partwise>\n");

    xml
}

/// Combined heuristic: skyline (highest pitch) + near-note continuity bias + outlier filter.
/// At each event point the highest active pitch is the base candidate, but when multiple
/// notes are active at the same pitch range, prefers the one closest to the previous melody
/// pitch (near-note continuity). Outliers more than `outlier_semitones` from the rolling
/// median are suppressed.
fn extract_melody_heuristic(notes: &[NoteEvent], outlier_semitones: u8) -> Vec<NoteEvent> {
    if notes.is_empty() {
        return Vec::new();
    }

    let mut events: Vec<(f32, u8, u8, bool)> = Vec::with_capacity(notes.len() * 2);
    for n in notes {
        if !n.start_time.is_finite() || !n.end_time.is_finite() || n.end_time <= n.start_time {
            continue;
        }
        events.push((n.start_time, n.pitch, n.velocity, true));
        events.push((n.end_time, n.pitch, n.velocity, false));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.3.cmp(&a.3)));

    // Pre-deduplicate: for any group of note-ons at the same time, keep only the
    // first note-on per pitch.  Multiple stems can produce identical note-ons,
    // but different pitches must be preserved so the skyline algorithm can
    // choose among them.
    let time_tolerance = 0.12;
    let mut deduped: Vec<(f32, u8, u8, bool)> = Vec::with_capacity(events.len());
    {
        let mut i = 0;
        while i < events.len() {
            let batch_time = events[i].0;
            let mut j = i;
            while j < events.len() && (events[j].0 - batch_time).abs() <= time_tolerance {
                j += 1;
            }
            let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
            for k in i..j {
                let (time, pitch, vel, is_start) = events[k];
                if is_start {
                    if seen.insert(pitch) {
                        deduped.push((time, pitch, vel, true));
                    }
                } else {
                    deduped.push((time, pitch, vel, false));
                }
            }
            i = j;
        }
    }

    let events = deduped;
    let mut active: Vec<(u8, u8)> = Vec::new();
    let mut melody_segments: Vec<NoteEvent> = Vec::new();
    let mut segment_start = 0.0f32;
    let mut last_melody_pitch: Option<u8> = None;
    let mut last_melody_vel: u8 = 90;
    let mut pitch_history: Vec<u8> = Vec::new();

    let mut i = 0;
    while i < events.len() {
        let batch_time = events[i].0;

        let mut batch_end = i;
        while batch_end < events.len() && (events[batch_end].0 - batch_time).abs() <= time_tolerance {
            batch_end += 1;
        }

        // Note-offs first
        for j in i..batch_end {
            let (_, pitch, _, is_start) = events[j];
            if !is_start {
                active.retain(|a| a.0 != pitch);
            }
        }

        // Note-ons second
        for j in i..batch_end {
            let (_, pitch, vel, is_start) = events[j];
            if is_start {
                active.push((pitch, vel));
            }
        }

        // One melody decision per batch
        if !active.is_empty() {
            let max_pitch = active.iter().map(|a| a.0).max().unwrap_or(0);
            let candidates: Vec<(u8, u8)> = active.iter().filter(|a| a.0 == max_pitch).copied().collect();
            let best = if candidates.len() == 1 {
                candidates[0]
            } else {
                let prev = last_melody_pitch.unwrap_or(max_pitch);
                candidates.into_iter().min_by_key(|c| (c.0 as i16 - prev as i16).abs()).unwrap()
            };

            if Some(best.0) != last_melody_pitch {
                let mut skip = false;
                if pitch_history.len() >= 3 {
                    let mut sorted = pitch_history.clone();
                    sorted.sort();
                    let median = sorted[sorted.len() / 2];
                    let diff = (best.0 as i16 - median as i16).abs();
                    if diff > outlier_semitones as i16 {
                        skip = true;
                    }
                }

                if !skip {
                    if let Some(prev_pitch) = last_melody_pitch {
                        if batch_time > segment_start {
                            melody_segments.push(NoteEvent {
                                id: (melody_segments.len() + 1) as u32,
                                pitch: prev_pitch,
                                start_time: segment_start,
                                end_time: batch_time,
                                velocity: last_melody_vel,
                                channel: None,
                            });
                        }
                    }

                    segment_start = batch_time;
                    last_melody_pitch = Some(best.0);
                    last_melody_vel = best.1;
                    pitch_history.push(best.0);
                    if pitch_history.len() > 8 {
                        pitch_history.remove(0);
                    }
                }
            }
        }

        i = batch_end;
    }

    if let Some(pitch) = last_melody_pitch {
        let end_t = events.last().map(|e| e.0).unwrap_or(segment_start + 1.0);
        if end_t > segment_start {
            melody_segments.push(NoteEvent {
                id: (melody_segments.len() + 1) as u32,
                pitch,
                start_time: segment_start,
                end_time: end_t,
                velocity: last_melody_vel,
                channel: None,
            });
        }
    }

    melody_segments
}

/// Pure skyline (no continuity bias) — highest pitch at each point, with outlier filter.
fn extract_melody_skyline(notes: &[NoteEvent], outlier_semitones: u8) -> Vec<NoteEvent> {
    if notes.is_empty() {
        return Vec::new();
    }

    // Build event list: note-on and note-off
    let mut events: Vec<(f32, u8, u8, bool)> = Vec::with_capacity(notes.len() * 2);
    for n in notes {
        if !n.start_time.is_finite() || !n.end_time.is_finite() || n.end_time <= n.start_time {
            continue;
        }
        events.push((n.start_time, n.pitch, n.velocity, true));
        events.push((n.end_time, n.pitch, n.velocity, false));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.3.cmp(&a.3)));

    // Sweep: batch-process events within a small time tolerance so that
    // near-simultaneous notes from different stems are treated as one group.
    let time_tolerance = 0.15; // covers ~0.42 beats at 170 BPM (safe for 0.5-beat snap bucket)
    let mut active: Vec<(u8, u8)> = Vec::new();
    let mut melody_segments: Vec<NoteEvent> = Vec::new();
    let mut segment_start = 0.0f32;
    let mut last_melody_pitch: Option<u8> = None;
    let mut last_melody_vel: u8 = 90;
    let mut pitch_history: Vec<u8> = Vec::new();

    let mut i = 0;
    while i < events.len() {
        let batch_time = events[i].0;

        // Gather all events within tolerance of batch_time
        let mut batch_end = i;
        while batch_end < events.len() && (events[batch_end].0 - batch_time).abs() <= time_tolerance {
            batch_end += 1;
        }

        // Process note-offs first for this batch
        for j in i..batch_end {
            let (_, pitch, _, is_start) = events[j];
            if !is_start {
                active.retain(|a| a.0 != pitch);
            }
        }

        // Process note-ons for this batch
        for j in i..batch_end {
            let (_, pitch, vel, is_start) = events[j];
            if is_start {
                active.push((pitch, vel));
            }
        }

        // Make one melody decision for this batch
        if !active.is_empty() {
            let max_pitch = active.iter().map(|a| a.0).max().unwrap_or(0);
            let candidates: Vec<(u8, u8)> = active.iter().filter(|a| a.0 == max_pitch).copied().collect();
            let best = if candidates.len() == 1 {
                candidates[0]
            } else {
                let prev = last_melody_pitch.unwrap_or(max_pitch);
                candidates.into_iter().min_by_key(|c| (c.0 as i16 - prev as i16).abs()).unwrap()
            };

            if Some(best.0) != last_melody_pitch {
                // Outlier filter
                let mut skip = false;
                if pitch_history.len() >= 3 {
                    let mut sorted = pitch_history.clone();
                    sorted.sort();
                    let median = sorted[sorted.len() / 2];
                    let diff = (best.0 as i16 - median as i16).abs();
                    if diff > outlier_semitones as i16 {
                        skip = true;
                    }
                }

                if !skip {
                    // Emit previous segment
                    if let Some(prev_pitch) = last_melody_pitch {
                        if batch_time > segment_start {
                            melody_segments.push(NoteEvent {
                                id: (melody_segments.len() + 1) as u32,
                                pitch: prev_pitch,
                                start_time: segment_start,
                                end_time: batch_time,
                                velocity: last_melody_vel,
                                channel: None,
                            });
                        }
                    }

                    segment_start = batch_time;
                    last_melody_pitch = Some(best.0);
                    last_melody_vel = best.1;
                    pitch_history.push(best.0);
                    if pitch_history.len() > 8 {
                        pitch_history.remove(0);
                    }
                }
            }
        }

        i = batch_end;
    }

    if let Some(pitch) = last_melody_pitch {
        let end_t = events.last().map(|e| e.0).unwrap_or(segment_start + 1.0);
        if end_t > segment_start {
            melody_segments.push(NoteEvent {
                id: (melody_segments.len() + 1) as u32,
                pitch,
                start_time: segment_start,
                end_time: end_t,
                velocity: last_melody_vel,
                channel: None,
            });
        }
    }

    melody_segments
}

fn merge_adjacent_notes(notes: &mut Vec<NoteEvent>, step: f32) {
    let merge_gap = (step * 2.0).max(0.03);
    
    notes.sort_by(|a, b| {
        a.pitch.cmp(&b.pitch).then_with(|| {
            a.start_time.partial_cmp(&b.start_time).unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut i = 0;
    while i + 1 < notes.len() {
        let a = &notes[i];
        let b = &notes[i + 1];
        if a.pitch == b.pitch && b.start_time - a.end_time < merge_gap {
            let end = a.end_time.max(b.end_time);
            let vel = a.velocity.max(b.velocity);
            notes[i].end_time = end;
            notes[i].velocity = vel;
            notes.remove(i + 1);
        } else {
            i += 1;
        }
    }

    notes.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });
}

fn notes_to_spans(notes: &[QuantizedNote], config: SheetEngravingConfig) -> Vec<NoteSpan> {
    // DEBUG: whole, half, quarter, 8th only (no 16th/32nd)
    let std_durations: [f32; 4] = [4.0, 2.0, 1.0, 0.5];

    fn snap_to_std(value: f32, candidates: &[f32]) -> f32 {
        let mut best = value;
        let mut best_err = f32::MAX;
        for &c in candidates {
            let err = (value - c).abs();
            if err < best_err {
                best_err = err;
                best = c;
            }
        }
        best
    }

    let mut out = Vec::new();
    for note in notes {
        let start_tick = (note.beat_start * MUSICXML_DIVISIONS as f32).round().max(0.0) as i32;
        let raw_dur = snap_to_std(note.beat_duration, &std_durations);
        let duration_ticks = (raw_dur * MUSICXML_DIVISIONS as f32).round().max(1.0) as i32;
        let staff = if config.is_lead_sheet || config.single_staff {
            1
        } else if note.pitch >= GRAND_STAFF_SPLIT_MIDI {
            1
        } else {
            2
        };
        out.push(NoteSpan {
            id: note.id,
            start_tick: start_tick.max(0),
            end_tick: (start_tick + duration_ticks).max(start_tick + 1),
            pitch: note.pitch,
            velocity: note.velocity,
            staff,
            articulation: note.articulation,
        });
    }

    out.sort_by_key(|n| n.start_tick);

    // Deduplicate: if two NoteSpans share the same pitch and start_tick,
    // keep only the longer one (prevents unison from monophonic reduction).
    let mut deduped: Vec<NoteSpan> = Vec::with_capacity(out.len());
    for span in out {
        if let Some(last) = deduped.last_mut() {
            if last.pitch == span.pitch && last.start_tick == span.start_tick {
                if span.end_tick > last.end_tick {
                    last.end_tick = span.end_tick;
                }
                continue;
            }
        }
        deduped.push(span);
    }
    out = deduped;

    out
}


fn write_harmony(xml: &mut String, offset: i32, symbol: &str) {
    let (root_pc, suffix, bass_pc) = parse_chord_symbol(symbol);
    let (root_step, root_alter) = pc_to_step_alter(root_pc);

    let _ = write!(xml, "      <harmony>\n");
    if offset > 0 {
        let _ = write!(xml, "        <offset>{offset}</offset>\n");
    }
    let _ = write!(xml, "        <root><root-step>{root_step}</root-step>");
    if root_alter != 0 {
        let _ = write!(xml, "<root-alter>{}</root-alter>", root_alter);
    }
    let _ = write!(xml, "</root>\n");

    let kind = chord_suffix_to_musicxml_kind(suffix);
    let _ = write!(
        xml,
        "        <kind{}>{}</kind>\n",
        if kind != "other" {
            ""
        } else {
            " text=\"other\""
        },
        kind
    );

    if let Some(bass_pc) = bass_pc {
        let (bass_step, bass_alter) = pc_to_step_alter(bass_pc);
        let _ = write!(xml, "        <bass><bass-step>{bass_step}</bass-step>");
        if bass_alter != 0 {
            let _ = write!(xml, "<bass-alter>{}</bass-alter>", bass_alter);
        }
        let _ = write!(xml, "</bass>\n");
    }

    let _ = write!(xml, "      </harmony>\n");
}

fn parse_chord_symbol(symbol: &str) -> (u8, &str, Option<u8>) {
    let (main, bass) = symbol.split_once('/').map(|(a, b)| (a, Some(b))).unwrap_or((symbol, None));

    let (root_pc, root_len) = parse_root_pc(main).unwrap_or((0, 1));
    let suffix = &main[root_len.min(main.len())..];
    let bass_pc = bass.and_then(|b| parse_root_pc(b).map(|(pc, _)| pc));

    (root_pc, suffix, bass_pc)
}

fn parse_root_pc(s: &str) -> Option<(u8, usize)> {
    let mut chars = s.chars();
    let first = chars.next()?;
    let base = match first {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };

    let second = s.chars().nth(1);
    match second {
        Some('#') => Some(((base + 1) % 12, 2)),
        Some('b') => Some(((base + 11) % 12, 2)),
        _ => Some((base, 1)),
    }
}

fn chord_suffix_to_musicxml_kind(suffix: &str) -> &'static str {
    match suffix {
        "" => "major",
        "-" => "minor",
        "7" => "dominant",
        "\u{0394}7" => "major-seventh",
        "-7" => "minor-seventh",
        "dim" => "diminished",
        "dim7" => "diminished-seventh",
        "aug" => "augmented",
        "sus2" => "suspended-second",
        "sus4" => "suspended-fourth",
        "-\u{0394}7" => "minor-major-seventh",
        "-7b5" => "half-diminished",
        "7#5" => "augmented-seventh",
        "9" => "dominant-ninth",
        "\u{0394}9" => "major-ninth",
        "-9" => "minor-ninth",
        "7b9" => "dominant-ninth",
        "7#9" => "dominant-ninth",
        "7#11" => "dominant-11th",
        "\u{0394}7#11" => "major-11th",
        "-11" => "minor-11th",
        "13" => "dominant-13th",
        "\u{0394}13" => "major-13th",
        "-13" => "minor-13th",
        "\u{0394}9#11" => "major-11th",
        _ => "other",
    }
}

fn pc_to_step_alter(pc: u8) -> (&'static str, i8) {
    match pc % 12 {
        0 => ("C", 0),
        1 => ("D", -1),
        2 => ("D", 0),
        3 => ("E", -1),
        4 => ("E", 0),
        5 => ("F", 0),
        6 => ("G", -1),
        7 => ("G", 0),
        8 => ("A", -1),
        9 => ("A", 0),
        10 => ("B", -1),
        _ => ("B", 0),
    }
}

fn write_rest_ticks(
    xml: &mut String,
    start_tick: i32,
    duration_ticks: i32,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let tokens = duration_tokens_for_ticks(duration_ticks, divisions, config);
    let mut cursor_tick = start_tick;
    for token in tokens {
        let _ = write!(
            xml,
            "      <note id=\"r{}_{}\">\n",
            cursor_tick,
            token.ticks.max(1)
        );
        let _ = write!(xml, "        <rest/>\n");
        let _ = write!(xml, "        <duration>{}</duration>\n", token.ticks.max(1));
        write_time_mod(xml, token.time_mod);
        let _ = write!(xml, "        <type>{}</type>\n", token.note_type);
        for _ in 0..token.dots {
            let _ = write!(xml, "        <dot/>\n");
        }
        let _ = write!(xml, "        <voice>{}</voice>\n", voice.max(1));
        let _ = write!(xml, "        <staff>{}</staff>\n", staff.max(1));
        if token.time_mod.is_some() {
            let _ = write!(xml, "        <notations>\n");
            write_tuplet_notation(xml, token.time_mod);
            let _ = write!(xml, "        </notations>\n");
        }
        let _ = write!(xml, "      </note>\n");
        cursor_tick += token.ticks.max(1);
    }
}



fn write_note_element(
    xml: &mut String,
    note_id: &str,
    pitch: u8,
    token: DurationToken,
    staff: u8,
    voice: u8,
    velocity: u8,
    is_chord: bool,
    tie_start: bool,
    tie_stop: bool,
    articulation: Articulation,
    beam: Option<&'static str>,
) {
    let (step, alter, octave) = midi_to_pitch_parts(pitch);
    let _ = write!(xml, "      <note id=\"{}\">\n", xml_escape(note_id));
    if is_chord {
        let _ = write!(xml, "        <chord/>\n");
    }
    if articulation == Articulation::Grace || token.note_type == "grace" {
        let _ = write!(xml, "        <grace/>\n");
    }
    if token.note_type != "grace" || articulation == Articulation::Grace {
        let _ = write!(xml, "        <pitch><step>{step}</step>");
        if alter != 0 {
            let _ = write!(xml, "<alter>{}</alter>", alter);
        }
        let _ = write!(xml, "<octave>{octave}</octave></pitch>\n");
    }
    if token.note_type != "grace" && articulation != Articulation::Grace {
        let _ = write!(xml, "        <duration>{}</duration>\n", token.ticks.max(1));
    }
    write_time_mod(xml, token.time_mod);
    let note_type = if articulation == Articulation::Grace { "eighth" } else { token.note_type };
    let _ = write!(xml, "        <type>{}</type>\n", note_type);
    for _ in 0..token.dots {
        let _ = write!(xml, "        <dot/>\n");
    }
    let (_, alter, _) = midi_to_pitch_parts(pitch);
    if alter != 0 && token.note_type != "grace" {
        let acc = match alter { -2 => "double-flat", -1 => "flat", 1 => "sharp", 2 => "double-sharp", _ => "sharp" };
        let _ = write!(xml, "        <accidental>{}</accidental>\n", acc);
    }
    if let Some(b) = beam {
        let _ = write!(xml, "        <beam number=\"1\">{}</beam>\n", b);
    }
    let _ = write!(xml, "        <voice>{}</voice>\n", voice.max(1));
    let _ = write!(xml, "        <staff>{}</staff>\n", staff.max(1));
    let _ = write!(xml, "        <velocity>{}</velocity>\n", velocity);

    if tie_stop {
        let _ = write!(xml, "        <tie type=\"stop\"/>\n");
    }
    if tie_start {
        let _ = write!(xml, "        <tie type=\"start\"/>\n");
    }

    let has_notations = tie_start || tie_stop || token.time_mod.is_some()
        || articulation == Articulation::Staccato
        || articulation == Articulation::Tenuto
        || articulation == Articulation::Accent;

    if has_notations {
        let _ = write!(xml, "        <notations>\n");
        if tie_stop {
            let _ = write!(xml, "          <tied type=\"stop\"/>\n");
        }
        if tie_start {
            let _ = write!(xml, "          <tied type=\"start\"/>\n");
        }
        match articulation {
            Articulation::Staccato => {
                let _ = write!(xml, "          <articulations>\n");
                let _ = write!(xml, "            <staccato/>\n");
                let _ = write!(xml, "          </articulations>\n");
            }
            Articulation::Tenuto => {
                let _ = write!(xml, "          <articulations>\n");
                let _ = write!(xml, "            <tenuto/>\n");
                let _ = write!(xml, "          </articulations>\n");
            }
            Articulation::Accent => {
                let _ = write!(xml, "          <articulations>\n");
                let _ = write!(xml, "            <accent/>\n");
                let _ = write!(xml, "          </articulations>\n");
            }
            _ => {}
        }
        write_tuplet_notation(xml, token.time_mod);
        let _ = write!(xml, "        </notations>\n");
    }

    let _ = write!(xml, "      </note>\n");
}

fn note_type_for_ticks(ticks: i32, divisions: i32) -> (&'static str, i32, i32) {
    let d = divisions.max(1);
    // DEBUG: whole, half, quarter, 8th only (no 16th/32nd)
    let types = [
        ("whole", d * 4, 0),
        ("half", d * 2, 0),
        ("quarter", d, 0),
        ("eighth", d / 2, 0),
    ];
    let mut best = ("quarter", d, 0i32);
    for &(name, expected, dots) in &types {
        if expected <= 0 {
            continue;
        }
        if (ticks - expected).abs() <= (ticks - best.1).abs() {
            best = (name, expected, dots);
        }
    }
    // Check dotted: 1.5x
    for &(name, base, _) in &types {
        if base <= 0 {
            continue;
        }
        let dotted = base + base / 2;
        if (ticks - dotted).abs() < (ticks - best.1).abs() {
            best = (name, dotted, 1);
        }
    }
    best
}

fn duration_tokens_for_ticks(
    duration_ticks: i32,
    divisions: i32,
    _config: SheetEngravingConfig,
) -> Vec<DurationToken> {
    let d = divisions.max(1);
    let min_tick = d / 2; // smallest unit = eighth note (240 at 480 div)

    // Floor to nearest 8th boundary so we never exceed the true duration
    let total = (duration_ticks.max(0) / min_tick) * min_tick;
    if total < min_tick {
        return Vec::new();
    }

    // DEBUG: undotted whole, half, quarter, eighth only
    let candidates = [
        DurationToken { ticks: d * 4, note_type: "whole", dots: 0, time_mod: None },
        DurationToken { ticks: d * 2, note_type: "half", dots: 0, time_mod: None },
        DurationToken { ticks: d, note_type: "quarter", dots: 0, time_mod: None },
        DurationToken { ticks: d / 2, note_type: "eighth", dots: 0, time_mod: None },
    ];

    let mut remaining = total;

    let mut out = Vec::new();
    while remaining > 0 {
        let mut chosen: Option<DurationToken> = None;
        for &candidate in &candidates {
            if candidate.ticks <= remaining {
                chosen = Some(candidate);
                break;
            }
        }

        match chosen {
            Some(token) => {
                remaining -= token.ticks;
                out.push(token);
            }
            None => {
                break;
            }
        }
    }

    out
}

fn build_voice_chunks(chunks: &[NoteChunk], staff: u8) -> BTreeMap<u8, Vec<NoteChunk>> {
    let mut out: BTreeMap<u8, Vec<NoteChunk>> = BTreeMap::new();
    for chunk in chunks {
        if chunk.staff != staff {
            continue;
        }
        out.entry(1).or_default().push(chunk.clone());
    }
    out
}

fn write_voice_sequence(
    xml: &mut String,
    chunks: &[NoteChunk],
    measure_start_tick: i32,
    measure_ticks: i32,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let min_rest_ticks = (divisions / 2).max(1); // no rests smaller than 8th

    // Pre-compute beam groups for consecutive eighth notes
    let mut beam_of_group: Vec<Option<&'static str>> = vec![None; chunks.len()];
    {
        let mut g = 0;
        let mut group_positions: Vec<(usize, bool)> = Vec::new();
        while g < chunks.len() {
            let pos = chunks[g].start_tick_in_measure;
            let mut ge = g;
            let mut max_dur = 0;
            while ge < chunks.len() && chunks[ge].start_tick_in_measure == pos {
                max_dur = max_dur.max(chunks[ge].duration_ticks);
                ge += 1;
            }
            let is_eighth = max_dur <= divisions / 2;
            group_positions.push((g, is_eighth));
            g = ge;
        }
        let mut gi = 0;
        while gi < group_positions.len() {
            if group_positions[gi].1 {
                let start = gi;
                while gi < group_positions.len() && group_positions[gi].1 { gi += 1; }
                let end = gi;
                if end - start >= 2 {
                    beam_of_group[group_positions[start].0] = Some("begin");
                    for k in (start + 1)..(end - 1) {
                        beam_of_group[group_positions[k].0] = Some("continue");
                    }
                    beam_of_group[group_positions[end - 1].0] = Some("end");
                }
            } else {
                gi += 1;
            }
        }
    }

    let mut cursor = 0i32;
    let mut i = 0;

    while i < chunks.len() {
        if cursor >= measure_ticks {
            break;
        }

        let nominal_pos = chunks[i].start_tick_in_measure;

        // If this chunk's position is behind the cursor, shift it to the cursor
        // (avo`ids going backwards in time within a single voice).
        let write_pos = if nominal_pos < cursor { cursor } else { nominal_pos };

        // Rest gap before write_pos
        if write_pos > cursor {
            let gap = write_pos.min(measure_ticks) - cursor;
            if gap >= min_rest_ticks {
                write_rest_ticks(
                    xml,
                    measure_start_tick + cursor,
                    gap,
                    divisions,
                    staff,
                    voice,
                    config,
                );
            }
            cursor = write_pos.min(measure_ticks);
            if cursor >= measure_ticks {
                break;
            }
        }

        // Gather all chunks starting at this *nominal* position (chord group)
        let mut group_end = i;
        while group_end < chunks.len() && chunks[group_end].start_tick_in_measure == nominal_pos {
            group_end += 1;
        }

        // Find max duration in the chord group
        let group_dur = chunks[i..group_end]
            .iter()
            .map(|c| c.duration_ticks)
            .max()
            .unwrap_or(0);
        let clamped_dur = group_dur.min(measure_ticks - cursor);

        // Write chord: first note normal, rest as chords.
        // ALL chord notes share the same duration (max of group) — MusicXML
        // requires chord notes to have identical durations, otherwise
        // renderers count each as consuming separate time.
        for (j, chunk) in chunks[i..group_end].iter().enumerate() {
            let is_chord = j > 0;
            if clamped_dur > 0 {
                let tokens = duration_tokens_for_ticks(clamped_dur, divisions, config);
                let mut current_token_tick = chunk.absolute_tick;
                
                for (idx, token) in tokens.iter().enumerate() {
                    let local_tie_stop = (idx > 0) || (idx == 0 && chunk.tie_stop);
                    let local_tie_start = (idx + 1 < tokens.len()) || (idx + 1 == tokens.len() && chunk.tie_start);
                    
                    let beam = if j == 0 && idx == 0 { beam_of_group[i] } else { None };
                    write_note_element(
                        xml,
                        &format!(
                            "n{}_{}_{}_{}",
                            chunk.id,
                            chunk.pitch,
                            current_token_tick,
                            token.ticks
                        ),
                        chunk.pitch,
                        *token,
                        staff,
                        voice,
                        chunk.velocity,
                        is_chord,
                        local_tie_start,
                        local_tie_stop,
                        chunk.articulation,
                        beam,
                    );
                    current_token_tick += token.ticks;
                }
            }
        }

        cursor = (cursor + clamped_dur).min(measure_ticks);
        i = group_end;
    }

    let remaining = measure_ticks - cursor;
    if remaining >= min_rest_ticks {
        write_rest_ticks(
            xml,
            measure_start_tick + cursor,
            remaining,
            divisions,
            staff,
            voice,
            config,
        );
    }
}


fn write_time_mod(xml: &mut String, time_mod: Option<(u8, u8)>) {
    if let Some((actual, normal)) = time_mod {
        let _ = write!(xml, "        <time-modification>\n");
        let _ = write!(xml, "          <actual-notes>{}</actual-notes>\n", actual);
        let _ = write!(xml, "          <normal-notes>{}</normal-notes>\n", normal);
        let _ = write!(xml, "        </time-modification>\n");
    }
}

fn write_tuplet_notation(xml: &mut String, time_mod: Option<(u8, u8)>) {
    if time_mod.is_some() {
        let _ = write!(xml, "          <tuplet type=\"start\"/>\n");
        let _ = write!(xml, "          <tuplet type=\"stop\"/>\n");
    }
}

fn midi_to_pitch_parts(midi: u8) -> (&'static str, i8, i32) {
    let octave = (midi as i32 / 12) - 1;
    match midi % 12 {
        0 => ("C", 0, octave),
        1 => ("C", 1, octave),
        2 => ("D", 0, octave),
        3 => ("D", 1, octave),
        4 => ("E", 0, octave),
        5 => ("F", 0, octave),
        6 => ("F", 1, octave),
        7 => ("G", 0, octave),
        8 => ("G", 1, octave),
        9 => ("A", 0, octave),
        10 => ("A", 1, octave),
        _ => ("B", 0, octave),
    }
}

fn midi_note_name(midi: u8) -> String {
    let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let octave = (midi as i32 / 12) - 1;
    format!("{}{}", names[(midi % 12) as usize], octave)
}

fn midi_note_name_opt(midi: u8) -> Option<String> {
    if midi >= 21 && midi <= 108 {
        Some(midi_note_name(midi))
    } else {
        None
    }
}

fn parse_note_input(input: &str) -> Option<u8> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }
    // Try as MIDI number first
    if let Ok(num) = input.parse::<u8>() {
        return Some(num.clamp(21, 108));
    }
    // Try as note name (e.g. C4, C#4, Db5, B-1)
    let names = [
        ("C", 0), ("C#", 1), ("Db", 1), ("D", 2), ("D#", 3), ("Eb", 3),
        ("E", 4), ("F", 5), ("F#", 6), ("Gb", 6), ("G", 7), ("G#", 8),
        ("Ab", 8), ("A", 9), ("A#", 10), ("Bb", 10), ("B", 11),
    ];
    let upper = input.to_uppercase();
    for (name, semitone) in &names {
        if let Some(rest) = upper.strip_prefix(name) {
            if let Ok(octave) = rest.trim().parse::<i32>() {
                let midi = (octave + 1) * 12 + semitone;
                if midi >= 21 && midi <= 108 {
                    return Some(midi as u8);
                }
            }
        }
    }
    None
}

fn sanitize_filename_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        let valid = c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.';
        if valid {
            out.push(c);
        } else if c.is_ascii_whitespace() {
            out.push('_');
        }
    }

    if out.is_empty() {
        "keyscribe-sheet".to_string()
    } else {
        out
    }
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn export_engraved_pdf_with_musescore(musicxml_path: &Path, pdf_path: &Path) -> Result<(), String> {
    let mut commands = vec![
        "musescore4".to_string(),
        "MuseScore4".to_string(),
        "mscore".to_string(),
        "MuseScore3".to_string(),
        "MuseScore".to_string(),
    ];

    if cfg!(windows) {
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files));
            commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files));
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files_x86));
            commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files_x86));
        }
    }

    let mut failures = Vec::<String>::new();
    for cmd in commands {
        let attempts = [
            vec![
                "-o".to_string(),
                pdf_path.to_string_lossy().to_string(),
                musicxml_path.to_string_lossy().to_string(),
            ],
            vec![
                musicxml_path.to_string_lossy().to_string(),
                "-o".to_string(),
                pdf_path.to_string_lossy().to_string(),
            ],
        ];

        for args in attempts {
            let status = Command::new(cmd.as_str()).args(args.as_slice()).status();
            match status {
                Ok(s) if s.success() => return Ok(()),
                Ok(s) => {
                    failures.push(format!("{} exited with {}", cmd, s));
                }
                Err(_) => {
                    // Try next command candidate.
                }
            }
        }
    }

    if failures.is_empty() {
        Err("MuseScore CLI was not found. Install MuseScore and ensure its CLI executable is on PATH.".to_string())
    } else {
        Err(format!(
            "MuseScore CLI failed to engrave PDF. Attempts: {}",
            failures.join(" | ")
        ))
    }
}

fn write_temp_musicxml(prefix: &str, xml: &str) -> Result<PathBuf, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("clock error: {e}"))?
        .as_millis();
    let path = std::env::temp_dir().join(format!("{}_{}.musicxml", prefix, now));
    fs::write(path.as_path(), xml.as_bytes())
        .map_err(|e| format!("failed to write temp musicxml: {e}"))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_readable_musicxml() {
        let foundation = LeadSheetFoundation {
            tempo_map: vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 8.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            }],
            time_signature_segments: vec![crate::leadsheet::TimeSignatureSegment {
                start_beat: 0.0,
                end_beat: 16.0,
                numerator: 4,
                denominator: 4,
                confidence: 0.9,
                meter_class: crate::leadsheet::MeterClass::SimpleQuadruple,
            }],
            tempo: crate::leadsheet::TempoEstimate {
                bpm: 120.0,
                beat_duration_sec: 0.5,
                confidence: 1.0,
            },
            quantized_notes: vec![
                QuantizedNote {
                    id: 1,
                    pitch: 60,
                    beat_start: 0.0,
                    beat_duration: 1.0,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
                QuantizedNote {
                    id: 2,
                    pitch: 64,
                    beat_start: 1.0,
                    beat_duration: 1.0,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
            ],
            melody_notes: vec![],
            chord_changes: vec![ChordSymbolChange {
                beat_start: 0.0,
                symbol: "C".to_string(),
            }],
            tied_notes: vec![],
            rhythm_confidence: 0.9,
            melodic_stem: None,
            separation_confidence: 0.0,
            aligned_notes: vec![],
            swing_sections: vec![],
            beats_per_bar: 4,
        };

        let xml = build_musicxml_document("Test", &foundation, SheetEngravingConfig {
            is_lead_sheet: false,
            ..SheetEngravingConfig::default()
        });
        assert!(xml.contains("<score-partwise"));
        assert!(xml.contains("<harmony>"));
        assert!(xml.contains("<measure number=\"1\">"));
        assert!(xml.contains("<staves>2</staves>"));
    }

    #[test]
    fn musicxml_contains_non_quarter_types_when_input_has_short_values() {
        let foundation = LeadSheetFoundation {
            tempo_map: vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 8.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            }],
            time_signature_segments: vec![crate::leadsheet::TimeSignatureSegment {
                start_beat: 0.0,
                end_beat: 16.0,
                numerator: 4,
                denominator: 4,
                confidence: 0.9,
                meter_class: crate::leadsheet::MeterClass::SimpleQuadruple,
            }],
            tempo: crate::leadsheet::TempoEstimate {
                bpm: 120.0,
                beat_duration_sec: 0.5,
                confidence: 1.0,
            },
            quantized_notes: vec![
                QuantizedNote {
                    id: 1,
                    pitch: 60,
                    beat_start: 0.0,
                    beat_duration: 0.5,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
                QuantizedNote {
                    id: 2,
                    pitch: 62,
                    beat_start: 0.5,
                    beat_duration: 0.5,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
            ],
            melody_notes: vec![],
            chord_changes: vec![ChordSymbolChange {
                beat_start: 0.0,
                symbol: "C".to_string(),
            }],
            tied_notes: vec![],
            rhythm_confidence: 0.9,
            melodic_stem: None,
            separation_confidence: 0.0,
            aligned_notes: vec![],
            swing_sections: vec![],
            beats_per_bar: 4,
        };

        let xml =
            build_musicxml_document("DurationTest", &foundation, SheetEngravingConfig::default());
        assert!(xml.contains("<type>eighth</type>"));
    }

    #[test]
    fn musicxml_emits_time_signature_changes() {
        let foundation = LeadSheetFoundation {
            tempo_map: vec![TempoSegment {
                start_time_sec: 0.0,
                end_time_sec: 12.0,
                bpm: 120.0,
                beat_duration_sec: 0.5,
                beat_offset: 0.0,
            }],
            time_signature_segments: vec![
                crate::leadsheet::TimeSignatureSegment {
                    start_beat: 0.0,
                    end_beat: 8.0,
                    numerator: 4,
                    denominator: 4,
                    confidence: 0.9,
                    meter_class: crate::leadsheet::MeterClass::SimpleQuadruple,
                },
                crate::leadsheet::TimeSignatureSegment {
                    start_beat: 8.0,
                    end_beat: 24.0,
                    numerator: 3,
                    denominator: 4,
                    confidence: 0.85,
                    meter_class: crate::leadsheet::MeterClass::SimpleTriple,
                },
            ],
            tempo: crate::leadsheet::TempoEstimate {
                bpm: 120.0,
                beat_duration_sec: 0.5,
                confidence: 1.0,
            },
            quantized_notes: vec![
                QuantizedNote {
                    id: 1,
                    pitch: 60,
                    beat_start: 0.0,
                    beat_duration: 1.0,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
                QuantizedNote {
                    id: 2,
                    pitch: 62,
                    beat_start: 8.0,
                    beat_duration: 1.0,
                    velocity: 96,
                    channel: None,
                    confidence: 1.0,
                    bar_index: 0,
                    beat_index: 0,
                    intra_beat_pos: 0.0,
                    articulation: Articulation::Normal,
                    swing_style: SwingStyle::Straight,
                    swing_feel: false,
                },
            ],
            melody_notes: vec![],
            chord_changes: vec![],
            tied_notes: vec![],
            rhythm_confidence: 0.85,
            melodic_stem: None,
            separation_confidence: 0.0,
            aligned_notes: vec![],
            swing_sections: vec![],
            beats_per_bar: 4,
        };

        let xml =
            build_musicxml_document("MeterChange", &foundation, SheetEngravingConfig::default());
        assert!(xml.contains("<beats>4</beats><beat-type>4</beat-type>"));
        assert!(xml.contains("<beats>3</beats><beat-type>4</beat-type>"));
    }
}
