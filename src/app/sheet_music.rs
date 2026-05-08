use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::leadsheet::{
    cross_validate_beat_sources, generate_lead_sheet_enhanced, generate_lead_sheet_foundation,
    generate_lead_sheet_with_tempo_map, tempo_map_from_beats, Articulation, BeatTrackConfig,
    ChordSymbolChange, CrossValidatedBeats, LeadSheetFoundation, LeadSheetPresetConfig, NoteEvent,
    QuantizedNote, SwingStyle, TimeSignatureSegment,
};
use verovioxide::{Options, Png, Svg, Toolkit};
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
    pub(super) fn draw_main_content_tabs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.main_content_tab, MainContentTab::Waveform, "Waveform");
            ui.selectable_value(
                &mut self.main_content_tab,
                MainContentTab::SheetMusic,
                "Sheet Music",
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(stems) = self.separated_stems.clone() {
                    // --- Visualization Selector ---
                    let visualize_btn_text = format!(
                        "Visualize: {} / {}",
                        self.enabled_stem_indices.len(),
                        stems.len()
                    );
                    
                    let visualize_resp = ui.add(egui::Button::new(visualize_btn_text).min_size(egui::vec2(220.0, 0.0)));
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
                                            if ui.button("All").clicked() {
                                                self.pending_stem_indices = (0..stems.len()).collect();
                                            }
                                            if ui.button("None").clicked() {
                                                self.pending_stem_indices.clear();
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
                    let listen_btn_text = format!(
                        "Listen: {}",
                        if self.enabled_listening_indices.is_empty() { 
                            "Original Mix".to_string() 
                        } else { 
                            format!("{}/{}", self.enabled_listening_indices.len(), stems.len()) 
                        },
                    );

                    let listen_resp = ui.add(egui::Button::new(listen_btn_text).min_size(egui::vec2(180.0, 0.0)));
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
        interaction_ready: bool,
        interaction_duration: f32,
        default_stack_spacing_y: f32,
        vertical_gap: f32,
    ) {
        ui.horizontal(|ui| {
            // Mode selector
            let mode_label = format!("Mode: {}", self.sheet_music_mode.label());
            egui::ComboBox::from_id_source("sheet_mode")
                .selected_text(&mode_label)
                .show_ui(ui, |ui| {
                    for mode in &[SheetMusicMode::LeadSheet, SheetMusicMode::PianoGrandStaff, SheetMusicMode::SingleStaff] {
                        let label = mode.label();
                        let selected = self.sheet_music_mode == *mode;
                        if ui.selectable_label(selected, label).clicked() {
                            self.sheet_music_mode = *mode;
                            self.sheet_preview_cache_key = None;
                            self.sheet_engraving_cache_key = None;
                            self.sheet_engraving_pages.clear();
                        }
                    }
                });

            ui.separator();

            // Melody source selector
            let mel_label = match self.melody_stem_index {
                Some(i) => format!("Melody: {}", self.stem_label(i)),
                None => "Melody: Full Mix".to_string(),
            };
            egui::ComboBox::from_id_source("melody_source")
                .selected_text(&mel_label)
                .show_ui(ui, |ui| {
                    let is_none = self.melody_stem_index.is_none();
                    if ui.selectable_label(is_none, "Full Mix").clicked() {
                        self.melody_stem_index = None;
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                    if let Some(stems) = self.separated_stems.as_ref() {
                        for (i, stem) in stems.iter().enumerate() {
                            let selected = self.melody_stem_index == Some(i);
                            if ui.selectable_label(selected, stem.stem_type.display_name().as_ref()).clicked() {
                                self.melody_stem_index = Some(i);
                                self.sheet_preview_cache_key = None;
                                self.sheet_engraving_cache_key = None;
                                self.sheet_engraving_pages.clear();
                            }
                        }
                    }
                });

            // Chord source selector
            let chord_label = if self.chord_skip {
                "Chords: off".to_string()
            } else {
                match self.chord_stem_index {
                    Some(i) => format!("Chords: {}", self.stem_label(i)),
                    None => "Chords: Full Mix".to_string(),
                }
            };
            egui::ComboBox::from_id_source("chord_source")
                .selected_text(&chord_label)
                .show_ui(ui, |ui| {
                    let is_off = self.chord_skip;
                    if ui.selectable_label(is_off, "No chords").clicked() {
                        self.chord_skip = true;
                        self.chord_stem_index = None;
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                    let is_none = !self.chord_skip && self.chord_stem_index.is_none();
                    if ui.selectable_label(is_none, "Full Mix").clicked() {
                        self.chord_skip = false;
                        self.chord_stem_index = None;
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                    if let Some(stems) = self.separated_stems.as_ref() {
                        for (i, stem) in stems.iter().enumerate() {
                            let selected = !self.chord_skip && self.chord_stem_index == Some(i);
                            if ui.selectable_label(selected, stem.stem_type.display_name().as_ref()).clicked() {
                                self.chord_skip = false;
                                self.chord_stem_index = Some(i);
                                self.sheet_preview_cache_key = None;
                                self.sheet_engraving_cache_key = None;
                                self.sheet_engraving_pages.clear();
                            }
                        }
                    }
                });
        });

        // BPM override (empty = auto-detect)
        ui.add_space(UI_VSPACE_TIGHT);
        ui.horizontal(|ui| {
            ui.label("BPM:");
            let bpm_resp = ui.add(
                egui::TextEdit::singleline(&mut self.bpm_input_str)
                    .desired_width(50.0)
                    .hint_text("Auto"),
            );
            if bpm_resp.lost_focus() {
                if self.bpm_input_str.trim().is_empty() {
                    self.manual_bpm = None;
                } else if let Ok(bpm) = self.bpm_input_str.trim().parse::<f32>() {
                    self.manual_bpm = Some(bpm.clamp(20.0, 300.0));
                }
            }
            if self.manual_bpm.is_some() {
                if ui.button("×2").clicked() {
                    let base = self.manual_bpm.unwrap_or(120.0);
                    self.manual_bpm = Some((base * 2.0).clamp(20.0, 300.0));
                }
                if ui.button("÷2").clicked() {
                    let base = self.manual_bpm.unwrap_or(120.0);
                    self.manual_bpm = Some((base / 2.0).clamp(20.0, 300.0));
                }
                if ui.button("Clear").clicked() {
                    self.manual_bpm = None;
                }
            } else {
                ui.label(egui::RichText::new("Auto").color(egui::Color32::GRAY).weak());
            }
        });

        ui.add_space(UI_VSPACE_TIGHT);
        ui.horizontal(|ui| {
            ui.label("Feel:");
            let current_swing = self.manual_swing;
            egui::ComboBox::from_id_source("swing_combo")
                .selected_text(match current_swing {
                    None => "Auto",
                    Some(crate::leadsheet::types::SwingStyle::Straight) => "Straight (Even)",
                    Some(crate::leadsheet::types::SwingStyle::Swing) => "Swung 8ths",
                    Some(crate::leadsheet::types::SwingStyle::Triplet) => "Swung 16ths / Triplet",
                })
                .show_ui(ui, |ui| {
                    if ui.selectable_label(current_swing == None, "Auto").clicked() {
                        self.manual_swing = None;
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                    if ui.selectable_label(current_swing == Some(crate::leadsheet::types::SwingStyle::Straight), "Straight (Even)").clicked() {
                        self.manual_swing = Some(crate::leadsheet::types::SwingStyle::Straight);
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                    if ui.selectable_label(current_swing == Some(crate::leadsheet::types::SwingStyle::Swing), "Swung 8ths").clicked() {
                        self.manual_swing = Some(crate::leadsheet::types::SwingStyle::Swing);
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                    if ui.selectable_label(current_swing == Some(crate::leadsheet::types::SwingStyle::Triplet), "Swung 16ths / Triplet").clicked() {
                        self.manual_swing = Some(crate::leadsheet::types::SwingStyle::Triplet);
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_pages.clear();
                    }
                });
        });

        // Note range filter for melody
        ui.add_space(UI_VSPACE_TIGHT);
        ui.horizontal(|ui| {
            ui.label("Note range:");
            let min_resp = ui.add(
                egui::TextEdit::singleline(&mut self.melody_min_input_str)
                    .desired_width(30.0)
                    .hint_text("min"),
            );
            if min_resp.lost_focus() {
                self.melody_min_note = parse_note_input(&self.melody_min_input_str);
                self.sheet_preview_cache_key = None;
                self.sheet_engraving_pages.clear();
            }
            ui.label("-");
            let max_resp = ui.add(
                egui::TextEdit::singleline(&mut self.melody_max_input_str)
                    .desired_width(30.0)
                    .hint_text("max"),
            );
            if max_resp.lost_focus() {
                self.melody_max_note = parse_note_input(&self.melody_max_input_str);
                self.sheet_preview_cache_key = None;
                self.sheet_engraving_pages.clear();
            }
            // Show parsed MIDI numbers as hint
            let min_hint = self.melody_min_note.and_then(midi_note_name_opt);
            let max_hint = self.melody_max_note.and_then(midi_note_name_opt);
            match (min_hint, max_hint) {
                (Some(a), Some(b)) => { ui.label(egui::RichText::new(format!("{}–{}", a, b)).weak()); }
                (Some(a), None) => { ui.label(egui::RichText::new(format!("{}–", a)).weak()); }
                (None, Some(b)) => { ui.label(egui::RichText::new(format!("–{}", b)).weak()); }
                _ => {}
            }
            if self.melody_min_note.is_some() || self.melody_max_note.is_some() {
                if ui.button("Clear range").clicked() {
                    self.melody_min_note = None;
                    self.melody_max_note = None;
                    self.sheet_preview_cache_key = None;
                    self.sheet_engraving_pages.clear();
                }
            }
        });

        // Generate button
        // Validate note range inputs
        let mut range_error: Option<String> = None;
        if !self.melody_min_input_str.trim().is_empty() && self.melody_min_note.is_none() {
            range_error = Some(format!("Invalid min note: '{}'", self.melody_min_input_str.trim()));
        } else if !self.melody_max_input_str.trim().is_empty() && self.melody_max_note.is_none() {
            range_error = Some(format!("Invalid max note: '{}'", self.melody_max_input_str.trim()));
        } else if let (Some(min), Some(max)) = (self.melody_min_note, self.melody_max_note) {
            if min > max {
                range_error = Some("Min note is above max note".to_string());
            }
        }

        if let Some(ref err) = range_error {
            ui.colored_label(egui::Color32::from_rgb(224, 112, 112), err);
        }

        ui.add_space(UI_VSPACE_TIGHT);
        ui.horizontal(|ui| {
            if ui.button("Generate Sheet Music").clicked() {
                self.sheet_preview_cache_key = None;
                self.sheet_engraving_cache_key = None;
                self.refresh_sheet_preview_if_needed();
                if let Some(preview) = self.sheet_preview_cache.as_ref().cloned() {
                    let file_stem = self.export_file_stem();
                    let engraving_config = SheetEngravingConfig {
                        allow_triplets: !SHEET_SWING_BIAS,
                        is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
                    };
                    let musicxml = build_musicxml_document(
                        file_stem.as_str(),
                        &preview.foundation,
                        engraving_config,
                    );
                    if let Some(key) = self.current_sheet_preview_key() {
                        self.start_sheet_render(ui.ctx(), &musicxml, key);
                    }
                }
            }
            if self.is_processing || !self.note_visuals_ready() {
                ui.label(
                    egui::RichText::new("(waiting for transcription...)")
                        .color(egui::Color32::GRAY),
                );
            }
        });

        let preview = self.sheet_preview_cache.clone();
        let preview_error = self.sheet_preview_error.clone();
        let engraving_error = self.sheet_engraving_error.clone();

        ui.horizontal_wrapped(|ui| {
            if let Some(data) = preview.as_ref() {
                let used_bpm = data.foundation.tempo.bpm;
                let source = data.bpm_source.as_str();
                let manual_label = if self.manual_bpm.is_some() { " (manual)" } else { "" };
                ui.label(format!(
                    "BPM: {:.1}{} | from {} | Segments: {} | Chords: {} | Notes: {} | Gate: {:.2}",
                    used_bpm,
                    manual_label,
                    source,
                    data.foundation.tempo_map.len(),
                    data.foundation.chord_changes.len(),
                    data.note_count,
                    data.threshold
                ));
            } else {
                ui.label("Sheet preview unavailable for the current transcription.");
            }

            ui.separator();

            let can_export = preview.is_some();
            if ui
                .add_enabled(can_export, egui::Button::new("Export MusicXML"))
                .clicked()
            {
                self.export_sheet_musicxml();
            }
            if ui
                .add_enabled(can_export, egui::Button::new("Export Engraved PDF"))
                .clicked()
            {
                self.export_sheet_pdf();
            }
        });

        if let Some(err) = preview_error.as_deref() {
            ui.colored_label(ERROR_RED, err);
        }
        if let Some(err) = engraving_error.as_deref() {
            ui.colored_label(ERROR_RED, err);
        }

        let remaining_stack_h = ui.available_height().max(0.0);
        let media_height = media_controls_height_for_width(ui.available_width())
            .min((remaining_stack_h - vertical_gap * 2.0).max(0.0));
        let content_height = (remaining_stack_h - (media_height + vertical_gap * 2.0)).max(0.0);

        ui.group(|ui| {
            ui.set_min_height(content_height);
            if let Some(data) = preview.as_ref() {
                let current_beat = data.foundation.beat_at_time(self.selected_time_sec.max(0.0));
                if self.sheet_engraving_pages.is_empty() {
                    ui.add_space(16.0);
                    ui.centered_and_justified(|ui| {
                        ui.label("Engraving is being prepared. Try reprocessing if this persists.");
                    });
                } else {
                    let sensitivity = self.key_color_sensitivity.clamp(0.0, 2.0);
                    let threshold = if sensitivity > 0.0 {
                        (NOTE_HIGHLIGHT_ACTIVATION_THRESHOLD / sensitivity).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    let active_notes: Vec<u8> = self
                        .note_probs_smoothed
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| **p >= threshold)
                        .map(|(i, _)| (PIANO_LOW_MIDI as usize + i) as u8)
                        .collect();
                    draw_scrollable_engraved_preview(
                        ui,
                        self.sheet_engraving_pages.as_slice(),
                        current_beat,
                        self.highlight_color,
                        &active_notes,
                    );
                }

                ui.add_space(UI_VSPACE_TIGHT);
                draw_horizontal_separator(ui, 0.0);
                ui.add_space(UI_VSPACE_TIGHT);

                egui::ScrollArea::vertical()
                    .max_height((content_height * 0.30).max(80.0))
                    .show(ui, |ui| {
                        egui::Grid::new("sheet_note_table")
                            .striped(true)
                            .num_columns(4)
                            .show(ui, |ui| {
                                ui.label("Beat");
                                ui.label("Duration");
                                ui.label("Pitch");
                                ui.label("Velocity");
                                ui.end_row();

                                for note in data
                                    .foundation
                                    .quantized_notes
                                    .iter()
                                    .take(SHEET_NOTE_TABLE_LIMIT)
                                {
                                    ui.label(format!("{:.3}", note.beat_start));
                                    ui.label(format!("{:.3}", note.beat_duration));
                                    ui.label(midi_note_name(note.pitch));
                                    ui.label(note.velocity.to_string());
                                    ui.end_row();
                                }
                            });
                    });
            } else {
                ui.add_space(24.0);
                ui.centered_and_justified(|ui| {
                    ui.label(
                        "Need stable transcription data before sheet engraving is possible.\nTry reprocessing or adjusting key sensitivity.",
                    );
                });
            }
        });

        ui.add_space(vertical_gap);
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.y = default_stack_spacing_y;
            draw_media_controls(self, ui, interaction_ready, interaction_duration);
        });
        ui.add_space(vertical_gap);
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

        Some(SheetPreviewCacheKey {
            timeline_ptr: Arc::as_ptr(&self.note_timeline) as usize,
            timeline_len: self.note_timeline.len(),
            timeline_step_bits: self.note_timeline_step_sec.to_bits(),
            threshold_bits: self.sheet_preview_threshold().to_bits(),
            separation_selection_bits: separation_bits,
            mode_bits: self.sheet_music_mode as u8,
            melody_stem_bit: self.melody_stem_index.map(|i| i as u8),
            chord_stem_bit: self.chord_stem_index.map(|i| i as u8),
            swing_style_bit: self.manual_swing.map(|s| s as u8),
            stem_analysis_key: stem_key,
            melody_note_range_bits: (note_min_bits << 8) | note_max_bits,
        })
    }

    fn refresh_sheet_preview_if_needed(&mut self) {
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

        match self.build_sheet_preview() {
            Ok(preview) => {
                self.sheet_preview_cache_key = Some(key);
                self.sheet_preview_cache = Some(preview);
                self.sheet_preview_error = None;
                self.sheet_engraving_cache_key = None;
                self.sheet_engraving_pages.clear();
                self.sheet_engraving_error = None;
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
        };
        let musicxml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );
        let total_beats = total_beats_for_foundation(&preview.foundation);

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

    fn build_sheet_preview(&mut self) -> Result<SheetPreviewData, String> {
        let threshold = self.sheet_preview_threshold();

        // Extract melody notes from the selected melody source (or combined timeline)
        let melody_events = self.extract_notes_for_stem(self.melody_stem_index, threshold);
        if melody_events.is_empty() {
            if self.melody_stem_index.is_some() {
                return Err("Selected melody stem analysis not yet ready. Please wait for per-stem analysis to complete.".to_string());
            }
            return Err("Not enough note events to infer tempo and sheet layout.".to_string());
        }
        if melody_events.len() < 4 {
            return Err("Not enough note events to infer tempo and sheet layout.".to_string());
        }

        // Apply note range filter
        let melody_events: Vec<NoteEvent> = melody_events
            .into_iter()
            .filter(|n| {
                let above_min = self.melody_min_note.map(|m| n.pitch >= m).unwrap_or(true);
                let below_max = self.melody_max_note.map(|m| n.pitch <= m).unwrap_or(true);
                above_min && below_max
            })
            .collect();
        if melody_events.len() < 4 {
            return Err("Not enough notes in the selected range. Widen the note range.".to_string());
        }

        // For lead sheet: reduce melody to monophonic
        let note_events = if self.sheet_music_mode.is_lead_sheet() {
            reduce_to_monophonic_melody(&melody_events)
        } else {
            melody_events
        };

        if note_events.len() < 4 {
            return Err("Not enough melody notes after reduction.".to_string());
        }

        // Cross-validated beat/downbeat detection via beat-this on drums + bass
        // Falls back: beat-this on full mix → autocorrelation on stems → note-onset
        let stems = self.separated_stems.as_ref();
        let sample_rate = stems
            .and_then(|s| s.first().map(|st| st.sample_rate))
            .or_else(|| self.audio_raw.as_ref().map(|r| r.sample_rate));

        let (beat_track, bpm_source) = if let Some(sr) = sample_rate {
            let bass_audio: Option<&[f32]> = stems.and_then(|stems| {
                stems
                    .iter()
                    .find(|s| s.stem_type == StemType::Bass)
                    .map(|s| s.samples_mono.as_ref().as_slice())
            });
            let drum_audio: Option<&[f32]> = stems.and_then(|stems| {
                stems
                    .iter()
                    .find(|s| s.stem_type == StemType::Drums)
                    .map(|s| s.samples_mono.as_ref().as_slice())
            });
            let full_mix: Option<&[f32]> =
                self.audio_raw.as_ref().map(|r| r.samples_mono.as_slice());
            let beat_config = BeatTrackConfig::default();
            match cross_validate_beat_sources(
                bass_audio,
                drum_audio,
                full_mix,
                sr,
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
                    // Fallback to autocorrelation on stems if ML model unavailable
                    eprintln!("BeatThis Python execution failed: {:?}", e);
                    let duration = self.source_duration();
                    match crate::leadsheet::detect_beats_from_stems(
                        bass_audio,
                        drum_audio,
                        full_mix,
                        sr,
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

        // Fallback: use note-onset BPM if audio-based detection failed
        let beat_track = beat_track.or_else(|| {
            crate::leadsheet::detect_beats_from_notes(&note_events).map(CrossValidatedBeats::from)
        });

        // Apply manual BPM override if set
        let beat_track = if let Some(manual_bpm) = self.manual_bpm {
            let beat_duration = 60.0 / manual_bpm.clamp(20.0, 300.0);
            let duration = self.source_duration();
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
        config.quantization.min_duration_beats = 0.25;
        config.quantization.grids = vec![1.0, 0.5, 0.25];
        config.quantization.duration_grids = vec![4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25];
        config.chord_analysis.skip = self.chord_skip;
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
        let foundation = foundation
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
            })?;

        if foundation.quantized_notes.is_empty() {
            return Err("No quantized notes available for engraving.".to_string());
        }

        let bpm_source = if self.manual_bpm.is_some() {
            "Manual".to_string()
        } else {
            bpm_source
        };

        Ok(SheetPreviewData {
            foundation,
            note_count: note_events.len(),
            threshold,
            bpm_source,
        })
    }

    /// Extract note events from a specific stem's timeline, or from the combined
    /// visualization timeline if no stem is selected.
    fn extract_notes_for_stem(
        &self,
        stem_index: Option<usize>,
        threshold: f32,
    ) -> Vec<NoteEvent> {
        if let Some(idx) = stem_index {
            // When a specific stem is selected, ONLY use that stem's analysis.
            // Never fall back to the combined timeline — that would silently
            // include notes from other stems and ignore the user's choice.
            if let Some(analysis) = self.stem_analyses.iter().find(|a| a.stem_index == idx) {
                if !analysis.timeline.is_empty() && analysis.step_sec > 0.0 {
                    return Self::extract_events_from_timeline_data(
                        &analysis.timeline,
                        analysis.step_sec,
                        threshold,
                    );
                }
            }
            // Analysis not ready yet — return empty instead of using wrong data
            return Vec::new();
        }
        // "Full Mix" mode: use combined timeline if available, else all non-drums stems
        if !self.stem_analyses.is_empty() && self.note_timeline.is_empty() {
            // Combine ALL non-drums stem analyses into one polyphonic view
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
                let start_time = start_idx as f32 * step_sec;
                let end_time = timeline.len() as f32 * step_sec;
                let end_time = end_time.max(start_time + step_sec);
                let velocity = (max_prob * 127.0).round().clamp(1.0, 127.0) as u8;
                out.push(NoteEvent {
                    pitch: (PIANO_LOW_MIDI as usize + note_idx) as u8,
                    start_time,
                    end_time,
                    velocity,
                    channel: None,
                });
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

    fn trigger_sheet_regenerate(&mut self, ctx: &egui::Context) {
        self.sheet_preview_cache_key = None;
        self.sheet_engraving_cache_key = None;
        self.refresh_sheet_preview_if_needed();
        if let Some(preview) = self.sheet_preview_cache.as_ref().cloned() {
            self.refresh_engraved_preview_if_needed(ctx, &preview);
        }
    }

    fn export_sheet_musicxml(&mut self) {
        self.refresh_sheet_preview_if_needed();
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
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

    fn export_sheet_pdf(&mut self) {
        self.refresh_sheet_preview_if_needed();
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
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
    let svg_node = match doc.root().descendants().find(|n| n.has_tag_name("svg")) {
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

    let mut positions = Vec::new();
    for node in doc.descendants() {
        if !node.has_tag_name("g") {
            continue;
        }
        let id = match node.attribute("id") {
            Some(id) if id.starts_with('n') => id,
            _ => continue,
        };
        let pitch = match id[1..].split('_').next().and_then(|s| s.parse::<u8>().ok()) {
            Some(p) if p >= 21 && p <= 108 => p,
            _ => continue,
        };

        // Try data-bounding-box first (most accurate)
        if let Some(bbox) = node.attribute("data-bounding-box") {
            let parts: Vec<f32> = bbox.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if parts.len() >= 4 && parts[2] > 0.0 && parts[3] > 0.0 {
                positions.push(NotePosition {
                    x: parts[0] / svg_w,
                    y: parts[1] / svg_h,
                    w: parts[2] / svg_w,
                    h: parts[3] / svg_h,
                    pitch,
                });
                continue;
            }
        }

        // Fallback: find the first <use> element (note head) for x,y
        if let Some(use_node) = node.descendants().find(|n| n.has_tag_name("use")) {
            let nx = use_node.attribute("x").and_then(|s| s.parse::<f32>().ok());
            let ny = use_node.attribute("y").and_then(|s| s.parse::<f32>().ok());
            if let (Some(nx), Some(ny)) = (nx, ny) {
                // Note heads ~12x12 SVG units
                let note_w = 14.0 / svg_w;
                let note_h = 14.0 / svg_h;
                positions.push(NotePosition {
                    x: (nx - 7.0) / svg_w,
                    y: (ny - 7.0) / svg_h,
                    w: note_w,
                    h: note_h,
                    pitch,
                });
            }
        }
    }
    positions
}

fn render_engraved_pages_with_verovioxide(
    ctx: &egui::Context,
    musicxml: &str,
    cache_key: &SheetPreviewCacheKey,
    _total_beats: f32,
) -> Result<Vec<EngravedSheetPage>, String> {
    let mut toolkit = Toolkit::new().map_err(|err| format!("verovioxide init failed: {err}"))?;
    toolkit
        .load_data(musicxml)
        .map_err(|err| format!("verovioxide could not parse MusicXML: {err}"))?;

    let opts = Options::builder()
        .svg_bounding_boxes(true)
        .build();
    toolkit.set_options(&opts)
        .map_err(|err| format!("verovioxide options failed: {err}"))?;

    let svg_pages: Vec<String> = toolkit
        .render(Svg::all_pages())
        .map_err(|err| format!("verovioxide SVG render failed: {err}"))?;

    let dpi_scale = ctx.pixels_per_point().max(1.0);
    let render_w = ((3000.0 * dpi_scale).ceil() as u32).min(4096);
    let png_pages: Vec<Vec<u8>> = toolkit
        .render(Png::all_pages().width(render_w).white_background())
        .map_err(|err| format!("verovioxide PNG render failed: {err}"))?;

    if png_pages.is_empty() {
        return Err("verovioxide returned zero rendered pages.".to_string());
    }

    let page_count = png_pages.len().max(1);
    let mut pages = Vec::with_capacity(page_count);

    for (page_idx, page_bytes) in png_pages.into_iter().enumerate() {
        let rgba = image::load_from_memory(page_bytes.as_slice())
            .map_err(|err| format!("failed decoding rendered PNG page {}: {err}", page_idx + 1))?
            .to_rgba8();

        let width_px = rgba.width() as usize;
        let height_px = rgba.height() as usize;
        if width_px == 0 || height_px == 0 {
            continue;
        }

        let color_image =
            egui::ColorImage::from_rgba_unmultiplied([width_px, height_px], rgba.as_raw());
        let texture = ctx.load_texture(
            format!(
                "sheet-engraved-{}-{}-{}-{}",
                cache_key.timeline_ptr,
                cache_key.timeline_len,
                cache_key.timeline_step_bits,
                page_idx
            ),
            color_image,
            egui::TextureOptions::LINEAR,
        );

        let note_positions = if page_idx < svg_pages.len() {
            parse_svg_note_positions(&svg_pages[page_idx])
        } else {
            Vec::new()
        };

        pages.push(EngravedSheetPage {
            texture,
            width_px,
            height_px,
            note_positions,
            beat_start: 0.0,
            beat_end: 0.0,
        });
    }

    if pages.is_empty() {
        Err("verovioxide returned pages, but no valid PNG image could be decoded.".to_string())
    } else {
        Ok(pages)
    }
}

fn draw_scrollable_engraved_preview(
    ui: &mut egui::Ui,
    pages: &[EngravedSheetPage],
    _current_beat: f32,
    accent: egui::Color32,
    active_notes: &[u8],
) {
    if pages.is_empty() {
        ui.label("No engraved pages available for preview.");
        return;
    }

    let highlight_fill = egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200);
    let highlight_stroke = egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 255);

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

                // Draw note highlights
                if !active_notes.is_empty() {
                    let active_set: std::collections::BTreeSet<u8> =
                        active_notes.iter().copied().collect();
                    for np in &page.note_positions {
                        if active_set.contains(&np.pitch) {
                            let rx = response.rect.left() + np.x * target_width;
                            let ry = response.rect.top() + np.y * target_height;
                            let rw = np.w * target_width;
                            let rh = np.h * target_height;
                            let note_rect = egui::Rect::from_min_size(
                                egui::pos2(rx, ry),
                                egui::vec2(rw, rh),
                            );
                            ui.painter().rect_filled(note_rect, 2.0, highlight_fill);
                            ui.painter().rect_stroke(
                                note_rect,
                                2.0,
                                egui::Stroke::new(1.5, highlight_stroke),
                            );
                        }
                    }
                }

                if idx + 1 < pages.len() {
                    ui.add_space(UI_VSPACE_COMPACT);
                }
            }
        });
}

#[derive(Clone)]
struct NoteSpan {
    start_tick: i32,
    end_tick: i32,
    pitch: u8,
    velocity: u8,
    staff: u8,
    articulation: Articulation,
}

#[derive(Clone)]
struct NoteChunk {
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
}

impl Default for SheetEngravingConfig {
    fn default() -> Self {
        Self {
            allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: true,
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

        let mut cursor = boundaries.last().copied().unwrap_or(0).max(seg_start_tick);
        while cursor < seg_end_tick.min(max_tick) {
            let next = (cursor + mw).min(seg_end_tick);
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

    // Determine per-measure clef for lead sheet based on average pitch
    let measure_clefs: Vec<&'static str> = if config.is_lead_sheet {
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

        if measure_idx == 0 || time_signature_change_by_measure.contains_key(&(measure_idx as i32)) {
            let _ = write!(xml, "      <attributes>\n");
            if measure_idx == 0 {
                let _ = write!(xml, "        <divisions>{}</divisions>\n", MUSICXML_DIVISIONS);
                let _ = write!(xml, "        <key><fifths>0</fifths></key>\n");
                if config.is_lead_sheet {
                    // Lead sheet: always start with treble clef
                    let _ = write!(xml, "        <clef><sign>G</sign><line>2</line></clef>\n");
                } else {
                    let _ = write!(xml, "        <staves>2</staves>\n");
                    let _ = write!(xml, "        <clef number=\"1\"><sign>G</sign><line>2</line></clef>\n");
                    let _ = write!(xml, "        <clef number=\"2\"><sign>F</sign><line>4</line></clef>\n");
                }
            }
            let _ = write!(
                xml,
                "        <time><beats>{}</beats><beat-type>{}</beat-type></time>\n",
                current_time_sig.0,
                current_time_sig.1
            );

            // Dynamic clef for lead sheet: switch per-measure based on avg pitch
            if config.is_lead_sheet {
                if let Some(&clef) = measure_clefs.get(measure_idx) {
                    if clef != current_clef {
                        current_clef = clef;
                        let (sign, line) = if clef == "F" { ("F", 4) } else { ("G", 2) };
                        let _ = write!(
                            xml,
                            "        <clef><sign>{}</sign><line>{}</line></clef>\n",
                            sign, line
                        );
                    }
                }
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

        if config.is_lead_sheet {
            let voice_map = build_voice_chunks(chunks.as_slice(), 1);
            let voice_count = voice_map.len().max(1);
            let mut rendered = 0usize;
            for (voice, mut voice_chunks) in voice_map {
                voice_chunks.sort_by_key(|chunk| chunk.start_tick_in_measure);
                write_voice_sequence(
                    &mut xml,
                    voice_chunks.as_slice(),
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

fn reduce_to_monophonic_melody(notes: &[NoteEvent]) -> Vec<NoteEvent> {
    if notes.is_empty() {
        return Vec::new();
    }

    let mut sorted = notes.to_vec();
    sorted.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.pitch.cmp(&a.pitch))
    });

    let mut melody: Vec<NoteEvent> = Vec::new();
    for note in sorted {
        let overlaps = melody.iter().any(|m| {
            note.start_time < m.end_time && note.end_time > m.start_time
        });
        if !overlaps {
            melody.push(note);
        }
    }

    melody.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    melody
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
    let std_durations: [f32; 12] = [
        4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625,
    ];

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
        let staff = if config.is_lead_sheet {
            1
        } else if note.pitch >= GRAND_STAFF_SPLIT_MIDI {
            1
        } else {
            2
        };
        out.push(NoteSpan {
            start_tick: start_tick.max(0),
            end_tick: (start_tick + duration_ticks).max(start_tick + 1),
            pitch: note.pitch,
            velocity: note.velocity,
            staff,
            articulation: note.articulation,
        });
    }

    out.sort_by_key(|n| n.start_tick);
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
        1 => ("C", 1),
        2 => ("D", 0),
        3 => ("D", 1),
        4 => ("E", 0),
        5 => ("F", 0),
        6 => ("F", 1),
        7 => ("G", 0),
        8 => ("G", 1),
        9 => ("A", 0),
        10 => ("A", 1),
        _ => ("B", 0),
    }
}

fn write_rest_ticks(
    xml: &mut String,
    duration_ticks: i32,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let tokens = duration_tokens_for_ticks(duration_ticks, divisions, config);
    for token in tokens {
        let _ = write!(xml, "      <note>\n");
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
    }
}

fn write_note_chunk(
    xml: &mut String,
    chunk: &NoteChunk,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let note_id = format!("n{}_{}", chunk.pitch, chunk.absolute_tick);
    if chunk.articulation == Articulation::Grace {
        let grace_token = DurationToken {
            ticks: 0,
            note_type: "grace",
            dots: 0,
            time_mod: None,
        };
        write_note_element(
            xml,
            &note_id,
            chunk.pitch,
            grace_token,
            staff,
            voice,
            chunk.velocity,
            false,
            false,
            false,
            chunk.articulation,
        );
        return;
    }

    let tokens = duration_tokens_for_ticks(chunk.duration_ticks, divisions, config);
    for (idx, token) in tokens.iter().enumerate() {
        let local_tie_stop = (idx > 0) || (idx == 0 && chunk.tie_stop);
        let local_tie_start =
            (idx + 1 < tokens.len()) || (idx + 1 == tokens.len() && chunk.tie_start);
        let token_note_id = if tokens.len() > 1 {
            format!("{}_t{}", note_id, idx)
        } else {
            note_id.clone()
        };
        write_note_element(
            xml,
            &token_note_id,
            chunk.pitch,
            *token,
            staff,
            voice,
            chunk.velocity,
            false,
            local_tie_start,
            local_tie_stop,
            chunk.articulation,
        );
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
) {
    let (step, alter, octave) = midi_to_pitch_parts(pitch);
    let _ = write!(xml, "      <note xml:id=\"{}\">\n", xml_escape(note_id));
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
    // (type_name, expected_ticks, dots)
    let types = [
        ("whole", d * 4, 0),
        ("half", d * 2, 0),
        ("quarter", d, 0),
        ("eighth", d / 2, 0),
        ("16th", d / 4, 0),
        ("32nd", d / 8, 0),
    ];
    let mut best = ("16th", d / 4, 0i32);
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
    config: SheetEngravingConfig,
) -> Vec<DurationToken> {
    let d = divisions.max(1);
    let candidates = [
        DurationToken {
            ticks: d * 4,
            note_type: "whole",
            dots: 0,
            time_mod: None,
        },
        DurationToken {
            ticks: d * 3,
            note_type: "half",
            dots: 1,
            time_mod: None,
        },
        DurationToken {
            ticks: d * 2,
            note_type: "half",
            dots: 0,
            time_mod: None,
        },
        DurationToken {
            ticks: d + d / 2,
            note_type: "quarter",
            dots: 1,
            time_mod: None,
        },
        DurationToken {
            ticks: d,
            note_type: "quarter",
            dots: 0,
            time_mod: None,
        },
        DurationToken {
            ticks: d / 2 + d / 4,
            note_type: "eighth",
            dots: 1,
            time_mod: None,
        },
        DurationToken {
            ticks: d / 2,
            note_type: "eighth",
            dots: 0,
            time_mod: None,
        },
        DurationToken {
            ticks: d / 3,
            note_type: "eighth",
            dots: 0,
            time_mod: Some((3, 2)),
        },
        DurationToken {
            ticks: d / 4 + d / 8,
            note_type: "16th",
            dots: 1,
            time_mod: None,
        },
        DurationToken {
            ticks: d / 4,
            note_type: "16th",
            dots: 0,
            time_mod: None,
        },
        DurationToken {
            ticks: d / 6,
            note_type: "16th",
            dots: 0,
            time_mod: Some((3, 2)),
        },
    ];

    let mut remaining = duration_ticks.max(1);
    let mut out = Vec::new();

    while remaining > 0 {
        let mut chosen = None;
        for candidate in &candidates {
            if candidate.ticks <= 0 {
                continue;
            }
            if !config.allow_triplets && candidate.time_mod.is_some() {
                continue;
            }
            if candidate.ticks <= remaining {
                chosen = Some(*candidate);
                break;
            }
        }

        let token = chosen.unwrap_or_else(|| {
            let (nt, _expected, _dots) = note_type_for_ticks(remaining, divisions);
            DurationToken {
                ticks: remaining,
                note_type: nt,
                dots: 0,
                time_mod: None,
            }
        });

        let tick = token.ticks.max(1).min(remaining);
        out.push(DurationToken {
            ticks: tick,
            note_type: token.note_type,
            dots: token.dots,
            time_mod: token.time_mod,
        });

        remaining -= tick;
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
    measure_ticks: i32,
    divisions: i32,
    staff: u8,
    voice: u8,
    config: SheetEngravingConfig,
) {
    let min_rest_ticks = (divisions / 4).max(1);
    let mut cursor = 0i32;

    for chunk in chunks {
        if chunk.start_tick_in_measure > cursor {
            let gap = chunk.start_tick_in_measure - cursor;
            if gap >= min_rest_ticks {
                write_rest_ticks(
                    xml,
                    gap.min(measure_ticks - cursor),
                    divisions,
                    staff,
                    voice,
                    config,
                );
            }
            cursor = chunk.start_tick_in_measure.min(measure_ticks);
        }

        if cursor >= measure_ticks {
            break;
        }

        let chunk_dur = chunk.duration_ticks.min(measure_ticks - cursor);
        if chunk_dur > 0 {
            write_note_chunk(xml, chunk, divisions, staff, voice, config);
            cursor = (chunk.start_tick_in_measure + chunk_dur).min(measure_ticks);
        } else {
            cursor = (chunk.start_tick_in_measure + chunk.duration_ticks).min(measure_ticks);
        }
    }

    let remaining = measure_ticks - cursor;
    if remaining >= min_rest_ticks {
        write_rest_ticks(xml, remaining, divisions, staff, voice, config);
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

#[allow(dead_code)]
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
