use crate::app::KeyScribeApp;
use eframe::egui;
use std::path::Path;

impl KeyScribeApp {
    pub(crate) fn draw_export_modals(&mut self, ctx: &egui::Context) {
        let mut stems_open = self.export_stems_modal_open;
        if stems_open {
            let mut close_modal = false;
            egui::Window::new("Export Stems (Audio)")
                .collapsible(false)
                .resizable(false)
                .open(&mut stems_open)
                .show(ctx, |ui| {
                    let current_model = self.current_separation_model_name();
                    if self.separated_stems.is_none() || self.loaded_stems_model_name.as_deref() != Some(&current_model) {
                        self.request_cached_stems(&current_model);
                    }
                    let model_display = self.separation_model_display_name(&current_model);
                    ui.label(egui::RichText::new(format!("Model: {model_display}")).strong());
                    ui.add_space(4.0);

                    self.draw_export_stem_selection(ui);
                    ui.add_space(8.0);
                    if ui.button("Select Destination & Export").clicked() {
                        // The folder picker runs on a worker thread (see
                        // `spawn_file_dialog`): the modal closes immediately
                        // and the export fires when a folder is chosen, so a
                        // long browsing session never stalls the event loop.
                        #[cfg(feature = "desktop-ui")]
                        self.spawn_file_dialog(
                            ui.ctx(),
                            super::runtime::FileDialogRequest::ExportStemsFolder,
                        );
                        close_modal = true;
                    }
                });
            self.export_stems_modal_open = stems_open && !close_modal;
        }

        let mut midi_open = self.export_midi_modal_open;
        if midi_open {
            let mut close_modal = false;
            egui::Window::new("Export MIDI")
                .collapsible(false)
                .resizable(false)
                .open(&mut midi_open)
                .show(ctx, |ui| {
                    // Original mix (full audio, no stem separation required).
                    ui.checkbox(&mut self.export_full_mix_midi, "Full Mix");
                    self.draw_export_stem_selection(ui);
                    ui.add_space(8.0);
                    
                    let mut ui_key_sensitivity = (self.key_color_sensitivity * 0.5).clamp(0.0, 1.0);
                    let slider_response = ui.horizontal(|ui| {
                        let changed = Self::top_bar_slider_with_input(
                            ui,
                            "Note Sensitivity",
                            &mut ui_key_sensitivity,
                            0.0,
                            1.0,
                            "",
                            0.01,
                            2,
                        );
                        if changed {
                            self.key_color_sensitivity = (ui_key_sensitivity * 2.0).clamp(0.0, 2.0);
                        }
                    }).response;
                    slider_response.on_hover_text("Adjust note sensitivity for the MIDI export. Higher sensitivity will detect more notes, while lower sensitivity will filter out softer/background notes.");
                    
                    ui.add_space(8.0);
                    if ui.button("Select Destination & Export").clicked() {
                        // Async folder picker (see above): the modal closes
                        // now, the export fires on choice.
                        #[cfg(feature = "desktop-ui")]
                        self.spawn_file_dialog(
                            ui.ctx(),
                            super::runtime::FileDialogRequest::ExportMidiFolder,
                        );
                        close_modal = true;
                    }
                });
            self.export_midi_modal_open = midi_open && !close_modal;
        }
    }

    fn draw_export_stem_selection(&mut self, ui: &mut egui::Ui) {
        ui.label("Select stems to export:");
        if let Some(stems) = &self.separated_stems {
            for stem in stems {
                let mut selected = self.export_selected_stems.contains(&stem.stem_type);
                if ui.checkbox(&mut selected, stem.stem_type.display_name().as_ref()).changed() {
                    if selected {
                        self.export_selected_stems.insert(stem.stem_type.clone());
                    } else {
                        self.export_selected_stems.remove(&stem.stem_type);
                    }
                }
            }
        }
    }

    pub(super) fn execute_export_stems(&mut self, dest_folder: &Path) {
        let current_model = self.current_separation_model_name();
        if self.separated_stems.is_none() || self.loaded_stems_model_name.as_deref() != Some(&current_model) {
            self.request_cached_stems(&current_model);
            self.last_error = Some(
                "Loading cached stems in the background — try the export again in a moment."
                    .to_string(),
            );
            return;
        }

        let mut exported_count = 0;
        if let Some(stems) = &self.separated_stems {
            for stem in stems {
                if self.export_selected_stems.contains(&stem.stem_type) {
                    let file_name = format!("{}.wav", stem.stem_type.display_name());
                    let path = dest_folder.join(file_name);
                    let spec = hound::WavSpec {
                        channels: stem.channels,
                        sample_rate: stem.sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    };
                    match hound::WavWriter::create(&path, spec) {
                        Ok(mut writer) => {
                            for &s in stem.samples_interleaved.iter() {
                                let _ = writer.write_sample(s);
                            }
                            if let Err(e) = writer.finalize() {
                                self.last_error = Some(format!("Failed to finalize WAV {:?}: {e}", path));
                            } else {
                                exported_count += 1;
                            }
                        }
                        Err(e) => {
                            self.last_error = Some(format!("Failed to create WAV file {:?}: {e}", path));
                        }
                    }
                }
            }
        }

        if exported_count > 0 {
            self.cache_status_message = Some(format!("Exported {exported_count} stem(s) to {:?}", dest_folder));
            self.cache_status_message_at = Some(std::time::Instant::now());
        }
    }

    pub(super) fn execute_export_midi(&self, dest_folder: &Path) {
        // Export the original/full mix when requested.
        if self.export_full_mix_midi && !self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0 {
            let mut next_id = 0;
            let notes = Self::extract_events_from_timeline_data(
                &self.note_timeline,
                self.note_timeline_step_sec,
                self.sheet_preview_threshold(),
                &mut next_id,
            );
            let path = dest_folder.join("Original Mix.mid");
            let _ = crate::midi::write_midi(&notes, &path, 120.0);
        }

        // Export per-stem MIDI when stems are available and selected.
        if let Some(stems) = &self.separated_stems {
            for (stem_idx, stem) in stems.iter().enumerate() {
                if self.export_selected_stems.contains(&stem.stem_type) {
                    if let Some(analysis) = self.stem_analyses.iter().find(|a| a.stem_index == stem_idx) {
                        let mut next_id = 0;
                        let notes = Self::extract_events_from_timeline_data(
                            &analysis.timeline,
                            analysis.step_sec,
                            self.sheet_preview_threshold(),
                            &mut next_id,
                        );

                        let file_name = format!("{}.mid", stem.stem_type.display_name());
                        let path = dest_folder.join(file_name);
                        let _ = crate::midi::write_midi(&notes, &path, 120.0);
                    }
                }
            }
        }
    }
}

