use super::*;

const TOOLBAR_MENU_MIN_WIDTH: f32 = 360.0;

impl KeyScribeApp {
    fn draw_toolbar_separator(ui: &mut egui::Ui) {
        let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
        let width = ui.available_width().max(0.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
        ui.painter().hline(rect.x_range(), rect.center().y, stroke);
    }

    pub(super) fn draw_audio_settings_menu(&mut self, ui: &mut egui::Ui) {
        ui.label("Audio Processing");
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

        Self::draw_toolbar_separator(ui);

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

        Self::draw_toolbar_separator(ui);

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

        if preprocess_changed || cqt_changed {
            self.request_rebuild_preserving_playback();
        }
    }

    pub(super) fn draw_preferences_menu(&mut self, ui: &mut egui::Ui) {
        setting_toggle_row(ui, &mut self.dark_mode, "Dark Mode");
        let _ = setting_toggle_row(ui, &mut self.show_note_hist_window, "Show Probability Pane");
        Self::draw_toolbar_separator(ui);

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

        Self::draw_toolbar_separator(ui);
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
    }

    #[cfg(not(feature = "desktop-ui"))]
    pub(super) fn draw_settings_menu(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(if self.is_touch_platform() {
            280.0
        } else {
            TOOLBAR_MENU_MIN_WIDTH
        });
        self.draw_audio_settings_menu(ui);
        Self::draw_toolbar_separator(ui);
        self.draw_preferences_menu(ui);
    }

    #[cfg(feature = "desktop-ui")]
    fn draw_desktop_menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let recent_files = self.recent_file_paths.clone();
        let mut selected_recent_file: Option<PathBuf> = None;

        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                ui.set_min_width(TOOLBAR_MENU_MIN_WIDTH);
                if ui.button("Open Audio...").clicked() {
                    self.import_audio_with_ctx(ctx);
                    ui.close_menu();
                }

                ui.menu_button("Open Recent", |ui| {
                    ui.set_min_width(TOOLBAR_MENU_MIN_WIDTH);
                    if recent_files.is_empty() {
                        ui.add_enabled(false, egui::Button::new("No recent files"));
                        return;
                    }

                    for recent_path in recent_files.iter() {
                        let label = recent_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("(unknown)");
                        let response = ui
                            .button(label)
                            .on_hover_text(recent_path.display().to_string());
                        if response.clicked() {
                            selected_recent_file = Some(recent_path.clone());
                            ui.close_menu();
                        }
                    }

                    Self::draw_toolbar_separator(ui);
                    if ui.button("Clear Recent").clicked() {
                        self.recent_file_paths.clear();
                        ui.close_menu();
                    }
                });
            });

            ui.menu_button("Settings", |ui| {
                ui.set_min_width(TOOLBAR_MENU_MIN_WIDTH);
                self.draw_audio_settings_menu(ui);
            });

            ui.menu_button("Preferences", |ui| {
                ui.set_min_width(TOOLBAR_MENU_MIN_WIDTH);
                self.draw_preferences_menu(ui);
            });
        });

        if let Some(path) = selected_recent_file {
            if let Err(err) = self.start_audio_loading_from_path(path, ctx) {
                self.last_error = Some(err);
            }
        }
    }

    pub(super) fn top_bar_slider_with_input(
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
        let row_height = 22.0;
        let slider_width = 142.0;
        let input_width = 74.0;

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
                    let (label_rect, _) = ui.allocate_exact_size(
                        egui::vec2(label_width, row_height),
                        egui::Sense::hover(),
                    );
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
                                [slider_width, row_height],
                                egui::Slider::new(value, min..=max)
                                    .show_value(false)
                                    .suffix(suffix),
                            )
                            .changed();

                        changed |= ui
                            .add_sized(
                                [input_width, row_height],
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

    pub(super) fn draw_top_controls_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            egui::Frame::none()
                .inner_margin(egui::Margin::symmetric(12.0, 10.0))
                .show(ui, |ui| {
                    #[cfg(feature = "desktop-ui")]
                    {
                        self.draw_desktop_menu_bar(ui, ctx);
                        Self::draw_toolbar_separator(ui);
                    }

                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 12.0;

                        #[cfg(not(feature = "desktop-ui"))]
                        if ui.button("Open Audio").clicked() {
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

                        #[cfg(not(feature = "desktop-ui"))]
                        let settings_popup_id = ui.make_persistent_id("settings_popup_menu");
                        #[cfg(not(feature = "desktop-ui"))]
                        let settings_response = icon_button(ui, GEAR, "Settings", true);
                        #[cfg(not(feature = "desktop-ui"))]
                        if settings_response.clicked() {
                            ui.memory_mut(|mem| mem.toggle_popup(settings_popup_id));
                        }
                        #[cfg(not(feature = "desktop-ui"))]
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

                    if self.is_touch_platform() {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Touch Navigation");
                            if ui
                                .selectable_label(!self.touch_loop_select_mode, "Pan")
                                .clicked()
                            {
                                self.touch_loop_select_mode = false;
                            }
                            if ui
                                .selectable_label(self.touch_loop_select_mode, "Loop Select")
                                .clicked()
                            {
                                self.touch_loop_select_mode = true;
                            }
                            ui.label("Tap to seek, drag to pan, pinch to zoom.");
                        });
                    }

                    #[cfg(not(feature = "desktop-ui"))]
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Audio Path");
                        let input_width = ui.available_width().clamp(180.0, 460.0);
                        ui.add_sized(
                            [input_width, 30.0],
                            egui::TextEdit::singleline(&mut self.manual_import_path)
                                .hint_text("/sdcard/Music/song.mp3"),
                        );
                        if ui.button("Open").clicked() {
                            self.import_audio_from_manual_path(ctx);
                        }
                    });

                    if let Some(err) = &self.last_error {
                        ui.colored_label(ERROR_RED, err);
                    }

                    if self.is_processing {
                        let msg = match self.active_rebuild_mode {
                            RebuildMode::Full if self.preprocess_audio => {
                                "Analyzing track in background... waveform and playback stay available."
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

                    if let Some(cache_msg) = self.cache_status_message.as_deref() {
                        let show_cache_msg = self.is_processing
                            || self
                                .cache_status_message_at
                                .map(|at| at.elapsed() <= Duration::from_secs(8))
                                .unwrap_or(false);
                        if show_cache_msg {
                            ui.label(cache_msg);
                        }
                    }

                    let buffered_sec = if self.loading_sample_rate > 0 {
                        self.loading_decoded_samples as f32 / self.loading_sample_rate as f32
                    } else {
                        0.0
                    };

                    let buffered_ratio = self
                        .loading_total_samples
                        .map(|total| {
                            if total == 0 {
                                0.0
                            } else {
                                (self.loading_decoded_samples as f32 / total as f32)
                                    .clamp(0.0, 1.0)
                            }
                        })
                        .unwrap_or_else(|| {
                            // Fall back to a saturating estimate when container metadata lacks total frames.
                            (buffered_sec / (buffered_sec + 30.0)).clamp(0.0, 0.95)
                        });

                    let total_sec = if self.loading_sample_rate > 0 {
                        self.loading_total_samples
                            .map(|total| total as f32 / self.loading_sample_rate as f32)
                    } else {
                        None
                    };

                    let show_rendered_row = self.is_audio_loading;
                    let transcription_ready_from_cache = self.preprocess_audio
                        && self.loading_cache_timeline_preloaded
                        && !self.note_timeline.is_empty()
                        && self.note_timeline_step_sec > 0.0;
                    let show_transcribed_row = if self.is_audio_loading {
                        true
                    } else {
                        self.is_processing
                            && self.preprocess_audio
                            && self.active_rebuild_mode == RebuildMode::Full
                            && !transcription_ready_from_cache
                    };

                    if show_rendered_row || show_transcribed_row {
                        let progress_label_width = 86.0;
                        let progress_bar_width = 220.0;

                        let draw_progress_row =
                            |ui: &mut egui::Ui,
                             label: &str,
                             ratio: f32,
                             animate: bool,
                             detail: &str| {
                                ui.horizontal(|ui| {
                                    ui.add_sized(
                                        [progress_label_width, 0.0],
                                        egui::Label::new(label),
                                    );
                                    ui.add(
                                        egui::ProgressBar::new(ratio)
                                            .desired_width(progress_bar_width)
                                            .animate(animate),
                                    );
                                    ui.label(detail);
                                });
                            };

                        if show_rendered_row {
                            let rendered_detail = if let Some(total_sec) = total_sec {
                                format!("{buffered_sec:.1}s / {total_sec:.1}s")
                            } else {
                                format!("{buffered_sec:.1}s buffered")
                            };

                            draw_progress_row(
                                ui,
                                "Rendered",
                                buffered_ratio,
                                self.loading_total_samples.is_none(),
                                rendered_detail.as_str(),
                            );
                        }

                        if show_transcribed_row {
                            let (transcribed_ratio, transcribed_detail, animate_bar) =
                                if self.is_audio_loading {
                                    let ratio = if self.preprocess_audio {
                                        if self.loading_cache_timeline_preloaded {
                                            1.0
                                        } else {
                                            (buffered_ratio * TRANSCRIBE_PROGRESS_LOADING_WEIGHT)
                                                .clamp(0.0, TRANSCRIBE_PROGRESS_LOADING_WEIGHT)
                                        }
                                    } else {
                                        buffered_ratio
                                    };

                                    let detail = if self.preprocess_audio {
                                        if self.loading_cache_timeline_preloaded {
                                            "Loaded from cache (ready before render ends)."
                                                .to_string()
                                        } else {
                                            format!(
                                                "Queued until render completes ({:.0}% rendered)",
                                                buffered_ratio * 100.0
                                            )
                                        }
                                    } else {
                                        format!("{:.0}%", buffered_ratio * 100.0)
                                    };

                                    (ratio, detail, !self.loading_cache_timeline_preloaded)
                                } else {
                                    let elapsed = self
                                        .processing_started_at
                                        .map(|t| t.elapsed().as_secs_f32())
                                        .unwrap_or(0.0);
                                    let estimate = self
                                        .processing_estimated_total_sec
                                        .max(0.5)
                                        .max(elapsed + 1.0e-3);
                                    let stage_ratio = (elapsed / estimate).clamp(0.0, 1.0);
                                    let remaining = (estimate - elapsed).max(0.0);

                                    let ratio = (TRANSCRIBE_PROGRESS_LOADING_WEIGHT
                                        + stage_ratio
                                            * (1.0 - TRANSCRIBE_PROGRESS_LOADING_WEIGHT))
                                        .min(TRANSCRIBE_PROGRESS_MAX_BEFORE_DONE);
                                    let detail = format!(
                                        "{:.0}% estimated ({:.0}s left)",
                                        stage_ratio * 100.0,
                                        remaining
                                    );

                                    (ratio, detail, true)
                                };

                            draw_progress_row(
                                ui,
                                "Transcribed",
                                transcribed_ratio,
                                animate_bar,
                                transcribed_detail.as_str(),
                            );
                        }
                    }
                });
        });
    }
}
