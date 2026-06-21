use super::*;

impl eframe::App for KeyScribeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_brand_theme(ctx, self.dark_mode, self.highlight_color);
        self.lock_startup_min_window_size_once(ctx);
        self.apply_mobile_ui_tweaks_once(ctx);

        let wants_keyboard = ctx.wants_keyboard_input();
        let (space_pressed, k_pressed, left_pressed, right_pressed, m_pressed, ctrl_held) = ctx.input(|i| {
            if wants_keyboard {
                (false, false, false, false, false, i.modifiers.ctrl)
            } else {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::K),
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                    i.key_pressed(egui::Key::M),
                    i.modifiers.ctrl,
                )
            }
        });

        if space_pressed {
            self.handle_space_replay();
        }
        if k_pressed {
            self.handle_toggle_play_pause();
        }
        if m_pressed {
            if let Some(hash) = &self.loaded_audio_hash {
                let markers = self.file_markers.entry(hash.clone()).or_default();
                markers.push(crate::app::MarkerData::Detailed { time: self.selected_time_sec, desc: String::new() });
                markers.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
            }
        }
        if left_pressed {
            if ctrl_held && self.shift_loop_by_seconds(-1.0) {
                // Ctrl+Arrow shifts loop range when looping is active.
            } else {
                self.skip_by_seconds(-SEEK_STEP_SEC);
            }
        }
        if right_pressed {
            if ctrl_held && self.shift_loop_by_seconds(1.0) {
                // Ctrl+Arrow shifts loop range when looping is active.
            } else {
                self.skip_by_seconds(SEEK_STEP_SEC);
            }
        }

        #[cfg(feature = "desktop-ui")]
        let (hovered_valid_drop, dropped_valid_drop, dropped_any_count) = ctx.input(|i| {
            let hovered_valid_drop = i
                .raw
                .hovered_files
                .iter()
                .filter_map(|file| file.path.as_ref())
                .find(|path| super::is_supported_media_extension(path.as_path()))
                .cloned();

            let dropped_valid_drop = i
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.as_ref())
                .find(|path| super::is_supported_media_extension(path.as_path()))
                .cloned();

            (
                hovered_valid_drop,
                dropped_valid_drop,
                i.raw.dropped_files.len(),
            )
        });

        #[cfg(feature = "desktop-ui")]
        if let Some(path) = dropped_valid_drop {
            match self.start_audio_loading_from_path(path, ctx) {
                Ok(()) => {
                    self.last_error = None;
                }
                Err(err) => {
                    self.last_error = Some(err);
                }
            }
        } else if dropped_any_count > 0 {
            self.last_error =
                Some("Drop a supported media file (Audio: wav, mp3, flac, ogg, m4a, aac | Video: mp4, mkv, avi, mov, webm).".to_string());
        }

        self.poll_audio_loading(ctx);

        // Auto-run separation once audio is fully loaded (not during streaming).
        if self.auto_separate
            && !self.is_audio_loading
            && self.audio_raw.is_some()
            && self.loaded_audio_hash.is_some()
            && self.separated_stems.is_none()
            && !self.is_separating
            && !self.separation_attempted
        {
            let duration = self.source_duration() as f64;
            if duration > super::AUTO_SEPARATE_MAX_DURATION_SEC {
                self.separation_attempted = true;
                self.cache_status_message = Some("Stem separation is manual for this source since it is longer than 10 minutes.".to_string());
                self.cache_status_message_at = Some(std::time::Instant::now());
            } else {
                self.run_instrument_separation();
            }
        }

        self.poll_processing_result();
        self.poll_separation_result();
        self.poll_stem_analysis_result();
        self.poll_streaming_playback();
        self.poll_sheet_preview(ctx);
        self.poll_sheet_rendering(ctx);
        self.sync_playhead_from_engine();

        self.draw_top_controls_panel(ctx);

        self.piano_panel_height_needs_init = false;
        if self.audio_raw.is_some() {
            let piano_panel_builder = egui::TopBottomPanel::bottom("piano_panel")
                .frame(
                    egui::Frame::none()
                        .inner_margin(egui::Margin::symmetric(UI_VSPACE_MEDIUM, 0.0)),
                )
                .show_separator_line(false)
                .resizable(false)
                .min_height(40.0);
            let piano_panel = piano_panel_builder.show(ctx, |ui| {
                let note_visuals_ready = self.note_visuals_ready();
                if !note_visuals_ready {
                    self.clear_note_visuals();
                }

                let pane_rect = ui.max_rect();
                let pane_hovered = ui.rect_contains_pointer(pane_rect);
                if pane_hovered && ui.input(|i| i.pointer.primary_clicked()) {
                    self.piano_has_focus = true;
                }

                let white_w_for_zoom =
                    keyboard_white_key_width(ui.available_width(), self.piano_zoom);
                let max_allowed_key_h = (white_w_for_zoom * WHITE_KEY_LENGTH_TO_WIDTH)
                    .clamp(MIN_PIANO_KEY_HEIGHT, 220.0);
                let ideal_key_h = max_allowed_key_h;
                let panel_visual_gap = UI_STACK_VSPACE;
                let default_item_spacing_y = ui.spacing().item_spacing.y;
                ui.spacing_mut().item_spacing.y = 0.0;

                let prob_strip_height_ideal = if self.show_note_hist_window && note_visuals_ready {
                    (ideal_key_h * 0.9).clamp(MIN_PROBABILITY_STRIP_HEIGHT, 120.0)
                } else {
                    0.0
                };

                let ideal_total_visual_h = prob_strip_height_ideal
                    + if prob_strip_height_ideal > 0.0 { panel_visual_gap } else { 0.0 }
                    + ideal_key_h;

                // Scale keys and probability pane together via the user-controlled
                // `piano_scale` factor (drag handle at top of panel).
                let height_scale = self.piano_scale;
                let key_h_for_frame = (ideal_key_h * height_scale).max(MIN_PIANO_KEY_HEIGHT);
                let prob_strip_height = (prob_strip_height_ideal * height_scale)
                    .max(if prob_strip_height_ideal > 0.0 { MIN_PROBABILITY_STRIP_HEIGHT } else { 0.0 });
                self.piano_key_height = key_h_for_frame;
                self.probability_panel_height = prob_strip_height;

                // Drag handle: drag up to shrink keys+probabilities together.
                let drag_h = 6.0;
                let drag_w = ui.available_width();
                let (drag_rect, drag_resp) =
                    ui.allocate_exact_size(egui::vec2(drag_w, drag_h), egui::Sense::drag());
                if drag_resp.hovered() || drag_resp.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
                if drag_resp.dragged() {
                    let delta_scale =
                        (-drag_resp.drag_delta().y) / ideal_total_visual_h.max(40.0);
                    self.piano_scale =
                        (self.piano_scale + delta_scale).clamp(0.25, 1.0);
                }
                let drag_color = if drag_resp.hovered() || drag_resp.dragged() {
                    self.highlight_color
                } else {
                    egui::Color32::from_gray(64)
                };
                ui.painter().rect_filled(
                    egui::Rect::from_center_size(
                        drag_rect.center(),
                        egui::vec2(drag_rect.width() * 0.3, drag_h.max(2.0)),
                    ),
                    2.0,
                    drag_color,
                );

                let mut max_scroll_px: f32 = 0.0;
                if self.show_note_hist_window && note_visuals_ready {
                    let prob_draw = draw_probability_pane(
                        ui,
                        &self.note_probs_smoothed,
                        self.note_probs.as_slice(),
                        &self.note_stem_colors,
                        self.piano_zoom,
                        self.piano_scroll_px,
                        prob_strip_height,
                        self.highlight_color,
                    );
                    max_scroll_px = max_scroll_px.max(prob_draw.max_scroll_px);
                    if self.show_chord_suggestions {
                        if let Some(chord) = &self.current_chord {
                            let pane = prob_draw.rect;
                            let overlay_w = pane.width().min(280.0);
                            let overlay_rect = egui::Rect::from_min_size(
                                egui::pos2(pane.left() + 6.0, pane.top() + 6.0),
                                egui::vec2(overlay_w, pane.height() - 12.0),
                            );
                            let painter = ui.painter();
                            painter.rect_filled(
                                overlay_rect,
                                8.0,
                                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 153),
                            );
                            painter.text(
                                egui::pos2(overlay_rect.left() + 10.0, overlay_rect.center().y),
                                egui::Align2::LEFT_CENTER,
                                chord,
                                egui::FontId::proportional(26.0),
                                egui::Color32::WHITE,
                            );
                        }
                    }
                    if prob_draw.clicked {
                        self.piano_has_focus = true;
                    }
                    ui.add_space(panel_visual_gap);
                }

                let piano_draw = draw_piano_view(
                    ui,
                    &self.note_probs_smoothed,
                    &self.note_stem_colors,
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

                let (raw, smooth, shift, ctrl, zoom_delta, pointer_down, pointer_pos) =
                    ui.ctx().input(|i| {
                        (
                            i.raw_scroll_delta,
                            i.smooth_scroll_delta,
                            i.modifiers.shift,
                            i.modifiers.ctrl,
                            i.zoom_delta_2d(),
                            i.pointer.primary_down(),
                            i.pointer.interact_pos(),
                        )
                    });
                let wheel_y = if raw.y.abs() > f32::EPSILON {
                    raw.y
                } else {
                    smooth.y
                };

                if self.piano_has_focus && pane_hovered {
                    if self.is_touch_platform() {
                        if pointer_down {
                            if let Some(pos) = pointer_pos {
                                if let Some(last_x) = self.piano_drag_last_x {
                                    let delta_x = pos.x - last_x;
                                    self.piano_scroll_px =
                                        (self.piano_scroll_px - delta_x).clamp(0.0, max_scroll_px);
                                }
                                self.piano_drag_last_x = Some(pos.x);
                            }
                        } else {
                            self.piano_drag_last_x = None;
                        }

                        if (zoom_delta.y - 1.0).abs() > f32::EPSILON {
                            self.piano_zoom = (self.piano_zoom * zoom_delta.y)
                                .clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX);
                        }
                    } else {
                        self.piano_drag_last_x = None;
                        if ctrl && wheel_y.abs() > f32::EPSILON {
                            let z = if wheel_y > 0.0 { 1.08 } else { 0.92 };
                            self.piano_zoom =
                                (self.piano_zoom * z).clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX);
                        } else if shift && wheel_y.abs() > f32::EPSILON {
                            self.piano_scroll_px =
                                (self.piano_scroll_px - wheel_y * 0.7).clamp(0.0, max_scroll_px);
                        }
                    }
                } else {
                    self.piano_drag_last_x = None;
                }

                ui.add_space(UI_VSPACE_TIGHT);
                ui.spacing_mut().item_spacing.y = default_item_spacing_y.min(UI_VSPACE_TIGHT);
                egui::Frame::none()
                    .inner_margin(egui::Margin::symmetric(12.0, UI_VSPACE_TIGHT))
                    .show(ui, |ui| {
                        let trio_gap = 12.0;
                        let trio_w = ui.available_width().max(0.0);
                        let stack_controls = trio_w < 780.0;
                        let slider_row_h = ui.spacing().interact_size.y.clamp(22.0, 30.0)
                            + UI_VSPACE_TIGHT * 2.0
                            + 2.0;
                        let mut visuals_changed = false;

                        if stack_controls {
                            ui.allocate_ui_with_layout(
                                egui::vec2(trio_w, 0.0),
                                egui::Layout::top_down(egui::Align::Center),
                                |ui| {
                                    let mut ui_key_sensitivity =
                                        (self.key_color_sensitivity * 0.5).clamp(0.0, 1.0);
                                    if Self::top_bar_slider_with_input(
                                        ui,
                                        "Key Color Sensitivity",
                                        &mut ui_key_sensitivity,
                                        0.0,
                                        1.0,
                                        "",
                                        0.01,
                                        2,
                                    ) {
                                        self.key_color_sensitivity =
                                            (ui_key_sensitivity * 2.0).clamp(0.0, 2.0);
                                        visuals_changed = true;
                                    }

                                    ui.add_space(UI_VSPACE_TIGHT);
                                    if Self::top_bar_slider_with_input(
                                        ui,
                                        "Max Key Highlight Time",
                                        &mut self.key_highlight_max_sec,
                                        KEY_HIGHLIGHT_MAX_SEC_MIN,
                                        KEY_HIGHLIGHT_MAX_SEC_MAX,
                                        " s",
                                        0.005,
                                        2,
                                    ) {
                                        self.key_highlight_max_sec = self
                                            .key_highlight_max_sec
                                            .clamp(KEY_HIGHLIGHT_MAX_SEC_MIN, KEY_HIGHLIGHT_MAX_SEC_MAX);
                                        visuals_changed = true;
                                    }

                                    ui.add_space(UI_VSPACE_TIGHT);
                                    if Self::top_bar_slider_with_input(
                                        ui,
                                        "Visualization Offset",
                                        &mut self.visualization_timing_offset_ms,
                                        VISUALIZATION_TIMING_OFFSET_MS_MIN,
                                        VISUALIZATION_TIMING_OFFSET_MS_MAX,
                                        " ms",
                                        1.0,
                                        0,
                                    ) {
                                        self.visualization_timing_offset_ms = self
                                            .visualization_timing_offset_ms
                                            .clamp(
                                                VISUALIZATION_TIMING_OFFSET_MS_MIN,
                                                VISUALIZATION_TIMING_OFFSET_MS_MAX,
                                            );
                                        visuals_changed = true;
                                    }

                                    ui.add_space(UI_VSPACE_TIGHT);
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
                                },
                            );
                        } else {
                            let col_w = ((trio_w - trio_gap * 3.0).max(0.0)) / 4.0;
                            ui.allocate_ui_with_layout(
                                egui::vec2(trio_w, slider_row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.spacing_mut().item_spacing.x = trio_gap;

                                    ui.allocate_ui_with_layout(
                                        egui::vec2(col_w, slider_row_h),
                                        egui::Layout::top_down(egui::Align::Center),
                                        |ui| {
                                            let mut ui_key_sensitivity =
                                                (self.key_color_sensitivity * 0.5).clamp(0.0, 1.0);
                                            if Self::top_bar_slider_with_input(
                                                ui,
                                                "Key Color Sensitivity",
                                                &mut ui_key_sensitivity,
                                                0.0,
                                                1.0,
                                                "",
                                                0.01,
                                                2,
                                            ) {
                                                self.key_color_sensitivity =
                                                    (ui_key_sensitivity * 2.0).clamp(0.0, 2.0);
                                                visuals_changed = true;
                                            }
                                        },
                                    );

                                    ui.allocate_ui_with_layout(
                                        egui::vec2(col_w, slider_row_h),
                                        egui::Layout::top_down(egui::Align::Center),
                                        |ui| {
                                            if Self::top_bar_slider_with_input(
                                                ui,
                                                "Max Key Highlight Time",
                                                &mut self.key_highlight_max_sec,
                                                KEY_HIGHLIGHT_MAX_SEC_MIN,
                                                KEY_HIGHLIGHT_MAX_SEC_MAX,
                                                " s",
                                                0.005,
                                                2,
                                            ) {
                                                self.key_highlight_max_sec = self
                                                    .key_highlight_max_sec
                                                    .clamp(
                                                        KEY_HIGHLIGHT_MAX_SEC_MIN,
                                                        KEY_HIGHLIGHT_MAX_SEC_MAX,
                                                    );
                                                visuals_changed = true;
                                            }
                                        },
                                    );

                                    ui.allocate_ui_with_layout(
                                        egui::vec2(col_w, slider_row_h),
                                        egui::Layout::top_down(egui::Align::Center),
                                        |ui| {
                                            if Self::top_bar_slider_with_input(
                                                ui,
                                                "Visualization Offset",
                                                &mut self.visualization_timing_offset_ms,
                                                VISUALIZATION_TIMING_OFFSET_MS_MIN,
                                                VISUALIZATION_TIMING_OFFSET_MS_MAX,
                                                " ms",
                                                1.0,
                                                0,
                                            ) {
                                                self.visualization_timing_offset_ms = self
                                                    .visualization_timing_offset_ms
                                                    .clamp(
                                                        VISUALIZATION_TIMING_OFFSET_MS_MIN,
                                                        VISUALIZATION_TIMING_OFFSET_MS_MAX,
                                                    );
                                                visuals_changed = true;
                                            }
                                        },
                                    );

                                    ui.allocate_ui_with_layout(
                                        egui::vec2(col_w, slider_row_h),
                                        egui::Layout::top_down(egui::Align::Center),
                                        |ui| {
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
                                        },
                                    );
                                },
                            );
                        }

                        if visuals_changed {
                            self.update_note_probabilities(true);
                        }

                        if self.note_probs.iter().any(|&p| p > 0.1) {
                            let _active_keys: Vec<u8> = self.note_probs.iter().enumerate()
                                .filter(|(_, &p)| p > 0.1)
                                .map(|(i, _)| (PIANO_LOW_MIDI as usize + i) as u8)
                                .collect();
                            // println!("[DEBUG] Visualization Active. ProbSum={:.2}, ActiveKeys={:?}", 
                            //    self.note_probs.iter().sum::<f32>(), active_keys);
                        }

                    });
                ui.add_space(UI_VSPACE_TIGHT);
                ui.spacing_mut().item_spacing.y = default_item_spacing_y;
            });
            self.piano_panel_height = piano_panel.response.rect.height().max(80.0);
            self.probability_panel_height = if self.show_note_hist_window && self.probability_panel_height > 0.0 {
                self.probability_panel_height
            } else {
                0.0
            };
        } else {
            self.piano_panel_height = 0.0;
            self.probability_panel_height = 0.0;
        }

        let waveform_central = egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(egui::Margin::symmetric(UI_VSPACE_MEDIUM, 0.0)))
            .show(ctx, |ui| {
                if self.audio_raw.is_none() && !self.is_audio_loading {
                    let import_surface_rect = ui.max_rect();
                    let import_surface_id = ui.make_persistent_id("empty_audio_import_surface");
                    let import_surface =
                        ui.interact(import_surface_rect, import_surface_id, egui::Sense::click());

                    if import_surface.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }

                    paint_audio_import_overlay(
                        ui.painter(),
                        import_surface_rect,
                        self.highlight_color,
                        168,
                        "Click Or Drag Media File",
                        "To Start Transcribing (Audio: wav, mp3, flac, ogg, m4a, aac | Video: mp4, mkv, mov, avi, webm)",
                    );

                    #[cfg(feature = "desktop-ui")]
                    if import_surface.clicked() {
                        self.import_audio_with_ctx(ctx);
                    }

                    return;
                }

                let source_duration = self.source_duration().max(0.01);
                let plot_duration = self.waveform_view_duration().max(source_duration).max(0.01);
                let interaction_duration = if self.is_audio_loading
                    && (self.loading_cache_waveform_preloaded
                        || self.loading_cache_timeline_preloaded)
                {
                    plot_duration
                } else {
                    source_duration
                };
                let waveform_visual_gap = UI_STACK_VSPACE;
                let analysis_ready =
                    !self.is_blocking_processing() && !self.processed_samples.is_empty();
                let interaction_ready = analysis_ready
                    || (self.is_audio_loading && self.loading_decoded_samples > 0);
                let default_stack_spacing_y = ui.spacing().item_spacing.y;
                ui.spacing_mut().item_spacing.y = 0.0;

                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.y = default_stack_spacing_y.min(UI_VSPACE_TIGHT);
                    self.draw_speed_pitch_controls(ui);
                });

                ui.add_space(UI_VSPACE_TIGHT);
                if !self.auto_separate || (self.separation_attempted && self.separated_stems.is_none()) {
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .add_enabled(
                                self.audio_raw.is_some(),
                                egui::Button::new("Separate Instruments"),
                            )
                            .clicked()
                        {
                            self.run_instrument_separation();
                        }

                        if self.separated_stems.is_some() {
                            ui.label(egui::RichText::new("Stem audio is loaded. Use the waveform tab controls to preview or enable instruments.").weak());
                        }
                    });
                }

                self.draw_main_content_tabs(ui);
                ui.add_space(UI_VSPACE_TIGHT);
                draw_horizontal_separator(ui, 0.0);
                ui.add_space(UI_VSPACE_TIGHT);

                // Absolute layout stability: calculate exact rects for content and footer
                let full_avail_h = ui.available_height().max(0.0);
                let full_avail_w = ui.available_width();
                let media_h = media_controls_height_for_width(full_avail_w);
                let gap = waveform_visual_gap;
                let footer_total_h = media_h + gap * 2.0;
                let content_h = (full_avail_h - footer_total_h).max(0.0);

                let start_pos = ui.cursor().min;
                let content_rect = egui::Rect::from_min_size(start_pos, egui::vec2(full_avail_w, content_h));
                let footer_rect = egui::Rect::from_min_size(
                    egui::pos2(start_pos.x, content_rect.bottom() + gap),
                    egui::vec2(full_avail_w, media_h)
                );

                // 1. Content Area (Strictly bounded)
                ui.allocate_ui_at_rect(content_rect, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    let mut remaining_h = content_h;
                    if self.main_content_tab != MainContentTab::SheetMusic {
                        if self.show_video_pane {
                            // The video follows the master audio clock (not
                            // selected_time_sec) so it stays locked to the
                            // audible audio, exactly like VLC's audio-master
                            // synchronization.
                            let audio_clock_sec = self
                                .master_clock
                                .map(|c| c.position_sec)
                                .unwrap_or(self.selected_time_sec);
                            let is_playing = self.is_playing();
                            if let Some(player) = &mut self.video_player {
                                let available_width = ui.available_width();
                                let max_video_h = (remaining_h - 100.0).max(50.0);
                                let video_h = self.video_panel_height.clamp(50.0, max_video_h);
                                
                                let (rect, _) = ui.allocate_exact_size(egui::vec2(available_width, video_h), egui::Sense::hover());
                                ui.allocate_ui_at_rect(rect, |ui| {
                                    player.draw(ui, audio_clock_sec, is_playing);
                                });
                                remaining_h -= video_h;
                                
                                let splitter_h = 8.0;
                                let (_, resp) = ui.allocate_exact_size(egui::vec2(available_width, splitter_h), egui::Sense::drag());
                                if resp.hovered() || resp.dragged() {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                                }
                                if resp.dragged() {
                                    self.video_panel_height += resp.drag_delta().y;
                                }
                                remaining_h -= splitter_h;

                                ui.add_space(waveform_visual_gap);
                                remaining_h -= waveform_visual_gap;
                            }
                        }
                    }

                    if self.main_content_tab == MainContentTab::SheetMusic {
                        self.draw_sheet_music_view(
                            ui,
                            interaction_ready,
                            interaction_duration,
                            default_stack_spacing_y,
                            waveform_visual_gap,
                            content_h,
                        );
                    } else {
                        let plot_resp = Plot::new("waveform_plot")
                            .height(remaining_h)
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

                                if let Some((a, b)) = self.loop_selection {
                                    let start = a.min(b);
                                    let end = a.max(b);
                                    self.refresh_loop_waveform_cache(start, end);

                                    if !self.loop_waveform_cache_pre.is_empty() {
                                        plot_ui.line(
                                            Line::new(PlotPoints::from_iter(
                                                self.loop_waveform_cache_pre.iter().copied(),
                                            ))
                                            .color(loop_wave_dim),
                                        );
                                    }
                                    if !self.loop_waveform_cache_mid.is_empty() {
                                        plot_ui.line(
                                            Line::new(PlotPoints::from_iter(
                                                self.loop_waveform_cache_mid.iter().copied(),
                                            ))
                                            .color(loop_wave_active),
                                        );
                                    }
                                    if !self.loop_waveform_cache_post.is_empty() {
                                        plot_ui.line(
                                            Line::new(PlotPoints::from_iter(
                                                self.loop_waveform_cache_post.iter().copied(),
                                            ))
                                            .color(loop_wave_dim),
                                        );
                                    }
                                } else {
                                    let line = Line::new(PlotPoints::from_iter(
                                        self.waveform.iter().copied(),
                                    ));
                                    plot_ui.line(line.color(self.highlight_color));
                                }

                                plot_ui.vline(
                                    VLine::new(self.selected_time_sec as f64)
                                        .color(accent_soft(self.highlight_color)),
                                );

                                let mut hovered_marker_idx = None;
                                if let Some(hash) = &self.loaded_audio_hash {
                                    if let Some(markers) = self.file_markers.get(hash) {
                                        for (i, mark_data) in markers.iter().enumerate() {
                                            let mark_sec = mark_data.time();
                                            let label = (b'A' + (i % 26) as u8) as char;
                                            let marker_color = if self.dark_mode { egui::Color32::WHITE } else { egui::Color32::BLACK };
                                            let bg_color = if self.dark_mode { egui::Color32::from_black_alpha(200) } else { egui::Color32::from_white_alpha(200) };
                                            plot_ui.vline(
                                                VLine::new(mark_sec as f64)
                                                    .color(marker_color)
                                            );
                                            plot_ui.text(
                                                egui_plot::Text::new(
                                                    egui_plot::PlotPoint::new(mark_sec as f64, 1.0),
                                                    egui::RichText::new(format!(" {label} "))
                                                        .size(16.0)
                                                        .color(marker_color)
                                                        .background_color(bg_color)
                                                )
                                                .anchor(egui::Align2::CENTER_TOP)
                                            );
                                        }

                                        let pointer = plot_ui.pointer_coordinate();
                                        if let Some(p) = pointer {
                                            if p.y > 0.4 {
                                                let x_span = plot_ui.plot_bounds().width();
                                                let mut min_dist = x_span * 0.02;
                                                for (i, mark) in markers.iter().enumerate() {
                                                    let dist = (p.x - mark.time() as f64).abs();
                                                    if dist < min_dist {
                                                        min_dist = dist;
                                                        hovered_marker_idx = Some(i);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if let Some((a, b)) = self.loop_selection {
                                    let start = a.min(b);
                                    let end = a.max(b);
                                    plot_ui.vline(VLine::new(start as f64).color(loop_edge));
                                    plot_ui.vline(VLine::new(end as f64).color(loop_edge));
                                }

                                // Keep Y scale fixed and clamp X so navigation stays within audio bounds.
                                let mut b = if self.waveform_reset_view {
                                    self.waveform_reset_view = false;
                                    PlotBounds::from_min_max(
                                        [0.0, -1.05],
                                        [plot_duration as f64, 1.05],
                                    )
                                } else {
                                    plot_ui.plot_bounds()
                                };

                                let pointer = plot_ui.pointer_coordinate();
                                let hovered = plot_ui.response().hovered();
                                let drag_started = plot_ui.response().drag_started_by(egui::PointerButton::Primary);
                                let dragged = plot_ui.response().dragged_by(egui::PointerButton::Primary);
                                let drag_stopped = plot_ui.response().drag_stopped_by(egui::PointerButton::Primary);
                                let clicked = plot_ui.response().clicked();
                                let (
                                    raw_scroll,
                                    smooth_scroll,
                                    shift_held,
                                    ctrl_held,
                                    zoom_delta,
                                    pointer_delta,
                                ) = plot_ui.ctx().input(|i| {
                                    (
                                        i.raw_scroll_delta,
                                        i.smooth_scroll_delta,
                                        i.modifiers.shift,
                                        i.modifiers.ctrl,
                                        i.zoom_delta_2d(),
                                        i.pointer.delta(),
                                    )
                                });
                                let touch_navigation = self.is_touch_platform();

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

                                    if touch_navigation
                                        && !self.touch_loop_select_mode
                                        && dragged
                                    {
                                        let drag_width = plot_ui.response().rect.width().max(1.0)
                                            as f64;
                                        let shift_amount =
                                            -(pointer_delta.x as f64) * (span / drag_width);
                                        b = PlotBounds::from_min_max(
                                            [b.min()[0] + shift_amount, b.min()[1]],
                                            [b.max()[0] + shift_amount, b.max()[1]],
                                        );
                                    } else if shift_held
                                        && (wheel_y.abs() > f32::EPSILON
                                            || wheel_x.abs() > f32::EPSILON)
                                    {
                                        let dominant_wheel = if wheel_x.abs() > wheel_y.abs() {
                                            wheel_x
                                        } else {
                                            wheel_y
                                        };
                                        let shift_amount =
                                            -(dominant_wheel as f64) * 0.0015 * span;
                                        b = PlotBounds::from_min_max(
                                            [b.min()[0] + shift_amount, b.min()[1]],
                                            [b.max()[0] + shift_amount, b.max()[1]],
                                        );
                                    }

                                    let touch_pinch = touch_navigation
                                        && (zoom_delta.y - 1.0).abs() > f32::EPSILON;
                                    if ctrl_held || touch_pinch {
                                        let zoom_from_wheel =
                                            if ctrl_held && wheel_y.abs() > f32::EPSILON {
                                                if wheel_y > 0.0 {
                                                    0.88
                                                } else {
                                                    1.14
                                                }
                                            } else {
                                                1.0
                                            };

                                        let zoom_from_input =
                                            if (zoom_delta.y - 1.0).abs() > f32::EPSILON {
                                                (1.0 / zoom_delta.y as f64).clamp(0.7, 1.4)
                                            } else {
                                                1.0
                                            };

                                        let zoom = zoom_from_wheel * zoom_from_input;

                                        if (zoom - 1.0).abs() > f64::EPSILON {
                                            let min_span =
                                                (plot_duration as f64 / 400.0).max(0.02);
                                            let max_span = plot_duration as f64;
                                            let new_span =
                                                (span * zoom).clamp(min_span, max_span);

                                            let center_x = pointer
                                                .map(|p| p.x)
                                                .unwrap_or((b.min()[0] + b.max()[0]) * 0.5)
                                                .clamp(0.0, plot_duration as f64);

                                            let left_ratio = ((center_x - b.min()[0]) / span)
                                                .clamp(0.0, 1.0);
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
                                let max_span = plot_duration as f64;
                                if x_span > max_span {
                                    x_span = max_span;
                                }

                                let min_x = if x_span >= max_span {
                                    0.0
                                } else {
                                    b.min()[0].clamp(0.0, max_span - x_span)
                                };
                                let max_x = (min_x + x_span).min(max_span);

                                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                                    [min_x, -1.05],
                                    [max_x, 1.05],
                                ));

                                let allow_loop_drag =
                                    !touch_navigation || self.touch_loop_select_mode;
                                if !allow_loop_drag {
                                    self.drag_select_anchor_sec = None;
                                }

                                if interaction_ready && allow_loop_drag && drag_started {
                                    self.drag_select_anchor_sec = pointer
                                        .map(|p| {
                                            p.x.clamp(0.0, interaction_duration as f64) as f32
                                        })
                                        .or(Some(self.selected_time_sec));
                                }

                                if interaction_ready && allow_loop_drag && dragged {
                                    if let (Some(anchor), Some(p)) = (
                                        self.drag_select_anchor_sec,
                                        pointer.map(|p| {
                                            p.x.clamp(0.0, interaction_duration as f64) as f32
                                        }),
                                    ) {
                                        self.loop_selection = Some((anchor, p));
                                    }
                                }

                                if interaction_ready && allow_loop_drag && drag_stopped {
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

                                if interaction_ready && clicked {
                                    if let Some(idx) = hovered_marker_idx {
                                        if let Some(hash) = &self.loaded_audio_hash {
                                            if let Some(markers) = self.file_markers.get(hash) {
                                                self.selected_time_sec = markers[idx].time().clamp(0.0, interaction_duration);
                                            }
                                        }
                                    } else if let Some(pointer) = pointer {
                                        self.selected_time_sec = pointer
                                            .x
                                            .clamp(0.0, interaction_duration as f64)
                                            as f32;
                                    }
                                    self.loop_selection = None;
                                    self.loop_playback_enabled = false;
                                    if self.is_playing() {
                                        self.play_from_selected();
                                    }
                                }

                                hovered_marker_idx
                            });

                        let hovered_marker_idx = plot_resp.inner;
                        let mut delete_mark_idx = None;
                        let mut add_mark_sec = None;

                        if plot_resp.response.secondary_clicked() {
                            self.context_menu_marker_idx = hovered_marker_idx;
                            if let Some(pos) = plot_resp.response.interact_pointer_pos() {
                                self.context_menu_plot_x = Some(plot_resp.transform.value_from_position(pos).x as f32);
                            } else {
                                self.context_menu_plot_x = None;
                            }
                        }

                        if let Some(hash) = &self.loaded_audio_hash {
                            plot_resp.response.context_menu(|ui| {
                                if let Some(idx) = self.context_menu_marker_idx {
                                    let label = (b'A' + (idx % 26) as u8) as char;
                                    ui.label(format!("Marker {label}"));
                                    ui.separator();
                                    ui.horizontal(|ui| {
                                        ui.label("Edit Time:");
                                        if self.marker_edit_index != Some(idx) {
                                            self.marker_edit_index = Some(idx);
                                            if let Some(markers) = self.file_markers.get(hash) {
                                                if idx < markers.len() {
                                                    let total_ms = (markers[idx].time() * 1000.0).round() as u32;
                                                    let ms = total_ms % 1000;
                                                    let s = (total_ms / 1000) % 60;
                                                    let m = (total_ms / 60000) % 60;
                                                    let h = total_ms / 3600000;
                                                    self.marker_edit_str = if h == 0 {
                                                        format!("{:02}:{:02}:{:03}", m, s, ms)
                                                    } else {
                                                        format!("{:02}:{:02}:{:02}:{:03}", h, m, s, ms)
                                                    };
                                                }
                                            }
                                        }
                                        
                                        if ui.add(egui::TextEdit::singleline(&mut self.marker_edit_str).desired_width(100.0)).changed() {
                                            let parts: Vec<&str> = self.marker_edit_str.split(':').collect();
                                            let parsed_val = if parts.len() == 4 {
                                                if let (Ok(h), Ok(m), Ok(s), Ok(ms)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>(), parts[2].parse::<f32>(), parts[3].parse::<f32>()) {
                                                    Some(h * 3600.0 + m * 60.0 + s + ms / 1000.0)
                                                } else { None }
                                            } else if parts.len() == 3 {
                                                if let (Ok(m), Ok(s), Ok(ms)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>(), parts[2].parse::<f32>()) {
                                                    Some(m * 60.0 + s + ms / 1000.0)
                                                } else { None }
                                            } else { None };

                                            if let Some(val) = parsed_val {
                                                if let Some(markers) = self.file_markers.get_mut(hash) {
                                                    if idx < markers.len() {
                                                        markers[idx].set_time(val.clamp(0.0, interaction_duration));
                                                    }
                                                }
                                            }
                                        }
                                    });

                                    ui.separator();
                                    
                                    if ui.add(egui::Button::new("Loop From Here").frame(false)).clicked() {
                                        if let Some(markers) = self.file_markers.get(hash) {
                                            if idx < markers.len() {
                                                let end = self.loop_selection.map(|(_, e)| e).unwrap_or(interaction_duration);
                                                self.loop_selection = Some((markers[idx].time(), end));
                                                self.loop_enabled = true;
                                                self.loop_playback_enabled = true;
                                                ui.close_menu();
                                            }
                                        }
                                    }
                                    if ui.add(egui::Button::new("Loop To Here").frame(false)).clicked() {
                                        if let Some(markers) = self.file_markers.get(hash) {
                                            if idx < markers.len() {
                                                let start = self.loop_selection.map(|(s, _)| s).unwrap_or(0.0);
                                                self.loop_selection = Some((start, markers[idx].time()));
                                                self.loop_enabled = true;
                                                self.loop_playback_enabled = true;
                                                ui.close_menu();
                                            }
                                        }
                                    }
                                    if ui.add(egui::Button::new(egui::RichText::new("Delete Mark").color(crate::theme::ERROR_RED)).frame(false)).clicked() {
                                        delete_mark_idx = Some(idx);
                                        ui.close_menu();
                                    }

                                    ui.separator();
                                    ui.label("Description:");
                                    let mut desc = String::new();
                                    if let Some(markers) = self.file_markers.get(hash) {
                                        if idx < markers.len() {
                                            desc = markers[idx].desc().to_string();
                                        }
                                    }
                                    egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                                        if ui.add(egui::TextEdit::multiline(&mut desc).desired_width(200.0).desired_rows(5)).changed() {
                                            if let Some(markers) = self.file_markers.get_mut(hash) {
                                                if idx < markers.len() {
                                                    markers[idx].set_desc(desc);
                                                }
                                            }
                                        }
                                    });
                                } else {
                                    if ui.button("Add Marker Here").clicked() {
                                        if let Some(x) = self.context_menu_plot_x {
                                            add_mark_sec = Some(x);
                                        }
                                        ui.close_menu();
                                    }
                                }
                            });

                            if plot_resp.response.drag_started_by(egui::PointerButton::Secondary) {
                                self.dragging_marker = hovered_marker_idx;
                            }
                            
                            if plot_resp.response.dragged_by(egui::PointerButton::Secondary) {
                                if let Some(idx) = self.dragging_marker {
                                    if let Some(p) = plot_resp.response.interact_pointer_pos() {
                                        let plot_p = plot_resp.transform.value_from_position(p);
                                        if let Some(markers) = self.file_markers.get_mut(hash) {
                                            markers[idx].set_time((plot_p.x as f32).clamp(0.0, interaction_duration));
                                        }
                                    }
                                }
                            }
                            
                            if plot_resp.response.drag_stopped_by(egui::PointerButton::Secondary) {
                                self.dragging_marker = None;
                                if let Some(markers) = self.file_markers.get_mut(hash) {
                                    markers.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
                                }
                            }

                            if let Some(idx) = delete_mark_idx {
                                if let Some(markers) = self.file_markers.get_mut(hash) {
                                    markers.remove(idx);
                                }
                            }

                            if let Some(sec) = add_mark_sec {
                                let markers = self.file_markers.entry(hash.clone()).or_default();
                                markers.push(crate::app::MarkerData::Detailed { time: sec, desc: String::new() });
                                markers.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
                            }
                        }
                    }
                });

                // 2. Media Footer (Pinned to bottom)
                ui.allocate_ui_at_rect(footer_rect, |ui| {
                    ui.scope(|ui| {
                        ui.spacing_mut().item_spacing.y = default_stack_spacing_y;
                        draw_media_controls(self, ui, interaction_ready, interaction_duration);
                    });
                });

                // Finish layout by advancing cursor past footer
                ui.advance_cursor_after_rect(egui::Rect::from_min_max(
                    start_pos,
                    egui::pos2(start_pos.x + full_avail_w, footer_rect.bottom() + gap)
                ));
                });        self.waveform_panel_height = waveform_central.response.rect.height().clamp(120.0, 5000.0);

        #[cfg(feature = "desktop-ui")]
        if hovered_valid_drop.is_some() && self.audio_raw.is_some() {
            let screen_rect = ctx.input(|i| i.screen_rect());
            let overlay_layer = egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("audio_file_drag_drop_overlay"),
            );
            let painter = ctx.layer_painter(overlay_layer);

            paint_audio_import_overlay(
                &painter,
                screen_rect,
                self.highlight_color,
                228,
                "Drop Audio/Video File To Import",
                "Audio: wav, mp3, flac, ogg, m4a, aac | Video: mp4, mkv, avi, mov, webm",
            );
        }

        // Keep high refresh only when motion or background work is active.
        let pointer_active = ctx.input(|i| i.pointer.any_down());
        let needs_fast_repaint = self.is_playing()
            || self.is_audio_loading
            || self.is_processing
            || self.pending_param_change
            || pointer_active;
        ctx.request_repaint_after(if needs_fast_repaint {
            ACTIVE_REPAINT_INTERVAL
        } else {
            IDLE_REPAINT_INTERVAL
        });

        self.draw_export_modals(ctx);

        if self.last_state_save_at.elapsed() >= Duration::from_secs(2) {
            self.save_state_to_disk();
            self.last_state_save_at = Instant::now();
        }
    }
}

impl Drop for KeyScribeApp {
    fn drop(&mut self) {
        self.save_state_to_disk();
        self.cancel_audio_loading();
        self.cancel_active_processing();
        self.stop();

        // Release large allocations eagerly so the OS can reclaim memory faster.
        self.processed_samples = Vec::new();
        self.processed_playback_samples = Arc::new(Vec::new());
        self.waveform = Vec::new();
        self.audio_raw = None;
        self.video_player = None;
        self.engine = None;
    }
}

fn paint_audio_import_overlay(
    painter: &egui::Painter,
    overlay_rect: egui::Rect,
    highlight_color: egui::Color32,
    backdrop_alpha: u8,
    title: &str,
    subtitle: &str,
) {
    painter.rect_filled(
        overlay_rect,
        0.0,
        egui::Color32::from_rgba_unmultiplied(6, 10, 16, backdrop_alpha),
    );

    let border_color = egui::Color32::from_rgb(
        highlight_color.r().saturating_add(20),
        highlight_color.g().saturating_add(20),
        highlight_color.b().saturating_add(20),
    );
    let frame_rect = overlay_rect.shrink(22.0);
    painter.rect_stroke(frame_rect, 12.0, egui::Stroke::new(2.0, border_color));

    painter.text(
        frame_rect.center() + egui::vec2(0.0, -10.0),
        egui::Align2::CENTER_CENTER,
        title,
        egui::FontId::proportional(26.0),
        egui::Color32::WHITE,
    );
    painter.text(
        frame_rect.center() + egui::vec2(0.0, 22.0),
        egui::Align2::CENTER_CENTER,
        subtitle,
        egui::FontId::proportional(16.0),
        egui::Color32::from_gray(216),
    );
}
