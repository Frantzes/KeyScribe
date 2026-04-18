use super::*;

impl eframe::App for KeyScribeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_brand_theme(ctx, self.dark_mode, self.highlight_color);
        self.lock_startup_min_window_size_once(ctx);
        self.apply_mobile_ui_tweaks_once(ctx);

        let (space_pressed, k_pressed, left_pressed, right_pressed, ctrl_held) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::K),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.modifiers.ctrl,
            )
        });

        if space_pressed {
            self.handle_space_replay();
        }
        if k_pressed {
            self.handle_toggle_play_pause();
        }
        if left_pressed {
            if ctrl_held && self.shift_loop_by_seconds(-SEEK_STEP_SEC) {
                // Ctrl+Arrow shifts loop range when looping is active.
            } else {
                self.skip_by_seconds(-SEEK_STEP_SEC);
            }
        }
        if right_pressed {
            if ctrl_held && self.shift_loop_by_seconds(SEEK_STEP_SEC) {
                // Ctrl+Arrow shifts loop range when looping is active.
            } else {
                self.skip_by_seconds(SEEK_STEP_SEC);
            }
        }

        self.poll_audio_loading(ctx);
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

            let note_visuals_ready = self.note_visuals_ready();
            if !note_visuals_ready {
                self.clear_note_visuals();
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

            let prob_strip_height = if self.show_note_hist_window && note_visuals_ready {
                (key_h_for_frame * 0.9).clamp(MIN_PROBABILITY_STRIP_HEIGHT, 120.0)
            } else {
                0.0
            };

            let keyboard_stack_h = key_h_for_frame
                + if self.show_note_hist_window && note_visuals_ready {
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
            if self.show_note_hist_window && note_visuals_ready {
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
                        self.piano_zoom =
                            (self.piano_zoom * zoom_delta.y).clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX);
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
            let plot_duration = self.waveform_view_duration().max(source_duration).max(0.01);
            let interaction_duration = if self.is_audio_loading
                && (self.loading_cache_waveform_preloaded || self.loading_cache_timeline_preloaded)
            {
                plot_duration
            } else {
                source_duration
            };
            let waveform_height = (ui.available_height() - 112.0).max(40.0);
            let analysis_ready =
                !self.is_blocking_processing() && !self.processed_samples.is_empty();
            let interaction_ready = analysis_ready
                || (self.is_audio_loading
                    && (self.loading_cache_waveform_preloaded
                        || self.loading_cache_timeline_preloaded));

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
                        PlotBounds::from_min_max([0.0, -1.05], [plot_duration as f64, 1.05])
                    } else {
                        plot_ui.plot_bounds()
                    };

                    let pointer = plot_ui.pointer_coordinate();
                    let hovered = plot_ui.response().hovered();
                    let drag_started = plot_ui.response().drag_started();
                    let dragged = plot_ui.response().dragged();
                    let drag_stopped = plot_ui.response().drag_stopped();
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

                        if touch_navigation && !self.touch_loop_select_mode && dragged {
                            let drag_width = plot_ui.response().rect.width().max(1.0) as f64;
                            let shift_amount = -(pointer_delta.x as f64) * (span / drag_width);
                            b = PlotBounds::from_min_max(
                                [b.min()[0] + shift_amount, b.min()[1]],
                                [b.max()[0] + shift_amount, b.max()[1]],
                            );
                        } else if shift_held
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
                        }

                        let touch_pinch =
                            touch_navigation && (zoom_delta.y - 1.0).abs() > f32::EPSILON;
                        if ctrl_held || touch_pinch {
                            let zoom_from_wheel = if ctrl_held && wheel_y.abs() > f32::EPSILON {
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
                                let min_span = (plot_duration as f64 / 400.0).max(0.02);
                                let max_span = plot_duration as f64;
                                let new_span = (span * zoom).clamp(min_span, max_span);

                                let center_x = pointer
                                    .map(|p| p.x)
                                    .unwrap_or((b.min()[0] + b.max()[0]) * 0.5)
                                    .clamp(0.0, plot_duration as f64);

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

                    plot_ui
                        .set_plot_bounds(PlotBounds::from_min_max([min_x, -1.05], [max_x, 1.05]));

                    let allow_loop_drag = !touch_navigation || self.touch_loop_select_mode;
                    if !allow_loop_drag {
                        self.drag_select_anchor_sec = None;
                    }

                    if interaction_ready && allow_loop_drag && drag_started {
                        self.drag_select_anchor_sec = pointer
                            .map(|p| p.x.clamp(0.0, interaction_duration as f64) as f32)
                            .or(Some(self.selected_time_sec));
                    }

                    if interaction_ready && allow_loop_drag && dragged {
                        if let (Some(anchor), Some(p)) = (
                            self.drag_select_anchor_sec,
                            pointer.map(|p| p.x.clamp(0.0, interaction_duration as f64) as f32),
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
                        if let Some(pointer) = pointer {
                            self.selected_time_sec =
                                pointer.x.clamp(0.0, interaction_duration as f64) as f32;
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
                draw_media_controls(self, ui, interaction_ready, interaction_duration);
            });
        });
        self.waveform_panel_height = waveform_central.response.rect.height().clamp(120.0, 5000.0);

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

        if self.last_state_save_at.elapsed() >= Duration::from_secs(2) {
            self.save_state_to_disk();
            self.last_state_save_at = Instant::now();
        }
    }
}

impl Drop for KeyScribeApp {
    fn drop(&mut self) {
        self.save_state_to_disk();
    }
}
