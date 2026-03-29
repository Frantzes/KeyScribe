use eframe::egui;
use egui_phosphor::regular::{
    FAST_FORWARD, PAUSE, PLAY, REPEAT, REWIND, SPEAKER_HIGH, SPEAKER_NONE,
};

use super::{TranscriberApp, SEEK_STEP_SEC};
use crate::ui::utils::format_time;
use crate::ui::widgets::{icon_button, icon_font_id, icon_toggle_button};

pub(super) fn setting_toggle_row(ui: &mut egui::Ui, value: &mut bool, label: &str) -> bool {
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

pub(super) fn draw_media_controls(
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

    let target_h = art_size + 24.0;

    let full_w = ui.available_width();
    let inner_w = (full_w - 28.0).max(0.0);

    ui.allocate_ui_with_layout(
        egui::vec2(full_w, target_h),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            egui::Frame::none()
                .fill(panel_fill)
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                .show(ui, |ui| {
                    // Force frame width to match the parent width so centering is stable.
                    ui.set_min_width(inner_w);
                    ui.set_max_width(inner_w);

                    ui.columns(3, |cols| {
                        cols[0].set_height(target_h);
                        cols[0].with_layout(
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
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
                                    painter.rect_filled(
                                        rect,
                                        6.0,
                                        egui::Color32::from_rgb(38, 49, 63),
                                    );
                                    painter.text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        PLAY,
                                        icon_font_id(20.0),
                                        egui::Color32::from_rgb(177, 192, 210),
                                    );
                                }

                                ui.add_space(8.0);

                                ui.vertical(|ui| {
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
                                });
                            },
                        );

                        cols[1].set_height(target_h);
                        cols[1].allocate_ui_with_layout(
                            egui::vec2(cols[1].available_width(), target_h),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                let play_w = 34.0_f32;
                                let side_w = ((ui.available_width() - play_w).max(0.0)) * 0.5;

                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 10.0;

                                    ui.allocate_ui_with_layout(
                                        egui::vec2(side_w, target_h),
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if icon_button(
                                                ui,
                                                REWIND,
                                                "Skip Back 5s",
                                                analysis_ready,
                                            )
                                            .clicked()
                                            {
                                                app.skip_by_seconds(-SEEK_STEP_SEC);
                                            }
                                        },
                                    );

                                    let is_playing = app.is_playing();
                                    let play_icon = if is_playing { PAUSE } else { PLAY };
                                    if icon_button(ui, play_icon, "Play / Pause", analysis_ready)
                                        .clicked()
                                    {
                                        let current_pos = app.current_position_sec();

                                        if is_playing {
                                            app.stop();
                                        } else if app.audio_raw.is_some() {
                                            if app.processed_samples.is_empty() {
                                                app.request_rebuild(false);
                                            }

                                            if current_pos <= 0.0 || current_pos >= duration - 0.01
                                            {
                                                app.play_from_selected();
                                            } else if let Some(engine) = &mut app.engine {
                                                engine.resume();
                                            }
                                        }
                                    }

                                    ui.allocate_ui_with_layout(
                                        egui::vec2(side_w, target_h),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            if icon_button(
                                                ui,
                                                FAST_FORWARD,
                                                "Skip Forward 5s",
                                                analysis_ready,
                                            )
                                            .clicked()
                                            {
                                                app.skip_by_seconds(SEEK_STEP_SEC);
                                            }

                                            ui.add_space(14.0);

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
                                                    app.loop_selection = None;
                                                    app.loop_playback_enabled = false;
                                                }
                                            }
                                        },
                                    );
                                });
                            },
                        );

                        cols[2].set_height(target_h);
                        cols[2].with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} / {}",
                                        format_time(app.selected_time_sec),
                                        format_time(duration)
                                    ))
                                    .color(egui::Color32::from_rgb(176, 188, 203)),
                                );

                                ui.add_space(8.0);

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

                                let vol_icon = if app.playback_volume <= 0.01 {
                                    SPEAKER_NONE
                                } else {
                                    SPEAKER_HIGH
                                };
                                ui.label(egui::RichText::new(vol_icon).font(icon_font_id(17.0)));
                            },
                        );
                    });
                });
        },
    );
}
