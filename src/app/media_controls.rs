use eframe::egui;
use egui_phosphor::regular::{
    FAST_FORWARD, PAUSE, PLAY, REPEAT, REWIND, SPEAKER_HIGH, SPEAKER_NONE,
};

use super::{KeyScribeApp, SEEK_STEP_SEC, UI_VSPACE_COMPACT, UI_VSPACE_MEDIUM};
use crate::theme::{
    MEDIA_PANEL_BG_DARK, MEDIA_PANEL_BG_LIGHT, SLIDER_RAIL_BG_ACTIVE_DARK,
    SLIDER_RAIL_BG_ACTIVE_LIGHT, SLIDER_RAIL_BG_DARK, SLIDER_RAIL_BG_HOVER_DARK,
    SLIDER_RAIL_BG_HOVER_LIGHT, SLIDER_RAIL_BG_LIGHT,
};
use crate::ui::utils::format_time;
use crate::ui::widgets::{
    icon_button, icon_font_id, icon_toggle_button, responsive_icon_button_size,
};

const VOLUME_SLIDER_MAX_WIDTH: f32 = 280.0;

fn channel_label(channels: u16) -> String {
    match channels.max(1) {
        1 => "Mono".to_string(),
        2 => "Stereo".to_string(),
        n => format!("{n}ch"),
    }
}

pub(super) fn media_controls_height_for_width(width: f32) -> f32 {
    if width < 560.0 {
        178.0
    } else if width < 820.0 {
        154.0
    } else {
        98.0
    }
}

fn draw_album_art(ui: &mut egui::Ui, texture: Option<&egui::TextureHandle>, art_size: f32) {
    if let Some(texture) = texture {
        ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(art_size, art_size)));
    } else {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(art_size, art_size), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(38, 49, 63));
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            PLAY,
            icon_font_id((art_size * 0.28).clamp(16.0, 22.0)),
            egui::Color32::from_rgb(177, 192, 210),
        );
    }
}

fn draw_track_meta(
    ui: &mut egui::Ui,
    title: &str,
    artist: &str,
    album: &str,
    channel_status: &str,
    art_size: f32,
    compact: bool,
) {
    ui.vertical(|ui| {
        let title_size = if compact { 15.0 } else { 17.0 };
        let title_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(title_size)));
        let artist_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(14.0)));
        let format_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(12.0)));
        let block_h = title_h + artist_h + format_h + ui.spacing().item_spacing.y * 2.0;
        let available_h = ui.available_height().max(art_size);
        let top_pad = ((available_h - block_h) * 0.5).max(0.0);
        if top_pad > 0.0 {
            ui.add_space(top_pad);
        }

        ui.add(egui::Label::new(egui::RichText::new(title).size(title_size)).wrap(true));
        let secondary = if album.is_empty() {
            artist.to_string()
        } else {
            format!("{artist} · {album}")
        };
        ui.add(
            egui::Label::new(
                egui::RichText::new(secondary).color(egui::Color32::from_rgb(166, 182, 202)),
            )
            .wrap(true),
        );
        ui.add(
            egui::Label::new(
                egui::RichText::new(channel_status)
                    .size(12.0)
                    .color(egui::Color32::from_rgb(145, 160, 182)),
            )
            .wrap(true),
        );
    });
}

fn draw_volume_time_row(
    ui: &mut egui::Ui,
    app: &mut KeyScribeApp,
    time_label: &str,
    slider_height: f32,
    icon_size: f32,
    min_slider_width: f32,
) {
    let time_color = egui::Color32::from_rgb(176, 188, 203);
    let time_font = egui::TextStyle::Body.resolve(ui.style());
    let (time_width, time_height) = ui.fonts(|f| {
        let size = f
            .layout_no_wrap(time_label.to_owned(), time_font.clone(), time_color)
            .size();
        (size.x, size.y)
    });
    let row_h = slider_height.max(time_height).max(icon_size + 2.0);

    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), row_h),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            let spacing_x = ui.spacing().item_spacing.x;
            let icon_slot_w = (icon_size + 8.0).max(18.0);

            let vol_icon = if app.playback_volume <= 0.01 {
                SPEAKER_NONE
            } else {
                SPEAKER_HIGH
            };
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(icon_slot_w, row_h), egui::Sense::hover());
            ui.painter().text(
                icon_rect.center(),
                egui::Align2::CENTER_CENTER,
                vol_icon,
                icon_font_id(icon_size),
                ui.visuals().text_color(),
            );

            let max_slider_width = (ui.available_width() - (time_width + spacing_x))
                .max(0.0)
                .min(VOLUME_SLIDER_MAX_WIDTH);
            let volume_width = if max_slider_width >= min_slider_width {
                max_slider_width
            } else {
                max_slider_width.max(72.0)
            };

            ui.spacing_mut().slider_width = volume_width;
            let (rail_fill, rail_fill_hover, rail_fill_active) = if app.dark_mode {
                (
                    SLIDER_RAIL_BG_DARK,
                    SLIDER_RAIL_BG_HOVER_DARK,
                    SLIDER_RAIL_BG_ACTIVE_DARK,
                )
            } else {
                (
                    SLIDER_RAIL_BG_LIGHT,
                    SLIDER_RAIL_BG_HOVER_LIGHT,
                    SLIDER_RAIL_BG_ACTIVE_LIGHT,
                )
            };
            let vol_changed = ui
                .scope(|ui| {
                    let visuals = ui.visuals_mut();
                    visuals.slider_trailing_fill = true;
                    visuals.widgets.inactive.bg_fill = rail_fill;
                    visuals.widgets.hovered.bg_fill = rail_fill_hover;
                    visuals.widgets.active.bg_fill = rail_fill_active;
                    visuals.widgets.inactive.weak_bg_fill = rail_fill;
                    visuals.widgets.hovered.weak_bg_fill = rail_fill_hover;
                    visuals.widgets.active.weak_bg_fill = rail_fill_active;

                    ui.add_sized(
                        [volume_width, row_h],
                        egui::Slider::new(&mut app.playback_volume, 0.0..=1.5).show_value(false),
                    )
                    .changed()
                })
                .inner;
            if vol_changed {
                if let Some(engine) = &mut app.engine {
                    engine.set_volume(app.playback_volume);
                }
            }

            let (time_rect, _) =
                ui.allocate_exact_size(egui::vec2(time_width, row_h), egui::Sense::hover());
            ui.painter().text(
                time_rect.center(),
                egui::Align2::CENTER_CENTER,
                time_label,
                time_font,
                time_color,
            );
        },
    );
}

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
    app: &mut KeyScribeApp,
    ui: &mut egui::Ui,
    analysis_ready: bool,
    duration: f32,
) {
    let full_w = ui.available_width();
    let compact_layout = full_w < 820.0;
    let art_size = if compact_layout {
        (full_w * 0.09).clamp(48.0, 64.0)
    } else {
        72.0
    };
    let preferred_h = media_controls_height_for_width(full_w);
    let target_h = ui.available_height().max(0.0).min(preferred_h);
    if target_h <= f32::EPSILON {
        return;
    }
    let button_size = responsive_icon_button_size(ui);

    let panel_fill = if app.dark_mode {
        MEDIA_PANEL_BG_DARK
    } else {
        MEDIA_PANEL_BG_LIGHT
    };

    let inner_w = (full_w - if compact_layout { 20.0 } else { 28.0 }).max(0.0);

    let fallback_name = app
        .loaded_path
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string();

    let title = app
        .audio_raw
        .as_ref()
        .and_then(|a| a.metadata.title.as_deref())
        .unwrap_or(fallback_name.as_str())
        .to_string();

    let artist = app
        .audio_raw
        .as_ref()
        .and_then(|a| a.metadata.artist.as_deref())
        .unwrap_or("Unknown Artist")
        .to_string();

    let album = app
        .audio_raw
        .as_ref()
        .and_then(|a| a.metadata.album.as_deref())
        .unwrap_or("")
        .to_string();

    let source_channels = app
        .audio_raw
        .as_ref()
        .map(|audio| audio.channels)
        .unwrap_or(app.loading_source_channels)
        .max(1);
    let playback_channels = if app.processed_playback_samples.is_empty() {
        if source_channels <= 1 {
            1
        } else {
            2
        }
    } else {
        app.processed_playback_channels.max(1)
    };
    let channel_status = format!(
        "Source: {} | Playback: {}",
        channel_label(source_channels),
        channel_label(playback_channels)
    );
    let time_label = format!(
        "{} / {}",
        format_time(app.selected_time_sec),
        format_time(duration)
    );

    ui.allocate_ui_with_layout(
        egui::vec2(full_w, target_h),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            ui.set_min_height(target_h);
            let clip_rect = ui.max_rect();
            ui.set_clip_rect(clip_rect);
            egui::Frame::none()
                .fill(panel_fill)
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(if compact_layout {
                    egui::Margin::symmetric(10.0, UI_VSPACE_MEDIUM)
                } else {
                    egui::Margin::symmetric(14.0, UI_VSPACE_MEDIUM)
                })
                .show(ui, |ui| {
                    // Force frame width to match the parent width so centering is stable.
                    ui.set_min_width(inner_w);
                    ui.set_max_width(inner_w);
                    let content_h = ui.available_height().max(0.0);

                    if compact_layout {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                draw_album_art(ui, app.album_art_texture.as_ref(), art_size);
                                ui.add_space(8.0);

                                let metadata_width = (ui.available_width() - 4.0).max(0.0);
                                ui.allocate_ui_with_layout(
                                    egui::vec2(metadata_width, art_size),
                                    egui::Layout::top_down(egui::Align::Min),
                                    |ui| {
                                        draw_track_meta(
                                            ui,
                                            title.as_str(),
                                            artist.as_str(),
                                            album.as_str(),
                                            channel_status.as_str(),
                                            art_size,
                                            true,
                                        );
                                    },
                                );
                            });

                            ui.add_space(UI_VSPACE_MEDIUM);
                            ui.horizontal_wrapped(|ui| {
                                if icon_button(ui, REWIND, "Skip Back 5s", analysis_ready).clicked()
                                {
                                    app.skip_by_seconds(-SEEK_STEP_SEC);
                                }

                                let is_playing = app.is_playing();
                                let play_icon = if is_playing { PAUSE } else { PLAY };
                                if icon_button(ui, play_icon, "Play / Pause", analysis_ready)
                                    .clicked()
                                {
                                    let current_pos = app.current_position_sec();

                                    if is_playing {
                                        app.stop();
                                    } else if app.audio_raw.is_some() {
                                        if app.processed_playback_samples.is_empty() {
                                            app.request_rebuild(false, super::RebuildMode::Full);
                                        }

                                        if current_pos <= 0.0 || current_pos >= duration - 0.01 {
                                            app.play_from_selected();
                                        } else if let Some(engine) = &mut app.engine {
                                            engine.resume();
                                        }
                                    }
                                }

                                if icon_button(ui, FAST_FORWARD, "Skip Forward 5s", analysis_ready)
                                    .clicked()
                                {
                                    app.skip_by_seconds(SEEK_STEP_SEC);
                                }

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
                            });

                            ui.add_space(UI_VSPACE_COMPACT);
                            draw_volume_time_row(
                                ui,
                                app,
                                time_label.as_str(),
                                (button_size * 0.62).clamp(18.0, 24.0),
                                16.0,
                                96.0,
                            );
                        });
                    } else {
                        ui.columns(3, |cols| {
                            cols[0].set_height(content_h);
                            cols[0].with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    draw_album_art(ui, app.album_art_texture.as_ref(), art_size);
                                    ui.add_space(8.0);

                                    let metadata_width = (ui.available_width() - 6.0).max(0.0);
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(metadata_width, content_h),
                                        egui::Layout::top_down(egui::Align::Min),
                                        |ui| {
                                            draw_track_meta(
                                                ui,
                                                title.as_str(),
                                                artist.as_str(),
                                                album.as_str(),
                                                channel_status.as_str(),
                                                art_size,
                                                false,
                                            );
                                        },
                                    );
                                },
                            );

                            cols[1].set_height(content_h);
                            cols[1].allocate_ui_with_layout(
                                egui::vec2(cols[1].available_width(), content_h),
                                egui::Layout::top_down(egui::Align::Center),
                                |ui| {
                                    let play_w = button_size;
                                    let side_w = ((ui.available_width() - play_w).max(0.0)) * 0.5;

                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 10.0;

                                        ui.allocate_ui_with_layout(
                                            egui::vec2(side_w, content_h),
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
                                        if icon_button(
                                            ui,
                                            play_icon,
                                            "Play / Pause",
                                            analysis_ready,
                                        )
                                        .clicked()
                                        {
                                            let current_pos = app.current_position_sec();

                                            if is_playing {
                                                app.stop();
                                            } else if app.audio_raw.is_some() {
                                                if app.processed_playback_samples.is_empty() {
                                                    app.request_rebuild(
                                                        false,
                                                        super::RebuildMode::Full,
                                                    );
                                                }

                                                if current_pos <= 0.0
                                                    || current_pos >= duration - 0.01
                                                {
                                                    app.play_from_selected();
                                                } else if let Some(engine) = &mut app.engine {
                                                    engine.resume();
                                                }
                                            }
                                        }

                                        ui.allocate_ui_with_layout(
                                            egui::vec2(side_w, content_h),
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

                            cols[2].set_height(content_h);
                            cols[2].allocate_ui_with_layout(
                                egui::vec2(cols[2].available_width(), content_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    let slider_height = (button_size * 0.62).clamp(18.0, 24.0);
                                    draw_volume_time_row(
                                        ui,
                                        app,
                                        time_label.as_str(),
                                        slider_height,
                                        17.0,
                                        120.0,
                                    );
                                },
                            );
                        });
                    }
                });
        },
    );
}
