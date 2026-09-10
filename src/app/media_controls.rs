use eframe::egui;
use egui_phosphor::regular::{
    FAST_FORWARD, PAUSE, PLAY, REPEAT, REWIND, SPEAKER_HIGH, SPEAKER_NONE, MINUS, PLUS,
};

use super::{
    KeyScribeApp, MixSnapshot, SEEK_STEP_SEC, UI_VSPACE_COMPACT, UI_VSPACE_MEDIUM,
};
use crate::theme::{
    MEDIA_PANEL_BG_DARK, MEDIA_PANEL_BG_LIGHT, SLIDER_RAIL_BG_ACTIVE_DARK,
    SLIDER_RAIL_BG_ACTIVE_LIGHT, SLIDER_RAIL_BG_DARK, SLIDER_RAIL_BG_HOVER_DARK,
    SLIDER_RAIL_BG_HOVER_LIGHT, SLIDER_RAIL_BG_LIGHT,
};
use crate::ui::utils::format_time;
use crate::ui::widgets::{
    icon_button, icon_font_id, icon_toggle_button, responsive_icon_button_size,
};


fn channel_label(channels: u16) -> String {
    match channels.max(1) {
        1 => "Mono".to_string(),
        2 => "Stereo".to_string(),
        n => format!("{n}ch"),
    }
}

/// Reserved footer height for a given panel width.
///
/// The loop start/end fields live in a popover under the loop toggle, so the
/// footer height no longer depends on the loop state. These are ceilings the
/// layout is designed to fit; if the window gives the footer less room than
/// this (very short windows), the panel falls back to a scrollable stacked
/// layout so every control stays reachable.
/// Widths below this switch the media panel to the compact (stacked) layout.
pub(super) const MEDIA_COMPACT_MAX_W: f32 = 700.0;

pub(super) fn media_controls_height_for_width(width: f32) -> f32 {
    if width < 560.0 {
        224.0
    } else if width < MEDIA_COMPACT_MAX_W {
        // Album-art row plus one shared transport/volume row plus seek bar.
        154.0
    } else {
        120.0
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
        let mut title_size = if compact { 15.0 } else { 17.0 };
        if title.len() > 30 {
            title_size = (title_size * (30.0 / title.len() as f32).max(0.6)).max(12.0);
        }
        
        let title_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(title_size)));
        let artist_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(14.0)));
        let format_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(12.0)));
        let block_h = title_h + artist_h + format_h + ui.spacing().item_spacing.y * 2.0;
        let available_h = ui.available_height().max(art_size);
        let top_pad = ((available_h - block_h) * 0.5).max(0.0);
        if top_pad > 0.0 {
            ui.add_space(top_pad);
        }

        ui.add(egui::Label::new(egui::RichText::new(title).size(title_size)).truncate(true));
        let secondary = if album.is_empty() {
            artist.to_string()
        } else {
            format!("{artist} · {album}")
        };
        ui.add(
            egui::Label::new(
                egui::RichText::new(secondary).color(egui::Color32::from_rgb(166, 182, 202)),
            )
            .truncate(true),
        );
        ui.add(
            egui::Label::new(
                egui::RichText::new(channel_status)
                    .size(12.0)
                    .color(egui::Color32::from_rgb(145, 160, 182)),
            )
            .truncate(true),
        );
    });
}

/// Inline horizontal volume control for the desktop (wide) layout: speaker
/// icon pinned to the row's right edge with the slider filling the width up
/// to `max_slider_w`.
fn draw_volume_slider_row(ui: &mut egui::Ui, app: &mut KeyScribeApp, max_slider_w: f32) {
    let icon_size: f32 = 17.0;
    let icon_slot_w = (icon_size + 8.0).max(20.0);
    let row_h = (icon_size + 12.0).max(22.0);

    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), row_h),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            ui.spacing_mut().item_spacing.x = 8.0;

            // Right_to_left places the first allocation at the right edge, so
            // the slider is allocated first and the icon ends up on its left.
            // Captured before the slider below mutates app in place.
            let pre_vol = app.playback_volume;
            let slider_w = ui.available_width().min(max_slider_w).max(60.0);
            ui.spacing_mut().slider_width = slider_w;
            let (rail_fill, rail_hover, rail_active) = if app.dark_mode {
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
                    visuals.widgets.hovered.bg_fill = rail_hover;
                    visuals.widgets.active.bg_fill = rail_active;
                    visuals.widgets.inactive.weak_bg_fill = rail_fill;
                    visuals.widgets.hovered.weak_bg_fill = rail_hover;
                    visuals.widgets.active.weak_bg_fill = rail_active;

                    ui.add_sized(
                        [slider_w, row_h],
                        egui::Slider::new(&mut app.playback_volume, 0.0..=1.5).show_value(false),
                    )
                    .changed()
                })
                .inner;
            if vol_changed {
                let pd = ui.input(|i| i.pointer.primary_down());
                let mut snap = MixSnapshot::capture(app);
                snap.playback_volume = pre_vol;
                app.push_mix_undo_with(pd, snap);
                if let Some(engine) = &mut app.engine {
                    engine.set_volume(app.playback_volume);
                }
            }

            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(icon_slot_w, row_h), egui::Sense::hover());
            let vol_icon = if app.playback_volume <= 0.01 {
                SPEAKER_NONE
            } else {
                SPEAKER_HIGH
            };
            ui.painter().text(
                icon_rect.center(),
                egui::Align2::CENTER_CENTER,
                vol_icon,
                icon_font_id(icon_size),
                ui.visuals().text_color(),
            );
        },
    );
}

/// Speaker button that opens a vertical volume slider popup above it
/// (Spotify-style popover). Replaces the always-visible inline slider.
fn draw_volume_button(ui: &mut egui::Ui, app: &mut KeyScribeApp, icon_size: f32) -> egui::Response {
    let size = (icon_size + 6.0).max(20.0);
    let id = ui.make_persistent_id("media_volume_popup");
    let vol_icon = if app.playback_volume <= 0.01 {
        SPEAKER_NONE
    } else {
        SPEAKER_HIGH
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let bg = if resp.hovered() || resp.is_pointer_button_down_on() {
        ui.visuals().widgets.hovered.bg_fill
    } else {
        egui::Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 4.0, bg);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        vol_icon,
        icon_font_id(icon_size),
        ui.visuals().text_color(),
    );
    let resp = resp.on_hover_text("Volume");
    if resp.clicked() {
        ui.memory_mut(|mem| mem.toggle_popup(id));
    }
    if ui.memory(|mem| mem.is_popup_open(id)) {
        // Centered above the button: egui's popup_above_or_below_widget
        // anchors the popup's left edge to the button's left edge, pushing
        // the box to the right. Same popup mechanics, but pivoted on the
        // button's top-center (with a small gap) instead.
        let pos = egui::pos2(resp.rect.center().x, resp.rect.top() - 4.0);
        egui::Area::new(id)
            .order(egui::Order::Foreground)
            .constrain(true)
            .fixed_pos(pos)
            .pivot(egui::Align2::CENTER_BOTTOM)
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    // Fixed content size: an auto-sized area offers
                    // unbounded height, which balloons the popup with dead
                    // space. 92px slider + spacing + measured % label.
                    let label_h = ui.fonts(|fonts| {
                        fonts.row_height(&egui::TextStyle::Small.resolve(ui.style()))
                    });
                    let content_h =
                        92.0 + ui.spacing().item_spacing.y + 2.0 + label_h;
                    ui.allocate_ui_with_layout(
                        egui::vec2(28.0, content_h),
                        egui::Layout::top_down(egui::Align::Center),
                        |ui| {
                            draw_vertical_volume_slider(ui, app);
                            ui.add_space(2.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}%",
                                    ((app.playback_volume / 1.5).clamp(0.0, 1.0)
                                        * 100.0)
                                        as i32
                                ))
                                .text_style(egui::TextStyle::Small)
                                .color(egui::Color32::from_rgb(176, 188, 203)),
                            );
                        },
                    );
                });
            });
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) || resp.clicked_elsewhere() {
            ui.memory_mut(|mem| mem.close_popup());
        }
    }
    resp
}

/// Compact vertical rail slider used inside the volume popup.
fn draw_vertical_volume_slider(ui: &mut egui::Ui, app: &mut KeyScribeApp) {
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(20.0, 92.0), egui::Sense::click_and_drag());
    let rail =
        egui::Rect::from_center_size(rect.center(), egui::vec2(5.0, rect.height() - 6.0));
    ui.painter()
        .rect_filled(rail, rail.width() * 0.5, SLIDER_RAIL_BG_DARK);
    let frac = (app.playback_volume / 1.5).clamp(0.0, 1.0);
    let handle_y = rail.bottom() - frac * rail.height();
    if frac > 0.0 {
        let fill = egui::Rect::from_min_max(
            egui::pos2(rail.left(), handle_y),
            egui::pos2(rail.right(), rail.bottom()),
        );
        ui.painter()
            .rect_filled(fill, rail.width() * 0.5, app.highlight_color);
    }
    let handle_fill = if resp.hovered() || resp.is_pointer_button_down_on() {
        ui.visuals().widgets.hovered.bg_fill
    } else {
        ui.visuals().widgets.inactive.bg_fill
    };
    ui.painter()
        .circle_filled(egui::pos2(rect.center().x, handle_y), 6.0, handle_fill);
    ui.painter().circle_stroke(
        egui::pos2(rect.center().x, handle_y),
        6.0,
        ui.visuals().widgets.inactive.fg_stroke,
    );
    // Captured before the slider below mutates app in place.
    let pre_popup_vol = app.playback_volume;
    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            // Drags coalesce into one undo step via pointer transitions;
            // taps count as discrete actions.
            let mut snap = MixSnapshot::capture(app);
            snap.playback_volume = pre_popup_vol;
            app.push_mix_undo_with(resp.dragged(), snap);
            let f = ((rail.bottom() - pos.y) / rail.height().max(1.0)).clamp(0.0, 1.0);
            app.playback_volume = f * 1.5;
            if let Some(engine) = &mut app.engine {
                engine.set_volume(app.playback_volume);
            }
        }
    }
    resp.on_hover_text("Drag to adjust volume");
}

/// Spotify-style progress row: thin rail with the elapsed portion in the
/// accent color, current time left of it and total duration right of it.
/// Click or drag to move the playhead, like any music player.
fn draw_seek_bar_row(ui: &mut egui::Ui, app: &mut KeyScribeApp, duration: f32) {
    let time_color = egui::Color32::from_rgb(176, 188, 203);
    let time_font = egui::TextStyle::Body.resolve(ui.style());
    let time_w = ui
        .fonts(|f| {
            f.layout_no_wrap("0:00:00".to_owned(), time_font.clone(), time_color)
                .size()
                .x
        })
        .max(40.0);
    let row_h = 18.0;
    let dark = ui.visuals().dark_mode;
    let (rail_fill, rail_fill_hover, rail_fill_active) = if dark {
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
    let accent = app.highlight_color;
    let enabled = duration > 0.0 && app.audio_raw.is_some();

    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), row_h),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.spacing_mut().item_spacing.x = 8.0;

            // Slot for the elapsed label, painted after the bar interaction
            // so it can preview the scrub target.
            let (cur_rect, _) =
                ui.allocate_exact_size(egui::vec2(time_w, row_h), egui::Sense::hover());

            let bar_w = (ui.available_width() - time_w - 8.0).max(40.0);
            let id = ui.make_persistent_id("media_seek_bar");
            let (rect, mut resp) = ui
                .push_id(id, |ui| {
                    ui.allocate_exact_size(
                        egui::vec2(bar_w, row_h),
                        if enabled {
                            egui::Sense::click_and_drag()
                        } else {
                            egui::Sense::hover()
                        },
                    )
                })
                .inner;

            let rail = egui::Rect::from_center_size(rect.center(), egui::vec2(bar_w, 4.0));
            let bg = if !enabled {
                rail_fill.gamma_multiply(0.4)
            } else if resp.is_pointer_button_down_on() {
                rail_fill_active
            } else if resp.hovered() {
                rail_fill_hover
            } else {
                rail_fill
            };
            ui.painter().rect_filled(rail, rail.height() * 0.5, bg);

            let frac = if duration > 0.0 {
                (app.selected_time_sec / duration).clamp(0.0, 1.0)
            } else {
                0.0
            };
            // While pressed, the handle previews the drag target.
            let handle_frac = if resp.is_pointer_button_down_on() {
                resp.interact_pointer_pos()
                    .map(|p| ((p.x - rail.left()) / rail.width().max(1.0)).clamp(0.0, 1.0))
                    .unwrap_or(frac)
            } else {
                frac
            };
            if handle_frac > 0.0 {
                let fill = egui::Rect::from_min_max(
                    egui::pos2(rail.left(), rail.top()),
                    egui::pos2(rail.left() + handle_frac * rail.width(), rail.bottom()),
                );
                let fill_color = if enabled {
                    accent
                } else {
                    accent.gamma_multiply(0.4)
                };
                ui.painter().rect_filled(fill, rail.height() * 0.5, fill_color);
            }
            let handle_center =
                egui::pos2(rail.left() + handle_frac * rail.width(), rect.center().y);
            let handle_fill = if !enabled {
                ui.visuals().widgets.inactive.bg_fill.gamma_multiply(0.4)
            } else if resp.hovered() || resp.is_pointer_button_down_on() {
                ui.visuals().widgets.hovered.bg_fill
            } else {
                ui.visuals().widgets.inactive.bg_fill
            };
            ui.painter().circle_filled(handle_center, 5.0, handle_fill);
            ui.painter().circle_stroke(
                handle_center,
                5.0,
                ui.visuals().widgets.inactive.fg_stroke,
            );

            if enabled {
                resp = resp.on_hover_text("Seek — click or drag");
            } else {
                resp = resp.on_hover_text("Load a track to seek");
            }

            // Elapsed time — shows the scrub preview while pressed.
            let cur_text = if enabled && resp.is_pointer_button_down_on() {
                resp.interact_pointer_pos()
                    .map(|p| {
                        let f = ((p.x - rail.left()) / rail.width().max(1.0))
                            .clamp(0.0, 1.0);
                        f * duration
                    })
                    .unwrap_or(app.selected_time_sec)
            } else {
                app.selected_time_sec
            };
            ui.painter().text(
                cur_rect.center(),
                egui::Align2::CENTER_CENTER,
                format_time(cur_text),
                time_font.clone(),
                time_color,
            );

            // Commit seeks on click and on drag release.
            let playing = app.is_playing();
            let mut seek_target: Option<f32> = None;
            if enabled && (resp.clicked() || resp.drag_stopped()) {
                if let Some(pos) = resp.interact_pointer_pos() {
                    let f = ((pos.x - rail.left()) / rail.width().max(1.0)).clamp(0.0, 1.0);
                    seek_target = Some(f * duration);
                }
            }
            if let Some(target) = seek_target {
                app.request_seek(target);
                if playing {
                    app.play_from_selected();
                }
            }

            // Total duration pinned to the right edge.
            let rest_w = ui.available_width().max(time_w);
            ui.allocate_ui_with_layout(
                egui::vec2(rest_w, row_h),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    let (total_rect, _) = ui.allocate_exact_size(
                        egui::vec2(time_w, row_h),
                        egui::Sense::hover(),
                    );
                    ui.painter().text(
                        total_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format_time(duration),
                        time_font.clone(),
                        time_color,
                    );
                },
            );
        },
    );
}

/// Spotify-style solid circular play/pause button (white disc in dark mode).
fn draw_play_button(
    ui: &mut egui::Ui,
    _app: &mut KeyScribeApp,
    is_playing: bool,
    enabled: bool,
    button_size: f32,
) -> egui::Response {
    ui.push_id("play_pause_circle", |ui| {
        let (rect, resp) =
            ui.allocate_exact_size(egui::vec2(button_size, button_size), egui::Sense::click());
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        // Disc takes the same gray as the other transport icons; the glyph
        // takes the panel fill so the pair always contrasts.
        let fg = ui.visuals().widgets.inactive.fg_stroke.color;
        let fg_hover = ui.visuals().widgets.hovered.fg_stroke.color;
        let fill = if !enabled {
            fg.gamma_multiply(0.35)
        } else if resp.is_pointer_button_down_on() {
            fg_hover.gamma_multiply(0.85)
        } else if resp.hovered() {
            fg_hover
        } else {
            fg
        };
        let radius = button_size * 0.5 - 2.0;
        ui.painter().circle_filled(rect.center(), radius, fill);
        let icon = if is_playing { PAUSE } else { PLAY };
        let icon_color = ui.visuals().panel_fill;
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            icon,
            icon_font_id((button_size * 0.42).round()),
            icon_color,
        );
        resp.on_hover_text("Play / Pause")
    })
    .inner
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
    let compact_layout = full_w < MEDIA_COMPACT_MAX_W;
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

    let inner_pad = if compact_layout { 10.0 } else { 14.0 };
    let inner_w = (full_w - 2.0 * inner_pad).max(0.0);

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
    ui.allocate_ui_with_layout(
        egui::vec2(full_w, target_h),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            ui.set_min_height(target_h);
            egui::Frame::none()
                .fill(panel_fill)
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(if compact_layout {
                    egui::Margin::symmetric(10.0, UI_VSPACE_MEDIUM)
                } else {
                    egui::Margin::symmetric(14.0, UI_VSPACE_MEDIUM)
                })
                // No outer gutter: the parent panel already insets its
                // content, so the fill spans exactly the same width as the
                // waveform plot and piano panes above/below it.
                .show(ui, |ui| {
                    // Force frame width to match the parent width so centering is stable.
                    ui.set_min_width(inner_w);
                    ui.set_max_width(inner_w);
                    // The trailing item_spacing egui appends after the last row
                    // would otherwise grow the frame `full_w + spacing` wide,
                    // pushing its right edge past the pane and clipping it at
                    // the window border. All rows below space themselves with
                    // explicit add_space calls.
                    ui.spacing_mut().item_spacing.x = 0.0;
                    // Explicit finite content height (was: unbounded
                    // available_height) so column centering math stays valid
                    // inside the scroll area below.
                    let content_h = (target_h - 2.0 * UI_VSPACE_MEDIUM).max(0.0);

                    // Safety net: if rows wrap beyond the reserved height
                    // (very narrow windows, huge fonts), scroll instead of
                    // clipping controls away.
                    egui::ScrollArea::vertical()
                        .id_source("media_panel_scroll")
                        // When the scrollbar shows it reserves ~10px and
                        // widens the frame past the pane, clipping the right
                        // edge against the window border. Hide it; scrolling
                        // still works as an overflow safety net.
                        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                        .max_height(content_h)
                        .show(ui, |ui| {
                    ui.set_min_width(inner_w);
                    // Clamp the content width too, so right-pinned widgets
                    // keep the frame's inner margin instead of touching the
                    // panel's rounded edge.
                    ui.set_max_width(inner_w);

                    if compact_layout {
                        ui.vertical(|ui| {
                            // Center artwork + details as one group.
                            let avail_w = ui.available_width();
                            let title_size = if title.len() > 30 {
                                (15.0_f32 * (30.0 / title.len() as f32).max(0.6)).max(12.0)
                            } else {
                                15.0
                            };
                            let secondary_text = if album.is_empty() {
                                artist.clone()
                            } else {
                                format!("{artist} · {album}")
                            };
                            let measure = |text: &str, size: f32| {
                                ui.fonts(|f| {
                                    f.layout_no_wrap(
                                        text.to_owned(),
                                        egui::FontId::proportional(size),
                                        egui::Color32::WHITE,
                                    )
                                    .size()
                                    .x
                                })
                            };
                            let meta_w = measure(title.as_str(), title_size)
                                .max(measure(secondary_text.as_str(), 14.0))
                                .max(measure(channel_status.as_str(), 12.0))
                                .min((avail_w - art_size - 12.0).max(60.0))
                                .max(60.0);
                            let group_w = art_size + 8.0 + meta_w;
                            let pad = ((avail_w - group_w) * 0.5).max(0.0);
                            ui.allocate_ui_with_layout(
                                egui::vec2(avail_w, art_size),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.add_space(pad);
                                    draw_album_art(
                                        ui,
                                        app.album_art_texture.as_ref(),
                                        art_size,
                                    );
                                    ui.add_space(8.0);
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(meta_w, art_size),
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
                                },
                            );

                            ui.add_space(UI_VSPACE_MEDIUM);

                            let transport_row = |ui: &mut egui::Ui,
                                                app: &mut KeyScribeApp,
                                                with_volume: bool,
                                                side_w: f32| {
                                // Slot layout: rewind right-aligned in the left
                                // slot keeps the play button dead-center.
                                ui.allocate_ui_with_layout(
                                    egui::vec2(side_w, button_size),
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
                                let play_resp = draw_play_button(
                                    ui,
                                    app,
                                    is_playing,
                                    analysis_ready,
                                    button_size,
                                );
                                if play_resp.clicked() {
                                    let current_pos = app.current_position_sec();

                                    if is_playing {
                                        app.stop();
                                    } else if app.audio_raw.is_some() {
                                        if app.processed_playback_samples.is_empty() && app.separated_stems.is_none() && !app.is_processing {
                                            app.request_rebuild(false, super::RebuildMode::Full);
                                        }

                                        if current_pos <= 0.0 || current_pos >= duration - 0.01 {
                                            app.play_from_selected();
                                        } else if let Some(engine) = &mut app.engine {
                                            engine.resume();
                                        }
                                    }
                                }

                                ui.allocate_ui_with_layout(
                                    egui::vec2(side_w, button_size),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                if icon_button(ui, FAST_FORWARD, "Skip Forward 5s", analysis_ready)
                                    .clicked()
                                {
                                    app.skip_by_seconds(SEEK_STEP_SEC);
                                }

                                ui.add_space(14.0);

                                let loop_resp = icon_toggle_button(
                                    ui,
                                    REPEAT,
                                    "Loop Selection",
                                    app.loop_enabled,
                                    analysis_ready,
                                    app.highlight_color,
                                );
                                if loop_resp.clicked() {
                                    app.toggle_loop();
                                }

                                if with_volume {
                                    let rest_w = ui.available_width().max(22.0);
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(rest_w, button_size),
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            draw_volume_button(ui, app, 16.0);
                                        },
                                    );
                                }
                                    },
                                );
                            };

                            // Slot layout: equal side slots keep the play
                            // button dead-center; the volume button rides the
                            // right edge, dropping to its own centered row
                            // only when there is no room in the right slot.
                            let avail_w = ui.available_width();
                            let side_w = ((avail_w - button_size) * 0.5).max(0.0);
                            let ff_loop_w = button_size * 2.0 + 14.0;
                            let volume_btn_w = 22.0_f32;
                            let volume_inline = side_w >= ff_loop_w + 10.0 + volume_btn_w;
                            ui.allocate_ui_with_layout(
                                egui::vec2(avail_w, button_size),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    transport_row(ui, app, volume_inline, side_w);
                                },
                            );
                            if !volume_inline {
                                ui.add_space(UI_VSPACE_COMPACT);
                                let vpad = ((avail_w - volume_btn_w) * 0.5).max(0.0);
                                ui.allocate_ui_with_layout(
                                    egui::vec2(avail_w, 26.0),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        ui.add_space(vpad);
                                        draw_volume_button(ui, app, 16.0);
                                    },
                                );
                            }
                        });
                    } else {
                        ui.columns(3, |cols| {
                            // Columns leave room for the seek bar row below.
                            let cols_h = content_h;
                            cols[0].set_height(cols_h);
                            cols[0].with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    draw_album_art(ui, app.album_art_texture.as_ref(), art_size);
                                    ui.add_space(8.0);

                                    let metadata_width = (ui.available_width() - 6.0).max(0.0);
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(metadata_width, cols_h),
                                        egui::Layout::top_down(egui::Align::Min)
                                            .with_main_align(egui::Align::Center),
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

                            cols[1].set_height(cols_h);
                            cols[1].allocate_ui_with_layout(
                                egui::vec2(cols[1].available_width(), cols_h),
                                egui::Layout::top_down(egui::Align::Center),
                                |ui| {
                                    let play_w = button_size;
                                    let side_w = ((ui.available_width() - play_w).max(0.0)) * 0.5;

                                    let play_row_height = button_size;
                                    // Spotify distributes transport + seek
                                    // vertically, centered as one group; the
                                    // seek bar spans wider than the buttons.
                                    let total_needed_h =
                                        play_row_height + 10.0 + 18.0;

                                    ui.add_space(((cols_h - total_needed_h) / 2.0).max(0.0));

                                    ui.horizontal(|ui| {
                                        // Default spacing is zeroed at the frame
                                        // level and stays zero here: the rewind
                                        // slot ends exactly at the column center
                                        // so any extra spacing would push the
                                        // play disc off-center.
                                        ui.spacing_mut().item_spacing.x = 0.0;

                                        ui.allocate_ui_with_layout(
                                            egui::vec2(side_w, play_row_height),
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                // Keep a gap to the play button
                                                // (default item_spacing.x is zeroed
                                                // at the frame level); right_to_left
                                                // placement means this shifts the
                                                // button left, off the slot edge.
                                                ui.add_space(12.0);
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
                                        let play_resp = draw_play_button(
                                            ui,
                                            app,
                                            is_playing,
                                            analysis_ready,
                                            button_size,
                                        );
                                        if play_resp.clicked() {
                                            let current_pos = app.current_position_sec();

                                            if is_playing {
                                                app.stop();
                                            } else if app.audio_raw.is_some() {
                                                if app.processed_playback_samples.is_empty() && app.separated_stems.is_none() {
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

                                        // Explicit gap to the fast-forward button
                                        // (default item_spacing.x is zeroed at the
                                        // frame level); placed after the play button
                                        // so the play disc stays column-centered.
                                        ui.add_space(12.0);

                                        // Fast-forward + loop are added directly to the row so
                                        // they share the exact same alignment as the play
                                        // button (nested centered slots offset them).
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

                                        let loop_resp = icon_toggle_button(
                                            ui,
                                            REPEAT,
                                            "Loop Selection",
                                            app.loop_enabled,
                                            analysis_ready,
                                            app.highlight_color,
                                        );
                                        if loop_resp.clicked() {
                                            app.toggle_loop();
                                        }
                                    });

                                    ui.add_space(10.0);
                                    draw_seek_bar_row(ui, app, duration);
                                },
                            );

                            cols[2].set_height(cols_h);
                            cols[2].allocate_ui_with_layout(
                                egui::vec2(cols[2].available_width(), cols_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    // Desktop layout keeps the volume slider
                                    // permanently visible, right-pinned and
                                    // vertically centered.
                                    draw_volume_slider_row(ui, app, 110.0);
                                },
                            );
                        });
                    }

                            if compact_layout {
                                ui.add_space(UI_VSPACE_COMPACT);
                                draw_seek_bar_row(ui, app, duration);
                            }
                        });
                });
        },
    );
}

/// Loop editor pill: floats over the bottom edge of the waveform, centered.
/// Appears while looping is active, disappears when it is not.
pub(crate) fn draw_loop_pill(
    ui: &mut egui::Ui,
    app: &mut KeyScribeApp,
    content_bottom: f32,
    content_left: f32,
    avail_w: f32,
) {
    if !app.loop_enabled {
        return;
    }
    // Compact overlay pill: it floats over the waveform, so every pixel
    // counts. Buttons stay finger-friendly but small; keep the width math in
    // sync with draw_loop_inputs below (fields 42 wide, 4px gaps).
    let widget_h = ui.spacing().interact_size.y.clamp(24.0, 32.0);
    // Exact content width: four step buttons + two time fields + the dash
    // separator + gaps (including the trailing spacing egui appends after
    // the last row item), plus the frame's side margins.
    let pill_w = (4.0 * widget_h + 2.0 * 42.0 + 12.0 + 7.0 * 4.0 + 16.0)
        .min(avail_w - 16.0)
        .max(180.0);
    // Frame inner margin is 5 top + 5 bottom: exact fit, so the row sits
    // vertically centered with no slack.
    let pill_h = widget_h + 10.0;
    let pill_rect = egui::Rect::from_min_size(
        egui::pos2(
            content_left + (avail_w - pill_w) * 0.5,
            content_bottom - pill_h + 8.0,
        ),
        egui::vec2(pill_w, pill_h),
    );
    ui.allocate_ui_at_rect(pill_rect, |ui| {
        let fill = if app.dark_mode {
            crate::theme::MEDIA_PANEL_BG_DARK
        } else {
            crate::theme::MEDIA_PANEL_BG_LIGHT
        };
        egui::Frame::none()
            .fill(fill)
            .rounding(egui::Rounding::same(10.0))
            .inner_margin(egui::Margin::symmetric(8.0, 5.0))
            .show(ui, |ui| {
                // Content box excludes the frame's own margins: sizing to the
                // full pill size would overflow the pill rect to the right.
                ui.set_min_size(egui::vec2(pill_w - 16.0, pill_h - 10.0));
                draw_loop_inputs(ui, app);
            });
    });
}

fn draw_loop_inputs(ui: &mut egui::Ui, app: &mut KeyScribeApp) {
    if !app.loop_enabled {
        return;
    }

    // Wrap so the time fields stay reachable on narrow panels instead of
    // overflowing (and getting clipped) past the panel edge.
    ui.horizontal_wrapped(|ui| {
    
    let (start, end) = app.loop_selection.unwrap_or((0.0, 0.0));
    
    let start_id = ui.make_persistent_id("loop_start_input");
    let end_id = ui.make_persistent_id("loop_end_input");

    if !ui.memory(|mem| mem.has_focus(start_id)) {
        app.loop_start_input_str = format_time(start);
    }
    if !ui.memory(|mem| mem.has_focus(end_id)) {
        app.loop_end_input_str = format_time(end);
    }

    ui.spacing_mut().item_spacing.x = 4.0;

    // Fixed widget height matching draw_loop_pill so the whole row stays
    // on one visual line (mixed auto heights drifted apart).
    let widget_h = ui.spacing().interact_size.y.clamp(24.0, 32.0);

    let mut new_start = start;
    let mut new_end = end;
    let mut changed = false;

    let duration = app.timeline_duration_sec();

    let small_step_button = |ui: &mut egui::Ui, id: &str, icon: &str, tooltip: &str| {
        ui.push_id(id, |ui| {
            ui.add_sized(
                [widget_h, widget_h],
                egui::Button::new(egui::RichText::new(icon).font(icon_font_id(12.0))),
            )
        })
        .inner
        .on_hover_text(tooltip)
        .clicked()
    };

    if small_step_button(ui, "start_minus", MINUS, "Subtract 1 second from loop start") {
        new_start = (start - 1.0).max(0.0);
        changed = true;
    }
    let start_resp = ui.add_sized(
        [42.0, widget_h],
        egui::TextEdit::singleline(&mut app.loop_start_input_str)
            .id(start_id)
            .margin(egui::vec2(4.0, (widget_h - 16.0) * 0.5)),
    );
    if small_step_button(ui, "start_plus", PLUS, "Add 1 second to loop start") {
        new_start = (start + 1.0).min(end - 0.1);
        changed = true;
    }

    ui.allocate_ui_with_layout(
        egui::vec2(12.0, widget_h),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(12.0, widget_h), egui::Sense::hover());
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "\u{2014}",
                egui::TextStyle::Body.resolve(ui.style()),
                ui.visuals().text_color(),
            );
        },
    );

    if small_step_button(ui, "end_minus", MINUS, "Subtract 1 second from loop end") {
        new_end = (end - 1.0).max(start + 0.1);
        changed = true;
    }
    let end_resp = ui.add_sized(
        [42.0, widget_h],
        egui::TextEdit::singleline(&mut app.loop_end_input_str)
            .id(end_id)
            .margin(egui::vec2(4.0, (widget_h - 16.0) * 0.5)),
    );
    if small_step_button(ui, "end_plus", PLUS, "Add 1 second to loop end") {
        new_end = (end + 1.0).min(duration);
        changed = true;
    }

    let marker_time = |input: &str| -> Option<f32> {
        let s = input.trim();
        if s.len() == 1 {
            let c = s.chars().next().unwrap().to_ascii_uppercase();
            if c >= 'A' && c <= 'Z' {
                let idx = (c as u8 - b'A') as usize;
                if let Some(hash) = &app.loaded_audio_hash {
                    if let Some(markers) = app.file_markers.get(hash) {
                        if idx < markers.len() {
                            return Some(markers[idx].time());
                        }
                    }
                }
            }
        }
        None
    };

    if start_resp.lost_focus() {
        let parsed = marker_time(&app.loop_start_input_str)
            .or_else(|| crate::ui::utils::parse_time(&app.loop_start_input_str));
        if let Some(parsed) = parsed {
            new_start = parsed.max(0.0).min(end - 0.1);
            changed = true;
        }
    }
    if end_resp.lost_focus() {
        let parsed = marker_time(&app.loop_end_input_str)
            .or_else(|| crate::ui::utils::parse_time(&app.loop_end_input_str));
        if let Some(parsed) = parsed {
            new_end = parsed.min(duration).max(start + 0.1);
            changed = true;
        }
    }

    if changed {
        app.loop_selection = Some((new_start, new_end));
        app.loop_playback_enabled = true;
        if app.is_playing() {
            let pos = app.current_position_sec();
            if pos < new_start || pos >= new_end {
                app.selected_time_sec = new_start;
                app.play_range(new_start, None);
            }
        }
    }
    });
}
