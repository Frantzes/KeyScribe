use eframe::egui;

use crate::analysis::{PIANO_HIGH_MIDI, PIANO_LOW_MIDI};

pub const PIANO_ZOOM_MIN: f32 = 0.35;
pub const PIANO_ZOOM_MAX: f32 = 1.0;
pub const WHITE_KEY_LENGTH_TO_WIDTH: f32 = 6.3;
pub const MIN_PIANO_KEY_HEIGHT: f32 = 16.0;
pub const MIN_PROBABILITY_STRIP_HEIGHT: f32 = 20.0;

pub struct KeyboardDrawResult {
    pub clicked: bool,
    pub max_scroll_px: f32,
}

pub fn keyboard_white_key_width(viewport_width: f32, zoom: f32) -> f32 {
    let white_count = (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI)
        .filter(|midi| !is_black_key(*midi))
        .count() as f32;
    let fit_width = (viewport_width / white_count.max(1.0)).max(1.0);
    fit_width * zoom.clamp(PIANO_ZOOM_MIN, PIANO_ZOOM_MAX)
}

fn white_index_before_midi(midi: u8) -> usize {
    (PIANO_LOW_MIDI..midi).filter(|m| !is_black_key(*m)).count()
}

pub fn draw_piano_view(
    ui: &mut egui::Ui,
    probs: &[f32],
    sensitivity: f32,
    zoom: f32,
    key_height: f32,
    scroll_px: f32,
    highlight_color: egui::Color32,
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
        // Optimized: use fast approximation instead of expensive pow operations
        let adjusted = (p * s).clamp(0.0, 1.0);
        let activation_threshold = 0.12;
        if adjusted >= activation_threshold {
            painter.rect_filled(key_rect, 0.0, highlight_color);
            painter.rect_stroke(
                key_rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(90)),
            );
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

        painter.rect_filled(key_rect, 2.0, egui::Color32::from_gray(55));
        painter.rect_stroke(
            key_rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(65)),
        );

        let idx = (midi - PIANO_LOW_MIDI) as usize;
        let p = probs.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let s = sensitivity.clamp(0.0, 2.0);
        // Optimized: use fast approximation instead of expensive pow operations
        let adjusted = (p * s).clamp(0.0, 1.0);
        let activation_threshold = 0.12;
        if adjusted >= activation_threshold {
            painter.rect_filled(key_rect, 2.0, highlight_color);
            painter.rect_stroke(
                key_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(65)),
            );
        }
    }

    if (PIANO_LOW_MIDI..=PIANO_HIGH_MIDI).contains(&60) {
        let c4_white_idx = white_index_before_midi(60);
        let cx = x_start + c4_white_idx as f32 * white_w + white_w * 0.5;
        if cx >= rect.left() && cx <= rect.right() {
            let marker_radius = 4.0;
            let marker_y = (rect.bottom() - marker_radius - 2.0).max(rect.top() + marker_radius);
            painter.circle_filled(
                egui::pos2(cx, marker_y),
                marker_radius,
                egui::Color32::from_gray(155),
            );
        }
    }

    KeyboardDrawResult {
        clicked: response.clicked(),
        max_scroll_px,
    }
}

pub fn draw_probability_pane(
    ui: &mut egui::Ui,
    probs_smoothed: &[f32],
    probs_raw: &[f32],
    zoom: f32,
    scroll_px: f32,
    strip_height: f32,
    highlight_color: egui::Color32,
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

        let h = p_raw * (rect.height() - 8.0);
        if h > 0.5 {
            let bar = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, rect.bottom() - h - 2.0),
                egui::pos2(x1 - 1.0, rect.bottom() - 2.0),
            );
            painter.rect_filled(bar, 1.0, highlight_color);
        }

        let glow_h = p_smooth * (rect.height() - 8.0);
        if glow_h > 0.5 {
            let glow = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, rect.bottom() - glow_h - 2.0),
                egui::pos2(x1 - 1.0, rect.bottom() - glow_h - 1.0),
            );
            let glow_color = egui::Color32::from_rgba_unmultiplied(
                highlight_color.r().saturating_add(28),
                highlight_color.g().saturating_add(28),
                highlight_color.b().saturating_add(28),
                180,
            );
            painter.rect_filled(glow, 1.0, glow_color);
        }

        white_index += 1;
    }

    let black_h = rect.height();
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
            painter.rect_filled(bar, 1.0, highlight_color);
        }

        let glow_h = p_smooth * (key_rect.height() - 4.0);
        if glow_h > 0.5 {
            let glow = egui::Rect::from_min_max(
                egui::pos2(x0 + 1.0, key_rect.bottom() - glow_h - 1.0),
                egui::pos2(x1 - 1.0, key_rect.bottom() - glow_h),
            );
            let glow_color = egui::Color32::from_rgba_unmultiplied(
                highlight_color.r().saturating_add(36),
                highlight_color.g().saturating_add(36),
                highlight_color.b().saturating_add(36),
                210,
            );
            painter.rect_filled(glow, 1.0, glow_color);
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
