use eframe::egui;

pub fn responsive_icon_button_size(ui: &egui::Ui) -> f32 {
    ui.spacing().interact_size.y.clamp(30.0, 42.0)
}

fn responsive_icon_font_size(button_size: f32) -> f32 {
    (button_size * 0.52).clamp(16.0, 22.0)
}

pub fn icon_button(ui: &mut egui::Ui, icon: &str, tooltip: &str, enabled: bool) -> egui::Response {
    icon_button_with_fill(ui, icon, tooltip, enabled, None, None)
}

pub fn icon_toggle_button(
    ui: &mut egui::Ui,
    icon: &str,
    tooltip: &str,
    enabled_state: bool,
    enabled: bool,
    accent_color: egui::Color32,
) -> egui::Response {
    let fill = if enabled_state {
        accent_color
    } else {
        ui.visuals().widgets.inactive.bg_fill
    };

    let text_color_override = if enabled && enabled_state {
        Some(egui::Color32::WHITE)
    } else {
        None
    };

    icon_button_with_fill(ui, icon, tooltip, enabled, Some(fill), text_color_override)
}

fn icon_button_with_fill(
    ui: &mut egui::Ui,
    icon: &str,
    tooltip: &str,
    enabled: bool,
    fill_override: Option<egui::Color32>,
    text_color_override: Option<egui::Color32>,
) -> egui::Response {
    let button_size = responsive_icon_button_size(ui);
    let icon_size = responsive_icon_font_size(button_size);
    let desired = egui::vec2(button_size, button_size);
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(desired, sense);
    let response = response.on_hover_text(tooltip);
    let visuals = ui.style().interact(&response);

    let mut bg_fill = fill_override.unwrap_or(visuals.bg_fill);
    if !enabled {
        bg_fill = ui.visuals().widgets.inactive.bg_fill;
    }

    ui.painter()
        .rect(rect, visuals.rounding, bg_fill, visuals.bg_stroke);

    let text_color = text_color_override.unwrap_or_else(|| {
        if enabled {
            visuals.text_color()
        } else {
            ui.visuals().widgets.inactive.text_color()
        }
    });

    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        icon_font_id(icon_size),
        text_color,
    );

    response
}

pub fn icon_font_id(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name("icons".into()))
}

const KNOB_START_DEG: f32 = 135.0;
const KNOB_SWEEP_DEG: f32 = 270.0;

fn knob_angle_pos(center: egui::Pos2, radius: f32, deg: f32) -> egui::Pos2 {
    let rad = deg.to_radians();
    egui::pos2(
        center.x + radius * rad.cos(),
        center.y + radius * rad.sin(),
    )
}

fn paint_knob_arc(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    from_deg: f32,
    to_deg: f32,
    stroke: egui::Stroke,
) {
    let sweep = (to_deg - from_deg).abs();
    if sweep < 0.1 {
        return;
    }
    let steps = (sweep / 6.0).ceil().clamp(2.0, 48.0) as usize;
    let points: Vec<egui::Pos2> = (0..=steps)
        .map(|i| {
            knob_angle_pos(
                center,
                radius,
                from_deg + (to_deg - from_deg) * (i as f32 / steps as f32),
            )
        })
        .collect();
    painter.add(egui::Shape::line(points.clone(), stroke));
    painter.circle_filled(points[0], stroke.width * 0.5, stroke.color);
    painter.circle_filled(points[steps], stroke.width * 0.5, stroke.color);
}

/// Synth/Ableton-style rotary knob: a 270° arc track around a disc with a
/// pointer line. Drag vertically to change (Shift for fine steps),
/// double-click to reset to `default`. With `bipolar` the value arc fills
/// from 12 o'clock so center stands for `default`-centered ranges.
/// `granularity` > 0 snaps values to a step (e.g. 0.25 for dB, 0.1 for st).
pub fn synth_knob(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    value: &mut f32,
    min: f32,
    max: f32,
    default: f32,
    size: f32,
    accent: egui::Color32,
    bipolar: bool,
    enabled: bool,
    colored_border: bool,
    granularity: f32,
) -> bool {
    let span = (max - min).max(f32::EPSILON);
    let mut changed = false;
    let (rect, mut resp) = if enabled {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::drag());
        (rect, resp.on_hover_cursor(egui::CursorIcon::ResizeVertical))
    } else {
        ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover())
    };

    if enabled {
        if resp.dragged() {
            let drag_div = if ui.input(|i| i.modifiers.shift) {
                600.0
            } else {
                150.0
            };
            let t = ((*value - min) / span).clamp(0.0, 1.0);
            let new_t = (t - resp.drag_delta().y / drag_div).clamp(0.0, 1.0);
            let mut new_value = min + span * new_t;
            if granularity > 0.0 {
                new_value = (new_value / granularity).round() * granularity;
            }
            if (new_value - *value).abs() > f32::EPSILON {
                *value = new_value.clamp(min, max);
                changed = true;
            }
        }
        if resp.double_clicked() {
            *value = default.clamp(min, max);
            changed = true;
        }
    } else {
        resp = resp.on_hover_text("Disabled — unmute or finish stem analysis");
    }

    // Paint. Angles run clockwise from the bottom-left (135°) through the
    // top (270°) to the bottom-right (45°), matching synth conventions.
    let painter = ui.painter();
    let center = rect.center();
    let t = ((*value - min) / span).clamp(0.0, 1.0);
    let angle = KNOB_START_DEG + t * KNOB_SWEEP_DEG;
    let visuals = ui.visuals();
    let dim = if enabled { 1.0 } else { 0.4 };
    let arc_r = size * 0.47;
    let disc_r = size * 0.35;

    paint_knob_arc(
        painter,
        center,
        arc_r,
        KNOB_START_DEG,
        KNOB_START_DEG + KNOB_SWEEP_DEG,
        egui::Stroke::new(
            (size * 0.06).max(1.5),
            visuals.widgets.inactive.fg_stroke.color.gamma_multiply(0.25 * dim),
        ),
    );

    let arc_start = if bipolar {
        KNOB_START_DEG + KNOB_SWEEP_DEG * 0.5
    } else {
        KNOB_START_DEG
    };
    if (angle - arc_start).abs() > 0.5 {
        let (a0, a1) = if angle > arc_start {
            (arc_start, angle)
        } else {
            (angle, arc_start)
        };
        paint_knob_arc(
            painter,
            center,
            arc_r,
            a0,
            a1,
            egui::Stroke::new((size * 0.075).max(2.0), accent.gamma_multiply(dim)),
        );
    }

    let hovered = enabled && (resp.hovered() || resp.dragged());
    let stroke_color = if hovered {
        accent
    } else if colored_border {
        accent.gamma_multiply(0.55 * dim)
    } else {
        visuals.widgets.inactive.fg_stroke.color.gamma_multiply(0.45 * dim)
    };
    painter.circle(
        center,
        disc_r,
        visuals.widgets.inactive.bg_fill.gamma_multiply(dim),
        egui::Stroke::new(1.0, stroke_color),
    );
    let p0 = knob_angle_pos(center, disc_r * 0.30, angle);
    let p1 = knob_angle_pos(center, disc_r * 0.80, angle);
    painter.line_segment(
        [p0, p1],
        egui::Stroke::new(2.0, visuals.text_color().gamma_multiply(dim)),
    );

    changed
}

/// Horizontal slider whose moved portion is painted in `accent`, matching
/// `synth_knob`'s value arc: unipolar fills from the range start, bipolar
/// (e.g. pitch ±st, centered ranges) fills outward from the middle.
///
/// Drag horizontally or click to jump. The rail follows the ambient widget
/// visuals, so callers can restyle it with a scope like anywhere else.
pub fn accent_slider(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    value: &mut f32,
    min: f32,
    max: f32,
    bipolar: bool,
    size: egui::Vec2,
    accent: egui::Color32,
) -> bool {
    let span = (max - min).max(f32::EPSILON);
    let mut changed = false;
    // Persistent id keeps the widget identity stable for drags that cross
    // frames; interaction itself is position-driven below.
    let _id = ui.make_persistent_id(id_salt);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    let resp = resp
        .on_hover_cursor(egui::CursorIcon::ResizeHorizontal)
        .on_hover_text("Drag to adjust");

    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let handle_r = (size.y * 0.33).clamp(5.0, 9.0);
            let travel_min = rect.left() + handle_r;
            let travel_max = rect.right() - handle_r;
            let frac =
                ((pos.x - travel_min) / (travel_max - travel_min).max(1.0)).clamp(0.0, 1.0);
            let new_value = (min + span * frac).clamp(min, max);
            if (new_value - *value).abs() > f32::EPSILON {
                *value = new_value;
                changed = true;
            }
        }
    }

    let painter = ui.painter();
    let visuals = ui.visuals();
    let handle_r = (size.y * 0.33).clamp(5.0, 9.0);
    let rail_h = (size.y * 0.22).clamp(3.0, 6.0);
    let travel_min = rect.left() + handle_r;
    let travel_max = rect.right() - handle_r;
    let rail = egui::Rect::from_min_max(
        egui::pos2(travel_min, rect.center().y - rail_h * 0.5),
        egui::pos2(travel_max, rect.center().y + rail_h * 0.5),
    );
    painter.rect_filled(rail, rail.height() * 0.5, visuals.widgets.inactive.bg_fill);

    // Moved portion: from the range start, or outward from the middle when
    // bipolar — the same rule as `synth_knob`'s value arc.
    let origin_t = if bipolar { 0.5 } else { 0.0 };
    let t_now = ((*value - min) / span).clamp(0.0, 1.0);
    let (f0, f1) = if t_now >= origin_t {
        (origin_t, t_now)
    } else {
        (t_now, origin_t)
    };
    if f1 - f0 > 1.0e-3 {
        let fill = egui::Rect::from_min_max(
            egui::pos2(travel_min + f0 * (travel_max - travel_min), rail.top()),
            egui::pos2(travel_min + f1 * (travel_max - travel_min), rail.bottom()),
        );
        painter.rect_filled(fill, rail.height() * 0.5, accent);
    }

    let active = resp.hovered() || resp.dragged();
    let handle_pos = egui::pos2(
        travel_min + t_now * (travel_max - travel_min),
        rect.center().y,
    );
    painter.circle_filled(
        handle_pos,
        handle_r,
        if active {
            visuals.widgets.hovered.bg_fill
        } else {
            visuals.widgets.inactive.bg_fill
        },
    );
    painter.circle_stroke(handle_pos, handle_r, visuals.widgets.inactive.fg_stroke);

    changed
}
