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
