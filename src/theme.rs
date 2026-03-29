use eframe::egui;

pub const ACCENT_ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 140, 45);
pub const ACCENT_ORANGE_SOFT: egui::Color32 = egui::Color32::from_rgb(255, 178, 110);
pub const BG_MAIN: egui::Color32 = egui::Color32::from_rgb(20, 24, 30);
pub const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(28, 33, 42);
pub const TEXT_MAIN: egui::Color32 = egui::Color32::from_rgb(236, 236, 236);
pub const ERROR_RED: egui::Color32 = egui::Color32::from_rgb(220, 70, 70);

pub fn apply_brand_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(TEXT_MAIN);
    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(36, 42, 52);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(50, 58, 72);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 70, 86);
    visuals.selection.bg_fill = ACCENT_ORANGE;
    visuals.panel_fill = BG_MAIN;
    visuals.window_fill = BG_PANEL;
    visuals.hyperlink_color = ACCENT_ORANGE_SOFT;
    ctx.set_visuals(visuals);
}
