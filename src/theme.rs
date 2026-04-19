use std::sync::OnceLock;

use eframe::egui;
use egui_phosphor::Variant;

pub const ACCENT_PURPLE: egui::Color32 = egui::Color32::from_rgb(148, 106, 255);
pub const BG_DARK_GREY: egui::Color32 = egui::Color32::from_rgb(16, 16, 19); // #101013
pub const BG_MAIN: egui::Color32 = BG_DARK_GREY;
pub const BG_PANEL: egui::Color32 = BG_DARK_GREY;
pub const MEDIA_PANEL_BG_DARK: egui::Color32 = BG_DARK_GREY;
pub const MEDIA_PANEL_BG_LIGHT: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_BG_DARK: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_BG_LIGHT: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_BG_HOVERED_DARK: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_BG_HOVERED_LIGHT: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_BG_ACTIVE_DARK: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_BG_ACTIVE_LIGHT: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_ROW_BG_DARK: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_ROW_BG_LIGHT: egui::Color32 = BG_DARK_GREY;
pub const SLIDER_ROW_STROKE_DARK: egui::Color32 = egui::Color32::from_rgb(82, 93, 108);
pub const SLIDER_ROW_STROKE_LIGHT: egui::Color32 = egui::Color32::from_rgb(34, 38, 44);
pub const SLIDER_RAIL_BG_DARK: egui::Color32 = egui::Color32::from_rgb(22, 24, 28);
pub const SLIDER_RAIL_BG_LIGHT: egui::Color32 = egui::Color32::from_rgb(22, 24, 28);
pub const SLIDER_RAIL_BG_HOVER_DARK: egui::Color32 = egui::Color32::from_rgb(28, 31, 36);
pub const SLIDER_RAIL_BG_HOVER_LIGHT: egui::Color32 = egui::Color32::from_rgb(28, 31, 36);
pub const SLIDER_RAIL_BG_ACTIVE_DARK: egui::Color32 = egui::Color32::from_rgb(34, 38, 44);
pub const SLIDER_RAIL_BG_ACTIVE_LIGHT: egui::Color32 = egui::Color32::from_rgb(34, 38, 44);
pub const PIANO_WHITE_KEY_BG: egui::Color32 = egui::Color32::from_gray(238);
pub const PIANO_WHITE_KEY_STROKE: egui::Color32 = egui::Color32::from_gray(90);
pub const PIANO_BLACK_KEY_BG: egui::Color32 = egui::Color32::from_rgb(10, 10, 12);
pub const PIANO_BLACK_KEY_STROKE: egui::Color32 = egui::Color32::from_rgb(26, 28, 34);
pub const PIANO_C4_MARKER: egui::Color32 = egui::Color32::from_gray(155);
pub const PROBABILITY_PANE_BG: egui::Color32 = BG_DARK_GREY;
pub const PROBABILITY_PANE_WHITE_KEY_STROKE: egui::Color32 = egui::Color32::from_rgb(74, 74, 74);
pub const PROBABILITY_PANE_BLACK_KEY_BG: egui::Color32 = PIANO_BLACK_KEY_BG;
pub const PROBABILITY_PANE_BLACK_KEY_STROKE: egui::Color32 = egui::Color32::from_rgb(96, 96, 102);
pub const TEXT_MAIN: egui::Color32 = egui::Color32::from_rgb(236, 236, 236);
pub const ERROR_RED: egui::Color32 = egui::Color32::from_rgb(220, 70, 70);

static FONTS_CONFIGURED: OnceLock<()> = OnceLock::new();

fn configure_fonts_once(ctx: &egui::Context) {
    if FONTS_CONFIGURED.get().is_some() {
        return;
    }

    let mut defs = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut defs, Variant::Regular);
    let mut icon_family = vec!["phosphor".to_owned()];
    if let Some(existing) = defs.families.get(&egui::FontFamily::Proportional) {
        for name in existing {
            if !icon_family.iter().any(|n| n == name) {
                icon_family.push(name.clone());
            }
        }
    }
    defs.families
        .insert(egui::FontFamily::Name("icons".into()), icon_family);

    let mut candidates = vec![
        "C:/Windows/Fonts/Jost-Regular.ttf".to_string(),
        "C:/Windows/Fonts/jost-regular.ttf".to_string(),
        "C:/Windows/Fonts/Jost VariableFont_wght.ttf".to_string(),
        "C:/Windows/Fonts/Jost-VariableFont_wght.ttf".to_string(),
        "C:/Windows/Fonts/Jost[wght].ttf".to_string(),
    ];

    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        let user_fonts = std::path::PathBuf::from(local_app_data).join("Microsoft/Windows/Fonts");
        for name in [
            "Jost-Regular.ttf",
            "jost-regular.ttf",
            "Jost VariableFont_wght.ttf",
            "Jost-VariableFont_wght.ttf",
            "Jost[wght].ttf",
        ] {
            candidates.push(user_fonts.join(name).to_string_lossy().replace('\\', "/"));
        }
    }

    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            defs.font_data.insert(
                "jost_ui".to_owned(),
                egui::FontData::from_owned(bytes).into(),
            );

            defs.families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "jost_ui".to_owned());
            defs.families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .insert(0, "jost_ui".to_owned());
            break;
        }
    }

    // Fallback: keep default egui text fonts if Jost is unavailable.
    ctx.set_fonts(defs);
    let _ = FONTS_CONFIGURED.set(());
}

pub fn apply_brand_theme(ctx: &egui::Context, dark_mode: bool, accent: egui::Color32) {
    configure_fonts_once(ctx);

    let mut visuals = if dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    if dark_mode {
        visuals.override_text_color = Some(TEXT_MAIN);
        visuals.widgets.noninteractive.bg_fill = BG_PANEL;
        visuals.widgets.inactive.bg_fill = SLIDER_BG_DARK;
        visuals.widgets.hovered.bg_fill = SLIDER_BG_HOVERED_DARK;
        visuals.widgets.active.bg_fill = SLIDER_BG_ACTIVE_DARK;
        visuals.widgets.inactive.weak_bg_fill = SLIDER_RAIL_BG_DARK;
        visuals.widgets.hovered.weak_bg_fill = SLIDER_RAIL_BG_HOVER_DARK;
        visuals.widgets.active.weak_bg_fill = SLIDER_RAIL_BG_ACTIVE_DARK;
        visuals.panel_fill = BG_MAIN;
        visuals.window_fill = BG_PANEL;
    } else {
        visuals.override_text_color = Some(TEXT_MAIN);
        visuals.widgets.noninteractive.bg_fill = BG_PANEL;
        visuals.widgets.inactive.bg_fill = SLIDER_BG_LIGHT;
        visuals.widgets.hovered.bg_fill = SLIDER_BG_HOVERED_LIGHT;
        visuals.widgets.active.bg_fill = SLIDER_BG_ACTIVE_LIGHT;
        visuals.widgets.inactive.weak_bg_fill = SLIDER_RAIL_BG_LIGHT;
        visuals.widgets.hovered.weak_bg_fill = SLIDER_RAIL_BG_HOVER_LIGHT;
        visuals.widgets.active.weak_bg_fill = SLIDER_RAIL_BG_ACTIVE_LIGHT;
        visuals.panel_fill = BG_MAIN;
        visuals.window_fill = BG_PANEL;
    }

    visuals.selection.bg_fill = accent;
    visuals.hyperlink_color = accent;
    ctx.set_visuals(visuals);
}
