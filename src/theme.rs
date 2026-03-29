use std::sync::OnceLock;

use eframe::egui;
use egui_phosphor::Variant;

pub const ACCENT_ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 140, 45);
pub const BG_MAIN: egui::Color32 = egui::Color32::from_rgb(20, 24, 30);
pub const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(28, 33, 42);
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
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(36, 42, 52);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(50, 58, 72);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 70, 86);
        visuals.panel_fill = BG_MAIN;
        visuals.window_fill = BG_PANEL;
    } else {
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(238, 239, 242);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(225, 228, 234);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(215, 220, 227);
        visuals.panel_fill = egui::Color32::from_rgb(247, 248, 251);
        visuals.window_fill = egui::Color32::from_rgb(241, 244, 248);
    }

    visuals.selection.bg_fill = accent;
    visuals.hyperlink_color = accent;
    ctx.set_visuals(visuals);
}
