use eframe::egui;

pub fn parse_hex_color(hex: &str) -> Option<egui::Color32> {
    let trimmed = hex.trim().trim_start_matches('#');
    if trimmed.len() != 6 {
        return None;
    }

    let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
    let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
    let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
    Some(egui::Color32::from_rgb(r, g, b))
}

pub fn color_to_hex(color: egui::Color32) -> String {
    format!("#{:02X}{:02X}{:02X}", color.r(), color.g(), color.b())
}

pub fn push_recent_color(recent: &mut Vec<String>, color: egui::Color32) {
    let hex = color_to_hex(color);
    recent.retain(|item| item != &hex);
    recent.insert(0, hex);
    if recent.len() > 10 {
        recent.truncate(10);
    }
}

pub fn accent_soft(color: egui::Color32) -> egui::Color32 {
    let r = ((color.r() as u16 + 255) / 2) as u8;
    let g = ((color.g() as u16 + 255) / 2) as u8;
    let b = ((color.b() as u16 + 255) / 2) as u8;
    egui::Color32::from_rgb(r, g, b)
}

pub fn format_time(sec: f32) -> String {
    let total = sec.max(0.0).floor() as u64;
    let m = total / 60;
    let s = total % 60;
    format!("{m:02}:{s:02}")
}
