#![cfg_attr(
    all(not(debug_assertions), feature = "desktop-ui"),
    windows_subsystem = "windows"
)]

#[cfg(feature = "desktop-ui")]
use eframe::egui;
#[cfg(feature = "desktop-ui")]
use transcriber::app::TranscriberApp;

#[cfg(feature = "desktop-ui")]
fn load_window_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!("../icon.png");
    let image = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (width, height) = image.dimensions();

    Some(egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    })
}

#[cfg(feature = "desktop-ui")]
fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport = native_options.viewport.with_maximized(true);
    if let Some(icon) = load_window_icon() {
        native_options.viewport = native_options.viewport.with_icon(icon);
    }

    eframe::run_native(
        "Audio Visual Transcriber",
        native_options,
        Box::new(|cc| Box::new(TranscriberApp::new(cc))),
    )
}

#[cfg(not(feature = "desktop-ui"))]
fn main() {
    eprintln!("Desktop UI is disabled. Enable the `desktop-ui` feature to run this binary.");
}
