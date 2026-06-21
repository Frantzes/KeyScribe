#![cfg_attr(
    all(not(debug_assertions), feature = "desktop-ui"),
    windows_subsystem = "windows"
)]

#[cfg(feature = "desktop-ui")]
use eframe::egui;
#[cfg(feature = "native-ui")]
use keyscribe_lib::app::KeyScribeApp;

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

#[cfg(feature = "native-ui")]
#[allow(unused_mut)]
fn run_native_app(mut native_options: eframe::NativeOptions) -> eframe::Result<()> {
    #[cfg(feature = "desktop-ui")]
    if let Some(icon) = load_window_icon() {
        native_options.viewport = native_options.viewport.with_icon(icon);
    }

    eframe::run_native(
        "Keyscribe",
        native_options,
        Box::new(|cc| Box::new(KeyScribeApp::new(cc))),
    )
}

#[cfg(feature = "native-ui")]
fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport = native_options.viewport.with_maximized(true);

    run_native_app(native_options)
}

#[cfg(not(feature = "native-ui"))]
fn main() {
    eprintln!("Native UI is disabled. Enable `desktop-ui` to run this binary.");
}
