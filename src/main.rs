#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod app;
mod audio_io;
mod cqt;
mod dsp;
mod inference;
mod pipeline;
mod playback;
mod preprocessing;
mod ring_buffer;
mod theme;
mod ui;
mod viterbi;

use app::TranscriberApp;
use eframe::egui;

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
