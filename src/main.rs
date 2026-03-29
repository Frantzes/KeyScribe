mod analysis;
mod app;
mod audio_io;
mod dsp;
mod playback;
mod theme;

use app::TranscriberApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Audio Visual Transcriber",
        native_options,
        Box::new(|cc| Box::new(TranscriberApp::new(cc))),
    )
}
