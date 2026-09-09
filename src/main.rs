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

/// Whether the NVIDIA + Wayland combination is in use.
///
/// On NVIDIA's egl-wayland backend a vsync'd `eglSwapBuffers` performs a
/// synchronous roundtrip waiting for a compositor frame callback. Hidden
/// surfaces (another workspace, minimized) never get callbacks, so the
/// main thread wedges inside eframe's present, stops answering compositor
/// pings, and the desktop raises "Application Not Responding" a few seconds
/// later — while audio keeps playing and everything resumes on return.
/// (Root-caused via gdb: main thread parked in
/// `glutin EGL SwapBuffers → libnvidia-egl-wayland →
/// wl_display_dispatch_queue` for the entire hidden period.)
/// `SwapInterval::DontWait` never waits, so it cannot wedge.
#[cfg(feature = "native-ui")]
fn nvidia_wayland_present_would_wedge() -> bool {
    // Honour an explicit X11 backend choice: XWayland/X11 presents don't
    // take the blocking egl-wayland path.
    if std::env::var("WINIT_UNIX_BACKEND")
        .map(|v| v.eq_ignore_ascii_case("x11"))
        .unwrap_or(false)
    {
        return false;
    }
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
        || std::env::var("XDG_SESSION_TYPE")
            .map(|v| v.eq_ignore_ascii_case("wayland"))
            .unwrap_or(false);
    wayland && std::path::Path::new("/dev/nvidia0").exists()
}

#[cfg(feature = "native-ui")]
fn main() -> eframe::Result<()> {
    let mut native_options = eframe::NativeOptions::default();
    // The Wayland app-id MUST match the desktop file id
    // (`com.frantzes.keyscribe.desktop`) so the compositor can associate the
    // window with the app. Without this the window shows up as "(unknown)"
    // and hang detection / task switching misbehave on Linux.
    native_options.viewport = native_options
        .viewport
        .with_maximized(true)
        .with_app_id("com.frantzes.keyscribe");

    // Vsync default: off on NVIDIA + Wayland (see
    // `nvidia_wayland_present_would_wedge`), on everywhere else.
    // `KEYSCRIBE_VSYNC=0` forces off, `KEYSCRIBE_VSYNC=1` forces on.
    let vsync = match std::env::var("KEYSCRIBE_VSYNC").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => false,
        Ok("1") | Ok("on") | Ok("true") => true,
        _ => !nvidia_wayland_present_would_wedge(),
    };
    native_options.vsync = vsync;

    run_native_app(native_options)
}

#[cfg(not(feature = "native-ui"))]
fn main() {
    eprintln!("Native UI is disabled. Enable `desktop-ui` to run this binary.");
}
