//! Platform abstraction seams: app sandbox paths and asset/model locations.
//!
//! Desktop resolves paths relative to the executable and the working
//! directory. Android has no meaningful working directory and the executable
//! lives inside a read-only APK install tree, so `android_main` hands us the
//! app sandbox paths via [`set_android_paths`] and model resolution consults
//! them first. Every other platform keeps the original search order.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// App sandbox paths captured from `AndroidApp` at startup.
#[derive(Clone, Debug)]
pub struct AndroidPaths {
    /// App-private files dir (`Context.getFilesDir()`): writable, backed up.
    pub files_dir: PathBuf,
    /// App-private cache dir (`Context.getCacheDir()`): evictable.
    pub cache_dir: PathBuf,
    /// App-specific external dir, when the shared volume is mounted.
    pub external_dir: Option<PathBuf>,
}

static ANDROID_PATHS: OnceLock<AndroidPaths> = OnceLock::new();

/// Install the Android sandbox paths. Idempotent: later calls are ignored.
pub fn set_android_paths(paths: AndroidPaths) {
    let _ = ANDROID_PATHS.set(paths);
}

pub fn android_paths() -> Option<&'static AndroidPaths> {
    ANDROID_PATHS.get()
}

pub fn is_android() -> bool {
    cfg!(target_os = "android")
}

/// File name of the ONNX Runtime shared library for the current platform.
/// Android uses the Linux `.so` naming; without this it falls through to the
/// macOS `.dylib` branch and transcription fails with "library not found".
pub fn onnxruntime_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "android") || cfg!(target_os = "linux") {
        "libonnxruntime.so"
    } else {
        "libonnxruntime.dylib"
    }
}

/// Base directory for persisted app state (settings, markers, per-file data).
pub fn data_dir() -> Option<PathBuf> {
    android_paths().map(|p| p.files_dir.clone())
}

/// Base directory for the analysis cache.
pub fn cache_dir() -> Option<PathBuf> {
    android_paths().map(|p| p.cache_dir.clone())
}

/// Directories searched for ONNX models and other data files, highest
/// priority first.
pub fn model_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(p) = android_paths() {
        dirs.push(p.files_dir.join("models"));
        dirs.push(p.files_dir.clone());
        if let Some(external) = &p.external_dir {
            dirs.push(external.join("models"));
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
            dirs.push(parent.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("models"));
        dirs.push(cwd);
    }

    dirs
}

/// Directory containing this library's shared object on Android, which is
/// also where the APK's bundled native libs (e.g. `libonnxruntime.so`) are
/// extracted. Resolved with `dladdr` on one of our own functions.
#[cfg(target_os = "android")]
pub fn android_native_lib_dir() -> Option<PathBuf> {
    use std::ffi::{c_void, CStr};

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const std::os::raw::c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const std::os::raw::c_char,
        dli_saddr: *mut c_void,
    }

    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> i32;
    }

    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };

    let addr = android_native_lib_dir as *const c_void;
    let found = unsafe { dladdr(addr, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return None;
    }

    let path = unsafe { CStr::from_ptr(info.dli_fname) }.to_string_lossy();
    PathBuf::from(path.as_ref())
        .parent()
        .map(|parent| parent.to_path_buf())
}

#[cfg(not(target_os = "android"))]
pub fn android_native_lib_dir() -> Option<PathBuf> {
    None
}

/// Android automation hook: if the app's external files dir contains a
/// `_keyscribe_autoload.txt` whose first line is a media path, return it.
///
/// Used by on-device tests/CI to trigger an import without UI input (MIUI
/// blocks `adb shell input` and runtime-permission grants). Inert unless the
/// marker file exists, so it cannot affect normal installs.
pub fn android_autoload_path() -> Option<PathBuf> {
    let external = android_paths()?.external_dir.clone()?;
    let marker = external.join("_keyscribe_autoload.txt");
    let contents = std::fs::read_to_string(marker).ok()?;
    let candidate = PathBuf::from(contents.lines().next()?.trim());
    candidate.exists().then_some(candidate)
}

/// Android automation hook: open the in-app audio browser on startup when the
/// app external dir contains `_keyscribe_picker`. Inert otherwise.
pub fn android_picker_requested() -> bool {
    android_paths()
        .and_then(|p| p.external_dir.clone())
        .map(|dir| dir.join("_keyscribe_picker").exists())
        .unwrap_or(false)
}

/// Android automation hook: an optional `_keyscribe_automation.txt` in the app
/// external files dir selects a startup action (`picker`, `settings`,
/// `mvsep`). Inert unless the file exists.
pub fn android_automation_marker() -> Option<String> {
    let external = android_paths()?.external_dir.clone()?;
    let contents = std::fs::read_to_string(external.join("_keyscribe_automation.txt")).ok()?;
    let value = contents.trim().to_string();
    (!value.is_empty()).then_some(value)
}

/// Resolve a model filename (or relative subpath) against the platform search
/// dirs, then as a bare relative path. Mirrors the historical
/// `demucs::resolve_model_path` order.
pub fn resolve_model_path(filename: &str) -> Option<PathBuf> {
    let given = Path::new(filename);
    if given.is_absolute() && given.exists() {
        return Some(given.to_path_buf());
    }

    for dir in model_search_dirs() {
        let candidate = dir.join(filename);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if given.exists() {
        return Some(given.to_path_buf());
    }
    None
}
