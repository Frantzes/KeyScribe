//! Android entry point and platform glue.
//!
//! `cargo-apk` packages this library as a `cdylib`; the `native-activity`
//! glue (pulled in by `android-activity`) looks up the `android_main` symbol
//! once the activity is created. The desktop `main.rs` entry point is not
//! used on Android.
//!
//! Also hosts the small amount of JNI needed for media import (runtime
//! media permission + scanning shared audio directories) and window insets.

use std::ffi::CString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::app::KeyScribeApp;
use crate::platform::{self, AndroidPaths};

/// Assets copied into the writable app sandbox on first launch. `cargo-apk`
/// bundles the contents of `android/assets/` at the APK asset root, so these
/// names are relative to that root.
const BUNDLED_ASSETS: &[&str] = &[
    "models/basic-pitch.onnx",
    "models/mel_spectrogram.onnx",
    "models/beat_this_small.onnx",
    "models/transition_matrix.json",
];

const MEDIA_PERMISSIONS: &[&str] = &[
    "android.permission.READ_MEDIA_AUDIO",
    "android.permission.READ_EXTERNAL_STORAGE",
];

const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "mp4", "mkv", "avi", "mov", "webm",
];

/// Shared media directories scanned by the in-app audio browser. Reading them
/// requires the runtime media permission; the app's own external dir does not.
const PUBLIC_MEDIA_ROOTS: &[&str] = &[
    "/storage/emulated/0/Music",
    "/storage/emulated/0/Download",
    "/storage/emulated/0/Documents",
    "/storage/emulated/0/Podcasts",
    "/storage/emulated/0/Recordings",
];

static ANDROID_APP: OnceLock<android_activity::AndroidApp> = OnceLock::new();

/// Path of an audio file picked through the native document picker, waiting
/// to be consumed by the UI thread.
static PICKED_AUDIO: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn android_main(app: android_activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    let _ = ANDROID_APP.set(app.clone());
    init_paths_and_assets(&app);

    let options = eframe::NativeOptions {
        event_loop_builder: Some(Box::new(move |builder| {
            builder.with_android_app(app);
        })),
        ..Default::default()
    };

    let _ = eframe::run_native(
        "KeyScribe",
        options,
        Box::new(|cc| Box::new(KeyScribeApp::new(cc))),
    );
}

/// Capture the app sandbox dirs and extract bundled ONNX models into the
/// writable files dir, where `platform::resolve_model_path` will find them.
fn init_paths_and_assets(app: &android_activity::AndroidApp) {
    let files_dir = app
        .internal_data_path()
        .or_else(|| app.external_data_path())
        .unwrap_or_else(|| PathBuf::from("."));

    let cache_dir = files_dir
        .parent()
        .map(|parent| parent.join("cache"))
        .filter(|dir| dir.exists())
        .unwrap_or_else(|| files_dir.clone());

    platform::set_android_paths(AndroidPaths {
        files_dir: files_dir.clone(),
        cache_dir,
        external_dir: app.external_data_path(),
    });

    let _ = std::fs::create_dir_all(files_dir.join("models"));

    let asset_manager = app.asset_manager();
    for relative in BUNDLED_ASSETS {
        let Ok(name) = CString::new(*relative) else {
            continue;
        };
        let Some(mut asset) = asset_manager.open(name.as_c_str()) else {
            continue;
        };
        let Ok(bytes) = asset.buffer() else {
            continue;
        };
        let dest = files_dir.join(relative);
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.len() as usize == bytes.len() {
                continue;
            }
        }
        let _ = write_asset(&dest, bytes);
    }
}

fn write_asset(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(dest)?;
    file.write_all(bytes)
}

// ---------------------------------------------------------------------------
// JNI helpers (media permission + content insets)
// ---------------------------------------------------------------------------

/// Run `f` with a JNI env and the Activity object. Returns `None` when the JNI
/// context is unavailable (e.g. before the activity is ready).
fn with_activity<R>(
    f: impl FnOnce(&mut jni::JNIEnv, &jni::objects::JObject) -> jni::errors::Result<R>,
) -> Option<R> {
    let ctx = ndk_context::android_context();
    let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) }.ok()?;
    let mut env = vm.attach_current_thread().ok()?;
    let activity = unsafe { jni::objects::JObject::from_raw(ctx.context().cast()) };
    let result = f(&mut env, &activity);
    // Never leak a pending exception: a later JNI call (e.g. oboe's audio
    // enumeration) aborts the process if one is still set.
    if matches!(env.exception_check(), Ok(true)) {
        let _ = env.exception_clear();
    }
    result.ok()
}

fn permission_granted(env: &mut jni::JNIEnv, activity: &jni::objects::JObject) -> bool {
    use jni::objects::JValue;
    for permission in MEDIA_PERMISSIONS {
        let Ok(name) = env.new_string(permission) else {
            continue;
        };
        let result = env.call_method(
            activity,
            "checkSelfPermission",
            "(Ljava/lang/String;)I",
            &[JValue::Object(&name)],
        );
        if let Ok(value) = result {
            if value.i().unwrap_or(-1) == 0 {
                return true;
            }
        }
    }
    false
}

/// Whether the app may read shared media (audio) files.
pub fn media_permission_granted() -> bool {
    with_activity(|env, activity| Ok(permission_granted(env, activity))).unwrap_or(false)
}

/// Ask the system to grant the runtime media permission (shows a dialog). The
/// result is not delivered as a callback; poll [`media_permission_granted`].
pub fn request_media_permission() {
    use jni::objects::JValue;
    let _ = with_activity(|env, activity| {
        let strings: Vec<_> = MEDIA_PERMISSIONS
            .iter()
            .filter_map(|p| env.new_string(p).ok())
            .collect();
        if strings.is_empty() {
            return Ok(());
        }
        let first = strings.first().unwrap();
        let array = env.new_object_array(strings.len() as i32, "java/lang/String", first)?;
        for (index, item) in strings.iter().enumerate() {
            env.set_object_array_element(&array, index as i32, item)?;
        }
        env.call_method(
            activity,
            "requestPermissions",
            "([Ljava/lang/String;I)V",
            &[JValue::Object(&array), JValue::Int(0x4b53)],
        )?;
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// Native document picker (SAF)
// ---------------------------------------------------------------------------

/// Launch the native Android document picker for audio files. The selection
/// is delivered asynchronously and consumed via [`poll_picked_audio`].
pub fn pick_audio_file() {
    let _ = with_activity(|env, activity| {
        // Instance method on the Activity: avoids FindClass, which cannot see
        // app classes from the native `android_main` thread.
        env.call_method(activity, "pickAudio", "()V", &[])?;
        Ok(())
    });
}

/// Put `text` on the Android system clipboard. Applied once the window has
/// focus (Android 10+ blocks clipboard writes from unfocused apps).
pub fn set_clipboard(text: &str) {
    use jni::objects::JValue;
    let _ = with_activity(|env, activity| {
        let jtext = env.new_string(text)?;
        env.call_method(
            activity,
            "setClipboard",
            "(Ljava/lang/String;)V",
            &[JValue::Object(&jtext)],
        )?;
        Ok(())
    });
}

/// Log an info message to logcat under the `KeyScribe` tag (used to audit
/// destructive operations like cache cleanup).
pub fn log_info(message: &str) {
    use jni::objects::JValue;
    let _ = with_activity(|env, _activity| {
        let tag = env.new_string("KeyScribe")?;
        let msg = env.new_string(message)?;
        let class = env.find_class("android/util/Log")?;
        env.call_static_method(
            class,
            "i",
            "(Ljava/lang/String;Ljava/lang/String;)I",
            &[JValue::Object(&tag), JValue::Object(&msg)],
        )?;
        Ok(())
    });
}

/// Read the current Android clipboard text, or `None` when unavailable.
pub fn get_clipboard() -> Option<String> {
    with_activity(|env, activity| {
        let result = env.call_method(activity, "getClipboardText", "()Ljava/lang/String;", &[])?;
        let value = result.l()?;
        if value.is_null() {
            return Ok(String::new());
        }
        let text = unsafe { jni::objects::JString::from_raw(value.into_raw()) };
        let text: String = env.get_string(&text).map(|s| s.into()).unwrap_or_default();
        Ok(text)
    })
}

/// Encrypt and persist the MVSep API key using the Android Keystore.
pub fn store_secret(value: &str) -> bool {
    use jni::objects::JValue;
    with_activity(|env, activity| {
        let jval = env.new_string(value)?;
        let res = env.call_method(
            activity,
            "storeSecret",
            "(Ljava/lang/String;)Z",
            &[JValue::Object(&jval)],
        )?;
        Ok(res.z().unwrap_or(false))
    })
    .unwrap_or(false)
}

/// Decrypt and return the MVSep API key, or `None` when absent/unavailable.
pub fn load_secret() -> Option<String> {
    with_activity(|env, activity| {
        let res = env.call_method(activity, "loadSecret", "()Ljava/lang/String;", &[])?;
        let obj = res.l()?;
        if obj.is_null() {
            return Ok(String::new());
        }
        let text = unsafe { jni::objects::JString::from_raw(obj.into_raw()) };
        let value: String = env.get_string(&text).map(|s| s.into()).unwrap_or_default();
        Ok(value)
    })
    .filter(|value| !value.is_empty())
}

/// Remove the stored secret and its Android Keystore key.
pub fn clear_secret() -> bool {
    with_activity(|env, activity| {
        let res = env.call_method(activity, "clearSecret", "()Z", &[])?;
        Ok(res.z().unwrap_or(false))
    })
    .unwrap_or(false)
}

/// Take the path of the most recently picked audio file, if any.
pub fn poll_picked_audio() -> Option<String> {
    PICKED_AUDIO.lock().ok().and_then(|mut slot| slot.take())
}

/// Called by `MainActivity.onActivityResult` with the copied cache path (or an
/// empty string when the user cancelled).
#[no_mangle]
pub extern "system" fn Java_com_frantzes_keyscribe_MainActivity_nativeOnAudioPicked(
    mut env: jni::JNIEnv,
    _class: jni::objects::JClass,
    path: jni::objects::JString,
) {
    let value: String = env.get_string(&path).map(|s| s.into()).unwrap_or_default();
    if let Ok(mut slot) = PICKED_AUDIO.lock() {
        *slot = if value.is_empty() { None } else { Some(value) };
    }
}

/// System bar insets in physical pixels as `(top, bottom)`, or `None` when
/// unavailable. Uses `WindowInsets` rather than the native content rect, which
/// some OEM ROMs report with an inflated top value.
pub fn system_bar_insets_px() -> Option<(i32, i32)> {
    with_activity(|env, activity| {
        let top = env.call_method(activity, "systemBarTopPx", "()I", &[])?;
        let bottom = env.call_method(activity, "systemBarBottomPx", "()I", &[])?;
        Ok((top.i().unwrap_or(0), bottom.i().unwrap_or(0)))
    })
}

/// Top/left/right/bottom window insets in logical points, derived from the
/// Android content rect. Returns zeros when unavailable so callers can apply
/// them unconditionally.
pub fn content_insets_points(
    pixels_per_point: f32,
    viewport_width_points: f32,
    viewport_height_points: f32,
) -> (f32, f32, f32, f32) {
    let Some(app) = ANDROID_APP.get() else {
        return (0.0, 0.0, 0.0, 0.0);
    };
    let rect = app.content_rect();
    if rect.right <= 0 || rect.bottom <= 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let ppp = pixels_per_point.max(0.01);
    let full_w = viewport_width_points * ppp;
    let full_h = viewport_height_points * ppp;
    let left = (rect.left as f32 / ppp).max(0.0);
    let top = (rect.top as f32 / ppp).max(0.0);
    let right = ((full_w - rect.right as f32) / ppp).max(0.0);
    let bottom = ((full_h - rect.bottom as f32) / ppp).max(0.0);
    (left, top, right, bottom)
}

/// Audio files currently visible to the app, as `(display name, absolute
/// path)`, sorted by directory then name.
pub fn scan_audio_files() -> Vec<(String, String)> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(external) = platform::android_paths().and_then(|p| p.external_dir.clone()) {
        roots.push(external);
    }
    roots.extend(PUBLIC_MEDIA_ROOTS.iter().map(PathBuf::from));

    let mut entries: Vec<(String, String)> = Vec::new();
    for root in roots {
        scan_dir(&root, 0, &mut entries);
    }
    entries.sort();
    entries.dedup();
    entries
}

fn scan_dir(dir: &Path, depth: usize, out: &mut Vec<(String, String)>) {
    if depth > 3 || out.len() >= 500 {
        return;
    }
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            scan_dir(&path, depth + 1, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if AUDIO_EXTENSIONS.iter().any(|a| a.eq_ignore_ascii_case(ext)) {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio")
                    .to_string();
                out.push((name, path.to_string_lossy().to_string()));
            }
        }
    }
}
