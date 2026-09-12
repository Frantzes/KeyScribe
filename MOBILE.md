# KeyScribe on Android

This document covers the Android build of KeyScribe. The app reuses the same
egui/eframe UI as the desktop build, gated by Cargo features so that
desktop-only capabilities are compiled out instead of being stubbed at
runtime.

## Status / scope

Android MVP feature set:

- Import an audio file and decode it (Symphonia).
- Transcribe with Basic Pitch + beat tracking (ONNX Runtime, arm64-v8a).
- Waveform + piano-roll visualization with playback, seek and A-B loop.
- Cloud stem separation via MVSep.

Not included on Android: local Demucs separation, video playback, sheet-music
engraving (Verovio) and MuseScore PDF export.

## Architecture

The UI was decoupled from platform specifics with a small seam:

| Piece | Location | Purpose |
|---|---|---|
| `Capability features` | `Cargo.toml` | `video`, `local-demucs`, `pdf-export`, `sheet-render`, `mp3-encode`. `desktop-ui` enables all; `android` enables none. |
| `platform` module | `src/platform.rs` | Storage dirs, model search paths, ONNX library name, Android hooks. |
| Android entry point | `src/android.rs` | `android_main`, sandbox paths, asset extraction, JNI (SAF picker, permission), window insets. |
| Java activity | `android/java/.../MainActivity.java` | Receives the SAF picker result (no activity-result API in `android-activity` 0.5). |
| Build/packaging | `scripts/build-android.ps1` | cargo + javac/d8 + aapt2/zipalign/apksigner. |

`KeyScribeApp` reads state and model paths through `platform`, so the same UI
code runs on both desktop and Android. `is_touch_platform()` is true on
Android and drives the touch layout and larger hit targets.

## Prerequisites

- Rust with the `aarch64-linux-android` target:
  `rustup target add aarch64-linux-android`
- Android SDK (build-tools + platform 35), NDK r27, and a JDK (Android
  Studio's JBR works).

## Build

```powershell
# Builds target/android/keyscribe.apk and installs it on the connected device.
./scripts/build-android.ps1 -Install
```

The script builds the Rust `cdylib` with cargo, compiles the Java
`MainActivity`, dexes it with `d8`, links the manifest/assets with `aapt2`,
adds the native libraries, then `zipalign`s and signs with the debug keystore.

> `cargo-apk` is not used: it hardcodes `android:hasCode="false"` and cannot
> package the Java activity the SAF picker needs.

> **MIUI / Xiaomi:** `adb install` is rejected (`INSTALL_FAILED_USER_RESTRICTED`)
> unless "Install via USB" is enabled in Developer options. The script disables
> the package verifier and now fails loudly if the install is rejected, instead
> of leaving the previous build installed.

## ONNX Runtime

`ort` is built with `load-dynamic`; the runtime is not linked at build time.
Download the official Android AAR and extract the arm64 library:

```powershell
./scripts/fetch-onnxruntime-android.ps1 -Version 1.24.2
# writes android/libs/arm64-v8a/libonnxruntime.so
```

The build script packages `android/libs/<abi>/*.so` (plus `libc++_shared.so`
from the NDK) into `lib/<abi>/`. At runtime the loader resolves
`libonnxruntime.so` by name from the app's native library directory;
`src/inference.rs` also probes that directory explicitly.

Keep the AAR version aligned with the `ort` crate's `api-NN` feature
(`ort 2.0.0-rc.12` -> `api-24` -> ONNX Runtime 1.24.x).

## Bundled models

Small models are bundled as APK assets (`android/assets/models/`) and copied
into the app sandbox on first launch:

- `basic-pitch.onnx`
- `mel_spectrogram.onnx`
- `beat_this_small.onnx`
- `transition_matrix.json`

The 271 MB Demucs model is intentionally not bundled (local Demucs is disabled
on Android).

## Import: native document picker

"Open Audio" launches the Android Storage Access Framework document picker
(`ACTION_OPEN_DOCUMENT`, `audio/*`). No media permission prompt is needed:
SAF grants per-file read access.

The flow spans three pieces:

1. `src/android.rs::pick_audio_file` calls `MainActivity.pickAudio()` over JNI.
2. Java `MainActivity.onActivityResult` copies the chosen content URI into the
   app cache and calls the native `nativeOnAudioPicked(path)`.
3. `update()` polls `poll_picked_audio()` and imports the path.

A Java class is required because `android-activity` 0.5 (pinned by
winit 0.29 / eframe 0.27) has no `startActivityForResult` API and the NDK
`ANativeActivity` callbacks have no activity-result hook.

An in-app browser (permission + directory scan) remains available as a
fallback through the `picker` automation marker.

## Device automation hooks

MIUI (and some OEMs) block `adb shell input` and `pm grant`, so inert markers
in the app's external files directory (`/sdcard/Android/data/com.frantzes.keyscribe/files/`)
allow scripted testing:

- `_keyscribe_autoload.txt` containing an absolute media path -> import it on
  launch.
- `_keyscribe_picker` (empty file) -> open the audio browser on launch.
- `_keyscribe_automation.txt` containing one of `pick` (native picker),
  `picker` (in-app browser), `settings`, `recent`, `mvsep`, `paste` (fill the
  MVSep key from the clipboard), `cleancache`, `mvsepkey:<key>`, or
  `zoom:<factor>` -> perform that action on launch.

None has any effect unless the file exists.

## Mobile UI differences

- The keyboard-settings cog expands to a compact 2x2 slider grid in a bounded
  scroll area, so the piano and media controls stay visible.
- The top-bar cog opens a scrollable, width-bounded settings window instead of
  the desktop popup menu.
- The Sheet Music / Waveform view toggles and their separator are hidden;
  mobile is waveform-only.
- The manual "Audio Path" field is hidden; imports go through the native
  document picker.
- Text and hit targets are sized down so the top bar stays compact, and
  slider knobs are kept small (the handle radius is derived from the widget
  height).
- The mobile toolbar is icon-only: import (file-plus), Recent (opens a
  scrollable list of recently opened files) and Settings. The Help button is
  hidden on touch.
- Tooltips are disabled (there is no hover on touch).
- The MVSep API key field has a Paste button in both the settings pane and the
  MVSep modal, since egui has no native text context menu on Android.
- The bottom panel reserves space for the Android navigation bar.
- The keyboard and probability pane zoom (1.0 = fit-width, up to 4x) and pan
  together; the chord-suggestion plate is pinned to the visible pane and sized
  to its text, so it stays visible while zooming/panning.

## MVSep API key storage

The key is encrypted at rest via the `secrets` module; it is **never** written
to a plaintext file:

- **Android**: AES-256-GCM with a key generated in the hardware-backed Android
  Keystore (`MainActivity.storeSecret`/`loadSecret`/`clearSecret`). The key
  material never leaves the Keystore; only `IV || ciphertext` is stored, in the
  app-private files dir (`mvsep.key`).
- **Windows**: DPAPI (`CryptProtectData`), bound to the current user profile.

`mvsep::find_mvsep_api_key` checks, in order: the `MVSEP_API_KEY` environment
variable, the platform keystore, then a legacy plaintext `.env` (which it
migrates into the keystore and deletes). On platforms without a keystore
integration (Linux/macOS) it falls back to the legacy `.env`.

The Paste button and the 2-second autosave both go through this path.

## Cache cleanup

"Clean cache" removes only three subdirectories of the app-private cache dir
(`<data>/<pkg>/cache/.keyscribe_cache`, `.transcriber_cache`, `stems`). The
code refuses any path that is not strictly inside the cache base, never
follows symlinks, and logs every path to logcat under `KeyScribe`. It cannot
touch shared storage, personal files, or system directories.

## Known limitations

- The alpha build signs with the debug keystore and is not Play-ready.
- Local Demucs and video are disabled; MVSep requires an API key entered in
  Settings.
- Sheet-music engraving/preview is unavailable (Verovio not compiled for
  Android).

## iOS

iOS is not wired up yet. `eframe` 0.27 has only partial iOS support; adding it
will likely require either upgrading egui/eframe or building a native
SwiftUI frontend over the same platform-agnostic core. The `platform` seam and
capability features were designed to make that possible.
