use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Copy the CPU ONNX Runtime (`libonnxruntime.so*`) next to dev binaries.
///
/// Since v0.3.0 ort uses `load-dynamic`: the runtime is resolved at startup
/// from beside the executable, and plain `cargo run` binaries never got one
/// (only the packaged bundles stage it). Without it the first session
/// construction deadlocks inside ort's global init instead of failing, so
/// every transcription wedges at "Analyzing..." with no message. Staging the
/// already-downloaded `vendor/ort-gpu` files beside the dev exe restores
/// zero-config `cargo run` transcription.
///
/// Best-effort and silent: without `vendor/ort-gpu` (fresh clones, CI) the
/// build proceeds and the app reports a clear "library not found" error at
/// runtime instead of hanging (see `inference::ensure_onnxruntime_loadable`).
/// Only the ~22 MB CPU runtime is staged; the 370 MB CUDA provider stays a
/// packaged-build concern (dev falls back to multi-threaded CPU).
#[cfg(target_os = "linux")]
fn stage_linux_ort_runtime_for_dev() {
    use std::fs;

    println!("cargo:rerun-if-changed=vendor/ort-gpu");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    if manifest_dir.as_os_str().is_empty() {
        return;
    }
    let vendor_dir = manifest_dir.join("vendor").join("ort-gpu");
    let Ok(entries) = fs::read_dir(&vendor_dir) else {
        return;
    };

    // OUT_DIR = <profile>/build/<pkg>-<hash>/out → ancestors land in <profile>/.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let Some(exe_dir) = out_dir
        .ancestors()
        .nth(3)
        .map(|p| p.to_path_buf())
    else {
        return;
    };

    let mut staged = 0u32;
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("libonnxruntime.so") {
            continue;
        }
        let dest = exe_dir.join(name.as_ref());
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            // Recreate the symlink (e.g. libonnxruntime.so ->
            // libonnxruntime.so.1.24.4) when missing or dangling.
            let points_at_usable_target = fs::metadata(&dest).is_ok();
            if points_at_usable_target {
                continue;
            }
            let _ = fs::remove_file(&dest);
            if let Ok(target) = fs::read_link(entry.path()) {
                #[cfg(target_os = "linux")]
                {
                    use std::os::unix::fs::symlink;
                    if symlink(&target, &dest).is_ok() {
                        staged += 1;
                    }
                }
            }
        } else if file_type.is_file() {
            let copy_needed = match (fs::metadata(entry.path()), fs::metadata(&dest)) {
                (Ok(src_meta), Ok(dest_meta)) => src_meta.len() != dest_meta.len(),
                _ => true,
            };
            if copy_needed && fs::copy(entry.path(), &dest).is_ok() {
                staged += 1;
            }
        }
    }
    if staged > 0 {
        println!(
            "cargo:warning=staged {staged} ONNX Runtime file(s) next to dev binary in {}",
            exe_dir.display()
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn stage_linux_ort_runtime_for_dev() {}

fn create_windows_icon_from_png(png_path: &Path, out_dir: &Path) -> Result<PathBuf, String> {
    let image = image::open(png_path)
        .map_err(|err| format!("failed to decode {}: {err}", png_path.display()))?
        .into_rgba8();
    let (width, height) = image.dimensions();

    if width != height {
        return Err(format!(
            "icon image must be square, got {}x{} from {}",
            width,
            height,
            png_path.display()
        ));
    }

    let icon_image = ico::IconImage::from_rgba_data(width, height, image.into_raw());
    let icon_entry = ico::IconDirEntry::encode(&icon_image)
        .map_err(|err| format!("failed to encode icon entry: {err}"))?;

    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);
    icon_dir.add_entry(icon_entry);

    let out_icon_path = out_dir.join("keyscribe-build-icon.ico");
    let mut out_file =
        File::create(&out_icon_path).map_err(|err| format!("failed to create icon file: {err}"))?;
    icon_dir
        .write(&mut out_file)
        .map_err(|err| format!("failed to write icon file: {err}"))?;

    Ok(out_icon_path)
}

fn main() {
    println!("cargo:rerun-if-changed=icon.png");
    println!("cargo:rerun-if-changed=icon.ico");

    stage_linux_ort_runtime_for_dev();

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is required for build script"),
    );
    let out_dir =
        PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is required for build script"));

    let png_icon = manifest_dir.join("icon.png");
    let fallback_ico = manifest_dir.join("icon.ico");

    let icon_path = if png_icon.is_file() {
        match create_windows_icon_from_png(&png_icon, &out_dir) {
            Ok(generated) => generated,
            Err(err) => {
                if fallback_ico.is_file() {
                    eprintln!(
                        "warning: failed to generate .ico from icon.png ({err}); falling back to icon.ico"
                    );
                    fallback_ico
                } else {
                    panic!("failed to generate Windows icon from icon.png: {err}");
                }
            }
        }
    } else if fallback_ico.is_file() {
        fallback_ico
    } else {
        panic!("missing icon assets: expected icon.png or icon.ico in project root");
    };

    let mut resource = winresource::WindowsResource::new();
    resource.set_icon(
        icon_path
            .to_str()
            .expect("Windows icon path must be valid UTF-8"),
    );
    resource.set("FileDescription", "Keyscribe");
    resource.set("ProductName", "Keyscribe");
    resource.set("InternalName", "Keyscribe");
    resource.set("OriginalFilename", "keyscribe.exe");

    if let Err(err) = resource.compile() {
        panic!("failed to compile Windows resources: {err}");
    }
}
