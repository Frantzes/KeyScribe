use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};

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
