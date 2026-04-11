fn main() {
    println!("cargo:rerun-if-changed=icon.ico");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("icon.ico");
        if let Err(err) = resource.compile() {
            panic!("failed to compile Windows resources: {err}");
        }
    }
}
