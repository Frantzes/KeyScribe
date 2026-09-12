use super::*;

impl KeyScribeApp {
    /// Refresh the audio list (and media permission state) and show the
    /// in-app audio browser. Android's activity-result API is not available
    /// through `android-activity` 0.5, so this replaces a SAF picker.
    pub(super) fn android_open_picker(&mut self) {
        self.android_media_permission = crate::android::media_permission_granted();
        if !self.android_media_permission {
            crate::android::request_media_permission();
        }
        self.android_picker_entries = crate::android::scan_audio_files();
        self.android_picker_status = if !self.android_media_permission {
            Some("Allow audio access to see files in Music/Download.".to_string())
        } else if self.android_picker_entries.is_empty() {
            Some("No audio files found in Music, Download or the app folder.".to_string())
        } else {
            None
        };
        self.android_picker_open = true;
    }

    pub(super) fn draw_android_import(&mut self, ctx: &egui::Context) {
        if !self.android_picker_open {
            return;
        }

        let mut open = true;
        let mut chosen: Option<String> = None;
        let mut request_again = false;
        let mut rescan = false;

        egui::Window::new("Open Audio")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Allow audio access").clicked() {
                        request_again = true;
                    }
                    if ui.button("Rescan").clicked() {
                        rescan = true;
                    }
                });
                if let Some(status) = &self.android_picker_status {
                    ui.colored_label(ERROR_RED, status);
                }
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(460.0)
                    .show(ui, |ui| {
                        for (name, path) in &self.android_picker_entries {
                            if ui.button(name).on_hover_text(path).clicked() {
                                chosen = Some(path.clone());
                            }
                        }
                    });
            });

        if request_again {
            crate::android::request_media_permission();
            rescan = true;
        }
        if rescan {
            self.android_media_permission = crate::android::media_permission_granted();
            self.android_picker_entries = crate::android::scan_audio_files();
            self.android_picker_status = if self.android_picker_entries.is_empty() {
                Some("No audio files found. Grant audio access and tap Rescan.".to_string())
            } else {
                None
            };
        }

        if let Some(path) = chosen {
            self.android_picker_open = false;
            match self.start_audio_loading_from_path(std::path::PathBuf::from(path), ctx) {
                Ok(()) => self.last_error = None,
                Err(err) => self.last_error = Some(err),
            }
        }
        if !open {
            self.android_picker_open = false;
        }
    }
}
