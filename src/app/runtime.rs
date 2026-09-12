use super::*;

/// One compact touch settings cell: a label, a small accent slider, and the
/// numeric value. The slider rail fills with the user's accent color up to the
/// current value, and the handle is small (like the seek bar).
fn compact_setting_slider(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    default: f32,
    bipolar: bool,
    decimals: usize,
    accent: egui::Color32,
) -> bool {
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(label).size(12.0));
        // Fill the column width so the two columns are evenly distributed.
        let width = (ui.available_width() - 12.0).clamp(90.0, 240.0);
        let changed = crate::ui::widgets::accent_slider(
            ui,
            id,
            value,
            min,
            max,
            default,
            bipolar,
            egui::vec2(width, 16.0),
            accent,
        );
        ui.label(egui::RichText::new(format!("{:.*}", decimals, *value)).size(11.0));
        changed
    })
    .inner
}

impl KeyScribeApp {
    pub(super) fn lock_startup_min_window_size_once(&mut self, _ctx: &egui::Context) {
        if self.startup_min_window_size_locked {
            return;
        }

        // Avoid locking min size to a maximized startup viewport.
        // That made the app appear non-resizable after restore/unmaximize.
        self.startup_min_window_size_locked = true;
    }

    pub(super) fn is_touch_platform(&self) -> bool {
        cfg!(any(target_os = "android", target_os = "ios"))
    }

    pub(super) fn apply_mobile_ui_tweaks_once(&mut self, ctx: &egui::Context) {
        if !self.is_touch_platform() || self.mobile_ui_tweaks_applied {
            return;
        }

        let mut style = (*ctx.style()).clone();
        // Touch-friendly hit targets, kept compact: the top controls and
        // progress rows otherwise eat a lot of vertical space, and the slider
        // handle radius is derived from the widget height.
        style.spacing.interact_size.x = style.spacing.interact_size.x.max(34.0);
        style.spacing.interact_size.y = style.spacing.interact_size.y.max(26.0);
        style.spacing.slider_width = style.spacing.slider_width.max(140.0);
        style.spacing.item_spacing.x = style.spacing.item_spacing.x.max(6.0);
        style.spacing.item_spacing.y = style.spacing.item_spacing.y.max(UI_VSPACE_TIGHT);

        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(13.0));
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(12.0));
        style
            .text_styles
            .insert(egui::TextStyle::Small, egui::FontId::proportional(10.5));

        // Touch has no hover, so tooltips only pop up accidentally on taps.
        // A huge delay makes egui never consider them ready to show.
        style.interaction.tooltip_delay = 1.0e9;

        ctx.set_style(style);
        self.mobile_ui_tweaks_applied = true;
    }

    /// Compact keyboard-settings panel for touch layouts: a 2x2 grid of small
    /// sliders inside a bounded scroll area, so opening the cog never pushes
    /// the piano or media controls off-screen.
    pub(super) fn draw_keyboard_settings_compact(&mut self, ui: &mut egui::Ui) {
        let max_h = (ui.ctx().screen_rect().height() * 0.30).max(104.0);
        let mut visuals_changed = false;
        let accent = self.highlight_color;

        egui::ScrollArea::vertical()
            .max_height(max_h)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(10.0, 8.0);

                // Two even columns (via `columns`) so the sliders spread across
                // the full width instead of packing to the left.
                ui.columns(2, |cols| {
                    // Narrower, precise ranges tuned for touch, with the
                    // accent-colored fill showing how far each has moved.
                    visuals_changed |= compact_setting_slider(
                        &mut cols[0],
                        ("kb_key_sens", 0),
                        "Key Sensitivity",
                        &mut self.key_color_sensitivity,
                        0.0,
                        0.55,
                        default_key_color_sensitivity(),
                        false,
                        2,
                        accent,
                    );
                    visuals_changed |= compact_setting_slider(
                        &mut cols[1],
                        ("kb_highlight", 1),
                        "Highlight Time (s)",
                        &mut self.key_highlight_max_sec,
                        0.0,
                        0.3,
                        default_key_highlight_max_sec(),
                        false,
                        2,
                        accent,
                    );
                    visuals_changed |= compact_setting_slider(
                        &mut cols[0],
                        ("kb_vis_offset", 2),
                        "Vis Offset (ms)",
                        &mut self.visualization_timing_offset_ms,
                        -100.0,
                        100.0,
                        default_visualization_timing_offset_ms(),
                        true,
                        0,
                        accent,
                    );
                    visuals_changed |= compact_setting_slider(
                        &mut cols[1],
                        ("kb_piano_zoom", 3),
                        "Piano Zoom",
                        &mut self.piano_zoom,
                        PIANO_ZOOM_MIN,
                        PIANO_ZOOM_MAX,
                        1.0,
                        false,
                        2,
                        accent,
                    );
                });
            });

        if visuals_changed {
            self.update_note_probabilities(true);
        }
    }

    pub(super) fn is_playing(&self) -> bool {
        self.engine
            .as_ref()
            .map(|e| e.is_playing())
            .unwrap_or(false)
    }

    pub(super) fn current_position_sec(&self) -> f32 {
        self.engine
            .as_ref()
            .map(|e| e.current_position())
            .unwrap_or(0.0)
    }

    pub(super) fn invalidate_waveform_cache(&mut self) {
        self.waveform_version = self.waveform_version.wrapping_add(1);
        self.loop_waveform_cache_version = u64::MAX;
        self.loop_waveform_cache_selection = None;
        self.loop_waveform_cache_pre.clear();
        self.loop_waveform_cache_mid.clear();
        self.loop_waveform_cache_post.clear();
    }

    pub(super) fn set_waveform_data(&mut self, waveform: Vec<[f64; 2]>, reset_view: bool) {
        self.waveform = waveform;
        self.invalidate_waveform_cache();
        if reset_view {
            self.waveform_reset_view = true;
        }
    }

    pub(super) fn clear_waveform_data(&mut self) {
        self.waveform.clear();
        self.invalidate_waveform_cache();
    }

    pub(super) fn should_rebuild_streaming_waveform(&self, processed_sample_len: usize) -> bool {
        if self.waveform.is_empty() {
            return true;
        }

        let delta = adaptive_rebuild_delta(
            STREAMING_WAVEFORM_REBUILD_SAMPLE_DELTA,
            self.loading_total_samples,
            self.loading_sample_rate,
        );

        if processed_sample_len.saturating_sub(self.loading_last_waveform_rebuild_samples)
            >= delta
        {
            return true;
        }

        self.loading_last_waveform_rebuild_at
            .map(|at| at.elapsed() >= STREAMING_WAVEFORM_REBUILD_INTERVAL)
            .unwrap_or(true)
    }

    pub(super) fn mark_streaming_waveform_rebuild(&mut self, processed_sample_len: usize) {
        self.loading_last_waveform_rebuild_at = Some(Instant::now());
        self.loading_last_waveform_rebuild_samples = processed_sample_len;
    }

    pub(super) fn refresh_loop_waveform_cache(&mut self, start_sec: f32, end_sec: f32) {
        let cache_key = Some((start_sec, end_sec));
        if self.loop_waveform_cache_version == self.waveform_version
            && self.loop_waveform_cache_selection == cache_key
        {
            return;
        }

        self.loop_waveform_cache_pre.clear();
        self.loop_waveform_cache_mid.clear();
        self.loop_waveform_cache_post.clear();

        let start = start_sec as f64;
        let end = end_sec as f64;
        for &pt in &self.waveform {
            if pt[0] < start {
                self.loop_waveform_cache_pre.push(pt);
            } else if pt[0] <= end {
                self.loop_waveform_cache_mid.push(pt);
            } else {
                self.loop_waveform_cache_post.push(pt);
            }
        }

        self.loop_waveform_cache_selection = cache_key;
        self.loop_waveform_cache_version = self.waveform_version;
    }

    pub(super) fn stop_if_playing(&mut self) -> bool {
        let was_playing = self.is_playing();
        if was_playing {
            self.stop();
        }
        was_playing
    }

    pub(super) fn request_rebuild_preserving_playback(&mut self) {
        if self.is_audio_loading {
            return;
        }

        let was_playing = self.stop_if_playing();
        self.request_rebuild(was_playing, RebuildMode::Full);
    }

    pub(super) fn request_rebuild_preserving_playback_and_waveform(&mut self) {
        if self.is_audio_loading {
            return;
        }

        self.request_rebuild(false, RebuildMode::VisualizationOnly);
    }

    pub(super) fn cancel_active_processing(&mut self) {
        let cancel_epoch = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1);
        self.processing_epoch.store(cancel_epoch, Ordering::Release);
        if let Some(flag) = &self.processing_cancel_flag {
            flag.store(true, Ordering::Release);
        }
        self.clear_processing_job();
        self.pending_param_change = false;
        self.last_param_change_at = None;
        self.queued_param_update = false;
        self.restart_playback_after_processing = false;
        self.cancel_streaming_stretch();
    }

    pub(super) fn refresh_audio_output_devices(&mut self) {
        self.audio_output_devices = available_output_devices();
        if let Some(selected) = self.audio_output_device_id.as_deref() {
            let exists = self.audio_output_devices.iter().any(|d| d.id == selected);
            if !exists {
                self.audio_output_device_id = None;
            }
        }
    }

    pub(super) fn apply_audio_output_device_change(&mut self, device_id: Option<String>) {
        if self.audio_output_device_id == device_id {
            return;
        }

        let was_playing = self.is_playing();
        let resume_pos = self.current_position_sec().min(self.source_duration());
        self.stop();

        match AudioEngine::new_with_output_device(device_id.as_deref()) {
            Ok(mut engine) => {
                engine.set_volume(self.playback_volume);
                self.engine = Some(engine);
                self.audio_output_device_id = device_id;
                self.last_error = None;

                if was_playing && !self.processed_playback_samples.is_empty() {
                    self.selected_time_sec = resume_pos;
                    self.play_from_selected();
                }
            }
            Err(err) => {
                self.last_error = Some(format!("Audio device error: {err}"));
                self.engine = AudioEngine::new().ok();
                if let Some(engine) = &mut self.engine {
                    engine.set_volume(self.playback_volume);
                }
                self.audio_output_device_id = None;
            }
        }
    }

    pub(super) fn cancel_streaming_stretch(&mut self) {
        if let Some(state) = self.streaming_stretch.take() {
            state.cancel.store(true, Ordering::Release);
        }
    }

    pub(super) fn request_param_update_preserving_playback(&mut self) {
        self.refresh_timeline_for_current_params();

        if self.is_audio_loading {
            return;
        }

        // Parameter-only rebuilds rely on an existing analyzed base timeline.
        // If analysis is not ready yet, force a full rebuild to avoid ending up
        // with an empty timeline state that blocks playback controls.
        let needs_full_rebuild = self.preprocess_audio
            && (self.base_note_timeline.is_empty() || self.base_note_timeline_step_sec <= 0.0);
        if needs_full_rebuild {
            let was_playing = self.stop_if_playing();
            self.param_tweak_rebuild = true;
            self.request_rebuild(was_playing, RebuildMode::Full);
            return;
        }

        // Cancel any in-flight processing and streaming — we're switching modes
        self.cancel_active_processing();
        self.cancel_streaming_stretch();

        let was_playing = self.is_playing();
        let resume_pos = self.current_position_sec().min(self.source_duration());

        // Swap buffers to raw samples. For identity, this is the final state.
        // For non-identity, streaming will stretch from raw in real-time.
        if let Some(raw) = &self.audio_raw {
            self.processed_samples = raw.samples_mono.to_vec();
            self.processed_playback_samples = Arc::clone(&raw.samples_interleaved);
            self.processed_playback_channels = raw.channels;

            // The waveform is in source-time coordinates (0..source_duration),
            // which doesn't change with speed — streaming uses timeline_rate =
            // speed so the clock maps back to source time. Skip the expensive
            // rebuild (full-sample scan + peak-normalize) on speed changes.
            if self.waveform.is_empty() {
                let waveform = build_waveform_for_processed(
                    &self.processed_samples,
                    raw.sample_rate,
                    self.audio_quality_mode.waveform_points(),
                    1.0,
                );
                self.set_waveform_data(waveform, false);
            }
        }

        self.note_timeline = Arc::clone(&self.base_note_timeline);
        self.note_timeline_step_sec = self.base_note_timeline_step_sec;
        self.selected_time_sec = resume_pos;

        if !was_playing {
            return;
        }

        if speed_pitch_is_identity(self.speed, self.pitch_semitones) {
            // Identity: play raw samples directly — instant, preserves stereo
            self.play_from_selected();
        } else {
            // Non-identity: start streaming time-stretch from current position.
            // First chunk arrives in <10ms, so playback starts almost instantly.
            // Pitch is preserved, channels are preserved, no UI freeze.
            self.start_streaming_playback(resume_pos);
        }
    }

    pub(super) fn refresh_timeline_for_current_params(&mut self) {
        if !self.preprocess_audio {
            return;
        }
        if self.base_note_timeline.is_empty() || self.base_note_timeline_step_sec <= 0.0 {
            return;
        }

        let idx = (self.selected_time_sec.max(0.0) / self.base_note_timeline_step_sec) as usize;
        let idx = idx.min(self.base_note_timeline.len().saturating_sub(1));
        let frame = &self.base_note_timeline[idx];

        self.note_probs = if self.pitch_semitones.abs() < 1.0e-6 {
            frame.clone()
        } else {
            Self::transpose_frame(frame, self.pitch_semitones)
        };

        let elapsed_sec = self.last_prob_update.elapsed().as_secs_f32();
        self.update_note_highlight_visuals(elapsed_sec);

        self.last_prob_update = Instant::now();
    }

    pub(super) fn clear_processing_job(&mut self) {
        self.is_processing = false;
        self.param_tweak_rebuild = false;
        self.processing_rx = None;
        self.active_job_id = None;
        self.active_rebuild_mode = RebuildMode::Full;
        self.processing_started_at = None;
        self.processing_estimated_total_sec = 0.0;
        self.processing_audio_duration_sec = 0.0;
        self.processing_cancel_flag = None;
    }

    pub(super) fn apply_processing_result(&mut self, result: ProcessingResult) {
        if let Some(song_hash) = result.source_hash.as_ref() {
            self.loaded_audio_hash = Some(song_hash.clone());
        }

        if let Some(cache_hit) = result.cache_lookup_hit {
            self.cache_status_message = Some(if cache_hit {
                "Analysis cache: loaded from cache.".to_string()
            } else {
                "Analysis cache: miss, rendering new analysis.".to_string()
            });
            self.cache_status_message_at = Some(Instant::now());
        }

        if result.mode == RebuildMode::Full && self.preprocess_audio {
            if let Some(started_at) = self.processing_started_at {
                let elapsed = started_at.elapsed().as_secs_f32();
                let audio_sec = self.processing_audio_duration_sec.max(1.0e-3);

                // Ignore near-instant jobs (usually cache hits) so ETA learning stays realistic.
                if elapsed > 0.2 {
                    let observed = (elapsed / audio_sec).clamp(0.02, 4.0);
                    self.analysis_seconds_per_audio_second_ema = Some(
                        self.analysis_seconds_per_audio_second_ema
                            .map(|prev| prev * 0.7 + observed * 0.3)
                            .unwrap_or(observed),
                    );
                }
            }
        }

        if result.mode == RebuildMode::ParametersPreview {
            if self.queued_param_update {
                self.queued_param_update = false;
                self.clear_processing_job();
                self.request_param_update_preserving_playback();
                return;
            }

            self.clear_processing_job();

            if self.restart_playback_after_processing {
                self.restart_playback_after_processing = false;
                if let Some(preview) = result.preview_playback {
                    let playback_rate = self.playback_rate();
                    if self.audio_raw.is_some() {
                        if let Some(engine) = &mut self.engine {
                            if let Err(err) = engine.play_chunk_at_timeline(
                                &preview.samples,
                                preview.channels,
                                preview.sample_rate,
                                preview.timeline_start_sec,
                                playback_rate,
                            ) {
                                self.last_error = Some(format!("Playback error: {err}"));
                                self.live_stream_playback = false;
                            } else {
                                self.playing_preview_buffer = true;
                                self.live_stream_playback = false;
                            }
                        }
                    }
                }
            }

            // Continue with full render in the background so seeking and waveform stay accurate.
            self.request_rebuild(false, RebuildMode::ParametersOnly);
            return;
        }

        if result.mode == RebuildMode::ParametersOnly && self.queued_param_update {
            self.queued_param_update = false;
            self.clear_processing_job();
            self.request_param_update_preserving_playback();
            return;
        }

        let handoff_pos = if result.mode == RebuildMode::ParametersOnly
            && self.playing_preview_buffer
            && self.is_playing()
        {
            Some(self.current_position_sec())
        } else {
            None
        };
        let handoff_loop_end = if self.loop_enabled && self.loop_playback_enabled {
            self.loop_selection.map(|(a, b)| a.max(b))
        } else {
            None
        };

        if result.mode != RebuildMode::VisualizationOnly {
            self.processed_samples = result.processed_samples;
            self.processed_playback_samples = result.processed_playback_samples;
            self.processed_playback_channels = result.processed_playback_channels;

            // Only reset waveform view on initial Full load, not on background param updates.
            let reset_view = result.mode == RebuildMode::Full && self.waveform.is_empty();
            self.set_waveform_data(result.waveform, reset_view);
        }

        self.note_timeline = result.note_timeline;
        self.note_timeline_step_sec = result.note_timeline_step_sec;
        self.base_note_timeline = result.base_note_timeline;
        self.base_note_timeline_step_sec = result.base_note_timeline_step_sec;
        self.clear_processing_job();
        self.selected_time_sec = self.selected_time_sec.min(self.source_duration());

        if let Some(err) = result.analysis_error {
            // Mirror to stderr: otherwise headless/terminal diagnosis of a
            // failed transcription is impossible (the message only shows in
            // the UI's error label).
            eprintln!("[keyscribe] analysis failed: {err}");
            self.last_error = Some(err);
        }
        self.update_note_probabilities(true);

        if result.mode == RebuildMode::Full && self.queued_param_update {
            self.queued_param_update = false;
            self.request_param_update_preserving_playback();
            return;
        }

        if self.restart_playback_after_processing {
            self.restart_playback_after_processing = false;
            self.play_from_selected();
        } else if let Some(source_pos) = handoff_pos {
            if let Some(loop_end) = handoff_loop_end {
                if loop_end - source_pos > LOOP_MIN_DURATION_SEC {
                    self.play_range(source_pos, None);
                } else {
                    self.play_from_selected();
                }
            } else {
                self.play_range(source_pos, None);
            }
            self.selected_time_sec = source_pos.min(self.source_duration());
            self.playing_preview_buffer = false;
        } else {
            self.playing_preview_buffer = false;
        }
    }

    pub(super) fn maybe_commit_pending_param_change(&mut self, pointer_down: bool) {
        if !self.pending_param_change {
            return;
        }

        let debounce_elapsed = self
            .last_param_change_at
            .map(|at| at.elapsed() >= PARAM_UPDATE_LIVE_DEBOUNCE)
            .unwrap_or(!pointer_down);

        if pointer_down {
            self.refresh_timeline_for_current_params();
            return;
        }

        if !debounce_elapsed {
            return;
        }

        // Never restart an in-flight full transcription because of speed/pitch edits.
        // Defer the parameter-only render until the baseline (1.0x / 0 st) timeline is ready.
        if self.is_processing && self.active_rebuild_mode == RebuildMode::Full {
            self.queued_param_update = true;
            self.pending_param_change = false;
            self.last_param_change_at = None;
            return;
        }

        self.refresh_timeline_for_current_params();

        if self.is_param_render_in_progress() {
            self.queued_param_update = true;
        } else {
            self.request_param_update_preserving_playback();
        }

        self.pending_param_change = false;
        self.last_param_change_at = None;
    }

    pub(super) fn is_blocking_processing(&self) -> bool {
        self.is_processing
            && self.active_rebuild_mode == RebuildMode::Full
            && self.processed_samples.is_empty()
    }

    pub(super) fn is_param_render_in_progress(&self) -> bool {
        self.is_processing
            && matches!(
                self.active_rebuild_mode,
                RebuildMode::ParametersOnly | RebuildMode::ParametersPreview
            )
    }

    pub(super) fn playback_rate(&self) -> f32 {
        self.speed.clamp(0.25, 4.0)
    }

    pub(super) fn source_duration(&self) -> f32 {
        if let Some(audio) = &self.audio_raw {
            if audio.sample_rate > 0 {
                let duration = audio.samples_mono.len() as f32 / audio.sample_rate as f32;
                if duration > 0.0 || !self.is_audio_loading {
                    return duration;
                }
            }
        }

        if self.is_audio_loading && self.loading_sample_rate > 0 {
            return self.loading_decoded_samples as f32 / self.loading_sample_rate as f32;
        }

        0.0
    }

    pub(super) fn timeline_duration_sec(&self) -> f32 {
        if self.is_audio_loading
            && (self.loading_cache_waveform_preloaded || self.loading_cache_timeline_preloaded)
        {
            self.waveform_view_duration().max(0.0)
        } else {
            self.source_duration().max(0.0)
        }
    }

    pub(super) fn play_preview_at(&mut self, start_sec: f32, end_sec: Option<f32>) -> bool {
        if !self.is_param_render_in_progress() {
            return false;
        }

        let _ = end_sec;
        self.selected_time_sec = start_sec.max(0.0);

        if self.active_rebuild_mode == RebuildMode::ParametersPreview {
            self.restart_playback_after_processing = true;
        } else if self.restart_playback_after_processing {
            // A preview handoff is already queued; do not restart the worker again.
            return true;
        } else {
            self.request_rebuild(true, RebuildMode::ParametersPreview);
        }

        true
    }

    pub(super) fn start_audio_loading_from_path(
        &mut self,
        input_path: PathBuf,
        ctx: &egui::Context,
    ) -> Result<(), String> {
        let path = if input_path.is_absolute() {
            input_path
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(&input_path),
                Err(_) => input_path,
            }
        };

        if !super::is_supported_media_extension(path.as_path()) {
            return Err("Unsupported media format. Use wav, mp3, flac, ogg, m4a, aac, mp4, mkv, avi, mov, or webm.".into());
        }
        if !path.is_file() {
            return Err(format!("Audio file not found: {}", path.display()));
        }

        self.manual_import_path = path.to_string_lossy().to_string();
        self.start_audio_loading(path, ctx, true);
        Ok(())
    }

    #[cfg(not(feature = "desktop-ui"))]
    #[allow(dead_code)] // Android uses the native picker instead.
    pub(super) fn import_audio_from_manual_path(&mut self, ctx: &egui::Context) {
        let path = self.manual_import_path.trim();
        if path.is_empty() {
            self.last_error = Some("Enter an audio file path before opening.".to_string());
            return;
        }

        match self.start_audio_loading_from_path(PathBuf::from(path), ctx) {
            Ok(()) => {
                self.last_error = None;
            }
            Err(err) => {
                self.last_error = Some(err);
            }
        }
    }

    #[cfg(feature = "desktop-ui")]
    pub(super) fn import_audio_with_ctx(&mut self, ctx: &egui::Context) {
        self.spawn_file_dialog(ctx, FileDialogRequest::OpenAudio);
    }

    #[cfg(all(not(feature = "desktop-ui"), target_os = "android"))]
    pub(super) fn import_audio_with_ctx(&mut self, ctx: &egui::Context) {
        let _ = ctx;
        // Native Storage Access Framework picker; the result arrives via the
        // `MainActivity` JNI callback and is polled in `update`.
        crate::android::pick_audio_file();
    }

    #[cfg(all(not(feature = "desktop-ui"), not(target_os = "android")))]
    pub(super) fn import_audio_with_ctx(&mut self, ctx: &egui::Context) {
        self.import_audio_from_manual_path(ctx);
    }

    pub(super) fn create_album_art_texture(
        &self,
        ctx: &egui::Context,
    ) -> Option<egui::TextureHandle> {
        let bytes = self
            .audio_raw
            .as_ref()
            .and_then(|a| a.metadata.artwork_bytes.as_deref())?;
        if bytes.len() > ALBUM_ART_MAX_BYTES {
            return None;
        }

        let image = image::load_from_memory(bytes).ok()?.to_rgba8();
        let width = image.width() as usize;
        let height = image.height() as usize;
        if width == 0
            || height == 0
            || width > ALBUM_ART_MAX_DIMENSION
            || height > ALBUM_ART_MAX_DIMENSION
        {
            return None;
        }

        let size = [width, height];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        Some(ctx.load_texture("album-art", color_image, egui::TextureOptions::LINEAR))
    }
}

#[cfg(feature = "desktop-ui")]
pub(super) enum FileDialogRequest {
    OpenAudio,
    ExportStemsFolder,
    ExportMidiFolder,
    SaveMusicXml(String),
    SavePdf(String),
}

#[cfg(feature = "desktop-ui")]
pub(super) enum FileDialogResult {
    OpenAudio(Option<PathBuf>),
    ExportStemsFolder(Option<PathBuf>),
    ExportMidiFolder(Option<PathBuf>),
    SaveMusicXml(Option<PathBuf>),
    SavePdf(Option<PathBuf>),
}

#[cfg(feature = "desktop-ui")]
impl KeyScribeApp {
    /// Open a native file dialog without blocking the event loop.
    ///
    /// The portal file chooser (`xdg-desktop-portal` / Zenity fallback) can
    /// stay open for minutes, and its synchronous API parks the calling
    /// thread the whole time. Called on the UI thread that wedges the event
    /// loop: compositor pings go unanswered and the desktop raises
    /// "Application Not Responding" after a few seconds of browsing files.
    /// Instead the dialog runs on a worker thread (via
    /// `rfd::AsyncFileDialog`) and the choice is applied in
    /// [`Self::poll_file_dialog_result`]. One dialog at a time; extra
    /// requests while one is open are ignored.
    pub(super) fn spawn_file_dialog(&mut self, ctx: &egui::Context, request: FileDialogRequest) {
        if self.file_dialog_rx.is_some() {
            return;
        }
        const MEDIA_EXTENSIONS: &[&str] = &[
            "wav", "mp3", "flac", "ogg", "m4a", "aac", "mp4", "mkv", "avi", "mov", "webm",
        ];
        let (tx, rx) = mpsc::channel::<FileDialogResult>();
        self.file_dialog_rx = Some(rx);
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = match request {
                FileDialogRequest::OpenAudio => FileDialogResult::OpenAudio(
                    pollster::block_on(
                        rfd::AsyncFileDialog::new()
                            .add_filter("Media", MEDIA_EXTENSIONS)
                            .pick_file(),
                    )
                    .map(|handle| handle.path().to_path_buf()),
                ),
                FileDialogRequest::ExportStemsFolder => {
                    FileDialogResult::ExportStemsFolder(
                        pollster::block_on(rfd::AsyncFileDialog::new().pick_folder())
                            .map(|handle| handle.path().to_path_buf()),
                    )
                }
                FileDialogRequest::ExportMidiFolder => FileDialogResult::ExportMidiFolder(
                    pollster::block_on(rfd::AsyncFileDialog::new().pick_folder())
                        .map(|handle| handle.path().to_path_buf()),
                ),
                FileDialogRequest::SaveMusicXml(stem) => FileDialogResult::SaveMusicXml(
                    pollster::block_on(
                        rfd::AsyncFileDialog::new()
                            .add_filter("MusicXML", &["musicxml", "xml"])
                            .set_file_name(&format!("{stem}.musicxml"))
                            .save_file(),
                    )
                    .map(|handle| handle.path().to_path_buf()),
                ),
                FileDialogRequest::SavePdf(stem) => FileDialogResult::SavePdf(
                    pollster::block_on(
                        rfd::AsyncFileDialog::new()
                            .add_filter("PDF", &["pdf"])
                            .set_file_name(&format!("{stem}.pdf"))
                            .save_file(),
                    )
                    .map(|handle| handle.path().to_path_buf()),
                ),
            };
            let _ = tx.send(result);
            repaint.request_repaint();
        });
    }

    /// Apply a finished native file dialog, if any. Called once per frame;
    /// never blocks (single `try_recv`).
    pub(super) fn poll_file_dialog_result(&mut self, ctx: &egui::Context) {
        let result = match &self.file_dialog_rx {
            Some(rx) => match rx.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.file_dialog_rx = None;
                    return;
                }
            },
            None => return,
        };
        self.file_dialog_rx = None;
        match result {
            FileDialogResult::OpenAudio(Some(path)) => {
                match self.start_audio_loading_from_path(path, ctx) {
                    Ok(()) => {
                        self.last_error = None;
                    }
                    Err(err) => {
                        self.last_error = Some(err);
                    }
                }
            }
            FileDialogResult::OpenAudio(None) => {}
            FileDialogResult::ExportStemsFolder(Some(folder)) => {
                self.execute_export_stems(folder.as_path());
            }
            FileDialogResult::ExportStemsFolder(None) => {}
            FileDialogResult::ExportMidiFolder(Some(folder)) => {
                self.execute_export_midi(folder.as_path());
            }
            FileDialogResult::ExportMidiFolder(None) => {}
            FileDialogResult::SaveMusicXml(Some(path)) => {
                self.finish_musicxml_export(path.as_path());
            }
            FileDialogResult::SaveMusicXml(None) => {}
            FileDialogResult::SavePdf(Some(path)) => {
                self.finish_pdf_export(path.as_path());
            }
            FileDialogResult::SavePdf(None) => {}
        }
    }
}
