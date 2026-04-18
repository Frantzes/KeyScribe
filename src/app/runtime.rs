use super::*;

impl KeyScribeApp {
    pub(super) fn lock_startup_min_window_size_once(&mut self, ctx: &egui::Context) {
        if self.startup_min_window_size_locked {
            return;
        }

        let viewport = ctx.input(|i| i.viewport().clone());
        let Some(inner_rect) = viewport.inner_rect else {
            return;
        };

        let size = inner_rect.size();
        if size.x <= 1.0 || size.y <= 1.0 {
            return;
        }

        let sized_like_fullscreen = viewport
            .monitor_size
            .map(|monitor| size.x >= monitor.x * 0.9 && size.y >= monitor.y * 0.9)
            .unwrap_or(false);

        let should_lock = viewport.maximized.unwrap_or(false)
            || viewport.fullscreen.unwrap_or(false)
            || sized_like_fullscreen;

        if should_lock {
            ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(size));
            self.startup_min_window_size_locked = true;
        }
    }

    pub(super) fn is_touch_platform(&self) -> bool {
        false
    }

    pub(super) fn apply_mobile_ui_tweaks_once(&mut self, ctx: &egui::Context) {
        if !self.is_touch_platform() || self.mobile_ui_tweaks_applied {
            return;
        }

        let mut style = (*ctx.style()).clone();
        style.spacing.interact_size.x = style.spacing.interact_size.x.max(42.0);
        style.spacing.interact_size.y = style.spacing.interact_size.y.max(42.0);
        style.spacing.slider_width = style.spacing.slider_width.max(176.0);
        style.spacing.item_spacing.x = style.spacing.item_spacing.x.max(10.0);
        style.spacing.item_spacing.y = style.spacing.item_spacing.y.max(10.0);

        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(18.0));
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(17.0));

        ctx.set_style(style);
        self.mobile_ui_tweaks_applied = true;
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

        if processed_sample_len.saturating_sub(self.loading_last_waveform_rebuild_samples)
            >= STREAMING_WAVEFORM_REBUILD_SAMPLE_DELTA
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

    pub(super) fn cancel_active_processing(&mut self) {
        let cancel_epoch = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1);
        self.processing_epoch.store(cancel_epoch, Ordering::Release);
        self.clear_processing_job();
        self.pending_param_change = false;
        self.last_param_change_at = None;
        self.queued_param_update = false;
        self.restart_playback_after_processing = false;
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
            self.request_rebuild(was_playing, RebuildMode::Full);
            return;
        }

        let was_playing = self.stop_if_playing();
        if was_playing {
            self.request_rebuild(true, RebuildMode::ParametersPreview);
        } else {
            self.request_rebuild(false, RebuildMode::ParametersOnly);
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

        for (smoothed, current) in self
            .note_probs_smoothed
            .iter_mut()
            .zip(self.note_probs.iter())
        {
            *smoothed = *smoothed * 0.78 + *current * 0.22;
        }

        self.last_prob_update = Instant::now();
    }

    pub(super) fn clear_processing_job(&mut self) {
        self.is_processing = false;
        self.processing_rx = None;
        self.active_job_id = None;
        self.active_rebuild_mode = RebuildMode::Full;
        self.processing_started_at = None;
        self.processing_estimated_total_sec = 0.0;
        self.processing_audio_duration_sec = 0.0;
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
                    if let Some(raw) = &self.audio_raw {
                        if let Some(engine) = &mut self.engine {
                            if let Err(err) = engine.play_chunk_at_timeline(
                                &preview.samples,
                                preview.channels,
                                raw.sample_rate,
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

        self.processed_samples = result.processed_samples;
        self.processed_playback_samples = result.processed_playback_samples;
        self.processed_playback_channels = result.processed_playback_channels;
        self.set_waveform_data(result.waveform, true);
        self.note_timeline = result.note_timeline;
        self.note_timeline_step_sec = result.note_timeline_step_sec;
        self.base_note_timeline = result.base_note_timeline;
        self.base_note_timeline_step_sec = result.base_note_timeline_step_sec;
        self.clear_processing_job();
        self.selected_time_sec = self.selected_time_sec.min(self.source_duration());

        if let Some(err) = result.analysis_error {
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
                    self.play_range(source_pos, Some(loop_end));
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

        if pointer_down && !debounce_elapsed {
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

    pub(super) fn source_to_output_time(&self, source_sec: f32) -> f32 {
        source_sec / self.playback_rate()
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

        if !is_supported_audio_extension(path.as_path()) {
            return Err("Unsupported audio format. Use wav, mp3, flac, ogg, m4a, or aac.".into());
        }
        if !path.is_file() {
            return Err(format!("Audio file not found: {}", path.display()));
        }

        self.manual_import_path = path.to_string_lossy().to_string();
        self.start_audio_loading(path, ctx);
        Ok(())
    }

    #[cfg(not(feature = "desktop-ui"))]
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
        let picked = FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "aac"])
            .pick_file();

        if let Some(path) = picked {
            if let Err(err) = self.start_audio_loading_from_path(path.to_path_buf(), ctx) {
                self.last_error = Some(err);
            }
        }
    }

    #[cfg(not(feature = "desktop-ui"))]
    pub(super) fn import_audio_with_ctx(&mut self, ctx: &egui::Context) {
        self.import_audio_from_manual_path(ctx);
    }

    #[allow(dead_code)]
    pub(super) fn apply_loaded_audio(
        &mut self,
        path: PathBuf,
        audio: AudioData,
        ctx: &egui::Context,
    ) {
        self.cancel_active_processing();
        self.loaded_audio_hash = None;
        self.loaded_path = Some(path);
        self.loading_preview_cache.clear();
        self.selected_time_sec = 0.0;
        self.audio_raw = Some(audio);
        self.note_timeline = Arc::new(Vec::new());
        self.note_timeline_step_sec = 0.0;
        self.base_note_timeline = Arc::new(Vec::new());
        self.base_note_timeline_step_sec = 0.0;
        if let Some(raw) = &self.audio_raw {
            self.processed_playback_channels = raw.channels.max(1);
            if speed_pitch_is_identity(self.speed, self.pitch_semitones) {
                self.processed_samples = raw.samples_mono.as_ref().to_vec();
                self.processed_playback_samples = raw.samples_interleaved.as_ref().to_vec();
                let waveform = build_waveform_for_processed(
                    self.processed_samples.as_slice(),
                    raw.sample_rate,
                    self.audio_quality_mode.waveform_points(),
                    1.0,
                );
                self.set_waveform_data(waveform, false);
            } else {
                self.processed_samples.clear();
                self.processed_playback_samples.clear();
                self.clear_waveform_data();
            }
        }
        self.waveform_reset_view = true;
        self.playing_preview_buffer = false;
        self.live_stream_playback = false;
        self.album_art_texture = self.create_album_art_texture(ctx);
        self.update_note_probabilities(true);
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
