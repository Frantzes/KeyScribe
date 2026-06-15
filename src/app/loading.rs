use super::*;
use crate::core::processing::{
    build_waveform_for_processed, estimate_processing_duration_sec, CoreRebuildMode,
};

impl KeyScribeApp {
    pub(super) fn cancel_audio_loading(&mut self) {
        if let Some(flag) = self.audio_loading_cancel.take() {
            flag.store(true, Ordering::Release);
        }
        self.audio_loading_rx = None;
        self.is_audio_loading = false;
        self.live_stream_playback = false;
        self.loading_cache_timeline_preloaded = false;
        self.loading_cache_waveform_preloaded = false;
        self.loading_preview_cache.clear();
    }

    pub(super) fn clear_note_visuals(&mut self) {
        for value in &mut self.note_probs {
            *value = 0.0;
        }
        for value in &mut self.note_probs_smoothed {
            *value = 0.0;
        }
        for value in &mut self.note_highlight_hold_remaining {
            *value = 0.0;
        }
        if !self.note_stem_colors.is_empty() {
            self.note_stem_colors
                .fill(self.highlight_color);
        }
    }

    pub(super) fn note_visuals_ready(&self) -> bool {
        let has_preprocessed_timeline =
            !self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0;
        let has_stem_analyses = !self.stem_analyses.is_empty();

        if self.separated_stems.is_some() {
            return self.audio_raw.is_some() && (has_stem_analyses || has_preprocessed_timeline);
        }

        if self.is_audio_loading {
            return self.preprocess_audio
                && self.loading_cache_timeline_preloaded
                && has_preprocessed_timeline;
        }

        if self.preprocess_audio {
            has_preprocessed_timeline
                && (!self.is_processing || self.loading_cache_timeline_preloaded)
        } else {
            !self.is_processing && !self.processed_samples.is_empty()
        }
    }

    pub(super) fn waveform_view_duration(&self) -> f32 {
        if self.is_audio_loading && self.loading_sample_rate > 0 {
            if let Some(total_samples) = self.loading_total_samples {
                if total_samples > 0 {
                    return total_samples as f32 / self.loading_sample_rate as f32;
                }
            }
        }

        self.source_duration()
    }

    pub(super) fn estimate_processing_duration_sec(
        &self,
        mode: RebuildMode,
        raw_sample_len: usize,
        sample_rate: u32,
    ) -> f32 {
        let core_mode = match mode {
            RebuildMode::Full => CoreRebuildMode::Full,
            RebuildMode::ParametersOnly => CoreRebuildMode::ParametersOnly,
            RebuildMode::ParametersPreview => CoreRebuildMode::ParametersPreview,
            RebuildMode::VisualizationOnly => CoreRebuildMode::ParametersPreview,
        };

        let quality_multiplier = match self.audio_quality_mode {
            AudioQualityMode::Draft => 0.75,
            AudioQualityMode::Balanced => 1.0,
            AudioQualityMode::Studio => 1.35,
        };

        estimate_processing_duration_sec(
            core_mode,
            raw_sample_len,
            sample_rate,
            self.preprocess_audio,
            self.use_cqt_analysis,
            quality_multiplier,
            self.analysis_seconds_per_audio_second_ema,
        )
    }

    pub(super) fn start_audio_loading(
        &mut self,
        path: PathBuf,
        ctx: &egui::Context,
        reset_state: bool,
    ) {
        self.cancel_audio_loading();
        self.cancel_active_processing();
        self.stop();

        self.loaded_path = Some(path.clone());
        push_recent_file_path(&mut self.recent_file_paths, path.as_path());
        self.loaded_audio_hash = None;
        self.selected_time_sec = 0.0;
        self.audio_raw = None;
        self.processed_samples.clear();
        self.processed_playback_samples = Arc::new(Vec::new());
        self.processed_playback_channels = 1;
        self.clear_waveform_data();
        self.note_timeline = Arc::new(Vec::new());
        self.note_timeline_step_sec = 0.0;
        self.base_note_timeline = Arc::new(Vec::new());
        self.base_note_timeline_step_sec = 0.0;
        self.loading_source_channels = 1;
        self.loading_sample_rate = 0;
        self.loading_total_samples = None;
        self.loading_decoded_samples = 0;
        self.loading_last_waveform_rebuild_at = None;
        self.loading_last_waveform_rebuild_samples = 0;
        self.loading_provisional_timeline.clear();
        self.loading_next_transcribe_time_sec = 0.0;
        self.loading_timeline_frames_pending_sync = 0;
        self.album_art_texture = None;
        self.waveform_reset_view = true;
        self.playing_preview_buffer = false;
        self.live_stream_playback = false;
        self.last_error = None;
        self.cache_status_message = None;
        self.cache_status_message_at = None;
        self.cache_precheck_done = false;
        self.loading_cache_timeline_preloaded = false;
        self.loading_cache_waveform_preloaded = false;
        self.loading_preview_cache.clear();

        // Reset speed/pitch to identity when loading new audio.
        // This avoids needing separate raw-vs-processed audio buffers during streaming
        // and eliminates per-chunk DSP overhead. User can re-apply speed/pitch after load.
        self.speed = 1.0;
        self.pitch_semitones = 0.0;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if matches!(ext.as_str(), "mp4" | "mkv" | "avi" | "mov" | "webm") {
            self.video_player = Some(crate::app::video_player::VideoPlayer::new(path.to_string_lossy().to_string()));
        } else {
            self.video_player = None;
        }

        if reset_state {
            // Reset loop state for new audio
            self.loop_selection = None;
            self.loop_enabled = false;
            self.loop_playback_enabled = false;
            self.drag_select_anchor_sec = None;

            // Clear all stem state from previous song
            self.saved_visualize_stem_indices = None;
            self.saved_listen_stem_indices = None;
            self.pending_stem_indices.clear();
            self.pending_listening_indices.clear();
            self.show_visualize_selector = false;
            self.show_listen_selector = false;
            self.melody_stem_indices.clear();
            self.chord_stem_indices.clear();
            self.current_chord = None;
        }

        if self.preprocess_audio {
            self.cache_status_message = Some("Analysis cache: precheck pending...".to_string());
            self.cache_status_message_at = Some(Instant::now());
        }
        // Clear all stem state from previous song
        self.separated_stems = None;
        self.stem_analyses.clear();
        self.stem_colors.clear();
        self.stem_analysis_rx = None;
        self.is_separating = false;
        self.separation_attempted = false;
        self.separation_rx = None;
        self.enabled_stem_indices.clear();
        self.enabled_listening_indices.clear();
        self.stem_playback_cache = None;
        self.clear_note_visuals();

        let (tx, rx) = mpsc::channel::<StreamingAudioEvent>();
        let tx_error = tx.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let worker_path = path;

        let hash_tx = tx.clone();
        let hash_path = worker_path.clone();
        thread::spawn(move || {
            let source_hash = compute_file_hash(hash_path.as_path());
            let _ = hash_tx.send(StreamingAudioEvent::SourceHash(source_hash));
        });

        thread::spawn(move || {
            if let Err(err) = load_audio_file_streaming(
                worker_path.as_path(),
                AUDIO_STREAM_CHUNK_SAMPLES,
                worker_cancel,
                tx,
            ) {
                let _ = tx_error.send(StreamingAudioEvent::Error(format!(
                    "Failed to load audio: {err}"
                )));
            }
        });

        self.audio_loading_rx = Some(rx);
        self.audio_loading_cancel = Some(cancel);
        self.is_audio_loading = true;
        ctx.request_repaint();
    }

    pub(super) fn handle_audio_loading_event(
        &mut self,
        event: StreamingAudioEvent,
        ctx: &egui::Context,
    ) {
        match event {
            StreamingAudioEvent::SourceHash(source_hash) => {
                self.loaded_audio_hash = source_hash;
                if self.loaded_audio_hash.is_none() {
                    self.cache_status_message = Some(
                        "Analysis cache: precheck unavailable (source hash failed).".to_string(),
                    );
                    self.cache_status_message_at = Some(Instant::now());
                    self.cache_precheck_done = true;
                    self.loading_cache_timeline_preloaded = false;
                    self.loading_cache_waveform_preloaded = false;
                }
                self.maybe_precheck_analysis_cache();
            }
            StreamingAudioEvent::Started {
                sample_rate,
                total_samples,
                channels,
                metadata,
            } => {
                self.loading_sample_rate = sample_rate;
                self.loading_total_samples = total_samples;
                self.loading_source_channels = channels.max(1);
                self.audio_raw = Some(AudioData {
                    sample_rate,
                    channels: self.loading_source_channels,
                    samples_interleaved: Arc::new(Vec::new()),
                    samples_mono: Arc::new(Vec::new()),
                    metadata,
                });

                // Pre-allocate vector capacities to avoid repeated reallocations
                // and memcpy cascades during streaming for long files.
                if let Some(total) = total_samples {
                    self.processed_samples.reserve(total);
                    Arc::make_mut(&mut self.processed_playback_samples)
                        .reserve(total * self.loading_source_channels as usize);
                }

                self.album_art_texture = self.create_album_art_texture(ctx);
                self.maybe_precheck_analysis_cache();
            }
            StreamingAudioEvent::Chunk {
                samples_mono,
                samples_interleaved,
                channels,
                decoded_samples,
                total_samples,
            } => {
                if self.loading_sample_rate == 0 {
                    return;
                }

                let playback_channels = channels.max(1);
                self.loading_decoded_samples = decoded_samples;
                self.loading_total_samples = total_samples.or(self.loading_total_samples);
                self.loading_source_channels = playback_channels;

                let was_empty = self.processed_samples.is_empty();
                self.processed_samples.extend_from_slice(&samples_mono);
                Arc::make_mut(&mut self.processed_playback_samples)
                    .extend_from_slice(&samples_interleaved);
                self.processed_playback_channels = playback_channels;
                let processed_len = self.processed_samples.len();
                let waveform_budget = adaptive_waveform_budget(
                    self.audio_quality_mode.waveform_points(),
                    self.loading_total_samples,
                    self.loading_sample_rate,
                );
                let should_rebuild = !self.loading_cache_waveform_preloaded
                    && self.should_rebuild_streaming_waveform(processed_len);
                if should_rebuild {
                    let waveform = build_waveform_for_processed(
                        &self.processed_samples,
                        self.loading_sample_rate,
                        waveform_budget,
                        1.0,
                    );
                    let should_reset_view = was_empty && !waveform.is_empty();
                    self.set_waveform_data(waveform, should_reset_view);
                    self.mark_streaming_waveform_rebuild(processed_len);
                }
            }
            StreamingAudioEvent::Finished {
                sample_rate,
                channels,
                decoded_samples,
                total_samples,
                metadata,
            } => {
                self.loading_sample_rate = sample_rate;
                self.loading_decoded_samples = decoded_samples;
                self.loading_total_samples = total_samples.or(self.loading_total_samples);
                self.loading_source_channels = channels.max(1);

                // Build AudioData from the accumulated processed samples.
                // Since speed/pitch is identity during loading, processed == raw.
                let mono = std::mem::take(&mut self.processed_samples);
                let interleaved = Arc::clone(&self.processed_playback_samples);
                self.audio_raw = Some(AudioData {
                    sample_rate,
                    channels: self.loading_source_channels,
                    samples_mono: Arc::new(mono.clone()),
                    samples_interleaved: interleaved,
                    metadata,
                });
                self.processed_samples = mono;
                // self.processed_playback_samples is already the interleaved data
                self.processed_playback_channels = self.loading_source_channels;
                self.album_art_texture = self.create_album_art_texture(ctx);

                // Build final waveform at full quality now that loading is complete.
                let final_waveform = build_waveform_for_processed(
                    &self.processed_samples,
                    self.loading_sample_rate,
                    self.audio_quality_mode.waveform_points(),
                    1.0,
                );
                self.set_waveform_data(final_waveform, true);

                self.is_audio_loading = false;
                self.audio_loading_rx = None;
                self.audio_loading_cancel = None;
                self.live_stream_playback = false;
                self.loading_cache_waveform_preloaded = false;
                self.loading_preview_cache.clear();

                // Only rebuild the waveform from scratch when the cache didn't already
                // provide one — otherwise keep the cached waveform to avoid unnecessary
                // full-array scans on every load of the same file.
                if self.waveform.is_empty() {
                    let final_waveform = build_waveform_for_processed(
                        &self.processed_samples,
                        self.loading_sample_rate,
                        self.audio_quality_mode.waveform_points(),
                        1.0,
                    );
                    self.set_waveform_data(final_waveform, true);
                }

                let rebuild_mode = if self.preprocess_audio {
                    if self.loading_cache_timeline_preloaded {
                        RebuildMode::ParametersOnly
                    } else {
                        RebuildMode::Full
                    }
                } else {
                    RebuildMode::ParametersOnly
                };
                self.request_rebuild(false, rebuild_mode);
            }
            StreamingAudioEvent::Error(message) => {
                self.is_audio_loading = false;
                self.audio_loading_rx = None;
                self.audio_loading_cancel = None;
                self.live_stream_playback = false;
                self.loading_cache_timeline_preloaded = false;
                self.loading_cache_waveform_preloaded = false;
                self.loading_preview_cache.clear();
                self.last_error = Some(message);
            }
        }
    }

    pub(super) fn poll_audio_loading(&mut self, ctx: &egui::Context) {
        let mut events = Vec::new();
        let mut disconnected = false;

        if let Some(rx) = &self.audio_loading_rx {
            for _ in 0..AUDIO_LOADING_MAX_EVENTS_PER_FRAME {
                match rx.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        let events_len = events.len();

        for event in events {
            self.handle_audio_loading_event(event, ctx);
        }

        if disconnected {
            self.audio_loading_rx = None;
            self.audio_loading_cancel = None;
            self.is_audio_loading = false;
        }

        if self.is_audio_loading {
            if !self.note_visuals_ready() {
                self.clear_note_visuals();
            }

            if events_len == AUDIO_LOADING_MAX_EVENTS_PER_FRAME {
                // More work is likely queued; request immediate redraw but keep each frame bounded.
                ctx.request_repaint();
            }
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }
}
