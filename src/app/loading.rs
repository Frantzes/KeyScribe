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
    }

    pub(super) fn note_visuals_ready(&self) -> bool {
        let has_preprocessed_timeline =
            !self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0;

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

    pub(super) fn start_audio_loading(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.cancel_audio_loading();
        self.cancel_active_processing();
        self.stop();

        self.loaded_path = Some(path.clone());
        push_recent_file_path(&mut self.recent_file_paths, path.as_path());
        self.loaded_audio_hash = None;
        self.selected_time_sec = 0.0;
        self.audio_raw = None;
        self.processed_samples.clear();
        self.processed_playback_samples.clear();
        self.processed_playback_channels = 1;
        self.clear_waveform_data();
        self.note_timeline = Arc::new(Vec::new());
        self.note_timeline_step_sec = 0.0;
        self.base_note_timeline = Arc::new(Vec::new());
        self.base_note_timeline_step_sec = 0.0;
        self.loading_raw_samples.clear();
        self.loading_raw_samples_interleaved.clear();
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
        if self.preprocess_audio {
            self.cache_status_message = Some("Analysis cache: precheck pending...".to_string());
            self.cache_status_message_at = Some(Instant::now());
        }
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

                self.loading_raw_samples.extend_from_slice(&samples_mono);
                self.loading_raw_samples_interleaved
                    .extend_from_slice(&samples_interleaved);

                let mut processed_chunk = samples_mono;
                if !speed_pitch_is_identity(self.speed, self.pitch_semitones) {
                    processed_chunk = apply_speed_and_pitch(
                        &processed_chunk,
                        self.loading_sample_rate,
                        self.speed,
                        self.pitch_semitones,
                    );
                }

                let processed_chunk_playback =
                    if speed_pitch_is_identity(self.speed, self.pitch_semitones) {
                        samples_interleaved
                    } else {
                        apply_speed_and_pitch_interleaved(
                            &samples_interleaved,
                            playback_channels,
                            self.loading_sample_rate,
                            self.speed,
                            self.pitch_semitones,
                        )
                    };

                let was_empty = self.processed_samples.is_empty();
                self.processed_samples.extend_from_slice(&processed_chunk);
                self.processed_playback_samples
                    .extend_from_slice(&processed_chunk_playback);
                self.processed_playback_channels = playback_channels;
                let processed_len = self.processed_samples.len();
                if !self.loading_cache_waveform_preloaded
                    && self.should_rebuild_streaming_waveform(processed_len)
                {
                    let waveform = build_waveform_for_processed(
                        &self.processed_samples,
                        self.loading_sample_rate,
                        self.audio_quality_mode.waveform_points(),
                        self.speed,
                    );
                    let should_reset_view = was_empty && !waveform.is_empty();
                    self.set_waveform_data(waveform, should_reset_view);
                    self.mark_streaming_waveform_rebuild(processed_len);
                }

                if self.live_stream_playback
                    && !self.playing_preview_buffer
                    && !self.loop_playback_enabled
                {
                    let playback_rate = self.playback_rate();
                    if let Some(engine) = &mut self.engine {
                        if engine.has_active_sink() {
                            if let Err(err) = engine.append_samples(
                                &processed_chunk_playback,
                                playback_channels,
                                self.loading_sample_rate,
                                playback_rate,
                            ) {
                                self.last_error =
                                    Some(format!("Playback stream append error: {err}"));
                                self.live_stream_playback = false;
                            }
                        } else {
                            self.live_stream_playback = false;
                        }
                    }
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

                let raw_samples = std::mem::take(&mut self.loading_raw_samples);
                let raw_interleaved = std::mem::take(&mut self.loading_raw_samples_interleaved);
                self.audio_raw = Some(AudioData {
                    sample_rate,
                    channels: self.loading_source_channels,
                    samples_interleaved: Arc::new(raw_interleaved),
                    samples_mono: Arc::new(raw_samples),
                    metadata,
                });
                self.album_art_texture = self.create_album_art_texture(ctx);

                self.is_audio_loading = false;
                self.audio_loading_rx = None;
                self.audio_loading_cancel = None;
                self.live_stream_playback = false;
                self.loading_cache_waveform_preloaded = false;
                self.loading_preview_cache.clear();

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
                self.loading_raw_samples.clear();
                self.loading_raw_samples_interleaved.clear();
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
