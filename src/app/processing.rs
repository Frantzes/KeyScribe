use super::*;

impl KeyScribeApp {
    pub(super) fn request_rebuild(&mut self, restart_playback: bool, mode: RebuildMode) {
        if self.is_audio_loading {
            return;
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };

        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let sample_rate = raw.sample_rate;
        let raw_samples: Arc<Vec<f32>> = Arc::clone(&raw.samples_mono);
        let raw_playback_samples: Arc<Vec<f32>> = Arc::clone(&raw.samples_interleaved);
        let raw_playback_channels = raw.channels;
        let speed = self.speed;
        let pitch_semitones = self.pitch_semitones;
        let audio_quality_mode = self.audio_quality_mode;
        let use_cqt = self.use_cqt_analysis;
        let preprocess_audio = self.preprocess_audio;
        let base_timeline = Arc::clone(&self.base_note_timeline);
        let base_step = self.base_note_timeline_step_sec;
        let selected_time_sec = self.selected_time_sec;
        let source_hash = self.loaded_audio_hash.clone();
        let source_path = self.loaded_path.clone();
        let processing_epoch = Arc::clone(&self.processing_epoch);

        let (tx, rx) = mpsc::channel::<ProcessingResult>();
        self.processing_rx = Some(rx);
        self.active_rebuild_mode = mode;
        self.active_job_id = Some(job_id);
        self.is_processing = true;
        self.processing_started_at = Some(Instant::now());
        self.processing_audio_duration_sec = if sample_rate > 0 {
            raw_samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        self.processing_estimated_total_sec =
            self.estimate_processing_duration_sec(mode, raw_samples.len(), sample_rate);
        if mode == RebuildMode::Full {
            self.cache_status_message = Some("Analysis cache: checking...".to_string());
            self.cache_status_message_at = Some(Instant::now());
        }
        self.restart_playback_after_processing |= restart_playback;
        self.processing_epoch.store(job_id, Ordering::Release);
        self.clear_note_visuals();

        thread::spawn(move || {
            if mode == RebuildMode::ParametersPreview {
                if processing_epoch.load(Ordering::Acquire) != job_id {
                    return;
                }

                let playback_channels = raw_playback_channels.max(1);
                let playback_channels_usize = playback_channels as usize;
                let total_playback_frames = raw_playback_samples.len() / playback_channels_usize;
                let preview_start_frame =
                    (selected_time_sec.max(0.0) * sample_rate as f32) as usize;
                let preview_len_frames = (PARAM_UPDATE_PREVIEW_SEC * sample_rate as f32) as usize;
                let preview_end_frame =
                    (preview_start_frame.saturating_add(preview_len_frames))
                        .min(total_playback_frames);

                let preview_samples = if preview_start_frame < preview_end_frame {
                    let preview_start_idx = preview_start_frame * playback_channels_usize;
                    let preview_end_idx = preview_end_frame * playback_channels_usize;
                    apply_speed_and_pitch_interleaved(
                        &raw_playback_samples[preview_start_idx..preview_end_idx],
                        playback_channels,
                        sample_rate,
                        speed,
                        pitch_semitones,
                    )
                } else {
                    Vec::new()
                };

                if processing_epoch.load(Ordering::Acquire) != job_id {
                    return;
                }

                let _ = tx.send(ProcessingResult {
                    job_id,
                    mode,
                    cache_lookup_hit: None,
                    source_hash: None,
                    processed_samples: Vec::new(),
                    processed_playback_samples: Vec::new(),
                    processed_playback_channels: playback_channels,
                    waveform: Vec::new(),
                    note_timeline: Arc::new(Vec::new()),
                    note_timeline_step_sec: 0.0,
                    base_note_timeline: Arc::new(Vec::new()),
                    base_note_timeline_step_sec: 0.0,
                    analysis_error: None,
                    preview_playback: Some(PreviewPlayback {
                        samples: preview_samples,
                        channels: playback_channels,
                        timeline_start_sec: selected_time_sec.max(0.0),
                    }),
                });
                return;
            }

            let file_hash = source_hash.or_else(|| {
                source_path
                    .as_ref()
                    .and_then(|path| compute_file_hash(path.as_path()))
            });
            let content_hash = compute_audio_content_hash(sample_rate, raw_samples.as_slice());

            let mut cache_hash_candidates = Vec::<String>::new();
            if let Some(hash) = file_hash {
                cache_hash_candidates.push(hash);
            }
            if !cache_hash_candidates.iter().any(|h| h == &content_hash) {
                cache_hash_candidates.push(content_hash.clone());
            }

            let resolved_source_hash = cache_hash_candidates.first().cloned();

            if mode == RebuildMode::Full {
                for song_hash in &cache_hash_candidates {
                    if let Some((
                        cached_processed_samples,
                        cached_base_note_timeline,
                        cached_base_step,
                    )) = Self::load_analysis_cache_for_variant(
                        song_hash,
                        sample_rate,
                        raw_samples.len(),
                        raw_samples.as_slice(),
                        audio_quality_mode,
                        speed,
                        pitch_semitones,
                        use_cqt,
                        preprocess_audio,
                    ) {
                        if processing_epoch.load(Ordering::Acquire) != job_id {
                            return;
                        }

                        let (note_timeline, note_timeline_step_sec) = Self::transform_note_timeline(
                            Arc::clone(&cached_base_note_timeline),
                            cached_base_step,
                            speed,
                            pitch_semitones,
                        );

                        let waveform = build_waveform_for_processed(
                            &cached_processed_samples,
                            sample_rate,
                            audio_quality_mode.waveform_points(),
                            speed,
                        );

                        let processed_playback_samples = if speed_pitch_is_identity(
                            speed,
                            pitch_semitones,
                        ) {
                            raw_playback_samples.as_ref().to_vec()
                        } else {
                            apply_speed_and_pitch_interleaved(
                                raw_playback_samples.as_slice(),
                                raw_playback_channels,
                                sample_rate,
                                speed,
                                pitch_semitones,
                            )
                        };

                        if processing_epoch.load(Ordering::Acquire) != job_id {
                            return;
                        }

                        for candidate_hash in &cache_hash_candidates {
                            if candidate_hash == song_hash {
                                continue;
                            }
                            Self::persist_analysis_cache(
                                candidate_hash,
                                sample_rate,
                                raw_samples.len(),
                                audio_quality_mode,
                                speed,
                                pitch_semitones,
                                use_cqt,
                                preprocess_audio,
                                cached_processed_samples.as_slice(),
                                cached_base_note_timeline.as_ref(),
                                cached_base_step,
                            );
                        }

                        let _ = tx.send(ProcessingResult {
                            job_id,
                            mode,
                            cache_lookup_hit: Some(true),
                            source_hash: resolved_source_hash.clone(),
                            processed_samples: cached_processed_samples,
                            processed_playback_samples,
                            processed_playback_channels: raw_playback_channels.max(1),
                            waveform,
                            note_timeline,
                            note_timeline_step_sec,
                            base_note_timeline: cached_base_note_timeline,
                            base_note_timeline_step_sec: cached_base_step,
                            analysis_error: None,
                            preview_playback: None,
                        });
                        return;
                    }
                }
            }

            let processed_samples = if processing_epoch.load(Ordering::Acquire) != job_id {
                None
            } else {
                Some(apply_speed_and_pitch(
                    raw_samples.as_slice(),
                    sample_rate,
                    speed,
                    pitch_semitones,
                ))
            };
            let Some(processed_samples) = processed_samples else {
                return;
            };

            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            let processed_playback_samples = if speed_pitch_is_identity(speed, pitch_semitones) {
                raw_playback_samples.as_ref().to_vec()
            } else {
                apply_speed_and_pitch_interleaved(
                    raw_playback_samples.as_slice(),
                    raw_playback_channels,
                    sample_rate,
                    speed,
                    pitch_semitones,
                )
            };

            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            let waveform = build_waveform_for_processed(
                &processed_samples,
                sample_rate,
                audio_quality_mode.waveform_points(),
                speed,
            );

            let (base_note_timeline, base_note_timeline_step_sec, analysis_error) = match mode {
                RebuildMode::Full => {
                    let (timeline, step, err) = Self::build_note_timeline(
                        raw_samples.as_slice(),
                        sample_rate,
                        audio_quality_mode.fft_window_size(),
                        use_cqt,
                        preprocess_audio,
                    );
                    (Arc::new(timeline), step, err)
                }
                RebuildMode::ParametersOnly => (base_timeline, base_step, None),
                RebuildMode::ParametersPreview => unreachable!("preview mode returns early"),
            };

            let (note_timeline, note_timeline_step_sec) = Self::transform_note_timeline(
                Arc::clone(&base_note_timeline),
                base_note_timeline_step_sec,
                speed,
                pitch_semitones,
            );

            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            if let Some(song_hash) = resolved_source_hash.as_ref() {
                Self::persist_analysis_cache(
                    song_hash,
                    sample_rate,
                    raw_samples.len(),
                    audio_quality_mode,
                    speed,
                    pitch_semitones,
                    use_cqt,
                    preprocess_audio,
                    processed_samples.as_slice(),
                    base_note_timeline.as_ref(),
                    base_note_timeline_step_sec,
                );

                if mode == RebuildMode::Full {
                    for candidate_hash in &cache_hash_candidates {
                        if candidate_hash == song_hash {
                            continue;
                        }
                        Self::persist_analysis_cache(
                            candidate_hash,
                            sample_rate,
                            raw_samples.len(),
                            audio_quality_mode,
                            speed,
                            pitch_semitones,
                            use_cqt,
                            preprocess_audio,
                            processed_samples.as_slice(),
                            base_note_timeline.as_ref(),
                            base_note_timeline_step_sec,
                        );
                    }
                }
            }

            let _ = tx.send(ProcessingResult {
                job_id,
                mode,
                cache_lookup_hit: if mode == RebuildMode::Full {
                    Some(false)
                } else {
                    None
                },
                source_hash: resolved_source_hash,
                processed_samples,
                processed_playback_samples,
                processed_playback_channels: raw_playback_channels.max(1),
                waveform,
                note_timeline,
                note_timeline_step_sec,
                base_note_timeline,
                base_note_timeline_step_sec,
                analysis_error,
                preview_playback: None,
            });
        });
    }

    pub(super) fn poll_processing_result(&mut self) {
        let Some(rx) = &self.processing_rx else {
            return;
        };

        match rx.try_recv() {
            Ok(result) => {
                if Some(result.job_id) == self.active_job_id {
                    self.apply_processing_result(result);
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.clear_processing_job();
            }
        }
    }

    pub(super) fn save_state_to_disk(&self) {
        let state = PersistedState {
            last_file: self.loaded_path.clone(),
            recent_files: self.recent_file_paths.clone(),
            selected_time_sec: self.selected_time_sec,
            speed: self.speed,
            pitch_semitones: self.pitch_semitones,
            key_color_sensitivity: self.key_color_sensitivity,
            piano_zoom: self.piano_zoom,
            piano_key_height: self.piano_key_height,
            waveform_panel_height: self.waveform_panel_height,
            probability_panel_height: self.probability_panel_height,
            piano_panel_height: self.piano_panel_height,
            show_note_hist_window: self.show_note_hist_window,
            use_cqt_analysis: self.use_cqt_analysis,
            preprocess_audio: self.preprocess_audio,
            playback_volume: self.playback_volume,
            audio_quality_mode: self.audio_quality_mode,
            audio_output_device_id: self.audio_output_device_id.clone(),
            loop_enabled: self.loop_enabled,
            dark_mode: self.dark_mode,
            highlight_hex: color_to_hex(self.highlight_color),
            recent_highlight_hex: self.recent_highlight_hex.clone(),
        };

        if let Ok(raw) = serde_json::to_string_pretty(&state) {
            let path = state_file_path();
            if ensure_parent_dir(path.as_path()) {
                let _ = fs::write(path, raw);
            }
        }
    }

    pub(super) fn update_note_probabilities(&mut self, force: bool) {
        if !force && self.last_prob_update.elapsed() < PROBABILITY_UPDATE_INTERVAL {
            return;
        }

        if !self.note_visuals_ready() {
            self.clear_note_visuals();
            self.last_prob_update = Instant::now();
            return;
        }

        if self.preprocess_audio
            && !self.note_timeline.is_empty()
            && self.note_timeline_step_sec > 0.0
        {
            let idx = (self.selected_time_sec.max(0.0) / self.note_timeline_step_sec) as usize;
            let idx = idx.min(self.note_timeline.len().saturating_sub(1));
            self.note_probs = self.note_timeline[idx].clone();
        } else {
            let Some(raw) = &self.audio_raw else {
                return;
            };
            if self.processed_samples.is_empty() {
                return;
            }

            let output_time_sec = self.source_to_output_time(self.selected_time_sec.max(0.0));
            let center = (output_time_sec * raw.sample_rate as f32) as usize;
            let fft_window_size = self.audio_quality_mode.fft_window_size();
            self.note_probs = if self.use_cqt_analysis {
                detect_note_probabilities_cqt_preview(
                    &self.processed_samples,
                    raw.sample_rate,
                    center,
                    fft_window_size,
                )
            } else {
                detect_note_probabilities(
                    &self.processed_samples,
                    raw.sample_rate,
                    center,
                    fft_window_size,
                )
            };
        }

        // Smooth the visual state to reduce rapid flicker between adjacent notes.
        for (smoothed, current) in self
            .note_probs_smoothed
            .iter_mut()
            .zip(self.note_probs.iter())
        {
            *smoothed = *smoothed * 0.78 + *current * 0.22;
        }

        self.last_prob_update = Instant::now();
    }

    pub(super) fn compute_fft_timeline(
        samples: &[f32],
        sample_rate: u32,
        step_sec: f32,
        fft_window_size: usize,
    ) -> Vec<Vec<f32>> {
        if samples.is_empty() || sample_rate == 0 || step_sec <= 0.0 || fft_window_size < 64 {
            return Vec::new();
        }

        let mut timeline = Vec::new();
        let total_sec = samples.len() as f32 / sample_rate as f32;
        let mut t = 0.0f32;

        while t <= total_sec {
            let center = (t * sample_rate as f32) as usize;
            timeline.push(detect_note_probabilities(
                samples,
                sample_rate,
                center,
                fft_window_size,
            ));
            t += step_sec;
        }

        if timeline.is_empty() {
            timeline.push(vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize]);
        }

        timeline
    }

    pub(super) fn build_note_timeline(
        source_samples: &[f32],
        sample_rate: u32,
        fft_window_size: usize,
        use_cqt: bool,
        preprocess_audio: bool,
    ) -> (Vec<Vec<f32>>, f32, Option<String>) {
        if !preprocess_audio {
            return (Vec::new(), 0.0, None);
        }

        if use_cqt {
            match analyze_with_full_pipeline(source_samples, sample_rate) {
                Ok((_smoothed, probs)) => {
                    let duration_sec = source_samples.len() as f32 / sample_rate.max(1) as f32;
                    let step_sec = if probs.is_empty() {
                        0.0
                    } else {
                        (duration_sec / probs.len() as f32).max(1e-3)
                    };
                    (probs, step_sec, None)
                }
                Err(err) => {
                    let fallback = Self::compute_fft_timeline(
                        source_samples,
                        sample_rate,
                        FFT_TIMELINE_STEP_SEC,
                        fft_window_size,
                    );
                    (
                        fallback,
                        FFT_TIMELINE_STEP_SEC,
                        Some(format!("Pro analysis failed, using FFT fallback: {err}")),
                    )
                }
            }
        } else {
            (
                Self::compute_fft_timeline(
                    source_samples,
                    sample_rate,
                    FFT_TIMELINE_STEP_SEC,
                    fft_window_size,
                ),
                FFT_TIMELINE_STEP_SEC,
                None,
            )
        }
    }

    pub(super) fn transpose_frame(frame: &[f32], semitones: f32) -> Vec<f32> {
        if frame.is_empty() {
            return Vec::new();
        }

        if semitones.abs() < 1.0e-6 {
            return frame.to_vec();
        }

        let n = frame.len() as f32;
        let mut out = vec![0.0f32; frame.len()];
        for (dst_idx, dst) in out.iter_mut().enumerate() {
            let src_idx = dst_idx as f32 - semitones;
            if src_idx < 0.0 || src_idx >= n - 1.0 {
                continue;
            }

            let i0 = src_idx.floor() as usize;
            let i1 = (i0 + 1).min(frame.len() - 1);
            let frac = src_idx - i0 as f32;
            *dst = frame[i0] * (1.0 - frac) + frame[i1] * frac;
        }

        out
    }

    pub(super) fn transform_note_timeline(
        base_timeline: Arc<Vec<Vec<f32>>>,
        base_step_sec: f32,
        speed: f32,
        pitch_semitones: f32,
    ) -> (Arc<Vec<Vec<f32>>>, f32) {
        if base_timeline.is_empty() || base_step_sec <= 0.0 {
            return (Arc::new(Vec::new()), 0.0);
        }

        let transformed = if pitch_semitones.abs() < 1.0e-6 {
            base_timeline
        } else {
            let transformed = if base_timeline.len() >= 256 {
                base_timeline
                    .par_iter()
                    .map(|frame| Self::transpose_frame(frame, pitch_semitones))
                    .collect()
            } else {
                base_timeline
                    .iter()
                    .map(|frame| Self::transpose_frame(frame, pitch_semitones))
                    .collect()
            };
            Arc::new(transformed)
        };

        let _ = speed;
        let step_sec = base_step_sec;
        (transformed, step_sec)
    }

}
