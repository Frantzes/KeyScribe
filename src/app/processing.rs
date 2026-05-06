use super::*;
use crate::leadsheet::StemType;

impl KeyScribeApp {
    pub(super) fn request_rebuild(&mut self, restart_playback: bool, mode: RebuildMode) {
        if self.is_audio_loading {
            return;
        }

        let (raw_sample_rate, raw_samples_mono, raw_samples_interleaved, raw_channels) = {
            let Some(raw) = &self.audio_raw else {
                return;
            };
            (
                raw.sample_rate,
                Arc::clone(&raw.samples_mono),
                Arc::clone(&raw.samples_interleaved),
                raw.channels,
            )
        };

        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let speed = self.speed;
        let pitch_semitones = self.pitch_semitones;
        let audio_quality_mode = self.audio_quality_mode;
        let use_cqt = self.use_cqt_analysis;
        let preprocess_audio = self.preprocess_audio;
        let base_timeline = Arc::clone(&self.base_note_timeline);
        let base_step = self.base_note_timeline_step_sec;
        let selected_time_sec = self.selected_time_sec;
        let stems_active = self.separated_stems.is_some();
        let source_hash = if stems_active {
            None
        } else {
            self.loaded_audio_hash.clone()
        };
        let source_path = if stems_active {
            None
        } else {
            self.loaded_path.clone()
        };
        let processing_epoch = Arc::clone(&self.processing_epoch);

        let stem_sample_rate = self
            .separated_stems
            .as_ref()
            .and_then(|stems| stems.first().map(|stem| stem.sample_rate));
        let maybe_stems = self.separated_stems.clone();
        let enabled_indices: Vec<usize> = self.enabled_stem_indices.iter().copied().collect();
        let listening_indices: Vec<usize> =
            self.enabled_listening_indices.iter().copied().collect();

        let (tx, rx) = mpsc::channel::<ProcessingResult>();
        self.processing_rx = Some(rx);
        self.active_rebuild_mode = mode;
        self.active_job_id = Some(job_id);
        self.is_processing = true;
        self.processing_started_at = Some(Instant::now());

        // Estimate processing duration
        self.processing_audio_duration_sec = if raw_sample_rate > 0 {
            raw_samples_mono.len() as f32 / raw_sample_rate as f32
        } else {
            0.0
        };
        self.processing_estimated_total_sec =
            self.estimate_processing_duration_sec(mode, raw_samples_mono.len(), raw_sample_rate);

        if mode == RebuildMode::Full {
            self.cache_status_message = Some("Analysis cache: checking...".to_string());
            self.cache_status_message_at = Some(Instant::now());
        }
        self.restart_playback_after_processing |= restart_playback;
        self.processing_epoch.store(job_id, Ordering::Release);

        let keep_cache_preloaded_visuals = (mode == RebuildMode::ParametersOnly
            || mode == RebuildMode::VisualizationOnly)
            && self.preprocess_audio
            && (self.loading_cache_timeline_preloaded || !self.note_timeline.is_empty())
            && self.note_timeline_step_sec > 0.0;
        if !keep_cache_preloaded_visuals {
            self.clear_note_visuals();
        }

        let processing_sample_rate = stem_sample_rate.unwrap_or(raw_sample_rate);

        thread::spawn(move || {
            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            let sample_rate = processing_sample_rate;

            let (
                raw_analysis_samples,
                raw_render_samples,
                raw_playback_samples,
                raw_playback_channels,
            ) = if let Some(stems) = maybe_stems {
                let enabled_stems: Vec<_> = enabled_indices
                    .into_iter()
                    .filter_map(|idx| stems.get(idx).cloned())
                    .collect();

                let melodic_stems: Vec<_> = enabled_stems
                    .iter()
                    .filter(|s| s.stem_type != StemType::Drums)
                    .cloned()
                    .collect();

                let mono_analysis = if !melodic_stems.is_empty() {
                    crate::leadsheet::blend_for_chords(melodic_stems.as_slice())
                } else {
                    let total_mono_len = stems
                        .iter()
                        .map(|s| s.samples_mono.len())
                        .max()
                        .unwrap_or(0);
                    Arc::new(vec![0.0f32; total_mono_len])
                };

                let (interleaved, channels) = if !listening_indices.is_empty() {
                    let listen_stems: Vec<_> = listening_indices
                        .iter()
                        .copied()
                        .filter_map(|idx| stems.get(idx).cloned())
                        .collect();

                    if listen_stems.is_empty() {
                        crate::leadsheet::blend_interleaved_stems(stems.as_slice())
                    } else if listen_stems.len() == 1 {
                        let s = &listen_stems[0];
                        (Arc::clone(&s.samples_interleaved), s.channels)
                    } else {
                        crate::leadsheet::blend_interleaved_stems(listen_stems.as_slice())
                    }
                } else {
                    // "Blend / All" -> use all stems for playback
                    crate::leadsheet::blend_interleaved_stems(stems.as_slice())
                };

                let mono_render = if !listening_indices.is_empty() {
                    let listen_stems: Vec<_> = listening_indices
                        .iter()
                        .copied()
                        .filter_map(|idx| stems.get(idx).cloned())
                        .collect();
                    if listen_stems.is_empty() {
                        raw_samples_mono.clone()
                    } else {
                        crate::leadsheet::blend_for_chords(listen_stems.as_slice())
                    }
                } else {
                    raw_samples_mono.clone()
                };

                (mono_analysis, mono_render, interleaved, channels)
            } else {
                (
                    raw_samples_mono.clone(),
                    raw_samples_mono.clone(),
                    raw_samples_interleaved,
                    raw_channels,
                )
            };

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
                let preview_end_frame = (preview_start_frame.saturating_add(preview_len_frames))
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
                        sample_rate,
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
            let content_hash =
                compute_audio_content_hash(sample_rate, raw_analysis_samples.as_slice());

            let mut cache_hash_candidates = Vec::<String>::new();
            if let Some(hash) = file_hash {
                cache_hash_candidates.push(hash);
            }
            if !cache_hash_candidates.iter().any(|h| h == &content_hash) {
                cache_hash_candidates.push(content_hash.clone());
            }

            let resolved_source_hash = cache_hash_candidates.first().cloned();
            let result_source_hash = if stems_active {
                None
            } else {
                resolved_source_hash.clone()
            };

            let allow_cache = !stems_active;
            if allow_cache && (mode == RebuildMode::Full || mode == RebuildMode::VisualizationOnly)
            {
                for song_hash in &cache_hash_candidates {
                    if let Some((
                        cached_processed_samples,
                        cached_base_note_timeline,
                        cached_base_step,
                        cached_waveform,
                    )) = Self::load_analysis_cache_for_variant(
                        song_hash,
                        sample_rate,
                        raw_analysis_samples.len(),
                        raw_analysis_samples.as_slice(),
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

                        let mut processed_playback_samples = Vec::new();
                        let mut waveform = Vec::new();
                        let mut processed_samples_out = Vec::new();

                        if mode != RebuildMode::VisualizationOnly {
                            let had_cached_waveform = cached_waveform.is_some();
                            waveform = cached_waveform.unwrap_or_else(|| {
                                build_waveform_for_processed(
                                    &cached_processed_samples,
                                    sample_rate,
                                    audio_quality_mode.waveform_points(),
                                    speed,
                                )
                            });

                            processed_playback_samples =
                                if speed_pitch_is_identity(speed, pitch_semitones) {
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
                            processed_samples_out = cached_processed_samples;

                            if !had_cached_waveform {
                                Self::persist_analysis_cache(
                                    song_hash,
                                    sample_rate,
                                    raw_analysis_samples.len(),
                                    audio_quality_mode,
                                    speed,
                                    pitch_semitones,
                                    use_cqt,
                                    preprocess_audio,
                                    processed_samples_out.as_slice(),
                                    waveform.as_slice(),
                                    cached_base_note_timeline.as_ref(),
                                    cached_base_step,
                                );
                            }

                            for candidate_hash in &cache_hash_candidates {
                                if candidate_hash == song_hash {
                                    continue;
                                }
                                Self::persist_analysis_cache(
                                    candidate_hash,
                                    sample_rate,
                                    raw_analysis_samples.len(),
                                    audio_quality_mode,
                                    speed,
                                    pitch_semitones,
                                    use_cqt,
                                    preprocess_audio,
                                    processed_samples_out.as_slice(),
                                    waveform.as_slice(),
                                    cached_base_note_timeline.as_ref(),
                                    cached_base_step,
                                );
                            }
                        }

                        if processing_epoch.load(Ordering::Acquire) != job_id {
                            return;
                        }

                        let _ = tx.send(ProcessingResult {
                            job_id,
                            mode,
                            cache_lookup_hit: Some(true),
                            source_hash: result_source_hash.clone(),
                            processed_samples: processed_samples_out,
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

            let processed_samples = if mode == RebuildMode::VisualizationOnly {
                None
            } else {
                if processing_epoch.load(Ordering::Acquire) != job_id {
                    None
                } else {
                    Some(apply_speed_and_pitch(
                        raw_render_samples.as_slice(),
                        sample_rate,
                        speed,
                        pitch_semitones,
                    ))
                }
            };

            let processed_playback_samples = if mode == RebuildMode::VisualizationOnly {
                Vec::new()
            } else {
                if speed_pitch_is_identity(speed, pitch_semitones) {
                    raw_playback_samples.as_ref().to_vec()
                } else {
                    apply_speed_and_pitch_interleaved(
                        raw_playback_samples.as_slice(),
                        raw_playback_channels,
                        sample_rate,
                        speed,
                        pitch_semitones,
                    )
                }
            };

            if processing_epoch.load(Ordering::Acquire) != job_id {
                return;
            }

            let waveform = if let Some(ps) = &processed_samples {
                build_waveform_for_processed(
                    ps,
                    sample_rate,
                    audio_quality_mode.waveform_points(),
                    speed,
                )
            } else {
                Vec::new()
            };

            let expected_duration_sec = if raw_sample_rate > 0 {
                raw_samples_mono.len() as f32 / raw_sample_rate as f32
            } else {
                0.0
            };

            let (base_note_timeline, base_note_timeline_step_sec, analysis_error) = match mode {
                RebuildMode::Full | RebuildMode::VisualizationOnly => {
                    let (timeline, step, err) = Self::build_note_timeline(
                        raw_analysis_samples.as_slice(),
                        sample_rate,
                        audio_quality_mode.fft_window_size(),
                        use_cqt,
                        preprocess_audio,
                        Some(expected_duration_sec),
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
                if let Some(ps) = &processed_samples {
                    Self::persist_analysis_cache(
                        song_hash,
                        sample_rate,
                        raw_analysis_samples.len(),
                        audio_quality_mode,
                        speed,
                        pitch_semitones,
                        use_cqt,
                        preprocess_audio,
                        ps.as_slice(),
                        waveform.as_slice(),
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
                                raw_analysis_samples.len(),
                                audio_quality_mode,
                                speed,
                                pitch_semitones,
                                use_cqt,
                                preprocess_audio,
                                ps.as_slice(),
                                waveform.as_slice(),
                                base_note_timeline.as_ref(),
                                base_note_timeline_step_sec,
                            );
                        }
                    }
                }
            }

            let cache_lookup_hit = if allow_cache
                && (mode == RebuildMode::Full || mode == RebuildMode::VisualizationOnly)
            {
                Some(false)
            } else {
                None
            };

            let _ = tx.send(ProcessingResult {
                job_id,
                mode,
                cache_lookup_hit,
                source_hash: result_source_hash,
                processed_samples: processed_samples.unwrap_or_default(),
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

    pub(super) fn poll_separation_result(&mut self) {
        let Some(rx) = &self.separation_rx else {
            return;
        };

        match rx.try_recv() {
            Ok(result) => {
                self.is_separating = false;
                self.separation_rx = None;
                if let Some(err) = result.error {
                    self.last_error = Some(err);
                } else {
                    self.separated_stems = Some(result.stems);
                    self.enabled_listening_indices.clear();
                    self.enabled_stem_indices = self
                        .separated_stems
                        .as_ref()
                        .map(|stems| (0..stems.len()).collect())
                        .unwrap_or_default();
                    self.refresh_note_timeline_from_selected_stems();
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.is_separating = false;
                self.separation_rx = None;
            }
        }
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
            key_highlight_max_sec: self.key_highlight_max_sec,
            visualization_timing_offset_ms: self.visualization_timing_offset_ms,
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

    pub(super) fn update_note_highlight_visuals(&mut self, elapsed_sec: f32) {
        if self.note_highlight_hold_remaining.len() != self.note_probs.len() {
            self.note_highlight_hold_remaining
                .resize(self.note_probs.len(), 0.0);
        }

        let dt = elapsed_sec.clamp(0.0, 0.25);
        let hold_max_sec = self
            .key_highlight_max_sec
            .clamp(KEY_HIGHLIGHT_MAX_SEC_MIN, KEY_HIGHLIGHT_MAX_SEC_MAX);
        let hold_floor = (NOTE_HIGHLIGHT_ACTIVATION_THRESHOLD
            / self.key_color_sensitivity.max(0.05))
        .clamp(0.0, 1.0);

        for ((smoothed, current), hold_remaining) in self
            .note_probs_smoothed
            .iter_mut()
            .zip(self.note_probs.iter())
            .zip(self.note_highlight_hold_remaining.iter_mut())
        {
            let current = current.clamp(0.0, 1.0);
            if current >= hold_floor {
                *hold_remaining = hold_max_sec;
            } else if dt > 0.0 {
                *hold_remaining = (*hold_remaining - dt).max(0.0);
            }

            let held_target = if *hold_remaining > 0.0 {
                hold_floor
            } else {
                0.0
            };
            let target = current.max(held_target);

            *smoothed = *smoothed * 0.86 + target * 0.14;
        }
    }

    pub(super) fn update_note_probabilities(&mut self, force: bool) {
        let elapsed_sec = self.last_prob_update.elapsed().as_secs_f32();
        if !force && elapsed_sec < PROBABILITY_UPDATE_INTERVAL.as_secs_f32() {
            return;
        }

        if !self.note_visuals_ready() {
            self.clear_note_visuals();
            self.last_prob_update = Instant::now();
            return;
        }

        let timing_offset_sec = self.visualization_timing_offset_ms / 1000.0;
        let current_time = if self.is_playing() {
            self.current_position_sec()
        } else {
            self.selected_time_sec
        };
        let current_time = (current_time + timing_offset_sec).max(0.0);

        // 1. Prioritize pre-computed timeline (works for both original audio and stems)
        if !self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0 {
            let idx = (current_time.max(0.0) / self.note_timeline_step_sec) as usize;
            let idx = idx.min(self.note_timeline.len().saturating_sub(1));
            self.note_probs = self.note_timeline[idx].clone();
        }
        // 2. Fallback to live analysis if timeline is not ready
        else if let Some((stem_audio, stem_sample_rate)) = self.visualizing_stem_audio() {
            if self.audio_raw.is_none() {
                return;
            }

            if stem_audio.len() >= 64 {
                let center = (current_time.max(0.0) * stem_sample_rate as f32) as usize;
                let fft_window_size = self.audio_quality_mode.fft_window_size();
                self.note_probs = detect_note_probabilities(
                    &stem_audio,
                    stem_sample_rate,
                    center.min(stem_audio.len().saturating_sub(1)),
                    fft_window_size,
                );
            } else {
                self.note_probs = vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize];
            }
        } else {
            let Some(raw) = &self.audio_raw else {
                return;
            };
            if self.processed_samples.is_empty() {
                return;
            }

            let output_time_sec = self.source_to_output_time(current_time.max(0.0));
            let center = (output_time_sec * raw.sample_rate as f32) as usize;
            let fft_window_size = self.audio_quality_mode.fft_window_size();
            self.note_probs = detect_note_probabilities(
                &self.processed_samples,
                raw.sample_rate,
                center,
                fft_window_size,
            );
        }

        self.update_note_highlight_visuals(elapsed_sec);

        self.current_chord = {
            let sensitivity = self.key_color_sensitivity.clamp(0.0, 2.0);
            let threshold = if sensitivity > 0.0 {
                (NOTE_HIGHLIGHT_ACTIVATION_THRESHOLD / sensitivity).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let active: Vec<u8> = self
                .note_probs_smoothed
                .iter()
                .enumerate()
                .filter(|(_, p)| **p >= threshold)
                .map(|(i, _)| (PIANO_LOW_MIDI as usize + i) as u8)
                .collect();
            if active.len() >= 2 {
                crate::leadsheet::chord::detect_chord_from_active_notes(&active)
            } else {
                None
            }
        };

        self.last_prob_update = Instant::now();
    }

    pub(super) fn refresh_note_timeline_from_selected_stems(&mut self) {
        self.request_rebuild(false, RebuildMode::Full);
    }

    pub(super) fn refresh_note_timeline_from_selected_stems_preserving(&mut self) {
        self.request_rebuild_preserving_playback_and_waveform();
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

        let total_sec = samples.len() as f32 / sample_rate as f32;
        let frame_count = ((total_sec / step_sec).floor() as usize).saturating_add(1);

        let timeline: Vec<Vec<f32>> = if frame_count >= 256 {
            (0..frame_count)
                .into_par_iter()
                .map(|idx| {
                    let t = idx as f32 * step_sec;
                    let center =
                        ((t * sample_rate as f32) as usize).min(samples.len().saturating_sub(1));
                    detect_note_probabilities(samples, sample_rate, center, fft_window_size)
                })
                .collect()
        } else {
            (0..frame_count)
                .map(|idx| {
                    let t = idx as f32 * step_sec;
                    let center =
                        ((t * sample_rate as f32) as usize).min(samples.len().saturating_sub(1));
                    detect_note_probabilities(samples, sample_rate, center, fft_window_size)
                })
                .collect()
        };

        if timeline.is_empty() {
            return vec![vec![0.0; (PIANO_HIGH_MIDI - PIANO_LOW_MIDI + 1) as usize]];
        }

        timeline
    }

    pub(super) fn build_note_timeline(
        source_samples: &[f32],
        sample_rate: u32,
        fft_window_size: usize,
        use_cqt: bool,
        preprocess_audio: bool,
        expected_duration_sec: Option<f32>,
    ) -> (Vec<Vec<f32>>, f32, Option<String>) {
        let _ = (fft_window_size, use_cqt);
        if !preprocess_audio {
            return (Vec::new(), 0.0, None);
        }

        match analyze_with_full_pipeline(source_samples, sample_rate) {
            Ok((_smoothed, probs)) => {
                let duration_sec = expected_duration_sec
                    .unwrap_or_else(|| source_samples.len() as f32 / sample_rate.max(1) as f32);
                let step_sec = if probs.is_empty() {
                    0.0
                } else {
                    (duration_sec / probs.len() as f32).max(1e-3)
                };
                (probs, step_sec, None)
            }
            Err(err) => (
                Vec::new(),
                0.0,
                Some(format!("Basic Pitch analysis failed: {err}")),
            ),
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
