use super::*;
use crate::leadsheet::StemType;

impl KeyScribeApp {
    fn loading_preview_chunk_start(start_sec: f32) -> f32 {
        let stride = LOADING_PREVIEW_CACHE_STRIDE_SEC.max(0.25);
        (start_sec.max(0.0) / stride).floor() * stride
    }

    fn prune_loading_preview_cache(&mut self) {
        while self.loading_preview_cache.len() > LOADING_PREVIEW_CACHE_MAX_ENTRIES {
            if let Some((oldest_idx, _)) = self
                .loading_preview_cache
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used_at)
            {
                self.loading_preview_cache.remove(oldest_idx);
            } else {
                break;
            }
        }
    }

    fn get_or_decode_loading_preview_chunk(
        &mut self,
        path: &std::path::Path,
        start_sec: f32,
    ) -> anyhow::Result<(std::sync::Arc<Vec<f32>>, u16, u32, f32)> {
        let source_key = normalize_recent_file_key(path);
        let chunk_start_sec = Self::loading_preview_chunk_start(start_sec);

        if let Some(entry) = self.loading_preview_cache.iter_mut().find(|entry| {
            entry.source_key == source_key
                && (entry.chunk_start_sec - chunk_start_sec).abs() < 1.0e-3
        }) {
            entry.last_used_at = std::time::Instant::now();
            return Ok((
                std::sync::Arc::clone(&entry.samples_interleaved),
                entry.channels,
                entry.sample_rate,
                entry.chunk_start_sec,
            ));
        }

        let preview =
            load_audio_preview_chunk(path, chunk_start_sec, LOADING_PREVIEW_CACHE_CHUNK_SEC)?;
        if preview.samples_interleaved.is_empty() {
            return Err(anyhow::anyhow!("Preview decode returned no audio frames"));
        }

        let AudioPreviewChunk {
            sample_rate,
            channels,
            samples_interleaved,
        } = preview;
        let samples_arc = std::sync::Arc::new(samples_interleaved);

        self.loading_preview_cache.push(LoadingPreviewCacheEntry {
            source_key,
            chunk_start_sec,
            sample_rate,
            channels,
            samples_interleaved: std::sync::Arc::clone(&samples_arc),
            last_used_at: std::time::Instant::now(),
        });
        self.prune_loading_preview_cache();

        Ok((samples_arc, channels, sample_rate, chunk_start_sec))
    }

    fn play_loading_preview_from_source(&mut self, start_sec: f32, end_sec: Option<f32>) -> bool {
        if !self.is_audio_loading {
            return false;
        }

        let Some(path) = self.loaded_path.clone() else {
            return false;
        };

        let preview_len_sec = end_sec
            .map(|end| (end - start_sec).max(0.25))
            .unwrap_or(PARAM_UPDATE_PREVIEW_SEC)
            .clamp(0.25, 30.0);

        let (cached_samples, channels, sample_rate, chunk_start_sec) =
            match self.get_or_decode_loading_preview_chunk(path.as_path(), start_sec) {
                Ok(preview) => preview,
                Err(err) => {
                    self.last_error = Some(format!("Preview decode error: {err}"));
                    return false;
                }
            };

        let channels_usize = channels.max(1) as usize;
        let total_frames = cached_samples.len() / channels_usize;
        if total_frames == 0 {
            return false;
        }

        let start_frame =
            (((start_sec - chunk_start_sec).max(0.0)) * sample_rate as f32).floor() as usize;
        if start_frame >= total_frames {
            return false;
        }

        let preview_frames = (preview_len_sec * sample_rate as f32).ceil().max(1.0) as usize;
        let end_frame = start_frame.saturating_add(preview_frames).min(total_frames);
        let start_idx = start_frame * channels_usize;
        let end_idx = end_frame * channels_usize;
        let raw_preview_slice = &cached_samples[start_idx..end_idx];

        let playback_samples = if speed_pitch_is_identity(self.speed, self.pitch_semitones) {
            raw_preview_slice.to_vec()
        } else {
            apply_speed_and_pitch_interleaved(
                raw_preview_slice,
                channels,
                sample_rate,
                self.speed,
                self.pitch_semitones,
            )
        };

        if playback_samples.is_empty() {
            return false;
        }

        let playback_rate = self.playback_rate();
        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_chunk_at_timeline(
                playback_samples.as_slice(),
                channels,
                sample_rate,
                start_sec.max(0.0),
                playback_rate,
            ) {
                self.last_error = Some(format!("Playback error: {err}"));
                self.playing_preview_buffer = false;
                self.live_stream_playback = false;
                return false;
            }

            self.playing_preview_buffer = true;
            self.live_stream_playback = false;
            return true;
        }

        self.last_error = Some("Audio engine unavailable on this machine".to_string());
        false
    }

    pub(super) fn visualizing_stem_audio(&self) -> Option<(Arc<Vec<f32>>, u32)> {
        let stems = self.separated_stems.as_ref()?;
        if stems.is_empty() {
            return None;
        }

        let stem_sample_rate = stems.first().map(|stem| stem.sample_rate)?;

        let enabled_indices: Vec<usize> = self.enabled_stem_indices.iter().copied().collect();
        let melodic_stems: Vec<_> = enabled_indices
            .into_iter()
            .filter_map(|idx| stems.get(idx).cloned())
            .filter(|s| s.stem_type != StemType::Drums)
            .collect();

        if melodic_stems.is_empty() {
            let total_len = stems
                .iter()
                .map(|s| s.samples_mono.len())
                .max()
                .unwrap_or(0);
            return Some((Arc::new(vec![0.0f32; total_len]), stem_sample_rate));
        }

        Some((
            crate::leadsheet::blend_for_chords(melodic_stems.as_slice()),
            stem_sample_rate,
        ))
    }

    pub(super) fn listening_stem_audio(&self) -> Option<(Arc<Vec<f32>>, u16, u32)> {
        let stems = self.separated_stems.as_ref()?;
        if stems.is_empty() {
            return None;
        }

        let stem_sample_rate = stems.first().map(|stem| stem.sample_rate)?;

        let enabled_indices: Vec<usize> = self.enabled_listening_indices.iter().copied().collect();
        let enabled_stems: Vec<_> = enabled_indices
            .into_iter()
            .filter_map(|idx| stems.get(idx).cloned())
            .collect();

        if enabled_stems.is_empty() {
            // If nothing is selected for listening, blend all available stems
            // (Standard "Original Mix" behavior)
            let (samples, channels) = crate::leadsheet::blend_interleaved_stems(stems.as_slice());
            Some((samples, channels, stem_sample_rate))
        } else if enabled_stems.len() == 1 {
            let stem = &enabled_stems[0];
            Some((
                Arc::clone(&stem.samples_interleaved),
                stem.channels,
                stem.sample_rate,
            ))
        } else {
            let (samples, channels) =
                crate::leadsheet::blend_interleaved_stems(enabled_stems.as_slice());
            Some((samples, channels, stem_sample_rate))
        }
    }

    pub(super) fn active_stem_audio(&self) -> Option<(Arc<Vec<f32>>, u16, u32)> {
        self.listening_stem_audio()
    }

    pub(super) fn play_from_selected(&mut self) {
        if self.play_preview_at(self.selected_time_sec, None) {
            self.live_stream_playback = false;
            return;
        }

        let available_duration = self.source_duration();
        if self.is_audio_loading
            && (self.selected_time_sec > available_duration + 0.01
                || self.processed_playback_samples.is_empty())
        {
            if self.play_loading_preview_from_source(self.selected_time_sec, None) {
                return;
            }
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };
        let playback_rate = self.playback_rate();

        let (playback_samples, channels, sample_rate) =
            if let Some((stems, ch, sr)) = self.active_stem_audio() {
                (stems, ch, sr)
            } else {
                (
                    Arc::new(self.processed_playback_samples.clone()),
                    self.processed_playback_channels,
                    raw.sample_rate,
                )
            };

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_from(
                &playback_samples,
                channels,
                sample_rate,
                self.selected_time_sec,
                playback_rate,
            ) {
                self.last_error = Some(format!("Playback error: {err}"));
                self.playing_preview_buffer = false;
                self.live_stream_playback = false;
            } else {
                self.playing_preview_buffer = false;
                self.live_stream_playback = self.is_audio_loading;
            }
        } else {
            self.last_error = Some("Audio engine unavailable on this machine".to_string());
            self.live_stream_playback = false;
        }
    }

    pub(super) fn shift_loop_by_seconds(&mut self, delta_sec: f32) -> bool {
        if !self.loop_enabled {
            return false;
        }

        let Some((a, b)) = self.loop_selection else {
            return false;
        };

        let start = a.min(b);
        let end = a.max(b);
        let loop_len = end - start;
        if loop_len <= LOOP_MIN_DURATION_SEC {
            return false;
        }

        let duration = self.timeline_duration_sec();
        let max_start = (duration - loop_len).max(0.0);
        let new_start = (start + delta_sec).clamp(0.0, max_start);
        let new_end = (new_start + loop_len).min(duration);

        self.loop_selection = Some((new_start, new_end));
        self.loop_playback_enabled = true;
        self.selected_time_sec = (self.selected_time_sec + delta_sec).clamp(new_start, new_end);
        self.update_note_probabilities(true);

        if self.is_playing() {
            self.play_range(self.selected_time_sec, Some(new_end));
        }

        true
    }

    pub(super) fn skip_by_seconds(&mut self, delta_sec: f32) {
        if self.audio_raw.is_none() {
            return;
        }

        let duration = self.timeline_duration_sec();
        let target = self.selected_time_sec + delta_sec;
        self.selected_time_sec = if self.loop_enabled {
            if let Some((a, b)) = self.loop_selection {
                let start = a.min(b);
                let end = a.max(b);
                if end - start > LOOP_MIN_DURATION_SEC {
                    target.clamp(start, end)
                } else {
                    target.clamp(0.0, duration)
                }
            } else {
                target.clamp(0.0, duration)
            }
        } else {
            target.clamp(0.0, duration)
        };
        self.update_note_probabilities(true);

        if self.is_playing() {
            if self.loop_enabled {
                if let Some((a, b)) = self.loop_selection {
                    let start = a.min(b);
                    let end = a.max(b);
                    if end - start > LOOP_MIN_DURATION_SEC {
                        self.loop_playback_enabled = true;
                        self.play_range(self.selected_time_sec, Some(end));
                        return;
                    }
                }
            }
            self.play_from_selected();
        }
    }

    pub(super) fn play_range(&mut self, start_sec: f32, end_sec: Option<f32>) {
        if self.play_preview_at(start_sec, end_sec) {
            self.live_stream_playback = false;
            return;
        }

        let available_duration = self.source_duration();
        if self.is_audio_loading
            && (start_sec > available_duration + 0.01 || self.processed_playback_samples.is_empty())
        {
            if self.play_loading_preview_from_source(start_sec, end_sec) {
                return;
            }
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };
        let playback_rate = self.playback_rate();

        let (playback_samples, channels, sample_rate) =
            if let Some((stems, ch, sr)) = self.active_stem_audio() {
                (stems, ch, sr)
            } else {
                (
                    Arc::new(self.processed_playback_samples.clone()),
                    self.processed_playback_channels,
                    raw.sample_rate,
                )
            };

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_range(
                &playback_samples,
                channels,
                sample_rate,
                start_sec,
                end_sec,
                playback_rate,
            ) {
                self.last_error = Some(format!("Playback error: {err}"));
                self.playing_preview_buffer = false;
                self.live_stream_playback = false;
            } else {
                self.playing_preview_buffer = false;
                self.live_stream_playback = false;
            }
        }
    }

    pub(super) fn handle_space_replay(&mut self) {
        if self.audio_raw.is_none() {
            return;
        }

        if self.loop_enabled {
            if let Some((a, b)) = self.loop_selection {
                let start = a.min(b);
                let end = a.max(b);
                if end - start > LOOP_MIN_DURATION_SEC {
                    self.loop_playback_enabled = true;
                    self.selected_time_sec = start;
                    self.play_range(start, Some(end));
                    return;
                }
            }
        }

        self.loop_playback_enabled = false;
        let is_playing = self.is_playing();

        if is_playing {
            if let Some(engine) = &mut self.engine {
                engine.pause();
            }
            return;
        }

        let current_pos = self.current_position_sec();

        if current_pos <= 0.0 {
            self.play_from_selected();
        } else if let Some(engine) = &mut self.engine {
            engine.resume();
        }
    }

    pub(super) fn handle_toggle_play_pause(&mut self) {
        if self.audio_raw.is_none() {
            return;
        }

        if self.is_playing() {
            if let Some(engine) = &mut self.engine {
                engine.pause();
            }
            return;
        }

        let can_resume_existing = self
            .engine
            .as_ref()
            .map(|engine| engine.has_active_sink())
            .unwrap_or(false);

        if self.loop_enabled {
            if let Some((a, b)) = self.loop_selection {
                let start = a.min(b);
                let end = a.max(b);
                if end - start > LOOP_MIN_DURATION_SEC {
                    self.loop_playback_enabled = true;

                    if can_resume_existing {
                        if let Some(engine) = &mut self.engine {
                            engine.resume();
                        }
                    } else {
                        let current_pos = self.current_position_sec();
                        let restart_from = if current_pos < start || current_pos >= end - 0.01 {
                            start
                        } else {
                            current_pos.clamp(start, end)
                        };
                        self.selected_time_sec = restart_from;
                        self.play_range(restart_from, Some(end));
                    }
                    return;
                }
            }
        }

        if can_resume_existing {
            if let Some(engine) = &mut self.engine {
                engine.resume();
            }
        } else {
            self.play_from_selected();
        }
    }

    pub(super) fn maybe_restart_playback_for_listen_sync(&mut self) {
        if !self.is_playing() {
            return;
        }

        if self.loop_enabled {
            if let Some((a, b)) = self.loop_selection {
                self.play_range(self.selected_time_sec, Some(b.max(a)));
                return;
            }
        }
        self.play_from_selected();
    }

    pub(super) fn stop(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.stop();
        }
        self.loop_playback_enabled = false;
        self.playing_preview_buffer = false;
        self.live_stream_playback = false;
    }

    pub(super) fn sync_playhead_from_engine(&mut self) {
        let param_render_in_progress = self.is_param_render_in_progress();
        if let Some(engine) = &mut self.engine {
            engine.sync_finished();
            if engine.is_playing() {
                self.selected_time_sec =
                    engine.current_position().min(self.timeline_duration_sec());
                self.update_note_probabilities(false);
            } else if self.loop_enabled && self.loop_playback_enabled {
                if let Some((a, b)) = self.loop_selection {
                    let start = a.min(b);
                    let end = a.max(b);
                    if end - start > LOOP_MIN_DURATION_SEC {
                        self.selected_time_sec = start;
                        if param_render_in_progress {
                            // Avoid repeatedly canceling/restarting parameter renders while looping.
                            self.restart_playback_after_processing = true;
                        } else {
                            self.play_range(start, Some(end));
                        }
                    }
                }
            } else {
                self.playing_preview_buffer = false;
                if !engine.has_active_sink() {
                    self.live_stream_playback = false;
                }
            }
        }
    }
}
