use super::*;

impl KeyScribeApp {
    pub(super) fn play_from_selected(&mut self) {
        if self.play_preview_at(self.selected_time_sec, None) {
            self.live_stream_playback = false;
            return;
        }

        let Some(raw) = &self.audio_raw else {
            return;
        };
        let playback_rate = self.playback_rate();

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_from(
                &self.processed_playback_samples,
                self.processed_playback_channels,
                raw.sample_rate,
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

        let duration = self.source_duration();
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
        if self.audio_raw.is_none() || self.processed_playback_samples.is_empty() {
            return;
        }

        let duration = self.source_duration();
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

        let Some(raw) = &self.audio_raw else {
            return;
        };
        let playback_rate = self.playback_rate();

        if let Some(engine) = &mut self.engine {
            if let Err(err) = engine.play_range(
                &self.processed_playback_samples,
                self.processed_playback_channels,
                raw.sample_rate,
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
        if self.audio_raw.is_none() || self.processed_playback_samples.is_empty() {
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
        if self.audio_raw.is_none() || self.processed_playback_samples.is_empty() {
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
                self.selected_time_sec = engine.current_position().min(self.source_duration());
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