use super::*;
use crate::core::processing::build_waveform_for_processed;

impl TranscriberApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_cached_timeline_for_variant(
        song_hash: &str,
        sample_rate: u32,
        raw_sample_len: usize,
        audio_quality_mode: AudioQualityMode,
        speed: f32,
        pitch_semitones: f32,
        use_cqt_analysis: bool,
        preprocess_audio: bool,
    ) -> (Option<(Arc<Vec<Vec<f32>>>, f32)>, CachePrecheckDiagnostics) {
        let mut diag = CachePrecheckDiagnostics::default();

        let variant_key = analysis_cache_variant_key(
            sample_rate,
            raw_sample_len,
            audio_quality_mode,
            speed,
            pitch_semitones,
            use_cqt_analysis,
            preprocess_audio,
        );

        let expected_speed_bits = speed.to_bits();
        let expected_pitch_bits = pitch_semitones.to_bits();

        let strict_paths = analysis_cache_candidate_file_paths(song_hash, &variant_key);
        let strict_count = strict_paths.len();
        let mut candidate_paths = strict_paths;
        for path in analysis_cache_song_file_paths(song_hash) {
            if !candidate_paths.iter().any(|existing| existing == &path) {
                candidate_paths.push(path);
            }
        }
        diag.total_candidates = candidate_paths.len();

        for (idx, cache_path) in candidate_paths.into_iter().enumerate() {
            if !cache_path.is_file() {
                continue;
            }
            diag.existing_files += 1;

            let Ok(bytes) = fs::read(cache_path) else {
                diag.read_failures += 1;
                continue;
            };
            let cache = match decode_analysis_cache_blob(bytes.as_slice()) {
                Ok(cache) => cache,
                Err(CacheBlobDecodeFailure::Decompress) => {
                    diag.decompress_failures += 1;
                    continue;
                }
                Err(CacheBlobDecodeFailure::Deserialize) => {
                    diag.deserialize_failures += 1;
                    continue;
                }
            };
            diag.parsed_blobs += 1;

            let cache_matches_shared = cache.cache_version == ANALYSIS_CACHE_VERSION
                && cache.sample_rate == sample_rate
                && cache.audio_quality_mode_code == audio_quality_mode.cache_code()
                && cache.speed_bits == expected_speed_bits
                && cache.pitch_bits == expected_pitch_bits
                && cache.use_cqt_analysis == use_cqt_analysis
                && cache.preprocess_audio == preprocess_audio;
            if !cache_matches_shared {
                diag.shared_param_mismatches += 1;
                continue;
            }

            // Strict pass (variant-key path) enforces exact raw length when known;
            // fallback pass over all blobs for this song hash tolerates metadata drift.
            if raw_sample_len > 0 && idx < strict_count && cache.raw_sample_len != raw_sample_len {
                diag.strict_len_mismatches += 1;
                continue;
            }

            if !cache.base_note_timeline_step_sec.is_finite()
                || cache.base_note_timeline_step_sec <= 0.0
                || !validate_cached_note_timeline(cache.base_note_timeline.as_slice())
            {
                diag.invalid_timeline_blobs += 1;
                continue;
            }

            return (
                Some((
                    Arc::new(cache.base_note_timeline),
                    cache.base_note_timeline_step_sec,
                )),
                diag,
            );
        }

        (None, diag)
    }

    pub(super) fn maybe_precheck_analysis_cache(&mut self) {
        if self.cache_precheck_done || !self.is_audio_loading || !self.preprocess_audio {
            return;
        }

        let Some(song_hash) = self.loaded_audio_hash.as_deref() else {
            return;
        };
        let raw_sample_len_opt = self.loading_total_samples;
        let raw_sample_len = raw_sample_len_opt.unwrap_or(0);
        if self.loading_sample_rate == 0 {
            return;
        }

        self.cache_precheck_done = true;

        let (cached_timeline, precheck_diag) = Self::load_cached_timeline_for_variant(
            song_hash,
            self.loading_sample_rate,
            raw_sample_len,
            self.audio_quality_mode,
            self.speed,
            self.pitch_semitones,
            self.use_cqt_analysis,
            self.preprocess_audio,
        );

        if let Some((base_timeline, base_step_sec)) = cached_timeline {
            self.base_note_timeline = Arc::clone(&base_timeline);
            self.base_note_timeline_step_sec = base_step_sec;
            let (note_timeline, note_step_sec) = Self::transform_note_timeline(
                base_timeline,
                base_step_sec,
                self.speed,
                self.pitch_semitones,
            );
            self.note_timeline = note_timeline;
            self.note_timeline_step_sec = note_step_sec;
            self.loading_cache_timeline_preloaded = true;
            self.update_note_probabilities(true);
            self.cache_status_message =
                Some("Analysis cache: transcription loaded during render.".to_string());
        } else {
            self.loading_cache_timeline_preloaded = false;
            let hash_short = &song_hash[..song_hash.len().min(8)];
            let mismatch_total = precheck_diag.shared_param_mismatches
                + precheck_diag.strict_len_mismatches
                + precheck_diag.invalid_timeline_blobs;
            let decode_failures = precheck_diag.read_failures
                + precheck_diag.decompress_failures
                + precheck_diag.deserialize_failures;
            self.cache_status_message = Some(if raw_sample_len_opt.is_none() {
                format!(
                    "Analysis cache: early hit not confirmed yet (missing frame metadata, hash {hash_short}, blobs {} / parsed {}).",
                    precheck_diag.existing_files,
                    precheck_diag.parsed_blobs
                )
            } else if precheck_diag.existing_files == 0 {
                format!(
                    "Analysis cache: no early hit (hash {hash_short}, no cache blobs found for this hash)."
                )
            } else {
                format!(
                    "Analysis cache: no early hit (hash {hash_short}, blobs {}, parsed {}, mismatches {}, failures {} [read {}, decompress {}, deserialize {}]).",
                    precheck_diag.existing_files,
                    precheck_diag.parsed_blobs,
                    mismatch_total,
                    decode_failures,
                    precheck_diag.read_failures,
                    precheck_diag.decompress_failures,
                    precheck_diag.deserialize_failures
                )
            });
        }
        self.cache_status_message_at = Some(Instant::now());
    }

    #[allow(dead_code)]
    pub(super) fn try_restore_analysis_cache(&mut self, source_path: &Path) -> bool {
        let Some(audio) = &self.audio_raw else {
            return false;
        };
        let sample_rate = audio.sample_rate;
        let raw_sample_len = audio.samples_mono.len();
        let raw_samples = Arc::clone(&audio.samples_mono);

        let song_hash = if let Some(hash) = self.loaded_audio_hash.clone() {
            hash
        } else {
            let Some(hash) = compute_file_hash(source_path) else {
                return false;
            };
            self.loaded_audio_hash = Some(hash.clone());
            hash
        };

        let Some((processed_samples, base_note_timeline, base_note_timeline_step_sec)) =
            Self::load_analysis_cache_for_variant(
                song_hash.as_str(),
                sample_rate,
                raw_sample_len,
                raw_samples.as_slice(),
                self.audio_quality_mode,
                self.speed,
                self.pitch_semitones,
                self.use_cqt_analysis,
                self.preprocess_audio,
            )
        else {
            return false;
        };

        if self.preprocess_audio
            && (base_note_timeline.is_empty() || base_note_timeline_step_sec <= 0.0)
        {
            return false;
        }

        let (note_timeline, note_timeline_step_sec) = if self.preprocess_audio {
            Self::transform_note_timeline(
                Arc::clone(&base_note_timeline),
                base_note_timeline_step_sec,
                self.speed,
                self.pitch_semitones,
            )
        } else {
            (Arc::new(Vec::new()), 0.0)
        };

        self.processed_samples = processed_samples;
        self.waveform = build_waveform_for_processed(
            self.processed_samples.as_slice(),
            sample_rate,
            self.audio_quality_mode.waveform_points(),
            self.speed,
        );
        self.note_timeline = note_timeline;
        self.note_timeline_step_sec = note_timeline_step_sec;
        self.base_note_timeline = base_note_timeline;
        self.base_note_timeline_step_sec = base_note_timeline_step_sec;
        self.waveform_reset_view = true;
        self.selected_time_sec = self.selected_time_sec.min(self.source_duration());
        self.playing_preview_buffer = false;
        self.update_note_probabilities(true);

        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_analysis_cache_for_variant(
        song_hash: &str,
        sample_rate: u32,
        raw_sample_len: usize,
        raw_samples: &[f32],
        audio_quality_mode: AudioQualityMode,
        speed: f32,
        pitch_semitones: f32,
        use_cqt_analysis: bool,
        preprocess_audio: bool,
    ) -> Option<(Vec<f32>, Arc<Vec<Vec<f32>>>, f32)> {
        let variant_key = analysis_cache_variant_key(
            sample_rate,
            raw_sample_len,
            audio_quality_mode,
            speed,
            pitch_semitones,
            use_cqt_analysis,
            preprocess_audio,
        );
        let expected_speed_bits = speed.to_bits();
        let expected_pitch_bits = pitch_semitones.to_bits();

        for cache_path in analysis_cache_candidate_file_paths(song_hash, &variant_key) {
            let Ok(bytes) = fs::read(cache_path) else {
                continue;
            };
            let Ok(cache) = decode_analysis_cache_blob(bytes.as_slice()) else {
                continue;
            };

            if !cache.base_note_timeline_step_sec.is_finite()
                || !validate_cached_note_timeline(cache.base_note_timeline.as_slice())
            {
                continue;
            }

            let cache_matches = cache.cache_version == ANALYSIS_CACHE_VERSION
                && cache.sample_rate == sample_rate
                && cache.raw_sample_len == raw_sample_len
                && cache.audio_quality_mode_code == audio_quality_mode.cache_code()
                && cache.speed_bits == expected_speed_bits
                && cache.pitch_bits == expected_pitch_bits
                && cache.use_cqt_analysis == use_cqt_analysis
                && cache.preprocess_audio == preprocess_audio;

            if !cache_matches {
                continue;
            }

            let processed_samples = if let Some(shuffled) = cache.processed_samples_shuffled_bytes {
                let max_processed_len = raw_sample_len.saturating_mul(8);
                if cache.processed_samples_len == 0
                    || cache.processed_samples_len > max_processed_len
                {
                    continue;
                }
                let Some(samples) = unshuffle_f32_bytes(&shuffled, cache.processed_samples_len)
                else {
                    continue;
                };
                if samples.is_empty() {
                    continue;
                }
                samples
            } else if speed_pitch_is_identity(speed, pitch_semitones) {
                raw_samples.to_vec()
            } else {
                continue;
            };

            let base_note_timeline = Arc::new(cache.base_note_timeline);
            let base_note_timeline_step_sec = cache.base_note_timeline_step_sec;

            if preprocess_audio
                && (base_note_timeline.is_empty() || base_note_timeline_step_sec <= 0.0)
            {
                continue;
            }

            return Some((
                processed_samples,
                base_note_timeline,
                base_note_timeline_step_sec,
            ));
        }

        None
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn persist_analysis_cache(
        song_hash: &str,
        sample_rate: u32,
        raw_sample_len: usize,
        audio_quality_mode: AudioQualityMode,
        speed: f32,
        pitch_semitones: f32,
        use_cqt_analysis: bool,
        preprocess_audio: bool,
        processed_samples: &[f32],
        base_note_timeline: &[Vec<f32>],
        base_note_timeline_step_sec: f32,
    ) {
        if processed_samples.is_empty() {
            return;
        }

        if preprocess_audio && (base_note_timeline.is_empty() || base_note_timeline_step_sec <= 0.0)
        {
            return;
        }

        let variant_key = analysis_cache_variant_key(
            sample_rate,
            raw_sample_len,
            audio_quality_mode,
            speed,
            pitch_semitones,
            use_cqt_analysis,
            preprocess_audio,
        );
        let song_dir = analysis_cache_library_dir().join(song_hash);
        if fs::create_dir_all(&song_dir).is_err() {
            return;
        }

        let store_processed = !speed_pitch_is_identity(speed, pitch_semitones);
        let (processed_samples_len, processed_samples_shuffled_bytes) = if store_processed {
            (
                processed_samples.len(),
                Some(shuffle_f32_bytes(processed_samples)),
            )
        } else {
            (0usize, None)
        };

        let snapshot = AnalysisCacheSnapshot {
            cache_version: ANALYSIS_CACHE_VERSION,
            sample_rate,
            raw_sample_len,
            audio_quality_mode_code: audio_quality_mode.cache_code(),
            speed_bits: speed.to_bits(),
            pitch_bits: pitch_semitones.to_bits(),
            use_cqt_analysis,
            preprocess_audio,
            processed_samples_len,
            processed_samples_shuffled_bytes: processed_samples_shuffled_bytes.as_deref(),
            base_note_timeline,
            base_note_timeline_step_sec,
        };

        let Ok(serialized) = bincode::serialize(&snapshot) else {
            return;
        };

        let Ok(compressed) = zstd::bulk::compress(&serialized, ANALYSIS_CACHE_ZSTD_LEVEL) else {
            return;
        };
        if compressed.len() > ANALYSIS_CACHE_MAX_COMPRESSED_BYTES {
            return;
        }

        let cache_path = analysis_cache_primary_file_path(song_hash, &variant_key);
        if ensure_parent_dir(cache_path.as_path()) {
            let _ = fs::write(cache_path, compressed);
        }
    }
}
