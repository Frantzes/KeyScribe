use super::*;

#[derive(Default)]
pub(super) struct CachePrecheckDiagnostics {
    total_candidates: usize,
    existing_files: usize,
    parsed_blobs: usize,
    read_failures: usize,
    decompress_failures: usize,
    deserialize_failures: usize,
    shared_param_mismatches: usize,
    strict_len_mismatches: usize,
    invalid_timeline_blobs: usize,
}

/// Result of the background analysis-cache precheck (see
/// [`KeyScribeApp::maybe_precheck_analysis_cache`]). Everything needed to
/// apply the outcome is captured at spawn time so the poll side never
/// touches the filesystem.
pub(super) struct CachePrecheckResult {
    song_hash: String,
    raw_sample_len_opt: Option<usize>,
    sample_rate: u32,
    cached: Option<(
        Arc<Vec<Vec<f32>>>,
        f32,
        Option<Vec<[f64; 2]>>,
    )>,
    diag: CachePrecheckDiagnostics,
}

impl KeyScribeApp {
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
    ) -> (
        Option<(Arc<Vec<Vec<f32>>>, f32, Option<Vec<[f64; 2]>>)>,
        CachePrecheckDiagnostics,
    ) {
        let mut diag = CachePrecheckDiagnostics::default();

        let variant_key = analysis_cache_variant_key(
            sample_rate,
            audio_quality_mode,
            speed,
            pitch_semitones,
            use_cqt_analysis,
            preprocess_audio,
        );

        let strict_paths = analysis_cache_candidate_file_paths(song_hash, &variant_key);

        let expected_speed_bits = speed.to_bits();
        let expected_pitch_bits = pitch_semitones.to_bits();
        let _strict_count = strict_paths.len();
        let mut candidate_paths = strict_paths;
        for path in analysis_cache_song_file_paths(song_hash) {
            if !candidate_paths.iter().any(|existing| existing == &path) {
                candidate_paths.push(path);
            }
        }
        diag.total_candidates = candidate_paths.len();

        for (_idx, cache_path) in candidate_paths.into_iter().enumerate() {
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

            // Strict pass (variant-key path) enforces exact raw length when known.
            // When the writer stored the actual-decoded count and the reader only
            // has the container's metadata n_frames, a slight mismatch is normal.
            // Instead of hard-rejecting, check whether the cached timeline duration
            // is close enough to the expected duration to trust the cache.
            if raw_sample_len > 0 && cache.raw_sample_len != raw_sample_len {
                let expected_dur = raw_sample_len as f64 / sample_rate as f64;
                let cached_dur = cache.base_note_timeline.len() as f64
                    * cache.base_note_timeline_step_sec as f64;
                let drift = if expected_dur > 0.0 {
                    (cached_dur - expected_dur).abs() / expected_dur
                } else {
                    1.0
                };
                // Within 2 % and 2 seconds → still a valid match (just metadata
                // rounding). Anything beyond that is genuinely a different file.
                if drift > 0.02 && (cached_dur - expected_dur).abs() > 2.0 {
                    diag.strict_len_mismatches += 1;
                    continue;
                }
                // Fall through — accept the cache even though raw_sample_len
                // doesn't match exactly.
            }

            if !cache.base_note_timeline_step_sec.is_finite()
                || cache.base_note_timeline_step_sec <= 0.0
                || !validate_cached_note_timeline(cache.base_note_timeline.as_slice())
            {
                diag.invalid_timeline_blobs += 1;
                continue;
            }

            let cached_waveform = cache
                .waveform_points
                .as_deref()
                .filter(|points| validate_cached_waveform_points(points))
                .map(unpack_waveform_points);

            return (
                Some((
                    Arc::new(cache.base_note_timeline),
                    cache.base_note_timeline_step_sec,
                    cached_waveform,
                )),
                diag,
            );
        }

        (None, diag)
    }

    /// Kick off the analysis-cache precheck on a worker thread.
    ///
    /// The precheck reads candidate cache blobs from disk and runs zstd
    /// decompression + bincode deserialization over them. On a warm cache
    /// that is tens of milliseconds; on a cold/wrong-variant cache it can
    /// chew through megabytes of blobs — either way it must never run on
    /// the event loop, where a stall of a few seconds makes the compositor
    /// declare the app hung ("Application Not Responding"), typically first
    /// noticed right after minimizing or switching workspaces. The outcome
    /// is applied in [`Self::poll_cache_precheck_result`].
    pub(super) fn maybe_precheck_analysis_cache(&mut self) {
        if self.cache_precheck_done || !self.is_audio_loading || !self.preprocess_audio {
            return;
        }

        let Some(song_hash) = self.loaded_audio_hash.as_deref() else {
            return;
        };
        if self.loading_sample_rate == 0 {
            return;
        }

        self.cache_precheck_done = true;

        let song_hash = song_hash.to_string();
        let raw_sample_len_opt = self.loading_total_samples;
        let raw_sample_len = raw_sample_len_opt.unwrap_or(0);
        let sample_rate = self.loading_sample_rate;
        let audio_quality_mode = self.audio_quality_mode;
        let speed = self.speed;
        let pitch_semitones = self.pitch_semitones;
        let use_cqt_analysis = self.use_cqt_analysis;
        let preprocess_audio = self.preprocess_audio;

        let (tx, rx) = mpsc::channel::<CachePrecheckResult>();
        self.cache_precheck_rx = Some(rx);
        thread::spawn(move || {
            let (cached_timeline, precheck_diag) = Self::load_cached_timeline_for_variant(
                &song_hash,
                sample_rate,
                raw_sample_len,
                audio_quality_mode,
                speed,
                pitch_semitones,
                use_cqt_analysis,
                preprocess_audio,
            );
            let _ = tx.send(CachePrecheckResult {
                song_hash,
                raw_sample_len_opt,
                sample_rate,
                cached: cached_timeline,
                diag: precheck_diag,
            });
        });
    }

    /// Apply a finished background cache precheck, if any. Called once per
    /// frame; never blocks (single `try_recv`).
    pub(super) fn poll_cache_precheck_result(&mut self) {
        let Some(rx) = &self.cache_precheck_rx else {
            return;
        };
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                self.cache_precheck_rx = None;
                return;
            }
        };
        self.cache_precheck_rx = None;

        // A different file may have been loaded while the worker ran; stale
        // results must not clobber the new load's state.
        if self.loaded_audio_hash.as_deref() != Some(result.song_hash.as_str()) {
            return;
        }

        let song_hash = result.song_hash;
        let raw_sample_len_opt = result.raw_sample_len_opt;
        if let Some((base_timeline, base_step_sec, cached_waveform)) = result.cached {
            // If the cached timeline covers a drastically different duration than
            // what the container metadata reports (e.g. a cache from a short
            // alternate audio track vs the real full-length program track),
            // discard the stale blob so a full re-render runs.
            if raw_sample_len_opt.is_some() && result.sample_rate > 0 {
                let raw_sample_len = raw_sample_len_opt.unwrap_or(0);
                let expected_dur = raw_sample_len as f32 / result.sample_rate as f32;
                let cached_dur = base_timeline.len() as f32 * base_step_sec;
                let drift = if expected_dur > 0.0 {
                    (cached_dur - expected_dur).abs() / expected_dur
                } else {
                    1.0
                };
                // Only flag caches where the duration is off by more than 30 %
                // or 60 seconds absolute. This is deliberately coarse — we only
                // want to catch the "wrong audio track" scenario (e.g. 6 min vs
                // 69 min).  Normal rounding / encoder-padding differences are
                // well inside these bounds so valid caches are never zapped.
                if drift > 0.30 && (cached_dur - expected_dur).abs() > 60.0 {
                    self.loading_cache_timeline_preloaded = false;
                    self.loading_cache_waveform_preloaded = false;
                    self.cache_status_message = Some(
                        "Analysis cache: stale (duration mismatch), re-rendering."
                            .to_string(),
                    );
                    self.cache_status_message_at = Some(Instant::now());
                } else {
                    // Valid cache — apply it.
                    self.apply_prechecked_cache(
                        base_timeline,
                        base_step_sec,
                        cached_waveform,
                    );
                }
            } else {
                // No metadata frame count to validate against — trust the cache.
                self.apply_prechecked_cache(
                    base_timeline,
                    base_step_sec,
                    cached_waveform,
                );
            }
        } else {
            self.loading_cache_timeline_preloaded = false;
            self.loading_cache_waveform_preloaded = false;
            let hash_short = &song_hash[..song_hash.len().min(8)];
            let precheck_diag = result.diag;
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
                    "Analysis cache: first-time transcription — no cached data yet (hash {hash_short})."
                )
            } else {
                format!(
                    "Analysis cache: cache miss — re-analysing (hash {hash_short}, blobs {}, parsed {}, mismatches {}, failures {} [read {}, decompress {}, deserialize {}]).",
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

    fn apply_prechecked_cache(
        &mut self,
        base_timeline: Arc<Vec<Vec<f32>>>,
        base_step_sec: f32,
        cached_waveform: Option<Vec<[f64; 2]>>,
    ) {
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
        if let Some(waveform) = cached_waveform {
            self.set_waveform_data(waveform, true);
            self.loading_cache_waveform_preloaded = true;
        } else {
            self.loading_cache_waveform_preloaded = false;
        }
        self.update_note_probabilities(true);
        self.cache_status_message =
            Some("Analysis cache: transcription loaded during render.".to_string());
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
    ) -> Option<(Vec<f32>, Arc<Vec<Vec<f32>>>, f32, Option<Vec<[f64; 2]>>)> {
        let variant_key = analysis_cache_variant_key(
            sample_rate,
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

            let mut cache_matches = cache.cache_version == ANALYSIS_CACHE_VERSION
                && cache.sample_rate == sample_rate
                && cache.audio_quality_mode_code == audio_quality_mode.cache_code()
                && cache.speed_bits == expected_speed_bits
                && cache.pitch_bits == expected_pitch_bits
                && cache.use_cqt_analysis == use_cqt_analysis
                && cache.preprocess_audio == preprocess_audio;

            if cache_matches && raw_sample_len > 0 && cache.raw_sample_len != raw_sample_len {
                let expected_dur = raw_sample_len as f64 / sample_rate as f64;
                let cached_dur = cache.base_note_timeline.len() as f64
                    * cache.base_note_timeline_step_sec as f64;
                let drift = if expected_dur > 0.0 {
                    (cached_dur - expected_dur).abs() / expected_dur
                } else {
                    1.0
                };
                if drift > 0.02 && (cached_dur - expected_dur).abs() > 2.0 {
                    cache_matches = false;
                }
            }

            if !cache_matches {
                continue;
            }

            let cached_waveform = cache
                .waveform_points
                .as_deref()
                .filter(|points| validate_cached_waveform_points(points))
                .map(unpack_waveform_points);

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
                cached_waveform,
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
        waveform: &[[f64; 2]],
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
        let packed_waveform = pack_waveform_points(waveform);

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
            waveform_points: if packed_waveform.is_empty() {
                None
            } else {
                Some(packed_waveform.as_slice())
            },
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
