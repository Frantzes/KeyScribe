use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::leadsheet::bpm::{detect_bpm, detect_bpm_from_audio, BpmDetectionConfig, TempoEstimate};
use crate::leadsheet::NoteEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatTrackResult {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BeatTrackConfig {
    pub model: String,
    pub device: BeatTrackDevice,
    pub dbn: bool,
}

impl Default for BeatTrackConfig {
    fn default() -> Self {
        Self {
            model: "final0".to_string(),
            device: BeatTrackDevice::Auto,
            dbn: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BeatTrackDevice {
    Auto,
    Cpu,
    Cuda,
}

impl BeatTrackDevice {
    fn as_str(self) -> &'static str {
        match self {
            BeatTrackDevice::Auto => "auto",
            BeatTrackDevice::Cpu => "cpu",
            BeatTrackDevice::Cuda => "cuda",
        }
    }
}

pub fn run_beat_this(audio_path: &Path, config: &BeatTrackConfig) -> Result<BeatTrackResult> {
    let mut results = run_beat_this_multi(&[audio_path], config)?;
    results.pop().ok_or_else(|| anyhow!("No result from beat tracker"))
}

/// Run beat_this on multiple audio files in a single Python process call.
/// The model is loaded once and reused across all files.
pub fn run_beat_this_multi(
    audio_paths: &[&Path],
    config: &BeatTrackConfig,
) -> Result<Vec<BeatTrackResult>> {
    if audio_paths.is_empty() {
        return Err(anyhow!("No audio files provided for beat tracking"));
    }
    for &p in audio_paths {
        if !p.is_file() {
            return Err(anyhow!("Beat tracking input is not a file: {}", p.display()));
        }
    }

    let python_exe = if cfg!(windows) {
        "python/src/.venv/Scripts/python.exe"
    } else {
        "python/src/.venv/bin/python"
    };
    let runner_script = "python/src/beat_this_runner.py";

    let mut cmd = Command::new(python_exe);
    cmd.arg(runner_script);
    for &p in audio_paths {
        cmd.arg(p);
    }
    cmd.arg("--model")
        .arg(&config.model)
        .arg("--device")
        .arg(config.device.as_str());

    if config.dbn {
        cmd.arg("--dbn");
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd.output().context("BeatThis process failed to start")?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Debug: save the raw output and errors from the Python model regardless of exit status
    let _ = std::fs::write("beat_this_debug.json", stdout.as_ref());
    let _ = std::fs::write("beat_this_debug.err", stderr.as_ref());

    if !output.status.success() {
        return Err(anyhow!(
            "BeatThis process failed with status {:?}: {}",
            output.status.code(),
            stderr.trim()
        ));
    }

    let mut results: Vec<BeatTrackResult> =
        serde_json::from_str(stdout.trim()).context("Failed to parse BeatThis JSON array")?;

    for r in &mut results {
        correct_beat_metric_level(r);
    }

    Ok(results)
}

fn temp_wav_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!("keyscribe_beat_this_{:020}.wav", ts));
    path
}

fn write_samples_to_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;
    let data_size = samples.len() as u32 * bytes_per_sample as u32;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * i16::MAX as f32) as i16;
        buf.extend_from_slice(&i16_sample.to_le_bytes());
    }

    std::fs::write(path, buf)?;
    Ok(())
}

/// Run beat_this on combined drum+bass stems (or fall back to full mix).
/// Writes the combined audio to a temporary WAV file for the Python model.
pub fn run_beat_this_combined(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
    sample_rate: u32,
    config: &BeatTrackConfig,
) -> Result<BeatTrackResult> {
    let combined = combined_audio(bass_samples, drum_samples, full_mix_samples)?;
    let temp_path = temp_wav_path();
    write_samples_to_wav(&temp_path, &combined, sample_rate)?;
    let result = run_beat_this(&temp_path, config);
    let _ = std::fs::remove_file(&temp_path);
    result
}

fn infer_beats_per_bar(downbeats: &[f32], beats: &[f32]) -> u32 {
    if downbeats.len() < 2 || beats.len() < 2 {
        return 4;
    }
    let dbi = median_of_values(downbeats);
    let bi = median_of_values(beats);
    if bi < 0.001 {
        return 4;
    }
    (dbi / bi).round() as u32
}

fn median_of_values(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let samples: Vec<f32> = if values.len() >= 2 {
        values.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0.001).collect()
    } else {
        return 0.5;
    };
    if samples.is_empty() {
        return 0.5;
    }
    let mut sorted = samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

impl From<BeatTrackResult> for CrossValidatedBeats {
    fn from(bt: BeatTrackResult) -> Self {
        let bi = median_of_values(&bt.beats);
        let bpb = infer_beats_per_bar(&bt.downbeats, &bt.beats).clamp(2, 8);
        Self {
            beats: bt.beats,
            downbeats: bt.downbeats,
            beats_per_bar: bpb,
            bpm: (60.0 / bi.max(0.001)).clamp(40.0, 260.0),
            confidence: 0.5,
            source_count: 1,
        }
    }
}

impl From<CrossValidatedBeats> for BeatTrackResult {
    fn from(cv: CrossValidatedBeats) -> Self {
        BeatTrackResult {
            beats: cv.beats,
            downbeats: cv.downbeats,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CrossValidatedBeats {
    /// Consensus downbeat positions (seconds).
    pub downbeats: Vec<f32>,
    /// Consensus beat positions (seconds).
    pub beats: Vec<f32>,
    /// Beats per bar inferred from the downbeat/beat intervals.
    pub beats_per_bar: u32,
    /// Median BPM across all sources.
    pub bpm: f32,
    /// Confidence 0.0–1.0 based on cross-source agreement.
    pub confidence: f32,
    /// Number of audio sources that contributed (1–3).
    pub source_count: u32,
}

/// Run beat-this on up to three audio sources (combined, drums-only, bass-only)
/// and cross-validate the results for robust downbeat detection.
///
/// Audio source priority:
///   1. drums + bass combined (always created if both stems present)
///   2. drums-only (if available)
///   3. bass-only (if available)
///   4. full mix (fallback when no stems)
///
/// Downbeats that appear in multiple sources are retained with higher confidence.
pub fn cross_validate_beat_sources(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
    sample_rate: u32,
    config: &BeatTrackConfig,
) -> Result<CrossValidatedBeats> {
    // ---- collect audio sources to analyse ----
    let mut source_labels: Vec<String> = Vec::new();
    let mut source_audios: Vec<Vec<f32>> = Vec::new();

    // always create the combined source (drums+bass or fallback)
    let combined = combined_audio(bass_samples, drum_samples, full_mix_samples)?;
    source_labels.push("combined".into());
    source_audios.push(combined);

    // individual drums
    if let Some(d) = drum_samples {
        source_labels.push("drums".into());
        source_audios.push(d.to_vec());
    }

    // individual bass
    if let Some(b) = bass_samples {
        source_labels.push("bass".into());
        source_audios.push(b.to_vec());
    }

    // ---- write temp WAV files ----
    let mut temp_paths: Vec<PathBuf> = Vec::with_capacity(source_audios.len());
    for samples in source_audios.iter() {
        let p = temp_wav_path();
        write_samples_to_wav(&p, samples, sample_rate)?;
        temp_paths.push(p);
    }

    // ---- run beat-this on all sources in one call ----
    let ref_paths: Vec<&Path> = temp_paths.iter().map(|p| p.as_path()).collect();
    let results = match run_beat_this_multi(&ref_paths, config) {
        Ok(r) => r,
        Err(e) => {
            for p in &temp_paths {
                let _ = std::fs::remove_file(p);
            }
            return Err(e);
        }
    };

    // clean up temp files
    for p in &temp_paths {
        let _ = std::fs::remove_file(p);
    }

    // ---- cross-validate ----
    cross_validate_results(&results, &source_labels)
}

fn combined_audio(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
) -> Result<Vec<f32>> {
    match (bass_samples, drum_samples) {
        (Some(b), Some(d)) => {
            let len = b.len().max(d.len());
            let mut buf = Vec::with_capacity(len);
            for i in 0..len {
                buf.push(b.get(i).copied().unwrap_or(0.0) + d.get(i).copied().unwrap_or(0.0));
            }
            Ok(buf)
        }
        (Some(b), None) => Ok(b.to_vec()),
        (None, Some(d)) => Ok(d.to_vec()),
        (None, None) => full_mix_samples
            .ok_or_else(|| anyhow!("No audio available for beat tracking"))
            .map(|s| s.to_vec()),
    }
}

fn cross_validate_results(
    results: &[BeatTrackResult],
    _labels: &[String],
) -> Result<CrossValidatedBeats> {
    if results.is_empty() {
        return Err(anyhow!("No beat tracking results to cross-validate"));
    }

    // ---- compute BPM from each source ----
    let mut bpms: Vec<f32> = Vec::new();
    for r in results {
        if r.beats.len() >= 2 {
            let intervals: Vec<f32> = r.beats.windows(2).map(|w| w[1] - w[0]).collect();
            let med = median_of(&intervals);
            if med > 0.001 {
                bpms.push(60.0 / med);
            }
        }
    }
    let bpm = if bpms.is_empty() {
        120.0
    } else {
        median_of(&mut bpms)
    };
    let beat_interval = 60.0 / bpm.max(1.0);

    // ---- cross-validate downbeats ----
    // Pair up downbeats across sources (within 70ms window)
    let tolerance = 0.070f32;
    let primary = &results[0]; // combined result is the reference

    let mut consensus_downbeats: Vec<f32> = Vec::new();
    let mut hit_counts: Vec<u32> = Vec::new();

    for &db in &primary.downbeats {
        let mut count = 1u32; // always present in combined
        // check other sources
        for other in &results[1..] {
            if other.downbeats.iter().any(|&od| (od - db).abs() < tolerance) {
                count += 1;
            }
        }
        consensus_downbeats.push(db);
        hit_counts.push(count);
    }

    // If less than 2 downbeats, beats-per-bar unknown; infer from BPM
    let beats_per_bar = if consensus_downbeats.len() >= 2 {
        let db_intervals: Vec<f32> = consensus_downbeats
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let med_db = median_of(&db_intervals);
        let bpb = (med_db / beat_interval).round() as u32;
        bpb.clamp(2, 8)
    } else {
        4
    };

    // ---- cross-validate beats ----
    // Use primary beats (combined source) as the reference
    let mut consensus_beats: Vec<f32> = primary.beats.clone();

    // If we have too few beats, generate from BPM grid
    if consensus_beats.len() < 4 && bpms.len() >= 1 {
        let duration = primary.beats.last().copied().unwrap_or(30.0) + 2.0;
        let mut t = 0.0f32;
        while t <= duration {
            consensus_beats.push(t);
            t += beat_interval;
        }
        consensus_beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        consensus_beats.dedup_by(|a, b| (*a - *b).abs() < 0.001);
    }

    // ---- confidence ----
    let source_count = results.len() as u32;
    let avg_hits = if consensus_downbeats.is_empty() {
        1.0
    } else {
        hit_counts.iter().sum::<u32>() as f32 / consensus_downbeats.len() as f32
    };
    let confidence = ((avg_hits - 1.0) / (source_count as f32 - 1.0).max(1.0)).clamp(0.0, 1.0);

    // ---- post-process downbeats for consistency ----
    postprocess_downbeats(&mut consensus_beats, &mut consensus_downbeats, beats_per_bar);

    Ok(CrossValidatedBeats {
        downbeats: consensus_downbeats,
        beats: consensus_beats,
        beats_per_bar,
        bpm,
        confidence,
        source_count,
    })
}

/// Post-process beat_this downbeats to fill gaps and ensure consistent bar spacing.
/// beat_this can sometimes miss downbeats in sections with weak percussion, leaving
/// large gaps between downbeats that cause wonky engraving with extended measures.
/// This function detects such gaps and inserts missing downbeats at regular intervals.
/// Also handles anacrusis by propagating a downbeat back to time 0 when the first
/// downbeat arrives after the music has already started.
fn postprocess_downbeats(beats: &mut Vec<f32>, downbeats: &mut Vec<f32>, beats_per_bar: u32) {
    if downbeats.len() < 2 || beats.len() < 4 {
        return;
    }

    // Compute median beat interval from the full beat sequence
    let bi: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0.001).collect();
    if bi.is_empty() {
        return;
    }
    let mut bi_sorted = bi.clone();
    bi_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_beat = bi_sorted[bi_sorted.len() / 2];
    let bar_duration = beats_per_bar as f32 * median_beat;

    // Build a new downbeat list ensuring no gap exceeds 1.5x the expected bar duration
    let mut new_downbeats: Vec<f32> = Vec::new();
    let tolerance = bar_duration * 1.5;

    // Handle anacrusis: if the first downbeat is far from time 0, insert a
    // downbeat at time 0 so the pickup notes have a proper bar reference.
    if downbeats[0] > bar_duration * 0.5 {
        new_downbeats.push(0.0);
    }

    for i in 0..downbeats.len() {
        let current = downbeats[i];
        new_downbeats.push(current);

        if i + 1 < downbeats.len() {
            let next = downbeats[i + 1];
            let gap = next - current;
            if gap > tolerance {
                // Insert missing downbeats at regular bar intervals
                let mut t = current + bar_duration;
                while t + median_beat < next {
                    new_downbeats.push(t);
                    // Also insert the beat positions that belong to these filled bars
                    for b in 1..beats_per_bar {
                        let beat_t = t + b as f32 * median_beat;
                        if beat_t < next && !beats.iter().any(|&x| (x - beat_t).abs() < median_beat * 0.3) {
                            beats.push(beat_t);
                        }
                    }
                    t += bar_duration;
                }
            }
        }
    }

    // Sort and deduplicate
    new_downbeats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    new_downbeats.dedup_by(|a, b| (*a - *b).abs() < median_beat * 0.3);
    *downbeats = new_downbeats;

    // Also fill in beats for the pickup region
    if downbeats.len() >= 2 && downbeats[0] < 0.001 {
        let first_real_db = downbeats[1];
        let mut t = median_beat;
        while t < first_real_db - median_beat * 0.3 {
            if !beats.iter().any(|&x| (x - t).abs() < median_beat * 0.3) {
                beats.push(t);
            }
            t += median_beat;
        }
    }

    beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    beats.dedup_by(|a, b| (*a - *b).abs() < median_beat * 0.3);
}

fn median_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn median_value(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Detect beats directly from note event onsets using autocorrelation.
/// This replaces the external beat-this Python model with a pure-Rust
/// algorithm that uses the transcribed note data for BPM detection.
pub fn detect_beats_from_notes(notes: &[NoteEvent]) -> Option<BeatTrackResult> {
    let config = BpmDetectionConfig::default();
    let tempo = detect_bpm(notes, config)?;

    let beat_duration = 60.0 / tempo.bpm;

    // Find first and last onset for phase alignment and range
    let first_onset = notes
        .iter()
        .filter_map(|n| {
            if n.start_time.is_finite() && n.start_time >= 0.0 {
                Some(n.start_time)
            } else {
                None
            }
        })
        .fold(f32::MAX, |a, b| a.min(b));

    let last_onset = notes
        .iter()
        .filter_map(|n| {
            if n.start_time.is_finite() && n.start_time >= 0.0 {
                Some(n.start_time)
            } else {
                None
            }
        })
        .fold(0.0f32, |a, b| a.max(b));

    if first_onset >= f32::MAX || last_onset <= 0.0 {
        return None;
    }

    // Align first beat: snap to the nearest beat grid position before first onset
    let phase = (first_onset / beat_duration).floor() * beat_duration;

    let end_time = last_onset + beat_duration;
    let mut beats: Vec<f32> = Vec::new();
    let mut t = phase;
    while t <= end_time + 1e-3 {
        beats.push(t);
        t += beat_duration;
    }

    if beats.len() < 4 {
        return None;
    }

    // Downbeats: every 4th beat (standard 4/4 time)
    let downbeats: Vec<f32> = beats.iter().step_by(4).copied().collect();

    Some(BeatTrackResult { beats, downbeats })
}

/// Detect beats from bass/drum stem audio for the most reliable BPM reference.
/// Falls back to full mix audio if no stems are available.
pub fn detect_beats_from_stems(
    bass_samples: Option<&[f32]>,
    drum_samples: Option<&[f32]>,
    full_mix_samples: Option<&[f32]>,
    sample_rate: u32,
    audio_duration_sec: f32,
) -> Option<(BeatTrackResult, String)> {
    let config = BpmDetectionConfig {
        min_bpm: 40.0,
        max_bpm: 200.0,
        ..Default::default()
    };

    // Try bass + drums combined first (best rhythmic reference)
    let audio = match (bass_samples, drum_samples) {
        (Some(b), Some(d)) => {
            let len = b.len().max(d.len());
            let mut combined = vec![0.0f32; len];
            for (i, &s) in b.iter().enumerate() {
                combined[i] += s;
            }
            for (i, &s) in d.iter().enumerate() {
                combined[i] += s;
            }
            Some((combined, "Bass + Drums".to_string()))
        }
        (Some(b), None) => Some((b.to_vec(), "Bass".to_string())),
        (None, Some(d)) => Some((d.to_vec(), "Drums".to_string())),
        (None, None) => full_mix_samples.map(|s| (s.to_vec(), "Full mix".to_string())),
    };

    let (samples, source_label) = audio?;
    let tempo = detect_bpm_from_audio(&samples, sample_rate, config)?;
    Some((
        generate_beats_from_tempo(&tempo, audio_duration_sec),
        source_label,
    ))
}

fn generate_beats_from_tempo(tempo: &TempoEstimate, duration_sec: f32) -> BeatTrackResult {
    let total_sec = duration_sec.max(10.0) + 2.0;
    let mut beats: Vec<f32> = Vec::new();
    let mut t = 0.0f32;
    while t <= total_sec + 1e-3 {
        beats.push(t);
        t += tempo.beat_duration_sec;
    }
    let downbeats: Vec<f32> = beats.iter().step_by(4).copied().collect();
    BeatTrackResult { beats, downbeats }
}

fn correct_beat_metric_level(result: &mut BeatTrackResult) {
    if result.beats.len() < 4 {
        return;
    }

    // Filter and sort beats
    result.beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result.beats.dedup_by(|a, b| (*a - *b).abs() < 1.0e-3);
    if result.beats.len() < 4 {
        return;
    }

    // Compute median beat interval
    let intervals: Vec<f32> = result.beats
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.001)
        .collect();
    if intervals.len() < 3 {
        return;
    }
    let mut intervals_copy = intervals.clone();
    let beat_interval = median_value(&mut intervals_copy);
    let bpm = 60.0 / beat_interval;

    // Use downbeats to cross-validate the metric level
    if result.downbeats.len() >= 2 {
        let db_intervals: Vec<f32> = result.downbeats
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&d| d > 0.001)
            .collect();
        if !db_intervals.is_empty() {
            let mut db_copy = db_intervals.clone();
            let db_interval = median_value(&mut db_copy);
            let beats_per_bar = (db_interval / beat_interval).round();

            // If beats-per-bar is implausible (< 2.5 or > 6.0), the metric level is wrong.
            // Try doubling (half-time) or halving (double-time) the beat count.
            if beats_per_bar < 2.5 && bpm < 70.0 {
                // Too few beats detected per measure → likely half-time
                // Double the number of beats by interpolating midpoints
                let mut new_beats = Vec::with_capacity(result.beats.len() * 2 - 1);
                for w in result.beats.windows(2) {
                    new_beats.push(w[0]);
                    new_beats.push((w[0] + w[1]) * 0.5);
                }
                new_beats.push(*result.beats.last().unwrap());
                result.beats = new_beats;
            } else if beats_per_bar > 6.0 && bpm > 160.0 {
                // Too many beats per bar → likely double-time
                // Halve the number of beats
                result.beats = result.beats.iter().step_by(2).copied().collect();
                if result.beats.len() < 4 {
                    result.beats = intervals_copy.into_iter().step_by(2).collect();
                }
            }
        }
    } else {
        // No downbeats: use simple BPM range heuristic
        if bpm < 50.0 {
            // Likely half-time: double the beat count
            let mut new_beats = Vec::with_capacity(result.beats.len() * 2 - 1);
            for w in result.beats.windows(2) {
                new_beats.push(w[0]);
                new_beats.push((w[0] + w[1]) * 0.5);
            }
            new_beats.push(*result.beats.last().unwrap());
            result.beats = new_beats;
        } else if bpm > 200.0 {
            // Likely double-time: halve the beat count
            result.beats = result.beats.iter().step_by(2).copied().collect();
        }
    }
}
