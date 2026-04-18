use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision, StandardTagKey, StandardVisualKey};
use symphonia::core::units::Time;
use symphonia::default::{get_codecs, get_probe};

#[derive(Clone, Default)]
pub struct AudioMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub artwork_bytes: Option<Vec<u8>>,
}

#[derive(Clone)]
pub struct AudioData {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_interleaved: Arc<Vec<f32>>,
    pub samples_mono: Arc<Vec<f32>>,
    pub metadata: AudioMetadata,
}

pub struct AudioPreviewChunk {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_interleaved: Vec<f32>,
}

pub enum StreamingAudioEvent {
    SourceHash(Option<String>),
    Started {
        sample_rate: u32,
        total_samples: Option<usize>,
        channels: u16,
        metadata: AudioMetadata,
    },
    Chunk {
        samples_mono: Vec<f32>,
        samples_interleaved: Vec<f32>,
        channels: u16,
        decoded_samples: usize,
        total_samples: Option<usize>,
    },
    Finished {
        sample_rate: u32,
        channels: u16,
        decoded_samples: usize,
        total_samples: Option<usize>,
        metadata: AudioMetadata,
    },
    Error(String),
}

fn first_non_empty(current: &mut Option<String>, candidate: String) {
    let trimmed = candidate.trim();
    if current.is_none() && !trimmed.is_empty() {
        *current = Some(trimmed.to_string());
    }
}

fn apply_metadata_revision(metadata: &mut AudioMetadata, revision: &MetadataRevision) {
    for tag in revision.tags() {
        let value = tag.value.to_string();

        match tag.std_key {
            Some(StandardTagKey::TrackTitle) => first_non_empty(&mut metadata.title, value),
            Some(StandardTagKey::Artist) | Some(StandardTagKey::Performer) => {
                first_non_empty(&mut metadata.artist, value)
            }
            Some(StandardTagKey::Album) => first_non_empty(&mut metadata.album, value),
            _ => {
                let key = tag.key.to_ascii_lowercase();
                if key.contains("title") {
                    first_non_empty(&mut metadata.title, value);
                } else if key.contains("artist") || key.contains("performer") {
                    first_non_empty(&mut metadata.artist, value);
                } else if key.contains("album") {
                    first_non_empty(&mut metadata.album, value);
                }
            }
        }
    }

    if metadata.artwork_bytes.is_none() {
        if let Some(best) = revision
            .visuals()
            .iter()
            .find(|v| matches!(v.usage, Some(StandardVisualKey::FrontCover)))
            .or_else(|| revision.visuals().iter().find(|v| v.media_type.starts_with("image/")))
            .or_else(|| revision.visuals().first())
        {
            metadata.artwork_bytes = Some(best.data.to_vec());
        }
    }
}

fn extract_audio_metadata(probed: &mut symphonia::core::probe::ProbeResult) -> AudioMetadata {
    let mut metadata = AudioMetadata::default();

    if let Some(mut probed_meta) = probed.metadata.get() {
        if let Some(revision) = probed_meta.skip_to_latest() {
            apply_metadata_revision(&mut metadata, revision);
        }
    }

    if let Some(revision) = probed.format.metadata().skip_to_latest() {
        apply_metadata_revision(&mut metadata, revision);
    }

    metadata
}

fn normalize_playback_channels(source_channels: usize) -> u16 {
    if source_channels <= 1 {
        1
    } else {
        2
    }
}

fn fold_multichannel_frame_to_stereo(frame: &[f32]) -> (f32, f32) {
    if frame.len() <= 1 {
        let mono = frame.first().copied().unwrap_or(0.0);
        return (mono, mono);
    }
    if frame.len() == 2 {
        return (frame[0], frame[1]);
    }

    let mut left_sum = 0.0f32;
    let mut left_count = 0usize;
    let mut right_sum = 0.0f32;
    let mut right_count = 0usize;

    for (idx, sample) in frame.iter().enumerate() {
        if idx % 2 == 0 {
            left_sum += *sample;
            left_count += 1;
        } else {
            right_sum += *sample;
            right_count += 1;
        }
    }

    let left = if left_count > 0 {
        left_sum / left_count as f32
    } else {
        0.0
    };
    let right = if right_count > 0 {
        right_sum / right_count as f32
    } else {
        left
    };

    (left, right)
}

#[allow(dead_code)]
pub fn load_audio_file(path: &Path) -> Result<AudioData> {
    let file = File::open(path).with_context(|| format!("Cannot open file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut probed = get_probe()
        .format(
            &Default::default(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Failed to probe audio format")?;

    let mut metadata = extract_audio_metadata(&mut probed);

    let format = &mut probed.format;
    let track = format
        .default_track()
        .context("No default audio track found")?;

    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Missing sample rate in codec params")?;

    let mut mono_samples = Vec::<f32>::new();
    let mut playback_samples = Vec::<f32>::new();
    let mut playback_channels = 1u16;

    loop {
        if let Some(revision) = format.metadata().skip_to_latest() {
            apply_metadata_revision(&mut metadata, revision);
        }

        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                return Err(anyhow::anyhow!(
                    "Decoder reset required; unsupported in this flow"
                ));
            }
            Err(err) => return Err(anyhow::anyhow!("Packet read error: {err}")),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(anyhow::anyhow!("Decode error: {err}")),
        };

        let channels = decoded.spec().channels.count().max(1);
        let mut sample_buffer =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buffer.copy_interleaved_ref(decoded);

        let data = sample_buffer.samples();
        playback_channels = normalize_playback_channels(channels);

        if channels <= 1 {
            mono_samples.extend_from_slice(data);
            playback_samples.extend_from_slice(data);
        } else if channels == 2 {
            playback_samples.extend_from_slice(data);
            for frame in data.chunks_exact(2) {
                mono_samples.push((frame[0] + frame[1]) * 0.5);
            }
        } else {
            for frame in data.chunks(channels) {
                let mut sum = 0.0;
                for &ch in frame {
                    sum += ch;
                }
                mono_samples.push(sum / channels as f32);

                let (left, right) = fold_multichannel_frame_to_stereo(frame);
                playback_samples.push(left);
                playback_samples.push(right);
            }
        }
    }

    Ok(AudioData {
        sample_rate,
        channels: playback_channels,
        samples_interleaved: Arc::new(playback_samples),
        samples_mono: Arc::new(mono_samples),
        metadata,
    })
}

pub fn load_audio_file_streaming(
    path: &Path,
    chunk_samples: usize,
    cancel_flag: Arc<AtomicBool>,
    tx: Sender<StreamingAudioEvent>,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("Cannot open file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut probed = get_probe()
        .format(
            &Default::default(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Failed to probe audio format")?;

    let mut metadata = extract_audio_metadata(&mut probed);

    let format = &mut probed.format;
    let track = format
        .default_track()
        .context("No default audio track found")?;

    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Missing sample rate in codec params")?;
    let total_samples = track
        .codec_params
        .n_frames
        .and_then(|v| usize::try_from(v).ok());
    let mut playback_channels = normalize_playback_channels(
        track
            .codec_params
            .channels
            .map(|channels| channels.count())
            .unwrap_or(1),
    );

    if tx
        .send(StreamingAudioEvent::Started {
            sample_rate,
            total_samples,
            channels: playback_channels,
            metadata: metadata.clone(),
        })
        .is_err()
    {
        return Ok(());
    }

    let target_chunk = chunk_samples.max(16_384);
    let mut decoded_samples = 0usize;
    let mut pending_chunk_mono = Vec::<f32>::with_capacity(target_chunk);
    let mut pending_chunk_interleaved = Vec::<f32>::with_capacity(
        target_chunk * playback_channels as usize,
    );

    loop {
        if cancel_flag.load(Ordering::Acquire) {
            return Ok(());
        }

        if let Some(revision) = format.metadata().skip_to_latest() {
            apply_metadata_revision(&mut metadata, revision);
        }

        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                return Err(anyhow::anyhow!(
                    "Decoder reset required; unsupported in this flow"
                ));
            }
            Err(err) => return Err(anyhow::anyhow!("Packet read error: {err}")),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(anyhow::anyhow!("Decode error: {err}")),
        };

        let channels = decoded.spec().channels.count().max(1);
        playback_channels = normalize_playback_channels(channels);
        let mut sample_buffer =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buffer.copy_interleaved_ref(decoded);

        let data = sample_buffer.samples();

        if channels <= 1 {
            pending_chunk_mono.extend_from_slice(data);
            pending_chunk_interleaved.extend_from_slice(data);
            decoded_samples = decoded_samples.saturating_add(data.len());
        } else if channels == 2 {
            pending_chunk_interleaved.extend_from_slice(data);
            for frame in data.chunks_exact(2) {
                pending_chunk_mono.push((frame[0] + frame[1]) * 0.5);
            }
            decoded_samples = decoded_samples.saturating_add(data.len() / 2);
        } else {
            for frame in data.chunks(channels) {
                let mut sum = 0.0;
                for &ch in frame {
                    sum += ch;
                }
                pending_chunk_mono.push(sum / channels as f32);

                let (left, right) = fold_multichannel_frame_to_stereo(frame);
                pending_chunk_interleaved.push(left);
                pending_chunk_interleaved.push(right);
            }
            decoded_samples = decoded_samples.saturating_add(data.len() / channels.max(1));
        }

        if pending_chunk_mono.len() >= target_chunk {
            let chunk_mono = std::mem::take(&mut pending_chunk_mono);
            let chunk_interleaved = std::mem::take(&mut pending_chunk_interleaved);
            if tx
                .send(StreamingAudioEvent::Chunk {
                    samples_mono: chunk_mono,
                    samples_interleaved: chunk_interleaved,
                    channels: playback_channels,
                    decoded_samples,
                    total_samples,
                })
                .is_err()
            {
                return Ok(());
            }
        }
    }

    if !pending_chunk_mono.is_empty() {
        if tx
            .send(StreamingAudioEvent::Chunk {
                samples_mono: pending_chunk_mono,
                samples_interleaved: pending_chunk_interleaved,
                channels: playback_channels,
                decoded_samples,
                total_samples,
            })
            .is_err()
        {
            return Ok(());
        }
    }

    let _ = tx.send(StreamingAudioEvent::Finished {
        sample_rate,
        channels: playback_channels,
        decoded_samples,
        total_samples,
        metadata,
    });

    Ok(())
}

pub fn load_audio_preview_chunk(
    path: &Path,
    start_sec: f32,
    duration_sec: f32,
) -> Result<AudioPreviewChunk> {
    let file = File::open(path).with_context(|| format!("Cannot open file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut probed = get_probe()
        .format(
            &Default::default(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Failed to probe audio format")?;

    let format = &mut probed.format;
    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .context("No default audio track found")?;
        (track.id, track.codec_params.clone())
    };

    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;

    let sample_rate = codec_params
        .sample_rate
        .context("Missing sample rate in codec params")?;
    let source_channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1)
        .max(1);
    let playback_channels = normalize_playback_channels(source_channels);

    let duration_sec = duration_sec.clamp(0.25, 30.0);
    let target_frames = (duration_sec * sample_rate as f32).ceil().max(1.0) as usize;
    let start_sec = start_sec.max(0.0);

    if start_sec > 0.0 {
        let seek_seconds = start_sec.floor() as u64;
        let seek_frac = (start_sec - seek_seconds as f32) as f64;
        let seek_target = SeekTo::Time {
            time: Time::new(seek_seconds, seek_frac),
            track_id: Some(track_id),
        };

        let _ = format.seek(SeekMode::Accurate, seek_target);
    }

    let mut out = Vec::<f32>::with_capacity(target_frames * playback_channels as usize);
    let mut decoded_frames = 0usize;

    while decoded_frames < target_frames {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                return Err(anyhow::anyhow!(
                    "Decoder reset required; unsupported in preview flow"
                ));
            }
            Err(err) => return Err(anyhow::anyhow!("Packet read error: {err}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(anyhow::anyhow!("Decode error: {err}")),
        };

        let channels = decoded.spec().channels.count().max(1);
        let mut sample_buffer =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buffer.copy_interleaved_ref(decoded);
        let data = sample_buffer.samples();

        if channels <= 1 {
            let available = data.len().min(target_frames.saturating_sub(decoded_frames));
            out.extend_from_slice(&data[..available]);
            decoded_frames = decoded_frames.saturating_add(available);
        } else if channels == 2 {
            let available_frames = (data.len() / 2).min(target_frames.saturating_sub(decoded_frames));
            out.extend_from_slice(&data[..available_frames * 2]);
            decoded_frames = decoded_frames.saturating_add(available_frames);
        } else {
            let remaining = target_frames.saturating_sub(decoded_frames);
            for frame in data.chunks(channels).take(remaining) {
                let (left, right) = fold_multichannel_frame_to_stereo(frame);
                out.push(left);
                out.push(right);
                decoded_frames += 1;
            }
        }
    }

    Ok(AudioPreviewChunk {
        sample_rate,
        channels: playback_channels,
        samples_interleaved: out,
    })
}
