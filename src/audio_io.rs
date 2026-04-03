use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision, StandardTagKey, StandardVisualKey};
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
    pub samples_mono: Arc<Vec<f32>>,
    pub metadata: AudioMetadata,
}

pub enum StreamingAudioEvent {
    SourceHash(Option<String>),
    Started {
        sample_rate: u32,
        total_samples: Option<usize>,
        metadata: AudioMetadata,
    },
    Chunk {
        samples_mono: Vec<f32>,
        decoded_samples: usize,
        total_samples: Option<usize>,
    },
    Finished {
        sample_rate: u32,
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

        let channels = decoded.spec().channels.count();
        let mut sample_buffer =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buffer.copy_interleaved_ref(decoded);

        let data = sample_buffer.samples();

        if channels <= 1 {
            mono_samples.extend_from_slice(data);
        } else {
            for frame in data.chunks(channels) {
                let mut sum = 0.0;
                for &ch in frame {
                    sum += ch;
                }
                mono_samples.push(sum / channels as f32);
            }
        }
    }

    Ok(AudioData {
        sample_rate,
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

    if tx
        .send(StreamingAudioEvent::Started {
            sample_rate,
            total_samples,
            metadata: metadata.clone(),
        })
        .is_err()
    {
        return Ok(());
    }

    let target_chunk = chunk_samples.max(16_384);
    let mut decoded_samples = 0usize;
    let mut pending_chunk = Vec::<f32>::with_capacity(target_chunk);

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

        let channels = decoded.spec().channels.count();
        let mut sample_buffer =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buffer.copy_interleaved_ref(decoded);

        let data = sample_buffer.samples();

        if channels <= 1 {
            pending_chunk.extend_from_slice(data);
            decoded_samples = decoded_samples.saturating_add(data.len());
        } else {
            for frame in data.chunks(channels) {
                let mut sum = 0.0;
                for &ch in frame {
                    sum += ch;
                }
                pending_chunk.push(sum / channels as f32);
            }
            decoded_samples = decoded_samples.saturating_add(data.len() / channels.max(1));
        }

        if pending_chunk.len() >= target_chunk {
            let chunk = std::mem::take(&mut pending_chunk);
            if tx
                .send(StreamingAudioEvent::Chunk {
                    samples_mono: chunk,
                    decoded_samples,
                    total_samples,
                })
                .is_err()
            {
                return Ok(());
            }
        }
    }

    if !pending_chunk.is_empty() {
        if tx
            .send(StreamingAudioEvent::Chunk {
                samples_mono: pending_chunk,
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
        decoded_samples,
        total_samples,
        metadata,
    });

    Ok(())
}
