use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::default::{get_codecs, get_probe};

#[derive(Clone)]
pub struct AudioData {
    pub sample_rate: u32,
    pub samples_mono: Arc<Vec<f32>>,
}

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
    })
}
