use crate::analysis::waveform_points;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreRebuildMode {
    Full,
    ParametersOnly,
    ParametersPreview,
}

pub fn build_waveform_for_processed(
    processed_samples: &[f32],
    sample_rate: u32,
    waveform_point_budget: usize,
    speed: f32,
) -> Vec<[f64; 2]> {
    let mut waveform = waveform_points(processed_samples, sample_rate, waveform_point_budget);
    let speed_for_waveform = speed.clamp(0.25, 4.0) as f64;
    if (speed_for_waveform - 1.0).abs() > f64::EPSILON {
        for pt in &mut waveform {
            pt[0] *= speed_for_waveform;
        }
    }
    waveform
}

pub fn estimate_processing_duration_sec(
    mode: CoreRebuildMode,
    raw_sample_len: usize,
    sample_rate: u32,
    preprocess_audio: bool,
    use_cqt_analysis: bool,
    quality_multiplier: f32,
    analysis_seconds_per_audio_second_ema: Option<f32>,
) -> f32 {
    let audio_sec = if sample_rate > 0 {
        raw_sample_len as f32 / sample_rate as f32
    } else {
        0.0
    };

    match mode {
        CoreRebuildMode::ParametersPreview => 0.25,
        CoreRebuildMode::ParametersOnly => (0.25 + audio_sec * 0.02).clamp(0.25, 8.0),
        CoreRebuildMode::Full => {
            if !preprocess_audio {
                (0.4 + audio_sec * 0.035).clamp(0.4, 30.0)
            } else {
                let default_per_audio = if use_cqt_analysis { 0.16 } else { 0.07 };
                let learned_per_audio =
                    analysis_seconds_per_audio_second_ema.unwrap_or(default_per_audio);
                let blended_per_audio =
                    (default_per_audio * 0.4 + learned_per_audio * 0.6) * quality_multiplier;

                (0.8 + audio_sec * blended_per_audio).clamp(0.8, 300.0)
            }
        }
    }
}
