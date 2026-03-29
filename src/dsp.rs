pub fn apply_speed_and_pitch(samples: &[f32], speed: f32, pitch_semitones: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let clamped_speed = speed.clamp(0.25, 4.0);

    // Tempo control: stretch/compress in time while mostly keeping perceived pitch.
    let tempo_adjusted = time_stretch_ola(samples, 1.0 / clamped_speed);

    if pitch_semitones.abs() < f32::EPSILON {
        return tempo_adjusted;
    }

    let pitch_ratio = 2.0_f32.powf(pitch_semitones / 12.0);

    // Pitch shift is approximated by resampling plus inverse time stretching.
    let resampled = resample_linear(&tempo_adjusted, pitch_ratio);
    time_stretch_ola(&resampled, pitch_ratio)
}

fn hann(n: usize, size: usize) -> f32 {
    if size <= 1 {
        return 1.0;
    }
    let phase = 2.0 * std::f32::consts::PI * n as f32 / (size - 1) as f32;
    0.5 - 0.5 * phase.cos()
}

fn time_stretch_ola(input: &[f32], stretch: f32) -> Vec<f32> {
    if input.len() < 1024 {
        return input.to_vec();
    }

    let stretch = stretch.clamp(0.25, 4.0);
    let win = 1024usize;
    let hop_in = 256usize;
    let hop_out = ((hop_in as f32) * stretch).max(1.0) as usize;

    let frames = (input.len().saturating_sub(win)) / hop_in + 1;
    let out_len = frames * hop_out + win + 1;

    let mut output = vec![0.0f32; out_len];
    let mut norm = vec![0.0f32; out_len];

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos + win <= input.len() {
        for i in 0..win {
            let w = hann(i, win);
            let dst = out_pos + i;
            output[dst] += input[in_pos + i] * w;
            norm[dst] += w * w;
        }
        in_pos += hop_in;
        out_pos += hop_out;
    }

    for (sample, n) in output.iter_mut().zip(norm.iter()) {
        if *n > 1.0e-6 {
            *sample /= *n;
        }
    }

    output
}

fn resample_linear(input: &[f32], ratio: f32) -> Vec<f32> {
    if input.len() < 2 {
        return input.to_vec();
    }

    let ratio = ratio.clamp(0.25, 4.0);
    let out_len = ((input.len() as f32) / ratio).max(2.0) as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src = (i as f32) * ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = src - i0 as f32;
        let v = input[i0] * (1.0 - frac) + input[i1] * frac;
        output.push(v);
    }

    output
}
