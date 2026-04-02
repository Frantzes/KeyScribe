use signalsmith_stretch::Stretch;

pub fn apply_speed_and_pitch(
    samples: &[f32],
    sample_rate: u32,
    speed: f32,
    pitch_semitones: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let clamped_speed = speed.clamp(0.25, 4.0);
    let speed_is_unity = (clamped_speed - 1.0).abs() < 1.0e-4;
    let pitch_is_zero = pitch_semitones.abs() < 1.0e-4;

    if speed_is_unity && pitch_is_zero {
        return samples.to_vec();
    }

    if sample_rate == 0 {
        return samples.to_vec();
    }

    let mut stretch = Stretch::preset_default(1, sample_rate);
    stretch.set_transpose_factor_semitones(pitch_semitones, None);

    let target_frames = ((samples.len() as f32) / clamped_speed).round().max(1.0) as usize;
    let input_latency = stretch.input_latency();
    let output_latency = stretch.output_latency();
    let silence_out_len = if input_latency > 0 {
        ((input_latency as f32) / clamped_speed).ceil().max(1.0) as usize
    } else {
        0
    };

    if input_latency > 0 {
        let mut seek_input = vec![0.0f32; input_latency];
        let copy = input_latency.min(samples.len());
        seek_input[..copy].copy_from_slice(&samples[..copy]);
        stretch.seek(&seek_input, clamped_speed as f64);
    }

    const BLOCK_OUT: usize = 4096;
    let mut output = Vec::with_capacity(target_frames + BLOCK_OUT);
    let mut in_pos = 0usize;
    let mut rendered = 0usize;
    let max_in_len = ((BLOCK_OUT as f32) * clamped_speed).ceil().max(1.0) as usize;
    let scratch_in_len = max_in_len.max(input_latency.max(1));
    let scratch_out_len = BLOCK_OUT.max(output_latency.max(silence_out_len).max(1));
    let mut input_scratch = vec![0.0f32; scratch_in_len];
    let mut output_chunk = vec![0.0f32; scratch_out_len];
    let mut skip_front = output_latency;

    #[inline]
    fn push_chunk_with_skip(output: &mut Vec<f32>, chunk: &[f32], skip_front: &mut usize) {
        if chunk.is_empty() {
            return;
        }

        let skip = (*skip_front).min(chunk.len());
        *skip_front -= skip;
        output.extend_from_slice(&chunk[skip..]);
    }

    while rendered < target_frames {
        let out_len = (target_frames - rendered).min(BLOCK_OUT);
        let in_len = ((out_len as f32) * clamped_speed).ceil().max(1.0) as usize;

        if in_pos + in_len <= samples.len() {
            let input_chunk = &samples[in_pos..in_pos + in_len];
            in_pos += in_len;
            stretch.process(input_chunk, &mut output_chunk[..out_len]);
        } else {
            let available = samples.len().saturating_sub(in_pos).min(in_len);
            if available > 0 {
                input_scratch[..available].copy_from_slice(&samples[in_pos..in_pos + available]);
                in_pos += available;
            }
            if available < in_len {
                input_scratch[available..in_len].fill(0.0);
            }
            stretch.process(&input_scratch[..in_len], &mut output_chunk[..out_len]);
        }

        push_chunk_with_skip(&mut output, &output_chunk[..out_len], &mut skip_front);
        rendered += out_len;
    }

    if input_latency > 0 && silence_out_len > 0 {
        let input_chunk = &input_scratch[..input_latency];
        stretch.process(input_chunk, &mut output_chunk[..silence_out_len]);
        push_chunk_with_skip(
            &mut output,
            &output_chunk[..silence_out_len],
            &mut skip_front,
        );
    }

    if output_latency > 0 {
        let mut flushed = 0usize;
        while flushed < output_latency {
            let len = (output_latency - flushed).min(output_chunk.len());
            stretch.flush(&mut output_chunk[..len]);
            push_chunk_with_skip(&mut output, &output_chunk[..len], &mut skip_front);
            flushed += len;
        }
    }

    if output.len() < target_frames {
        output.resize(target_frames, 0.0);
    } else {
        output.truncate(target_frames);
    }

    output
}
