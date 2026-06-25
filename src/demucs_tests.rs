//! Integration test for Demucs ONNX inference.
//! Runs only if models/htdemucs_6s.onnx exists.

#[cfg(test)]
mod integration {
    use super::super::*;

    #[test]
    fn test_demucs_single_chunk_basic() {
        let model_path = resolve_model_path("htdemucs_6s.onnx");
        if model_path.is_none() {
            eprintln!("Skipping test: htdemucs_6s.onnx not found");
            return;
        }

        let mut sep = DemucsSeparator::new("htdemucs_6s").expect("Failed to load model");

        // Simple 7.8s sine wave at 44100 Hz
        let n = SEGMENT_LENGTH;
        let mut stereo = vec![0.0f32; 2 * n];
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE as f32;
            let s = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            stereo[i * 2] = s;
            stereo[i * 2 + 1] = s * 0.9;
        }

        let stems = sep.separate_samples(&stereo, 2, None).expect("Separation failed");

        assert_eq!(stems.len(), 6, "Expected 6 stems");

        // Each stem should have the right length
        for stem in &stems {
            assert_eq!(stem.samples_interleaved.len(), 2 * n, "Stem {} wrong length", stem.stem_type.display_name());
            assert_eq!(stem.samples_mono.len(), n, "Stem {} wrong mono length", stem.stem_type.display_name());
            assert_eq!(stem.sample_rate, SAMPLE_RATE);
            assert_eq!(stem.channels, 2);
        }

        // Print stats for each stem
        for stem in &stems {
            let rms = (stem.samples_mono.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
            let max = stem.samples_mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            eprintln!("  {}: rms={:.4}, max={:.4}", stem.stem_type.display_name(), rms, max);
        }

        // At least one stem should have significant energy (the "other" stem for a tone)
        let total_rms: f32 = stems.iter().map(|s| {
            (s.samples_mono.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt()
        }).sum();
        assert!(total_rms > 0.01, "All stems are silent (total rms={})", total_rms);
    }

    #[test]
    fn test_demucs_multi_chunk() {
        let model_path = resolve_model_path("htdemucs_6s.onnx");
        if model_path.is_none() {
            eprintln!("Skipping test: htdemucs_6s.onnx not found");
            return;
        }

        let mut sep = DemucsSeparator::new("htdemucs_6s").expect("Failed to load model");

        // 30-second signal
        let n = SAMPLE_RATE as usize * 30;
        let mut stereo = vec![0.0f32; 2 * n];
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE as f32;
            let s = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                  + 0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            stereo[i * 2] = s;
            stereo[i * 2 + 1] = s * 0.9;
        }

        let stems = sep.separate_samples(&stereo, 2, None).expect("Separation failed");

        assert_eq!(stems.len(), 6);
        for stem in &stems {
            assert_eq!(stem.samples_interleaved.len(), 2 * n, "Stem {} wrong length", stem.stem_type.display_name());
        }

        // Check for NaN/Inf
        for stem in &stems {
            for &v in stem.samples_interleaved.iter() {
                assert!(v.is_finite(), "Stem {} has non-finite values", stem.stem_type.display_name());
            }
        }

        // Check for clicks at chunk boundaries
        let stride = STRIDE;
        for stem in &stems {
            let samples = &stem.samples_interleaved;
            for offset in (stride..n).step_by(stride) {
                let left = samples[(offset - 1) * 2];
                let right = samples[offset * 2];
                let diff = (right - left).abs();
                let local_max = samples[(offset.saturating_sub(10))..(offset + 10).min(n)]
                    .iter()
                    .map(|s| s.abs())
                    .fold(0.0f32, f32::max);
                if diff > 0.1 && diff > 10.0 * local_max + 0.01 {
                    eprintln!("  CLICK at {}s in {}: diff={:.4} local_max={:.4}",
                        offset as f32 / SAMPLE_RATE as f32,
                        stem.stem_type.display_name(), diff, local_max);
                }
            }
        }

        // Print stats
        for stem in &stems {
            let rms = (stem.samples_mono.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
            let max = stem.samples_mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            eprintln!("  {}: rms={:.4}, max={:.4}", stem.stem_type.display_name(), rms, max);
        }
    }
}
