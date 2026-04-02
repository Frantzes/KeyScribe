#![allow(dead_code)]

/// Viterbi decoding for note smoothing
/// Implements Hidden Markov Model-based smoothing to eliminate glitches
/// and ghost notes in transcription

/// Configuration for Viterbi smoothing
#[derive(Debug, Clone)]
pub struct ViterbiConfig {
    /// Transition cost between different notes (lower = more smoothing)
    pub transition_cost: f32,
    /// Observation likelihood scaling factor
    pub likelihood_scale: f32,
    /// Minimum confidence threshold for activating a note
    pub confidence_threshold: f32,
}

impl Default for ViterbiConfig {
    fn default() -> Self {
        Self {
            transition_cost: 0.2,      // Penalize note changes
            likelihood_scale: 1.0,      // Scale observation probabilities
            confidence_threshold: 0.6,  // Only notes above this threshold are active
        }
    }
}

/// Viterbi decoder for polyphonic note sequences
pub struct ViterbiDecoder {
    config: ViterbiConfig,
}

impl ViterbiDecoder {
    /// Create a new Viterbi decoder
    pub fn new(config: ViterbiConfig) -> Self {
        Self { config }
    }

    /// Decode note probabilities using Viterbi algorithm
    /// 
    /// # Arguments
    /// * `note_probs_sequence` - Sequence of note probability vectors (time, num_notes)
    /// * `look_ahead_frames` - Number of frames to look ahead for smoothing
    /// 
    /// # Returns
    /// Smoothed binary note activations
    pub fn decode(
        &self,
        note_probs_sequence: &[Vec<f32>],
        _look_ahead_frames: usize,
    ) -> Vec<Vec<bool>> {
        if note_probs_sequence.is_empty() {
            return vec![];
        }

        let num_frames = note_probs_sequence.len();
        let num_notes = note_probs_sequence[0].len();

        if num_notes == 0 {
            return vec![];
        }

        // Compute forward pass with Viterbi algorithm
        let mut viterbi_matrices = vec![vec![f32::NEG_INFINITY; num_notes]; num_frames];
        let mut backpointers = vec![vec![0usize; num_notes]; num_frames];

        // Initialize first frame
        for note in 0..num_notes {
            viterbi_matrices[0][note] =
                (note_probs_sequence[0][note] * self.config.likelihood_scale).ln();
        }

        // Forward pass
        for frame in 1..num_frames {
            for current_note in 0..num_notes {
                let mut best_prev_score = f32::NEG_INFINITY;
                let mut best_prev_note = 0;

                for prev_note in 0..num_notes {
                    // Transition cost (penalize large note jumps)
                    let note_distance = ((current_note as i32 - prev_note as i32).abs()) as f32;
                    let transition_penalty = self.config.transition_cost * note_distance;

                    let score =
                        viterbi_matrices[frame - 1][prev_note] -
                        transition_penalty +
                        (note_probs_sequence[frame][current_note] * self.config.likelihood_scale).ln();

                    if score > best_prev_score {
                        best_prev_score = score;
                        best_prev_note = prev_note;
                    }
                }

                viterbi_matrices[frame][current_note] = best_prev_score;
                backpointers[frame][current_note] = best_prev_note;
            }
        }

        // Backward pass: extract best path
        let mut best_path = vec![0usize; num_frames];

        // Find best ending state
        let mut best_end_note = 0;
        let mut best_end_score = f32::NEG_INFINITY;
        for note in 0..num_notes {
            if viterbi_matrices[num_frames - 1][note] > best_end_score {
                best_end_score = viterbi_matrices[num_frames - 1][note];
                best_end_note = note;
            }
        }

        best_path[num_frames - 1] = best_end_note;

        // Backtrack
        for frame in (1..num_frames).rev() {
            best_path[frame - 1] = backpointers[frame][best_path[frame]];
        }

        // Convert decoded path to binary note activations with confidence thresholding
        let mut result = vec![vec![false; num_notes]; num_frames];

        for frame in 0..num_frames {
            let active_note = best_path[frame];

            // Apply confidence threshold
            if note_probs_sequence[frame][active_note] >= self.config.confidence_threshold {
                result[frame][active_note] = true;
            }
        }

        result
    }

    /// Decode with a look-ahead buffer for better smoothing
    /// This version waits for future frames before making decisions
    pub fn decode_with_lookahead(
        &self,
        note_probs_sequence: &[Vec<f32>],
        lookahead_frames: usize,
    ) -> Vec<Vec<bool>> {
        if note_probs_sequence.is_empty() {
            return vec![];
        }

        let num_frames = note_probs_sequence.len();
        let num_notes = note_probs_sequence[0].len();

        if num_notes == 0 {
            return vec![];
        }

        // Split into windows
        let window_size = 1 + lookahead_frames;
        let mut result = vec![vec![false; num_notes]; num_frames];

        for start_frame in 0..num_frames {
            let end_frame = (start_frame + window_size).min(num_frames);
            let window = &note_probs_sequence[start_frame..end_frame];

            if window.is_empty() {
                continue;
            }

            // Find most likely note in this window
            let mut note_scores: Vec<f32> = vec![0.0; num_notes];

            for frame in window {
                for (note, &prob) in frame.iter().enumerate() {
                    note_scores[note] += prob;
                }
            }

            // Find best note
            let best_note = note_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Apply to first frame if confidence is high
            if note_scores[best_note] > self.config.confidence_threshold * window.len() as f32 {
                result[start_frame][best_note] = true;
            }
        }

        result
    }

    /// Smooth note on/off timing using state persistence
    /// Helps eliminate rapid on/off flickering
    pub fn apply_temporal_smoothing(
        &self,
        notes: &[Vec<bool>],
        min_duration_frames: usize,
    ) -> Vec<Vec<bool>> {
        if notes.is_empty() {
            return notes.to_vec();
        }

        let num_frames = notes.len();
        let num_notes = notes[0].len();
        let mut result = vec![vec![false; num_notes]; num_frames];

        for note in 0..num_notes {
            let mut in_note = false;
            let mut note_start = 0;

            for frame in 0..num_frames {
                if notes[frame][note] && !in_note {
                    // Note onset
                    in_note = true;
                    note_start = frame;
                } else if !notes[frame][note] && in_note {
                    // Note offset
                    let duration = frame - note_start;
                    if duration >= min_duration_frames {
                        // Note was long enough, keep it
                        for t in note_start..frame {
                            result[t][note] = true;
                        }
                    }
                    in_note = false;
                }
            }

            // Handle note that extends to end of sequence
            if in_note {
                let duration = num_frames - note_start;
                if duration >= min_duration_frames {
                    for t in note_start..num_frames {
                        result[t][note] = true;
                    }
                }
            }
        }

        result
    }

    /// Extract note onsets (attack times) from note sequences
    pub fn extract_onsets(notes: &[Vec<bool>]) -> Vec<Vec<usize>> {
        if notes.is_empty() {
            return vec![];
        }

        let num_notes = notes[0].len();
        let mut onsets: Vec<Vec<usize>> = vec![vec![]; num_notes];

        for note in 0..num_notes {
            let mut was_off = true;
            for (frame, frame_notes) in notes.iter().enumerate() {
                if frame_notes[note] && was_off {
                    onsets[note].push(frame);
                    was_off = false;
                } else if !frame_notes[note] && !was_off {
                    was_off = true;
                }
            }
        }

        onsets
    }

    /// Extract note offsets (offset times) from note sequences
    pub fn extract_offsets(notes: &[Vec<bool>]) -> Vec<Vec<usize>> {
        if notes.is_empty() {
            return vec![];
        }

        let num_notes = notes[0].len();
        let mut offsets: Vec<Vec<usize>> = vec![vec![]; num_notes];

        for note in 0..num_notes {
            let mut was_on = false;
            for (frame, frame_notes) in notes.iter().enumerate() {
                if frame_notes[note] && !was_on {
                    was_on = true;
                } else if !frame_notes[note] && was_on {
                    offsets[note].push(frame);
                    was_on = false;
                }
            }
        }

        offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viterbi_simple() {
        let config = ViterbiConfig {
            transition_cost: 0.1,
            likelihood_scale: 1.0,
            confidence_threshold: 0.5,
        };

        let decoder = ViterbiDecoder::new(config);

        // Three frames, 3 notes
        let probs = vec![
            vec![0.9, 0.05, 0.05],  // Frame 0: note 0 likely
            vec![0.1, 0.8, 0.1],    // Frame 1: note 1 likely
            vec![0.95, 0.02, 0.03], // Frame 2: note 0 likely
        ];

        let result = decoder.decode(&probs, 0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_temporal_smoothing() {
        let config = ViterbiConfig::default();
        let decoder = ViterbiDecoder::new(config);

        // Flickering note (on, off, on for single frames)
        let notes = vec![
            vec![true, false],
            vec![false, false],
            vec![true, false],
            vec![false, false],
        ];

        let smoothed = decoder.apply_temporal_smoothing(&notes, 2);
        assert_eq!(smoothed.len(), 4);
        // Single-frame notes should be removed
        assert!(!smoothed[0][0]); // Too short
    }

    #[test]
    fn test_onset_extraction() {
        let notes = vec![
            vec![false, false],
            vec![true, false],
            vec![true, false],
            vec![false, true],
            vec![true, true],
        ];

        let onsets = ViterbiDecoder::extract_onsets(&notes);
        assert_eq!(onsets[0], vec![1, 4]);
        assert_eq!(onsets[1], vec![3]);
    }
}
