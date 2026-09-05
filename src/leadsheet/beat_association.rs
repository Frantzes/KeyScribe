use crate::leadsheet::types::BeatAlignedNote;

#[derive(Debug, Clone, Copy)]
pub struct BeatAssociationConfig {
    pub max_beat_distance_ratio: f32,
}

impl Default for BeatAssociationConfig {
    fn default() -> Self {
        Self {
            max_beat_distance_ratio: 3.0,
        }
    }
}

pub fn associate_notes_with_beats(
    notes: &[BeatAlignedNote],
) -> Vec<BeatAlignedNote> {
    let mut out: Vec<BeatAlignedNote> = notes.to_vec();
    out.sort_by(|a, b| {
        a.original_start_time
            .partial_cmp(&b.original_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });
    out
}

pub fn associate_notes_to_beat_grid(
    start_times: &[f32],
    pitches: &[u8],
    velocities: &[u8],
    end_times: &[f32],
    rearticulations: &[bool],
    beat_times: &[f32],
    downbeat_times: &[f32],
    _config: &BeatAssociationConfig,
) -> Vec<BeatAlignedNote> {
    if beat_times.len() < 2 || start_times.is_empty() {
        return Vec::new();
    }

    let mut beats: Vec<f32> = beat_times
        .iter()
        .copied()
        .filter(|t| t.is_finite())
        .collect();
    let mut downbeats: Vec<f32> = downbeat_times
        .iter()
        .copied()
        .filter(|t| t.is_finite())
        .collect();
    beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    beats.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    downbeats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    downbeats.dedup_by(|a, b| (*a - *b).abs() < 1e-4);

    let beats_per_bar = beats_per_bar_from_downbeats(&downbeats, &beats);
    let has_anacrusis = beats.first().copied().unwrap_or(0.0) < downbeats.first().copied().unwrap_or(0.0) - 0.001;
    let bar_shift = if has_anacrusis { 1 } else { 0 };

    let max_len = start_times.len().min(pitches.len()).min(velocities.len()).min(end_times.len());
    let mut aligned = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let onset = start_times[i].max(0.0);
        let end = end_times[i].max(onset);
        let pitch = pitches[i];
        let velocity = velocities[i];

        let (prev_beat, next_beat) = match find_surrounding_beats(onset, &beats) {
            Some(pair) => pair,
            None => continue,
        };

        let beat_duration_sec = (next_beat - prev_beat).max(0.001);
        let intra_beat_pos = ((onset - prev_beat) / beat_duration_sec).clamp(0.0, 1.0);

        // Beat-boundary snap: if the note is within a small tempo-adaptive
        // margin of the next beat, reassign it to beat_index+1 at pos=0.0.
        // This fixes notes that land 5-20 ms before a beat and get assigned
        // to the previous beat at pos≈0.96, which cascades into a wrong
        // bar_index. The threshold scales with tempo: at 60 BPM 8% = 80ms
        // (too wide), at 208 BPM 8% = 23ms (correct). Clamp to [0.04, 0.10].
        let beat_dur_ms = beat_duration_sec * 1000.0;
        let snap_threshold = (15.0 / beat_dur_ms).clamp(0.04, 0.10);
        let (adj_intra, adj_beat_bump) = if intra_beat_pos > (1.0 - snap_threshold) {
            (0.0f32, 1u32) // snap forward to next beat
        } else if intra_beat_pos < snap_threshold {
            (0.0f32, 0u32) // snap to current beat start
        } else {
            (intra_beat_pos, 0u32)
        };

        let raw_beat_index = (find_beat_index(onset, &beats) + adj_beat_bump)
            .min((beats.len() - 1) as u32);

        // If the note moved to the next beat, refresh the surrounding beat
        // pair so the stored timings match the assigned beat index.
        let (prev_beat, next_beat) = if adj_beat_bump > 0 {
            let p = beats
                .get(raw_beat_index as usize)
                .copied()
                .unwrap_or(prev_beat);
            let n = beats
                .get(raw_beat_index as usize + 1)
                .copied()
                .unwrap_or_else(|| p + beat_duration_sec);
            (p, n)
        } else {
            (prev_beat, next_beat)
        };
        let beat_duration_sec = (next_beat - prev_beat).max(0.001);

        let (bar_index, beat_offset_in_bar) = find_structural_position(
            raw_beat_index,
            &beats,
            &downbeats,
            beats_per_bar,
            bar_shift,
        );
        let _structural_beat_index = bar_index * beats_per_bar + beat_offset_in_bar;

        aligned.push(BeatAlignedNote {
            id: i as u32,
            pitch,
            velocity,
            channel: None,
            original_start_time: onset,
            original_end_time: end,
            beat_index: raw_beat_index,
            bar_index,
            intra_beat_pos: adj_intra,
            prev_beat_time: prev_beat,
            next_beat_time: next_beat,
            beat_duration_sec,
            is_rearticulation: rearticulations[i],
        });

    }

    aligned
}

fn find_surrounding_beats(time: f32, beats: &[f32]) -> Option<(f32, f32)> {
    if beats.len() < 2 {
        return None;
    }
    if time < beats[0] {
        return Some((beats[0], beats[1]));
    }
    if time >= beats[beats.len() - 1] {
        let last = beats[beats.len() - 1];
        let prev = beats[beats.len() - 2];
        let dur = (last - prev).max(0.001);
        return Some((last, last + dur));
    }
    for i in 0..beats.len() - 1 {
        if time >= beats[i] && time < beats[i + 1] {
            return Some((beats[i], beats[i + 1]));
        }
    }
    None
}

fn find_beat_index(time: f32, beats: &[f32]) -> u32 {
    if time < beats[0] {
        return 0;
    }
    for i in (0..beats.len() - 1).rev() {
        if time >= beats[i] {
            return i as u32;
        }
    }
    0
}

fn find_structural_position(raw_beat_index: u32, beats: &[f32], downbeats: &[f32], beats_per_bar: u32, bar_shift: u32) -> (u32, u32) {
    if downbeats.is_empty() {
        return (raw_beat_index / beats_per_bar, raw_beat_index % beats_per_bar);
    }

    let beat_time = beats.get(raw_beat_index as usize).copied().unwrap_or(0.0);

    let mut nearest_downbeat_idx = 0;
    let mut nearest_downbeat_time = downbeats[0];

    for (i, &db_time) in downbeats.iter().enumerate().rev() {
        if beat_time >= db_time - 0.001 {
            nearest_downbeat_idx = i as u32;
            nearest_downbeat_time = db_time;
            break;
        }
    }

    if beat_time < downbeats[0] - 0.001 {
        let beats_before = beats.iter().filter(|&&b| b >= beat_time - 0.001 && b < downbeats[0] - 0.001).count() as u32;
        let offset = if beats_before >= beats_per_bar {
            0
        } else {
            beats_per_bar - beats_before
        };
        return (0, offset % beats_per_bar);
    }

    let beats_since_downbeat = beats.iter().filter(|&&b| b >= nearest_downbeat_time - 0.001 && b < beat_time + 0.001).count() as u32;
    let offset = beats_since_downbeat.saturating_sub(1);

    (nearest_downbeat_idx + bar_shift, offset % beats_per_bar)
}
pub fn associate_note_events(
    notes: &[crate::leadsheet::NoteEvent],
    beat_times: &[f32],
    downbeat_times: &[f32],
) -> Vec<BeatAlignedNote> {
    let mut start_times = Vec::with_capacity(notes.len());
    let mut pitches = Vec::with_capacity(notes.len());
    let mut velocities = Vec::with_capacity(notes.len());
    let mut end_times = Vec::with_capacity(notes.len());
    let mut rearticulations = Vec::with_capacity(notes.len());
    for n in notes {
        start_times.push(n.start_time);
        pitches.push(n.pitch);
        velocities.push(n.velocity);
        end_times.push(n.end_time);
        rearticulations.push(n.is_rearticulation);
    }
    associate_notes_to_beat_grid(
        &start_times,
        &pitches,
        &velocities,
        &end_times,
        &rearticulations,
        beat_times,
        downbeat_times,
        &BeatAssociationConfig::default(),
    )
}

pub fn beats_per_bar_from_downbeats(downbeats: &[f32], beats: &[f32]) -> u32 {
    if downbeats.len() < 2 || beats.len() < 2 {
        return 4;
    }
    let mut intervals: Vec<u32> = Vec::new();
    for pair in downbeats.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let count = beats.iter().filter(|&&b| b >= start - 0.001 && b < end - 0.001).count() as u32;
        if count > 0 {
            intervals.push(count);
        }
    }
    if intervals.is_empty() {
        return 4;
    }
    intervals.sort();
    let mid = intervals.len() / 2;
    intervals[mid]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn associates_note_between_two_beats() {
        let beats = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let aligned = associate_notes_to_beat_grid(
            &[0.25], &[60], &[100], &[0.4], &[false],
            &beats, &[0.0, 2.0],
            &BeatAssociationConfig::default(),
        );
        assert_eq!(aligned.len(), 1);
        let note = &aligned[0];
        assert!((note.intra_beat_pos - 0.5).abs() < 0.01);
        assert_eq!(note.beat_index, 0);
    }

    #[test]
    fn note_on_beat_gets_zero_intra_position() {
        let beats = vec![0.0, 0.5, 1.0];
        let aligned = associate_notes_to_beat_grid(
            &[0.0], &[60], &[100], &[0.3], &[false],
            &beats, &[0.0],
            &BeatAssociationConfig::default(),
        );
        assert!((aligned[0].intra_beat_pos - 0.0).abs() < 0.01);
    }

    #[test]
    fn detects_four_beats_per_bar() {
        let beats = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let downbeats = vec![0.0, 2.0, 4.0];
        assert_eq!(beats_per_bar_from_downbeats(&downbeats, &beats), 4);
    }

    #[test]
    fn note_before_beat_snaps_forward_to_next_beat() {
        // onset=0.49 (2 ms before the 0.5 beat) => intra 0.98, must snap to
        // beat_index=1 at pos 0.0, not beat_index=0 at pos 0.98.
        let beats = vec![0.0, 0.5, 1.0];
        let aligned = associate_notes_to_beat_grid(
            &[0.49], &[60], &[100], &[0.6], &[false],
            &beats, &[0.0],
            &BeatAssociationConfig::default(),
        );
        assert_eq!(aligned.len(), 1);
        let note = &aligned[0];
        assert_eq!(note.beat_index, 1);
        assert!((note.intra_beat_pos - 0.0).abs() < 0.01);
    }

    #[test]
    fn note_just_after_beat_snaps_to_beat_start() {
        // onset=0.01 => intra 0.02, snap to beat_index=0 at pos 0.0.
        let beats = vec![0.0, 0.5, 1.0];
        let aligned = associate_notes_to_beat_grid(
            &[0.01], &[60], &[100], &[0.2], &[false],
            &beats, &[0.0],
            &BeatAssociationConfig::default(),
        );
        assert_eq!(aligned.len(), 1);
        let note = &aligned[0];
        assert_eq!(note.beat_index, 0);
        assert!((note.intra_beat_pos - 0.0).abs() < 0.01);
    }

    #[test]
    fn genuine_half_beat_note_unchanged() {
        let beats = vec![0.0, 0.5, 1.0];
        let aligned = associate_notes_to_beat_grid(
            &[0.25], &[60], &[100], &[0.4], &[false],
            &beats, &[0.0],
            &BeatAssociationConfig::default(),
        );
        assert_eq!(aligned.len(), 1);
        let note = &aligned[0];
        assert_eq!(note.beat_index, 0);
        assert!((note.intra_beat_pos - 0.5).abs() < 0.01);
    }

    #[test]
    fn slow_tempo_guard_keeps_far_note_unsnapped() {
        // 60 BPM => beat_dur 1.0s, snap_threshold clamped to 0.04. intra 0.91
        // is below 0.96 so the note stays at 0.91 (not snapped forward).
        let beats = vec![0.0, 1.0, 2.0];
        let aligned = associate_notes_to_beat_grid(
            &[0.91], &[60], &[100], &[1.2], &[false],
            &beats, &[0.0],
            &BeatAssociationConfig::default(),
        );
        assert_eq!(aligned.len(), 1);
        let note = &aligned[0];
        assert_eq!(note.beat_index, 0);
        assert!((note.intra_beat_pos - 0.91).abs() < 0.01);
    }
}
