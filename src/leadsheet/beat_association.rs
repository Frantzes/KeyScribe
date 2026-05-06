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
    beat_times: &[f32],
    downbeat_times: &[f32],
    _config: &BeatAssociationConfig,
) -> Vec<BeatAlignedNote> {
    if beat_times.len() < 2 || start_times.is_empty() {
        return Vec::new();
    }

    let beats: Vec<f32> = beat_times
        .iter()
        .copied()
        .filter(|t| t.is_finite() && *t >= 0.0)
        .collect();
    let downbeats: Vec<f32> = downbeat_times
        .iter()
        .copied()
        .filter(|t| t.is_finite() && *t >= 0.0)
        .collect();

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

        let beat_index = find_beat_index(onset, &beats);
        let bar_index = find_bar_index_for_beat(beat_index, &beats, &downbeats);

        aligned.push(BeatAlignedNote {
            pitch,
            velocity,
            channel: None,
            original_start_time: onset,
            original_end_time: end,
            beat_index,
            bar_index,
            intra_beat_pos,
            prev_beat_time: prev_beat,
            next_beat_time: next_beat,
            beat_duration_sec,
        });
    }

    aligned
}

fn find_surrounding_beats(time: f32, beats: &[f32]) -> Option<(f32, f32)> {
    if time < beats[0] {
        return Some((beats[0] - (beats[1] - beats[0]), beats[0]));
    }
    if time >= beats[beats.len() - 1] {
        let last = beats[beats.len() - 1];
        let prev = if beats.len() >= 2 {
            beats[beats.len() - 2]
        } else {
            last - 0.5
        };
        let dur = (last - prev).max(0.001);
        return Some((last - dur, last + dur));
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

fn find_bar_index_for_beat(beat_index: u32, beats: &[f32], downbeats: &[f32]) -> u32 {
    if downbeats.is_empty() {
        return beat_index / 4;
    }
    let beat_time = beats.get(beat_index as usize).copied().unwrap_or(0.0);
    for i in (0..downbeats.len()).rev() {
        if beat_time >= downbeats[i] - 0.001 {
            return i as u32;
        }
    }
    0
}

pub fn associate_note_events(
    note_events: &[crate::leadsheet::NoteEvent],
    beat_times: &[f32],
    downbeat_times: &[f32],
) -> Vec<BeatAlignedNote> {
    let start_times: Vec<f32> = note_events.iter().map(|n| n.start_time).collect();
    let pitches: Vec<u8> = note_events.iter().map(|n| n.pitch).collect();
    let velocities: Vec<u8> = note_events.iter().map(|n| n.velocity).collect();
    let end_times: Vec<f32> = note_events.iter().map(|n| n.end_time).collect();

    associate_notes_to_beat_grid(
        &start_times,
        &pitches,
        &velocities,
        &end_times,
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
            &[0.25], &[60], &[100], &[0.4],
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
            &[0.0], &[60], &[100], &[0.3],
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
}
