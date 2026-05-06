use crate::leadsheet::types::{MeterClass, TimeSignatureSegment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RhythmState {
    pub tempo_bin: u8,
    pub meter_class: MeterClass,
    pub phase: u8,
}

impl RhythmState {
    pub fn new(tempo_bin: u8, meter_class: MeterClass, phase: u8) -> Self {
        Self {
            tempo_bin,
            meter_class,
            phase,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JointRhythmConfig {
    pub tempo_min_bin: u8,
    pub tempo_max_bin: u8,
    pub tempo_bin_resolution: f32,
    pub max_phase_per_meter: u8,
    pub tempo_transition_penalty: f32,
    pub meter_switch_penalty: f32,
    pub phase_transition_penalty: f32,
}

impl Default for JointRhythmConfig {
    fn default() -> Self {
        Self {
            tempo_min_bin: 20,
            tempo_max_bin: 80,
            tempo_bin_resolution: 1.0,
            max_phase_per_meter: 12,
            tempo_transition_penalty: 0.3,
            meter_switch_penalty: 1.5,
            phase_transition_penalty: 0.1,
        }
    }
}

#[allow(dead_code)]
pub struct JointRhythmTracker {
    config: JointRhythmConfig,
    states: Vec<RhythmState>,
    state_index: Vec<std::collections::HashMap<RhythmState, usize>>,
    transition_costs: Vec<Vec<f32>>,
}

impl JointRhythmTracker {
    pub fn new(config: JointRhythmConfig) -> Self {
        let states = Self::build_states(&config);
        let state_index = Self::build_state_index(&states);
        let transition_costs = Self::build_transition_costs(&states, &config);

        Self {
            config,
            states,
            state_index,
            transition_costs,
        }
    }

    fn build_states(config: &JointRhythmConfig) -> Vec<RhythmState> {
        let mut states = Vec::new();
        
        let meter_classes = [
            MeterClass::SimpleDuple,
            MeterClass::SimpleTriple,
            MeterClass::SimpleQuadruple,
            MeterClass::CompoundDuple,
            MeterClass::CompoundQuadruple,
        ];

        for tempo_bin in config.tempo_min_bin..=config.tempo_max_bin {
            for &meter_class in &meter_classes {
                let beats_per_measure = match meter_class {
                    MeterClass::SimpleDuple => 2,
                    MeterClass::SimpleTriple => 3,
                    MeterClass::SimpleQuadruple => 4,
                    MeterClass::CompoundDuple => 6,
                    MeterClass::CompoundQuadruple => 12,
                    MeterClass::Other => 4,
                };
                
                for phase in 0..beats_per_measure.min(config.max_phase_per_meter) {
                    states.push(RhythmState::new(tempo_bin, meter_class, phase as u8));
                }
            }
        }

        states
    }

    fn build_state_index(states: &[RhythmState]) -> Vec<std::collections::HashMap<RhythmState, usize>> {
        let mut index = vec![std::collections::HashMap::new(); states.len()];
        for (i, state) in states.iter().enumerate() {
            index[i] = std::collections::HashMap::new();
            index[i].insert(*state, i);
        }
        index
    }

    fn build_transition_costs(states: &[RhythmState], config: &JointRhythmConfig) -> Vec<Vec<f32>> {
        let n = states.len();
        let mut costs = vec![vec![f32::INFINITY; n]; n];

        for i in 0..n {
            for j in 0..n {
                let from = &states[i];
                let to = &states[j];

                let tempo_cost = if from.tempo_bin == to.tempo_bin {
                    0.0
                } else {
                    let drift = (from.tempo_bin as i32 - to.tempo_bin as i32).abs() as f32;
                    drift * config.tempo_transition_penalty
                };

                let meter_cost = if from.meter_class == to.meter_class {
                    0.0
                } else {
                    config.meter_switch_penalty
                };

                let beats_per_measure = match from.meter_class {
                    MeterClass::SimpleDuple => 2i32,
                    MeterClass::SimpleTriple => 3i32,
                    MeterClass::SimpleQuadruple => 4i32,
                    MeterClass::CompoundDuple => 6i32,
                    MeterClass::CompoundQuadruple => 12i32,
                    MeterClass::Other => 4i32,
                };

                let phase_cost = if from.phase == to.phase {
                    0.0
                } else if from.phase == 0 && to.phase == (beats_per_measure - 1) as u8 {
                    config.phase_transition_penalty
                } else if to.phase == from.phase + 1 || (from.phase == (beats_per_measure - 1) as u8 && to.phase == 0) {
                    config.phase_transition_penalty
                } else {
                    0.5
                };

                costs[i][j] = tempo_cost + meter_cost + phase_cost;
            }
        }

        costs
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn state_at(&self, index: usize) -> Option<&RhythmState> {
        self.states.get(index)
    }

    pub fn transition_cost(&self, from_idx: usize, to_idx: usize) -> f32 {
        self.transition_costs.get(from_idx).and_then(|row| row.get(to_idx)).copied().unwrap_or(f32::INFINITY)
    }

    pub fn viterbi_decode(&self, observations: &[Vec<f32>], initial_state: usize) -> Vec<usize> {
        let num_frames = observations.len();
        if num_frames == 0 {
            return Vec::new();
        }

        let num_states = self.states.len();
        if num_states == 0 {
            return Vec::new();
        }

        let mut viterbi = vec![vec![f32::NEG_INFINITY; num_states]; num_frames];
        let mut backpointers = vec![vec![0usize; num_states]; num_frames];

        for s in 0..num_states {
            viterbi[0][s] = observations[0].get(s).copied().unwrap_or(f32::NEG_INFINITY);
        }

        if initial_state < num_states {
            viterbi[0][initial_state] += 10.0;
        }

        for frame in 1..num_frames {
            for to in 0..num_states {
                let mut best_prev = 0;
                let mut best_score = f32::NEG_INFINITY;

                for from in 0..num_states {
                    let transition = self.transition_cost(from, to);
                    let score = viterbi[frame - 1][from] - transition;
                    if score > best_score {
                        best_score = score;
                        best_prev = from;
                    }
                }

                let obs = observations[frame].get(to).copied().unwrap_or(0.0);
                viterbi[frame][to] = best_score + obs;
                backpointers[frame][to] = best_prev;
            }
        }

        let mut path = vec![0usize; num_frames];
        let mut best_final = 0;
        let mut best_final_score = f32::NEG_INFINITY;
        for s in 0..num_states {
            if viterbi[num_frames - 1][s] > best_final_score {
                best_final_score = viterbi[num_frames - 1][s];
                best_final = s;
            }
        }
        path[num_frames - 1] = best_final;

        for frame in (1..num_frames).rev() {
            path[frame - 1] = backpointers[frame][path[frame]];
        }

        path
    }
}

pub fn extract_downbeats_from_path(path: &[usize], states: &[RhythmState]) -> Vec<(usize, f32)> {
    let mut downbeats = Vec::new();
    let mut prev_was_downbeat = false;

    for (frame_idx, &state_idx) in path.iter().enumerate() {
        let state = &states[state_idx];
        if state.phase == 0 && !prev_was_downbeat {
            downbeats.push((frame_idx, state.tempo_bin as f32));
            prev_was_downbeat = true;
        } else if state.phase != 0 {
            prev_was_downbeat = false;
        }
    }

    downbeats
}

pub fn collapse_to_tempo_segments(
    path: &[usize],
    states: &[RhythmState],
    frame_duration_sec: f32,
    segment_min_duration_sec: f32,
) -> Vec<(f32, f32, u8)> {
    if path.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_start_frame = 0;
    let mut current_tempo_bin = states[path[0]].tempo_bin;

    for frame in 1..path.len() {
        let tempo = states[path[frame]].tempo_bin;
        if tempo != current_tempo_bin {
            let duration_frames = frame - current_start_frame;
            let duration_sec = duration_frames as f32 * frame_duration_sec;
            
            if duration_sec >= segment_min_duration_sec {
                let start_sec = current_start_frame as f32 * frame_duration_sec;
                segments.push((start_sec, start_sec + duration_sec, current_tempo_bin));
            }
            
            current_start_frame = frame;
            current_tempo_bin = tempo;
        }
    }

    let last_start_sec = current_start_frame as f32 * frame_duration_sec;
    let total_duration = path.len() as f32 * frame_duration_sec;
    segments.push((last_start_sec, total_duration, current_tempo_bin));

    segments
}

pub fn collapse_to_time_signature_segments(
    path: &[usize],
    states: &[RhythmState],
    frame_duration_sec: f32,
    segment_min_duration_sec: f32,
) -> Vec<TimeSignatureSegment> {
    if path.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_start_frame = 0;
    let mut current_meter = states[path[0]].meter_class;

    for frame in 1..path.len() {
        let meter = states[path[frame]].meter_class;
        if meter != current_meter {
            let duration_frames = frame - current_start_frame;
            let duration_sec = duration_frames as f32 * frame_duration_sec;
            
            if duration_sec >= segment_min_duration_sec {
                let start_beat = current_start_frame as f32 * frame_duration_sec / 0.5;
                let (num, den) = meter_to_signature(current_meter);
                segments.push(TimeSignatureSegment {
                    start_beat,
                    end_beat: start_beat + duration_sec / 0.5,
                    numerator: num,
                    denominator: den,
                    confidence: 0.8,
                    meter_class: current_meter,
                });
            }
            
            current_start_frame = frame;
            current_meter = meter;
        }
    }

    let last_start_beat = current_start_frame as f32 * frame_duration_sec / 0.5;
    let total_beats = path.len() as f32 * frame_duration_sec / 0.5;
    let (num, den) = meter_to_signature(current_meter);
    segments.push(TimeSignatureSegment {
        start_beat: last_start_beat,
        end_beat: total_beats,
        numerator: num,
        denominator: den,
        confidence: 0.8,
        meter_class: current_meter,
    });

    segments
}

fn meter_to_signature(meter: MeterClass) -> (u8, u8) {
    match meter {
        MeterClass::SimpleDuple => (2, 4),
        MeterClass::SimpleTriple => (3, 4),
        MeterClass::SimpleQuadruple => (4, 4),
        MeterClass::CompoundDuple => (6, 8),
        MeterClass::CompoundQuadruple => (12, 8),
        MeterClass::Other => (4, 4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_valid_state_space() {
        let config = JointRhythmConfig::default();
        let tracker = JointRhythmTracker::new(config);
        
        assert!(tracker.num_states() > 100, "Should have reasonable state count");
        assert!(tracker.state_at(0).is_some());
    }

    #[test]
    fn viterbi_returns_valid_path() {
        let config = JointRhythmConfig::default();
        let tracker = JointRhythmTracker::new(config);
        
        let num_frames = 10;
        let observations: Vec<Vec<f32>> = (0..num_frames)
            .map(|_| (0..tracker.num_states()).map(|_| 0.0).collect())
            .collect();
        
        let path = tracker.viterbi_decode(&observations, 0);
        assert_eq!(path.len(), num_frames);
    }

    #[test]
    fn downbeat_detection() {
        let states = vec![
            RhythmState::new(60, MeterClass::SimpleQuadruple, 0),
            RhythmState::new(60, MeterClass::SimpleQuadruple, 1),
            RhythmState::new(60, MeterClass::SimpleQuadruple, 2),
            RhythmState::new(60, MeterClass::SimpleQuadruple, 3),
            RhythmState::new(60, MeterClass::SimpleQuadruple, 0),
        ];
        
        let path = vec![0, 1, 2, 3, 4];
        let downbeats = extract_downbeats_from_path(&path, &states);
        
        assert_eq!(downbeats.len(), 2);
        assert_eq!(downbeats[0].0, 0);
        assert_eq!(downbeats[1].0, 4);
    }
}