use crate::leadsheet::beat_association::{associate_note_events, beats_per_bar_from_downbeats};
use crate::leadsheet::bpm::{BpmDetectionConfig, TempoEstimate};
use crate::leadsheet::chord::{detect_chord_changes, ChordAnalysisConfig};
use crate::leadsheet::instrument_separation::{
    extract_melodic_audio, SeparationConfig, StemType,
};
use crate::leadsheet::joint_tracker::JointRhythmConfig;
use crate::leadsheet::quantize::{
    detect_articulation, detect_grace_notes, detect_swing, quantize_aligned_notes,
    quantize_notes_with_rhythm_map, quantize_notes_with_ties, QuantizationConfig,
    SwingDetectionConfig,
};
use crate::leadsheet::tempo_map::{
    beat_at_time, detect_tempo_map, detect_time_signature_segments, TempoMapConfig,
    TimeSignatureConfig,
};
use crate::leadsheet::types::{
    BeatAlignedNote, ChordSymbolChange, NoteEvent, QuantizedNote, SwingSection, TempoSegment,
    TimeSignatureSegment,
};

#[derive(Debug, Clone)]
pub struct LeadSheetPresetConfig {
    pub bpm: BpmDetectionConfig,
    pub tempo_map: TempoMapConfig,
    pub time_signature: TimeSignatureConfig,
    pub quantization: QuantizationConfig,
    pub chord_analysis: ChordAnalysisConfig,
    pub use_joint_tracker: bool,
    pub joint_rhythm: JointRhythmConfig,
    pub use_instrument_separation: bool,
    pub separation: SeparationConfig,
}

impl Default for LeadSheetPresetConfig {
    fn default() -> Self {
        Self {
            bpm: BpmDetectionConfig::default(),
            tempo_map: TempoMapConfig::default(),
            time_signature: TimeSignatureConfig::default(),
            quantization: QuantizationConfig::default(),
            chord_analysis: ChordAnalysisConfig::default(),
            use_joint_tracker: false,
            joint_rhythm: JointRhythmConfig::default(),
            use_instrument_separation: false,
            separation: SeparationConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LeadSheetFoundation {
    pub tempo_map: Vec<TempoSegment>,
    pub time_signature_segments: Vec<TimeSignatureSegment>,
    pub tempo: TempoEstimate,
    pub quantized_notes: Vec<QuantizedNote>,
    pub melody_notes: Vec<QuantizedNote>,
    pub chord_changes: Vec<ChordSymbolChange>,
    pub tied_notes: Vec<crate::leadsheet::TiedNote>,
    pub rhythm_confidence: f32,
    pub melodic_stem: Option<StemType>,
    pub separation_confidence: f32,
    /// Beat-aligned notes from the enhanced pipeline.
    pub aligned_notes: Vec<BeatAlignedNote>,
    /// Detected swing sections from analysis.
    pub swing_sections: Vec<SwingSection>,
    /// Beats per bar derived from downbeats.
    pub beats_per_bar: u32,
}

impl LeadSheetFoundation {
    pub fn beat_at_time(&self, time_sec: f32) -> f32 {
        beat_at_time(time_sec, self.tempo_map.as_slice())
    }
}

/// Stage-1 + Stage-2 baseline for lead-sheet generation.
pub fn generate_lead_sheet_foundation(
    notes: &[NoteEvent],
    config: &LeadSheetPresetConfig,
) -> Option<LeadSheetFoundation> {
    generate_lead_sheet_with_separation(notes, None, config)
}

pub fn generate_lead_sheet_with_separation(
    notes: &[NoteEvent],
    separated_stems: Option<&[crate::leadsheet::SeparatedStem]>,
    config: &LeadSheetPresetConfig,
) -> Option<LeadSheetFoundation> {
    let (tempo, tempo_map) = detect_tempo_map(notes, config.bpm, config.tempo_map)?;
    generate_lead_sheet_with_separation_and_tempo_map(
        notes,
        separated_stems,
        tempo,
        tempo_map,
        config,
    )
}

pub fn generate_lead_sheet_with_tempo_map(
    notes: &[NoteEvent],
    tempo: TempoEstimate,
    tempo_map: Vec<TempoSegment>,
    config: &LeadSheetPresetConfig,
) -> Option<LeadSheetFoundation> {
    generate_lead_sheet_with_separation_and_tempo_map(notes, None, tempo, tempo_map, config)
}

fn generate_lead_sheet_with_separation_and_tempo_map(
    notes: &[NoteEvent],
    separated_stems: Option<&[crate::leadsheet::SeparatedStem]>,
    tempo: TempoEstimate,
    tempo_map: Vec<TempoSegment>,
    config: &LeadSheetPresetConfig,
) -> Option<LeadSheetFoundation> {
    if tempo_map.is_empty() {
        return None;
    }

    let time_signature_segments =
        detect_time_signature_segments(notes, tempo_map.as_slice(), config.time_signature);
    let quantized_notes = quantize_notes_with_rhythm_map(
        notes,
        tempo_map.as_slice(),
        time_signature_segments.as_slice(),
        &config.quantization,
    );

    if quantized_notes.is_empty() {
        return None;
    }

    let tied_notes = quantize_notes_with_ties(
        notes,
        tempo_map.as_slice(),
        time_signature_segments.as_slice(),
        &config.quantization,
    );

    let (melody_notes, melodic_stem, separation_confidence) = if config.use_instrument_separation
        && separated_stems.is_some()
    {
        extract_melody_from_separated_stems(
            separated_stems.unwrap(),
            &tempo_map,
            &time_signature_segments,
            &config.quantization,
        )
    } else {
        let melody_events = extract_melody_events(notes);
        let notes = quantize_notes_with_rhythm_map(
            melody_events.as_slice(),
            tempo_map.as_slice(),
            time_signature_segments.as_slice(),
            &config.quantization,
        );
        (notes, None, 0.0)
    };

    let chord_changes = detect_chord_changes(quantized_notes.as_slice(), config.chord_analysis);
    let rhythm_confidence = compute_rhythm_confidence(&time_signature_segments, &tempo_map);

    Some(LeadSheetFoundation {
        tempo_map,
        time_signature_segments,
        tempo,
        quantized_notes,
        melody_notes,
        chord_changes,
        tied_notes,
        rhythm_confidence,
        melodic_stem,
        separation_confidence,
        aligned_notes: Vec::new(),
        swing_sections: Vec::new(),
        beats_per_bar: 4,
    })
}

/// Enhanced pipeline that uses beat-aligned notes and swing-aware quantization.
pub fn generate_lead_sheet_enhanced(
    notes: &[NoteEvent],
    beat_times: &[f32],
    downbeat_times: &[f32],
    config: &LeadSheetPresetConfig,
) -> Option<LeadSheetFoundation> {
    if beat_times.len() < 2 || notes.is_empty() {
        return None;
    }

    let aligned = associate_note_events(notes, beat_times, downbeat_times);
    if aligned.is_empty() {
        return None;
    }

    let beats_per_bar = beats_per_bar_from_downbeats(downbeat_times, beat_times);
    let swing_config = SwingDetectionConfig::default();
    let swing_sections = detect_swing(&aligned, beats_per_bar, &swing_config);

    let mut quantized = quantize_aligned_notes(&aligned, &swing_sections, beats_per_bar);
    if quantized.is_empty() {
        return None;
    }

    quantized = detect_grace_notes(&quantized, &aligned);
    quantized = detect_articulation(&quantized, &aligned);

    let beat_duration_sec = beat_times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.001)
        .fold(0.0f32, |acc, d| acc + d)
        / (beat_times.len().saturating_sub(1) as f32).max(1.0);

    let global_bpm = (60.0 / beat_duration_sec.max(0.001)).clamp(40.0, 260.0);
    let tempo = crate::leadsheet::TempoEstimate {
        bpm: global_bpm,
        beat_duration_sec,
        confidence: 0.8,
    };

    let tempo_map: Vec<TempoSegment> = beat_times
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let interval = (w[1] - w[0]).max(0.001);
            TempoSegment {
                start_time_sec: w[0],
                end_time_sec: w[1],
                bpm: (60.0 / interval).clamp(40.0, 260.0),
                beat_duration_sec: interval,
                beat_offset: i as f32,
            }
        })
        .collect();

    let time_sig_numerator = beats_per_bar as u8;
    let time_sig_denominator = 4u8;
    let time_signature_segments = vec![crate::leadsheet::TimeSignatureSegment {
        start_beat: 0.0,
        end_beat: (aligned.len() as f32).max(4.0),
        numerator: time_sig_numerator,
        denominator: time_sig_denominator,
        confidence: 0.8,
        meter_class: crate::leadsheet::MeterClass::from_signature(
            time_sig_numerator,
            time_sig_denominator,
        ),
    }];

    let chord_changes =
        detect_chord_changes(quantized.as_slice(), config.chord_analysis);
    let rhythm_confidence = if swing_sections.is_empty() {
        0.7
    } else {
        swing_sections
            .iter()
            .map(|s| s.confidence)
            .sum::<f32>()
            / swing_sections.len() as f32
    };

    Some(LeadSheetFoundation {
        tempo_map,
        time_signature_segments,
        tempo,
        quantized_notes: quantized.clone(),
        melody_notes: quantized,
        chord_changes,
        tied_notes: Vec::new(),
        rhythm_confidence,
        melodic_stem: None,
        separation_confidence: 0.0,
        aligned_notes: aligned,
        swing_sections,
        beats_per_bar,
    })
}

fn extract_melody_from_separated_stems(
    stems: &[crate::leadsheet::SeparatedStem],
    tempo_map: &[TempoSegment],
    time_signature_segments: &[TimeSignatureSegment],
    quantization: &QuantizationConfig,
) -> (Vec<QuantizedNote>, Option<StemType>, f32) {
    let melodic_stem = identify_melodic_stem_from_stems(stems);
    let separation_confidence = stems
        .iter()
        .find(|s| Some(s.stem_type.clone()) == melodic_stem)
        .map(|s| s.confidence)
        .unwrap_or(0.5);

    if let Some(stem_type) = melodic_stem {
        if let Some(melodic_audio) = extract_melodic_audio(stems, &stem_type) {
            let melody_events = audio_to_note_events(&melodic_audio, tempo_map);
            let melody_notes = quantize_notes_with_rhythm_map(
                melody_events.as_slice(),
                tempo_map,
                time_signature_segments,
                quantization,
            );
            return (melody_notes, Some(stem_type), separation_confidence);
        }
    }

    let all_notes: Vec<NoteEvent> = stems
        .iter()
        .flat_map(|s| audio_to_note_events(&s.samples_mono, tempo_map))
        .collect();
    let melody_notes = quantize_notes_with_rhythm_map(
        all_notes.as_slice(),
        tempo_map,
        time_signature_segments,
        quantization,
    );
    (melody_notes, None, separation_confidence)
}

fn identify_melodic_stem_from_stems(stems: &[crate::leadsheet::SeparatedStem]) -> Option<StemType> {
    let melodic_candidates: Vec<&crate::leadsheet::SeparatedStem> = stems
        .iter()
        .filter(|stem| stem.stem_type.is_melodic())
        .collect();
    let candidates = if melodic_candidates.is_empty() {
        stems.iter().collect::<Vec<_>>()
    } else {
        melodic_candidates
    };

    let mut best_stem = None;
    let mut best_score = -1.0f32;

    for stem in candidates {
        let pitch_variance = compute_pitch_variance(&stem.samples_mono);
        let spectral_energy = compute_spectral_energy(&stem.samples_mono);
        let score = (pitch_variance * 0.5 + spectral_energy * 0.5) * stem.confidence.max(0.1);

        if score > best_score {
            best_score = score;
            best_stem = Some(stem.stem_type.clone());
        }
    }

    best_stem.or_else(|| stems.first().map(|stem| stem.stem_type.clone()))
}

fn compute_pitch_variance(audio: &[f32]) -> f32 {
    if audio.len() < 1024 {
        return 0.5;
    }

    let window_size = 1024;
    let hop = 512;
    let mut pitch_changes = Vec::new();
    let mut prev_zero_crossings = 0isize;

    for i in (0..audio.len() - window_size).step_by(hop) {
        let mut zero_crossings = 0isize;
        for j in i..i + window_size - 1 {
            if (audio[j] >= 0.0) != (audio[j + 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        if i > 0 {
            let change = (zero_crossings - prev_zero_crossings).unsigned_abs() as f32;
            pitch_changes.push(change);
        }
        prev_zero_crossings = zero_crossings;
    }

    if pitch_changes.is_empty() {
        return 0.5;
    }

    let mean: f32 = pitch_changes.iter().sum::<f32>() / pitch_changes.len() as f32;
    let variance: f32 = pitch_changes.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
        / pitch_changes.len() as f32;

    (variance / 100.0).clamp(0.0, 1.0)
}

fn compute_spectral_energy(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = audio.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / audio.len() as f32).sqrt();

    (rms * 10.0).clamp(0.0, 1.0)
}

fn audio_to_note_events(audio: &[f32], tempo_map: &[TempoSegment]) -> Vec<NoteEvent> {
    if audio.is_empty() || tempo_map.is_empty() {
        return Vec::new();
    }

    let sample_rate = 22050.0f32;
    let beat_duration = tempo_map[0].beat_duration_sec;
    let window_sec = beat_duration * 0.5;
    let hop_sec = beat_duration * 0.25;
    let window_samples = (window_sec * sample_rate) as usize;
    let hop_samples = (hop_sec * sample_rate) as usize;

    let threshold = 0.05f32;
    let mut events = Vec::new();
    let mut in_note = false;
    let mut note_start = 0usize;
    let mut max_rms = 0.0f32;

    for (_frame_idx, start_sample) in (0..audio.len()).step_by(hop_samples).enumerate() {
        let end_sample = (start_sample + window_samples).min(audio.len());
        let window = &audio[start_sample..end_sample];

        let rms = if window.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = window.iter().map(|&x| x * x).sum();
            (sum_sq / window.len() as f32).sqrt()
        };

        if rms > threshold {
            if !in_note {
                in_note = true;
                note_start = start_sample;
                max_rms = rms;
            } else if rms > max_rms {
                max_rms = rms;
            }
        } else if in_note {
            in_note = false;
            let start_time = note_start as f32 / sample_rate;
            let end_time = start_sample as f32 / sample_rate;

            if end_time > start_time + 0.05 {
                events.push(NoteEvent {
                    pitch: 60,
                    start_time,
                    end_time,
                    velocity: ((max_rms * 127.0).round() as u8).clamp(1, 127),
                    channel: None,
                });
            }
        }
    }

    if in_note {
        let start_time = note_start as f32 / sample_rate;
        let end_time = audio.len() as f32 / sample_rate;
        if end_time > start_time + 0.05 {
            events.push(NoteEvent {
                pitch: 60,
                start_time,
                end_time,
                velocity: ((max_rms * 127.0).round() as u8).clamp(1, 127),
                channel: None,
            });
        }
    }

    events
}

fn compute_rhythm_confidence(time_sigs: &[TimeSignatureSegment], tempo_map: &[TempoSegment]) -> f32 {
    let ts_confidence = if time_sigs.is_empty() {
        0.5
    } else {
        time_sigs.iter().map(|s| s.confidence).fold(0.0f32, |acc, c| acc + c) / time_sigs.len() as f32
    };

    let tempo_confidence = compute_tempo_confidence(tempo_map, 0.8);

    (ts_confidence + tempo_confidence) / 2.0
}

fn compute_tempo_confidence(tempo_map: &[TempoSegment], base_confidence: f32) -> f32 {
    if tempo_map.is_empty() {
        return base_confidence;
    }

    let stability = if tempo_map.len() == 1 {
        1.0
    } else {
        let mut variance = 0.0f32;
        for seg in tempo_map {
            let ratio = seg.bpm / tempo_map[0].bpm.max(1.0);
            variance += (ratio - 1.0).abs();
        }
        (1.0 - (variance / tempo_map.len() as f32)).clamp(0.0, 1.0)
    };

    (base_confidence + stability) / 2.0
}

fn extract_melody_events(notes: &[NoteEvent]) -> Vec<NoteEvent> {
    if notes.is_empty() {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct EventPoint {
        time: f32,
        pitch: u8,
        velocity: u8,
        is_start: bool,
    }

    let mut events = Vec::<EventPoint>::new();
    for note in notes {
        if !note.start_time.is_finite() || !note.end_time.is_finite() {
            continue;
        }
        if note.end_time <= note.start_time {
            continue;
        }

        events.push(EventPoint {
            time: note.start_time.max(0.0),
            pitch: note.pitch,
            velocity: note.velocity,
            is_start: true,
        });
        events.push(EventPoint {
            time: note.end_time.max(0.0),
            pitch: note.pitch,
            velocity: note.velocity,
            is_start: false,
        });
    }

    events.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.is_start.cmp(&a.is_start))
            .then_with(|| b.pitch.cmp(&a.pitch))
    });

    let mut active: Vec<NoteEvent> = Vec::new();
    let mut out: Vec<NoteEvent> = Vec::new();

    let mut i = 0usize;
    let mut prev_t = events[0].time;
    let mut current_melody_pitch: Option<u8> = None;
    let mut current_melody_velocity: u8 = 90;
    let mut segment_start = prev_t;

    while i < events.len() {
        let t = events[i].time;

        if t > prev_t {
            if let Some(pitch) = current_melody_pitch {
                if t > segment_start {
                    out.push(NoteEvent {
                        pitch,
                        start_time: segment_start,
                        end_time: t,
                        velocity: current_melody_velocity,
                        channel: None,
                    });
                }
            }
            segment_start = t;
        }

        while i < events.len() && (events[i].time - t).abs() < 1.0e-6 {
            let ev = events[i];
            if ev.is_start {
                active.push(NoteEvent {
                    pitch: ev.pitch,
                    start_time: t,
                    end_time: t,
                    velocity: ev.velocity,
                    channel: None,
                });
            } else if let Some(pos) = active.iter().position(|n| n.pitch == ev.pitch) {
                active.swap_remove(pos);
            }
            i += 1;
        }

        let next_choice = choose_melody_note(active.as_slice(), current_melody_pitch);
        current_melody_pitch = next_choice.map(|(pitch, _)| pitch);
        current_melody_velocity = next_choice.map(|(_, vel)| vel).unwrap_or(90);
        prev_t = t;
    }

    merge_adjacent_note_events(out.as_slice())
}

fn choose_melody_note(active: &[NoteEvent], previous_pitch: Option<u8>) -> Option<(u8, u8)> {
    if active.is_empty() {
        return None;
    }

    // Prefer upper voice but keep continuity to avoid jumping every frame.
    let mut best: Option<(u8, u8, f32)> = None;
    for note in active {
        let continuity_bonus = previous_pitch
            .map(|prev| {
                let dist = (note.pitch as i32 - prev as i32).abs() as f32;
                (1.0 - (dist / 24.0)).clamp(0.0, 1.0) * 8.0
            })
            .unwrap_or(0.0);
        let score = note.pitch as f32 * 0.7 + note.velocity as f32 * 0.15 + continuity_bonus;

        match best {
            Some((_, _, best_score)) if score <= best_score => {}
            _ => best = Some((note.pitch, note.velocity, score)),
        }
    }

    best.map(|(pitch, velocity, _)| (pitch, velocity))
}

fn merge_adjacent_note_events(events: &[NoteEvent]) -> Vec<NoteEvent> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut sorted = events.to_vec();
    sorted.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pitch.cmp(&b.pitch))
    });

    let mut out: Vec<NoteEvent> = Vec::new();
    for note in sorted {
        if let Some(last) = out.last_mut() {
            if last.pitch == note.pitch && (note.start_time - last.end_time).abs() < 0.02 {
                last.end_time = note.end_time.max(last.end_time);
                last.velocity = last.velocity.max(note.velocity);
                continue;
            }
        }
        out.push(note);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_runs_end_to_end() {
        let mut notes = Vec::new();
        for i in 0..16 {
            let t = i as f32 * 0.5;
            notes.push(NoteEvent {
                pitch: 60,
                start_time: t,
                end_time: t + 0.2,
                velocity: 100,
                channel: None,
            });
        }

        let out = generate_lead_sheet_foundation(&notes, &LeadSheetPresetConfig::default())
            .expect("expected tempo + quantized notes");

        assert!((out.tempo.bpm - 120.0).abs() < 1.5);
        assert_eq!(out.quantized_notes.len(), notes.len());
        assert!(!out.tempo_map.is_empty());
        assert!(!out.melody_notes.is_empty());
        assert!(!out.tied_notes.is_empty());
        assert!(out.rhythm_confidence > 0.0);
    }

    #[test]
    fn melody_quantization_keeps_shorter_values() {
        let notes = vec![
            NoteEvent {
                pitch: 72,
                start_time: 0.0,
                end_time: 0.24,
                velocity: 110,
                channel: None,
            },
            NoteEvent {
                pitch: 74,
                start_time: 0.26,
                end_time: 0.48,
                velocity: 108,
                channel: None,
            },
            NoteEvent {
                pitch: 76,
                start_time: 0.50,
                end_time: 0.98,
                velocity: 112,
                channel: None,
            },
            NoteEvent {
                pitch: 77,
                start_time: 1.00,
                end_time: 1.24,
                velocity: 105,
                channel: None,
            },
            NoteEvent {
                pitch: 79,
                start_time: 1.26,
                end_time: 1.98,
                velocity: 118,
                channel: None,
            },
        ];

        let out = generate_lead_sheet_foundation(&notes, &LeadSheetPresetConfig::default())
            .expect("expected quantized melody");
        let has_non_quarter = out
            .melody_notes
            .iter()
            .any(|n| (n.beat_duration - 1.0).abs() > 0.05);
        assert!(has_non_quarter);
    }
}
