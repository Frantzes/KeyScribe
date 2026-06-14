use crate::app::KeyScribeApp;
use eframe::egui;
use std::path::{Path, PathBuf};
use crate::leadsheet::NoteEvent;

impl KeyScribeApp {
    pub(crate) fn draw_export_modals(&mut self, ctx: &egui::Context) {
        let mut stems_open = self.export_stems_modal_open;
        if stems_open {
            let mut close_modal = false;
            egui::Window::new("Export Stems (Audio)")
                .collapsible(false)
                .resizable(false)
                .open(&mut stems_open)
                .show(ctx, |ui| {
                    self.draw_export_stem_selection(ui);
                    ui.add_space(8.0);
                    if ui.button("Select Destination & Export").clicked() {
                        #[cfg(feature = "desktop-ui")]
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.execute_export_stems(&folder);
                        }
                        close_modal = true;
                    }
                });
            self.export_stems_modal_open = stems_open && !close_modal;
        }

        let mut midi_open = self.export_midi_modal_open;
        if midi_open {
            let mut close_modal = false;
            egui::Window::new("Export Stems (MIDI)")
                .collapsible(false)
                .resizable(false)
                .open(&mut midi_open)
                .show(ctx, |ui| {
                    self.draw_export_stem_selection(ui);
                    ui.add_space(8.0);
                    
                    let mut ui_key_sensitivity = (self.key_color_sensitivity * 0.5).clamp(0.0, 1.0);
                    let slider_response = ui.horizontal(|ui| {
                        let changed = Self::top_bar_slider_with_input(
                            ui,
                            "Note Sensitivity",
                            &mut ui_key_sensitivity,
                            0.0,
                            1.0,
                            "",
                            0.01,
                            2,
                        );
                        if changed {
                            self.key_color_sensitivity = (ui_key_sensitivity * 2.0).clamp(0.0, 2.0);
                        }
                    }).response;
                    slider_response.on_hover_text("Adjust note sensitivity for the MIDI export. Higher sensitivity will detect more notes, while lower sensitivity will filter out softer/background notes.");
                    
                    ui.add_space(8.0);
                    if ui.button("Select Destination & Export").clicked() {
                        #[cfg(feature = "desktop-ui")]
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.execute_export_midi(&folder);
                        }
                        close_modal = true;
                    }
                });
            self.export_midi_modal_open = midi_open && !close_modal;
        }
    }

    fn draw_export_stem_selection(&mut self, ui: &mut egui::Ui) {
        ui.label("Select stems to export:");
        if let Some(stems) = &self.separated_stems {
            for stem in stems {
                let mut selected = self.export_selected_stems.contains(&stem.stem_type);
                if ui.checkbox(&mut selected, stem.stem_type.display_name().as_ref()).changed() {
                    if selected {
                        self.export_selected_stems.insert(stem.stem_type.clone());
                    } else {
                        self.export_selected_stems.remove(&stem.stem_type);
                    }
                }
            }
        }
    }

    fn execute_export_stems(&self, dest_folder: &Path) {
        if let Some(stems) = &self.separated_stems {
            for stem in stems {
                if self.export_selected_stems.contains(&stem.stem_type) {
                    let file_name = format!("{}.wav", stem.stem_type.display_name());
                    let path = dest_folder.join(file_name);
                    let spec = hound::WavSpec {
                        channels: stem.channels,
                        sample_rate: stem.sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    };
                    if let Ok(mut writer) = hound::WavWriter::create(path, spec) {
                        for &s in stem.samples_interleaved.iter() {
                            let _ = writer.write_sample(s);
                        }
                    }
                }
            }
        }
    }

    fn execute_export_midi(&self, dest_folder: &Path) {
        if let Some(stems) = &self.separated_stems {
            for (stem_idx, stem) in stems.iter().enumerate() {
                if self.export_selected_stems.contains(&stem.stem_type) {
                    // Find the stem analysis
                    if let Some(analysis) = self.stem_analyses.iter().find(|a| a.stem_index == stem_idx) {
                        let mut next_id = 0;
                        let notes = Self::extract_events_from_timeline_data(
                            &analysis.timeline,
                            analysis.step_sec,
                            self.sheet_preview_threshold(),
                            &mut next_id,
                        );
                        
                        let file_name = format!("{}.mid", stem.stem_type.display_name());
                        let path = dest_folder.join(file_name);
                        write_midi(&notes, &path);
                    }
                }
            }
        }
    }
}

fn write_midi(notes: &[NoteEvent], path: &Path) {
    use midly::{Header, Format, Timing, Track, TrackEvent, TrackEventKind, MetaMessage, MidiMessage, Smf};
    
    let mut smf = Smf::new(Header::new(Format::SingleTrack, Timing::Metrical(480.into())));
    let mut track = Track::new();
    
    struct Event {
        time_ticks: u32,
        is_note_on: bool,
        pitch: u8,
        velocity: u8,
    }
    
    let mut events = Vec::new();
    for note in notes {
        events.push(Event {
            time_ticks: (note.start_time * 960.0) as u32,
            is_note_on: true,
            pitch: note.pitch,
            velocity: note.velocity,
        });
        events.push(Event {
            time_ticks: (note.end_time * 960.0) as u32,
            is_note_on: false,
            pitch: note.pitch,
            velocity: 0,
        });
    }
    
    events.sort_by_key(|e| e.time_ticks);
    
    let mut last_tick = 0;
    for e in events {
        let delta = e.time_ticks.saturating_sub(last_tick);
        last_tick = e.time_ticks;
        
        let message = if e.is_note_on {
            TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOn { key: e.pitch.into(), vel: e.velocity.into() },
            }
        } else {
            TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOff { key: e.pitch.into(), vel: e.velocity.into() },
            }
        };
        
        let delta_u28 = midly::num::u28::try_from(delta).unwrap_or(midly::num::u28::max_value());
        track.push(TrackEvent {
            delta: delta_u28,
            kind: message,
        });
    }
    
    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    
    smf.tracks.push(track);
    let _ = smf.save(path);
}
