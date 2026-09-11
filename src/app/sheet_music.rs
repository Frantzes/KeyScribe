use std::collections::HashMap;
use std::process::Command;
use std::sync::mpsc::TryRecvError;

use egui_phosphor::regular::{EYE, EYE_SLASH, MUSIC_NOTE, SLIDERS, SPEAKER_HIGH, SPEAKER_LOW, WAVEFORM};

use super::*;
use crate::theme::{MEDIA_PANEL_BG_DARK, MEDIA_PANEL_BG_LIGHT};
use crate::ui::widgets::{
    icon_button, icon_toggle_button, responsive_icon_button_size, synth_knob,
    toggle_switch_with_label,
};

/// Build the stem-strip value row (mute icon + dB number + eye icon) as one
/// text job so all three sections share a single baseline.
///
/// Icon fonts carry different vertical bearings than the digit font, so
/// centering separately-laid-out boxes can never line the ink up (it showed
/// as a ~1px float). One job = one baseline = optically aligned by
/// construction.
fn stem_value_row_job(
    mute_icon: &str,
    mute_color: egui::Color32,
    db_text: &str,
    db_color: egui::Color32,
    eye_icon: &str,
    eye_color: egui::Color32,
    icon_gap: f32,
) -> egui::text::LayoutJob {
    use crate::ui::widgets::icon_font_id;

    let mut job = egui::text::LayoutJob::default();
    job.wrap = egui::text::TextWrapping::no_max_width();
    job.append(
        mute_icon,
        0.0,
        egui::text::TextFormat {
            font_id: icon_font_id(11.0),
            color: mute_color,
            ..Default::default()
        },
    );
    job.append(
        db_text,
        icon_gap,
        egui::text::TextFormat {
            font_id: egui::FontId::monospace(10.0),
            color: db_color,
            ..Default::default()
        },
    );
    job.append(
        eye_icon,
        icon_gap,
        egui::text::TextFormat {
            font_id: icon_font_id(11.0),
            color: eye_color,
            ..Default::default()
        },
    );
    job
}

#[cfg(test)]
mod stem_row_tests {
    use super::*;

    /// The value row must lay out as exactly ONE row: that is what puts the
    /// icons and the dB number on a shared baseline. Multiple rows (wrapping)
    /// or an empty layout would break both alignment and the hitboxes.
    #[test]
    fn stem_value_row_lays_out_on_a_single_shared_baseline() {
        let ctx = egui::Context::default();
        crate::theme::apply_brand_theme(&ctx, true, crate::theme::ACCENT_PURPLE);

        let vols = [-12.5f32, 0.0, 7.2, 10.0];
        let jobs: Vec<_> = vols
            .iter()
            .map(|vol| {
                stem_value_row_job(
                    SPEAKER_HIGH,
                    egui::Color32::WHITE,
                    &format!("{:+.1} dB", vol),
                    egui::Color32::WHITE,
                    EYE,
                    egui::Color32::WHITE,
                    4.0,
                )
            })
            .collect();
        // Fonts are only usable inside a run pass.
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            for (job, vol) in jobs.into_iter().zip(vols.iter()) {
                assert_eq!(job.sections.len(), 3, "mute + dB + eye sections");
                let galley = ctx.fonts(|fonts| fonts.layout_job(job));
                assert_eq!(
                    galley.rows.len(),
                    1,
                    "value row must not wrap (vol={vol})"
                );
                assert!(
                    galley.size().x > 40.0,
                    "row should have real width, got {}",
                    galley.size().x
                );
                // Glyphs must span mute → dB → eye left-to-right.
                let glyphs = &galley.rows[0].glyphs;
                assert!(
                    glyphs.len() >= 3,
                    "expected icon + text + icon glyphs, got {}",
                    glyphs.len()
                );
                let first = glyphs.first().unwrap();
                let last = glyphs.last().unwrap();
                assert!(
                    first.pos.x < last.pos.x,
                    "glyphs should run left-to-right"
                );
            }
        });
    }
}

/// Thin vertical rule used to separate clusters inside a horizontal row.
fn draw_vertical_separator(ui: &mut egui::Ui, row_h: f32) {
    let h = (row_h * 0.55).clamp(16.0, 24.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(9.0, row_h), egui::Sense::hover());
    let center = rect.center();
    let color = ui.visuals().widgets.noninteractive.bg_stroke.color;
    ui.painter().line_segment(
        [
            egui::pos2(center.x, center.y - h * 0.5),
            egui::pos2(center.x, center.y + h * 0.5),
        ],
        egui::Stroke::new(1.0, color),
    );
}
use crate::leadsheet::{
    cross_validate_beat_sources, debug_chord_notes_to_json, detect_chord_changes_per_bar,
    generate_lead_sheet_enhanced, generate_lead_sheet_enhanced_with_timeline,
    generate_lead_sheet_foundation, generate_lead_sheet_with_tempo_map,
    quantize_notes_with_rhythm_map, refine_beat_phase, refine_beat_phase_fixed_bpm, tempo_map_from_beats,
    validate_downbeat_rotation, BeatTrackConfig, CrossValidatedBeats, LeadSheetFoundation,
    LeadSheetPresetConfig, NoteEvent, STEM_GAIN_DB_RANGE,
};
use crate::musicxml::{
    build_musicxml_document, export_engraved_pdf_with_musescore, extract_melody_heuristic,
    extract_melody_skyline, sanitize_filename_component, write_temp_musicxml, MUSICXML_DIVISIONS,
    SHEET_SWING_BIAS,
    SheetEngravingConfig,
};

impl KeyScribeApp {
    fn estimate_sheet_cursor_offset_sec(
        note_events: &[NoteEvent],
        foundation: &LeadSheetFoundation,
    ) -> f32 {
        if note_events.is_empty()
            || foundation.quantized_notes.is_empty()
            || foundation.tempo_map.is_empty()
        {
            return 0.0;
        }

        let mut by_id: HashMap<u32, f32> = HashMap::with_capacity(note_events.len());
        for note in note_events {
            if note.start_time.is_finite() {
                by_id.insert(note.id, note.start_time.max(0.0));
            }
        }

        let mut offsets: Vec<f32> = Vec::new();
        for note in &foundation.quantized_notes {
            let Some(start_time) = by_id.get(&note.id) else {
                continue;
            };
            let expected_time = crate::leadsheet::tempo_map::time_at_beat(
                note.beat_start,
                foundation.tempo_map.as_slice(),
            );
            if expected_time.is_finite() {
                let delta = *start_time - expected_time;
                if delta.abs() <= 2.0 {
                    offsets.push(delta);
                }
            }
        }

        if offsets.len() < 4 {
            return 0.0;
        }

        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = offsets.len() / 2;
        if offsets.len() % 2 == 0 {
            (offsets[mid - 1] + offsets[mid]) * 0.5
        } else {
            offsets[mid]
        }
    }

    fn active_note_id_for_time(note_events: &[NoteEvent], time_sec: f32) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for note in note_events {
            if time_sec >= note.start_time && time_sec < note.end_time {
                let start = note.start_time;
                if best.map_or(true, |(prev, _)| start < prev) {
                    best = Some((start, note.id));
                }
            }
        }
        best.map(|(_, id)| id)
    }

    /// View switcher row: waveform / sheet-music icon toggles pinned to the
    /// right. Rendered in its own row above the speed/pitch controls. The
    /// active view gets the accent fill.
    ///
    /// The row height must be bounded explicitly: `with_layout` alone on the
    /// full-height central panel would vertically center the icons in all
    /// remaining space and swallow the whole panel height as cursor advance.
    pub(super) fn draw_view_switcher_row(&mut self, ui: &mut egui::Ui) {
        // The knob cells (knob + value label) are slightly taller than the
        // bare icon buttons; keep one row height for the whole strip.
        let avail_w = ui.available_width().max(0.0);
        let compact = avail_w < 640.0;
        let row_h = if compact {
            64.0
        } else {
            responsive_icon_button_size(ui).max(46.0)
        };

        // Cluster first (right-aligned); the remaining space goes to the
        // Speed/Pitch controls (evenly distributed knobs when compact).
        ui.allocate_ui_with_layout(
            egui::vec2(avail_w, row_h),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                self.draw_view_cluster(ui);

                let rest = ui.available_width().max(0.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(rest, row_h),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        self.draw_speed_pitch_controls(ui, compact);
                    },
                );
            },
        );

    }

    /// Right-aligned cluster: view toggles + Separate Instruments action.
    fn draw_view_cluster(&mut self, ui: &mut egui::Ui) {
        let row_h = responsive_icon_button_size(ui);
        // right_to_left, so the music note is added first to land on the
        // right.
        let active = self.main_content_tab;
        if icon_toggle_button(
            ui,
            MUSIC_NOTE,
            "Sheet music view (experimental, WIP)",
            active == MainContentTab::SheetMusic,
            true,
            self.highlight_color,
        )
        .clicked()
        {
            self.main_content_tab = MainContentTab::SheetMusic;
        }
        if icon_toggle_button(
            ui,
            WAVEFORM,
            "Waveform view",
            active == MainContentTab::Waveform,
            true,
            self.highlight_color,
        )
        .clicked()
        {
            self.main_content_tab = MainContentTab::Waveform;
        }

        draw_vertical_separator(ui, row_h);

        if self.separated_stems.is_some() {
            // Stems loaded: the button toggles the stem mixer popup.
            let mixer_resp = icon_toggle_button(
                ui,
                SLIDERS,
                "Stem Mixer (listening + piano visibility)",
                self.show_stem_mixer,
                true,
                self.highlight_color,
            );
            self.stem_mixer_anchor = Some(mixer_resp.rect);
            if mixer_resp.clicked() {
                self.show_stem_mixer = !self.show_stem_mixer;
            }
        } else {
            // No stems yet: the button runs separation, shown under the
            // same conditions as the old text button.
            let show_separation = !self.auto_separate
                || (self.separation_attempted && self.separated_stems.is_none());
            if show_separation {
                let can_separate =
                    self.audio_raw.is_some() && !self.is_blocking_processing();
                let sep_resp =
                    icon_button(ui, SLIDERS, "Separate Instruments", can_separate);
                if sep_resp.clicked() {
                    self.run_instrument_separation();
                }
            }
        }
    }

    /// Unified stem mixer popup, anchored under the instant-mix button in
    /// the view switcher row. Replaces the old "Listen:" and "Visualize:"
    /// buttons: each stem row has an audible/mute toggle and a piano
    /// visibility toggle.
    /// Stem volume strip rendered underneath the piano keyboard, toggled by
    /// the SLIDERS button in the top bar: two columns of bipolar dB sliders
    /// (with mute and piano-visibility toggles) when the window is wide
    /// enough, compact synth knobs otherwise.
    pub(super) fn draw_stem_strip(&mut self, ui: &mut egui::Ui) {
        let Some(stems) = self.separated_stems.clone() else {
            return;
        };
        if !self.show_stem_mixer || stems.is_empty() {
            return;
        }
        let is_analyzing = self.stem_analysis_rx.is_some();
        let analysis_ready = !self.stem_analyses.is_empty();
        let frame_fill = if self.dark_mode {
            MEDIA_PANEL_BG_DARK
        } else {
            MEDIA_PANEL_BG_LIGHT
        };

        ui.add_space(UI_VSPACE_TIGHT);

        egui::Frame::none()
            .fill(frame_fill)
            .rounding(egui::Rounding::same(8.0))
            // Even vertical padding top and bottom so the content sits
            // symmetrically inside the rounded panel.
            .inner_margin(egui::Margin::symmetric(10.0, UI_VSPACE_MEDIUM))
            // No outer gutter: the parent panel already insets its content,
            // so the fill spans exactly the same width as the waveform pane.
            .show(ui, |ui| {
                // Content width accounts for the inner margin so the fill
                // lands exactly on the pane edges with rounded corners.
                let content_w = (ui.available_width() - 20.0).max(0.0);
                ui.set_min_width(content_w);

                if !analysis_ready {
                    ui.label(
                        egui::RichText::new(if is_analyzing {
                            "Analyzing stems..."
                        } else {
                            "Stem analysis unavailable"
                        })
                        .weak()
                        .size(11.0),
                    );
                    return;
                }

        // Header: master toggles. "Stem audio" routes playback through the
        // separated stems or the original mix; "Stem transcriptions" routes
        // the piano notes through the stem analyses or the original full-mix
        // transcription. They are independent so audio and visualization can
        // be mixed freely.
        let master_text_color = ui.visuals().text_color();
        let master_dim_color = master_text_color.gamma_multiply(0.45);
        let stem_audio_on = !self.enabled_listening_indices.is_empty();
        let stem_transcriptions_on = !self.enabled_stem_indices.is_empty();

        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;

            let mut audio_on = stem_audio_on;
            let audio_color = if audio_on { master_text_color } else { master_dim_color };
            if toggle_switch_with_label(
                ui,
                "stem_audio_master",
                &mut audio_on,
                true,
                self.highlight_color,
                "Stem audio",
                audio_color,
            ) {
                let pd = ui.input(|i| i.pointer.primary_down());
                self.push_mix_undo(pd);
                self.set_stem_audio_enabled(audio_on, stems.len());
            }

            let mut transcriptions_on = stem_transcriptions_on;
            let transcriptions_color = if transcriptions_on {
                master_text_color
            } else {
                master_dim_color
            };
            if toggle_switch_with_label(
                ui,
                "stem_transcriptions_master",
                &mut transcriptions_on,
                true,
                self.highlight_color,
                "Stem transcriptions",
                transcriptions_color,
            ) {
                let pd = ui.input(|i| i.pointer.primary_down());
                self.push_mix_undo(pd);
                self.set_stem_transcriptions_enabled(transcriptions_on, stems.len());
            }
        });
        // Clearly separate the master toggles from the stem rows below them.
        ui.add_space(UI_VSPACE_MEDIUM);

        let stem_audio_on = !self.enabled_listening_indices.is_empty();
        let stem_transcriptions_on = !self.enabled_stem_indices.is_empty();

        // Stem knobs: wrapping rows that stay balanced (3+3 rather than
        // 4+2) so no row leaves a wide empty stretch; cells distribute
        // equally across each row for visual symmetry.
        const STEM_CELL_W: f32 = 88.0;
        let fit = (((ui.available_width() + 8.0) / (STEM_CELL_W + 8.0)).floor() as usize).max(1);
        let n_stems = stems.len().max(1);
        let rows = n_stems.div_ceil(fit.min(n_stems).max(1));
        let per_row = n_stems.div_ceil(rows).clamp(1, n_stems);
        ui.columns(per_row, |cols| {
            for (ci, col) in cols.iter_mut().enumerate() {
                col.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    for idx in (ci..stems.len()).step_by(per_row) {
                        let stem = &stems[idx];
                        let label = stem.stem_type.display_name();
                        // The knob itself carries the stem color (border and
                        // value arc), so no color dot is needed.
                        let stem_color = self
                            .stem_colors
                            .get(idx)
                            .copied()
                            .unwrap_or(self.highlight_color);
                        let audible =
                            stem_audio_on && self.enabled_listening_indices.contains(&idx);
                        let visible = stem_transcriptions_on
                            && self.enabled_stem_indices.contains(&idx);
                        let mut vol = self
                            .stem_volumes
                            .get(label.as_ref())
                            .copied()
                            .unwrap_or(0.0);
                        let text_color = ui.visuals().text_color();
                        let dim_color = text_color.gamma_multiply(0.4);
                        // The instrument name is only dimmed when the stem is
                        // neither contributing audio nor transcription.
                        let stem_label_active = stem_audio_on || stem_transcriptions_on;

                        // Cell: name above the knob, then a value row with
                        // mute/eye flanking the dB number — all centered.
                        ui.vertical_centered(|ui| {
                            ui.spacing_mut().item_spacing.y = 2.0;
                            // Name alone, centered.
                            ui.label(
                                egui::RichText::new(label.as_ref())
                                    .size(11.0)
                                    .color(if stem_label_active { text_color } else { dim_color }),
                            );
                            let vol_changed = synth_knob(
                                ui,
                                ("stem_strip_knob", idx),
                                &mut vol,
                                -STEM_GAIN_DB_RANGE,
                                STEM_GAIN_DB_RANGE,
                                0.0,
                                30.0,
                                stem_color,
                                true,
                                audible,
                                true,
                                0.25,
                            );
                            // Value row: a single text job so the icons and
                            // the dB number share one baseline — icon fonts
                            // carry different vertical bearings than the
                            // digits, so centering separate boxes can never
                            // line the ink up. Hand-composed inside a single
                            // centered widget so it aligns exactly with the
                            // knob above (nested child Uis paint left-aligned
                            // and would break the centering).
                            let db_color = if audible { text_color } else { dim_color };
                            let icon_w = 18.0_f32;
                            let gap = 4.0_f32;
                            let mute_str =
                                if audible { SPEAKER_HIGH } else { SPEAKER_LOW };
                            let mute_color =
                                if audible { text_color } else { dim_color };
                            let eye_str = if visible { EYE } else { EYE_SLASH };
                            let eye_color =
                                if visible { text_color } else { dim_color };
                            let job = stem_value_row_job(
                                mute_str,
                                mute_color,
                                &format!("{:+.1} dB", vol),
                                db_color,
                                eye_str,
                                eye_color,
                                gap,
                            );
                            let row_galley =
                                ui.ctx().fonts(|fonts| fonts.layout_job(job));
                            let row_size = row_galley.size();
                            let row_w =
                                (row_size.x + gap * 2.0).max(icon_w * 2.0 + 8.0);
                            let (row_rect, _) = ui.allocate_exact_size(
                                egui::vec2(row_w, row_size.y.max(1.0)),
                                egui::Sense::hover(),
                            );
                            // Center the laid-out row inside the (possibly
                            // padded) hitbox row so it stays aligned with the
                            // knob when the dB text is narrow.
                            let row_left =
                                row_rect.center().x - row_size.x * 0.5;
                            ui.painter().galley(
                                egui::pos2(row_left, row_rect.min.y),
                                row_galley,
                                db_color,
                            );
                            let mute_rect = egui::Rect::from_min_size(
                                egui::pos2(row_rect.left(), row_rect.top()),
                                egui::vec2(icon_w, row_rect.height()),
                            );
                            let mute_resp = ui
                                .interact(
                                    mute_rect,
                                    egui::Id::new(("stem_mute", idx)),
                                    if stem_audio_on {
                                        egui::Sense::click()
                                    } else {
                                        egui::Sense::hover()
                                    },
                                )
                                .on_hover_text(if !stem_audio_on {
                                    "Turn on Stem audio to mix individual stems"
                                } else if audible {
                                    "Audible in playback — click to mute"
                                } else {
                                    "Muted — click to unmute"
                                });
                            if stem_audio_on && mute_resp.clicked() {
                                let pd = ui.input(|i| i.pointer.primary_down());
                                self.push_mix_undo(pd);
                                self.toggle_stem_mute(idx, label.as_ref());
                            }
                            let eye_rect = egui::Rect::from_min_size(
                                egui::pos2(
                                    row_rect.right() - icon_w,
                                    row_rect.top(),
                                ),
                                egui::vec2(icon_w, row_rect.height()),
                            );
                            let eye_resp = ui
                                .interact(
                                    eye_rect,
                                    egui::Id::new(("stem_eye", idx)),
                                    if stem_transcriptions_on {
                                        egui::Sense::click()
                                    } else {
                                        egui::Sense::hover()
                                    },
                                )
                                .on_hover_text(if !stem_transcriptions_on {
                                    "Turn on Stem transcriptions to show individual stems"
                                } else if visible {
                                    "Shown on piano — click to hide"
                                } else {
                                    "Hidden from piano — click to show"
                                });
                            if stem_transcriptions_on && eye_resp.clicked() {
                                let pd = ui.input(|i| i.pointer.primary_down());
                                self.push_mix_undo(pd);
                                self.toggle_stem_piano_visibility(idx);
                            }
                            if vol_changed {
                                let pd = ui.input(|i| i.pointer.primary_down());
                                self.push_mix_undo(pd);
                                self.apply_stem_volume_change(idx, label.as_ref(), vol);
                            }
                        });
                    }
                });
            }
        });
            });
        ui.add_space(UI_VSPACE_TIGHT);
    }

    /// Apply a stem volume change from the under-keyboard strip: persist the
    /// dB value, make the stem audible if it was muted, and live-sync gain.
    fn apply_stem_volume_change(&mut self, idx: usize, label: &str, vol: f32) {
        self.stem_volumes.insert(label.to_string(), vol);
        if let Some(hash) = &self.loaded_audio_hash {
            self.file_stem_volumes
                .insert(hash.clone(), self.stem_volumes.clone());
        }
        if !self.enabled_listening_indices.contains(&idx) {
            self.enabled_listening_indices.insert(idx);
        }
        let linear_gain = 10.0f32.powf(vol / 20.0);
        let live_synced = self.sync_stem_gain_live(label, linear_gain);
        self.stem_playback_cache = None;
        if !live_synced {
            self.maybe_restart_playback_for_listen_sync();
        }
    }

    /// Master switch: route playback through the separated stems (on) or the
    /// original mix (off, represented by an empty listening set).
    fn set_stem_audio_enabled(&mut self, enabled: bool, stems_len: usize) {
        self.stem_playback_cache = None;
        if enabled {
            self.enabled_listening_indices = (0..stems_len).collect();
            let live = self.sync_all_stem_gains_live();
            if !live {
                self.maybe_restart_playback_for_listen_sync();
            }
        } else {
            self.enabled_listening_indices.clear();
            self.maybe_restart_playback_for_listen_sync();
        }
    }

    /// Master switch: show the notes from the individual stem analyses (on) or
    /// the original full-mix transcription (off, an empty visible set).
    fn set_stem_transcriptions_enabled(&mut self, enabled: bool, stems_len: usize) {
        if enabled {
            self.enabled_stem_indices = (0..stems_len).collect();
        } else {
            self.enabled_stem_indices.clear();
        }
        self.note_timeline = Arc::new(Vec::new());
        self.note_timeline_step_sec = 0.0;
        self.refresh_note_timeline_from_selected_stems_preserving();
    }

    /// Show/hide a stem's notes on the piano from the under-keyboard strip.
    fn toggle_stem_piano_visibility(&mut self, idx: usize) {
        if self.enabled_stem_indices.contains(&idx) {
            self.enabled_stem_indices.remove(&idx);
        } else {
            self.enabled_stem_indices.insert(idx);
        }
        self.note_timeline = Arc::new(Vec::new());
        self.note_timeline_step_sec = 0.0;
        self.refresh_note_timeline_from_selected_stems_preserving();
    }

    /// Mute/unmute a stem from the under-keyboard strip.
    fn toggle_stem_mute(&mut self, idx: usize, label: &str) {
        let audible = self.enabled_listening_indices.contains(&idx);
        let mut live_synced = false;
        if audible {
            self.enabled_listening_indices.remove(&idx);
            if !self.enabled_listening_indices.is_empty() {
                live_synced = self.sync_stem_gain_live(label, 0.0);
            }
        } else {
            self.enabled_listening_indices.insert(idx);
            let vol = self.stem_volumes.get(label).copied().unwrap_or(0.0);
            let linear_gain = 10.0f32.powf(vol / 20.0);
            live_synced = self.sync_stem_gain_live(label, linear_gain);
        }
        self.stem_playback_cache = None;
        if !live_synced {
            self.maybe_restart_playback_for_listen_sync();
        }
    }

    pub(super) fn draw_sheet_music_view(
        &mut self,
        ui: &mut egui::Ui,
        _interaction_ready: bool,
        _interaction_duration: f32,
        _default_stack_spacing_y: f32,
        _vertical_gap: f32,
        content_height: f32,
    ) {
        // Show config modal if open
        if self.sheet_config_modal_open {
            self.draw_sheet_config_modal(ui);
        }

        let has_engraving = !self.sheet_engraving_pages.is_empty()
            || self.sheet_preview_cache.is_some();

        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;

            if !has_engraving {
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), content_height),
                    egui::Sense::click(),
                );

                // Draw the background box
                ui.painter().rect(
                    rect,
                    egui::Rounding::same(8.0),
                    ui.visuals().extreme_bg_color,
                    egui::Stroke::new(2.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
                );

                if self.sheet_preview_result_rx.is_some() {
                    ui.allocate_ui_at_rect(rect, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(content_height * 0.4);
                            ui.add(egui::Spinner::new().size(32.0));
                            ui.add_space(10.0);
                            ui.label(
                                egui::RichText::new("Analyzing audio and generating sheet music...")
                                    .strong(),
                            );
                        });
                    });
                } else {
                    // ---- large clickable placeholder ----
                    let text = "Click to configure sheet music";
                    let text_font = egui::FontId::proportional(20.0);
                    let text_color = ui.visuals().weak_text_color();
                    let galley = ui.painter().layout_no_wrap(text.to_owned(), text_font, text_color);
                    let galley_pos =
                        rect.center() - egui::vec2(galley.size().x / 2.0, galley.size().y / 2.0);
                    ui.painter().galley(galley_pos, galley, text_color);
                    if resp.clicked() {
                        self.sheet_config_modal_open = true;
                    }
                }
            } else {
                let preview = self.sheet_preview_cache.clone();

                // Row 1: Status + Re-generate + Updating Spinner
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Sheet music generated").strong());
                    if ui.button("Generate again").clicked() {
                        self.sheet_config_modal_open = true;
                    }
                    
                    if self.sheet_preview_result_rx.is_some() {
                        ui.add_space(12.0);
                        ui.add(egui::Spinner::new().size(14.0));
                        ui.label(egui::RichText::new("Updating...").weak());
                    }
                });

                ui.add_space(4.0);

                // Row 2: Export actions
                ui.horizontal(|ui| {
                    let can_export = preview.is_some();
                    if ui.add_enabled(can_export, egui::Button::new("Export MusicXML")).clicked() {
                        self.export_sheet_musicxml(ui.ctx());
                    }
                    if ui.add_enabled(can_export, egui::Button::new("Export Engraved PDF")).clicked() {
                        self.export_sheet_pdf(ui.ctx());
                    }
                    if ui.add_enabled(can_export, egui::Button::new("Open in MuseScore")).clicked() {
                        self.open_in_musescore(ui.ctx());
                    }
                });

                ui.add_space(4.0);

                // Error reporting
                if let Some(err) = self.sheet_preview_error.as_deref() {
                    ui.colored_label(ERROR_RED, err);
                }
                if let Some(err) = self.sheet_engraving_error.as_deref() {
                    ui.colored_label(ERROR_RED, err);
                }

                // Score Area - Fills all remaining space
                if let Some(data) = preview.as_ref() {
                    let clock_pos = self
                        .master_clock
                        .map(|c| c.position_sec)
                        .unwrap_or(self.selected_time_sec);
                    let playback_time = (clock_pos
                        + self.visualization_timing_offset_ms / 1000.0)
                        .max(0.0);
                    let active_note_id =
                        Self::active_note_id_for_time(&data.melody_events, playback_time);
                    let cursor_time = (playback_time - data.cursor_offset_sec).max(0.0);
                    let current_beat = data.foundation.beat_at_time(cursor_time);
                    
                    if self.sheet_engraving_pages.is_empty() && self.sheet_engraving_error.is_none() {
                        ui.centered_and_justified(|ui| {
                            ui.label("Engraving is being prepared...");
                        });
                    } else {
                        draw_scrollable_engraved_preview(
                            ui,
                            self.sheet_engraving_pages.as_slice(),
                            current_beat,
                            active_note_id,
                            self.highlight_color,
                        );
                    }
                }
            }
        });
    }

    fn draw_sheet_config_modal(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::Window::new("Sheet Music Configuration")
            .id(egui::Id::new("sheet_config_modal"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(&ctx, |ui| {
                egui::Grid::new("sheet_config_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(12.0, 6.0))
                    .striped(true)
                    .show(ui, |ui| {
                        // Mode
                        ui.label("Mode").on_hover_text("Lead Sheet: single staff with melody + chords. Piano Grand Staff: both hands. Single Staff: one staff.");
                        ui.horizontal(|ui| {
                            for mode in &[SheetMusicMode::LeadSheet, SheetMusicMode::PianoGrandStaff, SheetMusicMode::SingleStaff] {
                                ui.selectable_value(&mut self.sheet_music_mode, *mode, mode.label());
                            }
                        });
                        ui.end_row();

                        // Melody source
                        ui.label("Melody source").on_hover_text("Which stems to use for melody extraction. Empty = Full Mix (all enabled stems combined).");
                        let mel_label = if self.melody_stem_indices.is_empty() {
                            "Full Mix".to_string()
                        } else if let Some(stems) = self.separated_stems.as_ref() {
                            self.melody_stem_indices.iter()
                                .filter_map(|i| stems.get(*i))
                                .map(|s| s.stem_type.display_name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        } else {
                            format!("{} stem(s)", self.melody_stem_indices.len())
                        };
                        let mel_btn = ui.button(mel_label);
                        if mel_btn.clicked() {
                            self.melody_stem_selector_open = !self.melody_stem_selector_open;
                            self.chord_stem_selector_open = false;
                        }
                        if self.melody_stem_selector_open {
                            let popup_id = ui.make_persistent_id("modal_melody_selector");
                            egui::Area::new(popup_id)
                                .order(egui::Order::Foreground)
                                .fixed_pos(mel_btn.rect.left_bottom() + egui::vec2(0.0, 4.0))
                                .show(ui.ctx(), |ui| {
                                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                                        ui.set_min_width(180.0);
                                        ui.horizontal(|ui| {
                                            if ui.button("Full Mix").clicked() { self.melody_stem_indices.clear(); }
                                            if let Some(stems) = self.separated_stems.as_ref() {
                                                if ui.button("All").clicked() { self.melody_stem_indices = (0..stems.len()).collect(); }
                                            }
                                            if ui.button("None").clicked() { self.melody_stem_indices.clear(); }
                                        });
                                        if let Some(stems) = self.separated_stems.as_ref() {
                                            for (i, stem) in stems.iter().enumerate() {
                                                let mut enabled = self.melody_stem_indices.contains(&i);
                                                if ui.checkbox(&mut enabled, stem.stem_type.display_name()).changed() {
                                                    if enabled { self.melody_stem_indices.insert(i); } else { self.melody_stem_indices.remove(&i); }
                                                }
                                            }
                                        }
                                        if ui.button("Done").clicked() { self.melody_stem_selector_open = false; }
                                    });
                                });
                        }
                        ui.end_row();

                        // Chord source
                        ui.label("Chord source").on_hover_text("Which stems to use for chord detection. Empty = Full Mix. 'Off' disables chord symbols.");
                        let chord_label = if self.chord_skip {
                            "Off".to_string()
                        } else if self.chord_stem_indices.is_empty() {
                            "Full Mix".to_string()
                        } else if let Some(stems) = self.separated_stems.as_ref() {
                            self.chord_stem_indices.iter()
                                .filter_map(|i| stems.get(*i))
                                .map(|s| s.stem_type.display_name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        } else {
                            format!("{} stem(s)", self.chord_stem_indices.len())
                        };
                        let chord_btn = ui.button(chord_label);
                        if chord_btn.clicked() {
                            self.chord_stem_selector_open = !self.chord_stem_selector_open;
                            self.melody_stem_selector_open = false;
                        }
                        if self.chord_stem_selector_open {
                            let popup_id = ui.make_persistent_id("modal_chord_selector");
                            egui::Area::new(popup_id)
                                .order(egui::Order::Foreground)
                                .fixed_pos(chord_btn.rect.left_bottom() + egui::vec2(0.0, 4.0))
                                .show(ui.ctx(), |ui| {
                                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                                        ui.set_min_width(180.0);
                                        ui.horizontal(|ui| {
                                            if ui.button("Off").clicked() { self.chord_skip = true; self.chord_stem_indices.clear(); }
                                            if ui.button("Full Mix").clicked() { self.chord_skip = false; self.chord_stem_indices.clear(); }
                                            if let Some(stems) = self.separated_stems.as_ref() {
                                                if ui.button("All").clicked() { self.chord_skip = false; self.chord_stem_indices = (0..stems.len()).collect(); }
                                            }
                                        });
                                        if let Some(stems) = self.separated_stems.as_ref() {
                                            for (i, stem) in stems.iter().enumerate() {
                                                let mut enabled = self.chord_stem_indices.contains(&i);
                                                if ui.checkbox(&mut enabled, stem.stem_type.display_name()).changed() {
                                                    self.chord_skip = false;
                                                    if enabled { self.chord_stem_indices.insert(i); } else { self.chord_stem_indices.remove(&i); }
                                                }
                                            }
                                        }
                                        if ui.button("Done").clicked() { self.chord_stem_selector_open = false; }
                                    });
                                });
                        }
                        ui.end_row();

                        // BPM
                        ui.label("Tempo (BPM)").on_hover_text("Override detected tempo. Empty = auto-detect from audio.");
                        ui.horizontal(|ui| {
                            let mut bpm_invalid = false;
                            if !self.bpm_input_str.trim().is_empty() && self.manual_bpm.is_none() {
                                bpm_invalid = true;
                            }
                            let bpm_resp = ui.add(
                                egui::TextEdit::singleline(&mut self.bpm_input_str)
                                    .desired_width(60.0)
                                    .hint_text("Auto")
                                    .text_color(if bpm_invalid { egui::Color32::RED } else { egui::Color32::WHITE }),
                            );
                            if bpm_resp.lost_focus() {
                                let trimmed = self.bpm_input_str.trim().to_string();
                                if trimmed.is_empty() {
                                    self.manual_bpm = None;
                                } else if let Ok(bpm) = trimmed.parse::<f32>() {
                                    let clamped = bpm.clamp(30.0, 400.0);
                                    self.manual_bpm = Some(clamped);
                                    self.bpm_input_str = format!("{:.0}", clamped);
                                } else {
                                    self.manual_bpm = None;
                                }
                            }
                            if bpm_invalid {
                                ui.label(egui::RichText::new("invalid").color(egui::Color32::RED).weak());
                            }
                            if self.manual_bpm.is_some() {
                                if ui.button("×2").clicked() {
                                    let base = self.manual_bpm.unwrap_or(120.0);
                                    let clamped = (base * 2.0).clamp(30.0, 400.0);
                                    self.manual_bpm = Some(clamped);
                                    self.bpm_input_str = format!("{:.0}", clamped);
                                }
                                if ui.button("÷2").clicked() {
                                    let base = self.manual_bpm.unwrap_or(120.0);
                                    let clamped = (base / 2.0).clamp(30.0, 400.0);
                                    self.manual_bpm = Some(clamped);
                                    self.bpm_input_str = format!("{:.0}", clamped);
                                }
                                if ui.button("Clear").clicked() {
                                    self.manual_bpm = None;
                                    self.bpm_input_str.clear();
                                }
                            }
                        });
                        ui.end_row();

                        // Feel
                        ui.label("Rhythmic feel").on_hover_text("Override swing detection. 'Auto' detects from audio. 'Straight' forces even 8ths. 'Swing' forces swung 8ths.");
                        ui.horizontal(|ui| {
                            let feels = [
                                (None, "Auto"),
                                (Some(crate::leadsheet::SwingStyle::Straight), "Straight"),
                                (Some(crate::leadsheet::SwingStyle::Swing), "Swing"),
                                (Some(crate::leadsheet::SwingStyle::Triplet), "Triplet"),
                            ];
                            for (val, label) in &feels {
                                let selected = self.manual_swing == *val;
                                if ui.selectable_label(selected, *label).clicked() {
                                    self.manual_swing = *val;
                                }
                            }
                        });
                        ui.end_row();

                        // Polyphony
                        ui.label("Polyphony").on_hover_text("Monophonic: single note line. Polyphonic: preserves chords. Heuristic applies skyline + near-note continuity with outlier suppression.");
                        ui.horizontal(|ui| {
                            let mono = self.melody_mode == MelodyMode::Monophonic;
                            if ui.selectable_label(mono, "Monophonic").clicked() {
                                self.melody_mode = MelodyMode::Monophonic;
                            }
                            if ui.selectable_label(!mono, "Polyphonic").clicked() {
                                self.melody_mode = MelodyMode::Polyphonic;
                            }
                            if mono {
                                let mut h = self.melody_heuristic;
                                if ui.checkbox(&mut h, "Heuristic").changed() {
                                    self.melody_heuristic = h;
                                }
                                if self.melody_heuristic {
                                    ui.add(
                                        egui::Slider::new(&mut self.melody_outlier_semitones, 3u8..=24u8)
                                            .text("σ"),
                                    ).on_hover_text("Outlier threshold: melody jumps larger than this many semitones from the rolling median are suppressed. Lower = smoother line, higher = allows more leaps");
                                }
                            }
                        });
                        ui.end_row();
                    });

                ui.add_space(UI_VSPACE_MEDIUM);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.sheet_config_modal_open = false;
                    }
                    if ui.button("Generate Sheet Music").clicked() {
                        self.sheet_config_modal_open = false;
                        self.sheet_preview_cache_key = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                        self.refresh_sheet_preview_if_needed(ui.ctx());
                    }
                });
            });
    }

    pub(crate) fn sheet_preview_threshold(&self) -> f32 {
        (NOTE_HIGHLIGHT_ACTIVATION_THRESHOLD / self.key_color_sensitivity.max(0.05)).clamp(0.05, 0.95)
    }

    fn current_sheet_preview_key(&self) -> Option<SheetPreviewCacheKey> {
        // Per-stem analyses take priority; fall back to blended note_timeline
        let has_timeline = !self.stem_analyses.is_empty()
            || (!self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0);
        if !has_timeline {
            return None;
        }

        let mut separation_bits = 0u64;
        for &idx in &self.enabled_stem_indices {
            if idx < 64 {
                separation_bits |= 1 << idx;
            }
        }

        // When per-stem analyses are active, encode their combined state into the key
        let stem_key: u64 = if !self.stem_analyses.is_empty() {
            let mut hash: u64 = 0;
            for a in &self.stem_analyses {
                hash = hash.wrapping_mul(31).wrapping_add(a.stem_index as u64);
                hash = hash.wrapping_mul(31).wrapping_add(
                    Arc::as_ptr(&a.timeline) as usize as u64,
                );
                hash = hash.wrapping_mul(31).wrapping_add(a.step_sec.to_bits() as u64);
            }
            hash
        } else {
            0
        };

        let note_min_bits = self.melody_min_note.map(|m| m as u32).unwrap_or(0);
        let note_max_bits = self.melody_max_note.map(|m| m as u32).unwrap_or(0);

        // Encode melody stem indices as bitmask
        let mut melody_stem_bits = 0u64;
        for &idx in &self.melody_stem_indices {
            if idx < 64 {
                melody_stem_bits |= 1 << idx;
            }
        }

        // Encode chord stem indices as bitmask
        let mut chord_stem_bits = 0u64;
        for &idx in &self.chord_stem_indices {
            if idx < 64 {
                chord_stem_bits |= 1 << idx;
            }
        }

        Some(SheetPreviewCacheKey {
            timeline_ptr: Arc::as_ptr(&self.note_timeline) as usize,
            timeline_len: self.note_timeline.len(),
            timeline_step_bits: self.note_timeline_step_sec.to_bits(),
            threshold_bits: self.sheet_preview_threshold().to_bits(),
            separation_selection_bits: separation_bits,
            mode_bits: self.sheet_music_mode as u8,
            melody_stem_bits,
            chord_stem_bits,
            swing_style_bit: self.manual_swing.map(|s| s as u8),
            stem_analysis_key: stem_key,
            melody_note_range_bits: (note_min_bits << 8) | note_max_bits,
            melody_mode_bits: (self.melody_mode as u8 as u32) << 24
                | ((self.melody_heuristic as u8 as u32) << 23)
                | (self.melody_outlier_semitones as u32) << 16,
            use_musescore: self.sheet_use_musescore,
        })
    }

    fn refresh_sheet_preview_if_needed(&mut self, ctx: &egui::Context) {
        let Some(key) = self.current_sheet_preview_key() else {
            self.sheet_preview_cache_key = None;
            self.sheet_preview_cache = None;
            self.sheet_preview_error = Some("No timeline is available yet. Run transcription first.".to_string());
            self.sheet_engraving_cache_key = None;
            self.sheet_engraving_pages.clear();
            self.sheet_engraving_error = None;
            return;
        };

        if self.sheet_preview_cache_key == Some(key) {
            return;
        }

        if self.sheet_preview_result_rx.is_some() {
            return;
        }

        self.start_sheet_preview_build(ctx, key);
    }

    fn start_sheet_preview_build(&mut self, ctx: &egui::Context, key: SheetPreviewCacheKey) {
        let threshold = self.sheet_preview_threshold();

        // Extract melody notes from the selected melody source(s) or combined timeline
        let melody_events = self.extract_notes_for_stems(&self.melody_stem_indices, threshold);
        if melody_events.is_empty() {
            if !self.melody_stem_indices.is_empty() {
                self.sheet_preview_error = Some("Selected melody stem analysis not yet ready. Please wait for per-stem analysis to complete.".to_string());
            } else {
                self.sheet_preview_error = Some("Not enough note events to infer tempo and sheet layout.".to_string());
            }
            self.sheet_preview_cache_key = Some(key);
            self.sheet_preview_cache = None;
            return;
        }
        if melody_events.len() < 4 {
            self.sheet_preview_error = Some("Not enough note events to infer tempo and sheet layout.".to_string());
            self.sheet_preview_cache_key = Some(key);
            self.sheet_preview_cache = None;
            return;
        }

        let stems = self.separated_stems.as_ref();
        let sample_rate = stems
            .and_then(|s| s.first().map(|st| st.sample_rate))
            .or_else(|| self.audio_raw.as_ref().map(|r| r.sample_rate))
            .unwrap_or(44100);

        let bass_audio: Option<Vec<f32>> = stems.and_then(|stems| {
            stems
                .iter()
                .find(|s| s.stem_type == StemType::Bass)
                .map(|s| s.samples_mono.to_vec())
        });
        let drum_audio: Option<Vec<f32>> = stems.and_then(|stems| {
            stems
                .iter()
                .find(|s| s.stem_type == StemType::Drums)
                .map(|s| s.samples_mono.to_vec())
        });
        let full_mix: Option<Vec<f32>> =
            self.audio_raw.as_ref().map(|r| r.samples_mono.to_vec());

        let chord_notes = if !self.chord_skip && !self.chord_stem_indices.is_empty() {
            Some(self.extract_notes_for_stems(&self.chord_stem_indices, threshold))
        } else {
            None
        };

        let chord_timeline = {
            let tls = self.chord_timelines();
            if tls.is_empty() {
                None
            } else {
                Some(crate::leadsheet::TimelineChordInput {
                    timelines: tls,
                    onset_timelines: Vec::new(),
                })
            }
        };

        let job = SheetPreviewJob {
            key,
            threshold,
            melody_events,
            melody_mode: self.melody_mode,
            melody_outlier_semitones: self.melody_outlier_semitones,
            melody_heuristic: self.melody_heuristic,
            bass_audio,
            drum_audio,
            full_mix,
            sample_rate,
            manual_bpm: self.manual_bpm,
            swing_override: self.manual_swing,
            chord_skip: self.chord_skip,
            chord_notes,
            chord_timeline,
            source_duration: self.source_duration(),
        };

        let (tx, rx) = std::sync::mpsc::channel::<(SheetPreviewCacheKey, Result<SheetPreviewData, String>)>();
        self.sheet_preview_result_rx = Some(rx);

        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = Self::run_preview_background(job);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn run_preview_background(job: SheetPreviewJob) -> (SheetPreviewCacheKey, Result<SheetPreviewData, String>) {
        let _threshold = job.threshold;
        let melody_events = job.melody_events;

        // Reduce melody based on selected mode
        let note_events = match job.melody_mode {
            MelodyMode::Polyphonic => melody_events,
            MelodyMode::Monophonic if !job.melody_heuristic => {
                extract_melody_skyline(&melody_events, job.melody_outlier_semitones)
            }
            MelodyMode::Monophonic => {
                extract_melody_heuristic(&melody_events, job.melody_outlier_semitones)
            }
        };

        if note_events.len() < 4 {
            return (job.key, Err("Not enough melody notes after reduction.".to_string()));
        }

        let (beat_track, bpm_source) = if job.sample_rate > 0 {
            let beat_config = BeatTrackConfig::default();
            match cross_validate_beat_sources(
                job.bass_audio.as_deref(),
                job.drum_audio.as_deref(),
                job.full_mix.as_deref(),
                job.sample_rate,
                &beat_config,
            ) {
                Ok(cv) => {
                    let src = if cv.source_count > 1 {
                        format!("BeatThis cross-validated ({} sources)", cv.source_count)
                    } else {
                        "BeatThis ML".to_string()
                    };
                    (Some(cv), src)
                }
                Err(e) => {
                    eprintln!("BeatThis Python execution failed: {:?}", e);
                    let duration = job.source_duration;
                    match crate::leadsheet::detect_beats_from_stems(
                        job.bass_audio.as_deref(),
                        job.drum_audio.as_deref(),
                        job.full_mix.as_deref(),
                        job.sample_rate,
                        duration,
                    ) {
                        Some((bt, src)) => (Some(bt.into()), src),
                        None => (None, "Note onsets (fallback)".to_string()),
                    }
                }
            }
        } else {
            (None, "Unknown".to_string())
        };

        let beat_track = beat_track.or_else(|| {
            crate::leadsheet::detect_beats_from_notes(&note_events).map(CrossValidatedBeats::from)
        });

        // Mirror the CLI pipeline (`generate_sheet_inner` in headless.rs) so the
        // GUI benefits from the same measured improvements: lock a manual BPM to
        // a phase-refined grid (encoder padding otherwise displaces every note
        // by a 16th at bebop tempos), and refine + validate the ML beat grid
        // against the melody notes (16-step phase sweep, elastic per-measure
        // recalibration, downbeat-rotation guard).
        let beat_track = if let Some(manual_bpm) = job.manual_bpm {
            let syn = crate::headless::synthetic_beat_grid(manual_bpm, job.source_duration);
            Some(refine_beat_phase_fixed_bpm(&note_events, &syn))
        } else if let Some(mut refined) = beat_track {
            refined = refine_beat_phase(&note_events, &refined);
            validate_downbeat_rotation(&note_events, &mut refined, 0.25);
            Some(refined)
        } else {
            None
        };

        let mut config = LeadSheetPresetConfig::default();
        // Use the full default quantization grid (1.0, 0.5, 0.25) so 16th
        // notes are preserved, and the full duration grid (4.0, 3.0, 2.0,
        // 1.5, 1.0, 0.75, 0.5, 0.25) so dotted and compound durations are
        // available. The previous override limited everything to quarter
        // and eighth notes, which destroyed rhythmic accuracy.
        config.quantization.min_duration_beats = 0.25;
        config.chord_analysis.skip = job.chord_skip;
        // Match the CLI defaults (`SheetOptions` in headless.rs, as measured
        // by `keyscribe-cli eval-corpus`): collapse jazz extensions to 7ths,
        // one chord per bar (half-bar split off), learned quantizer with the
        // rhythm-coarsening pass, plus the swing feel from the config modal
        // (previously selected but never applied).
        config.chord_analysis.collapse_extensions = true;
        config.chord_analysis.split_threshold = 0.0;
        config.rhythm_coarsen.enabled = true;
        config.swing_override = job.swing_override;
        let mut foundation = None;

        if let Some(bt) = beat_track.as_ref() {
            foundation = match job.chord_timeline.as_ref() {
                Some(tl) => generate_lead_sheet_enhanced_with_timeline(
                    &note_events,
                    bt.beats.as_slice(),
                    bt.downbeats.as_slice(),
                    bt.beats_per_bar,
                    &config,
                    Some(tl),
                ),
                None => generate_lead_sheet_enhanced(
                    &note_events,
                    bt.beats.as_slice(),
                    bt.downbeats.as_slice(),
                    bt.beats_per_bar,
                    &config,
                ),
            };
        }

        let fallback_bt = beat_track.as_ref().map(|bt| bt.clone().into());
        let foundation_res = foundation
            .or_else(|| {
                fallback_bt.as_ref().and_then(|bt: &crate::leadsheet::BeatTrackResult| {
                    tempo_map_from_beats(bt.beats.as_slice()).and_then(|(tempo, tempo_map)| {
                        generate_lead_sheet_with_tempo_map(
                            &note_events,
                            tempo,
                            tempo_map,
                            &config,
                        )
                    })
                })
            })
            .or_else(|| generate_lead_sheet_foundation(&note_events, &config))
            .ok_or_else(|| {
                "Tempo-map detection/quantization failed for the current selection.".to_string()
            });

        let mut foundation = match foundation_res {
            Ok(f) => f,
            Err(e) => return (job.key, Err(e)),
        };

        if foundation.quantized_notes.is_empty() {
            return (job.key, Err("No quantized notes available for engraving.".to_string()));
        }

        if !job.chord_skip {
            if let Some(chord_notes) = job.chord_notes {
                if chord_notes.len() >= 4 {
                    let chord_quantized = quantize_notes_with_rhythm_map(
                        &chord_notes,
                        foundation.tempo_map.as_slice(),
                        foundation.time_signature_segments.as_slice(),
                        &config.quantization,
                    );
                    if !chord_quantized.is_empty() {
                        let mut chord_config = config.chord_analysis;
                        chord_config.skip = false;
                        foundation.chord_changes =
                            detect_chord_changes_per_bar(chord_quantized.as_slice(), foundation.beats_per_bar, chord_config);
                        debug_chord_notes_to_json(chord_quantized.as_slice(), foundation.beats_per_bar, chord_config);
                    }
                }
            }
        }

        let _bpm_source = if job.manual_bpm.is_some() {
            "Manual".to_string()
        } else {
            bpm_source
        };

        let cursor_offset_sec = Self::estimate_sheet_cursor_offset_sec(&note_events, &foundation);
        let melody_events = note_events;

        (job.key, Ok(SheetPreviewData {
            foundation,
            cursor_offset_sec,
            melody_events,
        }))
    }

    pub(super) fn poll_sheet_preview(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.sheet_preview_result_rx else { return };
        match rx.try_recv() {
            Ok((key, result)) => {
                self.sheet_preview_result_rx = None;
                match result {
                    Ok(preview) => {
                        self.sheet_preview_cache_key = Some(key);
                        self.sheet_preview_cache = Some(preview.clone());
                        self.sheet_preview_error = None;
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                        self.sheet_engraving_error = None;
                        
                        // Automatically trigger engraving now that we have the foundation
                        self.refresh_engraved_preview_if_needed(ctx, &preview);
                    }
                    Err(err) => {
                        self.sheet_preview_cache_key = Some(key);
                        self.sheet_preview_cache = None;
                        self.sheet_preview_error = Some(err);
                        self.sheet_engraving_cache_key = None;
                        self.sheet_engraving_pages.clear();
                        self.sheet_engraving_error = None;
                    }
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.sheet_preview_result_rx = None;
            }
        }
    }

    fn refresh_engraved_preview_if_needed(&mut self, ctx: &egui::Context, preview: &SheetPreviewData) {
        let Some(key) = self.current_sheet_preview_key() else {
            self.sheet_engraving_cache_key = None;
            self.sheet_engraving_pages.clear();
            self.sheet_engraving_error = None;
            return;
        };

        if self.sheet_engraving_cache_key == Some(key)
            && (!self.sheet_engraving_pages.is_empty() || self.sheet_engraving_error.is_some())
        {
            return;
        }

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            _allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let musicxml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        // Submit engraving job to background thread
        self.start_sheet_render(ctx, &musicxml, key);
    }

    fn start_sheet_render(
        &mut self,
        ctx: &egui::Context,
        musicxml: &str,
        key: SheetPreviewCacheKey,
    ) {
        let dpi_scale = ctx.pixels_per_point().max(1.0);
        let job = SheetRenderJob {
            musicxml: musicxml.to_string(),
            key,
        };
        let (tx, rx) = std::sync::mpsc::channel::<SheetRenderResult>();
        self.sheet_render_result_rx = Some(rx);

        thread::spawn(move || {
            let result = Self::run_render_background(&job, dpi_scale);
            let _ = tx.send(result);
        });
    }

    pub(super) fn poll_sheet_rendering(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.sheet_render_result_rx else { return };
        match rx.try_recv() {
            Ok(result) => {
                self.sheet_render_result_rx = None;
                if let Some(err) = result.error {
                    self.sheet_engraving_cache_key = Some(result.key);
                    self.sheet_engraving_pages.clear();
                    self.sheet_engraving_error = Some(err);
                } else {
                    // Convert raw RGBA data to egui textures on the main thread
                    let mut pages = Vec::with_capacity(result.pages.len());
                    for (page_idx, raw) in result.pages.iter().enumerate() {
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(
                            [raw.width_px, raw.height_px],
                            &raw.rgba_data,
                        );
                        let texture = ctx.load_texture(
                            format!(
                                "sheet-engraved-{}-{}-{}-{}",
                                result.key.timeline_ptr,
                                result.key.timeline_len,
                                result.key.timeline_step_bits,
                                page_idx
                            ),
                            color_image,
                            egui::TextureOptions::LINEAR,
                        );
                        pages.push(EngravedSheetPage {
                            texture,
                            width_px: raw.width_px,
                            height_px: raw.height_px,
                            note_positions: raw.note_positions.clone(),
                        });
                    }
                    self.sheet_engraving_cache_key = Some(result.key);
                    self.sheet_engraving_pages = pages;
                    self.sheet_engraving_error = None;
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.sheet_render_result_rx = None;
            }
        }
    }

    /// Runs in a background thread: MusicXML → verovioxide → raw RGBA pages + note positions
    fn run_render_background(
        job: &SheetRenderJob,
        dpi_scale: f32,
    ) -> SheetRenderResult {
        use verovioxide::{Options, Png, Svg, Toolkit};

        let key = job.key;
        let render_result = (|| -> Result<Vec<SheetRawPage>, String> {
            let mut toolkit = Toolkit::new().map_err(|err| format!("verovioxide init failed: {err}"))?;
            toolkit
                .load_data(&job.musicxml)
                .map_err(|err| format!("verovioxide could not parse MusicXML: {err}"))?;

            let opts = Options::builder()
                .svg_bounding_boxes(true)
                .build();
            toolkit.set_options(&opts)
                .map_err(|err| format!("verovioxide options failed: {err}"))?;

            let svg_pages: Vec<String> = toolkit
                .render(Svg::all_pages())
                .map_err(|err| format!("verovioxide SVG render failed: {err}"))?;

            let render_w = ((3000.0 * dpi_scale).ceil() as u32).min(4096);
            let png_pages: Vec<Vec<u8>> = toolkit
                .render(Png::all_pages().width(render_w).white_background())
                .map_err(|err| format!("verovioxide PNG render failed: {err}"))?;

            if png_pages.is_empty() {
                return Err("verovioxide returned zero rendered pages.".to_string());
            }

            let mut raw_pages = Vec::with_capacity(png_pages.len());
            for (page_idx, page_bytes) in png_pages.into_iter().enumerate() {
                let rgba = image::load_from_memory(&page_bytes)
                    .map_err(|err| format!("failed decoding rendered PNG page {}: {err}", page_idx + 1))?
                    .to_rgba8();

                let width_px = rgba.width() as usize;
                let height_px = rgba.height() as usize;
                if width_px == 0 || height_px == 0 {
                    continue;
                }

                let note_positions = if page_idx < svg_pages.len() {
                    parse_svg_note_positions(&svg_pages[page_idx])
                } else {
                    Vec::new()
                };

                raw_pages.push(SheetRawPage {
                    width_px,
                    height_px,
                    rgba_data: rgba.into_raw(),
                    note_positions,
                });
            }

            if raw_pages.is_empty() {
                Err("verovioxide returned pages but no valid image could be decoded.".to_string())
            } else {
                Ok(raw_pages)
            }
        })();

        match render_result {
            Ok(pages) => SheetRenderResult { key, pages, error: None },
            Err(err) => SheetRenderResult { key, pages: Vec::new(), error: Some(err) },
        }
    }

    /// Gather the probability timelines to use for harmonic chord detection:
    /// the selected chord stems when chosen, else the combined visualization
    /// timeline, else all enabled stem analyses.
    fn chord_timelines(&self) -> Vec<(Vec<Vec<f32>>, f32)> {
        if !self.chord_stem_indices.is_empty() {
            let mut tls = Vec::new();
            for &idx in &self.chord_stem_indices {
                if let Some(a) = self.stem_analyses.iter().find(|a| a.stem_index == idx) {
                    if !a.timeline.is_empty() && a.step_sec > 0.0 {
                        tls.push((a.timeline.to_vec(), a.step_sec));
                    }
                }
            }
            if !tls.is_empty() {
                return tls;
            }
        }
        if !self.note_timeline.is_empty() && self.note_timeline_step_sec > 0.0 {
            return vec![(self.note_timeline.to_vec(), self.note_timeline_step_sec)];
        }
        let mut tls = Vec::new();
        for a in &self.stem_analyses {
            if !self.enabled_stem_indices.contains(&a.stem_index) {
                continue;
            }
            if a.timeline.is_empty() || a.step_sec <= 0.0 {
                continue;
            }
            tls.push((a.timeline.to_vec(), a.step_sec));
        }
        tls
    }

    /// Extract note events from selected stem timelines, or from the combined
    /// visualization timeline if no stems are selected.
    fn extract_notes_for_stems(
        &self,
        stem_indices: &std::collections::BTreeSet<usize>,
        threshold: f32,
    ) -> Vec<NoteEvent> {
        let mut next_id: u32 = 1;
        if !stem_indices.is_empty() {
            // When specific stems are selected, merge their analyses
            let mut all_events: Vec<NoteEvent> = Vec::new();
            for &idx in stem_indices {
                if let Some(analysis) = self.stem_analyses.iter().find(|a| a.stem_index == idx) {
                    if !analysis.timeline.is_empty() && analysis.step_sec > 0.0 {
                        all_events.extend(Self::extract_events_from_timeline_data(
                            &analysis.timeline,
                            analysis.step_sec,
                            threshold,
                            &mut next_id,
                        ));
                    }
                }
            }
            if !all_events.is_empty() {
                all_events.sort_by(|a, b| {
                    a.start_time
                        .partial_cmp(&b.start_time)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.pitch.cmp(&b.pitch))
                });
                return all_events;
            }
            // Analysis not ready yet — return empty instead of using wrong data
            return Vec::new();
        }
        // "Full Mix" mode: use combined timeline if available, else all enabled stems
        if !self.stem_analyses.is_empty() && self.note_timeline.is_empty() {
            // Combine ALL enabled stem analyses into one polyphonic view
            let mut all_events: Vec<NoteEvent> = Vec::new();
            for analysis in &self.stem_analyses {
                if !self.enabled_stem_indices.contains(&analysis.stem_index) {
                    continue;
                }
                if analysis.timeline.is_empty() || analysis.step_sec <= 0.0 {
                    continue;
                }
                all_events.extend(Self::extract_events_from_timeline_data(
                    &analysis.timeline,
                    analysis.step_sec,
                    threshold,
                    &mut next_id,
                ));
            }
            all_events.sort_by(|a, b| {
                a.start_time
                    .partial_cmp(&b.start_time)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.pitch.cmp(&b.pitch))
            });
            return all_events;
        }
        self.extract_note_events_from_timeline(threshold)
    }

    /// Static helper to extract note events from any timeline data. Notes are
    /// split at re-articulations (staccato repeats) via an adaptive release
    /// threshold, and onsets are attack-adjusted so timing is accurate.
    pub(crate) fn extract_events_from_timeline_data(
        timeline: &[Vec<f32>],
        step_sec: f32,
        threshold: f32,
        next_id: &mut u32,
    ) -> Vec<NoteEvent> {
        // Delegate to the unified CLI extractor (headless.rs) so both paths
        // stay identical. The GUI has no onset-head timeline here, so the
        // onset-head split/refinement simply no-ops (the closure returns the
        // frame-head estimate when the onset timeline is absent).
        crate::headless::extract_notes_from_timeline(timeline, None, step_sec, threshold, next_id)
    }

    fn extract_note_events_from_timeline(&self, threshold: f32) -> Vec<NoteEvent> {
        if self.note_timeline.is_empty() || self.note_timeline_step_sec <= 0.0 {
            return Vec::new();
        }

        // Delegate to the unified static extractor so both code paths
        // (per-stem and full-mix) produce identical results. The previous
        // method version filtered short notes during extraction (before
        // merge), which dropped notes that could have been merged into
        // longer ones — producing different results from the static version.
        let mut next_id: u32 = 1;
        Self::extract_events_from_timeline_data(
            &self.note_timeline,
            self.note_timeline_step_sec,
            threshold,
            &mut next_id,
        )
    }

    fn export_sheet_musicxml(&mut self, ctx: &egui::Context) {
        self.refresh_sheet_preview_if_needed(ctx);
        if self.sheet_preview_cache.is_none() {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        }

        // The save dialog runs on a worker thread (see `spawn_file_dialog`)
        // so browsing for a destination never stalls the event loop; the
        // file is written in `finish_musicxml_export` on choice.
        let file_stem = self.export_file_stem();
        #[cfg(feature = "desktop-ui")]
        self.spawn_file_dialog(
            ctx,
            super::runtime::FileDialogRequest::SaveMusicXml(file_stem),
        );
        #[cfg(not(feature = "desktop-ui"))]
        {
            let _ = ctx;
            self.finish_musicxml_export(
                &app_data_dir().join(format!("{file_stem}.musicxml")),
            );
        }
    }

    pub(super) fn finish_musicxml_export(&mut self, path: &Path) {
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            _allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let xml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        if fs::write(path, xml.as_bytes()).is_ok() {
            self.last_error = None;
        } else {
            self.last_error = Some("Failed to write MusicXML export.".to_string());
        }
    }

    fn export_sheet_pdf(&mut self, ctx: &egui::Context) {
        self.refresh_sheet_preview_if_needed(ctx);
        if self.sheet_preview_cache.is_none() {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        }

        // Async save dialog (see above); written in `finish_pdf_export`.
        let file_stem = self.export_file_stem();
        #[cfg(feature = "desktop-ui")]
        self.spawn_file_dialog(ctx, super::runtime::FileDialogRequest::SavePdf(file_stem));
        #[cfg(not(feature = "desktop-ui"))]
        {
            let _ = ctx;
            self.finish_pdf_export(&app_data_dir().join(format!("{file_stem}.pdf")));
        }
    }

    pub(super) fn finish_pdf_export(&mut self, pdf_path: &Path) {
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to export.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            _allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let xml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        let sibling_xml_path = pdf_path.with_extension("musicxml");
        if fs::write(sibling_xml_path.as_path(), xml.as_bytes()).is_err() {
            self.last_error = Some("Failed to write intermediary MusicXML for PDF engraving.".to_string());
            return;
        }

        match export_engraved_pdf_with_musescore(sibling_xml_path.as_path(), pdf_path) {
            Ok(()) => {
                self.last_error = None;
            }
            Err(err) => {
                self.last_error = Some(format!(
                    "Engraved PDF export failed: {err}. MusicXML was still written to {}",
                    sibling_xml_path.display()
                ));
            }
        }
    }

    fn open_in_musescore(&mut self, ctx: &egui::Context) {
        self.refresh_sheet_preview_if_needed(ctx);
        let Some(preview) = self.sheet_preview_cache.as_ref() else {
            self.last_error = Some("No sheet preview available to open.".to_string());
            return;
        };

        let file_stem = self.export_file_stem();
        let engraving_config = SheetEngravingConfig {
            _allow_triplets: !SHEET_SWING_BIAS,
            is_lead_sheet: self.sheet_music_mode.is_lead_sheet(),
            single_staff: self.sheet_music_mode == SheetMusicMode::SingleStaff,
        };
        let xml = build_musicxml_document(
            file_stem.as_str(),
            &preview.foundation,
            engraving_config,
        );

        let temp_path = match write_temp_musicxml("keyscribe", &xml) {
            Ok(p) => p,
            Err(e) => {
                self.last_error = Some(format!("Failed to write temp MusicXML: {e}"));
                return;
            }
        };

        let mut commands = vec![
            "musescore4".to_string(),
            "MuseScore4".to_string(),
            "mscore".to_string(),
            "MuseScore3".to_string(),
            "MuseScore".to_string(),
        ];

        if cfg!(windows) {
            if let Ok(program_files) = std::env::var("ProgramFiles") {
                commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files));
                commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files));
            }
            if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
                commands.push(format!("{}\\MuseScore 4\\bin\\MuseScore4.exe", program_files_x86));
                commands.push(format!("{}\\MuseScore 3\\bin\\MuseScore3.exe", program_files_x86));
            }
        }

        for cmd in &commands {
            let mut cmd_obj = Command::new(cmd);
            cmd_obj.arg(temp_path.as_os_str());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd_obj.creation_flags(0x08000000);
            }
            match cmd_obj.spawn() {
                Ok(_) => {
                    self.last_error = None;
                    return;
                }
                Err(_) => {}
            }
        }

        self.last_error = Some(
            "MuseScore was not found. Install MuseScore and ensure its CLI executable is on PATH.".to_string(),
        );
    }

    fn export_file_stem(&self) -> String {
        let raw = self
            .loaded_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("keyscribe-sheet");

        sanitize_filename_component(raw)
    }
}

fn parse_svg_note_positions(svg_str: &str) -> Vec<NotePosition> {
    let doc = match roxmltree::Document::parse(svg_str) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let svg_node = match doc.root().descendants().find(|n| n.has_tag_name("svg") && n.attribute("viewBox").is_some()) {
        Some(n) => n,
        None => return Vec::new(),
    };
    let (svg_w, svg_h) = match svg_node.attribute("viewBox") {
        Some(vb) => {
            let parts: Vec<f32> = vb.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 4 { (parts[2], parts[3]) } else { return Vec::new() }
        }
        None => return Vec::new(),
    };
    if svg_w <= 0.0 || svg_h <= 0.0 {
        return Vec::new();
    }

    // Parse page-margin offset from the outermost <g class="page-margin"> transform
    let mut margin_ox = 0.0f32;
    let mut margin_oy = 0.0f32;
    for node in doc.descendants() {
        if node.has_tag_name("g")
            && node.attribute("class").map_or(false, |c| c == "page-margin")
        {
            if let Some(t) = node.attribute("transform") {
                let t = t.trim();
                if let Some(inner) = t.strip_prefix("translate(") {
                    if let Some(paren) = inner.find(')') {
                        let coords: Vec<f32> = inner[..paren]
                            .split(|c| c == ',' || c == ' ')
                            .filter_map(|s| {
                                let s = s.trim();
                                if s.is_empty() { None } else { s.parse().ok() }
                            })
                            .collect();
                        if coords.len() >= 2 {
                            margin_ox = coords[0];
                            margin_oy = coords[1];
                        }
                    }
                }
            }
            break;
        }
    }

    let mut positions = Vec::new();
    for node in doc.descendants() {
        if !node.has_tag_name("g") {
            continue;
        }
        let id = match node.attribute("id") {
            Some(id) => {
                if let Some(pos) = id.find('n') {
                    // Safety check to ensure it's our note ID (starts with n followed by digit)
                    if id[pos + 1..]
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false)
                    {
                        &id[pos..]
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let parts: Vec<&str> = id[1..].split('_').collect();
        let (note_id, _pitch, tick, duration_ticks) = if parts.len() >= 4 {
            let note_id = match parts[0].parse::<u32>().ok() {
                Some(id) => id,
                None => continue,
            };
            let pitch = match parts[1].parse::<u8>().ok() {
                Some(p) if p >= 21 && p <= 108 => p,
                _ => continue,
            };
            let tick = parts[2].parse::<i32>().ok().unwrap_or(0);
            let duration_ticks = parts[3].parse::<i32>().ok().unwrap_or(0);
            (note_id, pitch, tick, duration_ticks)
        } else if parts.len() >= 3 {
            let pitch = match parts[0].parse::<u8>().ok() {
                Some(p) if p >= 21 && p <= 108 => p,
                _ => continue,
            };
            let tick = parts[1].parse::<i32>().ok().unwrap_or(0);
            let duration_ticks = parts[2].parse::<i32>().ok().unwrap_or(0);
            (0, pitch, tick, duration_ticks)
        } else {
            continue;
        };

        // Try data-bounding-box first (most accurate)
        if let Some(bbox) = node.attribute("data-bounding-box") {
            let parts: Vec<f32> = bbox.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if parts.len() >= 4 && parts[2] > 0.0 && parts[3] > 0.0 {
                positions.push(NotePosition {
                    note_id,
                    x: (parts[0] + margin_ox) / svg_w,
                    y: (parts[1] + margin_oy) / svg_h,
                    w: parts[2] / svg_w,
                    h: parts[3] / svg_h,
                    tick,
                    duration_ticks,
                });
                continue;
            }
        }

        // Try bounding-box <rect> child (Verovio's svg-bounding-box option)
        if let Some(rect) = node.children().find(|n| n.has_tag_name("rect")) {
            let rx = rect.attribute("x").and_then(|s| s.parse::<f32>().ok());
            let ry = rect.attribute("y").and_then(|s| s.parse::<f32>().ok());
            let rw = rect.attribute("width").and_then(|s| s.parse::<f32>().ok());
            let rh = rect.attribute("height").and_then(|s| s.parse::<f32>().ok());
            if let (Some(rx), Some(ry), Some(rw), Some(rh)) = (rx, ry, rw, rh) {
                if rw > 0.0 && rh > 0.0 {
                    positions.push(NotePosition {
                        note_id,
                        x: (rx + margin_ox) / svg_w,
                        y: (ry + margin_oy) / svg_h,
                        w: rw / svg_w,
                        h: rh / svg_h,
                        tick,
                        duration_ticks,
                    });
                    continue;
                }
            }
        }

        // Fallback: find the first <use> element (note head) for x,y
        // Skip if this element already has a bbox child (avoids duplicating positions)
        let has_bbox_child = node.children().any(|n| {
            n.has_tag_name("g") && n.attribute("id").map_or(false, |id| id.contains("bbox-"))
        });
        if !has_bbox_child {
        if let Some(use_node) = node.descendants().find(|n| n.has_tag_name("use")) {
            // Verovio uses transform="translate(x,y)" instead of x/y attributes
            let nx = use_node.attribute("x").and_then(|s| s.parse::<f32>().ok());
            let ny = use_node.attribute("y").and_then(|s| s.parse::<f32>().ok());
            let coords = match (nx, ny) {
                (Some(x), Some(y)) => Some((x, y)),
                _ => {
                    use_node.attribute("transform").and_then(|t| {
                        let t = t.trim();
                        if t.starts_with("translate(") {
                            let inner = t.trim_start_matches("translate(");
                            if let Some(paren) = inner.find(')') {
                                let coords: Vec<f32> = inner[..paren]
                                    .split(|c| c == ',' || c == ' ')
                                    .filter_map(|s| {
                                        let s = s.trim();
                                        if s.is_empty() { None } else { s.parse().ok() }
                                    })
                                    .collect();
                                if coords.len() >= 2 {
                                    Some((coords[0], coords[1]))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                }
            };
            if let Some((ncx, ncy)) = coords {
                if ncx > 0.0 && ncy > 0.0 {
                    let note_w = 14.0 / svg_w;
                    let note_h = 14.0 / svg_h;
                    positions.push(NotePosition {
                        note_id,
                        // Apply the same margin offset as the bbox path
                        // (line 1742). The missing margin_ox caused the
                        // playback cursor to be horizontally misaligned
                        // with notes on this code path.
                        x: (ncx + margin_ox - 7.0) / svg_w,
                        y: (ncy + margin_oy - 7.0) / svg_h,
                        w: note_w,
                        h: note_h,
                        tick,
                        duration_ticks,
                    });
                }
            }
        }
        }
    }
    positions
}

fn draw_scrollable_engraved_preview(
    ui: &mut egui::Ui,
    pages: &[EngravedSheetPage],
    current_beat: f32,
    active_note_id: Option<u32>,
    accent: egui::Color32,
) {
    if pages.is_empty() {
        ui.label("No engraved pages available for preview.");
        return;
    }

    let cursor_color = egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 200);

    let current_tick = (current_beat * MUSICXML_DIVISIONS as f32).round() as i32;

    // First, find the global active note or bounding notes for rest interpolation
    let mut global_active_count = 0usize;
    let mut active_pages = Vec::new();
    
    // Also track the best predecessor and successor across all pages if no active note
    let mut best_prev: Option<(usize, &NotePosition)> = None;
    let mut best_next: Option<(usize, &NotePosition)> = None;

    for (p_idx, page) in pages.iter().enumerate() {
        let mut has_active = false;
        for np in &page.note_positions {
            let is_active = if let Some(id) = active_note_id {
                np.note_id == id
            } else {
                current_tick >= np.tick && current_tick < np.tick + np.duration_ticks
            };

            if is_active {
                has_active = true;
                global_active_count += 1;
            } else {
                if np.tick + np.duration_ticks <= current_tick {
                    if best_prev.map_or(true, |(_, p)| np.tick + np.duration_ticks > p.tick + p.duration_ticks) {
                        best_prev = Some((p_idx, np));
                    }
                } else if np.tick > current_tick {
                    if best_next.map_or(true, |(_, n)| np.tick < n.tick) {
                        best_next = Some((p_idx, np));
                    }
                }
            }
        }
        if has_active {
            active_pages.push(p_idx);
        }
    }

    egui::ScrollArea::vertical()
        .id_source("sheet_music_scroll")
        .max_height(ui.available_height().max(120.0))
        .show(ui, |ui| {
            for (idx, page) in pages.iter().enumerate() {
                let page_width = page.width_px.max(1) as f32;
                let page_height = page.height_px.max(1) as f32;
                let target_width = ui.available_width().max(260.0);
                let scale = target_width / page_width;
                let target_height = (page_height * scale).max(120.0);

                let image = egui::Image::new(&page.texture)
                    .fit_to_exact_size(egui::vec2(target_width, target_height));
                let response = ui.add(image);

                let mut cursor_x: Option<f32> = None;
                let mut active_center_y: f32 = 0.0;
                let mut active_count = 0usize;

                if global_active_count > 0 {
                    if active_pages.contains(&idx) {
                        for np in &page.note_positions {
                            let is_active = if let Some(id) = active_note_id {
                                np.note_id == id
                            } else {
                                current_tick >= np.tick && current_tick < np.tick + np.duration_ticks
                            };
                            
                            if is_active {
                                let cx = response.rect.left() + (np.x + np.w * 0.5) * target_width;
                                cursor_x = Some(cursor_x.map_or(cx, |prev| prev.max(cx)));
                                active_center_y += np.y + np.h * 0.5;
                                active_count += 1;
                            }
                        }
                    }
                } else {
                    // No active notes globally, this is a rest.
                    // Interpolate between best_prev and best_next.
                    let draw_on_this_page = match (best_prev, best_next) {
                        (Some((p_idx, _)), Some((n_idx, _))) => idx == p_idx || (idx == n_idx && p_idx != n_idx),
                        (Some((p_idx, _)), None) => idx == p_idx,
                        (None, Some((n_idx, _))) => idx == n_idx,
                        (None, None) => false,
                    };

                    if draw_on_this_page {
                        if let (Some((p_idx, prev)), Some((n_idx, next))) = (best_prev, best_next) {
                            if p_idx == n_idx && idx == p_idx {
                                // Both on this page
                                let y_diff = (prev.y - next.y).abs();
                                if y_diff < 0.025 { // same system
                                    let prev_end = prev.tick + prev.duration_ticks;
                                    let gap = (next.tick - prev_end).max(1);
                                    let t = (current_tick - prev_end).max(0) as f32 / gap as f32;
                                    let t = t.clamp(0.0, 1.0);
                                    
                                    let prev_cx = response.rect.left() + (prev.x + prev.w * 0.5) * target_width;
                                    let next_cx = response.rect.left() + (next.x + next.w * 0.5) * target_width;
                                    
                                    cursor_x = Some(prev_cx + t * (next_cx - prev_cx));
                                    active_center_y += prev.y + prev.h * 0.5;
                                    active_count += 1;
                                } else {
                                    // Different systems on same page
                                    let prev_end = prev.tick + prev.duration_ticks;
                                    let gap = (next.tick - prev_end).max(1);
                                    let t = (current_tick - prev_end).max(0) as f32 / gap as f32;
                                    if t < 0.5 {
                                        cursor_x = Some(response.rect.left() + (prev.x + prev.w * 0.5) * target_width);
                                        active_center_y += prev.y + prev.h * 0.5;
                                        active_count += 1;
                                    } else {
                                        cursor_x = Some(response.rect.left() + (next.x + next.w * 0.5) * target_width);
                                        active_center_y += next.y + next.h * 0.5;
                                        active_count += 1;
                                    }
                                }
                            } else {
                                // On different pages. 
                                let prev_end = prev.tick + prev.duration_ticks;
                                let gap = (next.tick - prev_end).max(1);
                                let t = (current_tick - prev_end).max(0) as f32 / gap as f32;
                                if t < 0.5 && idx == p_idx {
                                    cursor_x = Some(response.rect.left() + (prev.x + prev.w * 0.5) * target_width);
                                    active_center_y += prev.y + prev.h * 0.5;
                                    active_count += 1;
                                } else if t >= 0.5 && idx == n_idx {
                                    cursor_x = Some(response.rect.left() + (next.x + next.w * 0.5) * target_width);
                                    active_center_y += next.y + next.h * 0.5;
                                    active_count += 1;
                                }
                            }
                        } else if let Some((p_idx, prev)) = best_prev {
                            if idx == p_idx {
                                cursor_x = Some(response.rect.left() + (prev.x + prev.w * 0.5) * target_width);
                                active_center_y += prev.y + prev.h * 0.5;
                                active_count += 1;
                            }
                        } else if let Some((n_idx, next)) = best_next {
                            if idx == n_idx {
                                cursor_x = Some(response.rect.left() + (next.x + next.w * 0.5) * target_width);
                                active_center_y += next.y + next.h * 0.5;
                                active_count += 1;
                            }
                        }
                    }
                }

                // Draw vertical cursor line centered on the staff system
                if let Some(x) = cursor_x {
                    // Find the staff system containing the active notes by clustering
                    // all note Y-centers on this page. A gap > 0.025 norm = system boundary.
                    let mut y_centers: Vec<f32> = page.note_positions.iter()
                        .map(|np| np.y + np.h * 0.5).collect();
                    y_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    if active_count == 0 {
                        continue;
                    }

                    let active_y = active_center_y / active_count as f32;
                    let gap_thresh = 0.025;
                    let mut system_top = y_centers.first().copied().unwrap_or(0.0);
                    let mut system_bot = system_top;
                    let mut i = 1;
                    while i < y_centers.len() {
                        if y_centers[i] - y_centers[i - 1] > gap_thresh {
                            // Check if active note falls in the just-finished system
                            if active_y >= system_top && active_y <= system_bot {
                                break;
                            }
                            system_top = y_centers[i];
                            system_bot = y_centers[i];
                        } else {
                            system_bot = y_centers[i];
                        }
                        i += 1;
                    }

                    let pad = (system_bot - system_top) * 0.1;
                    let sy0 = (system_top - pad).max(0.0);
                    let sy1 = (system_bot + pad).min(1.0);

                    let y0 = response.rect.top() + sy0 * target_height;
                    let y1 = response.rect.top() + sy1 * target_height;
                    ui.painter().line_segment(
                        [egui::pos2(x, y0), egui::pos2(x, y1)],
                        egui::Stroke::new(2.0, cursor_color),
                    );
                }

                if idx + 1 < pages.len() {
                    ui.add_space(UI_VSPACE_COMPACT);
                }
            }
        });
}

/// Bipolar dB slider with a fixed-width rail.
///
/// Paints the track itself (like the general volume slider in the media
/// controls) and fills from the center detent to the handle with the
/// user-chosen accent color, so the direction and amount of the offset are
/// visible. Returns true when the value changed. When `enabled` is false the
/// slider is painted dimmed and does not react to clicks or drags.
#[allow(dead_code)]
fn bipolar_db_slider(
    ui: &mut egui::Ui,
    id: egui::Id,
    vol: &mut f32,
    accent: egui::Color32,
    enabled: bool,
) -> bool {
    const RAIL_W: f32 = 140.0;
    const RAIL_H: f32 = 6.0;
    const HANDLE_R: f32 = 7.0;
    const SNAP_DB: f32 = 0.25;
    const DIM: f32 = 0.4;

    let min = -STEM_GAIN_DB_RANGE;
    let max = STEM_GAIN_DB_RANGE;
    *vol = vol.clamp(min, max);

    let row_h = ui.spacing().interact_size.y.max(18.0);
    let (rect, mut resp) = ui.push_id(id, |ui| {
        ui.allocate_exact_size(
            egui::vec2(RAIL_W, row_h),
            if enabled {
                egui::Sense::click_and_drag()
            } else {
                egui::Sense::hover()
            },
        )
    }).inner;
    resp = resp.on_hover_text(if enabled {
        format!("{:+.1} dB — drag to adjust, double-click resets to 0 dB", *vol)
    } else {
        "Inactive — unmute the stem to adjust its volume".to_string()
    });

    // Rail background follows the themed weak fill, like the media-controls
    // volume slider does (hover/active variants included).
    let visuals = ui.visuals();
    let rail_bg = if !enabled {
        visuals.widgets.inactive.weak_bg_fill
    } else if resp.is_pointer_button_down_on() || resp.has_focus() {
        visuals.widgets.active.weak_bg_fill
    } else if resp.hovered() {
        visuals.widgets.hovered.weak_bg_fill
    } else {
        visuals.widgets.inactive.weak_bg_fill
    };
    let rail_bg = if enabled {
        rail_bg
    } else {
        rail_bg.gamma_multiply(DIM)
    };
    let rail = egui::Rect::from_center_size(rect.center(), egui::vec2(RAIL_W, RAIL_H));
    ui.painter()
        .rect_filled(rail, RAIL_H * 0.5, rail_bg);

    let frac = (*vol - min) / (max - min).max(f32::EPSILON);
    let thumb_x = rail.left() + frac.clamp(0.0, 1.0) * rail.width();
    let center_x = rail.center().x;

    // Center detent tick.
    ui.painter().line_segment(
        [
            egui::pos2(center_x, rail.top() - 2.0),
            egui::pos2(center_x, rail.bottom() + 2.0),
        ],
        egui::Stroke::new(1.5, ui.visuals().weak_text_color()),
    );

    // Accent fill from center to handle.
    if vol.abs() > 0.001 {
        let (left, right) = if thumb_x >= center_x {
            (center_x, thumb_x)
        } else {
            (thumb_x, center_x)
        };
        let fill = egui::Rect::from_min_max(
            egui::pos2(left, rail.top()),
            egui::pos2(right, rail.bottom()),
        );
        let fill_color = if enabled {
            accent
        } else {
            accent.gamma_multiply(DIM)
        };
        ui.painter().rect_filled(fill, RAIL_H * 0.5, fill_color);
    }

    // Handle in the standard dark-gray widget fill, like the other sliders.
    let thumb_center = egui::pos2(thumb_x, rect.center().y);
    let (mut handle_fill, mut handle_stroke) = if !enabled {
        (
            visuals.widgets.inactive.bg_fill,
            visuals.widgets.inactive.fg_stroke,
        )
    } else if resp.is_pointer_button_down_on() || resp.has_focus() {
        (
            visuals.widgets.active.bg_fill,
            visuals.widgets.active.fg_stroke,
        )
    } else if resp.hovered() {
        (
            visuals.widgets.hovered.bg_fill,
            visuals.widgets.hovered.fg_stroke,
        )
    } else {
        (
            visuals.widgets.inactive.bg_fill,
            visuals.widgets.inactive.fg_stroke,
        )
    };
    if !enabled {
        handle_fill = handle_fill.gamma_multiply(DIM);
        handle_stroke.color = handle_stroke.color.gamma_multiply(DIM);
    }
    ui.painter()
        .circle_filled(thumb_center, HANDLE_R, handle_fill);
    ui.painter()
        .circle_stroke(thumb_center, HANDLE_R, handle_stroke);

    // Interaction: click/drag maps pointer x to dB, with a center snap.
    let mut changed = false;
    if resp.double_clicked() {
        if *vol != 0.0 {
            *vol = 0.0;
            changed = true;
        }
    } else if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let x = pos.x.clamp(rail.left(), rail.right());
            let f = (x - rail.left()) / rail.width().max(f32::EPSILON);
            let mut next = min + f * (max - min);
            if next.abs() < SNAP_DB {
                next = 0.0;
            }
            let next = (next.clamp(min, max) * 10.0).round() / 10.0;
            if next != *vol {
                *vol = next;
                changed = true;
            }
        }
    }
    if changed {
        resp.mark_changed();
        ui.ctx().request_repaint();
    }
    changed || resp.changed()
}
