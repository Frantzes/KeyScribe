"""Visualize beat/downbeat detection from beat-this on an audio file.

Usage:
    python visualize_beats.py path/to/audio.wav [--model final0] [--dbn]

    Can also combine drum+bass stems (like the Rust app does):
    python visualize_beats.py path/to/original.mp3 --drum drum_stem.wav --bass bass_stem.wav

    Controls when plot window opens:
      - Scroll wheel / arrow keys: zoom/pan
      - 'b' key: toggle between full view and beat-zoom
      - 's' key: save current view as PNG
      - 'r' key: reset view
      - 'p' key: toggle audio playback (requires sounddevice)
"""

import argparse
from pathlib import Path

import numpy as np

# madmom compatibility: np.float was removed in NumPy 1.24+
if not hasattr(np, "float"):
    np.float = float

import soundfile as sf
import torch
from beat_this.inference import File2Beats
from matplotlib import pyplot as plt


def resolve_device(device: str) -> str:
    return "cuda" if device == "auto" and torch.cuda.is_available() else device


def make_figure(
    audio: np.ndarray,
    sample_rate: int,
    beats: list[float],
    downbeats: list[float],
    title: str = "Beat & Downbeat Analysis",
):
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0.0, duration, len(audio))

    # compute stats
    dbi = np.median(np.diff(downbeats)) if len(downbeats) >= 2 else 0.5
    bi = np.median(np.diff(beats)) if len(beats) >= 2 else 0.5
    bpm = 60.0 / bi
    bpb = round(dbi / bi) if dbi > 0 and bi > 0 else 4

    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [1, 1]}, sharex=False
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── full waveform ──────────────────────────────────────────────
    _draw_waveform(ax_full, time_axis, audio)
    ax_full.set_xlim(0, duration)
    ax_full.set_title(
        f"Full View  |  BPM: {bpm:.1f}  |  "
        f"{len(downbeats)} downbeats / {len(beats)} beats  |  "
        f"~{bpb} beats/bar"
    )
    _add_beats(ax_full, beats, downbeats, beats_per_bar=None)

    # ── zoom panel (first ~8 bars or up to 16s) ───────────────────
    window = min(16.0, float(dbi * 8))
    zoom_end = min(duration, window)
    _draw_waveform(ax_zoom, time_axis, audio)
    ax_zoom.set_xlabel("Time (seconds)")
    ax_zoom.set_xlim(0, zoom_end)
    ax_zoom.set_title(f"Zoom (first {zoom_end:.1f}s)")
    _add_beats(ax_zoom, beats, downbeats, beats_per_bar=bpb)

    ax_full.axvspan(0, zoom_end, color="royalblue", alpha=0.06, zorder=0)

    fig.tight_layout()
    return fig, (ax_full, ax_zoom), (bpm, bpb)


def _draw_waveform(ax, time_axis, audio):
    ax.plot(time_axis, audio, color="0.65", linewidth=0.4, alpha=0.7)
    ax.set_ylabel("Amplitude")


def _add_beats(ax, beats, downbeats, beats_per_bar: int | None):
    """Draw vertical lines for beats, prominent red lines for downbeats."""
    down_set = set(downbeats)
    offset = 0  # beat number within bar

    for t in beats:
        # find closest downbeat (within 1ms)
        is_down = any(abs(t - d) < 1e-3 for d in downbeats)
        if is_down:
            offset = 0
            bar_num = (downbeats.index(t) + 1) if t in downbeats else 0
            ax.axvline(t, color="red", linewidth=1.8, alpha=0.85, zorder=5)
            ax.text(
                t, 0.95, f"{bar_num}",
                transform=ax.get_xaxis_transform(),
                color="red", fontsize=10, fontweight="bold",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="red", alpha=0.85),
            )
        else:
            offset += 1
            ax.axvline(t, color="gray", linewidth=0.6, linestyle="--", alpha=0.4, zorder=3)
            if beats_per_bar and offset < beats_per_bar:
                ax.text(
                    t, -0.95, str(offset + 1),
                    transform=ax.get_xaxis_transform(),
                    color="gray", fontsize=7, ha="center", va="bottom", alpha=0.6,
                )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Visualize beat-this beat/downbeat detection."
    )
    ap.add_argument("input", help="Path to audio file")
    ap.add_argument("--drum", help="Drum stem audio (optional, combined with bass)")
    ap.add_argument("--bass", help="Bass stem audio (optional, combined with drums)")
    ap.add_argument("--model", default="final0")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--dbn", action="store_true", help="Enable DBN postprocessing")
    ap.add_argument("--no-play", action="store_true", help="Disable audio playback")
    args = ap.parse_args()

    device = resolve_device(args.device)

    # ── load / combine audio ───────────────────────────────────────
    if args.drum and args.bass:
        print("Combining drum + bass stems for beat detection...")
        drum_data, sr1 = sf.read(args.drum, always_2d=False)
        bass_data, sr2 = sf.read(args.bass, always_2d=False)
        if sr1 != sr2:
            from scipy.signal import resample
            target = max(sr1, sr2)
            if sr1 != target:
                drum_data = resample(drum_data, int(len(drum_data) * target / sr1))
                sr1 = target
            if sr2 != target:
                bass_data = resample(bass_data, int(len(bass_data) * target / sr2))
                sr2 = target
        audio = _to_mono(drum_data).astype(np.float32) + _to_mono(bass_data).astype(np.float32)
        sample_rate = sr1
        source_label = "drums+bass"
    else:
        audio, sample_rate = sf.read(args.input, always_2d=False, dtype="float32")
        audio = _to_mono(audio)
        source_label = "original"

    duration = len(audio) / sample_rate
    print(f"Audio ({source_label}): {duration:.1f}s @ {sample_rate}Hz, {len(audio)} samples")

    # ── run beat-this ──────────────────────────────────────────────
    print("Running beat-this inference...")
    tracker = File2Beats(checkpoint_path=args.model, device=device, dbn=args.dbn)
    with torch.no_grad():
        beats, downbeats = tracker(args.input)

    beats = [float(b) for b in beats]
    downbeats = [float(b) for b in downbeats]
    print(f"  → {len(beats)} beats, {len(downbeats)} downbeats")

    if len(beats) < 2:
        print("ERROR: Too few beats detected")
        return 1

    bi = np.median(np.diff(beats))
    bpm = 60.0 / bi
    print(f"  → BPM: {bpm:.1f}  (beat interval: {bi:.3f}s)")

    bpb = 4
    if downbeats:
        dbi = np.median(np.diff(downbeats))
        bpb = round(dbi / bi) if bi > 0 else 4
        print(f"  → Bar duration: {dbi:.3f}s  |  Beats/bar: {bpb}")

    # ── text beat table ────────────────────────────────────────────
    _print_beat_table(beats, downbeats, bpb)

    # ── plot ───────────────────────────────────────────────────────
    fig, (ax_full, ax_zoom), (bpm, bpb) = make_figure(
        audio, sample_rate, beats, downbeats,
        title=f"Beat-This: {Path(args.input).name}  [{source_label}]",
    )
    full_lim = (ax_full.get_xlim(), ax_zoom.get_xlim())
    zoom_active = [False]
    playing = [False]
    stream = [None]

    import sounddevice as sd

    def toggle_playback():
        if playing[0]:
            if stream[0] is not None:
                stream[0].stop(); stream[0].close(); stream[0] = None
            playing[0] = False
            print("Playback stopped")
            return
        cf = [0]

        def cb(outdata, frames, _ti, _st):
            end = min(cf[0] + frames, len(audio))
            outdata[:end - cf[0], 0] = audio[cf[0]:end]
            if end - cf[0] < frames:
                outdata[end - cf[0]:, 0] = 0.0
                raise sd.CallbackStop
            cf[0] = end

        cf[0] = 0
        playing[0] = True
        stream[0] = sd.OutputStream(samplerate=sample_rate, channels=1,
                                     callback=cb, blocksize=2048)
        stream[0].start()
        print("Playing (press 'p' to stop)")

    def on_key(event):
        if event.key == "b":
            zoom_active[0] = not zoom_active[0]
            if zoom_active[0] and len(beats) >= 4:
                c = (beats[0] + beats[min(31, len(beats) - 1)]) * 0.5
                h = max(8.0, (beats[min(31, len(beats) - 1)] - beats[0]) * 0.6)
                ax_full.set_xlim(c - h, c + h)
                ax_zoom.set_xlim(c - h, c + h)
            else:
                ax_full.set_xlim(full_lim[0])
                ax_zoom.set_xlim(full_lim[1])
            fig.canvas.draw_idle()
        elif event.key == "r":
            ax_full.set_xlim(full_lim[0])
            ax_zoom.set_xlim(full_lim[1])
            fig.canvas.draw_idle()
        elif event.key == "s":
            p = Path(args.input).stem + "_beats.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved: {p}")
        elif event.key == "p" and not args.no_play:
            toggle_playback()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return 0


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        return data.mean(axis=1)
    return data


def _print_beat_table(beats, downbeats, beats_per_bar):
    print(f"\n{'Bar':>4} {'Downbeat':>10}  Beats in bar")
    print("-" * 48)
    d = iter(downbeats)
    db = next(d, None)
    bar = 0
    for bt in beats:
        if db is not None and abs(bt - db) < 1e-3:
            bar += 1
            if bar > 16:
                break
            next_db = next(d, None)
            bar_beats = [b for b in beats if b >= bt and (next_db is None or b < next_db)]
            times = [f"{b:.2f}" for b in bar_beats[:beats_per_bar]]
            print(f"{bar:>4} {bt:>10.3f}  {times}")
            db = next_db


if __name__ == "__main__":
    raise SystemExit(main())
