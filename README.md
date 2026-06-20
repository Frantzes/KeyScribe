# KeyScribe

KeyScribe is a desktop application for polyphonic note detection, sheet music generation, and video transcription from recorded audio. Built in Rust, it is optimized for accuracy, responsiveness, and portable distribution.

[Visit website](https://keyscribe.frantzeselzaurdia.com/)

> **Status: Beta (active development).**
> The app is evolving quickly; behavior, UI, and outputs may change between releases.

## Features

- **Audio & Video Import:** Drag-and-drop or open via file dialog. Supports wav, mp3, flac, ogg, m4a, aac, mp4, mkv, avi, mov, webm.
- **Waveform & Piano Roll:** Real-time interactive waveform display with an 88-key piano roll that highlights detected notes as the playhead moves.
- **AI Stem Separation:** Separate audio into Vocals, Bass, Drums, and Other stems using Demucs. Visualize or listen to individual stems.
- **Per-Stem Piano Roll:** Toggle individual stems on the piano roll to see which instrument is playing what.
- **Video Playback:** Synchronized video with audio-master clock (VLC-style sync engine). Frame-accurate seeking, no accumulating drift.
- **Sheet Music Generation (Experimental):** Generate MusicXML from detected notes. In-app engraved preview via Verovio. Supports lead sheet, piano grand staff, and single staff modes. Export to MusicXML or PDF via MuseScore.
- **Chord Detection:** Automatic chord symbol extraction displayed on the piano roll and exported to sheet music.
- **Speed & Pitch Controls:** Adjust playback speed (0.5×–2×) and pitch (-12 to +12 semitones) independently using high-quality time-stretching.
- **Loop & Markers:** Create loop selections on the waveform. Add named markers with editable timestamps. Snap loop boundaries to markers.
- **Analysis Cache:** Processed results are cached by audio hash so re-opening a file loads instantly without re-analysis.
- **Audio Output Selection:** Choose your audio output device from the settings menu.
- **Keyboard Shortcuts:** Space (play/replay), K (play/pause), arrows (seek ±5s), Ctrl+arrows (shift loop), M (add marker).

## How It Works: The Pipeline

KeyScribe uses a hybrid architecture combining a high-performance Rust core with specialized Python subprocesses for state-of-the-art AI analysis.

### 1. Importing an Audio or Video File

When you drag and drop or import a file, KeyScribe initiates a high-concurrency pipeline:

- **Audio Decoding (Rust):** The audio track is decoded into a raw sample stream using `Symphonia`. A waveform visualization is generated and rendered instantly.
- **Video Decoding (FFmpeg):** For video files, FFmpeg is spawned as a subprocess to pipe raw RGBA frames for synchronized playback.
- **Hashing & Cache Check (Rust):** A unique hash of the audio is calculated. KeyScribe checks the local analysis cache. If a cache hit occurs, transcription data loads instantly.
- **Parallel Subprocesses (Python):** If no cache is found, Python subprocesses run in parallel for transcription and stem separation.

### 2. Stem Separation & Per-Stem Analysis

- **Stem Separation (Demucs):** The `demucs_runner.py` subprocess separates the audio into Vocals, Bass, Drums, and Other stems.
- **Background Stem Analysis:** Each stem is independently analyzed for note probabilities. The UI lets you toggle which stems appear on the piano roll ("Visualize") or in the playback mix ("Listen").

### 3. Sheet Music Generation

Once the initial analysis is complete, you can generate sheet music from the **Sheet Music** tab:

1. **Stem Selection:** Choose which stems feed the melody and chord detection (or use the full mix).
2. **Tempo & Beat Tracking:** The `beat_this_runner.py` subprocess establishes a precise tempo map.
3. **Quantization:** Detected notes are snapped to a musical grid (supports quarter, eighth, 16th, dotted, and triplet subdivisions).
4. **Melody Extraction:** Monophonic (skyline or heuristic with outlier suppression) or polyphonic mode.
5. **Chord Analysis:** Automatic chord symbol detection and per-bar chord changes.
6. **MusicXML Export:** Serialized to MusicXML format — openable in MuseScore, Sibelius, Finale, etc.
7. **In-App Preview:** Engraved sheet music rendered via Verovio with a live playback cursor.
8. **PDF Export:** If MuseScore is installed, export engraved PDFs directly.

### 4. Playback Sync Engine

KeyScribe uses a VLC-style audio-master clock for drift-free playback:

- **Master Clock:** The audio device's sample consumption is the single source of truth. All consumers (video, piano roll, waveform playhead) read from this clock.
- **Latency Compensation:** The clock subtracts the estimated output buffer latency so the playhead tracks the sample currently audible through the speakers, not the sample queued in the device buffer.
- **Video Sync:** The video player follows the master audio clock. Late frames are dropped, early frames are held, and hard seeks are only triggered on large drift (>400ms). Real frame rate is detected via ffprobe, eliminating PTS drift.
- **Keyboard Sync:** The piano roll reads from the same master clock with nearest-frame rounding, keeping key highlights locked to the audible audio.

## Quick Start

### Prerequisites

- **Rust 1.70+**
- **Python 3.10+** (with `torch` and `demucs` installed in the project environment)
- **FFmpeg** (for video playback)

### Build & Run

```bash
# Build release
cargo build --release

# Run app
cargo run --release
```

## License

GNU AGPL-3. See LICENSE.
