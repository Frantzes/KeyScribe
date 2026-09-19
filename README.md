# KeyScribe

KeyScribe is a desktop application for polyphonic note detection, sheet music generation, and video transcription from recorded audio. Built in Rust, it is optimized for accuracy, responsiveness, and portable distribution.

[Visit website](https://keyscribe.frantzeselzaurdia.com/)

> **Status: Beta (active development).**
> The app is evolving quickly; behavior, UI, and outputs may change between releases.

## Downloads

Latest release: [Releases page](https://github.com/Frantzes/KeyScribe/releases/latest)

| Platform | Portable bundle |
| --- | --- |
| Windows x64 | [keyscribe-windows-x64.zip](https://github.com/Frantzes/KeyScribe/releases/latest/download/keyscribe-windows-x64.zip) |
| Linux x64 | [keyscribe-linux-x64.zip](https://github.com/Frantzes/KeyScribe/releases/latest/download/keyscribe-linux-x64.zip) (CPU) or `keyscribe-linux-x86_64.flatpak` from the same release (includes GPU) |
| macOS Apple Silicon | [keyscribe-macos-arm64.dmg](https://github.com/Frantzes/KeyScribe/releases/latest/download/keyscribe-macos-arm64.dmg) |

### Runtime downloads

The bundles ship only the small core. The optional assets below are fetched
on first use, always over HTTPS, verified against the pinned byte size and
SHA-256, and discarded if either check fails. The exact constants live in
[`src/assets.rs`](src/assets.rs).

**1. Demucs stem-separation model** — only when stem separation runs with a
local Demucs model; the default MVSep separation is cloud-based and never
downloads this (272 MB).

- URL: https://github.com/Frantzes/KeyScribe/releases/download/assets-v1/htdemucs_6s.onnx
- SHA-256: `b268327119b709141d43ba4103e7bb5f54bbf33311d6f1af5054ef663bf83322`
- Source/license: Demucs v4 `htdemucs_6s` (MIT); ONNX conversion hosted on this repository's `assets-v1` release.

**2. FFmpeg (Windows)** — first time a file needs it: an audio format
Symphonia cannot decode (mkv, webm, wma, ...) or any video (121 MB).

- URL: https://github.com/Frantzes/KeyScribe/releases/download/assets-v1/keyscribe-ffmpeg-win64-v1.zip
- SHA-256: `502c913aca7e637314088b5d6c5d92359a65ac8ee5e8b91daa9b2fe4c1edc520`
- Source/license: [BtbN FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) build `ffmpeg-n9.0-latest-win64-gpl-9.0` (GPLv3), mirrored here. FFmpeg source: https://ffmpeg.org/download.html

**3. Windows GPU pack** — first local Demucs run on a machine with an NVIDIA
GPU (`nvidia-smi` present). Skipped entirely on other machines. ~1.5 GB total:

- ONNX Runtime GPU 1.24.4 CUDA provider — Microsoft (MIT), PyPI wheel:
  https://files.pythonhosted.org/packages/fa/bc/35f3a37226d7a28c84b8b456f52237ccd39eb7111114bcf9ac340178e1ec/onnxruntime_gpu-1.24.4-cp313-cp313-win_amd64.whl
  SHA-256 `6be8bf2048777c517fca33eb61e114969fa326619feaa789d8c75f24337ea762`
- NVIDIA cuDNN 9.3.0.75 for CUDA 12:
  https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.3.0.75_cuda12-archive.zip
  SHA-256 `864a85dc67c7f92b9a8639f323acb4af63ad65de2ca82dccdf2c0b6a701c27c0`
- CUDA 12.4 runtime components (`cuda_cudart`, `cuda_nvrtc`, `libcublas`, `libcufft`, `libcurand`) from
  https://developer.download.nvidia.com/compute/cuda/redist, using NVIDIA's
  published SHA-256 values from `redistrib_12.4.1.json`.

Downloaded files land next to the executable (`models/`, `cuda/`, `ffmpeg.exe`);
if that location is read-only (Program Files, AppImage, Flatpak), the per-user
data directory is used instead.

### Offline installation

Place the files next to `keyscribe` before first launch and no downloader
runs: `models/htdemucs_6s.onnx` for local Demucs separation; `ffmpeg.exe` and
`ffprobe.exe` on Windows (or install FFmpeg on `PATH`); GPU pack DLLs in the
`cuda/` folder next to the executable.

## Features

- **Audio & Video Import:** Drag-and-drop or open via file dialog. Supports wav, mp3, flac, ogg, m4a, aac, mp4, mkv, avi, mov, webm.
- **Waveform & Piano Roll:** Real-time interactive waveform display with an 88-key piano roll that highlights detected notes as the playhead moves.
- **AI Stem Separation:** Separate audio into Vocals, Bass, Drums, Piano, Guitar, and Other stems using Demucs. Visualize or listen to individual stems. GPU-accelerated on NVIDIA GPUs (CUDA + cuDNN); falls back to CPU on other hardware.
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

KeyScribe is a pure Rust application — all AI inference runs locally via ONNX Runtime (Demucs, Basic Pitch) and the `beat-this` Rust crate, with no Python dependency. Stem separation GPU acceleration requires an NVIDIA GPU with CUDA 12 and cuDNN 9 (downloaded automatically on first local separation on Windows, included in the Linux Flatpak); on other GPUs or CPUs it falls back to multi-threaded CPU inference.

### 1. Importing an Audio or Video File

When you drag and drop or import a file, KeyScribe initiates a high-concurrency pipeline:

- **Audio Decoding (Rust):** The audio track is decoded into a raw sample stream using `Symphonia`. A waveform visualization is generated and rendered instantly.
- **Video Decoding (FFmpeg):** For video files, FFmpeg is spawned as a subprocess to pipe raw RGBA frames for synchronized playback.
- **Hashing & Cache Check (Rust):** A unique hash of the audio is calculated. KeyScribe checks the local analysis cache. If a cache hit occurs, transcription data loads instantly.
- **Parallel Analysis (Rust):** If no cache is found, stem separation (Demucs), transcription (Basic Pitch), and beat tracking (beat-this) run concurrently.

### 2. Stem Separation & Per-Stem Analysis

- **Stem Separation (Demucs):** The ONNX-exported Demucs model separates the audio into Vocals, Bass, Drums, Piano, Guitar, and Other stems. GPU acceleration via NVIDIA CUDA + cuDNN (verified at startup with automatic CPU fallback if unavailable).
- **Background Stem Analysis:** Each stem is independently analyzed for note probabilities. The UI lets you toggle which stems appear on the piano roll ("Visualize") or in the playback mix ("Listen").

### 3. Sheet Music Generation

Once the initial analysis is complete, you can generate sheet music from the **Sheet Music** tab:

1. **Stem Selection:** Choose which stems feed the melody and chord detection (or use the full mix).
2. **Tempo & Beat Tracking:** The `beat-this` Rust crate establishes a precise tempo map.
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
- **FFmpeg** for audio fallback decoding and video playback (bundled on Linux, downloaded on first use on Windows, system install on macOS)

### Build & Run

```bash
# Build release
cargo build --release

# Run app
cargo run --release
```

## License

GNU AGPL-3. See LICENSE.
