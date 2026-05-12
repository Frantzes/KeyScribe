# KeyScribe

KeyScribe is a desktop application for polyphonic note detection and sheet music generation from recorded audio. Built in Rust, it is optimized for accuracy, responsiveness, and portable distribution.

[Visit website](https://keyscribe.frantzeselzaurdia.com/)

> **Status: Beta (active development).**
> The app is evolving quickly; behavior, UI, and outputs may change between releases.

## How It Works: The Pipeline

KeyScribe uses a hybrid architecture combining a high-performance Rust core with specialized Python subprocesses for state-of-the-art AI analysis.

### 1. Importing an Audio File
When you drag and drop or import an audio file, KeyScribe initiates a high-concurrency pipeline to provide immediate feedback:

*   **Audio Decoding (Rust):** The file is immediately decoded into a raw mono sample stream using `Symphonia`. A waveform visualization is generated and rendered instantly.
*   **Hashing & Cache Check (Rust):** A unique hash of the audio is calculated. KeyScribe checks the local analysis cache to see if this song has been processed before. If a cache hit occurs, transcription data is loaded instantly.
*   **Parallel Subprocesses (Python):** If no cache is found, KeyScribe spawns two independent Python subprocesses to run in parallel:
    *   **Subprocess A (Transcription):** Runs the `main.py` runner (using a polyphonic transcription model like Basic Pitch) to estimate note probabilities across the full mix.
    *   **Subprocess B (Stem Separation):** Runs the `demucs_runner.py` (using Meta's Demucs) to separate the audio into distinct stems (Vocals, Drums, Bass, Other).
*   **Background Stem Analysis:** As soon as the stems are separated, the Rust core begins a second pass, analyzing each individual stem for note probabilities. This is what allows you to eventually "Visualize" or "Listen" to specific instruments.

### 2. Sheet Music Generation
Once the initial analysis is complete, you can generate editable sheet music. This process is deterministic and follows these steps after you click **Generate**:

1.  **Stem Selection:** The generator takes the note probabilities from the specific stems you have enabled in the UI (e.g., just the "Other" stem for piano, or "Bass" for a bass clef part).
2.  **Temporal Decoding (Viterbi):** The raw frame-by-frame probabilities are passed through a Viterbi-style decoder to find the most likely sequence of note onsets and durations, filtering out "ghost notes" and noise.
3.  **Tempo & Beat Tracking:** The `beat_this_runner.py` subprocess is invoked to establish a precise tempo map (BPM) and align the detected notes with a musical grid.
4.  **Quantization:** Detected notes are snapped to the nearest musical subdivision (quarter notes, eighth notes, etc.) based on your quantization settings.
5.  **Chord Analysis:** The pipeline identifies harmonic structures to add chord symbols to the sheet.
6.  **MusicXML Export:** Finally, the data is serialized into `MusicXML` format, which can be opened in MuseScore, Sibelius, or Finale for further editing.

## Features

- **Real-time Visualization:** Waveform and 88-key piano roll interaction.
- **AI Stem Separation:** Isolate vocals, bass, and drums to improve transcription accuracy.
- **CQT Pro Mode:** Advanced Constant-Q Transform for cleaner note stability.
- **Format Support:** wav, mp3, flac, ogg, m4a, aac.
- **Cross-Platform:** Native bundles for Windows, macOS, and Linux.

## Quick Start

### Prerequisites
- **Rust 1.70+**
- **Python 3.10+** (with `torch` and `demucs` installed in the project environment)

### Build & Run
```bash
# Build release
cargo build --release

# Run app
cargo run --release
```

## License
GNU AGPL-3. See LICENSE.
