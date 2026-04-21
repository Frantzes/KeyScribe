# KeyScribe

KeyScribe is desktop app for polyphonic note detection from recorded audio.
Built in Rust. Optimized for accuracy, responsiveness, portable distribution.

[Visit website](https://keyscribe.frantzeselzaurdia.com/)

> Status: Beta (active development).
> The app is still evolving quickly and behavior, UI, and outputs may change between releases.

## Product Overview

- Real-time waveform and piano visualization.
- Audio import support: wav, mp3, flac, ogg, m4a, aac.
- Two analysis modes: Standard mode for fast feedback, CQT Pro mode for cleaner note stability and fewer ghost notes.
- Cross-platform desktop bundles for Windows, macOS, Linux.

## How It Works

KeyScribe processes full audio file in deterministic DSP pipeline:

1. Decode audio to mono sample stream.
2. Optional preprocessing separates harmonic content from percussive noise.
3. Compute spectral representation (FFT or CQT-based path).
4. Estimate per-frame note probabilities across piano range (A0 to C8).
5. Apply temporal smoothing (Viterbi-style decoding + duration filtering).
6. Render timeline, waveform, and 88-key activity in UI.

Result: stable note timeline better suited for transcription workflow than raw frame-by-frame peaks.

## Quick Start

Prerequisites:

- Rust 1.70+
- Windows, macOS, or Linux

Build release:

```bash
cargo build --release
```

Run app:

```bash
cargo run --release
```

## Usage Flow

1. Open audio file.
2. Pick analysis mode.
3. Play, seek, inspect waveform and piano roll.
4. Tune speed or pitch preview if needed.
5. Use note activity timeline for transcription decisions.

## Technical Notes

- UI: eframe/egui
- Audio decode: Symphonia
- Playback: rodio
- DSP core: rustfft + custom CQT, preprocessing, viterbi modules
- Optional model integration path: ONNX Runtime crate (ort)

## License
GNU AGPL-3. See LICENSE.
