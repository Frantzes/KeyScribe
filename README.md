# Transcriber

Transcriber is desktop app for polyphonic note detection from recorded audio.
Built in Rust. Optimized for accuracy, responsiveness, portable distribution.

## Product Overview

- Real-time waveform and piano visualization.
- Audio import support: wav, mp3, flac, ogg, m4a, aac.
- Two analysis modes: Standard mode for fast feedback, CQT Pro mode for cleaner note stability and fewer ghost notes.
- Cross-platform desktop bundles for Windows, macOS, Linux.

## How It Works

Transcriber processes full audio file in deterministic DSP pipeline:

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

## Portable Builds

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-windows.ps1
```

Outputs:

- build/windows/transcriber-windows-x64/
- build/windows/transcriber-windows-x64.zip

### macOS

```bash
chmod +x scripts/build-macos.sh
./scripts/build-macos.sh
```

Outputs:

- build/macos/transcriber-macos-arm64/ or build/macos/transcriber-macos-x64/
- build/macos/transcriber-macos-arm64.zip or build/macos/transcriber-macos-x64.zip

### Linux

```bash
chmod +x scripts/build-linux.sh
./scripts/build-linux.sh
```

Outputs:

- build/linux/transcriber-linux-x64/ or build/linux/transcriber-linux-arm64/
- build/linux/transcriber-linux-x64.zip or build/linux/transcriber-linux-arm64.zip

## Automated Releases (Tags)

Tag push triggers [.github/workflows/release-on-tag.yml](.github/workflows/release-on-tag.yml).
Workflow builds Windows, macOS, Linux bundles and uploads zip artifacts to GitHub Release.

```bash
git tag v0.1.0
git push origin v0.1.0
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

## Limitations

- File-based workflow only, no live microphone transcription yet.
- Outputs note activations, not chord names or notation export.
- CQT Pro mode introduces intentional smoothing latency for stability.

## More Documentation

- [CQT_PRO_MODE.md](CQT_PRO_MODE.md)
- [HFSFORMER_IMPLEMENTATION.md](HFSFORMER_IMPLEMENTATION.md)

## License

MIT. See LICENSE.
