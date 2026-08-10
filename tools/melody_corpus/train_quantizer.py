"""Train and export the small melody duration-token model.

This is a reproducible bootstrap trainer for the Rust MelodyQuantizerInference
interface. It uses score MusicXML as supervision and adds performed-style onset
and duration jitter, which is useful for calibrating the tokenizer before a
larger A-MAPS/ASAP/GuitarSet/Leduc corpus is available.
"""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TOKENS = np.asarray([4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=np.float32)
TRIPLET_TOKENS = {9, 10, 11}


def parse_score(path: Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    root = ET.parse(path).getroot()
    divisions_node = root.find(".//divisions")
    divisions = float(divisions_node.text) if divisions_node is not None else 480.0
    tempo_node = root.find(".//per-minute")
    tempo = float(tempo_node.text) if tempo_node is not None else 180.0
    beats_node = root.find(".//time/beats")
    beats_per_bar = int(beats_node.text) if beats_node is not None else 4

    rows: list[tuple[float, float, int, int]] = []
    for part in root.findall(".//part"):
        cursor = 0.0
        for measure in part.findall("./measure"):
            for node in list(measure):
                if node.tag in {"backup", "forward"}:
                    duration = float(node.findtext("duration", "0")) / divisions
                    cursor += duration if node.tag == "forward" else -duration
                    continue
                if node.tag != "note":
                    continue
                duration = float(node.findtext("duration", "0")) / divisions
                is_chord = node.find("chord") is not None
                is_rest = node.find("rest") is not None
                is_grace = node.find("grace") is not None
                onset = cursor
                if not is_chord and not is_grace:
                    cursor += duration
                if is_chord or is_rest or is_grace:
                    continue
                pitch = node.find("pitch")
                if pitch is None:
                    continue
                step = pitch.findtext("step", "C")
                alter = int(pitch.findtext("alter", "0"))
                octave = int(pitch.findtext("octave", "4"))
                pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
                midi = (octave + 1) * 12 + pcs[step] + alter
                rows.append((onset, duration, midi, len(rows) % beats_per_bar))

    rows.sort(key=lambda row: row[0])
    starts = np.asarray([row[0] for row in rows], dtype=np.float32)
    durations = np.asarray([row[1] for row in rows], dtype=np.float32)
    pitches = np.asarray([row[2] for row in rows], dtype=np.float32)
    bars = np.asarray([row[3] for row in rows], dtype=np.float32)
    return np.column_stack((starts, durations, pitches, bars)), starts, tempo, beats_per_bar


def token_for_duration(duration: float) -> int:
    return int(np.argmin(np.abs(TOKENS - duration)))


def make_examples(rows: np.ndarray, tempo: float, beats_per_bar: int, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for onset, duration, pitch, beat_in_bar in rows:
        token = token_for_duration(float(duration))
        for _ in range(repeats):
            onset_jitter = float(rng.normal(0.0, 0.035))
            duration_jitter = float(rng.normal(0.0, max(0.025, duration * 0.10)))
            raw_pos = (onset % 1.0 + onset_jitter) % 1.0
            raw_duration = max(1.0 / 12.0, duration + duration_jitter)
            style = [1.0, 0.0, 0.0]
            if token in TRIPLET_TOKENS:
                style = [0.0, 0.0, 1.0]
            features.append(np.asarray([
                raw_pos,
                raw_duration,
                (np.clip(tempo, 40.0, 260.0) - 40.0) / 220.0,
                beat_in_bar / max(1, beats_per_bar),
                style[0], style[1], style[2],
                pitch / 127.0,
                0.92,
            ], dtype=np.float32))
            labels.append(token)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("tests/data/Omnibook xml"))
    parser.add_argument("--output", type=Path, default=Path("models/melody_quantizer.onnx"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    files = sorted(args.corpus.glob("*.xml"))
    if not files:
        raise SystemExit(f"no MusicXML files found in {args.corpus}")
    for path in files:
        rows, _, tempo, beats_per_bar = parse_score(path)
        x, y = make_examples(rows, tempo, beats_per_bar, args.repeats)
        all_x.append(x)
        all_y.append(y)

    x = torch.from_numpy(np.concatenate(all_x))
    y = torch.from_numpy(np.concatenate(all_y))
    model = Quantizer()
    counts = torch.bincount(y, minlength=12).float().clamp_min(1.0)
    weights = (counts.sum() / counts).sqrt()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, weight=weights)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"epoch={epoch + 1} loss={loss.item():.4f} accuracy={accuracy:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 4, 9), dtype=torch.float32)
    torch.onnx.export(
        model.eval(),
        example,
        args.output,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {1: "seq"}, "logits": {1: "seq"}},
        opset_version=18,
    )
    print(f"exported {args.output} examples={len(y)}")


if __name__ == "__main__":
    main()
