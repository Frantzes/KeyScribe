"""Run beat_this inference on one or more audio files, print JSON array of results.

Always prints a JSON array:  [{beats:[...], downbeats:[...]}, ...]
"""

import argparse
import json
from typing import List

import numpy as np

if not hasattr(np, "float"):
    np.float = float

import torch
from beat_this.inference import File2Beats


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run beat_this inference on one or more files, print JSON array."
    )
    parser.add_argument(
        "input", nargs="+", help="Path(s) to input audio file(s)."
    )
    parser.add_argument("--model", default="final0", help="BeatThis checkpoint name.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device selection.",
    )
    parser.add_argument(
        "--dbn",
        action="store_true",
        help="Enable DBN postprocessing (requires madmom).",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    tracker = File2Beats(checkpoint_path=args.model, device=device, dbn=args.dbn)

    results: List[dict] = []
    for path in args.input:
        with torch.no_grad():
            beats, downbeats = tracker(path)
        results.append(
            {
                "beats": [float(b) for b in beats],
                "downbeats": [float(b) for b in downbeats],
            }
        )

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
