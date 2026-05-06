import argparse
import json
from typing import List, Tuple

import torch
from beat_this.inference import File2Beats


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run beat_this inference and print beats as JSON."
    )
    parser.add_argument("input", help="Path to input audio file.")
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

    with torch.no_grad():
        beats, downbeats = tracker(args.input)

    out = {
        "beats": [float(b) for b in beats],
        "downbeats": [float(b) for b in downbeats],
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
