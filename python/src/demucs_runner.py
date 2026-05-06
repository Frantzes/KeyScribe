import argparse
import sys
from typing import List

from demucs import separate
import torch


def build_args(
    input_path: str,
    output_dir: str,
    model: str,
    format_name: str,
    bitrate: int,
    device: str,
) -> List[str]:
    args = [
        "-n",
        model,
        input_path,
        "-o",
        output_dir,
        "--device",
        device,
    ]

    if format_name == "mp3":
        args.extend(["--mp3", "--mp3-bitrate", str(bitrate)])
    elif format_name == "flac":
        args.append("--flac")

    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run demucs separation with compressed output.")
    parser.add_argument("input", help="Path to input audio file.")
    parser.add_argument("-o", "--output", default="demucs_output", help="Output folder.")
    parser.add_argument("-m", "--model", default="htdemucs_6s", help="Demucs model name.")
    parser.add_argument(
        "-f",
        "--format",
        choices=["mp3", "flac"],
        default="mp3",
        help="Output audio format.",
    )
    parser.add_argument(
        "-b",
        "--bitrate",
        type=int,
        default=192,
        help="MP3 bitrate in kbps (used only for mp3 output).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Demucs device selection. Uses CUDA when available by default.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    demucs_args = build_args(
        args.input,
        args.output,
        args.model,
        args.format,
        args.bitrate,
        device,
    )

    separate.main(demucs_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
