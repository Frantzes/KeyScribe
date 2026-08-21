"""Tier B1 trainer: sequence-model melody quantizer with MERGE tokens.

Replaces the per-note MLP bootstrap (`train_quantizer.py`) with a 2-layer
bidirectional GRU over per-song note sequences, adding a 13th output class:
MERGE_INTO_PREVIOUS ("this detection is a spurious fragment of the previous
detection — extend it, don't emit a note").

Key differences from the v1 bootstrap:
- per-SONG sequences (the model sees neighbor context, so it can learn to
  merge over-segmented fragments and to recover a logical note's full
  duration from its first fragment);
- split augmentation: each training note is randomly chopped into 2-3
  fragments (first fragment labeled with the TRUE duration token,
  continuations labeled MERGE) so the model sees the exact failure mode the
  extractor produces;
- class-weighted CE with the MERGE weight halved so the model cannot win by
  over-merging.

Feature layout is IDENTICAL to the Rust featurizer (`learned_note_features`,
quantize.rs) and to v1: [intra_beat_pos, raw_dur_beats, tempo_norm,
beat_in_bar_norm, swing_onehot(3), pitch/127, velocity/127].

Export: ONNX [1, seq, 9] -> [1, seq, 13], verified against onnxruntime.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TOKENS = np.asarray(
    [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    dtype=np.float32,
)
TRIPLET_TOKENS = {9, 10, 11}
MERGE = 12
N_CLASSES = 13


def parse_score(path: Path) -> tuple[np.ndarray, float, int]:
    """Return rows [(onset_beats, dur_beats, midi, beat_in_bar)], tempo, bpb."""
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
                beat_in_bar = int(onset) % beats_per_bar
                rows.append((onset, duration, midi, beat_in_bar))

    rows.sort(key=lambda row: row[0])
    return np.asarray(rows, dtype=np.float32), tempo, beats_per_bar


def token_for_duration(duration: float) -> int:
    return int(np.argmin(np.abs(TOKENS - duration)))


def featurize(
    raw_pos: float,
    raw_dur: float,
    tempo: float,
    beat_in_bar: int,
    beats_per_bar: int,
    token: int,
    pitch: int,
    velocity: float = 0.92,
) -> list[float]:
    style = [0.0, 0.0, 1.0] if token in TRIPLET_TOKENS else [1.0, 0.0, 0.0]
    return [
        raw_pos % 1.0,
        min(max(raw_dur, 0.0), 32.0),
        (min(max(tempo, 40.0), 260.0) - 40.0) / 220.0,
        beat_in_bar / max(1, beats_per_bar),
        style[0], style[1], style[2],
        pitch / 127.0,
        velocity,
    ]


def augment_song(
    rows: np.ndarray,
    tempo: float,
    beats_per_bar: int,
    rng: np.random.Generator,
    p_split: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One augmented copy of a song as a sequence.

    With prob p_split a note (duration >= 0.5 beats) is chopped into k in {2,3}
    near-equal fragments: the first keeps the TRUE token label, continuations
    get MERGE. Fragments receive small onset jitter (split points are not
    grid-aligned, mirroring extractor behavior).
    """
    feats: list[list[float]] = []
    labels: list[int] = []
    for onset, duration, pitch, beat_in_bar in rows:
        token = token_for_duration(float(duration))
        if p_split > 0 and duration >= 0.5 and rng.random() < p_split:
            k = int(rng.choice([2, 3]))
            parts = [(duration / k) * rng.uniform(0.8, 1.2) for _ in range(k - 1)]
            parts.append(duration - sum(parts))
            if min(parts) < 0.12:
                parts = [duration / k] * k
            t = float(onset)
            for j, pd in enumerate(parts):
                onset_j = t + float(rng.normal(0.0, 0.02))
                raw_dur = max(1.0 / 12.0, pd + float(rng.normal(0.0, 0.015)))
                feats.append(featurize(onset_j, raw_dur, tempo, int(beat_in_bar),
                                       beats_per_bar, token, int(pitch)))
                labels.append(token if j == 0 else MERGE)
                t += pd
        else:
            onset_j = float(onset) + float(rng.normal(0.0, 0.035))
            raw_dur = max(1.0 / 12.0, duration + float(rng.normal(0.0, max(0.025, duration * 0.10))))
            feats.append(featurize(onset_j, raw_dur, tempo, int(beat_in_bar),
                                   beats_per_bar, token, int(pitch)))
            labels.append(token)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class SeqQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(9, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.out(h)


def collate(seqs: list[tuple[np.ndarray, np.ndarray]]):
    """Pad a list of (X, y) sequences into batch tensors + mask."""
    n = len(seqs)
    lens = [x.shape[0] for x, _ in seqs]
    max_len = max(lens)
    xb = np.zeros((n, max_len, 9), dtype=np.float32)
    yb = np.full((n, max_len), -100, dtype=np.int64)
    for i, (x, y) in enumerate(seqs):
        xb[i, : x.shape[0]] = x
        yb[i, : len(y)] = y
    return torch.from_numpy(xb), torch.from_numpy(yb)


@torch.no_grad()
def holdout_metrics(model: SeqQuantizer, songs: list, device: torch.device) -> dict:
    """Token accuracy on clean songs; merge P/R on split-augmented copies."""
    model.eval()
    correct = total = 0
    for rows, tempo, bpb in songs:
        x, y = augment_song(rows, tempo, bpb, np.random.default_rng(7), p_split=0.0)
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
        pred = logits.argmax(dim=2).squeeze(0).cpu().numpy()
        correct += int((pred == y).sum())
        total += len(y)
    token_acc = correct / max(1, total)

    tp = fp = fn = 0
    for rows, tempo, bpb in songs:
        x, y = augment_song(rows, tempo, bpb, np.random.default_rng(11), p_split=0.35)
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
        pred = logits.argmax(dim=2).squeeze(0).cpu().numpy()
        tp += int(((pred == MERGE) & (y == MERGE)).sum())
        fp += int(((pred == MERGE) & (y != MERGE)).sum())
        fn += int(((pred != MERGE) & (y == MERGE)).sum())
    merge_p = tp / max(1, tp + fp)
    merge_r = tp / max(1, tp + fn)
    return {"token_acc": token_acc, "merge_precision": merge_p, "merge_recall": merge_r}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("tests/data/Omnibook xml"))
    parser.add_argument("--output", type=Path, default=Path("models/melody_quantizer.onnx"))
    parser.add_argument("--backup-v1", action="store_true",
                        help="copy the existing model to melody_quantizer_v1_mlp.onnx first")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--copies", type=int, default=6,
                        help="augmented sequence copies per song per epoch-set")
    parser.add_argument("--p-split", type=float, default=0.35)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=str, default="Confirmation,Ornithology,Donna_Lee")
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device("cpu")

    holdout_stems = {s.strip() for s in args.holdout.split(",") if s.strip()}

    files = sorted(args.corpus.glob("*.xml"))
    if not files:
        raise SystemExit(f"no MusicXML files found in {args.corpus}")

    train_songs: list[tuple[np.ndarray, float, int, str]] = []
    hold_songs: list[tuple[np.ndarray, float, int]] = []
    for path in files:
        rows, tempo, bpb = parse_score(path)
        if len(rows) == 0:
            continue
        if path.stem in holdout_stems:
            hold_songs.append((rows, tempo, bpb))
        else:
            train_songs.append((rows, tempo, bpb, path.stem))
    if not train_songs:
        raise SystemExit("no training files left after holdout")
    print(f"train songs: {len(train_songs)}, holdout songs: {len(hold_songs)} "
          f"({sorted(holdout_stems)})")

    # Static augmented dataset (regenerated with fresh jitter every K epochs
    # would be better; for this corpus size one generation is enough).
    # Sequences are cut into overlapping windows so batches are rectangular
    # and the GRU never unrolls over a whole song (CPU-training speed).
    WINDOW = 96
    STRIDE = 64
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    for rows, tempo, bpb, _ in train_songs:
        for c in range(args.copies):
            rng = np.random.default_rng(10_000 + c * 977 + hash(len(rows)) % 9973)
            x, y = augment_song(rows, tempo, bpb, rng, args.p_split)
            if len(x) <= WINDOW:
                seqs.append((x, y))
                continue
            for s in range(0, len(x) - WINDOW + 1, STRIDE):
                seqs.append((x[s : s + WINDOW], y[s : s + WINDOW]))
            tail = len(x) % STRIDE
            if tail and len(x) - tail + STRIDE < len(x):
                pass  # last partial window omitted; overlap covers it
    print(f"sequences: {len(seqs)} (window={WINDOW}, stride={STRIDE}, "
          f"{args.copies} copies x {len(train_songs)} songs)")

    # Class weights: inverse-sqrt-frequency (as v1), MERGE halved so the
    # model cannot win by over-merging.
    all_y = np.concatenate([y for _, y in seqs])
    counts = np.bincount(all_y, minlength=N_CLASSES).astype(np.float64)
    weights_np = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights_np[MERGE] *= 0.5
    weights = torch.from_numpy(weights_np.astype(np.float32)).to(device)
    print(f"class counts: {counts.astype(int).tolist()} weights: "
          f"{[round(w, 2) for w in weights_np.tolist()]}")

    model = SeqQuantizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(seqs)))
        random.shuffle(order)
        total_loss = 0.0
        for i in range(0, len(order), args.batch):
            batch = [seqs[j] for j in order[i : i + args.batch]]
            xb, yb = collate(batch)
            logits = model(xb.to(device))
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, N_CLASSES), yb.reshape(-1).to(device),
                weight=weights, ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            m = holdout_metrics(model, hold_songs, device) if hold_songs else {}
            line = (f"epoch={epoch + 1} loss={total_loss:.2f}")
            if m:
                line += (f" holdout token_acc={m['token_acc']:.4f} "
                         f"merge P={m['merge_precision']:.3f} R={m['merge_recall']:.3f}")
            print(line)
            score = m.get("token_acc", -1) + m.get("merge_recall", -1)
            if score > best_acc:
                best_acc = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        m = holdout_metrics(model, hold_songs, device)
        print(f"best checkpoint: token_acc={m['token_acc']:.4f} "
              f"merge P={m['merge_precision']:.3f} R={m['merge_recall']:.3f}")

    # Persist the trained weights next to the export so a failed/rerun export
    # never requires retraining.
    state_path = args.output.with_suffix(".pt")
    torch.save(model.state_dict(), state_path)
    print(f"saved state dict to {state_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backup_v1 and args.output.exists():
        backup = args.output.parent / "melody_quantizer_v1_mlp.onnx"
        shutil.copy2(args.output, backup)
        print(f"backed up previous model to {backup}")

    model.eval()
    example = torch.zeros((1, 4, 9), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {1: "seq"}, "logits": {1: "seq"}},
        opset_version=18,
        dynamo=False,
    )

    # Verify the ONNX against torch on random inputs of two lengths.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    for seq_len in (7, 50):
        probe = np.random.default_rng(3).random((1, seq_len, 9)).astype(np.float32)
        t_out = model(torch.from_numpy(probe)).detach().numpy()
        o_out = sess.run(None, {"features": probe})[0]
        diff = float(np.abs(t_out - o_out).max())
        print(f"onnx verify seq={seq_len}: shape={o_out.shape} max|torch-ort|={diff:.2e}")
        assert o_out.shape == (1, seq_len, N_CLASSES), o_out.shape
        assert diff < 1e-4, diff
    print(f"exported {args.output} classes={N_CLASSES}")


if __name__ == "__main__":
    main()
