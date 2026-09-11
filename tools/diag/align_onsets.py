"""Estimate the tempo-scale + offset that best aligns detected onsets to the
reference melody, then report per-note residuals.

Usage: python tools/diag/align_onsets.py <ref.musicxml> <detected.txt>
"""
import sys
import xml.etree.ElementTree as ET

ref_path, det_path = sys.argv[1], sys.argv[2]

tree = ET.parse(ref_path)
root = tree.getroot()
tempo = 208.0
for perm in root.iter("per-minute"):
    tempo = float(perm.text)
    break
beat_sec = 60.0 / tempo

divisions = None
ref = []  # seconds, midi
beat_pos = 0.0
part = root.find("part")
for measure in part.iter("measure"):
    for el in measure:
        if el.tag == "attributes":
            d = el.find("divisions")
            if d is not None:
                divisions = int(d.text)
        elif el.tag == "note":
            dur = el.find("duration")
            chord = el.find("chord") is not None
            rest = el.find("rest") is not None
            pitch = el.find("pitch")
            midi = -1
            if pitch is not None:
                step = pitch.findtext("step")
                alter = int(pitch.findtext("alter", "0") or 0)
                octave = int(pitch.findtext("octave"))
                base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
                midi = (octave + 1) * 12 + base + alter
            if not chord and not rest and midi >= 0:
                ref.append((beat_pos * beat_sec, midi))
            if dur is not None and divisions:
                beat_pos += int(dur.text) / divisions

det = []
for line in open(det_path, encoding="utf-8"):
    p = line.split()
    if len(p) >= 2:
        det.append((float(p[0].rstrip("s")), p[1]))

# grid search scale/offset by note-index correspondence: match pitch sequences.
# Simple approach: take detected pitch sequence and ref pitch sequence, align
# by longest common subsequence-ish greedy walk, then regress.
# Here: brute force scale in [0.9, 1.1], offset in [-0.5, 0.5], maximize count
# of det onsets within 30ms of some ref onset.
best = (0, 0.0, 0.0)
s = 0.95
while s <= 1.05:
    o = -0.4
    while o <= 0.4:
        cnt = 0
        ri = 0
        transformed = [(t * s + o, m) for t, m in ref]
        for dt, dp in det:
            for rt, rm in transformed:
                if abs(rt - dt) < 0.030:
                    cnt += 1
                    break
        if cnt > best[0]:
            best = (cnt, s, o)
        o += 0.01
    s += 0.0005

cnt, s, o = best
print(f"best match: {cnt}/{len(det)} detected onsets within 30ms "
      f"(scale={s:.4f}, offset={o:+.2f}s => effective tempo {tempo/s:.2f}bpm)")

# residuals under best transform
trans = [(t * s + o, m) for t, m in ref]
print("\nper-note residual (detected - transformed ref), first 40 matches:")
shown = 0
for dt, dp in det:
    best_r = None
    for rt, rm in trans:
        if abs(rt - dt) < 0.25:
            r = dt - rt
            if best_r is None or abs(r) < abs(best_r):
                best_r = r
    if best_r is not None and shown < 40:
        sign = "+" if best_r >= 0 else ""
        print(f"  det {dt:7.3f}s residual {sign}{best_r*1000:6.1f}ms")
        shown += 1
