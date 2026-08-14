"""Side-by-side onset comparison (in beats) of reference vs transcribed
MusicXML melodies, with nearest-pitch matching to expose systematic offset.

Usage: python tools/diag/side_by_side.py <ref.musicxml> <trans.musicxml> [n]
"""
import sys
import xml.etree.ElementTree as ET

STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def load_onsets(path):
    """Return [(onset_beats, midi, dur_beats)] mirroring sheet_compare's parser
    (backup/forward cursor handling, chord notes at last onset)."""
    root = ET.parse(path).getroot()
    out = []
    divisions = 1
    part = root.find("part")
    cursor = 0.0
    last_onset = 0.0
    last_dur = 0.0
    for el in part.iter():
        if el.tag == "attributes":
            d = el.find("divisions")
            if d is not None:
                divisions = int(d.text)
        elif el.tag == "backup":
            cursor -= int(el.findtext("duration", "0")) / divisions
        elif el.tag == "forward":
            cursor += int(el.findtext("duration", "0")) / divisions
        elif el.tag == "note":
            chord = el.find("chord") is not None
            grace = el.find("grace") is not None
            dur = int(el.findtext("duration", "0")) / divisions
            pitch = el.find("pitch")
            midi = -1
            if pitch is not None:
                alter = int(pitch.findtext("alter", "0") or 0)
                midi = (int(pitch.findtext("octave")) + 1) * 12 + STEPS[pitch.findtext("step")] + alter
            onset = last_onset if chord else cursor
            if not chord and not grace:
                last_onset = onset
                last_dur = dur
                cursor += dur
            if midi >= 0 and not grace:
                out.append((onset, midi, dur if not chord else last_dur))
    return out


if __name__ == "__main__":
    ref = load_onsets(sys.argv[1])
    tr = load_onsets(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    # greedy nearest-with-same-pitch matching like compare_note_lists, tol 0.25
    used = [False] * len(ref)
    matched = 0
    errs = []
    for t in tr:
        best = None
        for i, r in enumerate(ref):
            if used[i] or r[1] != t[1]:
                continue
            e = abs(r[0] - t[0])
            if e <= 0.25 and (best is None or e < best[1]):
                best = (i, e)
        if best:
            used[best[0]] = True
            matched += 1
            errs.append((t[0], t[1], ref[best[0]][0] - t[0]))

    print(f"ref={len(ref)} trans={len(tr)} matched={matched} "
          f"note_acc={matched/max(1,len(tr)):.3f} recall={matched/max(1,len(ref)):.3f}")

    print("\nper-note onset delta (ref - trans, beats), first notes:")
    for t0, p, d in errs[:n]:
        print(f"  trans@{t0:8.3f} pitch={p:3d} delta={d:+.3f}")

    # histogram of deltas
    import collections
    hist = collections.Counter(round(d * 4) / 4 for _, _, d in errs)
    print("\ndelta histogram (quarter-beat bins):")
    for k in sorted(hist):
        print(f"  {k:+.2f}: {hist[k]}")
