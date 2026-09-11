"""Dump reference melody onsets (in beats at the XML tempo) vs detected onsets.

Usage: python tools/diag/compare_onsets.py <ref.musicxml> <detected.txt>
detected.txt: lines "start_sec pitch" (from KEYSCRIBE_RAW_NOTES_DEBUG).
Prints both series in beats for the first N bars.
"""
import sys
import xml.etree.ElementTree as ET

ref_path = sys.argv[1]
det_path = sys.argv[2] if len(sys.argv) > 2 else None

tree = ET.parse(ref_path)
root = tree.getroot()

# tempo: first sound tempo attr or metronome
tempo = None
for perm in root.iter("per-minute"):
    tempo = float(perm.text)
    break
if tempo is None:
    for p in root.iter("part"):
        for m in p.iter("measure"):
            s = m.find("sound")
            if s is not None and s.get("tempo"):
                tempo = float(s.get("tempo"))
                break
        if tempo:
            break
if tempo is None:
    tempo = 208.0
beat_sec = 60.0 / tempo

divisions = None
onsets = []  # (beat_offset, midi, is_rest)
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
            if pitch is not None:
                step = pitch.findtext("step")
                alter = int(pitch.findtext("alter", "0") or 0)
                octave = int(pitch.findtext("octave"))
                base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
                midi = (octave + 1) * 12 + base + alter
            else:
                midi = -1
            if not chord:
                onsets.append((beat_pos, midi, rest))
            if dur is not None and divisions:
                beat_pos += int(dur.text) / divisions

print(f"tempo={tempo} divisions={divisions} notes={len(onsets)}")
print("ref onsets (beats, pitch):")
for b, m, r in onsets[:80]:
    print(f"  {b:7.3f}  {'rest' if r else m}")

if det_path:
    print("\ndetected onsets (beats):")
    for line in open(det_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 2:
            t, p = float(parts[0]), parts[1]
            print(f"  {t / beat_sec:7.3f}  {p}")
