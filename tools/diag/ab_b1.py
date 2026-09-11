import json

base = "C:/Users/Fran/AppData/Local/Temp/opencode/corpus_b1_"
r = {}
for name in ["learned", "legacy"]:
    r[name] = json.load(open(base + name + ".json"))

keys = [
    "mean_pitch_accuracy", "mean_note_accuracy", "mean_recall",
    "mean_onset_error_beats", "mean_duration_error",
    "mean_root_match_rate", "mean_exact_match_rate", "mean_score",
]
print(f"{'metric':<24}{'learnedB1':>10}{'legacy':>10}   plan gate")
gates = {
    "mean_note_accuracy": ">= 0.42 FAIL",
    "mean_duration_error": "<= 0.40 FAIL",
    "mean_pitch_accuracy": ">= 0.826 ok",
    "mean_onset_error_beats": "<= 0.256 ok",
}
for k in keys:
    print(f"{k:<24}{r['learned'][k]:>10.4f}{r['legacy'][k]:>10.4f}   {gates.get(k, '')}")

# worst per-track regressions learned vs legacy
names = [t.get("name") or t.get("track") or f"track{i}" for i, t in enumerate(r["learned"]["tracks"])]
diffs = []
for i, name in enumerate(names):
    tl = r["learned"]["tracks"][i]
    tg = r["legacy"]["tracks"][i]
    d = tl["note_accuracy"] - tg["note_accuracy"]
    diffs.append((d, name, tl["note_accuracy"], tg["note_accuracy"]))
diffs.sort()
print("\nworst 5 regressions (note acc, learned - legacy):")
for d, n, a, b in diffs[:5]:
    print(f"  {n:<28} {a:.3f} vs {b:.3f}  ({d:+.3f})")
print("best 5 gains:")
for d, n, a, b in diffs[-5:]:
    print(f"  {n:<28} {a:.3f} vs {b:.3f}  ({d:+.3f})")
