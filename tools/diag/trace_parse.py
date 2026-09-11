import sys
sys.path.insert(0, "tools/diag")
from side_by_side import load_onsets

tr = load_onsets("C:/Users/Fran/AppData/Local/Temp/opencode/conf_fix2.musicxml")
ref = load_onsets("out/omnibook/Confirmation.musicxml")
print("trans[:6]:", [(round(a, 3), b, round(c, 3)) for a, b, c in tr[:6]])
print("ref[:6]:", [(round(a, 3), b, round(c, 3)) for a, b, c in ref[:6]])
