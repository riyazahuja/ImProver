import json

header = """import Mathlib.Tactic

def two : ℝ := 2"""

text = """example (x:ℝ) : 1+x = two → x = 1 := by
  intro h
  unfold two at h
  linarith"""

cmd = json.dumps({"cmd": header})
thm = json.dumps({"cmd": text})
print(cmd)
print()

print(thm)
