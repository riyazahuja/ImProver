import Mathlib.Tactic

def two : ℝ := 2

example (x:ℝ) : 1+x = two → x = 1 := by
  intro h
  unfold two at h
  linarith
