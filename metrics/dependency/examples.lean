import ImProver.metrics.tagger
import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Data.Nat.Prime.Basic

import ImProver.metrics.tagger

@[improver_example test, version unoptimized]
example : True := by
  sorry

@[improver_example test, version optimized]
example : True := by
  trivial


@[improver_example strong_tactics, version unoptimized]
theorem qux {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  apply Left.add_nonneg
  . exact ha
  . exact hb

@[improver_example strong_tactics, version optimized]
theorem qux' {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  linarith
