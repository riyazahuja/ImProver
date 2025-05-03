scoped[ProbabilityTheory] notation "𝔼[" X "|" m "]" =>
  MeasureTheory.condexp m MeasureTheory.MeasureSpace.volume X

-- `scoped[ProbabilityTheory]` isn't legal for `macro`s.

/-- `P[X]` is the expectation of `X` under the measure `P`.

Note that this notation can conflict with the `GetElem` notation for lists. Usually if you see an
error about ambiguous notation when trying to write `l[i]` for a list, it means that Lean could
not find `i < l.length`, and so fell back to trying this notation as well. -/
scoped macro:max P:term noWs "[" X:term "]" : term => `(∫ x, ↑($X x) ∂$P)

scoped[ProbabilityTheory] notation "𝔼[" X "]" => ∫ a, (X : _ → _) a


scoped[ProbabilityTheory] notation P "⟦" s "|" m "⟧" =>
  MeasureTheory.condexp m P (Set.indicator s fun ω => (1 : ℝ))


scoped[ProbabilityTheory] notation:50 X " =ₐₛ " Y:50 => X =ᵐ[MeasureTheory.MeasureSpace.volume] Y


scoped[ProbabilityTheory] notation:50 X " ≤ₐₛ " Y:50 => X ≤ᵐ[MeasureTheory.MeasureSpace.volume] Y


scoped[ProbabilityTheory] notation "∂" P "/∂" Q:100 => MeasureTheory.Measure.rnDeriv P Q


scoped[ProbabilityTheory] notation "ℙ" => MeasureTheory.MeasureSpace.volume

