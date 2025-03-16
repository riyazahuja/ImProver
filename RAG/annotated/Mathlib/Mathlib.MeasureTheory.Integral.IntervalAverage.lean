notation3 "⨍ "(...)" in "a".."b",
  "r:60:(scoped f => average (Measure.restrict volume (uIoc a b)) f) => r


theorem interval_average_symm (f : ℝ → E) (a b : ℝ) : (⨍ x in a..b, f x) = ⨍ x in b..a, f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    ⊢ Eq (MeasureTheory.average (MeasureTheory.MeasureSpace.volume.restrict (Set.u …
  -/
  rw [setAverage_eq, setAverage_eq, uIoc_comm]
  /-
    🎉 no goals
  -/


theorem interval_average_eq (f : ℝ → E) (a b : ℝ) :
    (⨍ x in a..b, f x) = (b - a)⁻¹ • ∫ x in a..b, f x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    ⊢ Eq (MeasureTheory.average (MeasureTheory.MeasureSpace.volume.restrict (Set.u …
  -/
  rcases le_or_lt a b with h | h
  · rw [setAverage_eq, uIoc_of_le h, Real.volume_Ioc, intervalIntegral.integral_of_le h,
      ENNReal.toReal_ofReal (sub_nonneg.2 h)]
  · rw [setAverage_eq, uIoc_of_ge h.le, Real.volume_Ioc, intervalIntegral.integral_of_ge h.le,
      ENNReal.toReal_ofReal (sub_nonneg.2 h.le), smul_neg, ← neg_smul, ← inv_neg, neg_sub]


theorem interval_average_eq_div (f : ℝ → ℝ) (a b : ℝ) :
    (⨍ x in a..b, f x) = (∫ x in a..b, f x) / (b - a) := by
  /-
    f : Real → Real
    a b : Real
    ⊢ Eq (MeasureTheory.average (MeasureTheory.MeasureSpace.volume.restrict (Set.u …
  -/
  rw [interval_average_eq, smul_eq_mul, div_eq_inv_mul]
  /-
    🎉 no goals
  -/

