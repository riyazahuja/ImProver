theorem isIntegral_int_I : IsIntegral ℤ I := by
  /-
    ⊢ IsIntegral Int Complex.I
  -/
  refine ⟨X ^ 2 + C 1, monic_X_pow_add_C _ two_ne_zero, ?_⟩
  /-
    ⊢ Eq (Polynomial.eval₂ (algebraMap Int Complex) Complex.I (HAdd.hAdd (HPow.hPo …
  -/
  rw [eval₂_add, eval₂_X_pow, eval₂_C, I_sq, eq_intCast, Int.cast_one, neg_add_cancel]
  /-
    🎉 no goals
  -/


theorem isIntegral_rat_I : IsIntegral ℚ I :=
  isIntegral_int_I.tower_top


