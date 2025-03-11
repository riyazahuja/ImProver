instance instNormedAddCommGroup : NormedAddCommGroup ℚ where
  norm r := ‖(r : ℝ)‖
                      /-
                        r₁ r₂ : Rat
                        ⊢ Eq (Dist.dist r₁ r₂) (Norm.norm (HSub.hSub r₁ r₂))
                      -/
  dist_eq r₁ r₂ := by simp only [Rat.dist_eq, norm, Rat.cast_sub]
                      /-
                        🎉 no goals
                      -/


@[norm_cast, simp 1001]
-- Porting note: increase priority to prevent the left-hand side from simplifying
theorem norm_cast_real (r : ℚ) : ‖(r : ℝ)‖ = ‖r‖ :=
  rfl


@[norm_cast, simp]
theorem _root_.Int.norm_cast_rat (m : ℤ) : ‖(m : ℚ)‖ = ‖m‖ := by
  /-
    m : Int
    ⊢ Eq (Norm.norm ↑m) (Norm.norm m)
  -/
  rw [← Rat.norm_cast_real, ← Int.norm_cast_real]; congr 1
                                                   /-
                                                     🎉 no goals
                                                   -/


