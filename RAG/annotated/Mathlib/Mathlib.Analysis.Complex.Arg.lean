theorem sameRay_iff : SameRay ℝ x y ↔ x = 0 ∨ y = 0 ∨ x.arg = y.arg := by
  /-
    x y : Complex
    ⊢ Iff (SameRay Real x y) (Or (Eq x 0) (Or (Eq y 0) (Eq x.arg y.arg)))
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case inl
      y : Complex
      ⊢ Iff (SameRay Real 0 y) (Or (Eq 0 0) (Or (Eq y 0) (Eq (Complex.arg 0) y.arg)))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    x y : Complex
    hx : Ne x 0
    ⊢ Iff (SameRay Real x y) (Or (Eq x 0) (Or (Eq y 0) (Eq x.arg y.arg)))
  -/
  rcases eq_or_ne y 0 with (rfl | hy)
    /-
      case inr.inl
      x : Complex
      hx : Ne x 0
      ⊢ Iff (SameRay Real x 0) (Or (Eq x 0) (Or (Eq 0 0) (Eq x.arg (Complex.arg 0))))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (SameRay Real x y) (Or (Eq x 0) (Or (Eq y 0) (Eq x.arg y.arg)))
  -/
  simp only [hx, hy, sameRay_iff_norm_smul_eq, arg_eq_arg_iff hx hy]
  /-
    case inr.inr
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (HSMul.hSMul (Norm.norm x) y) (HSMul.hSMul (Norm.norm y) x)) (Or Fal …
  -/
  field_simp [hx, hy]
  /-
    case inr.inr
    x y : Complex
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (HMul.hMul (↑(Complex.abs x)) y) (HMul.hMul (↑(Complex.abs y)) x)) ( …
  -/
  rw [mul_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem sameRay_iff_arg_div_eq_zero : SameRay ℝ x y ↔ arg (x / y) = 0 := by
  /-
    x y : Complex
    ⊢ Iff (SameRay Real x y) (Eq (HDiv.hDiv x y).arg 0)
  -/
  rw [← Real.Angle.toReal_zero, ← arg_coe_angle_eq_iff_eq_toReal, sameRay_iff]
  /-
    x y : Complex
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Eq x.arg y.arg))) (Eq (↑(HDiv.hDiv x y).arg) 0)
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    x y : Complex
    hx : Not (Eq x 0)
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Eq x.arg y.arg))) (Eq (↑(HDiv.hDiv x y).arg) 0)
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    x y : Complex
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Eq x.arg y.arg))) (Eq (↑(HDiv.hDiv x y).arg) 0)
  -/
  simp [hx, hy, arg_div_coe_angle, sub_eq_zero]
  /-
    🎉 no goals
  -/

-- Porting note: `(x + y).abs` stopped working.

theorem abs_add_eq_iff : abs (x + y) = abs x + abs y ↔ x = 0 ∨ y = 0 ∨ x.arg = y.arg :=
  sameRay_iff_norm_add.symm.trans sameRay_iff


theorem abs_sub_eq_iff : abs (x - y) = |abs x - abs y| ↔ x = 0 ∨ y = 0 ∨ x.arg = y.arg :=
  sameRay_iff_norm_sub.symm.trans sameRay_iff


theorem sameRay_of_arg_eq (h : x.arg = y.arg) : SameRay ℝ x y :=
  sameRay_iff.mpr <| Or.inr <| Or.inr h


theorem abs_add_eq (h : x.arg = y.arg) : abs (x + y) = abs x + abs y :=
  (sameRay_of_arg_eq h).norm_add


theorem abs_sub_eq (h : x.arg = y.arg) : abs (x - y) = ‖abs x - abs y‖ :=
  (sameRay_of_arg_eq h).norm_sub


