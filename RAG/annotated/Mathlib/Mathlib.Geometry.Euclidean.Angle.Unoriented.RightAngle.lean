/-- Pythagorean theorem, if-and-only-if vector angle form. -/
theorem norm_add_sq_eq_norm_sq_add_norm_sq_iff_angle_eq_pi_div_two (x y : V) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ ↔ angle x y = π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y)))  …
  -/
  rw [norm_add_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Inner.inner x y) 0) (Eq (InnerProductGeometry.angle x y) (HDiv.hDiv …
  -/
  exact inner_eq_zero_iff_angle_eq_pi_div_two x y
  /-
    🎉 no goals
  -/


/-- Pythagorean theorem, vector angle form. -/
theorem norm_add_sq_eq_norm_sq_add_norm_sq' (x y : V) (h : angle x y = π / 2) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ :=
  (norm_add_sq_eq_norm_sq_add_norm_sq_iff_angle_eq_pi_div_two x y).2 h


/-- Pythagorean theorem, subtracting vectors, if-and-only-if vector angle form. -/
theorem norm_sub_sq_eq_norm_sq_add_norm_sq_iff_angle_eq_pi_div_two (x y : V) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ ↔ angle x y = π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub x y)))  …
  -/
  rw [norm_sub_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Inner.inner x y) 0) (Eq (InnerProductGeometry.angle x y) (HDiv.hDiv …
  -/
  exact inner_eq_zero_iff_angle_eq_pi_div_two x y
  /-
    🎉 no goals
  -/


/-- Pythagorean theorem, subtracting vectors, vector angle form. -/
theorem norm_sub_sq_eq_norm_sq_add_norm_sq' (x y : V) (h : angle x y = π / 2) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ :=
  (norm_sub_sq_eq_norm_sq_add_norm_sq_iff_angle_eq_pi_div_two x y).2 h


/-- An angle in a right-angled triangle expressed using `arccos`. -/
theorem angle_add_eq_arccos_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    angle x (x + y) = Real.arccos (‖x‖ / ‖x + y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (InnerProductGeometry.angle x (HAdd.hAdd x y)) (Real.arccos (HDiv.hDiv (N …
  -/
  rw [angle, inner_add_right, h, add_zero, real_inner_self_eq_norm_mul_norm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (Real.arccos (HDiv.hDiv (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMu …
  -/
  by_cases hx : ‖x‖ = 0; · simp [hx]
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    hx : Not (Eq (Norm.norm x) 0)
    ⊢ Eq (Real.arccos (HDiv.hDiv (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMu …
  -/
  rw [div_mul_eq_div_div, mul_self_div_self]
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arcsin`. -/
theorem angle_add_eq_arcsin_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y ≠ 0) :
    angle x (x + y) = Real.arcsin (‖y‖ / ‖x + y‖) := by
  have hxy : ‖x + y‖ ^ 2 ≠ 0 := by
    rw [pow_two, norm_add_sq_eq_norm_sq_add_norm_sq_real h, ne_comm]
    refine ne_of_lt ?_
    rcases h0 with (h0 | h0)
    · exact
        Left.add_pos_of_pos_of_nonneg (mul_self_pos.2 (norm_ne_zero_iff.2 h0)) (mul_self_nonneg _)
    · exact
        Left.add_pos_of_nonneg_of_pos (mul_self_nonneg _) (mul_self_pos.2 (norm_ne_zero_iff.2 h0))
  rw [angle_add_eq_arccos_of_inner_eq_zero h,
    Real.arccos_eq_arcsin (div_nonneg (norm_nonneg _) (norm_nonneg _)), div_pow, one_sub_div hxy]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Ne y 0)
    hxy : Ne (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) 0
    ⊢ Eq (Real.arcsin (HDiv.hDiv (HSub.hSub (HPow.hPow (Norm.norm (HAdd.hAdd x y)) …
  -/
  nth_rw 1 [pow_two]
  rw [norm_add_sq_eq_norm_sq_add_norm_sq_real h, pow_two, add_sub_cancel_left, ← pow_two, ← div_pow,
    Real.sqrt_sq (div_nonneg (norm_nonneg _) (norm_nonneg _))]


/-- An angle in a right-angled triangle expressed using `arctan`. -/
theorem angle_add_eq_arctan_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0) :
    angle x (x + y) = Real.arctan (‖y‖ / ‖x‖) := by
  rw [angle_add_eq_arcsin_of_inner_eq_zero h (Or.inl h0), Real.arctan_eq_arcsin, ←
    div_mul_eq_div_div, norm_add_eq_sqrt_iff_real_inner_eq_zero.2 h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Ne x 0
    ⊢ Eq (Real.arcsin (HDiv.hDiv (Norm.norm y) (HAdd.hAdd (HMul.hMul (Norm.norm x) …
  -/
  nth_rw 3 [← Real.sqrt_sq (norm_nonneg x)]
  rw_mod_cast [← Real.sqrt_mul (sq_nonneg _), div_pow, pow_two, pow_two, mul_add, mul_one, mul_div,
    mul_comm (‖x‖ * ‖x‖), ← mul_div, div_self (mul_self_pos.2 (norm_ne_zero_iff.2 h0)).ne', mul_one]


/-- An angle in a non-degenerate right-angled triangle is positive. -/
theorem angle_add_pos_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x = 0 ∨ y ≠ 0) :
    0 < angle x (x + y) := by
  rw [angle_add_eq_arccos_of_inner_eq_zero h, Real.arccos_pos,
    norm_add_eq_sqrt_iff_real_inner_eq_zero.2 h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ LT.lt (HDiv.hDiv (Norm.norm x) (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.nor …
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  rw [div_lt_one (Real.sqrt_pos.2 (Left.add_pos_of_pos_of_nonneg (mul_self_pos.2
    (norm_ne_zero_iff.2 hx)) (mul_self_nonneg _))), Real.lt_sqrt (norm_nonneg _), pow_two]
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    hx : Not (Eq x 0)
    ⊢ LT.lt (HMul.hMul (Norm.norm x) (Norm.norm x)) (HAdd.hAdd (HMul.hMul (Norm.no …
  -/
  simpa [hx] using h0
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle is at most `π / 2`. -/
theorem angle_add_le_pi_div_two_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    angle x (x + y) ≤ π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ LE.le (InnerProductGeometry.angle x (HAdd.hAdd x y)) (HDiv.hDiv Real.pi 2)
  -/
  rw [angle_add_eq_arccos_of_inner_eq_zero h, Real.arccos_le_pi_div_two]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ LE.le 0 (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x y)))
  -/
  exact div_nonneg (norm_nonneg _) (norm_nonneg _)
  /-
    🎉 no goals
  -/


/-- An angle in a non-degenerate right-angled triangle is less than `π / 2`. -/
theorem angle_add_lt_pi_div_two_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0) :
    angle x (x + y) < π / 2 := by
  rw [angle_add_eq_arccos_of_inner_eq_zero h, Real.arccos_lt_pi_div_two,
    norm_add_eq_sqrt_iff_real_inner_eq_zero.2 h]
  exact div_pos (norm_pos_iff.2 h0) (Real.sqrt_pos.2 (Left.add_pos_of_pos_of_nonneg
    (mul_self_pos.2 (norm_ne_zero_iff.2 h0)) (mul_self_nonneg _)))


/-- The cosine of an angle in a right-angled triangle as a ratio of sides. -/
theorem cos_angle_add_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.cos (angle x (x + y)) = ‖x‖ / ‖x + y‖ := by
  rw [angle_add_eq_arccos_of_inner_eq_zero h,
    Real.cos_arccos (le_trans (by norm_num) (div_nonneg (norm_nonneg _) (norm_nonneg _)))
      (div_le_one_of_le₀ _ (norm_nonneg _))]
  rw [mul_self_le_mul_self_iff (norm_nonneg _) (norm_nonneg _),
    norm_add_sq_eq_norm_sq_add_norm_sq_real h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ LE.le (HMul.hMul (Norm.norm x) (Norm.norm x)) (HAdd.hAdd (HMul.hMul (Norm.no …
  -/
  exact le_add_of_nonneg_right (mul_self_nonneg _)
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle as a ratio of sides. -/
theorem sin_angle_add_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y ≠ 0) :
    Real.sin (angle x (x + y)) = ‖y‖ / ‖x + y‖ := by
  rw [angle_add_eq_arcsin_of_inner_eq_zero h h0,
    Real.sin_arcsin (le_trans (by norm_num) (div_nonneg (norm_nonneg _) (norm_nonneg _)))
      (div_le_one_of_le₀ _ (norm_nonneg _))]
  rw [mul_self_le_mul_self_iff (norm_nonneg _) (norm_nonneg _),
    norm_add_sq_eq_norm_sq_add_norm_sq_real h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Ne y 0)
    ⊢ LE.le (HMul.hMul (Norm.norm y) (Norm.norm y)) (HAdd.hAdd (HMul.hMul (Norm.no …
  -/
  exact le_add_of_nonneg_left (mul_self_nonneg _)
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle as a ratio of sides. -/
theorem tan_angle_add_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.tan (angle x (x + y)) = ‖y‖ / ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (Real.tan (InnerProductGeometry.angle x (HAdd.hAdd x y))) (HDiv.hDiv (Nor …
  -/
  by_cases h0 : x = 0; · simp [h0]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Not (Eq x 0)
    ⊢ Eq (Real.tan (InnerProductGeometry.angle x (HAdd.hAdd x y))) (HDiv.hDiv (Nor …
  -/
  rw [angle_add_eq_arctan_of_inner_eq_zero h h0, Real.tan_arctan]
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side. -/
theorem cos_angle_add_mul_norm_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.cos (angle x (x + y)) * ‖x + y‖ = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (HMul.hMul (Real.cos (InnerProductGeometry.angle x (HAdd.hAdd x y))) (Nor …
  -/
  rw [cos_angle_add_of_inner_eq_zero h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x y))) (Norm.no …
  -/
  by_cases hxy : ‖x + y‖ = 0
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      hxy : Eq (Norm.norm (HAdd.hAdd x y)) 0
      ⊢ Eq (HMul.hMul (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x y))) (Norm.no …
    -/
  · have h' := norm_add_sq_eq_norm_sq_add_norm_sq_real h
    rw [hxy, zero_mul, eq_comm,
      add_eq_zero_iff_of_nonneg (mul_self_nonneg ‖x‖) (mul_self_nonneg ‖y‖), mul_self_eq_zero] at h'
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      hxy : Eq (Norm.norm (HAdd.hAdd x y)) 0
      h' : And (Eq (Norm.norm x) 0) (Eq (HMul.hMul (Norm.norm y) (Norm.norm y)) 0)
      ⊢ Eq (HMul.hMul (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x y))) (Norm.no …
    -/
    simp [h'.1]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      hxy : Not (Eq (Norm.norm (HAdd.hAdd x y)) 0)
      ⊢ Eq (HMul.hMul (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x y))) (Norm.no …
    -/
  · exact div_mul_cancel₀ _ hxy
    /-
      🎉 no goals
    -/


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side. -/
theorem sin_angle_add_mul_norm_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.sin (angle x (x + y)) * ‖x + y‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HAdd.hAdd x y))) (Nor …
  -/
  by_cases h0 : x = 0 ∧ y = 0; · simp [h0]
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Not (And (Eq x 0) (Eq y 0))
    ⊢ Eq (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HAdd.hAdd x y))) (Nor …
  -/
  rw [not_and_or] at h0
  /-
    case neg
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Not (Eq x 0)) (Not (Eq y 0))
    ⊢ Eq (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HAdd.hAdd x y))) (Nor …
  -/
  rw [sin_angle_add_of_inner_eq_zero h h0, div_mul_cancel₀]
  /-
    case neg.h
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Not (Eq x 0)) (Not (Eq y 0))
    ⊢ Ne (Norm.norm (HAdd.hAdd x y)) 0
  -/
  rw [← mul_self_ne_zero, norm_add_sq_eq_norm_sq_add_norm_sq_real h]
  /-
    case neg.h
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Not (Eq x 0)) (Not (Eq y 0))
    ⊢ Ne (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMul (Norm.norm  …
  -/
  refine (ne_of_lt ?_).symm
  /-
    case neg.h
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Not (Eq x 0)) (Not (Eq y 0))
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMul (Norm. …
  -/
  rcases h0 with (h0 | h0)
    /-
      case neg.h.inl
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      h0 : Not (Eq x 0)
      ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMul (Norm. …
    -/
  · exact Left.add_pos_of_pos_of_nonneg (mul_self_pos.2 (norm_ne_zero_iff.2 h0)) (mul_self_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case neg.h.inr
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      h0 : Not (Eq y 0)
      ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMul (Norm. …
    -/
  · exact Left.add_pos_of_nonneg_of_pos (mul_self_nonneg _) (mul_self_pos.2 (norm_ne_zero_iff.2 h0))
    /-
      🎉 no goals
    -/


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side. -/
theorem tan_angle_add_mul_norm_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y = 0) :
    Real.tan (angle x (x + y)) * ‖x‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HMul.hMul (Real.tan (InnerProductGeometry.angle x (HAdd.hAdd x y))) (Nor …
  -/
  rw [tan_angle_add_of_inner_eq_zero h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Norm.norm y) (Norm.norm x)) (Norm.norm x)) (Norm.n …
  -/
                               /-
                                 🎉 no goals
                               -/
  rcases h0 with (h0 | h0) <;> simp [h0]
                               /-
                                 🎉 no goals
                               -/


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse. -/
theorem norm_div_cos_angle_add_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y = 0) :
    ‖x‖ / Real.cos (angle x (x + y)) = ‖x + y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (Real.cos (InnerProductGeometry.angle x (HAdd.hA …
  -/
  rw [cos_angle_add_of_inner_eq_zero h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x …
  -/
  rcases h0 with (h0 | h0)
    /-
      case inl
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      h0 : Ne x 0
      ⊢ Eq (HDiv.hDiv (Norm.norm x) (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x …
    -/
  · rw [div_div_eq_mul_div, mul_comm, div_eq_mul_inv, mul_inv_cancel_right₀ (norm_ne_zero_iff.2 h0)]
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      h0 : Eq y 0
      ⊢ Eq (HDiv.hDiv (Norm.norm x) (HDiv.hDiv (Norm.norm x) (Norm.norm (HAdd.hAdd x …
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse. -/
theorem norm_div_sin_angle_add_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x = 0 ∨ y ≠ 0) :
    ‖y‖ / Real.sin (angle x (x + y)) = ‖x + y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.sin (InnerProductGeometry.angle x (HAdd.hA …
  -/
  rcases h0 with (h0 | h0); · simp [h0]
                              /-
                                🎉 no goals
                              -/
  rw [sin_angle_add_of_inner_eq_zero h (Or.inr h0), div_div_eq_mul_div, mul_comm, div_eq_mul_inv,
    mul_inv_cancel_right₀ (norm_ne_zero_iff.2 h0)]


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side. -/
theorem norm_div_tan_angle_add_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x = 0 ∨ y ≠ 0) :
    ‖y‖ / Real.tan (angle x (x + y)) = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.tan (InnerProductGeometry.angle x (HAdd.hA …
  -/
  rw [tan_angle_add_of_inner_eq_zero h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (HDiv.hDiv (Norm.norm y) (Norm.norm x))) (Norm.n …
  -/
  rcases h0 with (h0 | h0)
    /-
      case inl
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      h0 : Eq x 0
      ⊢ Eq (HDiv.hDiv (Norm.norm y) (HDiv.hDiv (Norm.norm y) (Norm.norm x))) (Norm.n …
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (Inner.inner x y) 0
      h0 : Ne y 0
      ⊢ Eq (HDiv.hDiv (Norm.norm y) (HDiv.hDiv (Norm.norm y) (Norm.norm x))) (Norm.n …
    -/
  · rw [div_div_eq_mul_div, mul_comm, div_eq_mul_inv, mul_inv_cancel_right₀ (norm_ne_zero_iff.2 h0)]
    /-
      🎉 no goals
    -/


/-- An angle in a right-angled triangle expressed using `arccos`, version subtracting vectors. -/
theorem angle_sub_eq_arccos_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    angle x (x - y) = Real.arccos (‖x‖ / ‖x - y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arccos (HDiv.hDiv (N …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arccos (HDiv.hDiv (N …
  -/
  rw [sub_eq_add_neg, angle_add_eq_arccos_of_inner_eq_zero h]
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arcsin`, version subtracting vectors. -/
theorem angle_sub_eq_arcsin_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y ≠ 0) :
    angle x (x - y) = Real.arcsin (‖y‖ / ‖x - y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Ne y 0)
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arcsin (HDiv.hDiv (N …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Ne y 0)
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arcsin (HDiv.hDiv (N …
  -/
  rw [or_comm, ← neg_ne_zero, or_comm] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Ne (Neg.neg y) 0)
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arcsin (HDiv.hDiv (N …
  -/
  rw [sub_eq_add_neg, angle_add_eq_arcsin_of_inner_eq_zero h h0, norm_neg]
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle expressed using `arctan`, version subtracting vectors. -/
theorem angle_sub_eq_arctan_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0) :
    angle x (x - y) = Real.arctan (‖y‖ / ‖x‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Ne x 0
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arctan (HDiv.hDiv (N …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Ne x 0
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (Real.arctan (HDiv.hDiv (N …
  -/
  rw [sub_eq_add_neg, angle_add_eq_arctan_of_inner_eq_zero h h0, norm_neg]
  /-
    🎉 no goals
  -/


/-- An angle in a non-degenerate right-angled triangle is positive, version subtracting
vectors. -/
theorem angle_sub_pos_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x = 0 ∨ y ≠ 0) :
    0 < angle x (x - y) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ LT.lt 0 (InnerProductGeometry.angle x (HSub.hSub x y))
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ LT.lt 0 (InnerProductGeometry.angle x (HSub.hSub x y))
  -/
  rw [← neg_ne_zero] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne (Neg.neg y) 0)
    ⊢ LT.lt 0 (InnerProductGeometry.angle x (HSub.hSub x y))
  -/
  rw [sub_eq_add_neg]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne (Neg.neg y) 0)
    ⊢ LT.lt 0 (InnerProductGeometry.angle x (HAdd.hAdd x (Neg.neg y)))
  -/
  exact angle_add_pos_of_inner_eq_zero h h0
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle is at most `π / 2`, version subtracting vectors. -/
theorem angle_sub_le_pi_div_two_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    angle x (x - y) ≤ π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ LE.le (InnerProductGeometry.angle x (HSub.hSub x y)) (HDiv.hDiv Real.pi 2)
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ LE.le (InnerProductGeometry.angle x (HSub.hSub x y)) (HDiv.hDiv Real.pi 2)
  -/
  rw [sub_eq_add_neg]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ LE.le (InnerProductGeometry.angle x (HAdd.hAdd x (Neg.neg y))) (HDiv.hDiv Re …
  -/
  exact angle_add_le_pi_div_two_of_inner_eq_zero h
  /-
    🎉 no goals
  -/


/-- An angle in a non-degenerate right-angled triangle is less than `π / 2`, version subtracting
vectors. -/
theorem angle_sub_lt_pi_div_two_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0) :
    angle x (x - y) < π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Ne x 0
    ⊢ LT.lt (InnerProductGeometry.angle x (HSub.hSub x y)) (HDiv.hDiv Real.pi 2)
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Ne x 0
    ⊢ LT.lt (InnerProductGeometry.angle x (HSub.hSub x y)) (HDiv.hDiv Real.pi 2)
  -/
  rw [sub_eq_add_neg]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Ne x 0
    ⊢ LT.lt (InnerProductGeometry.angle x (HAdd.hAdd x (Neg.neg y))) (HDiv.hDiv Re …
  -/
  exact angle_add_lt_pi_div_two_of_inner_eq_zero h h0
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem cos_angle_sub_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.cos (angle x (x - y)) = ‖x‖ / ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (Real.cos (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ Eq (Real.cos (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [sub_eq_add_neg, cos_angle_add_of_inner_eq_zero h]
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem sin_angle_sub_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y ≠ 0) :
    Real.sin (angle x (x - y)) = ‖y‖ / ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Ne y 0)
    ⊢ Eq (Real.sin (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Ne y 0)
    ⊢ Eq (Real.sin (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [or_comm, ← neg_ne_zero, or_comm] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Ne (Neg.neg y) 0)
    ⊢ Eq (Real.sin (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [sub_eq_add_neg, sin_angle_add_of_inner_eq_zero h h0, norm_neg]
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle as a ratio of sides, version subtracting
vectors. -/
theorem tan_angle_sub_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.tan (angle x (x - y)) = ‖y‖ / ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (Real.tan (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ Eq (Real.tan (InnerProductGeometry.angle x (HSub.hSub x y))) (HDiv.hDiv (Nor …
  -/
  rw [sub_eq_add_neg, tan_angle_add_of_inner_eq_zero h, norm_neg]
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side, version subtracting vectors. -/
theorem cos_angle_sub_mul_norm_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.cos (angle x (x - y)) * ‖x - y‖ = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (HMul.hMul (Real.cos (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ Eq (HMul.hMul (Real.cos (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [sub_eq_add_neg, cos_angle_add_mul_norm_of_inner_eq_zero h]
  /-
    🎉 no goals
  -/


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side, version subtracting vectors. -/
theorem sin_angle_sub_mul_norm_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) :
    Real.sin (angle x (x - y)) * ‖x - y‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    ⊢ Eq (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [sub_eq_add_neg, sin_angle_add_mul_norm_of_inner_eq_zero h, norm_neg]
  /-
    🎉 no goals
  -/


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side, version subtracting vectors. -/
theorem tan_angle_sub_mul_norm_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y = 0) :
    Real.tan (angle x (x - y)) * ‖x‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HMul.hMul (Real.tan (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HMul.hMul (Real.tan (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [← neg_eq_zero] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Eq (Neg.neg y) 0)
    ⊢ Eq (HMul.hMul (Real.tan (InnerProductGeometry.angle x (HSub.hSub x y))) (Nor …
  -/
  rw [sub_eq_add_neg, tan_angle_add_mul_norm_of_inner_eq_zero h h0, norm_neg]
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse, version subtracting vectors. -/
theorem norm_div_cos_angle_sub_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x ≠ 0 ∨ y = 0) :
    ‖x‖ / Real.cos (angle x (x - y)) = ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (Real.cos (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Eq y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (Real.cos (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [← neg_eq_zero] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Ne x 0) (Eq (Neg.neg y) 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm x) (Real.cos (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [sub_eq_add_neg, norm_div_cos_angle_add_of_inner_eq_zero h h0]
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse, version subtracting vectors. -/
theorem norm_div_sin_angle_sub_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x = 0 ∨ y ≠ 0) :
    ‖y‖ / Real.sin (angle x (x - y)) = ‖x - y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.sin (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.sin (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [← neg_ne_zero] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne (Neg.neg y) 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.sin (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [sub_eq_add_neg, ← norm_neg, norm_div_sin_angle_add_of_inner_eq_zero h h0]
  /-
    🎉 no goals
  -/


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side, version subtracting vectors. -/
theorem norm_div_tan_angle_sub_of_inner_eq_zero {x y : V} (h : ⟪x, y⟫ = 0) (h0 : x = 0 ∨ y ≠ 0) :
    ‖y‖ / Real.tan (angle x (x - y)) = ‖x‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x y) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.tan (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [← neg_eq_zero, ← inner_neg_right] at h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne y 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.tan (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [← neg_ne_zero] at h0
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Inner.inner x (Neg.neg y)) 0
    h0 : Or (Eq x 0) (Ne (Neg.neg y) 0)
    ⊢ Eq (HDiv.hDiv (Norm.norm y) (Real.tan (InnerProductGeometry.angle x (HSub.hS …
  -/
  rw [sub_eq_add_neg, ← norm_neg, norm_div_tan_angle_add_of_inner_eq_zero h h0]
  /-
    🎉 no goals
  -/


/-- **Pythagorean theorem**, if-and-only-if angle-at-point form. -/
theorem dist_sq_eq_dist_sq_add_dist_sq_iff_angle_eq_pi_div_two (p1 p2 p3 : P) :
    dist p1 p3 * dist p1 p3 = dist p1 p2 * dist p1 p2 + dist p3 p2 * dist p3 p2 ↔
      ∠ p1 p2 p3 = π / 2 := by
  erw [dist_comm p3 p2, dist_eq_norm_vsub V p1 p3, dist_eq_norm_vsub V p1 p2,
    dist_eq_norm_vsub V p2 p3, ← norm_sub_sq_eq_norm_sq_add_norm_sq_iff_angle_eq_pi_div_two,
    vsub_sub_vsub_cancel_right p1, ← neg_vsub_eq_vsub_rev p2 p3, norm_neg]


/-- An angle in a right-angled triangle expressed using `arccos`. -/
theorem angle_eq_arccos_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2) :
    ∠ p₂ p₃ p₁ = Real.arccos (dist p₃ p₂ / dist p₁ p₃) := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  rw [angle, dist_eq_norm_vsub' V p₃ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, angle_add_eq_arccos_of_inner_eq_zero h]


/-- An angle in a right-angled triangle expressed using `arcsin`. -/
theorem angle_eq_arcsin_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ ≠ p₂ ∨ p₃ ≠ p₂) : ∠ p₂ p₃ p₁ = Real.arcsin (dist p₁ p₂ / dist p₁ p₃) := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Ne p₁ p₂) (Ne p₃ p₂)
    ⊢ Eq (EuclideanGeometry.angle p₂ p₃ p₁) (Real.arcsin (HDiv.hDiv (Dist.dist p₁  …
  -/
  rw [← @vsub_ne_zero V, @ne_comm _ p₃, ← @vsub_ne_zero V _ _ _ p₂, or_comm] at h0
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, angle_add_eq_arcsin_of_inner_eq_zero h h0]


/-- An angle in a right-angled triangle expressed using `arctan`. -/
theorem angle_eq_arctan_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₃ ≠ p₂) : ∠ p₂ p₃ p₁ = Real.arctan (dist p₁ p₂ / dist p₃ p₂) := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Ne p₃ p₂
    ⊢ Eq (EuclideanGeometry.angle p₂ p₃ p₁) (Real.arctan (HDiv.hDiv (Dist.dist p₁  …
  -/
  rw [ne_comm, ← @vsub_ne_zero V] at h0
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub' V p₃ p₂, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, angle_add_eq_arctan_of_inner_eq_zero h h0]


/-- An angle in a non-degenerate right-angled triangle is positive. -/
theorem angle_pos_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ ≠ p₂ ∨ p₃ = p₂) : 0 < ∠ p₂ p₃ p₁ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Ne p₁ p₂) (Eq p₃ p₂)
    ⊢ LT.lt 0 (EuclideanGeometry.angle p₂ p₃ p₁)
  -/
  rw [← @vsub_ne_zero V, eq_comm, ← @vsub_eq_zero_iff_eq V, or_comm] at h0
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Eq (VSub.vsub p₂ p₃) 0) (Ne (VSub.vsub p₁ p₂) 0)
    ⊢ LT.lt 0 (EuclideanGeometry.angle p₂ p₃ p₁)
  -/
  rw [angle, ← vsub_add_vsub_cancel p₁ p₂ p₃, add_comm]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Eq (VSub.vsub p₂ p₃) 0) (Ne (VSub.vsub p₁ p₂) 0)
    ⊢ LT.lt 0 (InnerProductGeometry.angle (VSub.vsub p₂ p₃) (HAdd.hAdd (VSub.vsub  …
  -/
  exact angle_add_pos_of_inner_eq_zero h h0
  /-
    🎉 no goals
  -/


/-- An angle in a right-angled triangle is at most `π / 2`. -/
theorem angle_le_pi_div_two_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2) :
    ∠ p₂ p₃ p₁ ≤ π / 2 := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    ⊢ LE.le (EuclideanGeometry.angle p₂ p₃ p₁) (HDiv.hDiv Real.pi 2)
  -/
  rw [angle, ← vsub_add_vsub_cancel p₁ p₂ p₃, add_comm]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    ⊢ LE.le (InnerProductGeometry.angle (VSub.vsub p₂ p₃) (HAdd.hAdd (VSub.vsub p₂ …
  -/
  exact angle_add_le_pi_div_two_of_inner_eq_zero h
  /-
    🎉 no goals
  -/


/-- An angle in a non-degenerate right-angled triangle is less than `π / 2`. -/
theorem angle_lt_pi_div_two_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₃ ≠ p₂) : ∠ p₂ p₃ p₁ < π / 2 := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Ne p₃ p₂
    ⊢ LT.lt (EuclideanGeometry.angle p₂ p₃ p₁) (HDiv.hDiv Real.pi 2)
  -/
  rw [ne_comm, ← @vsub_ne_zero V] at h0
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Ne (VSub.vsub p₂ p₃) 0
    ⊢ LT.lt (EuclideanGeometry.angle p₂ p₃ p₁) (HDiv.hDiv Real.pi 2)
  -/
  rw [angle, ← vsub_add_vsub_cancel p₁ p₂ p₃, add_comm]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Ne (VSub.vsub p₂ p₃) 0
    ⊢ LT.lt (InnerProductGeometry.angle (VSub.vsub p₂ p₃) (HAdd.hAdd (VSub.vsub p₂ …
  -/
  exact angle_add_lt_pi_div_two_of_inner_eq_zero h h0
  /-
    🎉 no goals
  -/


/-- The cosine of an angle in a right-angled triangle as a ratio of sides. -/
theorem cos_angle_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2) :
    Real.cos (∠ p₂ p₃ p₁) = dist p₃ p₂ / dist p₁ p₃ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  rw [angle, dist_eq_norm_vsub' V p₃ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, cos_angle_add_of_inner_eq_zero h]


/-- The sine of an angle in a right-angled triangle as a ratio of sides. -/
theorem sin_angle_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ ≠ p₂ ∨ p₃ ≠ p₂) : Real.sin (∠ p₂ p₃ p₁) = dist p₁ p₂ / dist p₁ p₃ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Ne p₁ p₂) (Ne p₃ p₂)
    ⊢ Eq (Real.sin (EuclideanGeometry.angle p₂ p₃ p₁)) (HDiv.hDiv (Dist.dist p₁ p₂ …
  -/
  rw [← @vsub_ne_zero V, @ne_comm _ p₃, ← @vsub_ne_zero V _ _ _ p₂, or_comm] at h0
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, sin_angle_add_of_inner_eq_zero h h0]


/-- The tangent of an angle in a right-angled triangle as a ratio of sides. -/
theorem tan_angle_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2) :
    Real.tan (∠ p₂ p₃ p₁) = dist p₁ p₂ / dist p₃ p₂ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub' V p₃ p₂, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, tan_angle_add_of_inner_eq_zero h]


/-- The cosine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
adjacent side. -/
theorem cos_angle_mul_dist_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2) :
    Real.cos (∠ p₂ p₃ p₁) * dist p₁ p₃ = dist p₃ p₂ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  rw [angle, dist_eq_norm_vsub' V p₃ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, cos_angle_add_mul_norm_of_inner_eq_zero h]


/-- The sine of an angle in a right-angled triangle multiplied by the hypotenuse equals the
opposite side. -/
theorem sin_angle_mul_dist_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2) :
    Real.sin (∠ p₂ p₃ p₁) * dist p₁ p₃ = dist p₁ p₂ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, sin_angle_add_mul_norm_of_inner_eq_zero h]


/-- The tangent of an angle in a right-angled triangle multiplied by the adjacent side equals
the opposite side. -/
theorem tan_angle_mul_dist_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ = p₂ ∨ p₃ ≠ p₂) : Real.tan (∠ p₂ p₃ p₁) * dist p₃ p₂ = dist p₁ p₂ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Eq p₁ p₂) (Ne p₃ p₂)
    ⊢ Eq (HMul.hMul (Real.tan (EuclideanGeometry.angle p₂ p₃ p₁)) (Dist.dist p₃ p₂ …
  -/
  rw [ne_comm, ← @vsub_ne_zero V, ← @vsub_eq_zero_iff_eq V, or_comm] at h0
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub' V p₃ p₂, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, tan_angle_add_mul_norm_of_inner_eq_zero h h0]


/-- A side of a right-angled triangle divided by the cosine of the adjacent angle equals the
hypotenuse. -/
theorem dist_div_cos_angle_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ = p₂ ∨ p₃ ≠ p₂) : dist p₃ p₂ / Real.cos (∠ p₂ p₃ p₁) = dist p₁ p₃ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Eq p₁ p₂) (Ne p₃ p₂)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₃ p₂) (Real.cos (EuclideanGeometry.angle p₂ p₃ p₁) …
  -/
  rw [ne_comm, ← @vsub_ne_zero V, ← @vsub_eq_zero_iff_eq V, or_comm] at h0
  rw [angle, dist_eq_norm_vsub' V p₃ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, norm_div_cos_angle_add_of_inner_eq_zero h h0]


/-- A side of a right-angled triangle divided by the sine of the opposite angle equals the
hypotenuse. -/
theorem dist_div_sin_angle_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ ≠ p₂ ∨ p₃ = p₂) : dist p₁ p₂ / Real.sin (∠ p₂ p₃ p₁) = dist p₁ p₃ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Ne p₁ p₂) (Eq p₃ p₂)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₁ p₂) (Real.sin (EuclideanGeometry.angle p₂ p₃ p₁) …
  -/
  rw [eq_comm, ← @vsub_ne_zero V, ← @vsub_eq_zero_iff_eq V, or_comm] at h0
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub V p₁ p₃, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, norm_div_sin_angle_add_of_inner_eq_zero h h0]


/-- A side of a right-angled triangle divided by the tangent of the opposite angle equals the
adjacent side. -/
theorem dist_div_tan_angle_of_angle_eq_pi_div_two {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π / 2)
    (h0 : p₁ ≠ p₂ ∨ p₃ = p₂) : dist p₁ p₂ / Real.tan (∠ p₂ p₃ p₁) = dist p₃ p₂ := by
  rw [angle, ← inner_eq_zero_iff_angle_eq_pi_div_two, real_inner_comm, ← neg_eq_zero, ←
    inner_neg_left, neg_vsub_eq_vsub_rev] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Inner.inner (VSub.vsub p₂ p₃) (VSub.vsub p₁ p₂)) 0
    h0 : Or (Ne p₁ p₂) (Eq p₃ p₂)
    ⊢ Eq (HDiv.hDiv (Dist.dist p₁ p₂) (Real.tan (EuclideanGeometry.angle p₂ p₃ p₁) …
  -/
  rw [eq_comm, ← @vsub_ne_zero V, ← @vsub_eq_zero_iff_eq V, or_comm] at h0
  rw [angle, dist_eq_norm_vsub V p₁ p₂, dist_eq_norm_vsub' V p₃ p₂, ← vsub_add_vsub_cancel p₁ p₂ p₃,
    add_comm, norm_div_tan_angle_add_of_inner_eq_zero h h0]


