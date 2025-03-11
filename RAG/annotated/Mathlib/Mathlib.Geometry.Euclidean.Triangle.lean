/-- **Law of cosines** (cosine rule), vector angle form. -/
theorem norm_sub_sq_eq_norm_sq_add_norm_sq_sub_two_mul_norm_mul_norm_mul_cos_angle (x y : V) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ - 2 * ‖x‖ * ‖y‖ * Real.cos (angle x y) := by
  rw [show 2 * ‖x‖ * ‖y‖ * Real.cos (angle x y) = 2 * (Real.cos (angle x y) * (‖x‖ * ‖y‖)) by ring,
    cos_angle_mul_norm_mul_norm, ← real_inner_self_eq_norm_mul_norm, ←
    real_inner_self_eq_norm_mul_norm, ← real_inner_self_eq_norm_mul_norm, real_inner_sub_sub_self,
    sub_add_eq_add_sub]


/-- **Pons asinorum**, vector angle form. -/
theorem angle_sub_eq_angle_sub_rev_of_norm_eq {x y : V} (h : ‖x‖ = ‖y‖) :
    angle x (x - y) = angle y (y - x) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Norm.norm x) (Norm.norm y)
    ⊢ Eq (InnerProductGeometry.angle x (HSub.hSub x y)) (InnerProductGeometry.angl …
  -/
  refine Real.injOn_cos ⟨angle_nonneg _ _, angle_le_pi _ _⟩ ⟨angle_nonneg _ _, angle_le_pi _ _⟩ ?_
  rw [cos_angle, cos_angle, h, ← neg_sub, norm_neg, neg_sub, inner_sub_right, inner_sub_right,
    real_inner_self_eq_norm_mul_norm, real_inner_self_eq_norm_mul_norm, h, real_inner_comm x y]


/-- **Converse of pons asinorum**, vector angle form. -/
theorem norm_eq_of_angle_sub_eq_angle_sub_rev_of_angle_ne_pi {x y : V}
    (h : angle x (x - y) = angle y (y - x)) (hpi : angle x y ≠ π) : ‖x‖ = ‖y‖ := by
  replace h := Real.arccos_injOn (abs_le.mp (abs_real_inner_div_norm_mul_norm_le_one x (x - y)))
    (abs_le.mp (abs_real_inner_div_norm_mul_norm_le_one y (y - x))) h
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hpi : Ne (InnerProductGeometry.angle x y) Real.pi
    h : Eq (HDiv.hDiv (Inner.inner x (HSub.hSub x y)) (HMul.hMul (Norm.norm x) (No …
    ⊢ Eq (Norm.norm x) (Norm.norm y)
  -/
  by_cases hxy : x = y
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hpi : Ne (InnerProductGeometry.angle x y) Real.pi
      h : Eq (HDiv.hDiv (Inner.inner x (HSub.hSub x y)) (HMul.hMul (Norm.norm x) (No …
      hxy : Eq x y
      ⊢ Eq (Norm.norm x) (Norm.norm y)
    -/
  · rw [hxy]
    /-
      🎉 no goals
    -/
  · rw [← norm_neg (y - x), neg_sub, mul_comm, mul_comm ‖y‖, div_eq_mul_inv, div_eq_mul_inv,
      mul_inv_rev, mul_inv_rev, ← mul_assoc, ← mul_assoc] at h
    replace h :=
      mul_right_cancel₀ (inv_ne_zero fun hz => hxy (eq_of_sub_eq_zero (norm_eq_zero.1 hz))) h
    rw [inner_sub_right, inner_sub_right, real_inner_comm x y, real_inner_self_eq_norm_mul_norm,
      real_inner_self_eq_norm_mul_norm, mul_sub_right_distrib, mul_sub_right_distrib,
      mul_self_mul_inv, mul_self_mul_inv, sub_eq_sub_iff_sub_eq_sub, ← mul_sub_left_distrib] at h
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hpi : Ne (InnerProductGeometry.angle x y) Real.pi
      hxy : Not (Eq x y)
      h : Eq (HSub.hSub (Norm.norm x) (Norm.norm y)) (HMul.hMul (Inner.inner x y) (H …
      ⊢ Eq (Norm.norm x) (Norm.norm y)
    -/
    by_cases hx0 : x = 0
      /-
        case pos
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        hpi : Ne (InnerProductGeometry.angle x y) Real.pi
        hxy : Not (Eq x y)
        h : Eq (HSub.hSub (Norm.norm x) (Norm.norm y)) (HMul.hMul (Inner.inner x y) (H …
        hx0 : Eq x 0
        ⊢ Eq (Norm.norm x) (Norm.norm y)
      -/
    · rw [hx0, norm_zero, inner_zero_left, zero_mul, zero_sub, neg_eq_zero] at h
      /-
        case pos
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        hpi : Ne (InnerProductGeometry.angle x y) Real.pi
        hxy : Not (Eq x y)
        h : Eq (Norm.norm y) 0
        hx0 : Eq x 0
        ⊢ Eq (Norm.norm x) (Norm.norm y)
      -/
      rw [hx0, norm_zero, h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        hpi : Ne (InnerProductGeometry.angle x y) Real.pi
        hxy : Not (Eq x y)
        h : Eq (HSub.hSub (Norm.norm x) (Norm.norm y)) (HMul.hMul (Inner.inner x y) (H …
        hx0 : Not (Eq x 0)
        ⊢ Eq (Norm.norm x) (Norm.norm y)
      -/
    · by_cases hy0 : y = 0
        /-
          case pos
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          h : Eq (HSub.hSub (Norm.norm x) (Norm.norm y)) (HMul.hMul (Inner.inner x y) (H …
          hx0 : Not (Eq x 0)
          hy0 : Eq y 0
          ⊢ Eq (Norm.norm x) (Norm.norm y)
        -/
      · rw [hy0, norm_zero, inner_zero_right, zero_mul, sub_zero] at h
        /-
          case pos
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          h : Eq (Norm.norm x) 0
          hx0 : Not (Eq x 0)
          hy0 : Eq y 0
          ⊢ Eq (Norm.norm x) (Norm.norm y)
        -/
        rw [hy0, norm_zero, h]
        /-
          🎉 no goals
        -/
      · rw [inv_sub_inv (fun hz => hx0 (norm_eq_zero.1 hz)) fun hz => hy0 (norm_eq_zero.1 hz), ←
          neg_sub, ← mul_div_assoc, mul_comm, mul_div_assoc, ← mul_neg_one] at h
        /-
          case neg
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          h : Eq (HMul.hMul (HSub.hSub (Norm.norm y) (Norm.norm x)) (-1)) (HMul.hMul (HS …
          hx0 : Not (Eq x 0)
          hy0 : Not (Eq y 0)
          ⊢ Eq (Norm.norm x) (Norm.norm y)
        -/
        symm
        /-
          case neg
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          h : Eq (HMul.hMul (HSub.hSub (Norm.norm y) (Norm.norm x)) (-1)) (HMul.hMul (HS …
          hx0 : Not (Eq x 0)
          hy0 : Not (Eq y 0)
          ⊢ Eq (Norm.norm y) (Norm.norm x)
        -/
        by_contra hyx
        /-
          case neg
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          h : Eq (HMul.hMul (HSub.hSub (Norm.norm y) (Norm.norm x)) (-1)) (HMul.hMul (HS …
          hx0 : Not (Eq x 0)
          hy0 : Not (Eq y 0)
          hyx : Not (Eq (Norm.norm y) (Norm.norm x))
          ⊢ False
        -/
        replace h := (mul_left_cancel₀ (sub_ne_zero_of_ne hyx) h).symm
        /-
          case neg
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          hx0 : Not (Eq x 0)
          hy0 : Not (Eq y 0)
          hyx : Not (Eq (Norm.norm y) (Norm.norm x))
          h : Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) ( …
          ⊢ False
        -/
        rw [real_inner_div_norm_mul_norm_eq_neg_one_iff, ← angle_eq_pi_iff] at h
        /-
          case neg
          V : Type u_1
          inst✝¹ : NormedAddCommGroup V
          inst✝ : InnerProductSpace Real V
          x y : V
          hpi : Ne (InnerProductGeometry.angle x y) Real.pi
          hxy : Not (Eq x y)
          hx0 : Not (Eq x 0)
          hy0 : Not (Eq y 0)
          hyx : Not (Eq (Norm.norm y) (Norm.norm x))
          h : Eq (InnerProductGeometry.angle x y) Real.pi
          ⊢ False
        -/
        exact hpi h
        /-
          🎉 no goals
        -/


/-- The cosine of the sum of two angles in a possibly degenerate
triangle (where two given sides are nonzero), vector angle form. -/
theorem cos_angle_sub_add_angle_sub_rev_eq_neg_cos_angle {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    Real.cos (angle x (x - y) + angle y (y - x)) = -Real.cos (angle x y) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (Real.cos (HAdd.hAdd (InnerProductGeometry.angle x (HSub.hSub x y)) (Inne …
  -/
  by_cases hxy : x = y
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Eq x y
      ⊢ Eq (Real.cos (HAdd.hAdd (InnerProductGeometry.angle x (HSub.hSub x y)) (Inne …
    -/
  · rw [hxy, angle_self hy]
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Eq x y
      ⊢ Eq (Real.cos (HAdd.hAdd (InnerProductGeometry.angle y (HSub.hSub y y)) (Inne …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      ⊢ Eq (Real.cos (HAdd.hAdd (InnerProductGeometry.angle x (HSub.hSub x y)) (Inne …
    -/
  · rw [Real.cos_add, cos_angle, cos_angle, cos_angle]
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner.inner x (HSub.hSub x y)) (HMul.hM …
    -/
    have hxn : ‖x‖ ≠ 0 := fun h => hx (norm_eq_zero.1 h)
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner.inner x (HSub.hSub x y)) (HMul.hM …
    -/
    have hyn : ‖y‖ ≠ 0 := fun h => hy (norm_eq_zero.1 h)
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner.inner x (HSub.hSub x y)) (HMul.hM …
    -/
    have hxyn : ‖x - y‖ ≠ 0 := fun h => hxy (eq_of_sub_eq_zero (norm_eq_zero.1 h))
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner.inner x (HSub.hSub x y)) (HMul.hM …
    -/
    apply mul_right_cancel₀ hxn
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner.inner x (HSub.hSub x y …
    -/
    apply mul_right_cancel₀ hyn
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner.inner x (HS …
    -/
    apply mul_right_cancel₀ hxyn
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul (HDiv.hDiv (Inner. …
    -/
    apply mul_right_cancel₀ hxyn
    have H1 :
      Real.sin (angle x (x - y)) * Real.sin (angle y (y - x)) * ‖x‖ * ‖y‖ * ‖x - y‖ * ‖x - y‖ =
        Real.sin (angle x (x - y)) * (‖x‖ * ‖x - y‖) *
          (Real.sin (angle y (y - x)) * (‖y‖ * ‖x - y‖)) := by
      ring
    have H2 :
      ⟪x, x⟫ * (⟪x, x⟫ - ⟪x, y⟫ - (⟪x, y⟫ - ⟪y, y⟫)) - (⟪x, x⟫ - ⟪x, y⟫) * (⟪x, x⟫ - ⟪x, y⟫) =
        ⟪x, x⟫ * ⟪y, y⟫ - ⟪x, y⟫ * ⟪x, y⟫ := by
      ring
    have H3 :
      ⟪y, y⟫ * (⟪y, y⟫ - ⟪x, y⟫ - (⟪x, y⟫ - ⟪x, x⟫)) - (⟪y, y⟫ - ⟪x, y⟫) * (⟪y, y⟫ - ⟪x, y⟫) =
        ⟪x, x⟫ * ⟪y, y⟫ - ⟪x, y⟫ * ⟪x, y⟫ := by
      ring
    rw [mul_sub_right_distrib, mul_sub_right_distrib, mul_sub_right_distrib, mul_sub_right_distrib,
      H1, sin_angle_mul_norm_mul_norm, norm_sub_rev x y, sin_angle_mul_norm_mul_norm,
      norm_sub_rev y x, inner_sub_left, inner_sub_left, inner_sub_right, inner_sub_right,
      inner_sub_right, inner_sub_right, real_inner_comm x y, H2, H3,
      Real.mul_self_sqrt (sub_nonneg_of_le (real_inner_mul_inner_self_le x y)),
      real_inner_self_eq_norm_mul_norm, real_inner_self_eq_norm_mul_norm,
      real_inner_eq_norm_mul_self_add_norm_mul_self_sub_norm_sub_mul_self_div_two]
    -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp [hxn, hyn, hxyn]`, but was really slow
    -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
    simp (disch := field_simp_discharge) only [sub_div', div_div, mul_div_assoc',
      div_mul_eq_mul_div, div_sub', neg_div', neg_sub, eq_div_iff, div_eq_iff]
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      H1 : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (Real.sin (Inne …
      H2 : Eq (HSub.hSub (HMul.hMul (Inner.inner x x) (HSub.hSub (HSub.hSub (Inner.i …
      H3 : Eq (HSub.hSub (HMul.hMul (Inner.inner y y) (HSub.hSub (HSub.hSub (Inner.i …
      ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.h …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The sine of the sum of two angles in a possibly degenerate
triangle (where two given sides are nonzero), vector angle form. -/
theorem sin_angle_sub_add_angle_sub_rev_eq_sin_angle {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    Real.sin (angle x (x - y) + angle y (y - x)) = Real.sin (angle x y) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (Real.sin (HAdd.hAdd (InnerProductGeometry.angle x (HSub.hSub x y)) (Inne …
  -/
  by_cases hxy : x = y
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Eq x y
      ⊢ Eq (Real.sin (HAdd.hAdd (InnerProductGeometry.angle x (HSub.hSub x y)) (Inne …
    -/
  · rw [hxy, angle_self hy]
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Eq x y
      ⊢ Eq (Real.sin (HAdd.hAdd (InnerProductGeometry.angle y (HSub.hSub y y)) (Inne …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      ⊢ Eq (Real.sin (HAdd.hAdd (InnerProductGeometry.angle x (HSub.hSub x y)) (Inne …
    -/
  · rw [Real.sin_add, cos_angle, cos_angle]
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HSub.hSub  …
    -/
    have hxn : ‖x‖ ≠ 0 := fun h => hx (norm_eq_zero.1 h)
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HSub.hSub  …
    -/
    have hyn : ‖y‖ ≠ 0 := fun h => hy (norm_eq_zero.1 h)
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HSub.hSub  …
    -/
    have hxyn : ‖x - y‖ ≠ 0 := fun h => hxy (eq_of_sub_eq_zero (norm_eq_zero.1 h))
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeometry.angle x (HSub.hSub  …
    -/
    apply mul_right_cancel₀ hxn
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeometry.angle x  …
    -/
    apply mul_right_cancel₀ hyn
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeomet …
    -/
    apply mul_right_cancel₀ hxyn
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HAdd.hAdd (HMul.hMul (Real.sin (InnerPr …
    -/
    apply mul_right_cancel₀ hxyn
    have H1 :
      Real.sin (angle x (x - y)) * (⟪y, y - x⟫ / (‖y‖ * ‖y - x‖)) * ‖x‖ * ‖y‖ * ‖x - y‖ =
        Real.sin (angle x (x - y)) * (‖x‖ * ‖x - y‖) * (⟪y, y - x⟫ / (‖y‖ * ‖y - x‖)) * ‖y‖ := by
      ring
    have H2 :
      ⟪x, x - y⟫ / (‖x‖ * ‖y - x‖) * Real.sin (angle y (y - x)) * ‖x‖ * ‖y‖ * ‖y - x‖ =
        ⟪x, x - y⟫ / (‖x‖ * ‖y - x‖) * (Real.sin (angle y (y - x)) * (‖y‖ * ‖y - x‖)) * ‖x‖ := by
      ring
    have H3 :
      ⟪x, x⟫ * (⟪x, x⟫ - ⟪x, y⟫ - (⟪x, y⟫ - ⟪y, y⟫)) - (⟪x, x⟫ - ⟪x, y⟫) * (⟪x, x⟫ - ⟪x, y⟫) =
        ⟪x, x⟫ * ⟪y, y⟫ - ⟪x, y⟫ * ⟪x, y⟫ := by
      ring
    have H4 :
      ⟪y, y⟫ * (⟪y, y⟫ - ⟪x, y⟫ - (⟪x, y⟫ - ⟪x, x⟫)) - (⟪y, y⟫ - ⟪x, y⟫) * (⟪y, y⟫ - ⟪x, y⟫) =
        ⟪x, x⟫ * ⟪y, y⟫ - ⟪x, y⟫ * ⟪x, y⟫ := by
      ring
    rw [right_distrib, right_distrib, right_distrib, right_distrib, H1, sin_angle_mul_norm_mul_norm,
      norm_sub_rev x y, H2, sin_angle_mul_norm_mul_norm, norm_sub_rev y x,
      mul_assoc (Real.sin (angle x y)), sin_angle_mul_norm_mul_norm, inner_sub_left, inner_sub_left,
      inner_sub_right, inner_sub_right, inner_sub_right, inner_sub_right, real_inner_comm x y, H3,
      H4, real_inner_self_eq_norm_mul_norm, real_inner_self_eq_norm_mul_norm,
      real_inner_eq_norm_mul_self_add_norm_mul_self_sub_norm_sub_mul_self_div_two]
    -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp [hxn, hyn, hxyn]`, but was really slow
    -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
    simp (disch := field_simp_discharge) only [mul_div_assoc', div_mul_eq_mul_div, div_div,
      sub_div', Real.sqrt_div', Real.sqrt_mul_self, add_div', div_add', eq_div_iff, div_eq_iff]
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hxy : Not (Eq x y)
      hxn : Ne (Norm.norm x) 0
      hyn : Ne (Norm.norm y) 0
      hxyn : Ne (Norm.norm (HSub.hSub x y)) 0
      H1 : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (Real.sin (InnerProductGeo …
      H2 : Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HDiv.hDiv (Inner.inner x  …
      H3 : Eq (HSub.hSub (HMul.hMul (Inner.inner x x) (HSub.hSub (HSub.hSub (Inner.i …
      H4 : Eq (HSub.hSub (HMul.hMul (Inner.inner y y) (HSub.hSub (HSub.hSub (Inner.i …
      ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HSub.h …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The cosine of the sum of the angles of a possibly degenerate
triangle (where two given sides are nonzero), vector angle form. -/
theorem cos_angle_add_angle_sub_add_angle_sub_eq_neg_one {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    Real.cos (angle x y + angle x (x - y) + angle y (y - x)) = -1 := by
  rw [add_assoc, Real.cos_add, cos_angle_sub_add_angle_sub_rev_eq_neg_cos_angle hx hy,
    sin_angle_sub_add_angle_sub_rev_eq_sin_angle hx hy, mul_neg, ← neg_add', add_comm, ← sq, ← sq,
    Real.sin_sq_add_cos_sq]


/-- The sine of the sum of the angles of a possibly degenerate
triangle (where two given sides are nonzero), vector angle form. -/
theorem sin_angle_add_angle_sub_add_angle_sub_eq_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    Real.sin (angle x y + angle x (x - y) + angle y (y - x)) = 0 := by
  rw [add_assoc, Real.sin_add, cos_angle_sub_add_angle_sub_rev_eq_neg_cos_angle hx hy,
    sin_angle_sub_add_angle_sub_rev_eq_sin_angle hx hy]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.sin (InnerProductGeometry.angle x y)) (Neg.ne …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The sum of the angles of a possibly degenerate triangle (where the
two given sides are nonzero), vector angle form. -/
theorem angle_add_angle_sub_add_angle_sub_eq_pi {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    angle x y + angle x (x - y) + angle y (y - x) = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
  -/
  have hcos := cos_angle_add_angle_sub_add_angle_sub_eq_neg_one hx hy
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
  -/
  have hsin := sin_angle_add_angle_sub_add_angle_sub_eq_zero hx hy
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
    hsin : Eq (Real.sin (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
  -/
  rw [Real.sin_eq_zero_iff] at hsin
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
    hsin : Exists fun n => Eq (HMul.hMul (↑n) Real.pi) (HAdd.hAdd (HAdd.hAdd (Inne …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
  -/
  cases' hsin with n hn
  /-
    case intro
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
    n : Int
    hn : Eq (HMul.hMul (↑n) Real.pi) (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.a …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
  -/
  symm at hn
  have h0 : 0 ≤ angle x y + angle x (x - y) + angle y (y - x) :=
    add_nonneg (add_nonneg (angle_nonneg _ _) (angle_nonneg _ _)) (angle_nonneg _ _)
  have h3lt : angle x y + angle x (x - y) + angle y (y - x) < π + π + π := by
    by_contra hnlt
    have hxy : angle x y = π := by
      by_contra hxy
      exact hnlt (add_lt_add_of_lt_of_le (add_lt_add_of_lt_of_le (lt_of_le_of_ne
        (angle_le_pi _ _) hxy) (angle_le_pi _ _)) (angle_le_pi _ _))
    rw [hxy] at hnlt
    rw [angle_eq_pi_iff] at hxy
    rcases hxy with ⟨hx, ⟨r, ⟨hr, hxr⟩⟩⟩
    rw [hxr, ← one_smul ℝ x, ← mul_smul, mul_one, ← sub_smul, one_smul, sub_eq_add_neg,
      angle_smul_right_of_pos _ _ (add_pos zero_lt_one (neg_pos_of_neg hr)), angle_self hx,
      add_zero] at hnlt
    apply hnlt
    rw [add_assoc]
    exact add_lt_add_left (lt_of_le_of_lt (angle_le_pi _ _) (lt_add_of_pos_right π Real.pi_pos)) _
  have hn0 : 0 ≤ n := by
    rw [hn, mul_nonneg_iff_left_nonneg_of_pos Real.pi_pos] at h0
    norm_cast at h0
  have hn3 : n < 3 := by
    rw [hn, show π + π + π = 3 * π by ring] at h3lt
    replace h3lt := lt_of_mul_lt_mul_right h3lt (le_of_lt Real.pi_pos)
    norm_cast at h3lt
  /-
    case intro
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
    n : Int
    hn : Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGe …
    h0 : LE.le 0 (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
    h3lt : LT.lt (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
    hn0 : LE.le 0 n
    hn3 : LT.lt n 3
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
  -/
  interval_cases n
    /-
      case intro.«0»
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
      n : Int
      h0 : LE.le 0 (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
      h3lt : LT.lt (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
      hn : Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGe …
      hn0 : LE.le 0 0
      hn3 : LT.lt 0 3
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
    -/
  · simp [hn] at hcos
    /-
      🎉 no goals
    -/
    /-
      case intro.«1»
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
      n : Int
      h0 : LE.le 0 (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
      h3lt : LT.lt (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
      hn : Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGe …
      hn0 : LE.le 0 1
      hn3 : LT.lt 1 3
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
    -/
  · norm_num [hn]
    /-
      🎉 no goals
    -/
    /-
      case intro.«2»
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      hcos : Eq (Real.cos (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (In …
      n : Int
      h0 : LE.le 0 (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
      h3lt : LT.lt (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProd …
      hn : Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGe …
      hn0 : LE.le 0 2
      hn3 : LT.lt 2 3
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (InnerProductGeometry.angle x y) (InnerProductGeome …
    -/
  · simp [hn] at hcos
    /-
      🎉 no goals
    -/


/-- **Law of cosines** (cosine rule), angle-at-point form. -/
theorem dist_sq_eq_dist_sq_add_dist_sq_sub_two_mul_dist_mul_dist_mul_cos_angle (p1 p2 p3 : P) :
    dist p1 p3 * dist p1 p3 = dist p1 p2 * dist p1 p2 + dist p3 p2 * dist p3 p2 -
      2 * dist p1 p2 * dist p3 p2 * Real.cos (∠ p1 p2 p3) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    ⊢ Eq (HMul.hMul (Dist.dist p1 p3) (Dist.dist p1 p3)) (HSub.hSub (HAdd.hAdd (HM …
  -/
  rw [dist_eq_norm_vsub V p1 p3, dist_eq_norm_vsub V p1 p2, dist_eq_norm_vsub V p3 p2]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    ⊢ Eq (HMul.hMul (Norm.norm (VSub.vsub p1 p3)) (Norm.norm (VSub.vsub p1 p3))) ( …
  -/
  unfold angle
  convert norm_sub_sq_eq_norm_sq_add_norm_sq_sub_two_mul_norm_mul_norm_mul_cos_angle
    (p1 -ᵥ p2 : V) (p3 -ᵥ p2 : V)
    /-
      case h.e'_2.h.e'_5.h.e'_3
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p1 p2 p3 : P
      ⊢ Eq (VSub.vsub p1 p3) (HSub.hSub (VSub.vsub p1 p2) (VSub.vsub p3 p2))
    -/
  · exact (vsub_sub_vsub_cancel_right p1 p3 p2).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_6.h.e'_3
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p1 p2 p3 : P
      ⊢ Eq (VSub.vsub p1 p3) (HSub.hSub (VSub.vsub p1 p2) (VSub.vsub p3 p2))
    -/
  · exact (vsub_sub_vsub_cancel_right p1 p3 p2).symm
    /-
      🎉 no goals
    -/


alias law_cos := dist_sq_eq_dist_sq_add_dist_sq_sub_two_mul_dist_mul_dist_mul_cos_angle


/-- **Isosceles Triangle Theorem**: Pons asinorum, angle-at-point form. -/
theorem angle_eq_angle_of_dist_eq {p1 p2 p3 : P} (h : dist p1 p2 = dist p1 p3) :
    ∠ p1 p2 p3 = ∠ p1 p3 p2 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Dist.dist p1 p2) (Dist.dist p1 p3)
    ⊢ Eq (EuclideanGeometry.angle p1 p2 p3) (EuclideanGeometry.angle p1 p3 p2)
  -/
  rw [dist_eq_norm_vsub V p1 p2, dist_eq_norm_vsub V p1 p3] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
    ⊢ Eq (EuclideanGeometry.angle p1 p2 p3) (EuclideanGeometry.angle p1 p3 p2)
  -/
  unfold angle
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) (InnerPr …
  -/
  convert angle_sub_eq_angle_sub_rev_of_norm_eq h
    /-
      case h.e'_2.h.e'_5
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p1 p2 p3 : P
      h : Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
      ⊢ Eq (VSub.vsub p3 p2) (HSub.hSub (VSub.vsub p1 p2) (VSub.vsub p1 p3))
    -/
  · exact (vsub_sub_vsub_cancel_left p3 p2 p1).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p1 p2 p3 : P
      h : Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
      ⊢ Eq (VSub.vsub p2 p3) (HSub.hSub (VSub.vsub p1 p3) (VSub.vsub p1 p2))
    -/
  · exact (vsub_sub_vsub_cancel_left p2 p3 p1).symm
    /-
      🎉 no goals
    -/


/-- Converse of pons asinorum, angle-at-point form. -/
theorem dist_eq_of_angle_eq_angle_of_angle_ne_pi {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = ∠ p1 p3 p2)
    (hpi : ∠ p2 p1 p3 ≠ π) : dist p1 p2 = dist p1 p3 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) (EuclideanGeometry.angle p1 p3 p2)
    hpi : Ne (EuclideanGeometry.angle p2 p1 p3) Real.pi
    ⊢ Eq (Dist.dist p1 p2) (Dist.dist p1 p3)
  -/
  unfold angle at h hpi
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) (Inner …
    hpi : Ne (InnerProductGeometry.angle (VSub.vsub p2 p1) (VSub.vsub p3 p1)) Real …
    ⊢ Eq (Dist.dist p1 p2) (Dist.dist p1 p3)
  -/
  rw [dist_eq_norm_vsub V p1 p2, dist_eq_norm_vsub V p1 p3]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) (Inner …
    hpi : Ne (InnerProductGeometry.angle (VSub.vsub p2 p1) (VSub.vsub p3 p1)) Real …
    ⊢ Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
  -/
  rw [← angle_neg_neg, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev] at hpi
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) (Inner …
    hpi : Ne (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p1 p3)) Real …
    ⊢ Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
  -/
  rw [← vsub_sub_vsub_cancel_left p3 p2 p1, ← vsub_sub_vsub_cancel_left p2 p3 p1] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (HSub.hSub (VSub.vsub p1  …
    hpi : Ne (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p1 p3)) Real …
    ⊢ Eq (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub p1 p3))
  -/
  exact norm_eq_of_angle_sub_eq_angle_sub_rev_of_angle_ne_pi h hpi
  /-
    🎉 no goals
  -/


/-- The **sum of the angles of a triangle** (possibly degenerate, where the
given vertex is distinct from the others), angle-at-point. -/
theorem angle_add_angle_add_angle_eq_pi {p1 p2 p3 : P} (h2 : p2 ≠ p1) (h3 : p3 ≠ p1) :
    ∠ p1 p2 p3 + ∠ p2 p3 p1 + ∠ p3 p1 p2 = π := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h2 : Ne p2 p1
    h3 : Ne p3 p1
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (EuclideanGeometry.angle p1 p2 p3) (EuclideanGeomet …
  -/
  rw [add_assoc, add_comm, add_comm (∠ p2 p3 p1), angle_comm p2 p3 p1]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h2 : Ne p2 p1
    h3 : Ne p3 p1
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (EuclideanGeometry.angle p3 p1 p2) (EuclideanGeomet …
  -/
  unfold angle
  rw [← angle_neg_neg (p1 -ᵥ p3), ← angle_neg_neg (p1 -ᵥ p2), neg_vsub_eq_vsub_rev,
    neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev, neg_vsub_eq_vsub_rev, ←
    vsub_sub_vsub_cancel_right p3 p2 p1, ← vsub_sub_vsub_cancel_right p2 p3 p1]
  exact angle_add_angle_sub_add_angle_sub_eq_pi (fun he => h3 (vsub_eq_zero_iff_eq.1 he)) fun he =>
    h2 (vsub_eq_zero_iff_eq.1 he)


/-- The **sum of the angles of a triangle** (possibly degenerate, where the triangle is a line),
oriented angles at point. -/
theorem oangle_add_oangle_add_oangle_eq_pi [Module.Oriented ℝ V (Fin 2)]
    [Fact (Module.finrank ℝ V = 2)] {p1 p2 p3 : P} (h21 : p2 ≠ p1) (h32 : p3 ≠ p2)
    (h13 : p1 ≠ p3) : ∡ p1 p2 p3 + ∡ p2 p3 p1 + ∡ p3 p1 p2 = π := by
  simpa only [neg_vsub_eq_vsub_rev] using
    positiveOrientation.oangle_add_cyc3_neg_left (vsub_ne_zero.mpr h21) (vsub_ne_zero.mpr h32)
      (vsub_ne_zero.mpr h13)


/-- **Stewart's Theorem**. -/
theorem dist_sq_mul_dist_add_dist_sq_mul_dist (a b c p : P) (h : ∠ b p c = π) :
    dist a b ^ 2 * dist c p + dist a c ^ 2 * dist b p =
    dist b c * (dist a p ^ 2 + dist b p * dist c p) := by
  rw [pow_two, pow_two, law_cos a p b, law_cos a p c,
    eq_sub_of_add_eq (angle_add_angle_eq_pi_of_angle_eq_pi a h), Real.cos_pi_sub,
    dist_eq_add_dist_of_angle_eq_pi h]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c p : P
    h : Eq (EuclideanGeometry.angle b p c) Real.pi
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (HAdd.hAdd (HMul.hMul (Dist.dist a p) (D …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Apollonius's Theorem**. -/
theorem dist_sq_add_dist_sq_eq_two_mul_dist_midpoint_sq_add_half_dist_sq (a b c : P) :
    dist a b ^ 2 + dist a c ^ 2 = 2 * (dist a (midpoint ℝ b c) ^ 2 + (dist b c / 2) ^ 2) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c : P
    ⊢ Eq (HAdd.hAdd (HPow.hPow (Dist.dist a b) 2) (HPow.hPow (Dist.dist a c) 2)) ( …
  -/
  by_cases hbc : b = c
    /-
      case pos
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c : P
      hbc : Eq b c
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Dist.dist a b) 2) (HPow.hPow (Dist.dist a c) 2)) ( …
    -/
  · simp [hbc, midpoint_self, dist_self, two_mul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c : P
      hbc : Not (Eq b c)
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Dist.dist a b) 2) (HPow.hPow (Dist.dist a c) 2)) ( …
    -/
  · let m := midpoint ℝ b c
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c : P
      hbc : Not (Eq b c)
      m : P := midpoint Real b c
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Dist.dist a b) 2) (HPow.hPow (Dist.dist a c) 2)) ( …
    -/
    have : dist b c ≠ 0 := (dist_pos.mpr hbc).ne'
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c : P
      hbc : Not (Eq b c)
      m : P := midpoint Real b c
      this : Ne (Dist.dist b c) 0
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Dist.dist a b) 2) (HPow.hPow (Dist.dist a c) 2)) ( …
    -/
    have hm := dist_sq_mul_dist_add_dist_sq_mul_dist a b c m (angle_midpoint_eq_pi b c hbc)
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c : P
      hbc : Not (Eq b c)
      m : P := midpoint Real b c
      this : Ne (Dist.dist b c) 0
      hm : Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (Dist.dist a b) 2) (Dist.dist c m)) ( …
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Dist.dist a b) 2) (HPow.hPow (Dist.dist a c) 2)) ( …
    -/
    simp only [m, dist_left_midpoint, dist_right_midpoint, Real.norm_two] at hm
    calc
      dist a b ^ 2 + dist a c ^ 2 = 2 / dist b c * (dist a b ^ 2 *
        ((2 : ℝ)⁻¹ * dist b c) + dist a c ^ 2 * (2⁻¹ * dist b c)) := by
        -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp`, but was really slow
        -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
        simp (disch := field_simp_discharge) only [inv_eq_one_div, div_mul_eq_mul_div, one_mul,
          mul_div_assoc', add_div', div_mul_cancel₀, div_div, eq_div_iff]
        ring
      _ = 2 * (dist a (midpoint ℝ b c) ^ 2 + (dist b c / 2) ^ 2) := by
        rw [hm]
        -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp`, but was really slow
        -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
        simp (disch := field_simp_discharge) only [inv_eq_one_div, div_mul_eq_mul_div, one_mul,
          mul_div_assoc', div_div, add_div', div_pow, eq_div_iff, div_eq_iff]
        ring


theorem dist_mul_of_eq_angle_of_dist_mul (a b c a' b' c' : P) (r : ℝ) (h : ∠ a' b' c' = ∠ a b c)
    (hab : dist a' b' = r * dist a b) (hcb : dist c' b' = r * dist c b) :
    dist a' c' = r * dist a c := by
  have h' : dist a' c' ^ 2 = (r * dist a c) ^ 2 := calc
    dist a' c' ^ 2 =
        dist a' b' ^ 2 + dist c' b' ^ 2 - 2 * dist a' b' * dist c' b' * Real.cos (∠ a' b' c') := by
      simp [pow_two, law_cos a' b' c']
    _ = r ^ 2 * (dist a b ^ 2 + dist c b ^ 2 - 2 * dist a b * dist c b * Real.cos (∠ a b c)) := by
      rw [h, hab, hcb]; ring
    _ = (r * dist a c) ^ 2 := by simp [pow_two, ← law_cos a b c, mul_pow]; ring
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c a' b' c' : P
    r : Real
    h : Eq (EuclideanGeometry.angle a' b' c') (EuclideanGeometry.angle a b c)
    hab : Eq (Dist.dist a' b') (HMul.hMul r (Dist.dist a b))
    hcb : Eq (Dist.dist c' b') (HMul.hMul r (Dist.dist c b))
    h' : Eq (HPow.hPow (Dist.dist a' c') 2) (HPow.hPow (HMul.hMul r (Dist.dist a c …
    ⊢ Eq (Dist.dist a' c') (HMul.hMul r (Dist.dist a c))
  -/
  by_cases hab₁ : a = b
  · have hab'₁ : a' = b' := by
      rw [← dist_eq_zero, hab, dist_eq_zero.mpr hab₁, mul_zero r]
    /-
      case pos
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c a' b' c' : P
      r : Real
      h : Eq (EuclideanGeometry.angle a' b' c') (EuclideanGeometry.angle a b c)
      hab : Eq (Dist.dist a' b') (HMul.hMul r (Dist.dist a b))
      hcb : Eq (Dist.dist c' b') (HMul.hMul r (Dist.dist c b))
      h' : Eq (HPow.hPow (Dist.dist a' c') 2) (HPow.hPow (HMul.hMul r (Dist.dist a c …
      hab₁ : Eq a b
      hab'₁ : Eq a' b'
      ⊢ Eq (Dist.dist a' c') (HMul.hMul r (Dist.dist a c))
    -/
    rw [hab₁, hab'₁, dist_comm b' c', dist_comm b c, hcb]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c a' b' c' : P
      r : Real
      h : Eq (EuclideanGeometry.angle a' b' c') (EuclideanGeometry.angle a b c)
      hab : Eq (Dist.dist a' b') (HMul.hMul r (Dist.dist a b))
      hcb : Eq (Dist.dist c' b') (HMul.hMul r (Dist.dist c b))
      h' : Eq (HPow.hPow (Dist.dist a' c') 2) (HPow.hPow (HMul.hMul r (Dist.dist a c …
      hab₁ : Not (Eq a b)
      ⊢ Eq (Dist.dist a' c') (HMul.hMul r (Dist.dist a c))
    -/
  · have h1 : 0 ≤ r * dist a b := by rw [← hab]; exact dist_nonneg
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c a' b' c' : P
      r : Real
      h : Eq (EuclideanGeometry.angle a' b' c') (EuclideanGeometry.angle a b c)
      hab : Eq (Dist.dist a' b') (HMul.hMul r (Dist.dist a b))
      hcb : Eq (Dist.dist c' b') (HMul.hMul r (Dist.dist c b))
      h' : Eq (HPow.hPow (Dist.dist a' c') 2) (HPow.hPow (HMul.hMul r (Dist.dist a c …
      hab₁ : Not (Eq a b)
      h1 : LE.le 0 (HMul.hMul r (Dist.dist a b))
      ⊢ Eq (Dist.dist a' c') (HMul.hMul r (Dist.dist a c))
    -/
    have h2 : 0 ≤ r := nonneg_of_mul_nonneg_left h1 (dist_pos.mpr hab₁)
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      a b c a' b' c' : P
      r : Real
      h : Eq (EuclideanGeometry.angle a' b' c') (EuclideanGeometry.angle a b c)
      hab : Eq (Dist.dist a' b') (HMul.hMul r (Dist.dist a b))
      hcb : Eq (Dist.dist c' b') (HMul.hMul r (Dist.dist c b))
      h' : Eq (HPow.hPow (Dist.dist a' c') 2) (HPow.hPow (HMul.hMul r (Dist.dist a c …
      hab₁ : Not (Eq a b)
      h1 : LE.le 0 (HMul.hMul r (Dist.dist a b))
      h2 : LE.le 0 r
      ⊢ Eq (Dist.dist a' c') (HMul.hMul r (Dist.dist a c))
    -/
    exact (sq_eq_sq₀ dist_nonneg (mul_nonneg h2 dist_nonneg)).mp h'
    /-
      🎉 no goals
    -/


