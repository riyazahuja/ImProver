/-- The undirected angle between two vectors. If either vector is 0,
this is π/2. See `Orientation.oangle` for the corresponding oriented angle
definition. -/
def angle (x y : V) : ℝ :=
  Real.arccos (⟪x, y⟫ / (‖x‖ * ‖y‖))


theorem continuousAt_angle {x : V × V} (hx1 : x.1 ≠ 0) (hx2 : x.2 ≠ 0) :
    ContinuousAt (fun y : V × V => angle y.1 y.2) x :=
  Real.continuous_arccos.continuousAt.comp <|
    continuous_inner.continuousAt.div
      ((continuous_norm.comp continuous_fst).mul (continuous_norm.comp continuous_snd)).continuousAt
          /-
            V : Type u_1
            inst✝¹ : NormedAddCommGroup V
            inst✝ : InnerProductSpace Real V
            x : Prod V V
            hx1 : Ne x.1 0
            hx2 : Ne x.2 0
            ⊢ Ne (HMul.hMul (Norm.norm x.1) (Norm.norm x.2)) 0
          -/
      (by simp [hx1, hx2])
          /-
            🎉 no goals
          -/


theorem angle_smul_smul {c : ℝ} (hc : c ≠ 0) (x y : V) : angle (c • x) (c • y) = angle x y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    c : Real
    hc : Ne c 0
    x y : V
    ⊢ Eq (InnerProductGeometry.angle (HSMul.hSMul c x) (HSMul.hSMul c y)) (InnerPr …
  -/
  have : c * c ≠ 0 := mul_ne_zero hc hc
  rw [angle, angle, real_inner_smul_left, inner_smul_right, norm_smul, norm_smul, Real.norm_eq_abs,
    mul_mul_mul_comm _ ‖x‖, abs_mul_abs_self, ← mul_assoc c c, mul_div_mul_left _ _ this]


@[simp]
theorem _root_.LinearIsometry.angle_map {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F]
    [InnerProductSpace ℝ E] [InnerProductSpace ℝ F] (f : E →ₗᵢ[ℝ] F) (u v : E) :
    angle (f u) (f v) = angle u v := by
  /-
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : InnerProductSpace Real E
    inst✝ : InnerProductSpace Real F
    f : LinearIsometry (RingHom.id Real) E F
    u v : E
    ⊢ Eq (InnerProductGeometry.angle (f u) (f v)) (InnerProductGeometry.angle u v)
  -/
  rw [angle, angle, f.inner_map_map, f.norm_map, f.norm_map]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem _root_.Submodule.angle_coe {s : Submodule ℝ V} (x y : s) :
    angle (x : V) (y : V) = angle x y :=
  s.subtypeₗᵢ.angle_map x y


/-- The cosine of the angle between two vectors. -/
theorem cos_angle (x y : V) : Real.cos (angle x y) = ⟪x, y⟫ / (‖x‖ * ‖y‖) :=
  Real.cos_arccos (abs_le.mp (abs_real_inner_div_norm_mul_norm_le_one x y)).1
    (abs_le.mp (abs_real_inner_div_norm_mul_norm_le_one x y)).2


/-- The angle between two vectors does not depend on their order. -/
theorem angle_comm (x y : V) : angle x y = angle y x := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (InnerProductGeometry.angle x y) (InnerProductGeometry.angle y x)
  -/
  unfold angle
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (Real.arccos (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm. …
  -/
  rw [real_inner_comm, mul_comm]
  /-
    🎉 no goals
  -/


/-- The angle between the negation of two vectors. -/
@[simp]
theorem angle_neg_neg (x y : V) : angle (-x) (-y) = angle x y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (InnerProductGeometry.angle (Neg.neg x) (Neg.neg y)) (InnerProductGeometr …
  -/
  unfold angle
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (Real.arccos (HDiv.hDiv (Inner.inner (Neg.neg x) (Neg.neg y)) (HMul.hMul  …
  -/
  rw [inner_neg_neg, norm_neg, norm_neg]
  /-
    🎉 no goals
  -/


/-- The angle between two vectors is nonnegative. -/
theorem angle_nonneg (x y : V) : 0 ≤ angle x y :=
  Real.arccos_nonneg _


/-- The angle between two vectors is at most π. -/
theorem angle_le_pi (x y : V) : angle x y ≤ π :=
  Real.arccos_le_pi _


/-- The angle between a vector and the negation of another vector. -/
theorem angle_neg_right (x y : V) : angle x (-y) = π - angle x y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (InnerProductGeometry.angle x (Neg.neg y)) (HSub.hSub Real.pi (InnerProdu …
  -/
  unfold angle
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (Real.arccos (HDiv.hDiv (Inner.inner x (Neg.neg y)) (HMul.hMul (Norm.norm …
  -/
  rw [← Real.arccos_neg, norm_neg, inner_neg_right, neg_div]
  /-
    🎉 no goals
  -/


/-- The angle between the negation of a vector and another vector. -/
theorem angle_neg_left (x y : V) : angle (-x) y = π - angle x y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (InnerProductGeometry.angle (Neg.neg x) y) (HSub.hSub Real.pi (InnerProdu …
  -/
  rw [← angle_neg_neg, neg_neg, angle_neg_right]
  /-
    🎉 no goals
  -/


/-- The angle between the zero vector and a vector. -/
@[simp]
theorem angle_zero_left (x : V) : angle 0 x = π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    ⊢ Eq (InnerProductGeometry.angle 0 x) (HDiv.hDiv Real.pi 2)
  -/
  unfold angle
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    ⊢ Eq (Real.arccos (HDiv.hDiv (Inner.inner 0 x) (HMul.hMul (Norm.norm 0) (Norm. …
  -/
  rw [inner_zero_left, zero_div, Real.arccos_zero]
  /-
    🎉 no goals
  -/


/-- The angle between a vector and the zero vector. -/
@[simp]
theorem angle_zero_right (x : V) : angle x 0 = π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    ⊢ Eq (InnerProductGeometry.angle x 0) (HDiv.hDiv Real.pi 2)
  -/
  unfold angle
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    ⊢ Eq (Real.arccos (HDiv.hDiv (Inner.inner x 0) (HMul.hMul (Norm.norm x) (Norm. …
  -/
  rw [inner_zero_right, zero_div, Real.arccos_zero]
  /-
    🎉 no goals
  -/


/-- The angle between a nonzero vector and itself. -/
@[simp]
theorem angle_self {x : V} (hx : x ≠ 0) : angle x x = 0 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    hx : Ne x 0
    ⊢ Eq (InnerProductGeometry.angle x x) 0
  -/
  unfold angle
  rw [← real_inner_self_eq_norm_mul_norm, div_self (inner_self_ne_zero.2 hx : ⟪x, x⟫ ≠ 0),
    Real.arccos_one]


/-- The angle between a nonzero vector and its negation. -/
@[simp]
theorem angle_self_neg_of_nonzero {x : V} (hx : x ≠ 0) : angle x (-x) = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    hx : Ne x 0
    ⊢ Eq (InnerProductGeometry.angle x (Neg.neg x)) Real.pi
  -/
  rw [angle_neg_right, angle_self hx, sub_zero]
  /-
    🎉 no goals
  -/


/-- The angle between the negation of a nonzero vector and that
vector. -/
@[simp]
theorem angle_neg_self_of_nonzero {x : V} (hx : x ≠ 0) : angle (-x) x = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x : V
    hx : Ne x 0
    ⊢ Eq (InnerProductGeometry.angle (Neg.neg x) x) Real.pi
  -/
  rw [angle_comm, angle_self_neg_of_nonzero hx]
  /-
    🎉 no goals
  -/


/-- The angle between a vector and a positive multiple of a vector. -/
@[simp]
theorem angle_smul_right_of_pos (x y : V) {r : ℝ} (hr : 0 < r) : angle x (r • y) = angle x y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (InnerProductGeometry.angle x (HSMul.hSMul r y)) (InnerProductGeometry.an …
  -/
  unfold angle
  rw [inner_smul_right, norm_smul, Real.norm_eq_abs, abs_of_nonneg (le_of_lt hr), ← mul_assoc,
    mul_comm _ r, mul_assoc, mul_div_mul_left _ _ (ne_of_gt hr)]


/-- The angle between a positive multiple of a vector and a vector. -/
@[simp]
theorem angle_smul_left_of_pos (x y : V) {r : ℝ} (hr : 0 < r) : angle (r • x) y = angle x y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (InnerProductGeometry.angle (HSMul.hSMul r x) y) (InnerProductGeometry.an …
  -/
  rw [angle_comm, angle_smul_right_of_pos y x hr, angle_comm]
  /-
    🎉 no goals
  -/


/-- The angle between a vector and a negative multiple of a vector. -/
@[simp]
theorem angle_smul_right_of_neg (x y : V) {r : ℝ} (hr : r < 0) :
    angle x (r • y) = angle x (-y) := by
  rw [← neg_neg r, neg_smul, angle_neg_right, angle_smul_right_of_pos x y (neg_pos_of_neg hr),
    angle_neg_right]


/-- The angle between a negative multiple of a vector and a vector. -/
@[simp]
theorem angle_smul_left_of_neg (x y : V) {r : ℝ} (hr : r < 0) : angle (r • x) y = angle (-x) y := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    r : Real
    hr : LT.lt r 0
    ⊢ Eq (InnerProductGeometry.angle (HSMul.hSMul r x) y) (InnerProductGeometry.an …
  -/
  rw [angle_comm, angle_smul_right_of_neg y x hr, angle_comm]
  /-
    🎉 no goals
  -/


/-- The cosine of the angle between two vectors, multiplied by the
product of their norms. -/
theorem cos_angle_mul_norm_mul_norm (x y : V) : Real.cos (angle x y) * (‖x‖ * ‖y‖) = ⟪x, y⟫ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (HMul.hMul (Real.cos (InnerProductGeometry.angle x y)) (HMul.hMul (Norm.n …
  -/
  rw [cos_angle, div_mul_cancel_of_imp]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0 → Eq (Inner.inner x y) 0
  -/
  simp +contextual [or_imp]
  /-
    🎉 no goals
  -/


/-- The sine of the angle between two vectors, multiplied by the
product of their norms. -/
theorem sin_angle_mul_norm_mul_norm (x y : V) :
    Real.sin (angle x y) * (‖x‖ * ‖y‖) = √(⟪x, x⟫ * ⟪y, y⟫ - ⟪x, y⟫ * ⟪x, y⟫) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (HMul.hMul (Real.sin (InnerProductGeometry.angle x y)) (HMul.hMul (Norm.n …
  -/
  unfold angle
  rw [Real.sin_arccos, ← Real.sqrt_mul_self (mul_nonneg (norm_nonneg x) (norm_nonneg y)),
    ← Real.sqrt_mul' _ (mul_self_nonneg _), sq,
    Real.sqrt_mul_self (mul_nonneg (norm_nonneg x) (norm_nonneg y)),
    real_inner_self_eq_norm_mul_norm, real_inner_self_eq_norm_mul_norm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Eq (HMul.hMul (HSub.hSub 1 (HMul.hMul (HDiv.hDiv (Inner.inner x y) (HMul.hMu …
  -/
  by_cases h : ‖x‖ * ‖y‖ = 0
  · rw [show ‖x‖ * ‖x‖ * (‖y‖ * ‖y‖) = ‖x‖ * ‖y‖ * (‖x‖ * ‖y‖) by ring, h, mul_zero,
      mul_zero, zero_sub]
    /-
      case pos
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
      ⊢ Eq (Real.sqrt 0) (Neg.neg (HMul.hMul (Inner.inner x y) (Inner.inner x y))).s …
    -/
    cases' eq_zero_or_eq_zero_of_mul_eq_zero h with hx hy
      /-
        case pos.inl
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        h : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
        hx : Eq (Norm.norm x) 0
        ⊢ Eq (Real.sqrt 0) (Neg.neg (HMul.hMul (Inner.inner x y) (Inner.inner x y))).s …
      -/
    · rw [norm_eq_zero] at hx
      /-
        case pos.inl
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        h : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
        hx : Eq x 0
        ⊢ Eq (Real.sqrt 0) (Neg.neg (HMul.hMul (Inner.inner x y) (Inner.inner x y))).s …
      -/
      rw [hx, inner_zero_left, zero_mul, neg_zero]
      /-
        🎉 no goals
      -/
      /-
        case pos.inr
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        h : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
        hy : Eq (Norm.norm y) 0
        ⊢ Eq (Real.sqrt 0) (Neg.neg (HMul.hMul (Inner.inner x y) (Inner.inner x y))).s …
      -/
    · rw [norm_eq_zero] at hy
      /-
        case pos.inr
        V : Type u_1
        inst✝¹ : NormedAddCommGroup V
        inst✝ : InnerProductSpace Real V
        x y : V
        h : Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
        hy : Eq y 0
        ⊢ Eq (Real.sqrt 0) (Neg.neg (HMul.hMul (Inner.inner x y) (Inner.inner x y))).s …
      -/
      rw [hy, inner_zero_right, zero_mul, neg_zero]
      /-
        🎉 no goals
      -/
  · -- takes 600ms; squeezing the "equivalent" simp call yields an invalid result
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Not (Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0)
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (HMul.hMul (HDiv.hDiv (Inner.inner x y) (HMul.hMu …
    -/
    field_simp [h]
    /-
      case neg
      V : Type u_1
      inst✝¹ : NormedAddCommGroup V
      inst✝ : InnerProductSpace Real V
      x y : V
      h : Not (Eq (HMul.hMul (Norm.norm x) (Norm.norm y)) 0)
      ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (Norm.norm x) (Norm.norm y)) (HMul.hMul  …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


/-- The angle between two vectors is zero if and only if they are
nonzero and one is a positive multiple of the other. -/
theorem angle_eq_zero_iff {x y : V} : angle x y = 0 ↔ x ≠ 0 ∧ ∃ r : ℝ, 0 < r ∧ y = r • x := by
  rw [angle, ← real_inner_div_norm_mul_norm_eq_one_iff, Real.arccos_eq_zero, LE.le.le_iff_eq,
    eq_comm]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ LE.le (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) 1
  -/
  exact (abs_le.mp (abs_real_inner_div_norm_mul_norm_le_one x y)).2
  /-
    🎉 no goals
  -/


/-- The angle between two vectors is π if and only if they are nonzero
and one is a negative multiple of the other. -/
theorem angle_eq_pi_iff {x y : V} : angle x y = π ↔ x ≠ 0 ∧ ∃ r : ℝ, r < 0 ∧ y = r • x := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (InnerProductGeometry.angle x y) Real.pi) (And (Ne x 0) (Exists fun  …
  -/
  rw [angle, ← real_inner_div_norm_mul_norm_eq_neg_one_iff, Real.arccos_eq_pi, LE.le.le_iff_eq]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ LE.le (-1) (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm  …
  -/
  exact (abs_le.mp (abs_real_inner_div_norm_mul_norm_le_one x y)).1
  /-
    🎉 no goals
  -/


/-- If the angle between two vectors is π, the angles between those
vectors and a third vector add to π. -/
theorem angle_add_angle_eq_pi_of_angle_eq_pi {x y : V} (z : V) (h : angle x y = π) :
    angle x z + angle y z = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y z : V
    h : Eq (InnerProductGeometry.angle x y) Real.pi
    ⊢ Eq (HAdd.hAdd (InnerProductGeometry.angle x z) (InnerProductGeometry.angle y …
  -/
  rcases angle_eq_pi_iff.1 h with ⟨_, ⟨r, ⟨hr, rfl⟩⟩⟩
  /-
    case intro.intro.intro
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x z : V
    left✝ : Ne x 0
    r : Real
    hr : LT.lt r 0
    h : Eq (InnerProductGeometry.angle x (HSMul.hSMul r x)) Real.pi
    ⊢ Eq (HAdd.hAdd (InnerProductGeometry.angle x z) (InnerProductGeometry.angle ( …
  -/
  rw [angle_smul_left_of_neg x z hr, angle_neg_left, add_sub_cancel]
  /-
    🎉 no goals
  -/


/-- Two vectors have inner product 0 if and only if the angle between
them is π/2. -/
theorem inner_eq_zero_iff_angle_eq_pi_div_two (x y : V) : ⟪x, y⟫ = 0 ↔ angle x y = π / 2 :=
                 /-
                   V : Type u_1
                   inst✝¹ : NormedAddCommGroup V
                   inst✝ : InnerProductSpace Real V
                   x y : V
                   ⊢ Iff (Eq (InnerProductGeometry.angle x y) (HDiv.hDiv Real.pi 2)) (Eq (Inner.i …
                 -/
  Iff.symm <| by simp +contextual [angle, or_imp]
                 /-
                   🎉 no goals
                 -/


/-- If the angle between two vectors is π, the inner product equals the negative product
of the norms. -/
theorem inner_eq_neg_mul_norm_of_angle_eq_pi {x y : V} (h : angle x y = π) :
    ⟪x, y⟫ = -(‖x‖ * ‖y‖) := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (InnerProductGeometry.angle x y) Real.pi
    ⊢ Eq (Inner.inner x y) (Neg.neg (HMul.hMul (Norm.norm x) (Norm.norm y)))
  -/
  simp [← cos_angle_mul_norm_mul_norm, h]
  /-
    🎉 no goals
  -/


/-- If the angle between two vectors is 0, the inner product equals the product of the norms. -/
theorem inner_eq_mul_norm_of_angle_eq_zero {x y : V} (h : angle x y = 0) : ⟪x, y⟫ = ‖x‖ * ‖y‖ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (InnerProductGeometry.angle x y) 0
    ⊢ Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  simp [← cos_angle_mul_norm_mul_norm, h]
  /-
    🎉 no goals
  -/


/-- The inner product of two non-zero vectors equals the negative product of their norms
if and only if the angle between the two vectors is π. -/
theorem inner_eq_neg_mul_norm_iff_angle_eq_pi {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ⟪x, y⟫ = -(‖x‖ * ‖y‖) ↔ angle x y = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (Inner.inner x y) (Neg.neg (HMul.hMul (Norm.norm x) (Norm.norm y)))) …
  -/
  refine ⟨fun h => ?_, inner_eq_neg_mul_norm_of_angle_eq_pi⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Inner.inner x y) (Neg.neg (HMul.hMul (Norm.norm x) (Norm.norm y)))
    ⊢ Eq (InnerProductGeometry.angle x y) Real.pi
  -/
  have h₁ : ‖x‖ * ‖y‖ ≠ 0 := (mul_pos (norm_pos_iff.mpr hx) (norm_pos_iff.mpr hy)).ne'
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Inner.inner x y) (Neg.neg (HMul.hMul (Norm.norm x) (Norm.norm y)))
    h₁ : Ne (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
    ⊢ Eq (InnerProductGeometry.angle x y) Real.pi
  -/
  rw [angle, h, neg_div, div_self h₁, Real.arccos_neg_one]
  /-
    🎉 no goals
  -/


/-- The inner product of two non-zero vectors equals the product of their norms
if and only if the angle between the two vectors is 0. -/
theorem inner_eq_mul_norm_iff_angle_eq_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ⟪x, y⟫ = ‖x‖ * ‖y‖ ↔ angle x y = 0 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) (Eq (Inne …
  -/
  refine ⟨fun h => ?_, inner_eq_mul_norm_of_angle_eq_zero⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))
    ⊢ Eq (InnerProductGeometry.angle x y) 0
  -/
  have h₁ : ‖x‖ * ‖y‖ ≠ 0 := (mul_pos (norm_pos_iff.mpr hx) (norm_pos_iff.mpr hy)).ne'
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))
    h₁ : Ne (HMul.hMul (Norm.norm x) (Norm.norm y)) 0
    ⊢ Eq (InnerProductGeometry.angle x y) 0
  -/
  rw [angle, h, div_self h₁, Real.arccos_one]
  /-
    🎉 no goals
  -/


/-- If the angle between two vectors is π, the norm of their difference equals
the sum of their norms. -/
theorem norm_sub_eq_add_norm_of_angle_eq_pi {x y : V} (h : angle x y = π) :
    ‖x - y‖ = ‖x‖ + ‖y‖ := by
  rw [← sq_eq_sq₀ (norm_nonneg (x - y)) (add_nonneg (norm_nonneg x) (norm_nonneg y)),
    norm_sub_pow_two_real, inner_eq_neg_mul_norm_of_angle_eq_pi h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (InnerProductGeometry.angle x y) Real.pi
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HPow.hPow (Norm.norm x) 2) (HMul.hMul 2 (Neg.neg ( …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- If the angle between two vectors is 0, the norm of their sum equals
the sum of their norms. -/
theorem norm_add_eq_add_norm_of_angle_eq_zero {x y : V} (h : angle x y = 0) :
    ‖x + y‖ = ‖x‖ + ‖y‖ := by
  rw [← sq_eq_sq₀ (norm_nonneg (x + y)) (add_nonneg (norm_nonneg x) (norm_nonneg y)),
    norm_add_pow_two_real, inner_eq_mul_norm_of_angle_eq_zero h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (InnerProductGeometry.angle x y) 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow (Norm.norm x) 2) (HMul.hMul 2 (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- If the angle between two vectors is 0, the norm of their difference equals
the absolute value of the difference of their norms. -/
theorem norm_sub_eq_abs_sub_norm_of_angle_eq_zero {x y : V} (h : angle x y = 0) :
    ‖x - y‖ = |‖x‖ - ‖y‖| := by
  rw [← sq_eq_sq₀ (norm_nonneg (x - y)) (abs_nonneg (‖x‖ - ‖y‖)), norm_sub_pow_two_real,
    inner_eq_mul_norm_of_angle_eq_zero h, sq_abs (‖x‖ - ‖y‖)]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (InnerProductGeometry.angle x y) 0
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HPow.hPow (Norm.norm x) 2) (HMul.hMul 2 (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The norm of the difference of two non-zero vectors equals the sum of their norms
if and only the angle between the two vectors is π. -/
theorem norm_sub_eq_add_norm_iff_angle_eq_pi {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ‖x - y‖ = ‖x‖ + ‖y‖ ↔ angle x y = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (Norm.norm (HSub.hSub x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))) …
  -/
  refine ⟨fun h => ?_, norm_sub_eq_add_norm_of_angle_eq_pi⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HSub.hSub x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    ⊢ Eq (InnerProductGeometry.angle x y) Real.pi
  -/
  rw [← inner_eq_neg_mul_norm_iff_angle_eq_pi hx hy]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HSub.hSub x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    ⊢ Eq (Inner.inner x y) (Neg.neg (HMul.hMul (Norm.norm x) (Norm.norm y)))
  -/
  obtain ⟨hxy₁, hxy₂⟩ := norm_nonneg (x - y), add_nonneg (norm_nonneg x) (norm_nonneg y)
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HSub.hSub x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    hxy₁ : LE.le 0 (Norm.norm (HSub.hSub x y))
    hxy₂ : LE.le 0 (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    ⊢ Eq (Inner.inner x y) (Neg.neg (HMul.hMul (Norm.norm x) (Norm.norm y)))
  -/
  rw [← sq_eq_sq₀ hxy₁ hxy₂, norm_sub_pow_two_real] at h
  calc
    ⟪x, y⟫ = (‖x‖ ^ 2 + ‖y‖ ^ 2 - (‖x‖ + ‖y‖) ^ 2) / 2 := by linarith
    _ = -(‖x‖ * ‖y‖) := by ring


/-- The norm of the sum of two non-zero vectors equals the sum of their norms
if and only the angle between the two vectors is 0. -/
theorem norm_add_eq_add_norm_iff_angle_eq_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ‖x + y‖ = ‖x‖ + ‖y‖ ↔ angle x y = 0 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))) …
  -/
  refine ⟨fun h => ?_, norm_add_eq_add_norm_of_angle_eq_zero⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    ⊢ Eq (InnerProductGeometry.angle x y) 0
  -/
  rw [← inner_eq_mul_norm_iff_angle_eq_zero hx hy]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    ⊢ Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  obtain ⟨hxy₁, hxy₂⟩ := norm_nonneg (x + y), add_nonneg (norm_nonneg x) (norm_nonneg y)
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    hxy₁ : LE.le 0 (Norm.norm (HAdd.hAdd x y))
    hxy₂ : LE.le 0 (HAdd.hAdd (Norm.norm x) (Norm.norm y))
    ⊢ Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  rw [← sq_eq_sq₀ hxy₁ hxy₂, norm_add_pow_two_real] at h
  calc
    ⟪x, y⟫ = ((‖x‖ + ‖y‖) ^ 2 - ‖x‖ ^ 2 - ‖y‖ ^ 2) / 2 := by linarith
    _ = ‖x‖ * ‖y‖ := by ring


/-- The norm of the difference of two non-zero vectors equals the absolute value
of the difference of their norms if and only the angle between the two vectors is 0. -/
theorem norm_sub_eq_abs_sub_norm_iff_angle_eq_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) :
    ‖x - y‖ = |‖x‖ - ‖y‖| ↔ angle x y = 0 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (Eq (Norm.norm (HSub.hSub x y)) (abs (HSub.hSub (Norm.norm x) (Norm.norm …
  -/
  refine ⟨fun h => ?_, norm_sub_eq_abs_sub_norm_of_angle_eq_zero⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HSub.hSub x y)) (abs (HSub.hSub (Norm.norm x) (Norm.norm y)))
    ⊢ Eq (InnerProductGeometry.angle x y) 0
  -/
  rw [← inner_eq_mul_norm_iff_angle_eq_zero hx hy]
  have h1 : ‖x - y‖ ^ 2 = (‖x‖ - ‖y‖) ^ 2 := by
    rw [h]
    exact sq_abs (‖x‖ - ‖y‖)
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    h : Eq (Norm.norm (HSub.hSub x y)) (abs (HSub.hSub (Norm.norm x) (Norm.norm y)))
    h1 : Eq (HPow.hPow (Norm.norm (HSub.hSub x y)) 2) (HPow.hPow (HSub.hSub (Norm. …
    ⊢ Eq (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  rw [norm_sub_pow_two_real] at h1
  calc
    ⟪x, y⟫ = ((‖x‖ + ‖y‖) ^ 2 - ‖x‖ ^ 2 - ‖y‖ ^ 2) / 2 := by linarith
    _ = ‖x‖ * ‖y‖ := by ring


/-- The norm of the sum of two vectors equals the norm of their difference if and only if
the angle between them is π/2. -/
theorem norm_add_eq_norm_sub_iff_angle_eq_pi_div_two (x y : V) :
    ‖x + y‖ = ‖x - y‖ ↔ angle x y = π / 2 := by
  rw [← sq_eq_sq₀ (norm_nonneg (x + y)) (norm_nonneg (x - y)),
    ← inner_eq_zero_iff_angle_eq_pi_div_two x y, norm_add_pow_two_real, norm_sub_pow_two_real]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow (Norm.norm x) 2) (HMul.hMul 2 (Inne …
  -/
                              /-
                                🎉 no goals
                              -/
  constructor <;> intro h <;> linarith
                              /-
                                🎉 no goals
                              -/


/-- The cosine of the angle between two vectors is 1 if and only if the angle is 0. -/
theorem cos_eq_one_iff_angle_eq_zero : cos (angle x y) = 1 ↔ angle x y = 0 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.cos (InnerProductGeometry.angle x y)) 1) (Eq (InnerProductGeom …
  -/
  rw [← cos_zero]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.cos (InnerProductGeometry.angle x y)) (Real.cos 0)) (Eq (Inner …
  -/
  exact injOn_cos.eq_iff ⟨angle_nonneg x y, angle_le_pi x y⟩ (left_mem_Icc.2 pi_pos.le)
  /-
    🎉 no goals
  -/


/-- The cosine of the angle between two vectors is 0 if and only if the angle is π / 2. -/
theorem cos_eq_zero_iff_angle_eq_pi_div_two : cos (angle x y) = 0 ↔ angle x y = π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.cos (InnerProductGeometry.angle x y)) 0) (Eq (InnerProductGeom …
  -/
  rw [← cos_pi_div_two]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.cos (InnerProductGeometry.angle x y)) (Real.cos (HDiv.hDiv Rea …
  -/
  apply injOn_cos.eq_iff ⟨angle_nonneg x y, angle_le_pi x y⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Membership.mem (Set.Icc 0 Real.pi) (HDiv.hDiv Real.pi 2)
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> linarith [pi_pos]
                  /-
                    🎉 no goals
                  -/


/-- The cosine of the angle between two vectors is -1 if and only if the angle is π. -/
theorem cos_eq_neg_one_iff_angle_eq_pi : cos (angle x y) = -1 ↔ angle x y = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.cos (InnerProductGeometry.angle x y)) (-1)) (Eq (InnerProductG …
  -/
  rw [← cos_pi]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.cos (InnerProductGeometry.angle x y)) (Real.cos Real.pi)) (Eq  …
  -/
  exact injOn_cos.eq_iff ⟨angle_nonneg x y, angle_le_pi x y⟩ (right_mem_Icc.2 pi_pos.le)
  /-
    🎉 no goals
  -/


/-- The sine of the angle between two vectors is 0 if and only if the angle is 0 or π. -/
theorem sin_eq_zero_iff_angle_eq_zero_or_angle_eq_pi :
    sin (angle x y) = 0 ↔ angle x y = 0 ∨ angle x y = π := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.sin (InnerProductGeometry.angle x y)) 0) (Or (Eq (InnerProduct …
  -/
  rw [sin_eq_zero_iff_cos_eq, cos_eq_one_iff_angle_eq_zero, cos_eq_neg_one_iff_angle_eq_pi]
  /-
    🎉 no goals
  -/


/-- The sine of the angle between two vectors is 1 if and only if the angle is π / 2. -/
theorem sin_eq_one_iff_angle_eq_pi_div_two : sin (angle x y) = 1 ↔ angle x y = π / 2 := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    ⊢ Iff (Eq (Real.sin (InnerProductGeometry.angle x y)) 1) (Eq (InnerProductGeom …
  -/
  refine ⟨fun h => ?_, fun h => by rw [h, sin_pi_div_two]⟩
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Real.sin (InnerProductGeometry.angle x y)) 1
    ⊢ Eq (InnerProductGeometry.angle x y) (HDiv.hDiv Real.pi 2)
  -/
  rw [← cos_eq_zero_iff_angle_eq_pi_div_two, ← abs_eq_zero, abs_cos_eq_sqrt_one_sub_sin_sq, h]
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    x y : V
    h : Eq (Real.sin (InnerProductGeometry.angle x y)) 1
    ⊢ Eq (HSub.hSub 1 (HPow.hPow 1 2)).sqrt 0
  -/
  simp
  /-
    🎉 no goals
  -/


