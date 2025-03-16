/-- The undirected angle at `p2` between the line segments to `p1` and
`p3`. If either of those points equals `p2`, this is π/2. Use
`open scoped EuclideanGeometry` to access the `∠ p1 p2 p3`
notation. -/
nonrec def angle (p1 p2 p3 : P) : ℝ :=
  angle (p1 -ᵥ p2 : V) (p3 -ᵥ p2)


@[inherit_doc] scoped notation "∠" => EuclideanGeometry.angle


theorem continuousAt_angle {x : P × P × P} (hx12 : x.1 ≠ x.2.1) (hx32 : x.2.2 ≠ x.2.1) :
    ContinuousAt (fun y : P × P × P => ∠ y.1 y.2.1 y.2.2) x := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    x : Prod P (Prod P P)
    hx12 : Ne x.1 x.2.1
    hx32 : Ne x.2.2 x.2.1
    ⊢ ContinuousAt (fun y => EuclideanGeometry.angle y.1 y.2.1 y.2.2) x
  -/
  let f : P × P × P → V × V := fun y => (y.1 -ᵥ y.2.1, y.2.2 -ᵥ y.2.1)
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    x : Prod P (Prod P P)
    hx12 : Ne x.1 x.2.1
    hx32 : Ne x.2.2 x.2.1
    f : Prod P (Prod P P) → Prod V V := fun y => { fst := VSub.vsub y.1 y.2.1, snd …
    ⊢ ContinuousAt (fun y => EuclideanGeometry.angle y.1 y.2.1 y.2.2) x
  -/
  have hf1 : (f x).1 ≠ 0 := by simp [f, hx12]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    x : Prod P (Prod P P)
    hx12 : Ne x.1 x.2.1
    hx32 : Ne x.2.2 x.2.1
    f : Prod P (Prod P P) → Prod V V := fun y => { fst := VSub.vsub y.1 y.2.1, snd …
    hf1 : Ne (f x).1 0
    ⊢ ContinuousAt (fun y => EuclideanGeometry.angle y.1 y.2.1 y.2.2) x
  -/
  have hf2 : (f x).2 ≠ 0 := by simp [f, hx32]
  exact (InnerProductGeometry.continuousAt_angle hf1 hf2).comp
    ((continuous_fst.vsub continuous_snd.fst).prod_mk
      (continuous_snd.snd.vsub continuous_snd.fst)).continuousAt


@[simp]
theorem _root_.AffineIsometry.angle_map {V₂ P₂ : Type*} [NormedAddCommGroup V₂]
    [InnerProductSpace ℝ V₂] [MetricSpace P₂] [NormedAddTorsor V₂ P₂]
    (f : P →ᵃⁱ[ℝ] P₂) (p₁ p₂ p₃ : P) : ∠ (f p₁) (f p₂) (f p₃) = ∠ p₁ p₂ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    V₂ : Type u_3
    P₂ : Type u_4
    inst✝³ : NormedAddCommGroup V₂
    inst✝² : InnerProductSpace Real V₂
    inst✝¹ : MetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineIsometry Real P P₂
    p₁ p₂ p₃ : P
    ⊢ Eq (EuclideanGeometry.angle (f p₁) (f p₂) (f p₃)) (EuclideanGeometry.angle p …
  -/
  simp_rw [angle, ← AffineIsometry.map_vsub, LinearIsometry.angle_map]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem _root_.AffineSubspace.angle_coe {s : AffineSubspace ℝ P} (p₁ p₂ p₃ : s) :
    haveI : Nonempty s := ⟨p₁⟩
    ∠ (p₁ : P) (p₂ : P) (p₃ : P) = ∠ p₁ p₂ p₃ :=
  haveI : Nonempty s := ⟨p₁⟩
  s.subtypeₐᵢ.angle_map p₁ p₂ p₃


/-- Angles are translation invariant -/
@[simp]
theorem angle_const_vadd (v : V) (p₁ p₂ p₃ : P) : ∠ (v +ᵥ p₁) (v +ᵥ p₂) (v +ᵥ p₃) = ∠ p₁ p₂ p₃ :=
  (AffineIsometryEquiv.constVAdd ℝ P v).toAffineIsometry.angle_map _ _ _


/-- Angles are translation invariant -/
@[simp]
theorem angle_vadd_const (v₁ v₂ v₃ : V) (p : P) : ∠ (v₁ +ᵥ p) (v₂ +ᵥ p) (v₃ +ᵥ p) = ∠ v₁ v₂ v₃ :=
  (AffineIsometryEquiv.vaddConst ℝ p).toAffineIsometry.angle_map _ _ _


/-- Angles are translation invariant -/
@[simp]
theorem angle_const_vsub (p p₁ p₂ p₃ : P) : ∠ (p -ᵥ p₁) (p -ᵥ p₂) (p -ᵥ p₃) = ∠ p₁ p₂ p₃ :=
  (AffineIsometryEquiv.constVSub ℝ p).toAffineIsometry.angle_map _ _ _


/-- Angles are translation invariant -/
@[simp]
theorem angle_vsub_const (p₁ p₂ p₃ p : P) : ∠ (p₁ -ᵥ p) (p₂ -ᵥ p) (p₃ -ᵥ p) = ∠ p₁ p₂ p₃ :=
  (AffineIsometryEquiv.vaddConst ℝ p).symm.toAffineIsometry.angle_map _ _ _


/-- Angles in a vector space are translation invariant -/
@[simp]
theorem angle_add_const (v₁ v₂ v₃ : V) (v : V) : ∠ (v₁ + v) (v₂ + v) (v₃ + v) = ∠ v₁ v₂ v₃ :=
  angle_vadd_const _ _ _ _


/-- Angles in a vector space are translation invariant -/
@[simp]
theorem angle_const_add (v : V) (v₁ v₂ v₃ : V) : ∠ (v + v₁) (v + v₂) (v + v₃) = ∠ v₁ v₂ v₃ :=
  angle_const_vadd _ _ _ _


/-- Angles in a vector space are translation invariant -/
@[simp]
theorem angle_sub_const (v₁ v₂ v₃ : V) (v : V) : ∠ (v₁ - v) (v₂ - v) (v₃ - v) = ∠ v₁ v₂ v₃ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    v₁ v₂ v₃ v : V
    ⊢ Eq (EuclideanGeometry.angle (HSub.hSub v₁ v) (HSub.hSub v₂ v) (HSub.hSub v₃  …
  -/
  simpa only [vsub_eq_sub] using angle_vsub_const v₁ v₂ v₃ v
  /-
    🎉 no goals
  -/


/-- Angles in a vector space are invariant to inversion -/
@[simp]
theorem angle_const_sub (v : V) (v₁ v₂ v₃ : V) : ∠ (v - v₁) (v - v₂) (v - v₃) = ∠ v₁ v₂ v₃ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    v v₁ v₂ v₃ : V
    ⊢ Eq (EuclideanGeometry.angle (HSub.hSub v v₁) (HSub.hSub v v₂) (HSub.hSub v v …
  -/
  simpa only [vsub_eq_sub] using angle_const_vsub v v₁ v₂ v₃
  /-
    🎉 no goals
  -/


/-- Angles in a vector space are invariant to inversion -/
@[simp]
theorem angle_neg (v₁ v₂ v₃ : V) : ∠ (-v₁) (-v₂) (-v₃) = ∠ v₁ v₂ v₃ := by
  /-
    V : Type u_1
    inst✝¹ : NormedAddCommGroup V
    inst✝ : InnerProductSpace Real V
    v₁ v₂ v₃ : V
    ⊢ Eq (EuclideanGeometry.angle (Neg.neg v₁) (Neg.neg v₂) (Neg.neg v₃)) (Euclide …
  -/
  simpa only [zero_sub] using angle_const_sub 0 v₁ v₂ v₃
  /-
    🎉 no goals
  -/


/-- The angle at a point does not depend on the order of the other two
points. -/
nonrec theorem angle_comm (p1 p2 p3 : P) : ∠ p1 p2 p3 = ∠ p3 p2 p1 :=
  angle_comm _ _


/-- The angle at a point is nonnegative. -/
nonrec theorem angle_nonneg (p1 p2 p3 : P) : 0 ≤ ∠ p1 p2 p3 :=
  angle_nonneg _ _


/-- The angle at a point is at most π. -/
nonrec theorem angle_le_pi (p1 p2 p3 : P) : ∠ p1 p2 p3 ≤ π :=
  angle_le_pi _ _


/-- The angle ∠AAB at a point is always `π / 2`. -/
@[simp] lemma angle_self_left (p₀ p : P) : ∠ p₀ p₀ p = π / 2 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₀ p : P
    ⊢ Eq (EuclideanGeometry.angle p₀ p₀ p) (HDiv.hDiv Real.pi 2)
  -/
  unfold angle
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₀ p : P
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p₀ p₀) (VSub.vsub p p₀)) (HDiv.hDi …
  -/
  rw [vsub_self]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₀ p : P
    ⊢ Eq (InnerProductGeometry.angle 0 (VSub.vsub p p₀)) (HDiv.hDiv Real.pi 2)
  -/
  exact angle_zero_left _
  /-
    🎉 no goals
  -/


/-- The angle ∠ABB at a point is always `π / 2`. -/
                                                                    /-
                                                                      V : Type u_1
                                                                      P : Type u_2
                                                                      inst✝³ : NormedAddCommGroup V
                                                                      inst✝² : InnerProductSpace Real V
                                                                      inst✝¹ : MetricSpace P
                                                                      inst✝ : NormedAddTorsor V P
                                                                      p₀ p : P
                                                                      ⊢ Eq (EuclideanGeometry.angle p p₀ p₀) (HDiv.hDiv Real.pi 2)
                                                                    -/
@[simp] lemma angle_self_right (p₀ p : P) : ∠ p p₀ p₀ = π / 2 := by rw [angle_comm, angle_self_left]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The angle ∠ABA at a point is `0`, unless `A = B`. -/
theorem angle_self_of_ne (h : p ≠ p₀) : ∠ p p₀ p = 0 := angle_self <| vsub_ne_zero.2 h


@[deprecated (since := "2024-02-14")] alias angle_eq_left := angle_self_left

@[deprecated (since := "2024-02-14")] alias angle_eq_right := angle_self_right

@[deprecated (since := "2024-02-14")] alias angle_eq_of_ne := angle_self_of_ne


/-- If the angle ∠ABC at a point is π, the angle ∠BAC is 0. -/
theorem angle_eq_zero_of_angle_eq_pi_left {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = π) : ∠ p2 p1 p3 = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    ⊢ Eq (EuclideanGeometry.angle p2 p1 p3) 0
  -/
  unfold angle at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) Real.pi
    ⊢ Eq (EuclideanGeometry.angle p2 p1 p3) 0
  -/
  rw [angle_eq_pi_iff] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : And (Ne (VSub.vsub p1 p2) 0) (Exists fun r => And (LT.lt r 0) (Eq (VSub.vs …
    ⊢ Eq (EuclideanGeometry.angle p2 p1 p3) 0
  -/
  rcases h with ⟨hp1p2, ⟨r, ⟨hr, hpr⟩⟩⟩
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne (VSub.vsub p1 p2) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p3 p2) (HSMul.hSMul r (VSub.vsub p1 p2))
    ⊢ Eq (EuclideanGeometry.angle p2 p1 p3) 0
  -/
  unfold angle
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne (VSub.vsub p1 p2) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p3 p2) (HSMul.hSMul r (VSub.vsub p1 p2))
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p2 p1) (VSub.vsub p3 p1)) 0
  -/
  rw [angle_eq_zero_iff]
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne (VSub.vsub p1 p2) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p3 p2) (HSMul.hSMul r (VSub.vsub p1 p2))
    ⊢ And (Ne (VSub.vsub p2 p1) 0) (Exists fun r => And (LT.lt 0 r) (Eq (VSub.vsub …
  -/
  rw [← neg_vsub_eq_vsub_rev, neg_ne_zero] at hp1p2
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne (VSub.vsub p2 p1) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p3 p2) (HSMul.hSMul r (VSub.vsub p1 p2))
    ⊢ And (Ne (VSub.vsub p2 p1) 0) (Exists fun r => And (LT.lt 0 r) (Eq (VSub.vsub …
  -/
  use hp1p2, -r + 1, add_pos (neg_pos_of_neg hr) zero_lt_one
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne (VSub.vsub p2 p1) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p3 p2) (HSMul.hSMul r (VSub.vsub p1 p2))
    ⊢ Eq (VSub.vsub p3 p1) (HSMul.hSMul (HAdd.hAdd (Neg.neg r) 1) (VSub.vsub p2 p1))
  -/
  rw [add_smul, ← neg_vsub_eq_vsub_rev p1 p2, smul_neg]
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne (VSub.vsub p2 p1) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p3 p2) (HSMul.hSMul r (VSub.vsub p1 p2))
    ⊢ Eq (VSub.vsub p3 p1) (HAdd.hAdd (Neg.neg (HSMul.hSMul (Neg.neg r) (VSub.vsub …
  -/
  simp [← hpr]
  /-
    🎉 no goals
  -/


/-- If the angle ∠ABC at a point is π, the angle ∠BCA is 0. -/
theorem angle_eq_zero_of_angle_eq_pi_right {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = π) :
    ∠ p2 p3 p1 = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    ⊢ Eq (EuclideanGeometry.angle p2 p3 p1) 0
  -/
  rw [angle_comm] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p3 p2 p1) Real.pi
    ⊢ Eq (EuclideanGeometry.angle p2 p3 p1) 0
  -/
  exact angle_eq_zero_of_angle_eq_pi_left h
  /-
    🎉 no goals
  -/


/-- If ∠BCD = π, then ∠ABC = ∠ABD. -/
theorem angle_eq_angle_of_angle_eq_pi (p1 : P) {p2 p3 p4 : P} (h : ∠ p2 p3 p4 = π) :
    ∠ p1 p2 p3 = ∠ p1 p2 p4 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (EuclideanGeometry.angle p2 p3 p4) Real.pi
    ⊢ Eq (EuclideanGeometry.angle p1 p2 p3) (EuclideanGeometry.angle p1 p2 p4)
  -/
  unfold angle at *
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) (InnerPr …
  -/
  rcases angle_eq_pi_iff.1 h with ⟨_, ⟨r, ⟨hr, hpr⟩⟩⟩
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    left✝ : Ne (VSub.vsub p2 p3) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p4 p3) (HSMul.hSMul r (VSub.vsub p2 p3))
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p3 p2)) (InnerPr …
  -/
  rw [eq_comm]
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    left✝ : Ne (VSub.vsub p2 p3) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p4 p3) (HSMul.hSMul r (VSub.vsub p2 p3))
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p4 p2)) (InnerPr …
  -/
  convert angle_smul_right_of_pos (p1 -ᵥ p2) (p3 -ᵥ p2) (add_pos (neg_pos_of_neg hr) zero_lt_one)
  /-
    case h.e'_2.h.e'_5
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    left✝ : Ne (VSub.vsub p2 p3) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p4 p3) (HSMul.hSMul r (VSub.vsub p2 p3))
    ⊢ Eq (VSub.vsub p4 p2) (HSMul.hSMul (HAdd.hAdd (Neg.neg r) 1) (VSub.vsub p3 p2))
  -/
  rw [add_smul, ← neg_vsub_eq_vsub_rev p2 p3, smul_neg, neg_smul, ← hpr]
  /-
    case h.e'_2.h.e'_5
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    left✝ : Ne (VSub.vsub p2 p3) 0
    r : Real
    hr : LT.lt r 0
    hpr : Eq (VSub.vsub p4 p3) (HSMul.hSMul r (VSub.vsub p2 p3))
    ⊢ Eq (VSub.vsub p4 p2) (HAdd.hAdd (Neg.neg (Neg.neg (VSub.vsub p4 p3))) (HSMul …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If ∠BCD = π, then ∠ACB + ∠ACD = π. -/
nonrec theorem angle_add_angle_eq_pi_of_angle_eq_pi (p1 : P) {p2 p3 p4 : P} (h : ∠ p2 p3 p4 = π) :
    ∠ p1 p3 p2 + ∠ p1 p3 p4 = π := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (EuclideanGeometry.angle p2 p3 p4) Real.pi
    ⊢ Eq (HAdd.hAdd (EuclideanGeometry.angle p1 p3 p2) (EuclideanGeometry.angle p1 …
  -/
  unfold angle at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    ⊢ Eq (HAdd.hAdd (EuclideanGeometry.angle p1 p3 p2) (EuclideanGeometry.angle p1 …
  -/
  rw [angle_comm p1 p3 p2, angle_comm p1 p3 p4]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    ⊢ Eq (HAdd.hAdd (EuclideanGeometry.angle p2 p3 p1) (EuclideanGeometry.angle p4 …
  -/
  unfold angle
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 p4 : P
    h : Eq (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p4 p3)) Real.pi
    ⊢ Eq (HAdd.hAdd (InnerProductGeometry.angle (VSub.vsub p2 p3) (VSub.vsub p1 p3 …
  -/
  exact angle_add_angle_eq_pi_of_angle_eq_pi _ h
  /-
    🎉 no goals
  -/


/-- **Vertical Angles Theorem**: angles opposite each other, formed by two intersecting straight
lines, are equal. -/
theorem angle_eq_angle_of_angle_eq_pi_of_angle_eq_pi {p1 p2 p3 p4 p5 : P} (hapc : ∠ p1 p5 p3 = π)
    (hbpd : ∠ p2 p5 p4 = π) : ∠ p1 p5 p2 = ∠ p3 p5 p4 := by
  linarith [angle_add_angle_eq_pi_of_angle_eq_pi p1 hbpd, angle_comm p4 p5 p1,
    angle_add_angle_eq_pi_of_angle_eq_pi p4 hapc, angle_comm p4 p5 p3]


/-- If ∠ABC = π then dist A B ≠ 0. -/
theorem left_dist_ne_zero_of_angle_eq_pi {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = π) : dist p1 p2 ≠ 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    ⊢ Ne (Dist.dist p1 p2) 0
  -/
  by_contra heq
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    heq : Eq (Dist.dist p1 p2) 0
    ⊢ False
  -/
  rw [dist_eq_zero] at heq
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    heq : Eq p1 p2
    ⊢ False
  -/
  rw [heq, angle_self_left] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (HDiv.hDiv Real.pi 2) Real.pi
    heq : Eq p1 p2
    ⊢ False
  -/
  exact Real.pi_ne_zero (by linarith)
  /-
    🎉 no goals
  -/


/-- If ∠ABC = π then dist C B ≠ 0. -/
theorem right_dist_ne_zero_of_angle_eq_pi {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = π) : dist p3 p2 ≠ 0 :=
  left_dist_ne_zero_of_angle_eq_pi <| (angle_comm _ _ _).trans h


/-- If ∠ABC = π, then (dist A C) = (dist A B) + (dist B C). -/
theorem dist_eq_add_dist_of_angle_eq_pi {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = π) :
    dist p1 p3 = dist p1 p2 + dist p3 p2 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    ⊢ Eq (Dist.dist p1 p3) (HAdd.hAdd (Dist.dist p1 p2) (Dist.dist p3 p2))
  -/
  rw [dist_eq_norm_vsub V, dist_eq_norm_vsub V, dist_eq_norm_vsub V, ← vsub_sub_vsub_cancel_right]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) Real.pi
    ⊢ Eq (Norm.norm (HSub.hSub (VSub.vsub p1 ?p₃) (VSub.vsub p3 ?p₃))) (HAdd.hAdd  …
  -/
  exact norm_sub_eq_add_norm_of_angle_eq_pi h
  /-
    🎉 no goals
  -/


/-- If A ≠ B and C ≠ B then ∠ABC = π if and only if (dist A C) = (dist A B) + (dist B C). -/
theorem dist_eq_add_dist_iff_angle_eq_pi {p1 p2 p3 : P} (hp1p2 : p1 ≠ p2) (hp3p2 : p3 ≠ p2) :
    dist p1 p3 = dist p1 p2 + dist p3 p2 ↔ ∠ p1 p2 p3 = π := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne p1 p2
    hp3p2 : Ne p3 p2
    ⊢ Iff (Eq (Dist.dist p1 p3) (HAdd.hAdd (Dist.dist p1 p2) (Dist.dist p3 p2))) ( …
  -/
  rw [dist_eq_norm_vsub V, dist_eq_norm_vsub V, dist_eq_norm_vsub V, ← vsub_sub_vsub_cancel_right]
  exact
    norm_sub_eq_add_norm_iff_angle_eq_pi (fun he => hp1p2 (vsub_eq_zero_iff_eq.1 he)) fun he =>
      hp3p2 (vsub_eq_zero_iff_eq.1 he)


/-- If ∠ABC = 0, then (dist A C) = abs ((dist A B) - (dist B C)). -/
theorem dist_eq_abs_sub_dist_of_angle_eq_zero {p1 p2 p3 : P} (h : ∠ p1 p2 p3 = 0) :
    dist p1 p3 = |dist p1 p2 - dist p3 p2| := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) 0
    ⊢ Eq (Dist.dist p1 p3) (abs (HSub.hSub (Dist.dist p1 p2) (Dist.dist p3 p2)))
  -/
  rw [dist_eq_norm_vsub V, dist_eq_norm_vsub V, dist_eq_norm_vsub V, ← vsub_sub_vsub_cancel_right]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (EuclideanGeometry.angle p1 p2 p3) 0
    ⊢ Eq (Norm.norm (HSub.hSub (VSub.vsub p1 ?p₃) (VSub.vsub p3 ?p₃))) (abs (HSub. …
  -/
  exact norm_sub_eq_abs_sub_norm_of_angle_eq_zero h
  /-
    🎉 no goals
  -/


/-- If A ≠ B and C ≠ B then ∠ABC = 0 if and only if (dist A C) = abs ((dist A B) - (dist B C)). -/
theorem dist_eq_abs_sub_dist_iff_angle_eq_zero {p1 p2 p3 : P} (hp1p2 : p1 ≠ p2) (hp3p2 : p3 ≠ p2) :
    dist p1 p3 = |dist p1 p2 - dist p3 p2| ↔ ∠ p1 p2 p3 = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    hp1p2 : Ne p1 p2
    hp3p2 : Ne p3 p2
    ⊢ Iff (Eq (Dist.dist p1 p3) (abs (HSub.hSub (Dist.dist p1 p2) (Dist.dist p3 p2 …
  -/
  rw [dist_eq_norm_vsub V, dist_eq_norm_vsub V, dist_eq_norm_vsub V, ← vsub_sub_vsub_cancel_right]
  exact
    norm_sub_eq_abs_sub_norm_iff_angle_eq_zero (fun he => hp1p2 (vsub_eq_zero_iff_eq.1 he))
      fun he => hp3p2 (vsub_eq_zero_iff_eq.1 he)


/-- If M is the midpoint of the segment AB, then ∠AMB = π. -/
theorem angle_midpoint_eq_pi (p1 p2 : P) (hp1p2 : p1 ≠ p2) : ∠ p1 (midpoint ℝ p1 p2) p2 = π := by
  simp only [angle, left_vsub_midpoint, invOf_eq_inv, right_vsub_midpoint, inv_pos, zero_lt_two,
    angle_smul_right_of_pos, angle_smul_left_of_pos]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 : P
    hp1p2 : Ne p1 p2
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (VSub.vsub p2 p1)) Real.pi
  -/
  rw [← neg_vsub_eq_vsub_rev p1 p2]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 : P
    hp1p2 : Ne p1 p2
    ⊢ Eq (InnerProductGeometry.angle (VSub.vsub p1 p2) (Neg.neg (VSub.vsub p1 p2)) …
  -/
  apply angle_self_neg_of_nonzero
  /-
    case hx
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 : P
    hp1p2 : Ne p1 p2
    ⊢ Ne (VSub.vsub p1 p2) 0
  -/
  simpa only [ne_eq, vsub_eq_zero_iff_eq]
  /-
    🎉 no goals
  -/


/-- If M is the midpoint of the segment AB and C is the same distance from A as it is from B
then ∠CMA = π / 2. -/
theorem angle_left_midpoint_eq_pi_div_two_of_dist_eq {p1 p2 p3 : P} (h : dist p3 p1 = dist p3 p2) :
    ∠ p3 (midpoint ℝ p1 p2) p1 = π / 2 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Dist.dist p3 p1) (Dist.dist p3 p2)
    ⊢ Eq (EuclideanGeometry.angle p3 (midpoint Real p1 p2) p1) (HDiv.hDiv Real.pi 2)
  -/
  let m : P := midpoint ℝ p1 p2
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Dist.dist p3 p1) (Dist.dist p3 p2)
    m : P := midpoint Real p1 p2
    ⊢ Eq (EuclideanGeometry.angle p3 (midpoint Real p1 p2) p1) (HDiv.hDiv Real.pi 2)
  -/
  have h1 : p3 -ᵥ p1 = p3 -ᵥ m - (p1 -ᵥ m) := (vsub_sub_vsub_cancel_right p3 p1 m).symm
  have h2 : p3 -ᵥ p2 = p3 -ᵥ m + (p1 -ᵥ m) := by
    rw [left_vsub_midpoint, ← midpoint_vsub_right, vsub_add_vsub_cancel]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Dist.dist p3 p1) (Dist.dist p3 p2)
    m : P := midpoint Real p1 p2
    h1 : Eq (VSub.vsub p3 p1) (HSub.hSub (VSub.vsub p3 m) (VSub.vsub p1 m))
    h2 : Eq (VSub.vsub p3 p2) (HAdd.hAdd (VSub.vsub p3 m) (VSub.vsub p1 m))
    ⊢ Eq (EuclideanGeometry.angle p3 (midpoint Real p1 p2) p1) (HDiv.hDiv Real.pi 2)
  -/
  rw [dist_eq_norm_vsub V p3 p1, dist_eq_norm_vsub V p3 p2, h1, h2] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    m : P := midpoint Real p1 p2
    h : Eq (Norm.norm (HSub.hSub (VSub.vsub p3 m) (VSub.vsub p1 m))) (Norm.norm (H …
    h1 : Eq (VSub.vsub p3 p1) (HSub.hSub (VSub.vsub p3 m) (VSub.vsub p1 m))
    h2 : Eq (VSub.vsub p3 p2) (HAdd.hAdd (VSub.vsub p3 m) (VSub.vsub p1 m))
    ⊢ Eq (EuclideanGeometry.angle p3 (midpoint Real p1 p2) p1) (HDiv.hDiv Real.pi 2)
  -/
  exact (norm_add_eq_norm_sub_iff_angle_eq_pi_div_two (p3 -ᵥ m) (p1 -ᵥ m)).mp h.symm
  /-
    🎉 no goals
  -/


/-- If M is the midpoint of the segment AB and C is the same distance from A as it is from B
then ∠CMB = π / 2. -/
theorem angle_right_midpoint_eq_pi_div_two_of_dist_eq {p1 p2 p3 : P} (h : dist p3 p1 = dist p3 p2) :
    ∠ p3 (midpoint ℝ p1 p2) p2 = π / 2 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 p3 : P
    h : Eq (Dist.dist p3 p1) (Dist.dist p3 p2)
    ⊢ Eq (EuclideanGeometry.angle p3 (midpoint Real p1 p2) p2) (HDiv.hDiv Real.pi 2)
  -/
  rw [midpoint_comm p1 p2, angle_left_midpoint_eq_pi_div_two_of_dist_eq h.symm]
  /-
    🎉 no goals
  -/


/-- If the second of three points is strictly between the other two, the angle at that point
is π. -/
theorem _root_.Sbtw.angle₁₂₃_eq_pi {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) : ∠ p₁ p₂ p₃ = π := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Sbtw Real p₁ p₂ p₃
    ⊢ Eq (EuclideanGeometry.angle p₁ p₂ p₃) Real.pi
  -/
  rw [angle, angle_eq_pi_iff]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Sbtw Real p₁ p₂ p₃
    ⊢ And (Ne (VSub.vsub p₁ p₂) 0) (Exists fun r => And (LT.lt r 0) (Eq (VSub.vsub …
  -/
  rcases h with ⟨⟨r, ⟨hr0, hr1⟩, hp₂⟩, hp₂p₁, hp₂p₃⟩
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    r : Real
    hp₂ : Eq ((AffineMap.lineMap p₁ p₃) r) p₂
    hr0 : LE.le 0 r
    hr1 : LE.le r 1
    hp₂p₁ : Ne p₂ p₁
    hp₂p₃ : Ne p₂ p₃
    ⊢ And (Ne (VSub.vsub p₁ p₂) 0) (Exists fun r => And (LT.lt r 0) (Eq (VSub.vsub …
  -/
  refine ⟨vsub_ne_zero.2 hp₂p₁.symm, -(1 - r) / r, ?_⟩
  have hr0' : r ≠ 0 := by
    rintro rfl
    rw [← hp₂] at hp₂p₁
    simp at hp₂p₁
  have hr1' : r ≠ 1 := by
    rintro rfl
    rw [← hp₂] at hp₂p₃
    simp at hp₂p₃
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    r : Real
    hp₂ : Eq ((AffineMap.lineMap p₁ p₃) r) p₂
    hr0 : LE.le 0 r
    hr1 : LE.le r 1
    hp₂p₁ : Ne p₂ p₁
    hp₂p₃ : Ne p₂ p₃
    hr0' : Ne r 0
    hr1' : Ne r 1
    ⊢ And (LT.lt (HDiv.hDiv (Neg.neg (HSub.hSub 1 r)) r) 0) (Eq (VSub.vsub p₃ p₂)  …
  -/
  replace hr0 := hr0.lt_of_ne hr0'.symm
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    r : Real
    hp₂ : Eq ((AffineMap.lineMap p₁ p₃) r) p₂
    hr1 : LE.le r 1
    hp₂p₁ : Ne p₂ p₁
    hp₂p₃ : Ne p₂ p₃
    hr0' : Ne r 0
    hr1' : Ne r 1
    hr0 : LT.lt 0 r
    ⊢ And (LT.lt (HDiv.hDiv (Neg.neg (HSub.hSub 1 r)) r) 0) (Eq (VSub.vsub p₃ p₂)  …
  -/
  replace hr1 := hr1.lt_of_ne hr1'
  /-
    case intro.intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    r : Real
    hp₂ : Eq ((AffineMap.lineMap p₁ p₃) r) p₂
    hp₂p₁ : Ne p₂ p₁
    hp₂p₃ : Ne p₂ p₃
    hr0' : Ne r 0
    hr1' : Ne r 1
    hr0 : LT.lt 0 r
    hr1 : LT.lt r 1
    ⊢ And (LT.lt (HDiv.hDiv (Neg.neg (HSub.hSub 1 r)) r) 0) (Eq (VSub.vsub p₃ p₂)  …
  -/
  refine ⟨div_neg_of_neg_of_pos (Left.neg_neg_iff.2 (sub_pos.2 hr1)) hr0, ?_⟩
  rw [← hp₂, AffineMap.lineMap_apply, vsub_vadd_eq_vsub_sub, vsub_vadd_eq_vsub_sub, vsub_self,
    zero_sub, smul_neg, smul_smul, div_mul_cancel₀ _ hr0', neg_smul, neg_neg, sub_eq_iff_eq_add, ←
    add_smul, sub_add_cancel, one_smul]


/-- If the second of three points is strictly between the other two, the angle at that point
(reversed) is π. -/
theorem _root_.Sbtw.angle₃₂₁_eq_pi {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) : ∠ p₃ p₂ p₁ = π := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Sbtw Real p₁ p₂ p₃
    ⊢ Eq (EuclideanGeometry.angle p₃ p₂ p₁) Real.pi
  -/
  rw [← h.angle₁₂₃_eq_pi, angle_comm]
  /-
    🎉 no goals
  -/


/-- The angle between three points is π if and only if the second point is strictly between the
other two. -/
theorem angle_eq_pi_iff_sbtw {p₁ p₂ p₃ : P} : ∠ p₁ p₂ p₃ = π ↔ Sbtw ℝ p₁ p₂ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (Eq (EuclideanGeometry.angle p₁ p₂ p₃) Real.pi) (Sbtw Real p₁ p₂ p₃)
  -/
  refine ⟨?_, fun h => h.angle₁₂₃_eq_pi⟩
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Eq (EuclideanGeometry.angle p₁ p₂ p₃) Real.pi → Sbtw Real p₁ p₂ p₃
  -/
  rw [angle, angle_eq_pi_iff]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ And (Ne (VSub.vsub p₁ p₂) 0) (Exists fun r => And (LT.lt r 0) (Eq (VSub.vsub …
  -/
  rintro ⟨hp₁p₂, r, hr, hp₃p₂⟩
  refine ⟨⟨1 / (1 - r), ⟨div_nonneg zero_le_one (sub_nonneg.2 (hr.le.trans zero_le_one)),
    (div_le_one (sub_pos.2 (hr.trans zero_lt_one))).2 ((le_sub_self_iff 1).2 hr.le)⟩, ?_⟩,
    (vsub_ne_zero.1 hp₁p₂).symm, ?_⟩
    /-
      case intro.intro.intro.refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr : LT.lt r 0
      hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
      ⊢ Eq ((AffineMap.lineMap p₁ p₃) (HDiv.hDiv 1 (HSub.hSub 1 r))) p₂
    -/
  · rw [← eq_vadd_iff_vsub_eq] at hp₃p₂
    rw [AffineMap.lineMap_apply, hp₃p₂, vadd_vsub_assoc, ← neg_vsub_eq_vsub_rev p₂ p₁, smul_neg, ←
      neg_smul, smul_add, smul_smul, ← add_smul, eq_comm, eq_vadd_iff_vsub_eq]
    /-
      case intro.intro.intro.refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr : LT.lt r 0
      hp₃p₂ : Eq p₃ (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₁ p₂)) p₂)
      ⊢ Eq (VSub.vsub p₂ p₁) (HSMul.hSMul (HAdd.hAdd (HMul.hMul (HDiv.hDiv 1 (HSub.h …
    -/
    convert (one_smul ℝ (p₂ -ᵥ p₁)).symm
    /-
      case h.e'_3.h.e'_5
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr : LT.lt r 0
      hp₃p₂ : Eq p₃ (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₁ p₂)) p₂)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv 1 (HSub.hSub 1 r)) (Neg.neg r)) (HDiv.hD …
    -/
    field_simp [(sub_pos.2 (hr.trans zero_lt_one)).ne.symm]
    /-
      case h.e'_3.h.e'_5
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr : LT.lt r 0
      hp₃p₂ : Eq p₃ (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p₁ p₂)) p₂)
      ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul r (HSub.hSub 1 r))) (HSub.hSub 1 r)) (HMul …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr : LT.lt r 0
      hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
      ⊢ Ne p₂ p₃
    -/
  · rw [ne_comm, ← @vsub_ne_zero V, hp₃p₂, smul_ne_zero_iff]
    /-
      case intro.intro.intro.refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr : LT.lt r 0
      hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
      ⊢ And (Ne r 0) (Ne (VSub.vsub p₁ p₂) 0)
    -/
    exact ⟨hr.ne, hp₁p₂⟩
    /-
      🎉 no goals
    -/


/-- If the second of three points is weakly between the other two, and not equal to the first,
the angle at the first point is zero. -/
theorem _root_.Wbtw.angle₂₁₃_eq_zero_of_ne {p₁ p₂ p₃ : P} (h : Wbtw ℝ p₁ p₂ p₃) (hp₂p₁ : p₂ ≠ p₁) :
    ∠ p₂ p₁ p₃ = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Wbtw Real p₁ p₂ p₃
    hp₂p₁ : Ne p₂ p₁
    ⊢ Eq (EuclideanGeometry.angle p₂ p₁ p₃) 0
  -/
  rw [angle, angle_eq_zero_iff]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Wbtw Real p₁ p₂ p₃
    hp₂p₁ : Ne p₂ p₁
    ⊢ And (Ne (VSub.vsub p₂ p₁) 0) (Exists fun r => And (LT.lt 0 r) (Eq (VSub.vsub …
  -/
  rcases h with ⟨r, ⟨hr0, hr1⟩, rfl⟩
  have hr0' : r ≠ 0 := by
    rintro rfl
    simp at hp₂p₁
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₃ : P
    r : Real
    hr0 : LE.le 0 r
    hr1 : LE.le r 1
    hp₂p₁ : Ne ((AffineMap.lineMap p₁ p₃) r) p₁
    hr0' : Ne r 0
    ⊢ And (Ne (VSub.vsub ((AffineMap.lineMap p₁ p₃) r) p₁) 0) (Exists fun r_1 => A …
  -/
  replace hr0 := hr0.lt_of_ne hr0'.symm
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₃ : P
    r : Real
    hr1 : LE.le r 1
    hp₂p₁ : Ne ((AffineMap.lineMap p₁ p₃) r) p₁
    hr0' : Ne r 0
    hr0 : LT.lt 0 r
    ⊢ And (Ne (VSub.vsub ((AffineMap.lineMap p₁ p₃) r) p₁) 0) (Exists fun r_1 => A …
  -/
  refine ⟨vsub_ne_zero.2 hp₂p₁, r⁻¹, inv_pos.2 hr0, ?_⟩
  rw [AffineMap.lineMap_apply, vadd_vsub_assoc, vsub_self, add_zero, smul_smul,
    inv_mul_cancel₀ hr0', one_smul]


/-- If the second of three points is strictly between the other two, the angle at the first point
is zero. -/
theorem _root_.Sbtw.angle₂₁₃_eq_zero {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) : ∠ p₂ p₁ p₃ = 0 :=
  h.wbtw.angle₂₁₃_eq_zero_of_ne h.ne_left


/-- If the second of three points is weakly between the other two, and not equal to the first,
the angle at the first point (reversed) is zero. -/
theorem _root_.Wbtw.angle₃₁₂_eq_zero_of_ne {p₁ p₂ p₃ : P} (h : Wbtw ℝ p₁ p₂ p₃) (hp₂p₁ : p₂ ≠ p₁) :
                         /-
                           V : Type u_1
                           P : Type u_2
                           inst✝³ : NormedAddCommGroup V
                           inst✝² : InnerProductSpace Real V
                           inst✝¹ : MetricSpace P
                           inst✝ : NormedAddTorsor V P
                           p₁ p₂ p₃ : P
                           h : Wbtw Real p₁ p₂ p₃
                           hp₂p₁ : Ne p₂ p₁
                           ⊢ Eq (EuclideanGeometry.angle p₃ p₁ p₂) 0
                         -/
    ∠ p₃ p₁ p₂ = 0 := by rw [← h.angle₂₁₃_eq_zero_of_ne hp₂p₁, angle_comm]
                         /-
                           🎉 no goals
                         -/


/-- If the second of three points is strictly between the other two, the angle at the first point
(reversed) is zero. -/
theorem _root_.Sbtw.angle₃₁₂_eq_zero {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) : ∠ p₃ p₁ p₂ = 0 :=
  h.wbtw.angle₃₁₂_eq_zero_of_ne h.ne_left


/-- If the second of three points is weakly between the other two, and not equal to the third,
the angle at the third point is zero. -/
theorem _root_.Wbtw.angle₂₃₁_eq_zero_of_ne {p₁ p₂ p₃ : P} (h : Wbtw ℝ p₁ p₂ p₃) (hp₂p₃ : p₂ ≠ p₃) :
    ∠ p₂ p₃ p₁ = 0 :=
  h.symm.angle₂₁₃_eq_zero_of_ne hp₂p₃


/-- If the second of three points is strictly between the other two, the angle at the third point
is zero. -/
theorem _root_.Sbtw.angle₂₃₁_eq_zero {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) : ∠ p₂ p₃ p₁ = 0 :=
  h.wbtw.angle₂₃₁_eq_zero_of_ne h.ne_right


/-- If the second of three points is weakly between the other two, and not equal to the third,
the angle at the third point (reversed) is zero. -/
theorem _root_.Wbtw.angle₁₃₂_eq_zero_of_ne {p₁ p₂ p₃ : P} (h : Wbtw ℝ p₁ p₂ p₃) (hp₂p₃ : p₂ ≠ p₃) :
    ∠ p₁ p₃ p₂ = 0 :=
  h.symm.angle₃₁₂_eq_zero_of_ne hp₂p₃


/-- If the second of three points is strictly between the other two, the angle at the third point
(reversed) is zero. -/
theorem _root_.Sbtw.angle₁₃₂_eq_zero {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) : ∠ p₁ p₃ p₂ = 0 :=
  h.wbtw.angle₁₃₂_eq_zero_of_ne h.ne_right


/-- The angle between three points is zero if and only if one of the first and third points is
weakly between the other two, and not equal to the second. -/
theorem angle_eq_zero_iff_ne_and_wbtw {p₁ p₂ p₃ : P} :
    ∠ p₁ p₂ p₃ = 0 ↔ p₁ ≠ p₂ ∧ Wbtw ℝ p₂ p₁ p₃ ∨ p₃ ≠ p₂ ∧ Wbtw ℝ p₂ p₃ p₁ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) (Or (And (Ne p₁ p₂) (Wbtw Real …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      ⊢ Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0 → Or (And (Ne p₁ p₂) (Wbtw Real p₂ p …
    -/
  · rw [angle, angle_eq_zero_iff]
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      ⊢ And (Ne (VSub.vsub p₁ p₂) 0) (Exists fun r => And (LT.lt 0 r) (Eq (VSub.vsub …
    -/
    rintro ⟨hp₁p₂, r, hr0, hp₃p₂⟩
    /-
      case mp.intro.intro.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
      r : Real
      hr0 : LT.lt 0 r
      hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
      ⊢ Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂ p₃ p₁))
    -/
    rcases le_or_lt 1 r with (hr1 | hr1)
      /-
        case mp.intro.intro.intro.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
        r : Real
        hr0 : LT.lt 0 r
        hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
        hr1 : LE.le 1 r
        ⊢ Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂ p₃ p₁))
      -/
    · refine Or.inl ⟨vsub_ne_zero.1 hp₁p₂, r⁻¹, ⟨(inv_pos.2 hr0).le, inv_le_one_of_one_le₀ hr1⟩, ?_⟩
      rw [AffineMap.lineMap_apply, hp₃p₂, smul_smul, inv_mul_cancel₀ hr0.ne.symm, one_smul,
        vsub_vadd]
      /-
        case mp.intro.intro.intro.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
        r : Real
        hr0 : LT.lt 0 r
        hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
        hr1 : LT.lt r 1
        ⊢ Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂ p₃ p₁))
      -/
    · refine Or.inr ⟨?_, r, ⟨hr0.le, hr1.le⟩, ?_⟩
        /-
          case mp.intro.intro.intro.inr.refine_1
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
          r : Real
          hr0 : LT.lt 0 r
          hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
          hr1 : LT.lt r 1
          ⊢ Ne p₃ p₂
        -/
      · rw [← @vsub_ne_zero V, hp₃p₂, smul_ne_zero_iff]
        /-
          case mp.intro.intro.intro.inr.refine_1
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
          r : Real
          hr0 : LT.lt 0 r
          hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
          hr1 : LT.lt r 1
          ⊢ And (Ne r 0) (Ne (VSub.vsub p₁ p₂) 0)
        -/
        exact ⟨hr0.ne.symm, hp₁p₂⟩
        /-
          🎉 no goals
        -/
        /-
          case mp.intro.intro.intro.inr.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          hp₁p₂ : Ne (VSub.vsub p₁ p₂) 0
          r : Real
          hr0 : LT.lt 0 r
          hp₃p₂ : Eq (VSub.vsub p₃ p₂) (HSMul.hSMul r (VSub.vsub p₁ p₂))
          hr1 : LT.lt r 1
          ⊢ Eq ((AffineMap.lineMap p₂ p₁) r) p₃
        -/
      · rw [AffineMap.lineMap_apply, ← hp₃p₂, vsub_vadd]
        /-
          🎉 no goals
        -/
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      ⊢ Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂ p₃ p₁ …
    -/
  · rintro (⟨hp₁p₂, h⟩ | ⟨hp₃p₂, h⟩)
      /-
        case mpr.inl.intro
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        hp₁p₂ : Ne p₁ p₂
        h : Wbtw Real p₂ p₁ p₃
        ⊢ Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0
      -/
    · exact h.angle₂₁₃_eq_zero_of_ne hp₁p₂
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        hp₃p₂ : Ne p₃ p₂
        h : Wbtw Real p₂ p₃ p₁
        ⊢ Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0
      -/
    · exact h.angle₃₁₂_eq_zero_of_ne hp₃p₂
      /-
        🎉 no goals
      -/


/-- The angle between three points is zero if and only if one of the first and third points is
strictly between the other two, or those two points are equal but not equal to the second. -/
theorem angle_eq_zero_iff_eq_and_ne_or_sbtw {p₁ p₂ p₃ : P} :
    ∠ p₁ p₂ p₃ = 0 ↔ p₁ = p₃ ∧ p₁ ≠ p₂ ∨ Sbtw ℝ p₂ p₁ p₃ ∨ Sbtw ℝ p₂ p₃ p₁ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) (Or (And (Eq p₁ p₃) (Ne p₁ p₂) …
  -/
  rw [angle_eq_zero_iff_ne_and_wbtw]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂  …
  -/
  by_cases hp₁p₂ : p₁ = p₂; · simp [hp₁p₂]
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
    p₁ p₂ p₃ : P
    hp₁p₂ : Not (Eq p₁ p₂)
    ⊢ Iff (Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂  …
  -/
  by_cases hp₁p₃ : p₁ = p₃; · simp [hp₁p₃]
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
    p₁ p₂ p₃ : P
    hp₁p₂ : Not (Eq p₁ p₂)
    hp₁p₃ : Not (Eq p₁ p₃)
    ⊢ Iff (Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂  …
  -/
  by_cases hp₃p₂ : p₃ = p₂; · simp [hp₃p₂]
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
    p₁ p₂ p₃ : P
    hp₁p₂ : Not (Eq p₁ p₂)
    hp₁p₃ : Not (Eq p₁ p₃)
    hp₃p₂ : Not (Eq p₃ p₂)
    ⊢ Iff (Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂  …
  -/
  simp [hp₁p₂, hp₁p₃, Ne.symm hp₁p₃, Sbtw, hp₃p₂]
  /-
    🎉 no goals
  -/


/-- Three points are collinear if and only if the first or third point equals the second or the
angle between them is 0 or π. -/
theorem collinear_iff_eq_or_eq_or_angle_eq_zero_or_angle_eq_pi {p₁ p₂ p₃ : P} :
    Collinear ℝ ({p₁, p₂, p₃} : Set P) ↔ p₁ = p₂ ∨ p₃ = p₂ ∨ ∠ p₁ p₂ p₃ = 0 ∨ ∠ p₁ p₂ p₃ = π := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Iff (Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      ⊢ Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) ( …
    -/
  · replace h := h.wbtw_or_wbtw_or_wbtw
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      h : Or (Wbtw Real p₁ p₂ p₃) (Or (Wbtw Real p₂ p₃ p₁) (Wbtw Real p₃ p₁ p₂))
      ⊢ Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) ( …
    -/
    by_cases h₁₂ : p₁ = p₂
      /-
        case pos
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h : Or (Wbtw Real p₁ p₂ p₃) (Or (Wbtw Real p₂ p₃ p₁) (Wbtw Real p₃ p₁ p₂))
        h₁₂ : Eq p₁ p₂
        ⊢ Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) ( …
      -/
    · exact Or.inl h₁₂
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
      p₁ p₂ p₃ : P
      h : Or (Wbtw Real p₁ p₂ p₃) (Or (Wbtw Real p₂ p₃ p₁) (Wbtw Real p₃ p₁ p₂))
      h₁₂ : Not (Eq p₁ p₂)
      ⊢ Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) ( …
    -/
    by_cases h₃₂ : p₃ = p₂
      /-
        case pos
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h : Or (Wbtw Real p₁ p₂ p₃) (Or (Wbtw Real p₂ p₃ p₁) (Wbtw Real p₃ p₁ p₂))
        h₁₂ : Not (Eq p₁ p₂)
        h₃₂ : Eq p₃ p₂
        ⊢ Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) ( …
      -/
    · exact Or.inr (Or.inl h₃₂)
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
      p₁ p₂ p₃ : P
      h : Or (Wbtw Real p₁ p₂ p₃) (Or (Wbtw Real p₂ p₃ p₁) (Wbtw Real p₃ p₁ p₂))
      h₁₂ : Not (Eq p₁ p₂)
      h₃₂ : Not (Eq p₃ p₂)
      ⊢ Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) ( …
    -/
    rw [or_iff_right h₁₂, or_iff_right h₃₂]
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      h : Or (Wbtw Real p₁ p₂ p₃) (Or (Wbtw Real p₂ p₃ p₁) (Wbtw Real p₃ p₁ p₂))
      h₁₂ : Not (Eq p₁ p₂)
      h₃₂ : Not (Eq p₃ p₂)
      ⊢ Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) (Eq (EuclideanGeometry.angle p₁ …
    -/
    rcases h with (h | h | h)
      /-
        case neg.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h₁₂ : Not (Eq p₁ p₂)
        h₃₂ : Not (Eq p₃ p₂)
        h : Wbtw Real p₁ p₂ p₃
        ⊢ Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) (Eq (EuclideanGeometry.angle p₁ …
      -/
    · exact Or.inr (angle_eq_pi_iff_sbtw.2 ⟨h, Ne.symm h₁₂, Ne.symm h₃₂⟩)
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h₁₂ : Not (Eq p₁ p₂)
        h₃₂ : Not (Eq p₃ p₂)
        h : Wbtw Real p₂ p₃ p₁
        ⊢ Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) (Eq (EuclideanGeometry.angle p₁ …
      -/
    · exact Or.inl (h.angle₃₁₂_eq_zero_of_ne h₃₂)
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h₁₂ : Not (Eq p₁ p₂)
        h₃₂ : Not (Eq p₃ p₂)
        h : Wbtw Real p₃ p₁ p₂
        ⊢ Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) (Eq (EuclideanGeometry.angle p₁ …
      -/
    · exact Or.inl (h.angle₂₃₁_eq_zero_of_ne h₁₂)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      p₁ p₂ p₃ : P
      h : Or (Eq p₁ p₂) (Or (Eq p₃ p₂) (Or (Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0) …
      ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
    -/
  · rcases h with (rfl | rfl | h | h)
      /-
        case refine_2.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₃ : P
        ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₁ (Singleton.singleton p₃)))
      -/
    · simpa using collinear_pair ℝ p₁ p₃
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₃ : P
        ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₃ (Singleton.singleton p₃)))
      -/
    · simpa using collinear_pair ℝ p₁ p₃
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.inr.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h : Eq (EuclideanGeometry.angle p₁ p₂ p₃) 0
        ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
      -/
    · rw [angle_eq_zero_iff_ne_and_wbtw] at h
      /-
        case refine_2.inr.inr.inl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h : Or (And (Ne p₁ p₂) (Wbtw Real p₂ p₁ p₃)) (And (Ne p₃ p₂) (Wbtw Real p₂ p₃  …
        ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
      -/
      rcases h with (⟨-, h⟩ | ⟨-, h⟩)
        /-
          case refine_2.inr.inr.inl.inl.intro
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          h : Wbtw Real p₂ p₁ p₃
          ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
        -/
      · rw [Set.insert_comm]
        /-
          case refine_2.inr.inr.inl.inl.intro
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          h : Wbtw Real p₂ p₁ p₃
          ⊢ Collinear Real (Insert.insert p₂ (Insert.insert p₁ (Singleton.singleton p₃)))
        -/
        exact h.collinear
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr.inr.inl.inr.intro
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          h : Wbtw Real p₂ p₃ p₁
          ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
        -/
      · rw [Set.insert_comm, Set.pair_comm]
        /-
          case refine_2.inr.inr.inl.inr.intro
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          p₁ p₂ p₃ : P
          h : Wbtw Real p₂ p₃ p₁
          ⊢ Collinear Real (Insert.insert p₂ (Insert.insert p₃ (Singleton.singleton p₁)))
        -/
        exact h.collinear
        /-
          🎉 no goals
        -/
      /-
        case refine_2.inr.inr.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h : Eq (EuclideanGeometry.angle p₁ p₂ p₃) Real.pi
        ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
      -/
    · rw [angle_eq_pi_iff_sbtw] at h
      /-
        case refine_2.inr.inr.inr
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        p₁ p₂ p₃ : P
        h : Sbtw Real p₁ p₂ p₃
        ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
      -/
      exact h.wbtw.collinear
      /-
        🎉 no goals
      -/


/-- If the angle between three points is 0, they are collinear. -/
theorem collinear_of_angle_eq_zero {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = 0) :
    Collinear ℝ ({p₁, p₂, p₃} : Set P) :=
  collinear_iff_eq_or_eq_or_angle_eq_zero_or_angle_eq_pi.2 <| Or.inr <| Or.inr <| Or.inl h


/-- If the angle between three points is π, they are collinear. -/
theorem collinear_of_angle_eq_pi {p₁ p₂ p₃ : P} (h : ∠ p₁ p₂ p₃ = π) :
    Collinear ℝ ({p₁, p₂, p₃} : Set P) :=
  collinear_iff_eq_or_eq_or_angle_eq_zero_or_angle_eq_pi.2 <| Or.inr <| Or.inr <| Or.inr h


/-- If three points are not collinear, the angle between them is nonzero. -/
theorem angle_ne_zero_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear ℝ ({p₁, p₂, p₃} : Set P)) :
    ∠ p₁ p₂ p₃ ≠ 0 :=
  mt collinear_of_angle_eq_zero h


/-- If three points are not collinear, the angle between them is not π. -/
theorem angle_ne_pi_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear ℝ ({p₁, p₂, p₃} : Set P)) :
    ∠ p₁ p₂ p₃ ≠ π :=
  mt collinear_of_angle_eq_pi h


/-- If three points are not collinear, the angle between them is positive. -/
theorem angle_pos_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear ℝ ({p₁, p₂, p₃} : Set P)) :
    0 < ∠ p₁ p₂ p₃ :=
  (angle_nonneg _ _ _).lt_of_ne (angle_ne_zero_of_not_collinear h).symm


/-- If three points are not collinear, the angle between them is less than π. -/
theorem angle_lt_pi_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear ℝ ({p₁, p₂, p₃} : Set P)) :
    ∠ p₁ p₂ p₃ < π :=
  (angle_le_pi _ _ _).lt_of_ne <| angle_ne_pi_of_not_collinear h


/-- The cosine of the angle between three points is 1 if and only if the angle is 0. -/
nonrec theorem cos_eq_one_iff_angle_eq_zero {p₁ p₂ p₃ : P} :
    Real.cos (∠ p₁ p₂ p₃) = 1 ↔ ∠ p₁ p₂ p₃ = 0 :=
  cos_eq_one_iff_angle_eq_zero


/-- The cosine of the angle between three points is 0 if and only if the angle is π / 2. -/
nonrec theorem cos_eq_zero_iff_angle_eq_pi_div_two {p₁ p₂ p₃ : P} :
    Real.cos (∠ p₁ p₂ p₃) = 0 ↔ ∠ p₁ p₂ p₃ = π / 2 :=
  cos_eq_zero_iff_angle_eq_pi_div_two


/-- The cosine of the angle between three points is -1 if and only if the angle is π. -/
nonrec theorem cos_eq_neg_one_iff_angle_eq_pi {p₁ p₂ p₃ : P} :
    Real.cos (∠ p₁ p₂ p₃) = -1 ↔ ∠ p₁ p₂ p₃ = π :=
  cos_eq_neg_one_iff_angle_eq_pi


/-- The sine of the angle between three points is 0 if and only if the angle is 0 or π. -/
nonrec theorem sin_eq_zero_iff_angle_eq_zero_or_angle_eq_pi {p₁ p₂ p₃ : P} :
    Real.sin (∠ p₁ p₂ p₃) = 0 ↔ ∠ p₁ p₂ p₃ = 0 ∨ ∠ p₁ p₂ p₃ = π :=
  sin_eq_zero_iff_angle_eq_zero_or_angle_eq_pi


/-- The sine of the angle between three points is 1 if and only if the angle is π / 2. -/
nonrec theorem sin_eq_one_iff_angle_eq_pi_div_two {p₁ p₂ p₃ : P} :
    Real.sin (∠ p₁ p₂ p₃) = 1 ↔ ∠ p₁ p₂ p₃ = π / 2 :=
  sin_eq_one_iff_angle_eq_pi_div_two


/-- Three points are collinear if and only if the first or third point equals the second or
the sine of the angle between three points is zero. -/
theorem collinear_iff_eq_or_eq_or_sin_eq_zero {p₁ p₂ p₃ : P} :
    Collinear ℝ ({p₁, p₂, p₃} : Set P) ↔ p₁ = p₂ ∨ p₃ = p₂ ∨ Real.sin (∠ p₁ p₂ p₃) = 0 := by
  rw [sin_eq_zero_iff_angle_eq_zero_or_angle_eq_pi,
    collinear_iff_eq_or_eq_or_angle_eq_zero_or_angle_eq_pi]


/-- If three points are not collinear, the sine of the angle between them is positive. -/
theorem sin_pos_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear ℝ ({p₁, p₂, p₃} : Set P)) :
    0 < Real.sin (∠ p₁ p₂ p₃) :=
  Real.sin_pos_of_pos_of_lt_pi (angle_pos_of_not_collinear h) (angle_lt_pi_of_not_collinear h)


/-- If three points are not collinear, the sine of the angle between them is nonzero. -/
theorem sin_ne_zero_of_not_collinear {p₁ p₂ p₃ : P} (h : ¬Collinear ℝ ({p₁, p₂, p₃} : Set P)) :
    Real.sin (∠ p₁ p₂ p₃) ≠ 0 :=
  ne_of_gt (sin_pos_of_not_collinear h)


/-- If the sine of the angle between three points is 0, they are collinear. -/
theorem collinear_of_sin_eq_zero {p₁ p₂ p₃ : P} (h : Real.sin (∠ p₁ p₂ p₃) = 0) :
    Collinear ℝ ({p₁, p₂, p₃} : Set P) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    h : Eq (Real.sin (EuclideanGeometry.angle p₁ p₂ p₃)) 0
    ⊢ Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃)))
  -/
  revert h
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Eq (Real.sin (EuclideanGeometry.angle p₁ p₂ p₃)) 0 → Collinear Real (Insert. …
  -/
  contrapose
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p₁ p₂ p₃ : P
    ⊢ Not (Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton …
  -/
  exact sin_ne_zero_of_not_collinear
  /-
    🎉 no goals
  -/


