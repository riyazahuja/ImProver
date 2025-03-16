/-- The Monge point of a simplex (in 2 or more dimensions) is a
generalization of the orthocenter of a triangle.  It is defined to be
the intersection of the Monge planes, where a Monge plane is the
(n-1)-dimensional affine subspace of the subspace spanned by the
simplex that passes through the centroid of an (n-2)-dimensional face
and is orthogonal to the opposite edge (in 2 dimensions, this is the
same as an altitude).  The circumcenter O, centroid G and Monge point
M are collinear in that order on the Euler line, with OG : GM = (n-1): 2.
Here, we use that ratio to define the Monge point (so resulting
in a point that equals the centroid in 0 or 1 dimensions), and then
show in subsequent lemmas that the point so defined lies in the Monge
planes and is their unique point of intersection. -/
def mongePoint {n : ℕ} (s : Simplex ℝ P n) : P :=
  (((n + 1 : ℕ) : ℝ) / ((n - 1 : ℕ) : ℝ)) •
      ((univ : Finset (Fin (n + 1))).centroid ℝ s.points -ᵥ s.circumcenter) +ᵥ
    s.circumcenter


/-- The position of the Monge point in relation to the circumcenter
and centroid. -/
theorem mongePoint_eq_smul_vsub_vadd_circumcenter {n : ℕ} (s : Simplex ℝ P n) :
    s.mongePoint =
      (((n + 1 : ℕ) : ℝ) / ((n - 1 : ℕ) : ℝ)) •
          ((univ : Finset (Fin (n + 1))).centroid ℝ s.points -ᵥ s.circumcenter) +ᵥ
        s.circumcenter :=
  rfl


/-- The Monge point lies in the affine span. -/
theorem mongePoint_mem_affineSpan {n : ℕ} (s : Simplex ℝ P n) :
    s.mongePoint ∈ affineSpan ℝ (Set.range s.points) :=
  smul_vsub_vadd_mem _ _ (centroid_mem_affineSpan_of_card_eq_add_one ℝ _ (card_fin (n + 1)))
    s.circumcenter_mem_affineSpan s.circumcenter_mem_affineSpan


/-- Two simplices with the same points have the same Monge point. -/
theorem mongePoint_eq_of_range_eq {n : ℕ} {s₁ s₂ : Simplex ℝ P n}
    (h : Set.range s₁.points = Set.range s₂.points) : s₁.mongePoint = s₂.mongePoint := by
  simp_rw [mongePoint_eq_smul_vsub_vadd_circumcenter, centroid_eq_of_range_eq h,
    circumcenter_eq_of_range_eq h]


/-- The weights for the Monge point of an (n+2)-simplex, in terms of
`pointsWithCircumcenter`. -/
def mongePointWeightsWithCircumcenter (n : ℕ) : PointsWithCircumcenterIndex (n + 2) → ℝ
  | pointIndex _ => ((n + 1 : ℕ) : ℝ)⁻¹
  | circumcenterIndex => -2 / ((n + 1 : ℕ) : ℝ)


/-- `mongePointWeightsWithCircumcenter` sums to 1. -/
@[simp]
theorem sum_mongePointWeightsWithCircumcenter (n : ℕ) :
    ∑ i, mongePointWeightsWithCircumcenter n i = 1 := by
  simp_rw [sum_pointsWithCircumcenter, mongePointWeightsWithCircumcenter, sum_const, card_fin,
    nsmul_eq_mul]
  -- Porting note: replaced
  -- have hn1 : (n + 1 : ℝ) ≠ 0 := mod_cast Nat.succ_ne_zero _
  -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp [n.cast_add_one_ne_zero]`, but was really slow
  -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
  simp (disch := field_simp_discharge) only [Nat.cast_add, Nat.cast_ofNat, Nat.cast_one,
    inv_eq_one_div, mul_div_assoc', mul_one, add_div', div_mul_cancel₀, div_eq_iff, one_mul]
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (↑n) 2) 1) (-2)) (HAdd.hAdd (↑n) 1)
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The Monge point of an (n+2)-simplex, in terms of
`pointsWithCircumcenter`. -/
theorem mongePoint_eq_affineCombination_of_pointsWithCircumcenter {n : ℕ}
    (s : Simplex ℝ P (n + 2)) :
    s.mongePoint =
      (univ : Finset (PointsWithCircumcenterIndex (n + 2))).affineCombination ℝ
        s.pointsWithCircumcenter (mongePointWeightsWithCircumcenter n) := by
  rw [mongePoint_eq_smul_vsub_vadd_circumcenter,
    centroid_eq_affineCombination_of_pointsWithCircumcenter,
    circumcenter_eq_affineCombination_of_pointsWithCircumcenter, affineCombination_vsub,
    ← LinearMap.map_smul, weightedVSub_vadd_affineCombination]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    ⊢ Eq ((Finset.affineCombination Real Finset.univ s.pointsWithCircumcenter) (HA …
  -/
  congr with i
  /-
    case h.e_6.h.h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i : Affine.Simplex.PointsWithCircumcenterIndex (HAdd.hAdd n 2)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HSub. …
  -/
  rw [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.sub_apply]
  -- Porting note: replaced
  -- have hn1 : (n + 1 : ℝ) ≠ 0 := mod_cast Nat.succ_ne_zero _
  /-
    case h.e_6.h.h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i : Affine.Simplex.PointsWithCircumcenterIndex (HAdd.hAdd n 2)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HSub.hS …
  -/
  have hn1 : (n + 1 : ℝ) ≠ 0 := n.cast_add_one_ne_zero
  /-
    case h.e_6.h.h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i : Affine.Simplex.PointsWithCircumcenterIndex (HAdd.hAdd n 2)
    hn1 : Ne (HAdd.hAdd (↑n) 1) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HSub.hS …
  -/
  cases i <;>
      simp_rw [centroidWeightsWithCircumcenter, circumcenterWeightsWithCircumcenter,
        mongePointWeightsWithCircumcenter] <;>
    /-
      case h.e_6.h.h.pointIndex
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      hn1 : Ne (HAdd.hAdd (↑n) 1) 0
      a✝ : Fin (HAdd.hAdd (HAdd.hAdd n 2) 1)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HSub.hS …
    -/
    rw [add_tsub_assoc_of_le (by decide : 1 ≤ 2), (by decide : 2 - 1 = 1)]
    /-
      case h.e_6.h.h.pointIndex
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      hn1 : Ne (HAdd.hAdd (↑n) 1) 0
      a✝ : Fin (HAdd.hAdd (HAdd.hAdd n 2) 1)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HAdd.hA …
    -/
  · rw [if_pos (mem_univ _), sub_zero, add_zero, card_fin]
    -- Porting note: replaced
    -- have hn3 : (n + 2 + 1 : ℝ) ≠ 0 := mod_cast Nat.succ_ne_zero _
    /-
      case h.e_6.h.h.pointIndex
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      hn1 : Ne (HAdd.hAdd (↑n) 1) 0
      a✝ : Fin (HAdd.hAdd (HAdd.hAdd n 2) 1)
      ⊢ Eq (HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HAdd.hAdd n 1)) (I …
    -/
    have hn3 : (n + 2 + 1 : ℝ) ≠ 0 := by norm_cast
    /-
      case h.e_6.h.h.pointIndex
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      hn1 : Ne (HAdd.hAdd (↑n) 1) 0
      a✝ : Fin (HAdd.hAdd (HAdd.hAdd n 2) 1)
      hn3 : Ne (HAdd.hAdd (HAdd.hAdd (↑n) 2) 1) 0
      ⊢ Eq (HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd (HAdd.hAdd n 2) 1) ↑(HAdd.hAdd n 1)) (I …
    -/
    field_simp [hn1, hn3, mul_comm]
    /-
      🎉 no goals
    -/
  · -- TODO(https://github.com/leanprover-community/mathlib4/issues/15486): used to be `field_simp [hn1]`, but was really slow
  -- replaced by `simp only ...` to speed up. Reinstate `field_simp` once it is faster.
    simp (disch := field_simp_discharge) only
      [Nat.cast_add, Nat.cast_ofNat, Nat.cast_one, zero_sub, mul_neg, mul_one, neg_div',
      neg_add_rev, div_add', one_mul, eq_div_iff, div_mul_cancel₀]
    /-
      case h.e_6.h.h.circumcenterIndex
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      hn1 : Ne (HAdd.hAdd (↑n) 1) 0
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (-1) (HAdd.hAdd (Neg.neg 2) (Neg.neg ↑n))) (HAdd.hA …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The weights for the Monge point of an (n+2)-simplex, minus the
centroid of an n-dimensional face, in terms of
`pointsWithCircumcenter`.  This definition is only valid when `i₁ ≠ i₂`. -/
def mongePointVSubFaceCentroidWeightsWithCircumcenter {n : ℕ} (i₁ i₂ : Fin (n + 3)) :
    PointsWithCircumcenterIndex (n + 2) → ℝ
  | pointIndex i => if i = i₁ ∨ i = i₂ then ((n + 1 : ℕ) : ℝ)⁻¹ else 0
  | circumcenterIndex => -2 / ((n + 1 : ℕ) : ℝ)


/-- `mongePointVSubFaceCentroidWeightsWithCircumcenter` is the
result of subtracting `centroidWeightsWithCircumcenter` from
`mongePointWeightsWithCircumcenter`. -/
theorem mongePointVSubFaceCentroidWeightsWithCircumcenter_eq_sub {n : ℕ} {i₁ i₂ : Fin (n + 3)}
    (h : i₁ ≠ i₂) :
    mongePointVSubFaceCentroidWeightsWithCircumcenter i₁ i₂ =
      mongePointWeightsWithCircumcenter n - centroidWeightsWithCircumcenter {i₁, i₂}ᶜ := by
  /-
    n : Nat
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Ne i₁ i₂
    ⊢ Eq (Affine.Simplex.mongePointVSubFaceCentroidWeightsWithCircumcenter i₁ i₂)  …
  -/
  ext i
  /-
    case h
    n : Nat
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Ne i₁ i₂
    i : Affine.Simplex.PointsWithCircumcenterIndex (HAdd.hAdd n 2)
    ⊢ Eq (Affine.Simplex.mongePointVSubFaceCentroidWeightsWithCircumcenter i₁ i₂ i …
  -/
  cases' i with i
  · rw [Pi.sub_apply, mongePointWeightsWithCircumcenter, centroidWeightsWithCircumcenter,
      mongePointVSubFaceCentroidWeightsWithCircumcenter]
    have hu : #{i₁, i₂}ᶜ = n + 1 := by
      simp [card_compl, Fintype.card_fin, h]
    /-
      case h.pointIndex
      n : Nat
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      h : Ne i₁ i₂
      i : Fin (HAdd.hAdd (HAdd.hAdd n 2) 1)
      hu : Eq (HasCompl.compl (Insert.insert i₁ (Singleton.singleton i₂))).card (HAd …
      ⊢ Eq (ite (Or (Eq i i₁) (Eq i i₂)) (Inv.inv ↑(HAdd.hAdd n 1)) 0) (HSub.hSub (I …
    -/
    rw [hu]
    /-
      case h.pointIndex
      n : Nat
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      h : Ne i₁ i₂
      i : Fin (HAdd.hAdd (HAdd.hAdd n 2) 1)
      hu : Eq (HasCompl.compl (Insert.insert i₁ (Singleton.singleton i₂))).card (HAd …
      ⊢ Eq (ite (Or (Eq i i₁) (Eq i i₂)) (Inv.inv ↑(HAdd.hAdd n 1)) 0) (HSub.hSub (I …
    -/
                                      /-
                                        🎉 no goals
                                      -/
    by_cases hi : i = i₁ ∨ i = i₂ <;> simp [compl_eq_univ_sdiff, hi]
                                      /-
                                        🎉 no goals
                                      -/
  · simp [mongePointWeightsWithCircumcenter, centroidWeightsWithCircumcenter,
      mongePointVSubFaceCentroidWeightsWithCircumcenter]


/-- `mongePointVSubFaceCentroidWeightsWithCircumcenter` sums to 0. -/
@[simp]
theorem sum_mongePointVSubFaceCentroidWeightsWithCircumcenter {n : ℕ} {i₁ i₂ : Fin (n + 3)}
    (h : i₁ ≠ i₂) : ∑ i, mongePointVSubFaceCentroidWeightsWithCircumcenter i₁ i₂ i = 0 := by
  /-
    n : Nat
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Ne i₁ i₂
    ⊢ Eq (Finset.univ.sum fun i => Affine.Simplex.mongePointVSubFaceCentroidWeight …
  -/
  rw [mongePointVSubFaceCentroidWeightsWithCircumcenter_eq_sub h]
  /-
    n : Nat
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Ne i₁ i₂
    ⊢ Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.mongePointWeightsWith …
  -/
  simp_rw [Pi.sub_apply, sum_sub_distrib, sum_mongePointWeightsWithCircumcenter]
  /-
    n : Nat
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Ne i₁ i₂
    ⊢ Eq (HSub.hSub 1 (Finset.univ.sum fun x => Affine.Simplex.centroidWeightsWith …
  -/
  rw [sum_centroidWeightsWithCircumcenter, sub_self]
  /-
    n : Nat
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Ne i₁ i₂
    ⊢ (HasCompl.compl (Insert.insert i₁ (Singleton.singleton i₂))).Nonempty
  -/
  simp [← card_pos, card_compl, h]
  /-
    🎉 no goals
  -/


/-- The Monge point of an (n+2)-simplex, minus the centroid of an
n-dimensional face, in terms of `pointsWithCircumcenter`. -/
theorem mongePoint_vsub_face_centroid_eq_weightedVSub_of_pointsWithCircumcenter {n : ℕ}
    (s : Simplex ℝ P (n + 2)) {i₁ i₂ : Fin (n + 3)} (h : i₁ ≠ i₂) :
    s.mongePoint -ᵥ ({i₁, i₂}ᶜ : Finset (Fin (n + 3))).centroid ℝ s.points =
      (univ : Finset (PointsWithCircumcenterIndex (n + 2))).weightedVSub s.pointsWithCircumcenter
        (mongePointVSubFaceCentroidWeightsWithCircumcenter i₁ i₂) := by
  simp_rw [mongePoint_eq_affineCombination_of_pointsWithCircumcenter,
    centroid_eq_affineCombination_of_pointsWithCircumcenter, affineCombination_vsub,
    mongePointVSubFaceCentroidWeightsWithCircumcenter_eq_sub h]


/-- The Monge point of an (n+2)-simplex, minus the centroid of an
n-dimensional face, is orthogonal to the difference of the two
vertices not in that face. -/
theorem inner_mongePoint_vsub_face_centroid_vsub {n : ℕ} (s : Simplex ℝ P (n + 2))
    {i₁ i₂ : Fin (n + 3)} :
    ⟪s.mongePoint -ᵥ ({i₁, i₂}ᶜ : Finset (Fin (n + 3))).centroid ℝ s.points,
        s.points i₁ -ᵥ s.points i₂⟫ =
      0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    ⊢ Eq (Inner.inner (VSub.vsub s.mongePoint (Finset.centroid Real (HasCompl.comp …
  -/
  by_cases h : i₁ = i₂
    /-
      case pos
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      h : Eq i₁ i₂
      ⊢ Eq (Inner.inner (VSub.vsub s.mongePoint (Finset.centroid Real (HasCompl.comp …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  simp_rw [mongePoint_vsub_face_centroid_eq_weightedVSub_of_pointsWithCircumcenter s h,
    point_eq_affineCombination_of_pointsWithCircumcenter, affineCombination_vsub]
  have hs : ∑ i, (pointWeightsWithCircumcenter i₁ - pointWeightsWithCircumcenter i₂) i = 0 := by
    simp
  rw [inner_weightedVSub _ (sum_mongePointVSubFaceCentroidWeightsWithCircumcenter h) _ hs,
    sum_pointsWithCircumcenter, pointsWithCircumcenter_eq_circumcenter]
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Not (Eq i₁ i₂)
    hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
    ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (Finset.univ.sum fun i => Finset.univ.sum  …
  -/
  simp only [mongePointVSubFaceCentroidWeightsWithCircumcenter, pointsWithCircumcenter_point]
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Not (Eq i₁ i₂)
    hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
    ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (Finset.univ.sum fun x => Finset.univ.sum  …
  -/
  let fs : Finset (Fin (n + 3)) := {i₁, i₂}
  have hfs : ∀ i : Fin (n + 3), i ∉ fs → i ≠ i₁ ∧ i ≠ i₂ := by
    intro i hi
    constructor <;> · intro hj; simp [fs, ← hj] at hi
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    h : Not (Eq i₁ i₂)
    hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
    fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
    hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
    ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (Finset.univ.sum fun x => Finset.univ.sum  …
  -/
  rw [← sum_subset fs.subset_univ _]
  · simp_rw [sum_pointsWithCircumcenter, pointsWithCircumcenter_eq_circumcenter,
      pointsWithCircumcenter_point, Pi.sub_apply, pointWeightsWithCircumcenter]
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      h : Not (Eq i₁ i₂)
      hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
      fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
      hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
      ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (fs.sum fun x => HAdd.hAdd (Finset.univ.su …
    -/
    rw [← sum_subset fs.subset_univ _]
      /-
        case neg
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 2)
        i₁ i₂ : Fin (HAdd.hAdd n 3)
        h : Not (Eq i₁ i₂)
        hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
        fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
        hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
        ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (fs.sum fun x => HAdd.hAdd (Finset.univ.su …
      -/
    · simp_rw [fs, sum_insert (not_mem_singleton.2 h), sum_singleton]
      /-
        case neg
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 2)
        i₁ i₂ : Fin (HAdd.hAdd n 3)
        h : Not (Eq i₁ i₂)
        hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
        fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
        hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
        ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Finset.univ.sum fun …
      -/
      repeat rw [← sum_subset fs.subset_univ _]
        /-
          case neg
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          n : Nat
          s : Affine.Simplex Real P (HAdd.hAdd n 2)
          i₁ i₂ : Fin (HAdd.hAdd n 3)
          h : Not (Eq i₁ i₂)
          hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
          fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
          hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
          ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (fs.sum fun x => HMu …
        -/
      · simp_rw [fs, sum_insert (not_mem_singleton.2 h), sum_singleton]
        /-
          case neg
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          n : Nat
          s : Affine.Simplex Real P (HAdd.hAdd n 2)
          i₁ i₂ : Fin (HAdd.hAdd n 3)
          h : Not (Eq i₁ i₂)
          hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
          fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
          hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
          ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMu …
        -/
        simp [h, Ne.symm h, dist_comm (s.points i₁)]
        /-
          🎉 no goals
        -/
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 2)
        i₁ i₂ : Fin (HAdd.hAdd n 3)
        h : Not (Eq i₁ i₂)
        hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
        fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
        hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
        ⊢ ∀ (x : Fin (HAdd.hAdd n 3)), Membership.mem Finset.univ x → Not (Membership. …
      -/
      all_goals intro i _ hi; simp [hfs i hi]
      /-
        🎉 no goals
      -/
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 2)
        i₁ i₂ : Fin (HAdd.hAdd n 3)
        h : Not (Eq i₁ i₂)
        hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
        fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
        hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
        ⊢ ∀ (x : Fin (HAdd.hAdd n 3)), Membership.mem Finset.univ x → Not (Membership. …
      -/
    · intro i _ hi
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 2)
        i₁ i₂ : Fin (HAdd.hAdd n 3)
        h : Not (Eq i₁ i₂)
        hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
        fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
        hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
        i : Fin (HAdd.hAdd n 3)
        a✝ : Membership.mem Finset.univ i
        hi : Not (Membership.mem fs i)
        ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (-2) ↑(HAdd.hAdd n 1)) (HSub.hSub (ite ( …
      -/
      simp [hfs i hi, pointsWithCircumcenter]
      /-
        🎉 no goals
      -/
    /-
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      h : Not (Eq i₁ i₂)
      hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
      fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
      hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
      ⊢ ∀ (x : Fin (HAdd.hAdd n 3)), Membership.mem Finset.univ x → Not (Membership. …
    -/
  · intro i _ hi
    /-
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      h : Not (Eq i₁ i₂)
      hs : Eq (Finset.univ.sum fun i => HSub.hSub (Affine.Simplex.pointWeightsWithCi …
      fs : Finset (Fin (HAdd.hAdd n 3)) := Insert.insert i₁ (Singleton.singleton i₂)
      hfs : ∀ (i : Fin (HAdd.hAdd n 3)), Not (Membership.mem fs i) → And (Ne i i₁) ( …
      i : Fin (HAdd.hAdd n 3)
      a✝ : Membership.mem Finset.univ i
      hi : Not (Membership.mem fs i)
      ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (HMul.hMul (ite (Or (Eq i i₁) (Eq i i …
    -/
    simp [hfs i hi]
    /-
      🎉 no goals
    -/


/-- A Monge plane of an (n+2)-simplex is the (n+1)-dimensional affine
subspace of the subspace spanned by the simplex that passes through
the centroid of an n-dimensional face and is orthogonal to the
opposite edge (in 2 dimensions, this is the same as an altitude).
This definition is only intended to be used when `i₁ ≠ i₂`. -/
def mongePlane {n : ℕ} (s : Simplex ℝ P (n + 2)) (i₁ i₂ : Fin (n + 3)) : AffineSubspace ℝ P :=
  mk' (({i₁, i₂}ᶜ : Finset (Fin (n + 3))).centroid ℝ s.points) (ℝ ∙ s.points i₁ -ᵥ s.points i₂)ᗮ ⊓
    affineSpan ℝ (Set.range s.points)


/-- The definition of a Monge plane. -/
theorem mongePlane_def {n : ℕ} (s : Simplex ℝ P (n + 2)) (i₁ i₂ : Fin (n + 3)) :
    s.mongePlane i₁ i₂ =
      mk' (({i₁, i₂}ᶜ : Finset (Fin (n + 3))).centroid ℝ s.points)
          (ℝ ∙ s.points i₁ -ᵥ s.points i₂)ᗮ ⊓
        affineSpan ℝ (Set.range s.points) :=
  rfl


/-- The Monge plane associated with vertices `i₁` and `i₂` equals that
associated with `i₂` and `i₁`. -/
theorem mongePlane_comm {n : ℕ} (s : Simplex ℝ P (n + 2)) (i₁ i₂ : Fin (n + 3)) :
    s.mongePlane i₁ i₂ = s.mongePlane i₂ i₁ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    ⊢ Eq (s.mongePlane i₁ i₂) (s.mongePlane i₂ i₁)
  -/
  simp_rw [mongePlane_def]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    ⊢ Eq (Min.min (AffineSubspace.mk' (Finset.centroid Real (HasCompl.compl (Inser …
  -/
  congr 3
    /-
      case e_a.e_p.e_s
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      ⊢ Eq (HasCompl.compl (Insert.insert i₁ (Singleton.singleton i₂))) (HasCompl.co …
    -/
  · congr 1
    /-
      case e_a.e_p.e_s.e_a
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      ⊢ Eq (Insert.insert i₁ (Singleton.singleton i₂)) (Insert.insert i₂ (Singleton. …
    -/
    exact pair_comm _ _
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_direction.e_K
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      ⊢ Eq (Submodule.span Real (Singleton.singleton (VSub.vsub (s.points i₁) (s.poi …
    -/
  · ext
    /-
      case e_a.e_direction.e_K.h
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      x✝ : V
      ⊢ Iff (Membership.mem (Submodule.span Real (Singleton.singleton (VSub.vsub (s. …
    -/
    simp_rw [Submodule.mem_span_singleton]
    /-
      case e_a.e_direction.e_K.h
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      x✝ : V
      ⊢ Iff (Exists fun a => Eq (HSMul.hSMul a (VSub.vsub (s.points i₁) (s.points i₂ …
    -/
    constructor
    /-
      case e_a.e_direction.e_K.h.mp
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 2)
      i₁ i₂ : Fin (HAdd.hAdd n 3)
      x✝ : V
      ⊢ (Exists fun a => Eq (HSMul.hSMul a (VSub.vsub (s.points i₁) (s.points i₂)))  …
    -/
    all_goals rintro ⟨r, rfl⟩; use -r; rw [neg_smul, ← smul_neg, neg_vsub_eq_vsub_rev]
    /-
      🎉 no goals
    -/


/-- The Monge point lies in the Monge planes. -/
theorem mongePoint_mem_mongePlane {n : ℕ} (s : Simplex ℝ P (n + 2)) {i₁ i₂ : Fin (n + 3)} :
    s.mongePoint ∈ s.mongePlane i₁ i₂ := by
  rw [mongePlane_def, mem_inf_iff, ← vsub_right_mem_direction_iff_mem (self_mem_mk' _ _),
    direction_mk', Submodule.mem_orthogonal']
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    ⊢ And (∀ (u : V), Membership.mem (Submodule.span Real (Singleton.singleton (VS …
  -/
  refine ⟨?_, s.mongePoint_mem_affineSpan⟩
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    ⊢ ∀ (u : V), Membership.mem (Submodule.span Real (Singleton.singleton (VSub.vs …
  -/
  intro v hv
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    v : V
    hv : Membership.mem (Submodule.span Real (Singleton.singleton (VSub.vsub (s.po …
    ⊢ Eq (Inner.inner (VSub.vsub s.mongePoint (Finset.centroid Real (HasCompl.comp …
  -/
  rcases Submodule.mem_span_singleton.mp hv with ⟨r, rfl⟩
  /-
    case intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ i₂ : Fin (HAdd.hAdd n 3)
    r : Real
    hv : Membership.mem (Submodule.span Real (Singleton.singleton (VSub.vsub (s.po …
    ⊢ Eq (Inner.inner (VSub.vsub s.mongePoint (Finset.centroid Real (HasCompl.comp …
  -/
  rw [inner_smul_right, s.inner_mongePoint_vsub_face_centroid_vsub, mul_zero]
  /-
    🎉 no goals
  -/


/-- The direction of a Monge plane. -/
theorem direction_mongePlane {n : ℕ} (s : Simplex ℝ P (n + 2)) {i₁ i₂ : Fin (n + 3)} :
    (s.mongePlane i₁ i₂).direction =
      (ℝ ∙ s.points i₁ -ᵥ s.points i₂)ᗮ ⊓ vectorSpan ℝ (Set.range s.points) := by
  rw [mongePlane_def, direction_inf_of_mem_inf s.mongePoint_mem_mongePlane, direction_mk',
    direction_affineSpan]


/-- The Monge point is the only point in all the Monge planes from any
one vertex. -/
theorem eq_mongePoint_of_forall_mem_mongePlane {n : ℕ} {s : Simplex ℝ P (n + 2)} {i₁ : Fin (n + 3)}
    {p : P} (h : ∀ i₂, i₁ ≠ i₂ → p ∈ s.mongePlane i₁ i₂) : p = s.mongePoint := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ : Fin (HAdd.hAdd n 3)
    p : P
    h : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (s.mongePlane i₁ i …
    ⊢ Eq p s.mongePoint
  -/
  rw [← @vsub_eq_zero_iff_eq V]
  have h' : ∀ i₂, i₁ ≠ i₂ → p -ᵥ s.mongePoint ∈
      (ℝ ∙ s.points i₁ -ᵥ s.points i₂)ᗮ ⊓ vectorSpan ℝ (Set.range s.points) := by
    intro i₂ hne
    rw [← s.direction_mongePlane, vsub_right_mem_direction_iff_mem s.mongePoint_mem_mongePlane]
    exact h i₂ hne
  have hi : p -ᵥ s.mongePoint ∈ ⨅ i₂ : { i // i₁ ≠ i }, (ℝ ∙ s.points i₁ -ᵥ s.points i₂)ᗮ := by
    rw [Submodule.mem_iInf]
    exact fun i => (Submodule.mem_inf.1 (h' i i.property)).1
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ : Fin (HAdd.hAdd n 3)
    p : P
    h : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (s.mongePlane i₁ i …
    h' : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (Min.min (Submodu …
    hi : Membership.mem (iInf fun i₂ => (Submodule.span Real (Singleton.singleton  …
    ⊢ Eq (VSub.vsub p s.mongePoint) 0
  -/
  rw [Submodule.iInf_orthogonal, ← Submodule.span_iUnion] at hi
  have hu :
    ⋃ i : { i // i₁ ≠ i }, ({s.points i₁ -ᵥ s.points i} : Set V) =
      (s.points i₁ -ᵥ ·) '' (s.points '' (Set.univ \ {i₁})) := by
    rw [Set.image_image]
    ext x
    simp_rw [Set.mem_iUnion, Set.mem_image, Set.mem_singleton_iff, Set.mem_diff_singleton]
    constructor
    · rintro ⟨i, rfl⟩
      use i, ⟨Set.mem_univ _, i.property.symm⟩
    · rintro ⟨i, ⟨-, hi⟩, rfl⟩
      -- Porting note: was `use ⟨i, hi.symm⟩, rfl`
      exact ⟨⟨i, hi.symm⟩, rfl⟩
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ : Fin (HAdd.hAdd n 3)
    p : P
    h : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (s.mongePlane i₁ i …
    h' : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (Min.min (Submodu …
    hi : Membership.mem (Submodule.span Real (Set.iUnion fun i => Singleton.single …
    hu : Eq (Set.iUnion fun i => Singleton.singleton (VSub.vsub (s.points i₁) (s.p …
    ⊢ Eq (VSub.vsub p s.mongePoint) 0
  -/
  rw [hu, ← vectorSpan_image_eq_span_vsub_set_left_ne ℝ _ (Set.mem_univ _), Set.image_univ] at hi
  have hv : p -ᵥ s.mongePoint ∈ vectorSpan ℝ (Set.range s.points) := by
    let s₁ : Finset (Fin (n + 3)) := univ.erase i₁
    obtain ⟨i₂, h₂⟩ := card_pos.1 (show 0 < #s₁ by simp [s₁, card_erase_of_mem])
    have h₁₂ : i₁ ≠ i₂ := (ne_of_mem_erase h₂).symm
    exact (Submodule.mem_inf.1 (h' i₂ h₁₂)).2
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 2)
    i₁ : Fin (HAdd.hAdd n 3)
    p : P
    h : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (s.mongePlane i₁ i …
    h' : ∀ (i₂ : Fin (HAdd.hAdd n 3)), Ne i₁ i₂ → Membership.mem (Min.min (Submodu …
    hi : Membership.mem (vectorSpan Real (Set.range s.points)).orthogonal (VSub.vs …
    hu : Eq (Set.iUnion fun i => Singleton.singleton (VSub.vsub (s.points i₁) (s.p …
    hv : Membership.mem (vectorSpan Real (Set.range s.points)) (VSub.vsub p s.mong …
    ⊢ Eq (VSub.vsub p s.mongePoint) 0
  -/
  exact Submodule.disjoint_def.1 (vectorSpan ℝ (Set.range s.points)).orthogonal_disjoint _ hv hi
  /-
    🎉 no goals
  -/


/-- An altitude of a simplex is the line that passes through a vertex
and is orthogonal to the opposite face. -/
def altitude {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) : AffineSubspace ℝ P :=
  mk' (s.points i) (affineSpan ℝ (s.points '' ↑(univ.erase i))).directionᗮ ⊓
    affineSpan ℝ (Set.range s.points)


/-- The definition of an altitude. -/
theorem altitude_def {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) :
    s.altitude i =
      mk' (s.points i) (affineSpan ℝ (s.points '' ↑(univ.erase i))).directionᗮ ⊓
        affineSpan ℝ (Set.range s.points) :=
  rfl


/-- A vertex lies in the corresponding altitude. -/
theorem mem_altitude {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) :
    s.points i ∈ s.altitude i :=
  (mem_inf_iff _ _ _).2 ⟨self_mem_mk' _ _, mem_affineSpan ℝ (Set.mem_range_self _)⟩


/-- The direction of an altitude. -/
theorem direction_altitude {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) :
    (s.altitude i).direction =
      (vectorSpan ℝ (s.points '' ↑(Finset.univ.erase i)))ᗮ ⊓ vectorSpan ℝ (Set.range s.points) := by
  rw [altitude_def,
    direction_inf_of_mem (self_mem_mk' (s.points i) _) (mem_affineSpan ℝ (Set.mem_range_self _)),
    direction_mk', direction_affineSpan, direction_affineSpan]


/-- The vector span of the opposite face lies in the direction
orthogonal to an altitude. -/
theorem vectorSpan_isOrtho_altitude_direction {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) :
    vectorSpan ℝ (s.points '' ↑(Finset.univ.erase i)) ⟂ (s.altitude i).direction := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    ⊢ (vectorSpan Real (Set.image s.points ↑(Finset.univ.erase i))).IsOrtho (s.alt …
  -/
  rw [direction_altitude]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    ⊢ (vectorSpan Real (Set.image s.points ↑(Finset.univ.erase i))).IsOrtho (Min.m …
  -/
  exact (Submodule.isOrtho_orthogonal_right _).mono_right inf_le_left
  /-
    🎉 no goals
  -/


/-- An altitude is finite-dimensional. -/
instance finiteDimensional_direction_altitude {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) :
    FiniteDimensional ℝ (s.altitude i).direction := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    ⊢ FiniteDimensional Real (Subtype fun x => Membership.mem (s.altitude i).direc …
  -/
  rw [direction_altitude]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    ⊢ FiniteDimensional Real (Subtype fun x => Membership.mem (Min.min (vectorSpan …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- An altitude is one-dimensional (i.e., a line). -/
@[simp]
theorem finrank_direction_altitude {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2)) :
    finrank ℝ (s.altitude i).direction = 1 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (Module.finrank Real (Subtype fun x => Membership.mem (s.altitude i).dire …
  -/
  rw [direction_altitude]
  have h := Submodule.finrank_add_inf_finrank_orthogonal
    (vectorSpan_mono ℝ (Set.image_subset_range s.points ↑(univ.erase i)))
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    h : Eq (HAdd.hAdd (Module.finrank Real (Subtype fun x => Membership.mem (vecto …
    ⊢ Eq (Module.finrank Real (Subtype fun x => Membership.mem (Min.min (vectorSpa …
  -/
  have hc : #(univ.erase i) = n + 1 := by rw [card_erase_of_mem (mem_univ _)]; simp
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    h : Eq (HAdd.hAdd (Module.finrank Real (Subtype fun x => Membership.mem (vecto …
    hc : Eq (Finset.univ.erase i).card (HAdd.hAdd n 1)
    ⊢ Eq (Module.finrank Real (Subtype fun x => Membership.mem (Min.min (vectorSpa …
  -/
  refine add_left_cancel (_root_.trans h ?_)
  classical
  rw [s.independent.finrank_vectorSpan (Fintype.card_fin _), ← Finset.coe_image,
    s.independent.finrank_vectorSpan_image_finset hc]


/-- A line through a vertex is the altitude through that vertex if and
only if it is orthogonal to the opposite face. -/
theorem affineSpan_pair_eq_altitude_iff {n : ℕ} (s : Simplex ℝ P (n + 1)) (i : Fin (n + 2))
    (p : P) :
    line[ℝ, p, s.points i] = s.altitude i ↔
      p ≠ s.points i ∧
        p ∈ affineSpan ℝ (Set.range s.points) ∧
          p -ᵥ s.points i ∈ (affineSpan ℝ (s.points '' ↑(Finset.univ.erase i))).directionᗮ := by
  rw [eq_iff_direction_eq_of_mem (mem_affineSpan ℝ (Set.mem_insert_of_mem _ (Set.mem_singleton _)))
      (s.mem_altitude _),
    ← vsub_right_mem_direction_iff_mem (mem_affineSpan ℝ (Set.mem_range_self i)) p,
    direction_affineSpan, direction_affineSpan, direction_affineSpan]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    n : Nat
    s : Affine.Simplex Real P (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 2)
    p : P
    ⊢ Iff (Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i)) …
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
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 1)
      i : Fin (HAdd.hAdd n 2)
      p : P
      ⊢ Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i)))) (s …
    -/
  · intro h
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 1)
      i : Fin (HAdd.hAdd n 2)
      p : P
      h : Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i))))  …
      ⊢ And (Ne p (s.points i)) (And (Membership.mem (vectorSpan Real (Set.range s.p …
    -/
    constructor
      /-
        case mp.left
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        h : Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i))))  …
        ⊢ Ne p (s.points i)
      -/
    · intro heq
      /-
        case mp.left
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        h : Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i))))  …
        heq : Eq p (s.points i)
        ⊢ False
      -/
      rw [heq, Set.pair_eq_singleton, vectorSpan_singleton] at h
      /-
        case mp.left
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        h : Eq Bot.bot (s.altitude i).direction
        heq : Eq p (s.points i)
        ⊢ False
      -/
      have hd : finrank ℝ (s.altitude i).direction = 0 := by rw [← h, finrank_bot]
      /-
        case mp.left
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        h : Eq Bot.bot (s.altitude i).direction
        heq : Eq p (s.points i)
        hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem (s.altitude i).d …
        ⊢ False
      -/
      simp at hd
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        h : Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i))))  …
        ⊢ And (Membership.mem (vectorSpan Real (Set.range s.points)) (VSub.vsub p (s.p …
      -/
    · rw [← Submodule.mem_inf, _root_.inf_comm, ← direction_altitude, ← h]
      exact
        vsub_mem_vectorSpan ℝ (Set.mem_insert _ _) (Set.mem_insert_of_mem _ (Set.mem_singleton _))
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 1)
      i : Fin (HAdd.hAdd n 2)
      p : P
      ⊢ And (Ne p (s.points i)) (And (Membership.mem (vectorSpan Real (Set.range s.p …
    -/
  · rintro ⟨hne, h⟩
    /-
      case mpr.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 1)
      i : Fin (HAdd.hAdd n 2)
      p : P
      hne : Ne p (s.points i)
      h : And (Membership.mem (vectorSpan Real (Set.range s.points)) (VSub.vsub p (s …
      ⊢ Eq (vectorSpan Real (Insert.insert p (Singleton.singleton (s.points i)))) (s …
    -/
    rw [← Submodule.mem_inf, _root_.inf_comm, ← direction_altitude] at h
    rw [vectorSpan_eq_span_vsub_set_left_ne ℝ (Set.mem_insert _ _),
      Set.insert_diff_of_mem _ (Set.mem_singleton _),
      Set.diff_singleton_eq_self fun h => hne (Set.mem_singleton_iff.1 h), Set.image_singleton]
    /-
      case mpr.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      n : Nat
      s : Affine.Simplex Real P (HAdd.hAdd n 1)
      i : Fin (HAdd.hAdd n 2)
      p : P
      hne : Ne p (s.points i)
      h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
      ⊢ Eq (Submodule.span Real (Singleton.singleton (VSub.vsub p (s.points i)))) (s …
    -/
    refine Submodule.eq_of_le_of_finrank_eq ?_ ?_
      /-
        case mpr.intro.refine_1
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        hne : Ne p (s.points i)
        h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
        ⊢ LE.le (Submodule.span Real (Singleton.singleton (VSub.vsub p (s.points i)))) …
      -/
    · rw [Submodule.span_le]
      /-
        case mpr.intro.refine_1
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        hne : Ne p (s.points i)
        h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
        ⊢ HasSubset.Subset (Singleton.singleton (VSub.vsub p (s.points i))) ↑(s.altitu …
      -/
      simpa using h
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.refine_2
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        n : Nat
        s : Affine.Simplex Real P (HAdd.hAdd n 1)
        i : Fin (HAdd.hAdd n 2)
        p : P
        hne : Ne p (s.points i)
        h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
        ⊢ Eq (Module.finrank Real (Subtype fun x => Membership.mem (Submodule.span Rea …
      -/
    · rw [finrank_direction_altitude, finrank_span_set_eq_card]
        /-
          case mpr.intro.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          n : Nat
          s : Affine.Simplex Real P (HAdd.hAdd n 1)
          i : Fin (HAdd.hAdd n 2)
          p : P
          hne : Ne p (s.points i)
          h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
          ⊢ Eq (Singleton.singleton (VSub.vsub p (s.points i))).toFinset.card 1
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          n : Nat
          s : Affine.Simplex Real P (HAdd.hAdd n 1)
          i : Fin (HAdd.hAdd n 2)
          p : P
          hne : Ne p (s.points i)
          h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
          ⊢ LinearIndependent Real Subtype.val
        -/
      · refine linearIndependent_singleton ?_
        /-
          case mpr.intro.refine_2
          V : Type u_1
          P : Type u_2
          inst✝³ : NormedAddCommGroup V
          inst✝² : InnerProductSpace Real V
          inst✝¹ : MetricSpace P
          inst✝ : NormedAddTorsor V P
          n : Nat
          s : Affine.Simplex Real P (HAdd.hAdd n 1)
          i : Fin (HAdd.hAdd n 2)
          p : P
          hne : Ne p (s.points i)
          h : Membership.mem (s.altitude i).direction (VSub.vsub p (s.points i))
          ⊢ Ne (VSub.vsub p (s.points i)) 0
        -/
        simpa using hne
        /-
          🎉 no goals
        -/


/-- The orthocenter of a triangle is the intersection of its
altitudes.  It is defined here as the 2-dimensional case of the
Monge point. -/
def orthocenter (t : Triangle ℝ P) : P :=
  t.mongePoint


/-- The orthocenter equals the Monge point. -/
theorem orthocenter_eq_mongePoint (t : Triangle ℝ P) : t.orthocenter = t.mongePoint :=
  rfl


/-- The position of the orthocenter in relation to the circumcenter
and centroid. -/
theorem orthocenter_eq_smul_vsub_vadd_circumcenter (t : Triangle ℝ P) :
    t.orthocenter =
      (3 : ℝ) • ((univ : Finset (Fin 3)).centroid ℝ t.points -ᵥ t.circumcenter : V) +ᵥ
        t.circumcenter := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    ⊢ Eq t.orthocenter (HVAdd.hVAdd (HSMul.hSMul 3 (VSub.vsub (Finset.centroid Rea …
  -/
  rw [orthocenter_eq_mongePoint, mongePoint_eq_smul_vsub_vadd_circumcenter]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul (HDiv.hDiv ↑(HAdd.hAdd 2 1) ↑(HSub.hSub 2 1)) ( …
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- The orthocenter lies in the affine span. -/
theorem orthocenter_mem_affineSpan (t : Triangle ℝ P) :
    t.orthocenter ∈ affineSpan ℝ (Set.range t.points) :=
  t.mongePoint_mem_affineSpan


/-- Two triangles with the same points have the same orthocenter. -/
theorem orthocenter_eq_of_range_eq {t₁ t₂ : Triangle ℝ P}
    (h : Set.range t₁.points = Set.range t₂.points) : t₁.orthocenter = t₂.orthocenter :=
  mongePoint_eq_of_range_eq h


/-- In the case of a triangle, altitudes are the same thing as Monge
planes. -/
theorem altitude_eq_mongePlane (t : Triangle ℝ P) {i₁ i₂ i₃ : Fin 3} (h₁₂ : i₁ ≠ i₂) (h₁₃ : i₁ ≠ i₃)
    (h₂₃ : i₂ ≠ i₃) : t.altitude i₁ = t.mongePlane i₂ i₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    ⊢ Eq (Affine.Simplex.altitude t i₁) (Affine.Simplex.mongePlane t i₂ i₃)
  -/
  have hs : ({i₂, i₃}ᶜ : Finset (Fin 3)) = {i₁} := by decide +revert
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    hs : Eq (HasCompl.compl (Insert.insert i₂ (Singleton.singleton i₃))) (Singleto …
    ⊢ Eq (Affine.Simplex.altitude t i₁) (Affine.Simplex.mongePlane t i₂ i₃)
  -/
  have he : univ.erase i₁ = {i₂, i₃} := by decide +revert
  rw [mongePlane_def, altitude_def, direction_affineSpan, hs, he, centroid_singleton, coe_insert,
    coe_singleton, vectorSpan_image_eq_span_vsub_set_left_ne ℝ _ (Set.mem_insert i₂ _)]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    h₁₃ : Ne i₁ i₃
    h₂₃ : Ne i₂ i₃
    hs : Eq (HasCompl.compl (Insert.insert i₂ (Singleton.singleton i₃))) (Singleto …
    he : Eq (Finset.univ.erase i₁) (Insert.insert i₂ (Singleton.singleton i₃))
    ⊢ Eq (Min.min (AffineSubspace.mk' (t.points i₁) (Submodule.span Real (Set.imag …
  -/
  simp [h₂₃, Submodule.span_insert_eq_span]
  /-
    🎉 no goals
  -/


/-- The orthocenter lies in the altitudes. -/
theorem orthocenter_mem_altitude (t : Triangle ℝ P) {i₁ : Fin 3} :
    t.orthocenter ∈ t.altitude i₁ := by
  obtain ⟨i₂, i₃, h₁₂, h₂₃, h₁₃⟩ : ∃ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ := by
    decide +revert
  /-
    case intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    h₂₃ : Ne i₂ i₃
    h₁₃ : Ne i₁ i₃
    ⊢ Membership.mem (Affine.Simplex.altitude t i₁) t.orthocenter
  -/
  rw [orthocenter_eq_mongePoint, t.altitude_eq_mongePlane h₁₂ h₁₃ h₂₃]
  /-
    case intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ i₃ : Fin 3
    h₁₂ : Ne i₁ i₂
    h₂₃ : Ne i₂ i₃
    h₁₃ : Ne i₁ i₃
    ⊢ Membership.mem (Affine.Simplex.mongePlane t i₂ i₃) (Affine.Simplex.mongePoin …
  -/
  exact t.mongePoint_mem_mongePlane
  /-
    🎉 no goals
  -/


/-- The orthocenter is the only point lying in any two of the
altitudes. -/
theorem eq_orthocenter_of_forall_mem_altitude {t : Triangle ℝ P} {i₁ i₂ : Fin 3} {p : P}
    (h₁₂ : i₁ ≠ i₂) (h₁ : p ∈ t.altitude i₁) (h₂ : p ∈ t.altitude i₂) : p = t.orthocenter := by
  obtain ⟨i₃, h₂₃, h₁₃⟩ : ∃ i₃, i₂ ≠ i₃ ∧ i₁ ≠ i₃ := by
    clear h₁ h₂
    decide +revert
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    p : P
    h₁₂ : Ne i₁ i₂
    h₁ : Membership.mem (Affine.Simplex.altitude t i₁) p
    h₂ : Membership.mem (Affine.Simplex.altitude t i₂) p
    i₃ : Fin 3
    h₂₃ : Ne i₂ i₃
    h₁₃ : Ne i₁ i₃
    ⊢ Eq p t.orthocenter
  -/
  rw [t.altitude_eq_mongePlane h₁₃ h₁₂ h₂₃.symm] at h₁
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    p : P
    h₁₂ : Ne i₁ i₂
    h₂ : Membership.mem (Affine.Simplex.altitude t i₂) p
    i₃ : Fin 3
    h₁ : Membership.mem (Affine.Simplex.mongePlane t i₃ i₂) p
    h₂₃ : Ne i₂ i₃
    h₁₃ : Ne i₁ i₃
    ⊢ Eq p t.orthocenter
  -/
  rw [t.altitude_eq_mongePlane h₂₃ h₁₂.symm h₁₃.symm] at h₂
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    p : P
    h₁₂ : Ne i₁ i₂
    i₃ : Fin 3
    h₂ : Membership.mem (Affine.Simplex.mongePlane t i₃ i₁) p
    h₁ : Membership.mem (Affine.Simplex.mongePlane t i₃ i₂) p
    h₂₃ : Ne i₂ i₃
    h₁₃ : Ne i₁ i₃
    ⊢ Eq p t.orthocenter
  -/
  rw [orthocenter_eq_mongePoint]
  have ha : ∀ i, i₃ ≠ i → p ∈ t.mongePlane i₃ i := by
    intro i hi
    obtain rfl | rfl : i₁ = i ∨ i₂ = i := by omega
    all_goals assumption
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    p : P
    h₁₂ : Ne i₁ i₂
    i₃ : Fin 3
    h₂ : Membership.mem (Affine.Simplex.mongePlane t i₃ i₁) p
    h₁ : Membership.mem (Affine.Simplex.mongePlane t i₃ i₂) p
    h₂₃ : Ne i₂ i₃
    h₁₃ : Ne i₁ i₃
    ha : ∀ (i : Fin 3), Ne i₃ i → Membership.mem (Affine.Simplex.mongePlane t i₃ i …
    ⊢ Eq p (Affine.Simplex.mongePoint t)
  -/
  exact eq_mongePoint_of_forall_mem_mongePlane ha
  /-
    🎉 no goals
  -/


/-- The distance from the orthocenter to the reflection of the
circumcenter in a side equals the circumradius. -/
theorem dist_orthocenter_reflection_circumcenter (t : Triangle ℝ P) {i₁ i₂ : Fin 3} (h : i₁ ≠ i₂) :
    dist t.orthocenter (reflection (affineSpan ℝ (t.points '' {i₁, i₂})) t.circumcenter) =
      t.circumradius := by
  rw [← mul_self_inj_of_nonneg dist_nonneg t.circumradius_nonneg,
    t.reflection_circumcenter_eq_affineCombination_of_pointsWithCircumcenter h,
    t.orthocenter_eq_mongePoint, mongePoint_eq_affineCombination_of_pointsWithCircumcenter,
    dist_affineCombination t.pointsWithCircumcenter (sum_mongePointWeightsWithCircumcenter _)
      (sum_reflectionCircumcenterWeightsWithCircumcenter h)]
  simp_rw [sum_pointsWithCircumcenter, Pi.sub_apply, mongePointWeightsWithCircumcenter,
    reflectionCircumcenterWeightsWithCircumcenter]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    h : Ne i₁ i₂
    ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (Finset.univ.sum fun x => HAdd.hAdd (Finse …
  -/
  have hu : ({i₁, i₂} : Finset (Fin 3)) ⊆ univ := subset_univ _
  obtain ⟨i₃, hi₃, hi₃₁, hi₃₂⟩ :
      ∃ i₃, univ \ ({i₁, i₂} : Finset (Fin 3)) = {i₃} ∧ i₃ ≠ i₁ ∧ i₃ ≠ i₂ := by
    decide +revert
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    h : Ne i₁ i₂
    hu : HasSubset.Subset (Insert.insert i₁ (Singleton.singleton i₂)) Finset.univ
    i₃ : Fin 3
    hi₃ : Eq (SDiff.sdiff Finset.univ (Insert.insert i₁ (Singleton.singleton i₂))) …
    hi₃₁ : Ne i₃ i₁
    hi₃₂ : Ne i₃ i₂
    ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (Finset.univ.sum fun x => HAdd.hAdd (Finse …
  -/
  simp_rw [← sum_sdiff hu, hi₃]
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    h : Ne i₁ i₂
    hu : HasSubset.Subset (Insert.insert i₁ (Singleton.singleton i₂)) Finset.univ
    i₃ : Fin 3
    hi₃ : Eq (SDiff.sdiff Finset.univ (Insert.insert i₁ (Singleton.singleton i₂))) …
    hi₃₁ : Ne i₃ i₁
    hi₃₂ : Ne i₃ i₂
    ⊢ Eq (HDiv.hDiv (Neg.neg (HAdd.hAdd (HAdd.hAdd ((Singleton.singleton i₃).sum f …
  -/
  norm_num [hi₃₁, hi₃₂]
  /-
    🎉 no goals
  -/


/-- The distance from the orthocenter to the reflection of the
circumcenter in a side equals the circumradius, variant using a
`Finset`. -/
theorem dist_orthocenter_reflection_circumcenter_finset (t : Triangle ℝ P) {i₁ i₂ : Fin 3}
    (h : i₁ ≠ i₂) :
    dist t.orthocenter
        (reflection (affineSpan ℝ (t.points '' ↑({i₁, i₂} : Finset (Fin 3)))) t.circumcenter) =
      t.circumradius := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    h : Ne i₁ i₂
    ⊢ Eq (Dist.dist t.orthocenter ((EuclideanGeometry.reflection (affineSpan Real  …
  -/
  simp only [mem_singleton, coe_insert, coe_singleton, Set.mem_singleton_iff]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i₁ i₂ : Fin 3
    h : Ne i₁ i₂
    ⊢ Eq (Dist.dist t.orthocenter ((EuclideanGeometry.reflection (affineSpan Real  …
  -/
  exact dist_orthocenter_reflection_circumcenter _ h
  /-
    🎉 no goals
  -/


/-- The affine span of the orthocenter and a vertex is contained in
the altitude. -/
theorem affineSpan_orthocenter_point_le_altitude (t : Triangle ℝ P) (i : Fin 3) :
    line[ℝ, t.orthocenter, t.points i] ≤ t.altitude i := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i : Fin 3
    ⊢ LE.le (affineSpan Real (Insert.insert t.orthocenter (Singleton.singleton (t. …
  -/
  refine spanPoints_subset_coe_of_subset_coe ?_
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i : Fin 3
    ⊢ HasSubset.Subset (Insert.insert t.orthocenter (Singleton.singleton (t.points …
  -/
  rw [Set.insert_subset_iff, Set.singleton_subset_iff]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    i : Fin 3
    ⊢ And (Membership.mem (↑(Affine.Simplex.altitude t i)) t.orthocenter) (Members …
  -/
  exact ⟨t.orthocenter_mem_altitude, t.mem_altitude i⟩
  /-
    🎉 no goals
  -/


/-- Suppose we are given a triangle `t₁`, and replace one of its
vertices by its orthocenter, yielding triangle `t₂` (with vertices not
necessarily listed in the same order).  Then an altitude of `t₂` from
a vertex that was not replaced is the corresponding side of `t₁`. -/
theorem altitude_replace_orthocenter_eq_affineSpan {t₁ t₂ : Triangle ℝ P}
    {i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3} (hi₁₂ : i₁ ≠ i₂) (hi₁₃ : i₁ ≠ i₃) (hi₂₃ : i₂ ≠ i₃) (hj₁₂ : j₁ ≠ j₂)
    (hj₁₃ : j₁ ≠ j₃) (hj₂₃ : j₂ ≠ j₃) (h₁ : t₂.points j₁ = t₁.orthocenter)
    (h₂ : t₂.points j₂ = t₁.points i₂) (h₃ : t₂.points j₃ = t₁.points i₃) :
    t₂.altitude j₂ = line[ℝ, t₁.points i₁, t₁.points i₂] := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    ⊢ Eq (Affine.Simplex.altitude t₂ j₂) (affineSpan Real (Insert.insert (t₁.point …
  -/
  symm
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    ⊢ Eq (affineSpan Real (Insert.insert (t₁.points i₁) (Singleton.singleton (t₁.p …
  -/
  rw [← h₂, t₂.affineSpan_pair_eq_altitude_iff]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    ⊢ And (Ne (t₁.points i₁) (t₂.points j₂)) (And (Membership.mem (affineSpan Real …
  -/
  rw [h₂]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    ⊢ And (Ne (t₁.points i₁) (t₁.points i₂)) (And (Membership.mem (affineSpan Real …
  -/
  use t₁.independent.injective.ne hi₁₂
  have he : affineSpan ℝ (Set.range t₂.points) = affineSpan ℝ (Set.range t₁.points) := by
    refine ext_of_direction_eq ?_
      ⟨t₁.points i₃, mem_affineSpan ℝ ⟨j₃, h₃⟩, mem_affineSpan ℝ (Set.mem_range_self _)⟩
    refine Submodule.eq_of_le_of_finrank_eq (direction_le (spanPoints_subset_coe_of_subset_coe ?_))
      ?_
    · have hu : (Finset.univ : Finset (Fin 3)) = {j₁, j₂, j₃} := by
        clear h₁ h₂ h₃
        decide +revert
      rw [← Set.image_univ, ← Finset.coe_univ, hu, Finset.coe_insert, Finset.coe_insert,
        Finset.coe_singleton, Set.image_insert_eq, Set.image_insert_eq, Set.image_singleton, h₁, h₂,
        h₃, Set.insert_subset_iff, Set.insert_subset_iff, Set.singleton_subset_iff]
      exact
        ⟨t₁.orthocenter_mem_affineSpan, mem_affineSpan ℝ (Set.mem_range_self _),
          mem_affineSpan ℝ (Set.mem_range_self _)⟩
    · rw [direction_affineSpan, direction_affineSpan,
        t₁.independent.finrank_vectorSpan (Fintype.card_fin _),
        t₂.independent.finrank_vectorSpan (Fintype.card_fin _)]
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    he : Eq (affineSpan Real (Set.range t₂.points)) (affineSpan Real (Set.range t₁ …
    ⊢ And (Membership.mem (affineSpan Real (Set.range t₂.points)) (t₁.points i₁))  …
  -/
  rw [he]
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    he : Eq (affineSpan Real (Set.range t₂.points)) (affineSpan Real (Set.range t₁ …
    ⊢ And (Membership.mem (affineSpan Real (Set.range t₁.points)) (t₁.points i₁))  …
  -/
  use mem_affineSpan ℝ (Set.mem_range_self _)
  have hu : Finset.univ.erase j₂ = {j₁, j₃} := by
    clear h₁ h₂ h₃
    decide +revert
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    he : Eq (affineSpan Real (Set.range t₂.points)) (affineSpan Real (Set.range t₁ …
    hu : Eq (Finset.univ.erase j₂) (Insert.insert j₁ (Singleton.singleton j₃))
    ⊢ Membership.mem (affineSpan Real (Set.image t₂.points ↑(Finset.univ.erase j₂) …
  -/
  rw [hu, Finset.coe_insert, Finset.coe_singleton, Set.image_insert_eq, Set.image_singleton, h₁, h₃]
  have hle : (t₁.altitude i₃).directionᗮ ≤ line[ℝ, t₁.orthocenter, t₁.points i₃].directionᗮ :=
    Submodule.orthogonal_le (direction_le (affineSpan_orthocenter_point_le_altitude _ _))
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    he : Eq (affineSpan Real (Set.range t₂.points)) (affineSpan Real (Set.range t₁ …
    hu : Eq (Finset.univ.erase j₂) (Insert.insert j₁ (Singleton.singleton j₃))
    hle : LE.le (Affine.Simplex.altitude t₁ i₃).direction.orthogonal (affineSpan R …
    ⊢ Membership.mem (affineSpan Real (Insert.insert t₁.orthocenter (Singleton.sin …
  -/
  refine hle ((t₁.vectorSpan_isOrtho_altitude_direction i₃) ?_)
  have hui : Finset.univ.erase i₃ = {i₁, i₂} := by
    clear hle h₂ h₃
    decide +revert
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    he : Eq (affineSpan Real (Set.range t₂.points)) (affineSpan Real (Set.range t₁ …
    hu : Eq (Finset.univ.erase j₂) (Insert.insert j₁ (Singleton.singleton j₃))
    hle : LE.le (Affine.Simplex.altitude t₁ i₃).direction.orthogonal (affineSpan R …
    hui : Eq (Finset.univ.erase i₃) (Insert.insert i₁ (Singleton.singleton i₂))
    ⊢ Membership.mem (vectorSpan Real (Set.image t₁.points ↑(Finset.univ.erase i₃) …
  -/
  rw [hui, Finset.coe_insert, Finset.coe_singleton, Set.image_insert_eq, Set.image_singleton]
  /-
    case right
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    he : Eq (affineSpan Real (Set.range t₂.points)) (affineSpan Real (Set.range t₁ …
    hu : Eq (Finset.univ.erase j₂) (Insert.insert j₁ (Singleton.singleton j₃))
    hle : LE.le (Affine.Simplex.altitude t₁ i₃).direction.orthogonal (affineSpan R …
    hui : Eq (Finset.univ.erase i₃) (Insert.insert i₁ (Singleton.singleton i₂))
    ⊢ Membership.mem (vectorSpan Real (Insert.insert (t₁.points i₁) (Singleton.sin …
  -/
  exact vsub_mem_vectorSpan ℝ (Set.mem_insert _ _) (Set.mem_insert_of_mem _ (Set.mem_singleton _))
  /-
    🎉 no goals
  -/


/-- Suppose we are given a triangle `t₁`, and replace one of its
vertices by its orthocenter, yielding triangle `t₂` (with vertices not
necessarily listed in the same order).  Then the orthocenter of `t₂`
is the vertex of `t₁` that was replaced. -/
theorem orthocenter_replace_orthocenter_eq_point {t₁ t₂ : Triangle ℝ P} {i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3}
    (hi₁₂ : i₁ ≠ i₂) (hi₁₃ : i₁ ≠ i₃) (hi₂₃ : i₂ ≠ i₃) (hj₁₂ : j₁ ≠ j₂) (hj₁₃ : j₁ ≠ j₃)
    (hj₂₃ : j₂ ≠ j₃) (h₁ : t₂.points j₁ = t₁.orthocenter) (h₂ : t₂.points j₂ = t₁.points i₂)
    (h₃ : t₂.points j₃ = t₁.points i₃) : t₂.orthocenter = t₁.points i₁ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t₁ t₂ : Affine.Triangle Real P
    i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
    hi₁₂ : Ne i₁ i₂
    hi₁₃ : Ne i₁ i₃
    hi₂₃ : Ne i₂ i₃
    hj₁₂ : Ne j₁ j₂
    hj₁₃ : Ne j₁ j₃
    hj₂₃ : Ne j₂ j₃
    h₁ : Eq (t₂.points j₁) t₁.orthocenter
    h₂ : Eq (t₂.points j₂) (t₁.points i₂)
    h₃ : Eq (t₂.points j₃) (t₁.points i₃)
    ⊢ Eq t₂.orthocenter (t₁.points i₁)
  -/
  refine (Triangle.eq_orthocenter_of_forall_mem_altitude hj₂₃ ?_ ?_).symm
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t₁ t₂ : Affine.Triangle Real P
      i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
      hi₁₂ : Ne i₁ i₂
      hi₁₃ : Ne i₁ i₃
      hi₂₃ : Ne i₂ i₃
      hj₁₂ : Ne j₁ j₂
      hj₁₃ : Ne j₁ j₃
      hj₂₃ : Ne j₂ j₃
      h₁ : Eq (t₂.points j₁) t₁.orthocenter
      h₂ : Eq (t₂.points j₂) (t₁.points i₂)
      h₃ : Eq (t₂.points j₃) (t₁.points i₃)
      ⊢ Membership.mem (Affine.Simplex.altitude t₂ j₂) (t₁.points i₁)
    -/
  · rw [altitude_replace_orthocenter_eq_affineSpan hi₁₂ hi₁₃ hi₂₃ hj₁₂ hj₁₃ hj₂₃ h₁ h₂ h₃]
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t₁ t₂ : Affine.Triangle Real P
      i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
      hi₁₂ : Ne i₁ i₂
      hi₁₃ : Ne i₁ i₃
      hi₂₃ : Ne i₂ i₃
      hj₁₂ : Ne j₁ j₂
      hj₁₃ : Ne j₁ j₃
      hj₂₃ : Ne j₂ j₃
      h₁ : Eq (t₂.points j₁) t₁.orthocenter
      h₂ : Eq (t₂.points j₂) (t₁.points i₂)
      h₃ : Eq (t₂.points j₃) (t₁.points i₃)
      ⊢ Membership.mem (affineSpan Real (Insert.insert (t₁.points i₁) (Singleton.sin …
    -/
    exact mem_affineSpan ℝ (Set.mem_insert _ _)
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
      t₁ t₂ : Affine.Triangle Real P
      i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
      hi₁₂ : Ne i₁ i₂
      hi₁₃ : Ne i₁ i₃
      hi₂₃ : Ne i₂ i₃
      hj₁₂ : Ne j₁ j₂
      hj₁₃ : Ne j₁ j₃
      hj₂₃ : Ne j₂ j₃
      h₁ : Eq (t₂.points j₁) t₁.orthocenter
      h₂ : Eq (t₂.points j₂) (t₁.points i₂)
      h₃ : Eq (t₂.points j₃) (t₁.points i₃)
      ⊢ Membership.mem (Affine.Simplex.altitude t₂ j₃) (t₁.points i₁)
    -/
  · rw [altitude_replace_orthocenter_eq_affineSpan hi₁₃ hi₁₂ hi₂₃.symm hj₁₃ hj₁₂ hj₂₃.symm h₁ h₃ h₂]
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t₁ t₂ : Affine.Triangle Real P
      i₁ i₂ i₃ j₁ j₂ j₃ : Fin 3
      hi₁₂ : Ne i₁ i₂
      hi₁₃ : Ne i₁ i₃
      hi₂₃ : Ne i₂ i₃
      hj₁₂ : Ne j₁ j₂
      hj₁₃ : Ne j₁ j₃
      hj₂₃ : Ne j₂ j₃
      h₁ : Eq (t₂.points j₁) t₁.orthocenter
      h₂ : Eq (t₂.points j₂) (t₁.points i₂)
      h₃ : Eq (t₂.points j₃) (t₁.points i₃)
      ⊢ Membership.mem (affineSpan Real (Insert.insert (t₁.points i₁) (Singleton.sin …
    -/
    exact mem_affineSpan ℝ (Set.mem_insert _ _)
    /-
      🎉 no goals
    -/


/-- Four points form an orthocentric system if they consist of the
vertices of a triangle and its orthocenter. -/
def OrthocentricSystem (s : Set P) : Prop :=
  ∃ t : Triangle ℝ P,
    t.orthocenter ∉ Set.range t.points ∧ s = insert t.orthocenter (Set.range t.points)


/-- This is an auxiliary lemma giving information about the relation
of two triangles in an orthocentric system; it abstracts some
reasoning, with no geometric content, that is common to some other
lemmas.  Suppose the orthocentric system is generated by triangle `t`,
and we are given three points `p` in the orthocentric system.  Then
either we can find indices `i₁`, `i₂` and `i₃` for `p` such that `p
i₁` is the orthocenter of `t` and `p i₂` and `p i₃` are points `j₂`
and `j₃` of `t`, or `p` has the same points as `t`. -/
theorem exists_of_range_subset_orthocentricSystem {t : Triangle ℝ P}
    (ho : t.orthocenter ∉ Set.range t.points) {p : Fin 3 → P}
    (hps : Set.range p ⊆ insert t.orthocenter (Set.range t.points)) (hpi : Function.Injective p) :
    (∃ i₁ i₂ i₃ j₂ j₃ : Fin 3,
      i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ (∀ i : Fin 3, i = i₁ ∨ i = i₂ ∨ i = i₃) ∧
        p i₁ = t.orthocenter ∧ j₂ ≠ j₃ ∧ t.points j₂ = p i₂ ∧ t.points j₃ = p i₃) ∨
      Set.range p = Set.range t.points := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    t : Affine.Triangle Real P
    ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
    hpi : Function.Injective p
    ⊢ Or (Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exis …
  -/
  by_cases h : t.orthocenter ∈ Set.range p
    /-
      case pos
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      h : Membership.mem (Set.range p) t.orthocenter
      ⊢ Or (Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exis …
    -/
  · left
    /-
      case pos.h
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      h : Membership.mem (Set.range p) t.orthocenter
      ⊢ Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exists f …
    -/
    rcases h with ⟨i₁, h₁⟩
    obtain ⟨i₂, i₃, h₁₂, h₁₃, h₂₃, h₁₂₃⟩ :
        ∃ i₂ i₃ : Fin 3, i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ ∀ i : Fin 3, i = i₁ ∨ i = i₂ ∨ i = i₃ := by
      clear h₁
      decide +revert
    have h : ∀ i, i₁ ≠ i → ∃ j : Fin 3, t.points j = p i := by
      intro i hi
      replace hps := Set.mem_of_mem_insert_of_ne
        (Set.mem_of_mem_of_subset (Set.mem_range_self i) hps) (h₁ ▸ hpi.ne hi.symm)
      exact hps
    /-
      case pos.h.intro.intro.intro.intro.intro.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ : Fin 3
      h₁ : Eq (p i₁) t.orthocenter
      i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h : ∀ (i : Fin 3), Ne i₁ i → Exists fun j => Eq (t.points j) (p i)
      ⊢ Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exists f …
    -/
    rcases h i₂ h₁₂ with ⟨j₂, h₂⟩
    /-
      case pos.h.intro.intro.intro.intro.intro.intro.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ : Fin 3
      h₁ : Eq (p i₁) t.orthocenter
      i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h : ∀ (i : Fin 3), Ne i₁ i → Exists fun j => Eq (t.points j) (p i)
      j₂ : Fin 3
      h₂ : Eq (t.points j₂) (p i₂)
      ⊢ Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exists f …
    -/
    rcases h i₃ h₁₃ with ⟨j₃, h₃⟩
    have hj₂₃ : j₂ ≠ j₃ := by
      intro he
      rw [he, h₃] at h₂
      exact h₂₃.symm (hpi h₂)
    /-
      case pos.h.intro.intro.intro.intro.intro.intro.intro.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ : Fin 3
      h₁ : Eq (p i₁) t.orthocenter
      i₂ i₃ : Fin 3
      h₁₂ : Ne i₁ i₂
      h₁₃ : Ne i₁ i₃
      h₂₃ : Ne i₂ i₃
      h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h : ∀ (i : Fin 3), Ne i₁ i → Exists fun j => Eq (t.points j) (p i)
      j₂ : Fin 3
      h₂ : Eq (t.points j₂) (p i₂)
      j₃ : Fin 3
      h₃ : Eq (t.points j₃) (p i₃)
      hj₂₃ : Ne j₂ j₃
      ⊢ Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exists f …
    -/
    exact ⟨i₁, i₂, i₃, j₂, j₃, h₁₂, h₁₃, h₂₃, h₁₂₃, h₁, hj₂₃, h₂, h₃⟩
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
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      h : Not (Membership.mem (Set.range p) t.orthocenter)
      ⊢ Or (Exists fun i₁ => Exists fun i₂ => Exists fun i₃ => Exists fun j₂ => Exis …
    -/
  · right
    /-
      case neg.h
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      h : Not (Membership.mem (Set.range p) t.orthocenter)
      ⊢ Eq (Set.range p) (Set.range t.points)
    -/
    have hs := Set.subset_diff_singleton hps h
    /-
      case neg.h
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      h : Not (Membership.mem (Set.range p) t.orthocenter)
      hs : HasSubset.Subset (Set.range p) (SDiff.sdiff (Insert.insert t.orthocenter  …
      ⊢ Eq (Set.range p) (Set.range t.points)
    -/
    rw [Set.insert_diff_self_of_not_mem ho] at hs
    classical
    refine Set.eq_of_subset_of_card_le hs ?_
    rw [Set.card_range_of_injective hpi, Set.card_range_of_injective t.independent.injective]


/-- For any three points in an orthocentric system generated by
triangle `t`, there is a point in the subspace spanned by the triangle
from which the distance of all those three points equals the circumradius. -/
theorem exists_dist_eq_circumradius_of_subset_insert_orthocenter {t : Triangle ℝ P}
    (ho : t.orthocenter ∉ Set.range t.points) {p : Fin 3 → P}
    (hps : Set.range p ⊆ insert t.orthocenter (Set.range t.points)) (hpi : Function.Injective p) :
    ∃ c ∈ affineSpan ℝ (Set.range t.points), ∀ p₁ ∈ Set.range p, dist p₁ c = t.circumradius := by
  rcases exists_of_range_subset_orthocentricSystem ho hps hpi with
    (⟨i₁, i₂, i₃, j₂, j₃, _, _, _, h₁₂₃, h₁, hj₂₃, h₂, h₃⟩ | hs)
  · use reflection (affineSpan ℝ (t.points '' {j₂, j₃})) t.circumcenter,
      reflection_mem_of_le_of_mem (affineSpan_mono ℝ (Set.image_subset_range _ _))
        t.circumcenter_mem_affineSpan
    /-
      case right
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ i₂ i₃ j₂ j₃ : Fin 3
      left✝² : Ne i₁ i₂
      left✝¹ : Ne i₁ i₃
      left✝ : Ne i₂ i₃
      h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h₁ : Eq (p i₁) t.orthocenter
      hj₂₃ : Ne j₂ j₃
      h₂ : Eq (t.points j₂) (p i₂)
      h₃ : Eq (t.points j₃) (p i₃)
      ⊢ ∀ (p₁ : P), Membership.mem (Set.range p) p₁ → Eq (Dist.dist p₁ ((EuclideanGe …
    -/
    intro p₁ hp₁
    /-
      case right
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ i₂ i₃ j₂ j₃ : Fin 3
      left✝² : Ne i₁ i₂
      left✝¹ : Ne i₁ i₃
      left✝ : Ne i₂ i₃
      h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h₁ : Eq (p i₁) t.orthocenter
      hj₂₃ : Ne j₂ j₃
      h₂ : Eq (t.points j₂) (p i₂)
      h₃ : Eq (t.points j₃) (p i₃)
      p₁ : P
      hp₁ : Membership.mem (Set.range p) p₁
      ⊢ Eq (Dist.dist p₁ ((EuclideanGeometry.reflection (affineSpan Real (Set.image  …
    -/
    rcases hp₁ with ⟨i, rfl⟩
    /-
      case right.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ i₂ i₃ j₂ j₃ : Fin 3
      left✝² : Ne i₁ i₂
      left✝¹ : Ne i₁ i₃
      left✝ : Ne i₂ i₃
      h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h₁ : Eq (p i₁) t.orthocenter
      hj₂₃ : Ne j₂ j₃
      h₂ : Eq (t.points j₂) (p i₂)
      h₃ : Eq (t.points j₃) (p i₃)
      i : Fin 3
      ⊢ Eq (Dist.dist (p i) ((EuclideanGeometry.reflection (affineSpan Real (Set.ima …
    -/
    have h₁₂₃ := h₁₂₃ i
    /-
      case right.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      i₁ i₂ i₃ j₂ j₃ : Fin 3
      left✝² : Ne i₁ i₂
      left✝¹ : Ne i₁ i₃
      left✝ : Ne i₂ i₃
      h₁₂₃✝ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      h₁ : Eq (p i₁) t.orthocenter
      hj₂₃ : Ne j₂ j₃
      h₂ : Eq (t.points j₂) (p i₂)
      h₃ : Eq (t.points j₃) (p i₃)
      i : Fin 3
      h₁₂₃ : Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
      ⊢ Eq (Dist.dist (p i) ((EuclideanGeometry.reflection (affineSpan Real (Set.ima …
    -/
    repeat' cases' h₁₂₃ with h₁₂₃ h₁₂₃
      /-
        case right.intro.inl.refl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        t : Affine.Triangle Real P
        ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
        p : Fin 3 → P
        hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
        hpi : Function.Injective p
        i₁ i₂ i₃ j₂ j₃ : Fin 3
        left✝² : Ne i₁ i₂
        left✝¹ : Ne i₁ i₃
        left✝ : Ne i₂ i₃
        h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
        h₁ : Eq (p i₁) t.orthocenter
        hj₂₃ : Ne j₂ j₃
        h₂ : Eq (t.points j₂) (p i₂)
        h₃ : Eq (t.points j₃) (p i₃)
        ⊢ Eq (Dist.dist (p i₁) ((EuclideanGeometry.reflection (affineSpan Real (Set.im …
      -/
    · convert Triangle.dist_orthocenter_reflection_circumcenter t hj₂₃
      /-
        🎉 no goals
      -/
    · rw [← h₂, dist_reflection_eq_of_mem _
       (mem_affineSpan ℝ (Set.mem_image_of_mem _ (Set.mem_insert _ _)))]
      /-
        case right.intro.inr.inl.refl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        t : Affine.Triangle Real P
        ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
        p : Fin 3 → P
        hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
        hpi : Function.Injective p
        i₁ i₂ i₃ j₂ j₃ : Fin 3
        left✝² : Ne i₁ i₂
        left✝¹ : Ne i₁ i₃
        left✝ : Ne i₂ i₃
        h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
        h₁ : Eq (p i₁) t.orthocenter
        hj₂₃ : Ne j₂ j₃
        h₂ : Eq (t.points j₂) (p i₂)
        h₃ : Eq (t.points j₃) (p i₃)
        ⊢ Eq (Dist.dist (t.points j₂) (Affine.Simplex.circumcenter t)) (Affine.Simplex …
      -/
      exact t.dist_circumcenter_eq_circumradius _
      /-
        🎉 no goals
      -/
    · rw [← h₃,
        dist_reflection_eq_of_mem _
          (mem_affineSpan ℝ
            (Set.mem_image_of_mem _ (Set.mem_insert_of_mem _ (Set.mem_singleton _))))]
      /-
        case right.intro.inr.inr.refl
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        t : Affine.Triangle Real P
        ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
        p : Fin 3 → P
        hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
        hpi : Function.Injective p
        i₁ i₂ i₃ j₂ j₃ : Fin 3
        left✝² : Ne i₁ i₂
        left✝¹ : Ne i₁ i₃
        left✝ : Ne i₂ i₃
        h₁₂₃ : ∀ (i : Fin 3), Or (Eq i i₁) (Or (Eq i i₂) (Eq i i₃))
        h₁ : Eq (p i₁) t.orthocenter
        hj₂₃ : Ne j₂ j₃
        h₂ : Eq (t.points j₂) (p i₂)
        h₃ : Eq (t.points j₃) (p i₃)
        ⊢ Eq (Dist.dist (t.points j₃) (Affine.Simplex.circumcenter t)) (Affine.Simplex …
      -/
      exact t.dist_circumcenter_eq_circumradius _
      /-
        🎉 no goals
      -/
    /-
      case inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      hs : Eq (Set.range p) (Set.range t.points)
      ⊢ Exists fun c => And (Membership.mem (affineSpan Real (Set.range t.points)) c …
    -/
  · use t.circumcenter, t.circumcenter_mem_affineSpan
    /-
      case right
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      hs : Eq (Set.range p) (Set.range t.points)
      ⊢ ∀ (p₁ : P), Membership.mem (Set.range p) p₁ → Eq (Dist.dist p₁ (Affine.Simpl …
    -/
    intro p₁ hp₁
    /-
      case right
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      hs : Eq (Set.range p) (Set.range t.points)
      p₁ : P
      hp₁ : Membership.mem (Set.range p) p₁
      ⊢ Eq (Dist.dist p₁ (Affine.Simplex.circumcenter t)) (Affine.Simplex.circumradi …
    -/
    rw [hs] at hp₁
    /-
      case right
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      hs : Eq (Set.range p) (Set.range t.points)
      p₁ : P
      hp₁ : Membership.mem (Set.range t.points) p₁
      ⊢ Eq (Dist.dist p₁ (Affine.Simplex.circumcenter t)) (Affine.Simplex.circumradi …
    -/
    rcases hp₁ with ⟨i, rfl⟩
    /-
      case right.intro
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      t : Affine.Triangle Real P
      ho : Not (Membership.mem (Set.range t.points) t.orthocenter)
      p : Fin 3 → P
      hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
      hpi : Function.Injective p
      hs : Eq (Set.range p) (Set.range t.points)
      i : Fin (HAdd.hAdd 2 1)
      ⊢ Eq (Dist.dist (t.points i) (Affine.Simplex.circumcenter t)) (Affine.Simplex. …
    -/
    exact t.dist_circumcenter_eq_circumradius _
    /-
      🎉 no goals
    -/


/-- Any three points in an orthocentric system are affinely independent. -/
theorem OrthocentricSystem.affineIndependent {s : Set P} (ho : OrthocentricSystem s) {p : Fin 3 → P}
    (hps : Set.range p ⊆ s) (hpi : Function.Injective p) : AffineIndependent ℝ p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    ho : EuclideanGeometry.OrthocentricSystem s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ⊢ AffineIndependent Real p
  -/
  rcases ho with ⟨t, hto, hst⟩
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hst : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    ⊢ AffineIndependent Real p
  -/
  rw [hst] at hps
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hpi : Function.Injective p
    t : Affine.Triangle Real P
    hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hst : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    ⊢ AffineIndependent Real p
  -/
  rcases exists_dist_eq_circumradius_of_subset_insert_orthocenter hto hps hpi with ⟨c, _, hc⟩
  /-
    case intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hpi : Function.Injective p
    t : Affine.Triangle Real P
    hps : HasSubset.Subset (Set.range p) (Insert.insert t.orthocenter (Set.range t …
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hst : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    c : P
    left✝ : Membership.mem (affineSpan Real (Set.range t.points)) c
    hc : ∀ (p₁ : P), Membership.mem (Set.range p) p₁ → Eq (Dist.dist p₁ c) (Affine …
    ⊢ AffineIndependent Real p
  -/
  exact Cospherical.affineIndependent ⟨c, t.circumradius, hc⟩ Set.Subset.rfl hpi
  /-
    🎉 no goals
  -/


/-- Any three points in an orthocentric system span the same subspace
as the whole orthocentric system. -/
theorem affineSpan_of_orthocentricSystem {s : Set P} (ho : OrthocentricSystem s) {p : Fin 3 → P}
    (hps : Set.range p ⊆ s) (hpi : Function.Injective p) :
    affineSpan ℝ (Set.range p) = affineSpan ℝ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    ho : EuclideanGeometry.OrthocentricSystem s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ⊢ Eq (affineSpan Real (Set.range p)) (affineSpan Real s)
  -/
  have ha := ho.affineIndependent hps hpi
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    ho : EuclideanGeometry.OrthocentricSystem s
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ha : AffineIndependent Real p
    ⊢ Eq (affineSpan Real (Set.range p)) (affineSpan Real s)
  -/
  rcases ho with ⟨t, _, hts⟩
  have hs : affineSpan ℝ s = affineSpan ℝ (Set.range t.points) := by
    rw [hts, affineSpan_insert_eq_affineSpan ℝ t.orthocenter_mem_affineSpan]
  refine ext_of_direction_eq ?_
    ⟨p 0, mem_affineSpan ℝ (Set.mem_range_self _), mem_affineSpan ℝ (hps (Set.mem_range_self _))⟩
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ha : AffineIndependent Real p
    t : Affine.Triangle Real P
    left✝ : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    hs : Eq (affineSpan Real s) (affineSpan Real (Set.range t.points))
    ⊢ Eq (affineSpan Real (Set.range p)).direction (affineSpan Real s).direction
  -/
  have hfd : FiniteDimensional ℝ (affineSpan ℝ s).direction := by rw [hs]; infer_instance
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ha : AffineIndependent Real p
    t : Affine.Triangle Real P
    left✝ : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    hs : Eq (affineSpan Real s) (affineSpan Real (Set.range t.points))
    hfd : FiniteDimensional Real (Subtype fun x => Membership.mem (affineSpan Real …
    ⊢ Eq (affineSpan Real (Set.range p)).direction (affineSpan Real s).direction
  -/
  haveI := hfd
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    p : Fin 3 → P
    hps : HasSubset.Subset (Set.range p) s
    hpi : Function.Injective p
    ha : AffineIndependent Real p
    t : Affine.Triangle Real P
    left✝ : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    hs : Eq (affineSpan Real s) (affineSpan Real (Set.range t.points))
    hfd this : FiniteDimensional Real (Subtype fun x => Membership.mem (affineSpan …
    ⊢ Eq (affineSpan Real (Set.range p)).direction (affineSpan Real s).direction
  -/
  refine Submodule.eq_of_le_of_finrank_eq (direction_le (affineSpan_mono ℝ hps)) ?_
  rw [hs, direction_affineSpan, direction_affineSpan, ha.finrank_vectorSpan (Fintype.card_fin _),
    t.independent.finrank_vectorSpan (Fintype.card_fin _)]


/-- All triangles in an orthocentric system have the same circumradius. -/
theorem OrthocentricSystem.exists_circumradius_eq {s : Set P} (ho : OrthocentricSystem s) :
    ∃ r : ℝ, ∀ t : Triangle ℝ P, Set.range t.points ⊆ s → t.circumradius = r := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    ho : EuclideanGeometry.OrthocentricSystem s
    ⊢ Exists fun r => ∀ (t : Affine.Triangle Real P), HasSubset.Subset (Set.range  …
  -/
  rcases ho with ⟨t, hto, hts⟩
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    ⊢ Exists fun r => ∀ (t : Affine.Triangle Real P), HasSubset.Subset (Set.range  …
  -/
  use t.circumradius
  /-
    case h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    ⊢ ∀ (t_1 : Affine.Triangle Real P), HasSubset.Subset (Set.range t_1.points) s  …
  -/
  intro t₂ ht₂
  /-
    case h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    t₂ : Affine.Triangle Real P
    ht₂ : HasSubset.Subset (Set.range t₂.points) s
    ⊢ Eq (Affine.Simplex.circumradius t₂) (Affine.Simplex.circumradius t)
  -/
  have ht₂s := ht₂
  /-
    case h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    t₂ : Affine.Triangle Real P
    ht₂ ht₂s : HasSubset.Subset (Set.range t₂.points) s
    ⊢ Eq (Affine.Simplex.circumradius t₂) (Affine.Simplex.circumradius t)
  -/
  rw [hts] at ht₂
  rcases exists_dist_eq_circumradius_of_subset_insert_orthocenter hto ht₂
      t₂.independent.injective with
    ⟨c, hc, h⟩
  /-
    case h.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    t₂ : Affine.Triangle Real P
    ht₂ : HasSubset.Subset (Set.range t₂.points) (Insert.insert t.orthocenter (Set …
    ht₂s : HasSubset.Subset (Set.range t₂.points) s
    c : P
    hc : Membership.mem (affineSpan Real (Set.range t.points)) c
    h : ∀ (p₁ : P), Membership.mem (Set.range t₂.points) p₁ → Eq (Dist.dist p₁ c)  …
    ⊢ Eq (Affine.Simplex.circumradius t₂) (Affine.Simplex.circumradius t)
  -/
  rw [Set.forall_mem_range] at h
  have hs : Set.range t.points ⊆ s := by
    rw [hts]
    exact Set.subset_insert _ _
  rw [affineSpan_of_orthocentricSystem ⟨t, hto, hts⟩ hs t.independent.injective,
    ← affineSpan_of_orthocentricSystem ⟨t, hto, hts⟩ ht₂s t₂.independent.injective] at hc
  /-
    case h.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    hto : Not (Membership.mem (Set.range t.points) t.orthocenter)
    hts : Eq s (Insert.insert t.orthocenter (Set.range t.points))
    t₂ : Affine.Triangle Real P
    ht₂ : HasSubset.Subset (Set.range t₂.points) (Insert.insert t.orthocenter (Set …
    ht₂s : HasSubset.Subset (Set.range t₂.points) s
    c : P
    hc : Membership.mem (affineSpan Real (Set.range t₂.points)) c
    h : ∀ (i : Fin 3), Eq (Dist.dist (t₂.points i) c) (Affine.Simplex.circumradius …
    hs : HasSubset.Subset (Set.range t.points) s
    ⊢ Eq (Affine.Simplex.circumradius t₂) (Affine.Simplex.circumradius t)
  -/
  exact (t₂.eq_circumradius_of_dist_eq hc h).symm
  /-
    🎉 no goals
  -/


/-- Given any triangle in an orthocentric system, the fourth point is
its orthocenter. -/
theorem OrthocentricSystem.eq_insert_orthocenter {s : Set P} (ho : OrthocentricSystem s)
    {t : Triangle ℝ P} (ht : Set.range t.points ⊆ s) :
    s = insert t.orthocenter (Set.range t.points) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    ho : EuclideanGeometry.OrthocentricSystem s
    t : Affine.Triangle Real P
    ht : HasSubset.Subset (Set.range t.points) s
    ⊢ Eq s (Insert.insert t.orthocenter (Set.range t.points))
  -/
  rcases ho with ⟨t₀, ht₀o, ht₀s⟩
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s : Set P
    t : Affine.Triangle Real P
    ht : HasSubset.Subset (Set.range t.points) s
    t₀ : Affine.Triangle Real P
    ht₀o : Not (Membership.mem (Set.range t₀.points) t₀.orthocenter)
    ht₀s : Eq s (Insert.insert t₀.orthocenter (Set.range t₀.points))
    ⊢ Eq s (Insert.insert t.orthocenter (Set.range t.points))
  -/
  rw [ht₀s] at ht
  rcases exists_of_range_subset_orthocentricSystem ht₀o ht t.independent.injective with
    (⟨i₁, i₂, i₃, j₂, j₃, h₁₂, h₁₃, h₂₃, h₁₂₃, h₁, hj₂₃, h₂, h₃⟩ | hs)
  · obtain ⟨j₁, hj₁₂, hj₁₃, hj₁₂₃⟩ :
        ∃ j₁ : Fin 3, j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ ∀ j : Fin 3, j = j₁ ∨ j = j₂ ∨ j = j₃ := by
      clear h₂ h₃
      decide +revert
    suffices h : t₀.points j₁ = t.orthocenter by
      have hui : (Set.univ : Set (Fin 3)) = {i₁, i₂, i₃} := by ext x; simpa using h₁₂₃ x
      have huj : (Set.univ : Set (Fin 3)) = {j₁, j₂, j₃} := by ext x; simpa using hj₁₂₃ x
      rw [← h, ht₀s, ← Set.image_univ, huj, ← Set.image_univ, hui]
      simp_rw [Set.image_insert_eq, Set.image_singleton, h₁, ← h₂, ← h₃]
      rw [Set.insert_comm]
    exact
      (Triangle.orthocenter_replace_orthocenter_eq_point hj₁₂ hj₁₃ hj₂₃ h₁₂ h₁₃ h₂₃ h₁ h₂.symm
          h₃.symm).symm
    /-
      case intro.intro.inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      t t₀ : Affine.Triangle Real P
      ht : HasSubset.Subset (Set.range t.points) (Insert.insert t₀.orthocenter (Set. …
      ht₀o : Not (Membership.mem (Set.range t₀.points) t₀.orthocenter)
      ht₀s : Eq s (Insert.insert t₀.orthocenter (Set.range t₀.points))
      hs : Eq (Set.range t.points) (Set.range t₀.points)
      ⊢ Eq s (Insert.insert t.orthocenter (Set.range t.points))
    -/
  · rw [hs]
    /-
      case intro.intro.inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      t t₀ : Affine.Triangle Real P
      ht : HasSubset.Subset (Set.range t.points) (Insert.insert t₀.orthocenter (Set. …
      ht₀o : Not (Membership.mem (Set.range t₀.points) t₀.orthocenter)
      ht₀s : Eq s (Insert.insert t₀.orthocenter (Set.range t₀.points))
      hs : Eq (Set.range t.points) (Set.range t₀.points)
      ⊢ Eq s (Insert.insert t.orthocenter (Set.range t₀.points))
    -/
    convert ht₀s using 2
    /-
      case h.e'_3.h.e'_4
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      s : Set P
      t t₀ : Affine.Triangle Real P
      ht : HasSubset.Subset (Set.range t.points) (Insert.insert t₀.orthocenter (Set. …
      ht₀o : Not (Membership.mem (Set.range t₀.points) t₀.orthocenter)
      ht₀s : Eq s (Insert.insert t₀.orthocenter (Set.range t₀.points))
      hs : Eq (Set.range t.points) (Set.range t₀.points)
      ⊢ Eq t.orthocenter t₀.orthocenter
    -/
    exact Triangle.orthocenter_eq_of_range_eq hs
    /-
      🎉 no goals
    -/


