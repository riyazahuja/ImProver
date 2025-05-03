/-- Inverse of the `sin` function, returns values in the range `-π / 2 ≤ arcsin x ≤ π / 2`.
It defaults to `-π / 2` on `(-∞, -1)` and to `π / 2` to `(1, ∞)`. -/
@[pp_nodot]
noncomputable def arcsin : ℝ → ℝ :=
  Subtype.val ∘ IccExtend (neg_le_self zero_le_one) sinOrderIso.symm


theorem arcsin_mem_Icc (x : ℝ) : arcsin x ∈ Icc (-(π / 2)) (π / 2) :=
  Subtype.coe_prop _


@[simp]
theorem range_arcsin : range arcsin = Icc (-(π / 2)) (π / 2) := by
  /-
    ⊢ Eq (Set.range Real.arcsin) (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hD …
  -/
  rw [arcsin, range_comp Subtype.val]
  /-
    ⊢ Eq (Set.image Subtype.val (Set.range (Set.IccExtend Real.arcsin.proof_2 ⇑Rea …
  -/
  simp [Icc]
  /-
    🎉 no goals
  -/


theorem arcsin_le_pi_div_two (x : ℝ) : arcsin x ≤ π / 2 :=
  (arcsin_mem_Icc x).2


theorem neg_pi_div_two_le_arcsin (x : ℝ) : -(π / 2) ≤ arcsin x :=
  (arcsin_mem_Icc x).1


theorem arcsin_projIcc (x : ℝ) :
    arcsin (projIcc (-1) 1 (neg_le_self zero_le_one) x) = arcsin x := by
  rw [arcsin, Function.comp_apply, IccExtend_val, Function.comp_apply, IccExtend,
        Function.comp_apply]


theorem sin_arcsin' {x : ℝ} (hx : x ∈ Icc (-1 : ℝ) 1) : sin (arcsin x) = x := by
  simpa [arcsin, IccExtend_of_mem _ _ hx, -OrderIso.apply_symm_apply] using
    Subtype.ext_iff.1 (sinOrderIso.apply_symm_apply ⟨x, hx⟩)


theorem sin_arcsin {x : ℝ} (hx₁ : -1 ≤ x) (hx₂ : x ≤ 1) : sin (arcsin x) = x :=
  sin_arcsin' ⟨hx₁, hx₂⟩


theorem arcsin_sin' {x : ℝ} (hx : x ∈ Icc (-(π / 2)) (π / 2)) : arcsin (sin x) = x :=
                                        /-
                                          x : Real
                                          hx : Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
                                          ⊢ Eq (Real.sin (Real.arcsin (Real.sin x))) (Real.sin x)
                                        -/
  injOn_sin (arcsin_mem_Icc _) hx <| by rw [sin_arcsin (neg_one_le_sin _) (sin_le_one _)]
                                        /-
                                          🎉 no goals
                                        -/


theorem arcsin_sin {x : ℝ} (hx₁ : -(π / 2) ≤ x) (hx₂ : x ≤ π / 2) : arcsin (sin x) = x :=
  arcsin_sin' ⟨hx₁, hx₂⟩


theorem strictMonoOn_arcsin : StrictMonoOn arcsin (Icc (-1) 1) :=
  (Subtype.strictMono_coe _).comp_strictMonoOn <|
    sinOrderIso.symm.strictMono.strictMonoOn_IccExtend _


theorem monotone_arcsin : Monotone arcsin :=
  (Subtype.mono_coe _).comp <| sinOrderIso.symm.monotone.IccExtend _


theorem injOn_arcsin : InjOn arcsin (Icc (-1) 1) :=
  strictMonoOn_arcsin.injOn


theorem arcsin_inj {x y : ℝ} (hx₁ : -1 ≤ x) (hx₂ : x ≤ 1) (hy₁ : -1 ≤ y) (hy₂ : y ≤ 1) :
    arcsin x = arcsin y ↔ x = y :=
  injOn_arcsin.eq_iff ⟨hx₁, hx₂⟩ ⟨hy₁, hy₂⟩


@[continuity]
theorem continuous_arcsin : Continuous arcsin :=
  continuous_subtype_val.comp sinOrderIso.symm.continuous.Icc_extend'


theorem continuousAt_arcsin {x : ℝ} : ContinuousAt arcsin x :=
  continuous_arcsin.continuousAt


theorem arcsin_eq_of_sin_eq {x y : ℝ} (h₁ : sin x = y) (h₂ : x ∈ Icc (-(π / 2)) (π / 2)) :
    arcsin y = x := by
  /-
    x y : Real
    h₁ : Eq (Real.sin x) y
    h₂ : Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    ⊢ Eq (Real.arcsin y) x
  -/
  subst y
  /-
    x : Real
    h₂ : Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    ⊢ Eq (Real.arcsin (Real.sin x)) x
  -/
  exact injOn_sin (arcsin_mem_Icc _) h₂ (sin_arcsin' (sin_mem_Icc x))
  /-
    🎉 no goals
  -/


@[simp]
theorem arcsin_zero : arcsin 0 = 0 :=
  arcsin_eq_of_sin_eq sin_zero ⟨neg_nonpos.2 pi_div_two_pos.le, pi_div_two_pos.le⟩


@[simp]
theorem arcsin_one : arcsin 1 = π / 2 :=
  arcsin_eq_of_sin_eq sin_pi_div_two <| right_mem_Icc.2 (neg_le_self pi_div_two_pos.le)


theorem arcsin_of_one_le {x : ℝ} (hx : 1 ≤ x) : arcsin x = π / 2 := by
  /-
    x : Real
    hx : LE.le 1 x
    ⊢ Eq (Real.arcsin x) (HDiv.hDiv Real.pi 2)
  -/
  rw [← arcsin_projIcc, projIcc_of_right_le _ hx, Subtype.coe_mk, arcsin_one]
  /-
    🎉 no goals
  -/


theorem arcsin_neg_one : arcsin (-1) = -(π / 2) :=
                          /-
                            ⊢ Eq (Real.sin (Neg.neg (HDiv.hDiv Real.pi 2))) (-1)
                          -/
  arcsin_eq_of_sin_eq (by rw [sin_neg, sin_pi_div_two]) <|
                          /-
                            🎉 no goals
                          -/
    left_mem_Icc.2 (neg_le_self pi_div_two_pos.le)


theorem arcsin_of_le_neg_one {x : ℝ} (hx : x ≤ -1) : arcsin x = -(π / 2) := by
  /-
    x : Real
    hx : LE.le x (-1)
    ⊢ Eq (Real.arcsin x) (Neg.neg (HDiv.hDiv Real.pi 2))
  -/
  rw [← arcsin_projIcc, projIcc_of_le_left _ hx, Subtype.coe_mk, arcsin_neg_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem arcsin_neg (x : ℝ) : arcsin (-x) = -arcsin x := by
  /-
    x : Real
    ⊢ Eq (Real.arcsin (Neg.neg x)) (Neg.neg (Real.arcsin x))
  -/
  rcases le_total x (-1) with hx₁ | hx₁
    /-
      case inl
      x : Real
      hx₁ : LE.le x (-1)
      ⊢ Eq (Real.arcsin (Neg.neg x)) (Neg.neg (Real.arcsin x))
    -/
  · rw [arcsin_of_le_neg_one hx₁, neg_neg, arcsin_of_one_le (le_neg.2 hx₁)]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : Real
    hx₁ : LE.le (-1) x
    ⊢ Eq (Real.arcsin (Neg.neg x)) (Neg.neg (Real.arcsin x))
  -/
  rcases le_total 1 x with hx₂ | hx₂
    /-
      case inr.inl
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LE.le 1 x
      ⊢ Eq (Real.arcsin (Neg.neg x)) (Neg.neg (Real.arcsin x))
    -/
  · rw [arcsin_of_one_le hx₂, arcsin_of_le_neg_one (neg_le_neg hx₂)]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    x : Real
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    ⊢ Eq (Real.arcsin (Neg.neg x)) (Neg.neg (Real.arcsin x))
  -/
  refine arcsin_eq_of_sin_eq ?_ ?_
    /-
      case inr.inr.refine_1
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LE.le x 1
      ⊢ Eq (Real.sin (Neg.neg (Real.arcsin x))) (Neg.neg x)
    -/
  · rw [sin_neg, sin_arcsin hx₁ hx₂]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.refine_2
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LE.le x 1
      ⊢ Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2 …
    -/
  · exact ⟨neg_le_neg (arcsin_le_pi_div_two _), neg_le.2 (neg_pi_div_two_le_arcsin _)⟩
    /-
      🎉 no goals
    -/


theorem arcsin_le_iff_le_sin {x y : ℝ} (hx : x ∈ Icc (-1 : ℝ) 1) (hy : y ∈ Icc (-(π / 2)) (π / 2)) :
    arcsin x ≤ y ↔ x ≤ sin y := by
  /-
    x y : Real
    hx : Membership.mem (Set.Icc (-1) 1) x
    hy : Membership.mem (Set.Icc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    ⊢ Iff (LE.le (Real.arcsin x) y) (LE.le x (Real.sin y))
  -/
  rw [← arcsin_sin' hy, strictMonoOn_arcsin.le_iff_le hx (sin_mem_Icc _), arcsin_sin' hy]
  /-
    🎉 no goals
  -/


theorem arcsin_le_iff_le_sin' {x y : ℝ} (hy : y ∈ Ico (-(π / 2)) (π / 2)) :
    arcsin x ≤ y ↔ x ≤ sin y := by
  /-
    x y : Real
    hy : Membership.mem (Set.Ico (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    ⊢ Iff (LE.le (Real.arcsin x) y) (LE.le x (Real.sin y))
  -/
  rcases le_total x (-1) with hx₁ | hx₁
    /-
      case inl
      x y : Real
      hy : Membership.mem (Set.Ico (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
      hx₁ : LE.le x (-1)
      ⊢ Iff (LE.le (Real.arcsin x) y) (LE.le x (Real.sin y))
    -/
  · simp [arcsin_of_le_neg_one hx₁, hy.1, hx₁.trans (neg_one_le_sin _)]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x y : Real
    hy : Membership.mem (Set.Ico (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    hx₁ : LE.le (-1) x
    ⊢ Iff (LE.le (Real.arcsin x) y) (LE.le x (Real.sin y))
  -/
  cases' lt_or_le 1 x with hx₂ hx₂
    /-
      case inr.inl
      x y : Real
      hy : Membership.mem (Set.Ico (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
      hx₁ : LE.le (-1) x
      hx₂ : LT.lt 1 x
      ⊢ Iff (LE.le (Real.arcsin x) y) (LE.le x (Real.sin y))
    -/
  · simp [arcsin_of_one_le hx₂.le, hy.2.not_le, (sin_le_one y).trans_lt hx₂]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    x y : Real
    hy : Membership.mem (Set.Ico (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    ⊢ Iff (LE.le (Real.arcsin x) y) (LE.le x (Real.sin y))
  -/
  exact arcsin_le_iff_le_sin ⟨hx₁, hx₂⟩ (mem_Icc_of_Ico hy)
  /-
    🎉 no goals
  -/


theorem le_arcsin_iff_sin_le {x y : ℝ} (hx : x ∈ Icc (-(π / 2)) (π / 2)) (hy : y ∈ Icc (-1 : ℝ) 1) :
    x ≤ arcsin y ↔ sin x ≤ y := by
  rw [← neg_le_neg_iff, ← arcsin_neg,
    arcsin_le_iff_le_sin ⟨neg_le_neg hy.2, neg_le.2 hy.1⟩ ⟨neg_le_neg hx.2, neg_le.2 hx.1⟩, sin_neg,
    neg_le_neg_iff]


theorem le_arcsin_iff_sin_le' {x y : ℝ} (hx : x ∈ Ioc (-(π / 2)) (π / 2)) :
    x ≤ arcsin y ↔ sin x ≤ y := by
  rw [← neg_le_neg_iff, ← arcsin_neg, arcsin_le_iff_le_sin' ⟨neg_le_neg hx.2, neg_lt.2 hx.1⟩,
    sin_neg, neg_le_neg_iff]


theorem arcsin_lt_iff_lt_sin {x y : ℝ} (hx : x ∈ Icc (-1 : ℝ) 1) (hy : y ∈ Icc (-(π / 2)) (π / 2)) :
    arcsin x < y ↔ x < sin y :=
  not_le.symm.trans <| (not_congr <| le_arcsin_iff_sin_le hy hx).trans not_le


theorem arcsin_lt_iff_lt_sin' {x y : ℝ} (hy : y ∈ Ioc (-(π / 2)) (π / 2)) :
    arcsin x < y ↔ x < sin y :=
  not_le.symm.trans <| (not_congr <| le_arcsin_iff_sin_le' hy).trans not_le


theorem lt_arcsin_iff_sin_lt {x y : ℝ} (hx : x ∈ Icc (-(π / 2)) (π / 2)) (hy : y ∈ Icc (-1 : ℝ) 1) :
    x < arcsin y ↔ sin x < y :=
  not_le.symm.trans <| (not_congr <| arcsin_le_iff_le_sin hy hx).trans not_le


theorem lt_arcsin_iff_sin_lt' {x y : ℝ} (hx : x ∈ Ico (-(π / 2)) (π / 2)) :
    x < arcsin y ↔ sin x < y :=
  not_le.symm.trans <| (not_congr <| arcsin_le_iff_le_sin' hx).trans not_le


theorem arcsin_eq_iff_eq_sin {x y : ℝ} (hy : y ∈ Ioo (-(π / 2)) (π / 2)) :
    arcsin x = y ↔ x = sin y := by
  simp only [le_antisymm_iff, arcsin_le_iff_le_sin' (mem_Ico_of_Ioo hy),
    le_arcsin_iff_sin_le' (mem_Ioc_of_Ioo hy)]


@[simp]
theorem arcsin_nonneg {x : ℝ} : 0 ≤ arcsin x ↔ 0 ≤ x :=
  (le_arcsin_iff_sin_le' ⟨neg_lt_zero.2 pi_div_two_pos, pi_div_two_pos.le⟩).trans <| by
    /-
      x : Real
      ⊢ Iff (LE.le (Real.sin 0) x) (LE.le 0 x)
    -/
    rw [sin_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem arcsin_nonpos {x : ℝ} : arcsin x ≤ 0 ↔ x ≤ 0 :=
  neg_nonneg.symm.trans <| arcsin_neg x ▸ arcsin_nonneg.trans neg_nonneg


@[simp]
                                                                /-
                                                                  x : Real
                                                                  ⊢ Iff (Eq (Real.arcsin x) 0) (Eq x 0)
                                                                -/
theorem arcsin_eq_zero_iff {x : ℝ} : arcsin x = 0 ↔ x = 0 := by simp [le_antisymm_iff]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem zero_eq_arcsin_iff {x} : 0 = arcsin x ↔ x = 0 :=
  eq_comm.trans arcsin_eq_zero_iff


@[simp]
theorem arcsin_pos {x : ℝ} : 0 < arcsin x ↔ 0 < x :=
  lt_iff_lt_of_le_iff_le arcsin_nonpos


@[simp]
theorem arcsin_lt_zero {x : ℝ} : arcsin x < 0 ↔ x < 0 :=
  lt_iff_lt_of_le_iff_le arcsin_nonneg


@[simp]
theorem arcsin_lt_pi_div_two {x : ℝ} : arcsin x < π / 2 ↔ x < 1 :=
  (arcsin_lt_iff_lt_sin' (right_mem_Ioc.2 <| neg_lt_self pi_div_two_pos)).trans <| by
    /-
      x : Real
      ⊢ Iff (LT.lt x (Real.sin (HDiv.hDiv Real.pi 2))) (LT.lt x 1)
    -/
    rw [sin_pi_div_two]
    /-
      🎉 no goals
    -/


@[simp]
theorem neg_pi_div_two_lt_arcsin {x : ℝ} : -(π / 2) < arcsin x ↔ -1 < x :=
  (lt_arcsin_iff_sin_lt' <| left_mem_Ico.2 <| neg_lt_self pi_div_two_pos).trans <| by
    /-
      x : Real
      ⊢ Iff (LT.lt (Real.sin (Neg.neg (HDiv.hDiv Real.pi 2))) x) (LT.lt (-1) x)
    -/
    rw [sin_neg, sin_pi_div_two]
    /-
      🎉 no goals
    -/


@[simp]
theorem arcsin_eq_pi_div_two {x : ℝ} : arcsin x = π / 2 ↔ 1 ≤ x :=
  ⟨fun h => not_lt.1 fun h' => (arcsin_lt_pi_div_two.2 h').ne h, arcsin_of_one_le⟩


@[simp]
theorem pi_div_two_eq_arcsin {x} : π / 2 = arcsin x ↔ 1 ≤ x :=
  eq_comm.trans arcsin_eq_pi_div_two


@[simp]
theorem pi_div_two_le_arcsin {x} : π / 2 ≤ arcsin x ↔ 1 ≤ x :=
  (arcsin_le_pi_div_two x).le_iff_eq.trans pi_div_two_eq_arcsin


@[simp]
theorem arcsin_eq_neg_pi_div_two {x : ℝ} : arcsin x = -(π / 2) ↔ x ≤ -1 :=
  ⟨fun h => not_lt.1 fun h' => (neg_pi_div_two_lt_arcsin.2 h').ne' h, arcsin_of_le_neg_one⟩


@[simp]
theorem neg_pi_div_two_eq_arcsin {x} : -(π / 2) = arcsin x ↔ x ≤ -1 :=
  eq_comm.trans arcsin_eq_neg_pi_div_two


@[simp]
theorem arcsin_le_neg_pi_div_two {x} : arcsin x ≤ -(π / 2) ↔ x ≤ -1 :=
  (neg_pi_div_two_le_arcsin x).le_iff_eq.trans arcsin_eq_neg_pi_div_two


@[simp]
theorem pi_div_four_le_arcsin {x} : π / 4 ≤ arcsin x ↔ √2 / 2 ≤ x := by
  /-
    x : Real
    ⊢ Iff (LE.le (HDiv.hDiv Real.pi 4) (Real.arcsin x)) (LE.le (HDiv.hDiv (Real.sq …
  -/
  rw [← sin_pi_div_four, le_arcsin_iff_sin_le']
  /-
    x : Real
    ⊢ Membership.mem (Set.Ioc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2 …
  -/
  have := pi_pos
  /-
    x : Real
    this : LT.lt 0 Real.pi
    ⊢ Membership.mem (Set.Ioc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2 …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> linarith
                  /-
                    🎉 no goals
                  -/


theorem mapsTo_sin_Ioo : MapsTo sin (Ioo (-(π / 2)) (π / 2)) (Ioo (-1) 1) := fun x h => by
  /-
    x : Real
    h : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi …
    ⊢ Membership.mem (Set.Ioo (-1) 1) (Real.sin x)
  -/
  rwa [mem_Ioo, ← arcsin_lt_pi_div_two, ← neg_pi_div_two_lt_arcsin, arcsin_sin h.1.le h.2.le]
  /-
    🎉 no goals
  -/


/-- `Real.sin` as a `PartialHomeomorph` between `(-π / 2, π / 2)` and `(-1, 1)`. -/
@[simp]
def sinPartialHomeomorph : PartialHomeomorph ℝ ℝ where
  toFun := sin
  invFun := arcsin
  source := Ioo (-(π / 2)) (π / 2)
  target := Ioo (-1) 1
  map_source' := mapsTo_sin_Ioo
  map_target' _ hy := ⟨neg_pi_div_two_lt_arcsin.2 hy.1, arcsin_lt_pi_div_two.2 hy.2⟩
  left_inv' _ hx := arcsin_sin hx.1.le hx.2.le
  right_inv' _ hy := sin_arcsin hy.1.le hy.2.le
  open_source := isOpen_Ioo
  open_target := isOpen_Ioo
  continuousOn_toFun := continuous_sin.continuousOn
  continuousOn_invFun := continuous_arcsin.continuousOn


theorem cos_arcsin_nonneg (x : ℝ) : 0 ≤ cos (arcsin x) :=
  cos_nonneg_of_mem_Icc ⟨neg_pi_div_two_le_arcsin _, arcsin_le_pi_div_two _⟩

-- The junk values for `arcsin` and `sqrt` make this true even outside `[-1, 1]`.

theorem cos_arcsin (x : ℝ) : cos (arcsin x) = √(1 - x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  by_cases hx₁ : -1 ≤ x; swap
    /-
      case neg
      x : Real
      hx₁ : Not (LE.le (-1) x)
      ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
  · rw [not_le] at hx₁
    /-
      case neg
      x : Real
      hx₁ : LT.lt x (-1)
      ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
    rw [arcsin_of_le_neg_one hx₁.le, cos_neg, cos_pi_div_two, sqrt_eq_zero_of_nonpos]
    /-
      case neg
      x : Real
      hx₁ : LT.lt x (-1)
      ⊢ LE.le (HSub.hSub 1 (HPow.hPow x 2)) 0
    -/
    nlinarith
    /-
      🎉 no goals
    -/
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  by_cases hx₂ : x ≤ 1; swap
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : Not (LE.le x 1)
      ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
  · rw [not_le] at hx₂
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LT.lt 1 x
      ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
    rw [arcsin_of_one_le hx₂.le, cos_pi_div_two, sqrt_eq_zero_of_nonpos]
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LT.lt 1 x
      ⊢ LE.le (HSub.hSub 1 (HPow.hPow x 2)) 0
    -/
    nlinarith
    /-
      🎉 no goals
    -/
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  have : sin (arcsin x) ^ 2 + cos (arcsin x) ^ 2 = 1 := sin_sq_add_cos_sq (arcsin x)
  rw [← eq_sub_iff_add_eq', ← sqrt_inj (sq_nonneg _) (sub_nonneg.2 (sin_sq_le_one (arcsin x))), sq,
    sqrt_mul_self (cos_arcsin_nonneg _)] at this
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    this : Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow (Real.sin (Real.a …
    ⊢ Eq (Real.cos (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  rw [this, sin_arcsin hx₁ hx₂]
  /-
    🎉 no goals
  -/

-- The junk values for `arcsin` and `sqrt` make this true even outside `[-1, 1]`.

theorem tan_arcsin (x : ℝ) : tan (arcsin x) = x / √(1 - x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.tan (Real.arcsin x)) (HDiv.hDiv x (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
  -/
  rw [tan_eq_sin_div_cos, cos_arcsin]
  /-
    x : Real
    ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
  -/
  by_cases hx₁ : -1 ≤ x; swap
    /-
      case neg
      x : Real
      hx₁ : Not (LE.le (-1) x)
      ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
    -/
  · have h : √(1 - x ^ 2) = 0 := sqrt_eq_zero_of_nonpos (by nlinarith)
    /-
      case neg
      x : Real
      hx₁ : Not (LE.le (-1) x)
      h : Eq (HSub.hSub 1 (HPow.hPow x 2)).sqrt 0
      ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
    -/
    rw [h]
    /-
      case neg
      x : Real
      hx₁ : Not (LE.le (-1) x)
      h : Eq (HSub.hSub 1 (HPow.hPow x 2)).sqrt 0
      ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) 0) (HDiv.hDiv x 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
  -/
  by_cases hx₂ : x ≤ 1; swap
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : Not (LE.le x 1)
      ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
    -/
  · have h : √(1 - x ^ 2) = 0 := sqrt_eq_zero_of_nonpos (by nlinarith)
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : Not (LE.le x 1)
      h : Eq (HSub.hSub 1 (HPow.hPow x 2)).sqrt 0
      ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
    -/
    rw [h]
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : Not (LE.le x 1)
      h : Eq (HSub.hSub 1 (HPow.hPow x 2)).sqrt 0
      ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) 0) (HDiv.hDiv x 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    ⊢ Eq (HDiv.hDiv (Real.sin (Real.arcsin x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt) …
  -/
  rw [sin_arcsin hx₁ hx₂]
  /-
    🎉 no goals
  -/


/-- Inverse of the `cos` function, returns values in the range `0 ≤ arccos x` and `arccos x ≤ π`.
  It defaults to `π` on `(-∞, -1)` and to `0` to `(1, ∞)`. -/
@[pp_nodot]
noncomputable def arccos (x : ℝ) : ℝ :=
  π / 2 - arcsin x


theorem arccos_eq_pi_div_two_sub_arcsin (x : ℝ) : arccos x = π / 2 - arcsin x :=
  rfl


                                                                                    /-
                                                                                      x : Real
                                                                                      ⊢ Eq (Real.arcsin x) (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arccos x))
                                                                                    -/
theorem arcsin_eq_pi_div_two_sub_arccos (x : ℝ) : arcsin x = π / 2 - arccos x := by simp [arccos]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem arccos_le_pi (x : ℝ) : arccos x ≤ π := by
  /-
    x : Real
    ⊢ LE.le (Real.arccos x) Real.pi
  -/
  unfold arccos; linarith [neg_pi_div_two_le_arcsin x]
                 /-
                   🎉 no goals
                 -/


theorem arccos_nonneg (x : ℝ) : 0 ≤ arccos x := by
  /-
    x : Real
    ⊢ LE.le 0 (Real.arccos x)
  -/
  unfold arccos; linarith [arcsin_le_pi_div_two x]
                 /-
                   🎉 no goals
                 -/


@[simp]
                                                        /-
                                                          x : Real
                                                          ⊢ Iff (LT.lt 0 (Real.arccos x)) (LT.lt x 1)
                                                        -/
theorem arccos_pos {x : ℝ} : 0 < arccos x ↔ x < 1 := by simp [arccos]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem cos_arccos {x : ℝ} (hx₁ : -1 ≤ x) (hx₂ : x ≤ 1) : cos (arccos x) = x := by
  /-
    x : Real
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    ⊢ Eq (Real.cos (Real.arccos x)) x
  -/
  rw [arccos, cos_pi_div_two_sub, sin_arcsin hx₁ hx₂]
  /-
    🎉 no goals
  -/


theorem arccos_cos {x : ℝ} (hx₁ : 0 ≤ x) (hx₂ : x ≤ π) : arccos (cos x) = x := by
  /-
    x : Real
    hx₁ : LE.le 0 x
    hx₂ : LE.le x Real.pi
    ⊢ Eq (Real.arccos (Real.cos x)) x
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  rw [arccos, ← sin_pi_div_two_sub, arcsin_sin] <;> simp [sub_eq_add_neg] <;> linarith
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


lemma arccos_eq_of_eq_cos (hy₀ : 0 ≤ y) (hy₁ : y ≤ π) (hxy : x = cos y) : arccos x = y := by
  /-
    x y : Real
    hy₀ : LE.le 0 y
    hy₁ : LE.le y Real.pi
    hxy : Eq x (Real.cos y)
    ⊢ Eq (Real.arccos x) y
  -/
  rw [hxy, arccos_cos hy₀ hy₁]
  /-
    🎉 no goals
  -/


theorem strictAntiOn_arccos : StrictAntiOn arccos (Icc (-1) 1) := fun _ hx _ hy h =>
  sub_lt_sub_left (strictMonoOn_arcsin hx hy h) _


theorem arccos_injOn : InjOn arccos (Icc (-1) 1) :=
  strictAntiOn_arccos.injOn


theorem arccos_inj {x y : ℝ} (hx₁ : -1 ≤ x) (hx₂ : x ≤ 1) (hy₁ : -1 ≤ y) (hy₂ : y ≤ 1) :
    arccos x = arccos y ↔ x = y :=
  arccos_injOn.eq_iff ⟨hx₁, hx₂⟩ ⟨hy₁, hy₂⟩


@[simp]
                                             /-
                                               ⊢ Eq (Real.arccos 0) (HDiv.hDiv Real.pi 2)
                                             -/
theorem arccos_zero : arccos 0 = π / 2 := by simp [arccos]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                        /-
                                          ⊢ Eq (Real.arccos 1) 0
                                        -/
theorem arccos_one : arccos 1 = 0 := by simp [arccos]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
                                               /-
                                                 ⊢ Eq (Real.arccos (-1)) Real.pi
                                               -/
theorem arccos_neg_one : arccos (-1) = π := by simp [arccos, add_halves]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                        /-
                                                          x : Real
                                                          ⊢ Iff (Eq (Real.arccos x) 0) (LE.le 1 x)
                                                        -/
theorem arccos_eq_zero {x} : arccos x = 0 ↔ 1 ≤ x := by simp [arccos, sub_eq_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
                                                                  /-
                                                                    x : Real
                                                                    ⊢ Iff (Eq (Real.arccos x) (HDiv.hDiv Real.pi 2)) (Eq x 0)
                                                                  -/
theorem arccos_eq_pi_div_two {x} : arccos x = π / 2 ↔ x = 0 := by simp [arccos]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem arccos_eq_pi {x} : arccos x = π ↔ x ≤ -1 := by
  /-
    x : Real
    ⊢ Iff (Eq (Real.arccos x) Real.pi) (LE.le x (-1))
  -/
  rw [arccos, sub_eq_iff_eq_add, ← sub_eq_iff_eq_add', div_two_sub_self, neg_pi_div_two_eq_arcsin]
  /-
    🎉 no goals
  -/


theorem arccos_neg (x : ℝ) : arccos (-x) = π - arccos x := by
  /-
    x : Real
    ⊢ Eq (Real.arccos (Neg.neg x)) (HSub.hSub Real.pi (Real.arccos x))
  -/
  rw [← add_halves π, arccos, arcsin_neg, arccos, add_sub_assoc, sub_sub_self, sub_neg_eq_add]
  /-
    🎉 no goals
  -/


theorem arccos_of_one_le {x : ℝ} (hx : 1 ≤ x) : arccos x = 0 := by
  /-
    x : Real
    hx : LE.le 1 x
    ⊢ Eq (Real.arccos x) 0
  -/
  rw [arccos, arcsin_of_one_le hx, sub_self]
  /-
    🎉 no goals
  -/


theorem arccos_of_le_neg_one {x : ℝ} (hx : x ≤ -1) : arccos x = π := by
  /-
    x : Real
    hx : LE.le x (-1)
    ⊢ Eq (Real.arccos x) Real.pi
  -/
  rw [arccos, arcsin_of_le_neg_one hx, sub_neg_eq_add, add_halves]
  /-
    🎉 no goals
  -/

-- The junk values for `arccos` and `sqrt` make this true even outside `[-1, 1]`.

theorem sin_arccos (x : ℝ) : sin (arccos x) = √(1 - x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  by_cases hx₁ : -1 ≤ x; swap
    /-
      case neg
      x : Real
      hx₁ : Not (LE.le (-1) x)
      ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
  · rw [not_le] at hx₁
    /-
      case neg
      x : Real
      hx₁ : LT.lt x (-1)
      ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
    rw [arccos_of_le_neg_one hx₁.le, sin_pi, sqrt_eq_zero_of_nonpos]
    /-
      case neg
      x : Real
      hx₁ : LT.lt x (-1)
      ⊢ LE.le (HSub.hSub 1 (HPow.hPow x 2)) 0
    -/
    nlinarith
    /-
      🎉 no goals
    -/
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  by_cases hx₂ : x ≤ 1; swap
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : Not (LE.le x 1)
      ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
  · rw [not_le] at hx₂
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LT.lt 1 x
      ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
    -/
    rw [arccos_of_one_le hx₂.le, sin_zero, sqrt_eq_zero_of_nonpos]
    /-
      case neg
      x : Real
      hx₁ : LE.le (-1) x
      hx₂ : LT.lt 1 x
      ⊢ LE.le (HSub.hSub 1 (HPow.hPow x 2)) 0
    -/
    nlinarith
    /-
      🎉 no goals
    -/
  /-
    case pos
    x : Real
    hx₁ : LE.le (-1) x
    hx₂ : LE.le x 1
    ⊢ Eq (Real.sin (Real.arccos x)) (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  rw [arccos_eq_pi_div_two_sub_arcsin, sin_pi_div_two_sub, cos_arcsin]
  /-
    🎉 no goals
  -/


@[simp]
                                                                  /-
                                                                    x : Real
                                                                    ⊢ Iff (LE.le (Real.arccos x) (HDiv.hDiv Real.pi 2)) (LE.le 0 x)
                                                                  -/
theorem arccos_le_pi_div_two {x} : arccos x ≤ π / 2 ↔ 0 ≤ x := by simp [arccos]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
                                                                      /-
                                                                        x : Real
                                                                        ⊢ Iff (LT.lt (Real.arccos x) (HDiv.hDiv Real.pi 2)) (LT.lt 0 x)
                                                                      -/
theorem arccos_lt_pi_div_two {x : ℝ} : arccos x < π / 2 ↔ 0 < x := by simp [arccos]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem arccos_le_pi_div_four {x} : arccos x ≤ π / 4 ↔ √2 / 2 ≤ x := by
  /-
    x : Real
    ⊢ Iff (LE.le (Real.arccos x) (HDiv.hDiv Real.pi 4)) (LE.le (HDiv.hDiv (Real.sq …
  -/
  rw [arccos, ← pi_div_four_le_arcsin]
  /-
    x : Real
    ⊢ Iff (LE.le (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arcsin x)) (HDiv.hDiv Real …
  -/
  constructor <;>
      /-
        case mp
        x : Real
        ⊢ LE.le (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arcsin x)) (HDiv.hDiv Real.pi 4 …
      -/
      /-
        case mp
        x : Real
        a✝ : LE.le (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arcsin x)) (HDiv.hDiv Real.p …
        ⊢ LE.le (HDiv.hDiv Real.pi 4) (Real.arcsin x)
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr
        x : Real
        a✝ : LE.le (HDiv.hDiv Real.pi 4) (Real.arcsin x)
        ⊢ LE.le (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arcsin x)) (HDiv.hDiv Real.pi 4)
      -/
      linarith
      /-
        🎉 no goals
      -/


@[continuity]
theorem continuous_arccos : Continuous arccos :=
  continuous_const.sub continuous_arcsin

-- The junk values for `arccos` and `sqrt` make this true even outside `[-1, 1]`.

theorem tan_arccos (x : ℝ) : tan (arccos x) = √(1 - x ^ 2) / x := by
  /-
    x : Real
    ⊢ Eq (Real.tan (Real.arccos x)) (HDiv.hDiv (HSub.hSub 1 (HPow.hPow x 2)).sqrt x)
  -/
  rw [arccos, tan_pi_div_two_sub, tan_arcsin, inv_div]
  /-
    🎉 no goals
  -/

-- The junk values for `arccos` and `sqrt` make this true even for `1 < x`.

theorem arccos_eq_arcsin {x : ℝ} (h : 0 ≤ x) : arccos x = arcsin (√(1 - x ^ 2)) :=
  (arcsin_eq_of_sin_eq (sin_arccos _)
                                                        /-
                                                          x : Real
                                                          h : LE.le 0 x
                                                          ⊢ LE.le 0 2
                                                        -/
      ⟨(Left.neg_nonpos_iff.2 (div_nonneg pi_pos.le (by norm_num))).trans (arccos_nonneg _),
                                                        /-
                                                          🎉 no goals
                                                        -/
        arccos_le_pi_div_two.2 h⟩).symm

-- The junk values for `arcsin` and `sqrt` make this true even for `1 < x`.

theorem arcsin_eq_arccos {x : ℝ} (h : 0 ≤ x) : arcsin x = arccos (√(1 - x ^ 2)) := by
  /-
    x : Real
    h : LE.le 0 x
    ⊢ Eq (Real.arcsin x) (Real.arccos (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
  -/
  rw [eq_comm, ← cos_arcsin]
  exact
    arccos_cos (arcsin_nonneg.2 h)
      ((arcsin_le_pi_div_two _).trans (div_le_self pi_pos.le one_le_two))


