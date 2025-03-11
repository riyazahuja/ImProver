theorem deriv_arcsin_aux {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
    HasStrictDerivAt arcsin (1 / √(1 - x ^ 2)) x ∧ ContDiffAt ℝ ω arcsin x := by
  /-
    x : Real
    h₁ : Ne x (-1)
    h₂ : Ne x 1
    ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
  -/
  cases' h₁.lt_or_lt with h₁ h₁
    /-
      case inl
      x : Real
      h₁✝ : Ne x (-1)
      h₂ : Ne x 1
      h₁ : LT.lt x (-1)
      ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
    -/
  · have : 1 - x ^ 2 < 0 := by nlinarith [h₁]
    /-
      case inl
      x : Real
      h₁✝ : Ne x (-1)
      h₂ : Ne x 1
      h₁ : LT.lt x (-1)
      this : LT.lt (HSub.hSub 1 (HPow.hPow x 2)) 0
      ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
    -/
    rw [sqrt_eq_zero'.2 this.le, div_zero]
    have : arcsin =ᶠ[𝓝 x] fun _ => -(π / 2) :=
      (gt_mem_nhds h₁).mono fun y hy => arcsin_of_le_neg_one hy.le
    exact ⟨(hasStrictDerivAt_const x _).congr_of_eventuallyEq this.symm,
      contDiffAt_const.congr_of_eventuallyEq this⟩
  /-
    case inr
    x : Real
    h₁✝ : Ne x (-1)
    h₂ : Ne x 1
    h₁ : LT.lt (-1) x
    ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
  -/
  cases' h₂.lt_or_lt with h₂ h₂
    /-
      case inr.inl
      x : Real
      h₁✝ : Ne x (-1)
      h₂✝ : Ne x 1
      h₁ : LT.lt (-1) x
      h₂ : LT.lt x 1
      ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
    -/
  · have : 0 < √(1 - x ^ 2) := sqrt_pos.2 (by nlinarith [h₁, h₂])
    /-
      case inr.inl
      x : Real
      h₁✝ : Ne x (-1)
      h₂✝ : Ne x 1
      h₁ : LT.lt (-1) x
      h₂ : LT.lt x 1
      this : LT.lt 0 (HSub.hSub 1 (HPow.hPow x 2)).sqrt
      ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
    -/
    simp only [← cos_arcsin, one_div] at this ⊢
    exact ⟨sinPartialHomeomorph.hasStrictDerivAt_symm ⟨h₁, h₂⟩ this.ne' (hasStrictDerivAt_sin _),
      sinPartialHomeomorph.contDiffAt_symm_deriv this.ne' ⟨h₁, h₂⟩ (hasDerivAt_sin _)
        contDiff_sin.contDiffAt⟩
    /-
      case inr.inr
      x : Real
      h₁✝ : Ne x (-1)
      h₂✝ : Ne x 1
      h₁ : LT.lt (-1) x
      h₂ : LT.lt 1 x
      ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
    -/
  · have : 1 - x ^ 2 < 0 := by nlinarith [h₂]
    /-
      case inr.inr
      x : Real
      h₁✝ : Ne x (-1)
      h₂✝ : Ne x 1
      h₁ : LT.lt (-1) x
      h₂ : LT.lt 1 x
      this : LT.lt (HSub.hSub 1 (HPow.hPow x 2)) 0
      ⊢ And (HasStrictDerivAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)) …
    -/
    rw [sqrt_eq_zero'.2 this.le, div_zero]
    /-
      case inr.inr
      x : Real
      h₁✝ : Ne x (-1)
      h₂✝ : Ne x 1
      h₁ : LT.lt (-1) x
      h₂ : LT.lt 1 x
      this : LT.lt (HSub.hSub 1 (HPow.hPow x 2)) 0
      ⊢ And (HasStrictDerivAt Real.arcsin 0 x) (ContDiffAt Real Top.top Real.arcsin x)
    -/
    have : arcsin =ᶠ[𝓝 x] fun _ => π / 2 := (lt_mem_nhds h₂).mono fun y hy => arcsin_of_one_le hy.le
    exact ⟨(hasStrictDerivAt_const x _).congr_of_eventuallyEq this.symm,
      contDiffAt_const.congr_of_eventuallyEq this⟩


theorem hasStrictDerivAt_arcsin {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
    HasStrictDerivAt arcsin (1 / √(1 - x ^ 2)) x :=
  (deriv_arcsin_aux h₁ h₂).1


theorem hasDerivAt_arcsin {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
    HasDerivAt arcsin (1 / √(1 - x ^ 2)) x :=
  (hasStrictDerivAt_arcsin h₁ h₂).hasDerivAt


theorem contDiffAt_arcsin {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) {n : WithTop ℕ∞} :
    ContDiffAt ℝ n arcsin x :=
  (deriv_arcsin_aux h₁ h₂).2.of_le le_top


theorem hasDerivWithinAt_arcsin_Ici {x : ℝ} (h : x ≠ -1) :
    HasDerivWithinAt arcsin (1 / √(1 - x ^ 2)) (Ici x) x := by
  /-
    x : Real
    h : Ne x (-1)
    ⊢ HasDerivWithinAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt …
  -/
  rcases eq_or_ne x 1 with (rfl | h')
    /-
      case inl
      h : Ne 1 (-1)
      ⊢ HasDerivWithinAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow 1 2)).sqrt …
    -/
  · convert (hasDerivWithinAt_const (1 : ℝ) _ (π / 2)).congr _ _ <;>
      /-
        case h.e'_9
        h : Ne 1 (-1)
        ⊢ Eq (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow 1 2)).sqrt) 0
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp +contextual [arcsin_of_one_le]
      /-
        🎉 no goals
      -/
    /-
      case inr
      x : Real
      h : Ne x (-1)
      h' : Ne x 1
      ⊢ HasDerivWithinAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt …
    -/
  · exact (hasDerivAt_arcsin h h').hasDerivWithinAt
    /-
      🎉 no goals
    -/


theorem hasDerivWithinAt_arcsin_Iic {x : ℝ} (h : x ≠ 1) :
    HasDerivWithinAt arcsin (1 / √(1 - x ^ 2)) (Iic x) x := by
  /-
    x : Real
    h : Ne x 1
    ⊢ HasDerivWithinAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt …
  -/
  rcases em (x = -1) with (rfl | h')
    /-
      case inl
      h : Ne (-1) 1
      ⊢ HasDerivWithinAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow (-1) 2)).s …
    -/
  · convert (hasDerivWithinAt_const (-1 : ℝ) _ (-(π / 2))).congr _ _ <;>
      /-
        case h.e'_9
        h : Ne (-1) 1
        ⊢ Eq (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow (-1) 2)).sqrt) 0
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp +contextual [arcsin_of_le_neg_one]
      /-
        🎉 no goals
      -/
    /-
      case inr
      x : Real
      h : Ne x 1
      h' : Not (Eq x (-1))
      ⊢ HasDerivWithinAt Real.arcsin (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt …
    -/
  · exact (hasDerivAt_arcsin h' h).hasDerivWithinAt
    /-
      🎉 no goals
    -/


theorem differentiableWithinAt_arcsin_Ici {x : ℝ} :
    DifferentiableWithinAt ℝ arcsin (Ici x) x ↔ x ≠ -1 := by
  /-
    x : Real
    ⊢ Iff (DifferentiableWithinAt Real Real.arcsin (Set.Ici x) x) (Ne x (-1))
  -/
  refine ⟨?_, fun h => (hasDerivWithinAt_arcsin_Ici h).differentiableWithinAt⟩
  /-
    x : Real
    ⊢ DifferentiableWithinAt Real Real.arcsin (Set.Ici x) x → Ne x (-1)
  -/
  rintro h rfl
  have : sin ∘ arcsin =ᶠ[𝓝[≥] (-1 : ℝ)] id := by
    filter_upwards [Icc_mem_nhdsGE (neg_lt_self zero_lt_one)] with x using sin_arcsin'
  /-
    h : DifferentiableWithinAt Real Real.arcsin (Set.Ici (-1)) (-1)
    this : (nhdsWithin (-1) (Set.Ici (-1))).EventuallyEq (Function.comp Real.sin R …
    ⊢ False
  -/
  have := h.hasDerivWithinAt.sin.congr_of_eventuallyEq this.symm (by simp)
  /-
    h : DifferentiableWithinAt Real Real.arcsin (Set.Ici (-1)) (-1)
    this✝ : (nhdsWithin (-1) (Set.Ici (-1))).EventuallyEq (Function.comp Real.sin  …
    this : HasDerivWithinAt id (HMul.hMul (Real.cos (Real.arcsin (-1))) (derivWith …
    ⊢ False
  -/
  simpa using (uniqueDiffOn_Ici _ _ left_mem_Ici).eq_deriv _ this (hasDerivWithinAt_id _ _)
  /-
    🎉 no goals
  -/


theorem differentiableWithinAt_arcsin_Iic {x : ℝ} :
    DifferentiableWithinAt ℝ arcsin (Iic x) x ↔ x ≠ 1 := by
  /-
    x : Real
    ⊢ Iff (DifferentiableWithinAt Real Real.arcsin (Set.Iic x) x) (Ne x 1)
  -/
  refine ⟨fun h => ?_, fun h => (hasDerivWithinAt_arcsin_Iic h).differentiableWithinAt⟩
  /-
    x : Real
    h : DifferentiableWithinAt Real Real.arcsin (Set.Iic x) x
    ⊢ Ne x 1
  -/
  rw [← neg_neg x, ← image_neg_Ici] at h
  /-
    x : Real
    h : DifferentiableWithinAt Real Real.arcsin (Set.image Neg.neg (Set.Ici (Neg.n …
    ⊢ Ne x 1
  -/
  have := (h.comp (-x) differentiableWithinAt_id.neg (mapsTo_image _ _)).neg
  /-
    x : Real
    h : DifferentiableWithinAt Real Real.arcsin (Set.image Neg.neg (Set.Ici (Neg.n …
    this : DifferentiableWithinAt Real (fun y => Neg.neg (Function.comp Real.arcsi …
    ⊢ Ne x 1
  -/
  simpa [(· ∘ ·), differentiableWithinAt_arcsin_Ici] using this
  /-
    🎉 no goals
  -/


theorem differentiableAt_arcsin {x : ℝ} : DifferentiableAt ℝ arcsin x ↔ x ≠ -1 ∧ x ≠ 1 :=
  ⟨fun h => ⟨differentiableWithinAt_arcsin_Ici.1 h.differentiableWithinAt,
      differentiableWithinAt_arcsin_Iic.1 h.differentiableWithinAt⟩,
    fun h => (hasDerivAt_arcsin h.1 h.2).differentiableAt⟩


@[simp]
theorem deriv_arcsin : deriv arcsin = fun x => 1 / √(1 - x ^ 2) := by
  /-
    ⊢ Eq (deriv Real.arcsin) fun x => HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt
  -/
  funext x
  /-
    case h
    x : Real
    ⊢ Eq (deriv Real.arcsin x) (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
  -/
  by_cases h : x ≠ -1 ∧ x ≠ 1
    /-
      case pos
      x : Real
      h : And (Ne x (-1)) (Ne x 1)
      ⊢ Eq (deriv Real.arcsin x) (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
    -/
  · exact (hasDerivAt_arcsin h.1 h.2).deriv
    /-
      🎉 no goals
    -/
    /-
      case neg
      x : Real
      h : Not (And (Ne x (-1)) (Ne x 1))
      ⊢ Eq (deriv Real.arcsin x) (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
    -/
  · rw [deriv_zero_of_not_differentiableAt (mt differentiableAt_arcsin.1 h)]
    /-
      case neg
      x : Real
      h : Not (And (Ne x (-1)) (Ne x 1))
      ⊢ Eq 0 (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
    -/
    simp only [not_and_or, Ne, Classical.not_not] at h
    /-
      case neg
      x : Real
      h : Or (Eq x (-1)) (Eq x 1)
      ⊢ Eq 0 (HDiv.hDiv 1 (HSub.hSub 1 (HPow.hPow x 2)).sqrt)
    -/
                                  /-
                                    🎉 no goals
                                  -/
    rcases h with (rfl | rfl) <;> simp
                                  /-
                                    🎉 no goals
                                  -/


theorem differentiableOn_arcsin : DifferentiableOn ℝ arcsin {-1, 1}ᶜ := fun _x hx =>
  (differentiableAt_arcsin.2
      ⟨fun h => hx (Or.inl h), fun h => hx (Or.inr h)⟩).differentiableWithinAt


theorem contDiffOn_arcsin {n : WithTop ℕ∞} : ContDiffOn ℝ n arcsin {-1, 1}ᶜ := fun _x hx =>
  (contDiffAt_arcsin (mt Or.inl hx) (mt Or.inr hx)).contDiffWithinAt


theorem contDiffAt_arcsin_iff {x : ℝ} {n : WithTop ℕ∞} :
    ContDiffAt ℝ n arcsin x ↔ n = 0 ∨ x ≠ -1 ∧ x ≠ 1 :=
  ⟨fun h => or_iff_not_imp_left.2 fun hn => differentiableAt_arcsin.1 <| h.differentiableAt <|
      ENat.one_le_iff_ne_zero_withTop.mpr hn,
    fun h => h.elim (fun hn => hn.symm ▸ (contDiff_zero.2 continuous_arcsin).contDiffAt) fun hx =>
      contDiffAt_arcsin hx.1 hx.2⟩


theorem hasStrictDerivAt_arccos {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
    HasStrictDerivAt arccos (-(1 / √(1 - x ^ 2))) x :=
  (hasStrictDerivAt_arcsin h₁ h₂).const_sub (π / 2)


theorem hasDerivAt_arccos {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
    HasDerivAt arccos (-(1 / √(1 - x ^ 2))) x :=
  (hasDerivAt_arcsin h₁ h₂).const_sub (π / 2)


theorem contDiffAt_arccos {x : ℝ} (h₁ : x ≠ -1) (h₂ : x ≠ 1) {n : WithTop ℕ∞} :
    ContDiffAt ℝ n arccos x :=
  contDiffAt_const.sub (contDiffAt_arcsin h₁ h₂)


theorem hasDerivWithinAt_arccos_Ici {x : ℝ} (h : x ≠ -1) :
    HasDerivWithinAt arccos (-(1 / √(1 - x ^ 2))) (Ici x) x :=
  (hasDerivWithinAt_arcsin_Ici h).const_sub _


theorem hasDerivWithinAt_arccos_Iic {x : ℝ} (h : x ≠ 1) :
    HasDerivWithinAt arccos (-(1 / √(1 - x ^ 2))) (Iic x) x :=
  (hasDerivWithinAt_arcsin_Iic h).const_sub _


theorem differentiableWithinAt_arccos_Ici {x : ℝ} :
    DifferentiableWithinAt ℝ arccos (Ici x) x ↔ x ≠ -1 :=
  (differentiableWithinAt_const_sub_iff _).trans differentiableWithinAt_arcsin_Ici


theorem differentiableWithinAt_arccos_Iic {x : ℝ} :
    DifferentiableWithinAt ℝ arccos (Iic x) x ↔ x ≠ 1 :=
  (differentiableWithinAt_const_sub_iff _).trans differentiableWithinAt_arcsin_Iic


theorem differentiableAt_arccos {x : ℝ} : DifferentiableAt ℝ arccos x ↔ x ≠ -1 ∧ x ≠ 1 :=
  (differentiableAt_const _).sub_iff_right.trans differentiableAt_arcsin


@[simp]
theorem deriv_arccos : deriv arccos = fun x => -(1 / √(1 - x ^ 2)) :=
                                                  /-
                                                    x : Real
                                                    ⊢ Eq (Neg.neg (deriv Real.arcsin x)) (Neg.neg (HDiv.hDiv 1 (HSub.hSub 1 (HPow. …
                                                  -/
  funext fun x => (deriv_const_sub _).trans <| by simp only [deriv_arcsin]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem differentiableOn_arccos : DifferentiableOn ℝ arccos {-1, 1}ᶜ :=
  differentiableOn_arcsin.const_sub _


theorem contDiffOn_arccos {n : WithTop ℕ∞} : ContDiffOn ℝ n arccos {-1, 1}ᶜ :=
  contDiffOn_const.sub contDiffOn_arcsin


theorem contDiffAt_arccos_iff {x : ℝ} {n : WithTop ℕ∞} :
    ContDiffAt ℝ n arccos x ↔ n = 0 ∨ x ≠ -1 ∧ x ≠ 1 := by
  /-
    x : Real
    n : WithTop ENat
    ⊢ Iff (ContDiffAt Real n Real.arccos x) (Or (Eq n 0) (And (Ne x (-1)) (Ne x 1)))
  -/
  refine Iff.trans ⟨fun h => ?_, fun h => ?_⟩ contDiffAt_arcsin_iff <;>
    /-
      case refine_1
      x : Real
      n : WithTop ENat
      h : ContDiffAt Real n Real.arccos x
      ⊢ ContDiffAt Real n Real.arcsin x
    -/
    /-
      🎉 no goals
    -/
    simpa [arccos] using (contDiffAt_const (c := π / 2)).sub h
    /-
      🎉 no goals
    -/


