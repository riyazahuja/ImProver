/-- The polar coordinates partial homeomorphism in `ℝ^2`, mapping `(r cos θ, r sin θ)` to `(r, θ)`.
It is a homeomorphism between `ℝ^2 - (-∞, 0]` and `(0, +∞) × (-π, π)`. -/
@[simps]
def polarCoord : PartialHomeomorph (ℝ × ℝ) (ℝ × ℝ) where
  toFun q := (√(q.1 ^ 2 + q.2 ^ 2), Complex.arg (Complex.equivRealProd.symm q))
  invFun p := (p.1 * cos p.2, p.1 * sin p.2)
  source := {q | 0 < q.1} ∪ {q | q.2 ≠ 0}
  target := Ioi (0 : ℝ) ×ˢ Ioo (-π) π
  map_target' := by
    /-
      ⊢ ∀ ⦃x : Prod Real Real⦄, Membership.mem (SProd.sprod (Set.Ioi 0) (Set.Ioo (Ne …
    -/
    rintro ⟨r, θ⟩ ⟨hr, hθ⟩
    /-
      case mk.intro
      r θ : Real
      hr : Membership.mem (Set.Ioi 0) { fst := r, snd := θ }.1
      hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) { fst := r, snd := θ }.2
      ⊢ Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => Ne  …
    -/
    dsimp at hr hθ
    /-
      case mk.intro
      r θ : Real
      hr : Membership.mem (Set.Ioi 0) r
      hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
      ⊢ Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => Ne  …
    -/
    rcases eq_or_ne θ 0 with (rfl | h'θ)
      /-
        case mk.intro.inl
        r : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) 0
        ⊢ Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => Ne  …
      -/
    · simpa using hr
      /-
        🎉 no goals
      -/
    /-
      ⊢ ∀ ⦃x : Prod Real Real⦄, Membership.mem (Union.union (setOf fun q => LT.lt 0  …
    -/
      /-
        case mk.intro.inr
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        h'θ : Ne θ 0
        ⊢ Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => Ne  …
      -/
    · right
      /-
        case mk.intro.inr.h
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        h'θ : Ne θ 0
        ⊢ Membership.mem (setOf fun q => Ne q.2 0) ((fun p => { fst := HMul.hMul p.1 ( …
      -/
    /-
      case mk
      x y : Real
      hxy : Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => …
      ⊢ And (LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))) (Or (LE.le 0 (Comp …
    -/
      simp at hr
      /-
        case mk.left
        x y : Real
        hxy : Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => …
        ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
      -/
      simpa only [ne_of_gt hr, Ne, mem_setOf_eq, mul_eq_zero, false_or,
        /-
          case mk.left.inl
          x y : Real
          hxy : Membership.mem (setOf fun q => LT.lt 0 q.1) { fst := x, snd := y }
          ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
        -/
        sin_eq_zero_iff_of_lt_of_lt hθ.1 hθ.2] using h'θ
                      /-
                        🎉 no goals
                      -/
        /-
          case mk.left.inr
          x y : Real
          hxy : Membership.mem (setOf fun q => Ne q.2 0) { fst := x, snd := y }
          ⊢ LT.lt 0 (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2))
        -/
  map_source' := by
        /-
          🎉 no goals
        -/
      /-
        case mk.right
        x y : Real
        hxy : Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q => …
        ⊢ Or (LE.le 0 (Complex.equivRealProd.symm { fst := x, snd := y }).re) (Ne (Com …
      -/
    rintro ⟨x, y⟩ hxy
        /-
          case mk.right.inl
          x y : Real
          hxy : Membership.mem (setOf fun q => LT.lt 0 q.1) { fst := x, snd := y }
          ⊢ Or (LE.le 0 (Complex.equivRealProd.symm { fst := x, snd := y }).re) (Ne (Com …
        -/
    simp only [prod_mk_mem_set_prod_eq, mem_Ioi, sqrt_pos, mem_Ioo, Complex.neg_pi_lt_arg,
        /-
          🎉 no goals
        -/
        /-
          case mk.right.inr
          x y : Real
          hxy : Membership.mem (setOf fun q => Ne q.2 0) { fst := x, snd := y }
          ⊢ Or (LE.le 0 (Complex.equivRealProd.symm { fst := x, snd := y }).re) (Ne (Com …
        -/
      true_and, Complex.arg_lt_pi_iff]
        /-
          🎉 no goals
        -/
    constructor
    · cases' hxy with hxy hxy
      · dsimp at hxy; linarith [sq_pos_of_ne_zero hxy.ne', sq_nonneg y]
      · linarith [sq_nonneg x, sq_pos_of_ne_zero hxy]
    · cases' hxy with hxy hxy
      · exact Or.inl (le_of_lt hxy)
      · exact Or.inr hxy
  right_inv' := by
    /-
      ⊢ ∀ ⦃x : Prod Real Real⦄, Membership.mem (SProd.sprod (Set.Ioi 0) (Set.Ioo (Ne …
    -/
    rintro ⟨r, θ⟩ ⟨hr, hθ⟩
    /-
      case mk.intro
      r θ : Real
      hr : Membership.mem (Set.Ioi 0) { fst := r, snd := θ }.1
      hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) { fst := r, snd := θ }.2
      ⊢ Eq ((fun q => { fst := (HAdd.hAdd (HPow.hPow q.1 2) (HPow.hPow q.2 2)).sqrt, …
    -/
    dsimp at hr hθ
    /-
      case mk.intro
      r θ : Real
      hr : Membership.mem (Set.Ioi 0) r
      hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
      ⊢ Eq ((fun q => { fst := (HAdd.hAdd (HPow.hPow q.1 2) (HPow.hPow q.2 2)).sqrt, …
    -/
    simp only [Prod.mk.inj_iff]
    /-
      case mk.intro
      r θ : Real
      hr : Membership.mem (Set.Ioi 0) r
      hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
      ⊢ And (Eq (HAdd.hAdd (HPow.hPow (HMul.hMul r (Real.cos θ)) 2) (HPow.hPow (HMul …
    -/
    constructor
      /-
        case mk.intro.left
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul r (Real.cos θ)) 2) (HPow.hPow (HMul.hMul …
      -/
    · conv_rhs => rw [← sqrt_sq (le_of_lt hr), ← one_mul (r ^ 2), ← sin_sq_add_cos_sq θ]
      /-
        case mk.intro.left
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul r (Real.cos θ)) 2) (HPow.hPow (HMul.hMul …
      -/
      congr 1
    /-
      ⊢ ∀ ⦃x : Prod Real Real⦄, Membership.mem (Union.union (setOf fun q => LT.lt 0  …
    -/
      /-
        case mk.intro.left.e_x
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        ⊢ Eq (HAdd.hAdd (HPow.hPow (HMul.hMul r (Real.cos θ)) 2) (HPow.hPow (HMul.hMul …
      -/
      ring
      /-
        🎉 no goals
      -/
    /-
      case mk
      x y : Real
      a✝ : Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q =>  …
      A : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)).sqrt (Complex.abs (HAdd.hAd …
      ⊢ Eq ((fun p => { fst := HMul.hMul p.1 (Real.cos p.2), snd := HMul.hMul p.1 (R …
    -/
      /-
        case mk.intro.right
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        ⊢ Eq (Complex.equivRealProd.symm { fst := HMul.hMul r (Real.cos θ), snd := HMu …
      -/
    · convert Complex.arg_mul_cos_add_sin_mul_I hr ⟨hθ.1, hθ.2.le⟩
      simp only [Complex.equivRealProd_symm_apply, Complex.ofReal_mul, Complex.ofReal_cos,
    /-
      case mk
      x y : Real
      a✝ : Membership.mem (Union.union (setOf fun q => LT.lt 0 q.1) (setOf fun q =>  …
      A : Eq (HAdd.hAdd (HPow.hPow x 2) (HPow.hPow y 2)).sqrt (Complex.abs (HAdd.hAd …
      Z : Eq (HAdd.hAdd (↑(HMul.hMul (Complex.abs (HAdd.hAdd (↑x) (HMul.hMul (↑y) Co …
      ⊢ Eq ((fun p => { fst := HMul.hMul p.1 (Real.cos p.2), snd := HMul.hMul p.1 (R …
    -/
        Complex.ofReal_sin]
    /-
      🎉 no goals
    -/
      /-
        case h.e'_2.h.e'_1
        r θ : Real
        hr : Membership.mem (Set.Ioi 0) r
        hθ : Membership.mem (Set.Ioo (Neg.neg Real.pi) Real.pi) θ
        ⊢ Eq (HAdd.hAdd (HMul.hMul (↑r) (Complex.cos ↑θ)) (HMul.hMul (HMul.hMul (↑r) ( …
      -/
      ring
      /-
        🎉 no goals
      -/
  left_inv' := by
    rintro ⟨x, y⟩ _
    have A : √(x ^ 2 + y ^ 2) = Complex.abs (x + y * Complex.I) := by
      rw [Complex.abs_apply, Complex.normSq_add_mul_I]
    have Z := Complex.abs_mul_cos_add_sin_mul_I (x + y * Complex.I)
    simp only [← Complex.ofReal_cos, ← Complex.ofReal_sin, mul_add, ← Complex.ofReal_mul, ←
      mul_assoc] at Z
    simp [A]
  open_target := isOpen_Ioi.prod isOpen_Ioo
  open_source :=
    (isOpen_lt continuous_const continuous_fst).union
      (isOpen_ne_fun continuous_snd continuous_const)
  continuousOn_invFun :=
    ((continuous_fst.mul (continuous_cos.comp continuous_snd)).prod_mk
        (continuous_fst.mul (continuous_sin.comp continuous_snd))).continuousOn
  continuousOn_toFun := by
    /-
      ⊢ ContinuousOn ↑{ toFun := fun q => { fst := (HAdd.hAdd (HPow.hPow q.1 2) (HPo …
    -/
    apply ((continuous_fst.pow 2).add (continuous_snd.pow 2)).sqrt.continuousOn.prod
    have A : MapsTo Complex.equivRealProd.symm ({q : ℝ × ℝ | 0 < q.1} ∪ {q : ℝ × ℝ | q.2 ≠ 0})
        Complex.slitPlane := by
      rintro ⟨x, y⟩ hxy; simpa only using hxy
    refine ContinuousOn.comp (f := Complex.equivRealProd.symm)
      (g := Complex.arg) (fun z hz => ?_) ?_ A
      /-
        case refine_1
        A : Set.MapsTo (⇑Complex.equivRealProd.symm) (Union.union (setOf fun q => LT.l …
        z : Complex
        hz : Membership.mem Complex.slitPlane z
        ⊢ ContinuousWithinAt Complex.arg Complex.slitPlane z
      -/
    · exact (Complex.continuousAt_arg hz).continuousWithinAt
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        A : Set.MapsTo (⇑Complex.equivRealProd.symm) (Union.union (setOf fun q => LT.l …
        ⊢ ContinuousOn ⇑Complex.equivRealProd.symm { toFun := fun q => { fst := (HAdd. …
      -/
    · exact Complex.equivRealProdCLM.symm.continuous.continuousOn
      /-
        🎉 no goals
      -/


theorem hasFDerivAt_polarCoord_symm (p : ℝ × ℝ) :
    HasFDerivAt polarCoord.symm
      (LinearMap.toContinuousLinearMap (Matrix.toLin (Basis.finTwoProd ℝ) (Basis.finTwoProd ℝ)
        !![cos p.2, -p.1 * sin p.2; sin p.2, p.1 * cos p.2])) p := by
  /-
    p : Prod Real Real
    ⊢ HasFDerivAt (↑polarCoord.symm) (LinearMap.toContinuousLinearMap ((Matrix.toL …
  -/
  rw [Matrix.toLin_finTwoProd_toContinuousLinearMap]
  convert HasFDerivAt.prod (𝕜 := ℝ)
    (hasFDerivAt_fst.mul ((hasDerivAt_cos p.2).comp_hasFDerivAt p hasFDerivAt_snd))
    (hasFDerivAt_fst.mul ((hasDerivAt_sin p.2).comp_hasFDerivAt p hasFDerivAt_snd)) using 2 <;>
  /-
    case h.e'_12.h.h.h.e'_15.h.h.h
    p : Prod Real Real
    e_4✝¹ : Eq Prod.instAddCommGroup SeminormedAddCommGroup.toAddCommGroup
    he✝² : Eq Prod.instModule NormedSpace.toModule
    e_6✝ : Eq instTopologicalSpaceProd UniformSpace.toTopologicalSpace
    e_4✝ : Eq instTopologicalSpaceProd UniformSpace.toTopologicalSpace
    e_5✝ : Eq Prod.instAddCommMonoid AddCommGroup.toAddCommMonoid
    he✝¹ : Eq Prod.instModule NormedSpace.toModule
    e_9✝ : Eq NonUnitalNonAssocSemiring.toAddCommMonoid AddCommGroup.toAddCommMonoid
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Real.cos p.2) (ContinuousLinearMap.fst Real Real …
  -/
  /-
    🎉 no goals
  -/
  simp [smul_smul, add_comm, neg_mul, smul_neg, neg_smul _ (ContinuousLinearMap.snd ℝ ℝ ℝ)]
  /-
    🎉 no goals
  -/


theorem det_fderiv_polarCoord_symm (p : ℝ × ℝ) :
    (LinearMap.toContinuousLinearMap (Matrix.toLin (Basis.finTwoProd ℝ) (Basis.finTwoProd ℝ)
      !![cos p.2, -p.1 * sin p.2; sin p.2, p.1 * cos p.2])).det = p.1 := by
  /-
    p : Prod Real Real
    ⊢ Eq (LinearMap.toContinuousLinearMap ((Matrix.toLin (Basis.finTwoProd Real) ( …
  -/
  conv_rhs => rw [← one_mul p.1, ← cos_sq_add_sin_sq p.2]
  simp only [neg_mul, LinearMap.det_toContinuousLinearMap, LinearMap.det_toLin,
    Matrix.det_fin_two_of, sub_neg_eq_add]
  /-
    p : Prod Real Real
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.cos p.2) (HMul.hMul p.1 (Real.cos p.2))) (HMu …
  -/
  ring
  /-
    🎉 no goals
  -/

-- Porting note: this instance is needed but not automatically synthesised

instance : Measure.IsAddHaarMeasure volume (G := ℝ × ℝ) :=
  Measure.prod.instIsAddHaarMeasure _ _


theorem polarCoord_source_ae_eq_univ : polarCoord.source =ᵐ[volume] univ := by
  have A : polarCoord.sourceᶜ ⊆ LinearMap.ker (LinearMap.snd ℝ ℝ ℝ) := by
    intro x hx
    simp only [polarCoord_source, compl_union, mem_inter_iff, mem_compl_iff, mem_setOf_eq, not_lt,
      Classical.not_not] at hx
    exact hx.2
  have B : volume (LinearMap.ker (LinearMap.snd ℝ ℝ ℝ) : Set (ℝ × ℝ)) = 0 := by
    apply Measure.addHaar_submodule
    rw [Ne, LinearMap.ker_eq_top]
    intro h
    have : (LinearMap.snd ℝ ℝ ℝ) (0, 1) = (0 : ℝ × ℝ →ₗ[ℝ] ℝ) (0, 1) := by rw [h]
    simp at this
  /-
    A : HasSubset.Subset (HasCompl.compl polarCoord.source) ↑(LinearMap.ker (Linea …
    B : Eq (MeasureTheory.MeasureSpace.volume ↑(LinearMap.ker (LinearMap.snd Real  …
    ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq polarCoord …
  -/
  simp only [ae_eq_univ]
  /-
    A : HasSubset.Subset (HasCompl.compl polarCoord.source) ↑(LinearMap.ker (Linea …
    B : Eq (MeasureTheory.MeasureSpace.volume ↑(LinearMap.ker (LinearMap.snd Real  …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (HasCompl.compl polarCoord.source)) 0
  -/
  exact le_antisymm ((measure_mono A).trans (le_of_eq B)) bot_le
  /-
    🎉 no goals
  -/


theorem integral_comp_polarCoord_symm {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (f : ℝ × ℝ → E) :
    (∫ p in polarCoord.target, p.1 • f (polarCoord.symm p)) = ∫ p, f p := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Prod Real Real → E
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict polar …
  -/
  symm
  calc
    ∫ p, f p = ∫ p in polarCoord.source, f p := by
      rw [← setIntegral_univ]
      apply setIntegral_congr_set
      exact polarCoord_source_ae_eq_univ.symm
    _ = ∫ p in polarCoord.target, |p.1| • f (polarCoord.symm p) := by
      rw [← PartialHomeomorph.symm_target, integral_target_eq_integral_abs_det_fderiv_smul volume
      (fun p _ ↦ hasFDerivAt_polarCoord_symm p), PartialHomeomorph.symm_source]
      simp_rw [det_fderiv_polarCoord_symm]
    _ = ∫ p in polarCoord.target, p.1 • f (polarCoord.symm p) := by
      apply setIntegral_congr_fun polarCoord.open_target.measurableSet fun x hx => ?_
      rw [abs_of_pos hx.1]


theorem lintegral_comp_polarCoord_symm (f : ℝ × ℝ → ℝ≥0∞) :
    ∫⁻ (p : ℝ × ℝ) in polarCoord.target, ENNReal.ofReal p.1 • f (polarCoord.symm p) =
      ∫⁻ (p : ℝ × ℝ), f p := by
  /-
    f : Prod Real Real → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict pola …
  -/
  symm
  calc
    _ = ∫⁻ p in polarCoord.symm '' polarCoord.target, f p := by
      rw [← setLIntegral_univ, setLIntegral_congr polarCoord_source_ae_eq_univ.symm,
        polarCoord.symm_image_target_eq_source ]
    _ = ∫⁻ (p : ℝ × ℝ) in polarCoord.target, ENNReal.ofReal |p.1| • f (polarCoord.symm p) := by
      rw [lintegral_image_eq_lintegral_abs_det_fderiv_mul volume _
        (fun p _ ↦ (hasFDerivAt_polarCoord_symm p).hasFDerivWithinAt)]
      · simp_rw [det_fderiv_polarCoord_symm]; rfl
      exacts [polarCoord.symm.injOn, measurableSet_Ioi.prod measurableSet_Ioo]
    _ = ∫⁻ (p : ℝ × ℝ) in polarCoord.target, ENNReal.ofReal p.1 • f (polarCoord.symm p) := by
      refine setLIntegral_congr_fun polarCoord.open_target.measurableSet ?_
      filter_upwards with _ hx using by rw [abs_of_pos hx.1]


/-- The polar coordinates partial homeomorphism in `ℂ`, mapping `r (cos θ + I * sin θ)` to `(r, θ)`.
It is a homeomorphism between `ℂ - ℝ≤0` and `(0, +∞) × (-π, π)`. -/
protected noncomputable def polarCoord : PartialHomeomorph ℂ (ℝ × ℝ) :=
  equivRealProdCLM.toHomeomorph.transPartialHomeomorph polarCoord


protected theorem polarCoord_apply (a : ℂ) :
    Complex.polarCoord a = (Complex.abs a, Complex.arg a) := by
  /-
    a : Complex
    ⊢ Eq (↑Complex.polarCoord a) { fst := Complex.abs a, snd := a.arg }
  -/
  simp_rw [Complex.abs_def, Complex.normSq_apply, ← pow_two]
  /-
    a : Complex
    ⊢ Eq (↑Complex.polarCoord a) { fst := (HAdd.hAdd (HPow.hPow a.re 2) (HPow.hPow …
  -/
  rfl
  /-
    🎉 no goals
  -/


protected theorem polarCoord_source : Complex.polarCoord.source = slitPlane := rfl


protected theorem polarCoord_target :
    Complex.polarCoord.target = Set.Ioi (0 : ℝ) ×ˢ Set.Ioo (-π) π := rfl


@[simp]
protected theorem polarCoord_symm_apply (p : ℝ × ℝ) :
    Complex.polarCoord.symm p = p.1 * (Real.cos p.2 + Real.sin p.2 * Complex.I) := by
  /-
    p : Prod Real Real
    ⊢ Eq (↑Complex.polarCoord.symm p) (HMul.hMul (↑p.1) (HAdd.hAdd (↑(Real.cos p.2 …
  -/
  simp [Complex.polarCoord, equivRealProdCLM_symm_apply, mul_add, mul_assoc]
  /-
    🎉 no goals
  -/


theorem measurableEquivRealProd_symm_polarCoord_symm_apply (p : ℝ × ℝ) :
    (measurableEquivRealProd.symm (polarCoord.symm p)) = Complex.polarCoord.symm p := rfl


theorem polarCoord_symm_abs (p : ℝ × ℝ) :
                                                          /-
                                                            p : Prod Real Real
                                                            ⊢ Eq (Complex.abs (↑Complex.polarCoord.symm p)) (_root_.abs p.1)
                                                          -/
    Complex.abs (Complex.polarCoord.symm p) = |p.1| := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[deprecated (since := "2024-07-15")] alias polardCoord_symm_abs := polarCoord_symm_abs


protected theorem integral_comp_polarCoord_symm {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (f : ℂ → E) :
    (∫ p in polarCoord.target, p.1 • f (Complex.polarCoord.symm p)) = ∫ p, f p := by
  rw [← (Complex.volume_preserving_equiv_real_prod.symm).integral_comp
    measurableEquivRealProd.symm.measurableEmbedding, ← integral_comp_polarCoord_symm]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Complex → E
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict polar …
  -/
  simp_rw [measurableEquivRealProd_symm_polarCoord_symm_apply]
  /-
    🎉 no goals
  -/


protected theorem lintegral_comp_polarCoord_symm (f : ℂ → ℝ≥0∞) :
    (∫⁻ p in polarCoord.target, ENNReal.ofReal p.1 • f (Complex.polarCoord.symm p)) =
      ∫⁻ p, f p := by
  rw [← (volume_preserving_equiv_real_prod.symm).lintegral_comp_emb
    measurableEquivRealProd.symm.measurableEmbedding, ← lintegral_comp_polarCoord_symm]
  /-
    f : Complex → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict pola …
  -/
  simp_rw [measurableEquivRealProd_symm_polarCoord_symm_apply]
  /-
    🎉 no goals
  -/


