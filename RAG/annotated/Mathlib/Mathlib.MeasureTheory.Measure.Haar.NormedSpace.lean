instance MapLinearEquiv.isAddHaarMeasure (e : G ≃ₗ[𝕜] H) : IsAddHaarMeasure (μ.map e) :=
  e.toContinuousLinearEquiv.isAddHaarMeasure_map _


/-- The integral of `f (R • x)` with respect to an additive Haar measure is a multiple of the
integral of `f`. The formula we give works even when `f` is not integrable or `R = 0`
thanks to the convention that a non-integrable function has integral zero. -/
theorem integral_comp_smul (f : E → F) (R : ℝ) :
    ∫ x, f (R • x) ∂μ = |(R ^ finrank ℝ E)⁻¹| • ∫ x, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul R x)) (HSMul.hSMul (abs …
  -/
  by_cases hF : CompleteSpace F; swap
    /-
      case neg
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : MeasurableSpace E
      inst✝⁴ : BorelSpace E
      inst✝³ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝² : μ.IsAddHaarMeasure
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      R : Real
      hF : Not (CompleteSpace F)
      ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul R x)) (HSMul.hSMul (abs …
    -/
  · simp [integral, hF]
    /-
      🎉 no goals
    -/
  /-
    case pos
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    hF : CompleteSpace F
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul R x)) (HSMul.hSMul (abs …
  -/
  rcases eq_or_ne R 0 with (rfl | hR)
    /-
      case pos.inl
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : MeasurableSpace E
      inst✝⁴ : BorelSpace E
      inst✝³ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝² : μ.IsAddHaarMeasure
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      hF : CompleteSpace F
      ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul 0 x)) (HSMul.hSMul (abs …
    -/
  · simp only [zero_smul, integral_const]
    /-
      case pos.inl
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : MeasurableSpace E
      inst✝⁴ : BorelSpace E
      inst✝³ : FiniteDimensional Real E
      μ : MeasureTheory.Measure E
      inst✝² : μ.IsAddHaarMeasure
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : E → F
      hF : CompleteSpace F
      ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (f 0)) (HSMul.hSMul (abs (Inv.inv (HPow. …
    -/
    rcases Nat.eq_zero_or_pos (finrank ℝ E) with (hE | hE)
      /-
        case pos.inl.inl
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : MeasurableSpace E
        inst✝⁴ : BorelSpace E
        inst✝³ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝² : μ.IsAddHaarMeasure
        F : Type u_2
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : E → F
        hF : CompleteSpace F
        hE : Eq (Module.finrank Real E) 0
        ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (f 0)) (HSMul.hSMul (abs (Inv.inv (HPow. …
      -/
    · have : Subsingleton E := finrank_zero_iff.1 hE
      /-
        case pos.inl.inl
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : MeasurableSpace E
        inst✝⁴ : BorelSpace E
        inst✝³ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝² : μ.IsAddHaarMeasure
        F : Type u_2
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : E → F
        hF : CompleteSpace F
        hE : Eq (Module.finrank Real E) 0
        this : Subsingleton E
        ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (f 0)) (HSMul.hSMul (abs (Inv.inv (HPow. …
      -/
      have : f = fun _ => f 0 := by ext x; rw [Subsingleton.elim x 0]
      /-
        case pos.inl.inl
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : MeasurableSpace E
        inst✝⁴ : BorelSpace E
        inst✝³ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝² : μ.IsAddHaarMeasure
        F : Type u_2
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : E → F
        hF : CompleteSpace F
        hE : Eq (Module.finrank Real E) 0
        this✝ : Subsingleton E
        this : Eq f fun x => f 0
        ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (f 0)) (HSMul.hSMul (abs (Inv.inv (HPow. …
      -/
      conv_rhs => rw [this]
      /-
        case pos.inl.inl
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : MeasurableSpace E
        inst✝⁴ : BorelSpace E
        inst✝³ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝² : μ.IsAddHaarMeasure
        F : Type u_2
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : E → F
        hF : CompleteSpace F
        hE : Eq (Module.finrank Real E) 0
        this✝ : Subsingleton E
        this : Eq f fun x => f 0
        ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (f 0)) (HSMul.hSMul (abs (Inv.inv (HPow. …
      -/
      simp only [hE, pow_zero, inv_one, abs_one, one_smul, integral_const]
      /-
        🎉 no goals
      -/
      /-
        case pos.inl.inr
        E : Type u_1
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : MeasurableSpace E
        inst✝⁴ : BorelSpace E
        inst✝³ : FiniteDimensional Real E
        μ : MeasureTheory.Measure E
        inst✝² : μ.IsAddHaarMeasure
        F : Type u_2
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace Real F
        f : E → F
        hF : CompleteSpace F
        hE : GT.gt (Module.finrank Real E) 0
        ⊢ Eq (HSMul.hSMul (μ Set.univ).toReal (f 0)) (HSMul.hSMul (abs (Inv.inv (HPow. …
      -/
    · have : Nontrivial E := finrank_pos_iff.1 hE
      simp only [zero_pow hE.ne', measure_univ_of_isAddLeftInvariant, ENNReal.top_toReal, zero_smul,
        inv_zero, abs_zero]
  · calc
      (∫ x, f (R • x) ∂μ) = ∫ y, f y ∂Measure.map (fun x => R • x) μ :=
        (integral_map_equiv (Homeomorph.smul (isUnit_iff_ne_zero.2 hR).unit).toMeasurableEquiv
            f).symm
      _ = |(R ^ finrank ℝ E)⁻¹| • ∫ x, f x ∂μ := by
        simp only [map_addHaar_smul μ hR, integral_smul_measure, ENNReal.toReal_ofReal, abs_nonneg]


/-- The integral of `f (R • x)` with respect to an additive Haar measure is a multiple of the
integral of `f`. The formula we give works even when `f` is not integrable or `R = 0`
thanks to the convention that a non-integrable function has integral zero. -/
theorem integral_comp_smul_of_nonneg (f : E → F) (R : ℝ) {hR : 0 ≤ R} :
    ∫ x, f (R • x) ∂μ = (R ^ finrank ℝ E)⁻¹ • ∫ x, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    hR : LE.le 0 R
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul R x)) (HSMul.hSMul (Inv …
  -/
  rw [integral_comp_smul μ f R, abs_of_nonneg (inv_nonneg.2 (pow_nonneg hR _))]
  /-
    🎉 no goals
  -/


/-- The integral of `f (R⁻¹ • x)` with respect to an additive Haar measure is a multiple of the
integral of `f`. The formula we give works even when `f` is not integrable or `R = 0`
thanks to the convention that a non-integrable function has integral zero. -/
theorem integral_comp_inv_smul (f : E → F) (R : ℝ) :
    ∫ x, f (R⁻¹ • x) ∂μ = |R ^ finrank ℝ E| • ∫ x, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul (Inv.inv R) x)) (HSMul. …
  -/
  rw [integral_comp_smul μ f R⁻¹, inv_pow, inv_inv]
  /-
    🎉 no goals
  -/


/-- The integral of `f (R⁻¹ • x)` with respect to an additive Haar measure is a multiple of the
integral of `f`. The formula we give works even when `f` is not integrable or `R = 0`
thanks to the convention that a non-integrable function has integral zero. -/
theorem integral_comp_inv_smul_of_nonneg (f : E → F) {R : ℝ} (hR : 0 ≤ R) :
    ∫ x, f (R⁻¹ • x) ∂μ = R ^ finrank ℝ E • ∫ x, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    hR : LE.le 0 R
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul (Inv.inv R) x)) (HSMul. …
  -/
  rw [integral_comp_inv_smul μ f R, abs_of_nonneg (pow_nonneg hR _)]
  /-
    🎉 no goals
  -/


theorem setIntegral_comp_smul (f : E → F) {R : ℝ} (s : Set E) (hR : R ≠ 0) :
    ∫ x in s, f (R • x) ∂μ = |(R ^ finrank ℝ E)⁻¹| • ∫ x in R • s, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    s : Set E
    hR : Ne R 0
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f (HSMul.hSMul R x)) (HSM …
  -/
  let e : E ≃ᵐ E := (Homeomorph.smul (Units.mk0 R hR)).toMeasurableEquiv
  calc
  ∫ x in s, f (R • x) ∂μ
    = ∫ x in e ⁻¹' (e.symm ⁻¹' s), f (e x) ∂μ := by simp [← preimage_comp]; rfl
  _ = ∫ y in e.symm ⁻¹' s, f y ∂map (fun x ↦ R • x) μ := (setIntegral_map_equiv _ _ _).symm
  _ = |(R ^ finrank ℝ E)⁻¹| • ∫ y in e.symm ⁻¹' s, f y ∂μ := by
    simp [map_addHaar_smul μ hR, integral_smul_measure, ENNReal.toReal_ofReal, abs_nonneg]
  _ = |(R ^ finrank ℝ E)⁻¹| • ∫ x in R • s, f x ∂μ := by
    congr
    ext y
    rw [mem_smul_set_iff_inv_smul_mem₀ hR]
    rfl


@[deprecated (since := "2024-04-17")]
alias set_integral_comp_smul := setIntegral_comp_smul


theorem setIntegral_comp_smul_of_pos (f : E → F) {R : ℝ} (s : Set E) (hR : 0 < R) :
    ∫ x in s, f (R • x) ∂μ = (R ^ finrank ℝ E)⁻¹ • ∫ x in R • s, f x ∂μ := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝² : μ.IsAddHaarMeasure
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : E → F
    R : Real
    s : Set E
    hR : LT.lt 0 R
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f (HSMul.hSMul R x)) (HSM …
  -/
  rw [setIntegral_comp_smul μ f s hR.ne', abs_of_nonneg (inv_nonneg.2 (pow_nonneg hR.le _))]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_comp_smul_of_pos := setIntegral_comp_smul_of_pos


theorem integral_comp_mul_left (g : ℝ → F) (a : ℝ) :
    (∫ x : ℝ, g (a * x)) = |a⁻¹| • ∫ y : ℝ, g y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    g : Real → F
    a : Real
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => g (HMu …
  -/
  simp_rw [← smul_eq_mul, Measure.integral_comp_smul, Module.finrank_self, pow_one]
  /-
    🎉 no goals
  -/


theorem integral_comp_inv_mul_left (g : ℝ → F) (a : ℝ) :
    (∫ x : ℝ, g (a⁻¹ * x)) = |a| • ∫ y : ℝ, g y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    g : Real → F
    a : Real
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => g (HMu …
  -/
  simp_rw [← smul_eq_mul, Measure.integral_comp_inv_smul, Module.finrank_self, pow_one]
  /-
    🎉 no goals
  -/


theorem integral_comp_mul_right (g : ℝ → F) (a : ℝ) :
    (∫ x : ℝ, g (x * a)) = |a⁻¹| • ∫ y : ℝ, g y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    g : Real → F
    a : Real
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => g (HMu …
  -/
  simpa only [mul_comm] using integral_comp_mul_left g a
  /-
    🎉 no goals
  -/


theorem integral_comp_inv_mul_right (g : ℝ → F) (a : ℝ) :
    (∫ x : ℝ, g (x * a⁻¹)) = |a| • ∫ y : ℝ, g y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    g : Real → F
    a : Real
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => g (HMu …
  -/
  simpa only [mul_comm] using integral_comp_inv_mul_left g a
  /-
    🎉 no goals
  -/


theorem integral_comp_div (g : ℝ → F) (a : ℝ) : (∫ x : ℝ, g (x / a)) = |a| • ∫ y : ℝ, g y :=
  integral_comp_inv_mul_right g a


theorem integrable_comp_smul_iff {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] (μ : Measure E) [IsAddHaarMeasure μ]
    (f : E → F) {R : ℝ} (hR : R ≠ 0) : Integrable (fun x => f (R • x)) μ ↔ Integrable f μ := by
  -- reduce to one-way implication
  suffices
    ∀ {g : E → F} (_ : Integrable g μ) {S : ℝ} (_ : S ≠ 0), Integrable (fun x => g (S • x)) μ by
    refine ⟨fun hf => ?_, fun hf => this hf hR⟩
    convert this hf (inv_ne_zero hR)
    rw [← mul_smul, mul_inv_cancel₀ hR, one_smul]
  -- now prove
  /-
    F : Type u_1
    inst✝⁶ : NormedAddCommGroup F
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    R : Real
    hR : Ne R 0
    ⊢ ∀ {g : E → F}, MeasureTheory.Integrable g μ → ∀ {S : Real}, Ne S 0 → Measure …
  -/
  intro g hg S hS
  /-
    F : Type u_1
    inst✝⁶ : NormedAddCommGroup F
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    R : Real
    hR : Ne R 0
    g : E → F
    hg : MeasureTheory.Integrable g μ
    S : Real
    hS : Ne S 0
    ⊢ MeasureTheory.Integrable (fun x => g (HSMul.hSMul S x)) μ
  -/
  let t := ((Homeomorph.smul (isUnit_iff_ne_zero.2 hS).unit).toMeasurableEquiv : E ≃ᵐ E)
  /-
    F : Type u_1
    inst✝⁶ : NormedAddCommGroup F
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    R : Real
    hR : Ne R 0
    g : E → F
    hg : MeasureTheory.Integrable g μ
    S : Real
    hS : Ne S 0
    t : MeasurableEquiv E E := (Homeomorph.smul ⋯.unit).toMeasurableEquiv
    ⊢ MeasureTheory.Integrable (fun x => g (HSMul.hSMul S x)) μ
  -/
  refine (integrable_map_equiv t g).mp (?_ : Integrable g (map (S • ·) μ))
  /-
    F : Type u_1
    inst✝⁶ : NormedAddCommGroup F
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    R : Real
    hR : Ne R 0
    g : E → F
    hg : MeasureTheory.Integrable g μ
    S : Real
    hS : Ne S 0
    t : MeasurableEquiv E E := (Homeomorph.smul ⋯.unit).toMeasurableEquiv
    ⊢ MeasureTheory.Integrable g (MeasureTheory.Measure.map (fun x => HSMul.hSMul  …
  -/
  rwa [map_addHaar_smul μ hS, integrable_smul_measure _ ENNReal.ofReal_ne_top]
  /-
    F : Type u_1
    inst✝⁶ : NormedAddCommGroup F
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    R : Real
    hR : Ne R 0
    g : E → F
    hg : MeasureTheory.Integrable g μ
    S : Real
    hS : Ne S 0
    t : MeasurableEquiv E E := (Homeomorph.smul ⋯.unit).toMeasurableEquiv
    ⊢ Ne (ENNReal.ofReal (abs (Inv.inv (HPow.hPow S (Module.finrank Real E))))) 0
  -/
  simpa only [Ne, ENNReal.ofReal_eq_zero, not_le, abs_pos] using inv_ne_zero (pow_ne_zero _ hS)
  /-
    🎉 no goals
  -/


theorem Integrable.comp_smul {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] {μ : Measure E} [IsAddHaarMeasure μ]
    {f : E → F} (hf : Integrable f μ) {R : ℝ} (hR : R ≠ 0) : Integrable (fun x => f (R • x)) μ :=
  (integrable_comp_smul_iff μ f hR).2 hf


theorem integrable_comp_mul_left_iff (g : ℝ → F) {R : ℝ} (hR : R ≠ 0) :
     /-
       F : Type u_1
       inst✝ : NormedAddCommGroup F
       g : Real → F
       R : Real
       hR : Ne R 0
       ⊢ MeasureTheory.Measure Real
     -/
     /-
       🎉 no goals
     -/
    (Integrable fun x => g (R * x)) ↔ Integrable g := by
                                      /-
                                        🎉 no goals
                                      -/
  /-
    F : Type u_1
    inst✝ : NormedAddCommGroup F
    g : Real → F
    R : Real
    hR : Ne R 0
    ⊢ Iff (MeasureTheory.Integrable (fun x => g (HMul.hMul R x)) MeasureTheory.Mea …
  -/
  simpa only [smul_eq_mul] using integrable_comp_smul_iff volume g hR
  /-
    🎉 no goals
  -/


                                                    /-
                                                      F : Type u_1
                                                      inst✝ : NormedAddCommGroup F
                                                      g : Real → F
                                                      ⊢ MeasureTheory.Measure Real
                                                    -/
theorem Integrable.comp_mul_left' {g : ℝ → F} (hg : Integrable g) {R : ℝ} (hR : R ≠ 0) :
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      F : Type u_1
      inst✝ : NormedAddCommGroup F
      g : Real → F
      hg : MeasureTheory.Integrable g MeasureTheory.MeasureSpace.volume
      R : Real
      hR : Ne R 0
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable fun x => g (R * x) :=
    /-
      🎉 no goals
    -/
  (integrable_comp_mul_left_iff g hR).2 hg


theorem integrable_comp_mul_right_iff (g : ℝ → F) {R : ℝ} (hR : R ≠ 0) :
     /-
       F : Type u_1
       inst✝ : NormedAddCommGroup F
       g : Real → F
       R : Real
       hR : Ne R 0
       ⊢ MeasureTheory.Measure Real
     -/
     /-
       🎉 no goals
     -/
    (Integrable fun x => g (x * R)) ↔ Integrable g := by
                                      /-
                                        🎉 no goals
                                      -/
  /-
    F : Type u_1
    inst✝ : NormedAddCommGroup F
    g : Real → F
    R : Real
    hR : Ne R 0
    ⊢ Iff (MeasureTheory.Integrable (fun x => g (HMul.hMul x R)) MeasureTheory.Mea …
  -/
  simpa only [mul_comm] using integrable_comp_mul_left_iff g hR
  /-
    🎉 no goals
  -/


                                                     /-
                                                       F : Type u_1
                                                       inst✝ : NormedAddCommGroup F
                                                       g : Real → F
                                                       ⊢ MeasureTheory.Measure Real
                                                     -/
theorem Integrable.comp_mul_right' {g : ℝ → F} (hg : Integrable g) {R : ℝ} (hR : R ≠ 0) :
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      F : Type u_1
      inst✝ : NormedAddCommGroup F
      g : Real → F
      hg : MeasureTheory.Integrable g MeasureTheory.MeasureSpace.volume
      R : Real
      hR : Ne R 0
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable fun x => g (x * R) :=
    /-
      🎉 no goals
    -/
  (integrable_comp_mul_right_iff g hR).2 hg


theorem integrable_comp_div_iff (g : ℝ → F) {R : ℝ} (hR : R ≠ 0) :
     /-
       F : Type u_1
       inst✝ : NormedAddCommGroup F
       g : Real → F
       R : Real
       hR : Ne R 0
       ⊢ MeasureTheory.Measure Real
     -/
     /-
       🎉 no goals
     -/
    (Integrable fun x => g (x / R)) ↔ Integrable g :=
                                      /-
                                        🎉 no goals
                                      -/
  integrable_comp_mul_right_iff g (inv_ne_zero hR)


                                              /-
                                                F : Type u_1
                                                inst✝ : NormedAddCommGroup F
                                                g : Real → F
                                                ⊢ MeasureTheory.Measure Real
                                              -/
theorem Integrable.comp_div {g : ℝ → F} (hg : Integrable g) {R : ℝ} (hR : R ≠ 0) :
                                              /-
                                                🎉 no goals
                                              -/
    /-
      F : Type u_1
      inst✝ : NormedAddCommGroup F
      g : Real → F
      hg : MeasureTheory.Integrable g MeasureTheory.MeasureSpace.volume
      R : Real
      hR : Ne R 0
      ⊢ MeasureTheory.Measure Real
    -/
    Integrable fun x => g (x / R) :=
    /-
      🎉 no goals
    -/
  (integrable_comp_div_iff g hR).2 hg


                                       /-
                                         F : Type u_1
                                         inst✝¹¹ : NormedAddCommGroup F
                                         E' : Type u_2
                                         F' : Type u_3
                                         A : Type u_4
                                         inst✝¹⁰ : NormedAddCommGroup E'
                                         inst✝⁹ : InnerProductSpace Real E'
                                         inst✝⁸ : FiniteDimensional Real E'
                                         inst✝⁷ : MeasurableSpace E'
                                         inst✝⁶ : BorelSpace E'
                                         inst✝⁵ : NormedAddCommGroup F'
                                         inst✝⁴ : InnerProductSpace Real F'
                                         inst✝³ : FiniteDimensional Real F'
                                         inst✝² : MeasurableSpace F'
                                         inst✝¹ : BorelSpace F'
                                         f : LinearIsometryEquiv (RingHom.id Real) E' F'
                                         inst✝ : NormedAddCommGroup A
                                         g : F' → A
                                         ⊢ MeasureTheory.Measure E'
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
theorem integrable_comp (g : F' → A) : Integrable (g ∘ f) ↔ Integrable g :=
                                                            /-
                                                              🎉 no goals
                                                            -/
  f.measurePreserving.integrable_comp_emb f.toMeasureEquiv.measurableEmbedding


theorem integral_comp [NormedSpace ℝ A] (g : F' → A) : ∫ (x : E'), g (f x) = ∫ (y : F'), g y :=
  f.measurePreserving.integral_comp' (f := f.toMeasureEquiv) g


