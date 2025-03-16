/-- Exponentially tilted measure. When `x ↦ exp (f x)` is integrable, `μ.tilted f` is the
probability measure with density with respect to `μ` proportional to `exp (f x)`. Otherwise it is 0.
-/
noncomputable
def Measure.tilted (μ : Measure α) (f : α → ℝ) : Measure α :=
  μ.withDensity (fun x ↦ ENNReal.ofReal (exp (f x) / ∫ x, exp (f x) ∂μ))


@[simp]
lemma tilted_of_not_integrable (hf : ¬ Integrable (fun x ↦ exp (f x)) μ) : μ.tilted f = 0 := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
    ⊢ Eq (μ.tilted f) 0
  -/
  rw [Measure.tilted, integral_undef hf]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
    ⊢ Eq (μ.withDensity fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) 0)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma tilted_of_not_aemeasurable (hf : ¬ AEMeasurable f μ) : μ.tilted f = 0 := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Not (AEMeasurable f μ)
    ⊢ Eq (μ.tilted f) 0
  -/
  refine tilted_of_not_integrable ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Not (AEMeasurable f μ)
    ⊢ Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
  -/
  suffices ¬ AEMeasurable (fun x ↦ exp (f x)) μ by exact fun h ↦ this h.1.aemeasurable
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : Not (AEMeasurable f μ)
    ⊢ Not (AEMeasurable (fun x => Real.exp (f x)) μ)
  -/
  exact fun h ↦ hf (aemeasurable_of_aemeasurable_exp h)
  /-
    🎉 no goals
  -/


@[simp]
                                                                           /-
                                                                             α : Type u_1
                                                                             mα : MeasurableSpace α
                                                                             f : α → Real
                                                                             ⊢ Eq (MeasureTheory.Measure.tilted 0 f) 0
                                                                           -/
lemma tilted_zero_measure (f : α → ℝ) : (0 : Measure α).tilted f = 0 := by simp [Measure.tilted]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
lemma tilted_const' (μ : Measure α) (c : ℝ) :
    μ.tilted (fun _ ↦ c) = (μ Set.univ)⁻¹ • μ := by
  cases eq_zero_or_neZero μ with
  | inl h => rw [h]; simp
  | inr h0 =>
    simp only [Measure.tilted, withDensity_const, integral_const, smul_eq_mul]
    by_cases h_univ : μ Set.univ = ∞
    · simp only [h_univ, ENNReal.top_toReal, zero_mul, log_zero, div_zero, ENNReal.ofReal_zero,
        zero_smul, ENNReal.inv_top]
    congr
    rw [div_eq_mul_inv, mul_inv, mul_comm, mul_assoc, inv_mul_cancel₀ (exp_pos _).ne', mul_one,
      ← ENNReal.toReal_inv, ENNReal.ofReal_toReal]
    simp [h0.out]


lemma tilted_const (μ : Measure α) [IsProbabilityMeasure μ] (c : ℝ) :
                                   /-
                                     α : Type u_1
                                     mα : MeasurableSpace α
                                     μ : MeasureTheory.Measure α
                                     inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                     c : Real
                                     ⊢ Eq (μ.tilted fun x => c) μ
                                   -/
    μ.tilted (fun _ ↦ c) = μ := by simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma tilted_zero' (μ : Measure α) : μ.tilted 0 = (μ Set.univ)⁻¹ • μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.tilted 0) (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)
  -/
  change μ.tilted (fun _ ↦ 0) = (μ Set.univ)⁻¹ • μ
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ⊢ Eq (μ.tilted fun x => 0) (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    mα : MeasurableSpace α
                                                                                    μ : MeasureTheory.Measure α
                                                                                    inst✝ : MeasureTheory.IsProbabilityMeasure μ
                                                                                    ⊢ Eq (μ.tilted 0) μ
                                                                                  -/
lemma tilted_zero (μ : Measure α) [IsProbabilityMeasure μ] : μ.tilted 0 = μ := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


lemma tilted_congr {g : α → ℝ} (hfg : f =ᵐ[μ] g) :
    μ.tilted f = μ.tilted g := by
  have h_int_eq : ∫ x, exp (f x) ∂μ = ∫ x, exp (g x) ∂μ := by
    refine integral_congr_ae ?_
    filter_upwards [hfg] with x hx
    rw [hx]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    h_int_eq : Eq (MeasureTheory.integral μ fun x => Real.exp (f x)) (MeasureTheor …
    ⊢ Eq (μ.tilted f) (μ.tilted g)
  -/
  refine withDensity_congr_ae ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    h_int_eq : Eq (MeasureTheory.integral μ fun x => Real.exp (f x)) (MeasureTheor …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ENNReal.ofReal (HDiv.hDiv (Real. …
  -/
  filter_upwards [hfg] with x hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    h_int_eq : Eq (MeasureTheory.integral μ fun x => Real.exp (f x)) (MeasureTheor …
    x : α
    hx : Eq (f x) (g x)
    ⊢ Eq (ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTheory.integral μ fun …
  -/
  rw [h_int_eq, hx]
  /-
    🎉 no goals
  -/


lemma tilted_eq_withDensity_nnreal (μ : Measure α) (f : α → ℝ) :
    μ.tilted f = μ.withDensity (fun x ↦ ((↑) : ℝ≥0 → ℝ≥0∞)
                                          /-
                                            α : Type u_1
                                            mα : MeasurableSpace α
                                            μ✝ : MeasureTheory.Measure α
                                            f✝ : α → Real
                                            μ : MeasureTheory.Measure α
                                            f : α → Real
                                            x : α
                                            ⊢ LE.le 0 (HDiv.hDiv (Real.exp (f x)) (MeasureTheory.integral μ fun x => Real. …
                                          -/
      (⟨exp (f x) / ∫ x, exp (f x) ∂μ, by positivity⟩ : ℝ≥0)) := by
                                          /-
                                            🎉 no goals
                                          -/
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ Eq (μ.tilted f) (μ.withDensity fun x => ↑⟨HDiv.hDiv (Real.exp (f x)) (Measur …
  -/
  rw [Measure.tilted]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ Eq (μ.withDensity fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (Measu …
  -/
  congr with x
  /-
    case e_f.h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    x : α
    ⊢ Eq (ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTheory.integral μ fun …
  -/
  rw [ENNReal.ofReal_eq_coe_nnreal]
  /-
    🎉 no goals
  -/


lemma tilted_apply' (μ : Measure α) (f : α → ℝ) {s : Set α} (hs : MeasurableSet s) :
    μ.tilted f s = ∫⁻ a in s, ENNReal.ofReal (exp (f a) / ∫ x, exp (f x) ∂μ) ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((μ.tilted f) s) (MeasureTheory.lintegral (μ.restrict s) fun a => ENNReal …
  -/
  rw [Measure.tilted, withDensity_apply _ hs]
  /-
    🎉 no goals
  -/


lemma tilted_apply (μ : Measure α) [SFinite μ] (f : α → ℝ) (s : Set α) :
    μ.tilted f s = ∫⁻ a in s, ENNReal.ofReal (exp (f a) / ∫ x, exp (f x) ∂μ) ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Real
    s : Set α
    ⊢ Eq ((μ.tilted f) s) (MeasureTheory.lintegral (μ.restrict s) fun a => ENNReal …
  -/
  rw [Measure.tilted, withDensity_apply' _ s]
  /-
    🎉 no goals
  -/


lemma tilted_apply_eq_ofReal_integral' {s : Set α} (f : α → ℝ) (hs : MeasurableSet s) :
    μ.tilted f s = ENNReal.ofReal (∫ a in s, exp (f a) / ∫ x, exp (f x) ∂μ ∂μ) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → Real
    hs : MeasurableSet s
    ⊢ Eq ((μ.tilted f) s) (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) f …
  -/
  by_cases hf : Integrable (fun x ↦ exp (f x)) μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → Real
      hs : MeasurableSet s
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ Eq ((μ.tilted f) s) (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) f …
    -/
  · rw [tilted_apply' _ _ hs, ← ofReal_integral_eq_lintegral_ofReal]
      /-
        case pos.hfi
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        f : α → Real
        hs : MeasurableSet s
        hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
        ⊢ MeasureTheory.Integrable (fun a => HDiv.hDiv (Real.exp (f a)) (MeasureTheory …
      -/
    · exact hf.integrableOn.div_const _
      /-
        🎉 no goals
      -/
      /-
        case pos.f_nn
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        f : α → Real
        hs : MeasurableSet s
        hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
        ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 fun a => HDiv.hDiv (Real.ex …
      -/
    · exact ae_of_all _ (fun _ ↦ by positivity)
      /-
        🎉 no goals
      -/
  · simp only [hf, not_false_eq_true, tilted_of_not_integrable, Measure.coe_zero,
      Pi.zero_apply, integral_undef hf, div_zero, integral_zero, ENNReal.ofReal_zero]


lemma tilted_apply_eq_ofReal_integral [SFinite μ] (f : α → ℝ) (s : Set α) :
    μ.tilted f s = ENNReal.ofReal (∫ a in s, exp (f a) / ∫ x, exp (f x) ∂μ ∂μ) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Real
    s : Set α
    ⊢ Eq ((μ.tilted f) s) (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) f …
  -/
  by_cases hf : Integrable (fun x ↦ exp (f x)) μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      s : Set α
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ Eq ((μ.tilted f) s) (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) f …
    -/
  · rw [tilted_apply _ _, ← ofReal_integral_eq_lintegral_ofReal]
      /-
        case pos.hfi
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        s : Set α
        hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
        ⊢ MeasureTheory.Integrable (fun a => HDiv.hDiv (Real.exp (f a)) (MeasureTheory …
      -/
    · exact hf.integrableOn.div_const _
      /-
        🎉 no goals
      -/
      /-
        case pos.f_nn
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        s : Set α
        hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
        ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 fun a => HDiv.hDiv (Real.ex …
      -/
    · exact ae_of_all _ (fun _ ↦ by positivity)
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      s : Set α
      hf : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq ((μ.tilted f) s) (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) f …
    -/
  · simp [tilted_of_not_integrable hf, integral_undef hf]
    /-
      🎉 no goals
    -/


instance isFiniteMeasure_tilted : IsFiniteMeasure (μ.tilted f) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ MeasureTheory.IsFiniteMeasure (μ.tilted f)
  -/
  by_cases hf : Integrable (fun x ↦ exp (f x)) μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ MeasureTheory.IsFiniteMeasure (μ.tilted f)
    -/
  · refine isFiniteMeasure_withDensity_ofReal ?_
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => HDiv.hDiv (Real.exp (f x)) (Measur …
    -/
    suffices Integrable (fun x ↦ exp (f x) / ∫ x, exp (f x) ∂μ) μ by exact this.2
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ MeasureTheory.Integrable (fun x => HDiv.hDiv (Real.exp (f x)) (MeasureTheory …
    -/
    exact hf.div_const _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ MeasureTheory.IsFiniteMeasure (μ.tilted f)
    -/
  · simp only [hf, not_false_eq_true, tilted_of_not_integrable]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ MeasureTheory.IsFiniteMeasure 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma isProbabilityMeasure_tilted [NeZero μ] (hf : Integrable (fun x ↦ exp (f x)) μ) :
    IsProbabilityMeasure (μ.tilted f) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : NeZero μ
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    ⊢ MeasureTheory.IsProbabilityMeasure (μ.tilted f)
  -/
  constructor
  simp_rw [tilted_apply' _ _ MeasurableSet.univ, setLIntegral_univ,
    ENNReal.ofReal_div_of_pos (integral_exp_pos hf), div_eq_mul_inv]
  rw [lintegral_mul_const'' _ hf.1.aemeasurable.ennreal_ofReal,
    ← ofReal_integral_eq_lintegral_ofReal hf (ae_of_all _ fun _ ↦ (exp_pos _).le),
    ENNReal.mul_inv_cancel]
    /-
      case measure_univ.h0
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      inst✝ : NeZero μ
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ Ne (ENNReal.ofReal (MeasureTheory.integral μ fun x => Real.exp (f x))) 0
    -/
  · simp only [ne_eq, ENNReal.ofReal_eq_zero, not_le]
    /-
      case measure_univ.h0
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      inst✝ : NeZero μ
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ LT.lt 0 (MeasureTheory.integral μ fun x => Real.exp (f x))
    -/
    exact integral_exp_pos hf
    /-
      🎉 no goals
    -/
    /-
      case measure_univ.ht
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      inst✝ : NeZero μ
      hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
      ⊢ Ne (ENNReal.ofReal (MeasureTheory.integral μ fun x => Real.exp (f x))) Top.top
    -/
  · simp
    /-
      🎉 no goals
    -/


lemma setLIntegral_tilted' (f : α → ℝ) (g : α → ℝ≥0∞) {s : Set α} (hs : MeasurableSet s) :
    ∫⁻ x in s, g x ∂(μ.tilted f)
      = ∫⁻ x in s, ENNReal.ofReal (exp (f x) / ∫ x, exp (f x) ∂μ) * g x ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    g : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((μ.tilted f).restrict s) fun x => g x) (Measure …
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      g : α → ENNReal
      s : Set α
      hs : MeasurableSet s
      hf : AEMeasurable f μ
      ⊢ Eq (MeasureTheory.lintegral ((μ.tilted f).restrict s) fun x => g x) (Measure …
    -/
  · rw [Measure.tilted, setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀]
      /-
        case pos
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        g : α → ENNReal
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HMul.hMul (fun x => ENNR …
      -/
    · simp only [Pi.mul_apply]
      /-
        🎉 no goals
      -/
      /-
        case pos.hf
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        g : α → ENNReal
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ AEMeasurable (fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTh …
      -/
    · refine AEMeasurable.restrict ?_
      /-
        case pos.hf
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        g : α → ENNReal
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ AEMeasurable (fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTh …
      -/
      exact ((measurable_exp.comp_aemeasurable hf).div_const _).ennreal_ofReal
      /-
        🎉 no goals
      -/
      /-
        case pos.hs
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        g : α → ENNReal
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ MeasurableSet s
      -/
    · exact hs
      /-
        🎉 no goals
      -/
      /-
        case pos.h'f
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        g : α → ENNReal
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ Filter.Eventually (fun x => LT.lt (ENNReal.ofReal (HDiv.hDiv (Real.exp (f x) …
      -/
    · filter_upwards
      /-
        case pos.h'f.h
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        g : α → ENNReal
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ ∀ (a : α), LT.lt (ENNReal.ofReal (HDiv.hDiv (Real.exp (f a)) (MeasureTheory. …
      -/
      simp only [ENNReal.ofReal_lt_top, implies_true]
      /-
        🎉 no goals
      -/
  · have hf' : ¬ Integrable (fun x ↦ exp (f x)) μ := by
      exact fun h ↦ hf (aemeasurable_of_aemeasurable_exp h.1.aemeasurable)
    simp only [hf, not_false_eq_true, tilted_of_not_aemeasurable, Measure.restrict_zero,
      lintegral_zero_measure]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      g : α → ENNReal
      s : Set α
      hs : MeasurableSet s
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.lintegral (μ.restrict s) fun x => HMul.hMul (ENNReal.ofR …
    -/
    rw [integral_undef hf']
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      g : α → ENNReal
      s : Set α
      hs : MeasurableSet s
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.lintegral (μ.restrict s) fun x => HMul.hMul (ENNReal.ofR …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_tilted' := setLIntegral_tilted'


lemma setLIntegral_tilted [SFinite μ] (f : α → ℝ) (g : α → ℝ≥0∞) (s : Set α) :
    ∫⁻ x in s, g x ∂(μ.tilted f)
      = ∫⁻ x in s, ENNReal.ofReal (exp (f x) / ∫ x, exp (f x) ∂μ) * g x ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Real
    g : α → ENNReal
    s : Set α
    ⊢ Eq (MeasureTheory.lintegral ((μ.tilted f).restrict s) fun x => g x) (Measure …
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      g : α → ENNReal
      s : Set α
      hf : AEMeasurable f μ
      ⊢ Eq (MeasureTheory.lintegral ((μ.tilted f).restrict s) fun x => g x) (Measure …
    -/
  · rw [Measure.tilted, setLIntegral_withDensity_eq_setLIntegral_mul_non_measurable₀']
      /-
        case pos
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => HMul.hMul (fun x => ENNR …
      -/
    · simp only [Pi.mul_apply]
      /-
        🎉 no goals
      -/
      /-
        case pos.hf
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        ⊢ AEMeasurable (fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTh …
      -/
    · refine AEMeasurable.restrict ?_
      /-
        case pos.hf
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        ⊢ AEMeasurable (fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTh …
      -/
      exact ((measurable_exp.comp_aemeasurable hf).div_const _).ennreal_ofReal
      /-
        🎉 no goals
      -/
      /-
        case pos.h'f
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        ⊢ Filter.Eventually (fun x => LT.lt (ENNReal.ofReal (HDiv.hDiv (Real.exp (f x) …
      -/
    · filter_upwards
      /-
        case pos.h'f.h
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → ENNReal
        s : Set α
        hf : AEMeasurable f μ
        ⊢ ∀ (a : α), LT.lt (ENNReal.ofReal (HDiv.hDiv (Real.exp (f a)) (MeasureTheory. …
      -/
      simp only [ENNReal.ofReal_lt_top, implies_true]
      /-
        🎉 no goals
      -/
  · have hf' : ¬ Integrable (fun x ↦ exp (f x)) μ := by
      exact fun h ↦ hf (aemeasurable_of_aemeasurable_exp h.1.aemeasurable)
    simp only [hf, not_false_eq_true, tilted_of_not_aemeasurable, Measure.restrict_zero,
      lintegral_zero_measure]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      g : α → ENNReal
      s : Set α
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.lintegral (μ.restrict s) fun x => HMul.hMul (ENNReal.ofR …
    -/
    rw [integral_undef hf']
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      g : α → ENNReal
      s : Set α
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.lintegral (μ.restrict s) fun x => HMul.hMul (ENNReal.ofR …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_tilted := setLIntegral_tilted


lemma lintegral_tilted (f : α → ℝ) (g : α → ℝ≥0∞) :
    ∫⁻ x, g x ∂(μ.tilted f)
      = ∫⁻ x, ENNReal.ofReal (exp (f x) / ∫ x, exp (f x) ∂μ) * (g x) ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    g : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.tilted f) fun x => g x) (MeasureTheory.linteg …
  -/
  rw [← setLIntegral_univ, setLIntegral_tilted' f g MeasurableSet.univ, setLIntegral_univ]
  /-
    🎉 no goals
  -/


lemma setIntegral_tilted' (f : α → ℝ) (g : α → E) {s : Set α} (hs : MeasurableSet s) :
    ∫ x in s, g x ∂(μ.tilted f) = ∫ x in s, (exp (f x) / ∫ x, exp (f x) ∂μ) • (g x) ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → Real
    g : α → E
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral ((μ.tilted f).restrict s) fun x => g x) (MeasureT …
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → Real
      g : α → E
      s : Set α
      hs : MeasurableSet s
      hf : AEMeasurable f μ
      ⊢ Eq (MeasureTheory.integral ((μ.tilted f).restrict s) fun x => g x) (MeasureT …
    -/
  · rw [tilted_eq_withDensity_nnreal, setIntegral_withDensity_eq_setIntegral_smul₀ _ _ hs]
      /-
        case pos
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → Real
        g : α → E
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul ⟨HDiv.hDiv (R …
      -/
    · congr
      /-
        🎉 no goals
      -/
    · suffices AEMeasurable (fun x ↦ exp (f x) / ∫ x, exp (f x) ∂μ) μ by
        rw [← aemeasurable_coe_nnreal_real_iff]
        refine AEMeasurable.restrict ?_
        simpa only [NNReal.coe_mk]
      /-
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → Real
        g : α → E
        s : Set α
        hs : MeasurableSet s
        hf : AEMeasurable f μ
        ⊢ AEMeasurable (fun x => HDiv.hDiv (Real.exp (f x)) (MeasureTheory.integral μ  …
      -/
      exact (measurable_exp.comp_aemeasurable hf).div_const _
      /-
        🎉 no goals
      -/
  · have hf' : ¬ Integrable (fun x ↦ exp (f x)) μ := by
      exact fun h ↦ hf (aemeasurable_of_aemeasurable_exp h.1.aemeasurable)
    simp only [hf, not_false_eq_true, tilted_of_not_aemeasurable, Measure.restrict_zero,
      integral_zero_measure]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → Real
      g : α → E
      s : Set α
      hs : MeasurableSet s
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (HDiv.hDiv  …
    -/
    rw [integral_undef hf']
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → Real
      g : α → E
      s : Set α
      hs : MeasurableSet s
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (HDiv.hDiv  …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_tilted' := setIntegral_tilted'


lemma setIntegral_tilted [SFinite μ] (f : α → ℝ) (g : α → E) (s : Set α) :
    ∫ x in s, g x ∂(μ.tilted f) = ∫ x in s, (exp (f x) / ∫ x, exp (f x) ∂μ) • (g x) ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : α → Real
    g : α → E
    s : Set α
    ⊢ Eq (MeasureTheory.integral ((μ.tilted f).restrict s) fun x => g x) (MeasureT …
  -/
  by_cases hf : AEMeasurable f μ
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      g : α → E
      s : Set α
      hf : AEMeasurable f μ
      ⊢ Eq (MeasureTheory.integral ((μ.tilted f).restrict s) fun x => g x) (MeasureT …
    -/
  · rw [tilted_eq_withDensity_nnreal, setIntegral_withDensity_eq_setIntegral_smul₀']
      /-
        case pos
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → E
        s : Set α
        hf : AEMeasurable f μ
        ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul ⟨HDiv.hDiv (R …
      -/
    · congr
      /-
        🎉 no goals
      -/
    · suffices AEMeasurable (fun x ↦ exp (f x) / ∫ x, exp (f x) ∂μ) μ by
        rw [← aemeasurable_coe_nnreal_real_iff]
        refine AEMeasurable.restrict ?_
        simpa only [NNReal.coe_mk]
      /-
        case pos.hf
        α : Type u_1
        mα : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_2
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : MeasureTheory.SFinite μ
        f : α → Real
        g : α → E
        s : Set α
        hf : AEMeasurable f μ
        ⊢ AEMeasurable (fun x => HDiv.hDiv (Real.exp (f x)) (MeasureTheory.integral μ  …
      -/
      exact (measurable_exp.comp_aemeasurable hf).div_const _
      /-
        🎉 no goals
      -/
  · have hf' : ¬ Integrable (fun x ↦ exp (f x)) μ := by
      exact fun h ↦ hf (aemeasurable_of_aemeasurable_exp h.1.aemeasurable)
    simp only [hf, not_false_eq_true, tilted_of_not_aemeasurable, Measure.restrict_zero,
      integral_zero_measure]
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      g : α → E
      s : Set α
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (HDiv.hDiv  …
    -/
    rw [integral_undef hf']
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : MeasureTheory.SFinite μ
      f : α → Real
      g : α → E
      s : Set α
      hf : Not (AEMeasurable f μ)
      hf' : Not (MeasureTheory.Integrable (fun x => Real.exp (f x)) μ)
      ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (HDiv.hDiv  …
    -/
    simp
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_tilted := setIntegral_tilted


lemma integral_tilted (f : α → ℝ) (g : α → E) :
    ∫ x, g x ∂(μ.tilted f) = ∫ x, (exp (f x) / ∫ x, exp (f x) ∂μ) • (g x) ∂μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → Real
    g : α → E
    ⊢ Eq (MeasureTheory.integral (μ.tilted f) fun x => g x) (MeasureTheory.integra …
  -/
  rw [← setIntegral_univ, setIntegral_tilted' f g MeasurableSet.univ, setIntegral_univ]
  /-
    🎉 no goals
  -/


lemma integral_exp_tilted (f g : α → ℝ) :
    ∫ x, exp (g x) ∂(μ.tilted f) = (∫ x, exp ((f + g) x) ∂μ) / ∫ x, exp (f x) ∂μ := by
  cases eq_zero_or_neZero μ with
  | inl h => rw [h]; simp
  | inr h0 =>
    rw [integral_tilted f]
    simp_rw [smul_eq_mul]
    have : ∀ x, (rexp (f x) / ∫ (x : α), rexp (f x) ∂μ) * rexp (g x)
        = (rexp ((f + g) x) / ∫ (x : α), rexp (f x) ∂μ) := by
      intro x
      rw [Pi.add_apply, exp_add]
      ring
    simp_rw [this, div_eq_mul_inv]
    rw [integral_mul_right]


lemma tilted_tilted (hf : Integrable (fun x ↦ exp (f x)) μ) (g : α → ℝ) :
    (μ.tilted f).tilted g = μ.tilted (f + g) := by
  cases eq_zero_or_neZero μ with
  | inl h => simp [h]
  | inr h0 =>
    ext1 s hs
    rw [tilted_apply' _ _ hs, tilted_apply' _ _ hs, setLIntegral_tilted' f _ hs]
    congr with x
    rw [← ENNReal.ofReal_mul (by positivity),
      integral_exp_tilted f, Pi.add_apply, exp_add]
    congr 1
    simp only [Pi.add_apply]
    field_simp
    ring_nf
    congr 1
    rw [mul_assoc, mul_inv_cancel₀, mul_one]
    exact (integral_exp_pos hf).ne'


lemma tilted_comm (hf : Integrable (fun x ↦ exp (f x)) μ) {g : α → ℝ}
    (hg : Integrable (fun x ↦ exp (g x)) μ) :
    (μ.tilted f).tilted g = (μ.tilted g).tilted f := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    g : α → Real
    hg : MeasureTheory.Integrable (fun x => Real.exp (g x)) μ
    ⊢ Eq ((μ.tilted f).tilted g) ((μ.tilted g).tilted f)
  -/
  rw [tilted_tilted hf, add_comm, tilted_tilted hg]
  /-
    🎉 no goals
  -/


@[simp]
lemma tilted_neg_same' (hf : Integrable (fun x ↦ exp (f x)) μ) :
    (μ.tilted f).tilted (-f) = (μ Set.univ)⁻¹ • μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    ⊢ Eq ((μ.tilted f).tilted (Neg.neg f)) (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)
  -/
  rw [tilted_tilted hf]; simp
                         /-
                           🎉 no goals
                         -/


@[simp]
lemma tilted_neg_same [IsProbabilityMeasure μ] (hf : Integrable (fun x ↦ exp (f x)) μ) :
    (μ.tilted f).tilted (-f) = μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) μ
    ⊢ Eq ((μ.tilted f).tilted (Neg.neg f)) μ
  -/
  simp [hf]
  /-
    🎉 no goals
  -/


lemma tilted_absolutelyContinuous (μ : Measure α) (f : α → ℝ) : μ.tilted f ≪ μ :=
  withDensity_absolutelyContinuous _ _


lemma absolutelyContinuous_tilted (hf : Integrable (fun x ↦ exp (f x)) μ) : μ ≪ μ.tilted f := by
  cases eq_zero_or_neZero μ with
  | inl h => simp only [h, tilted_zero_measure]; exact fun _ _ ↦ by simp
  | inr h0 =>
    refine withDensity_absolutelyContinuous' ?_ ?_
    · exact (hf.1.aemeasurable.div_const _).ennreal_ofReal
    · filter_upwards
      simp only [ne_eq, ENNReal.ofReal_eq_zero, not_le]
      exact fun _ ↦ div_pos (exp_pos _) (integral_exp_pos hf)


lemma rnDeriv_tilted_right (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν]
    (hf : Integrable (fun x ↦ exp (f x)) ν) :
    μ.rnDeriv (ν.tilted f)
      =ᵐ[ν] fun x ↦ ENNReal.ofReal (exp (- f x) * ∫ x, exp (f x) ∂ν) * μ.rnDeriv ν x := by
  cases eq_zero_or_neZero ν with
  | inl h => simp_rw [h, ae_zero, Filter.EventuallyEq]; exact Filter.eventually_bot
  | inr h0 =>
    refine (Measure.rnDeriv_withDensity_right μ ν ?_ ?_ ?_).trans ?_
    · exact (hf.1.aemeasurable.div_const _).ennreal_ofReal
    · filter_upwards
      simp only [ne_eq, ENNReal.ofReal_eq_zero, not_le]
      exact fun _ ↦ div_pos (exp_pos _) (integral_exp_pos hf)
    · refine ae_of_all _ (by simp)
    · filter_upwards with x
      congr
      rw [← ENNReal.ofReal_inv_of_pos, inv_div', ← exp_neg, div_eq_mul_inv, inv_inv]
      exact div_pos (exp_pos _) (integral_exp_pos hf)


lemma toReal_rnDeriv_tilted_right (μ ν : Measure α) [SigmaFinite μ] [SigmaFinite ν]
    (hf : Integrable (fun x ↦ exp (f x)) ν) :
    (fun x ↦ (μ.rnDeriv (ν.tilted f) x).toReal)
      =ᵐ[ν] fun x ↦ exp (- f x) * (∫ x, exp (f x) ∂ν) * (μ.rnDeriv ν x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    f : α → Real
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => (μ.rnDeriv (ν.tilted f) x).toRea …
  -/
  filter_upwards [rnDeriv_tilted_right μ ν hf] with x hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    f : α → Real
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) ν
    x : α
    hx : Eq (μ.rnDeriv (ν.tilted f) x) (HMul.hMul (ENNReal.ofReal (HMul.hMul (Real …
    ⊢ Eq (μ.rnDeriv (ν.tilted f) x).toReal (HMul.hMul (HMul.hMul (Real.exp (Neg.ne …
  -/
  rw [hx]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    f : α → Real
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) ν
    x : α
    hx : Eq (μ.rnDeriv (ν.tilted f) x) (HMul.hMul (ENNReal.ofReal (HMul.hMul (Real …
    ⊢ Eq (HMul.hMul (ENNReal.ofReal (HMul.hMul (Real.exp (Neg.neg (f x))) (Measure …
  -/
  simp only [ENNReal.toReal_mul, gt_iff_lt, mul_eq_mul_right_iff, ENNReal.toReal_ofReal_eq_iff]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    f : α → Real
    μ ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hf : MeasureTheory.Integrable (fun x => Real.exp (f x)) ν
    x : α
    hx : Eq (μ.rnDeriv (ν.tilted f) x) (HMul.hMul (ENNReal.ofReal (HMul.hMul (Real …
    ⊢ Or (LE.le 0 (HMul.hMul (Real.exp (Neg.neg (f x))) (MeasureTheory.integral ν  …
  -/
  exact Or.inl (by positivity)
  /-
    🎉 no goals
  -/


variable (μ) in
lemma rnDeriv_tilted_left {ν : Measure α} [SigmaFinite μ] [SigmaFinite ν] (hfν : AEMeasurable f ν) :
    (μ.tilted f).rnDeriv ν
      =ᵐ[ν] fun x ↦ ENNReal.ofReal (exp (f x) / (∫ x, exp (f x) ∂μ)) * μ.rnDeriv ν x := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.tilted f).rnDeriv ν) fun x => HMul.hMu …
  -/
  let g := fun x ↦ ENNReal.ofReal (exp (f x) / (∫ x, exp (f x) ∂μ))
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    g : α → ENNReal := fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (Measur …
    ⊢ (MeasureTheory.ae ν).EventuallyEq ((μ.tilted f).rnDeriv ν) fun x => HMul.hMu …
  -/
  refine Measure.rnDeriv_withDensity_left (μ := μ) (ν := ν) (f := g) ?_ ?_
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hfν : AEMeasurable f ν
      g : α → ENNReal := fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (Measur …
      ⊢ AEMeasurable g ν
    -/
  · exact ((measurable_exp.comp_aemeasurable hfν).div_const _).ennreal_ofReal
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      ν : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SigmaFinite μ
      inst✝ : MeasureTheory.SigmaFinite ν
      hfν : AEMeasurable f ν
      g : α → ENNReal := fun x => ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (Measur …
      ⊢ Filter.Eventually (fun x => Ne (g x) Top.top) (MeasureTheory.ae μ)
    -/
  · exact ae_of_all _ (fun x ↦ by simp [g])
    /-
      🎉 no goals
    -/


variable (μ) in
lemma toReal_rnDeriv_tilted_left {ν : Measure α} [SigmaFinite μ] [SigmaFinite ν]
    (hfν : AEMeasurable f ν) :
    (fun x ↦ ((μ.tilted f).rnDeriv ν x).toReal)
      =ᵐ[ν] fun x ↦ exp (f x) / (∫ x, exp (f x) ∂μ) * (μ.rnDeriv ν x).toReal := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    ⊢ (MeasureTheory.ae ν).EventuallyEq (fun x => ((μ.tilted f).rnDeriv ν x).toRea …
  -/
  filter_upwards [rnDeriv_tilted_left μ hfν] with x hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    x : α
    hx : Eq ((μ.tilted f).rnDeriv ν x) (HMul.hMul (ENNReal.ofReal (HDiv.hDiv (Real …
    ⊢ Eq ((μ.tilted f).rnDeriv ν x).toReal (HMul.hMul (HDiv.hDiv (Real.exp (f x))  …
  -/
  rw [hx]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    x : α
    hx : Eq ((μ.tilted f).rnDeriv ν x) (HMul.hMul (ENNReal.ofReal (HDiv.hDiv (Real …
    ⊢ Eq (HMul.hMul (ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTheory.int …
  -/
  simp only [ENNReal.toReal_mul, mul_eq_mul_right_iff, ENNReal.toReal_ofReal_eq_iff]
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ν : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SigmaFinite μ
    inst✝ : MeasureTheory.SigmaFinite ν
    hfν : AEMeasurable f ν
    x : α
    hx : Eq ((μ.tilted f).rnDeriv ν x) (HMul.hMul (ENNReal.ofReal (HDiv.hDiv (Real …
    ⊢ Or (LE.le 0 (HDiv.hDiv (Real.exp (f x)) (MeasureTheory.integral μ fun x => R …
  -/
  exact Or.inl (by positivity)
  /-
    🎉 no goals
  -/


lemma rnDeriv_tilted_left_self [SigmaFinite μ] (hf : AEMeasurable f μ) :
    (μ.tilted f).rnDeriv μ =ᵐ[μ] fun x ↦ ENNReal.ofReal (exp (f x) / ∫ x, exp (f x) ∂μ) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.SigmaFinite μ
    hf : AEMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ((μ.tilted f).rnDeriv μ) fun x => ENNReal. …
  -/
  refine (rnDeriv_tilted_left μ hf).trans ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.SigmaFinite μ
    hf : AEMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => HMul.hMul (ENNReal.ofReal (HDiv. …
  -/
  filter_upwards [Measure.rnDeriv_self μ] with x hx
  /-
    case h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.SigmaFinite μ
    hf : AEMeasurable f μ
    x : α
    hx : Eq (μ.rnDeriv μ x) 1
    ⊢ Eq (HMul.hMul (ENNReal.ofReal (HDiv.hDiv (Real.exp (f x)) (MeasureTheory.int …
  -/
  rw [hx, mul_one]
  /-
    🎉 no goals
  -/


lemma log_rnDeriv_tilted_left_self [SigmaFinite μ] (hf : Integrable (fun x ↦ exp (f x)) μ) :
    (fun x ↦ log ((μ.tilted f).rnDeriv μ x).toReal)
      =ᵐ[μ] fun x ↦ f x - log (∫ x, exp (f x) ∂μ) := by
  cases eq_zero_or_neZero μ with
  | inl h => simp_rw [h, ae_zero, Filter.EventuallyEq]; exact Filter.eventually_bot
  | inr h0 =>
    have hf' : AEMeasurable f μ := aemeasurable_of_aemeasurable_exp hf.1.aemeasurable
    filter_upwards [rnDeriv_tilted_left_self hf'] with x hx
    rw [hx, ENNReal.toReal_ofReal (by positivity), log_div (exp_pos _).ne', log_exp]
    exact (integral_exp_pos hf).ne'


