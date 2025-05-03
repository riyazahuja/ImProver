lemma apply_le_nndist_zero {X : Type*} [TopologicalSpace X] (f : X →ᵇ ℝ≥0) (x : X) :
    f x ≤ nndist 0 f := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    f : BoundedContinuousFunction X NNReal
    x : X
    ⊢ LE.le (f x) (NNDist.nndist 0 f)
  -/
  convert nndist_coe_le_nndist x
  /-
    case h.e'_3
    X : Type u_1
    inst✝ : TopologicalSpace X
    f : BoundedContinuousFunction X NNReal
    x : X
    ⊢ Eq (f x) (NNDist.nndist (0 x) (f x))
  -/
  simp only [coe_zero, Pi.zero_apply, NNReal.nndist_zero_eq_val]
  /-
    🎉 no goals
  -/


lemma lintegral_le_edist_mul (f : X →ᵇ ℝ≥0) (μ : Measure X) :
    (∫⁻ x, f x ∂μ) ≤ edist 0 f * (μ Set.univ) :=
                                                                                            /-
                                                                                              X : Type u_1
                                                                                              inst✝¹ : MeasurableSpace X
                                                                                              inst✝ : TopologicalSpace X
                                                                                              f : BoundedContinuousFunction X NNReal
                                                                                              μ : MeasureTheory.Measure X
                                                                                              ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(NNDist.nndist 0 f)) (HMul.hMul ( …
                                                                                            -/
  le_trans (lintegral_mono (fun x ↦ ENNReal.coe_le_coe.mpr (f.apply_le_nndist_zero x))) (by simp)
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem measurable_coe_ennreal_comp [OpensMeasurableSpace X] (f : X →ᵇ ℝ≥0) :
    Measurable fun x ↦ (f x : ℝ≥0∞) :=
  measurable_coe_nnreal_ennreal.comp f.continuous.measurable


theorem lintegral_lt_top_of_nnreal (f : X →ᵇ ℝ≥0) : ∫⁻ x, f x ∂μ < ∞ := by
  /-
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X NNReal
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ↑(f x)) Top.top
  -/
  apply IsFiniteMeasure.lintegral_lt_top_of_bounded_to_ennreal
  /-
    case f_bdd
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X NNReal
    ⊢ Exists fun c => ∀ (x : X), LE.le ↑(f x) ↑c
  -/
  refine ⟨nndist f 0, fun x ↦ ?_⟩
  /-
    case f_bdd
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X NNReal
    x : X
    ⊢ LE.le ↑(f x) ↑(NNDist.nndist f 0)
  -/
  have key := BoundedContinuousFunction.NNReal.upper_bound f x
  /-
    case f_bdd
    X : Type u_1
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X NNReal
    x : X
    key : LE.le (f x) (NNDist.nndist f 0)
    ⊢ LE.le ↑(f x) ↑(NNDist.nndist f 0)
  -/
  rwa [ENNReal.coe_le_coe]
  /-
    🎉 no goals
  -/


theorem integrable_of_nnreal [OpensMeasurableSpace X] (f : X →ᵇ ℝ≥0) :
    Integrable (((↑) : ℝ≥0 → ℝ) ∘ ⇑f) μ := by
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : OpensMeasurableSpace X
    f : BoundedContinuousFunction X NNReal
    ⊢ MeasureTheory.Integrable (Function.comp NNReal.toReal ⇑f) μ
  -/
  refine ⟨(NNReal.continuous_coe.comp f.continuous).measurable.aestronglyMeasurable, ?_⟩
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : OpensMeasurableSpace X
    f : BoundedContinuousFunction X NNReal
    ⊢ MeasureTheory.HasFiniteIntegral (Function.comp NNReal.toReal ⇑f) μ
  -/
  simp only [hasFiniteIntegral_iff_nnnorm, Function.comp_apply, NNReal.nnnorm_eq]
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : OpensMeasurableSpace X
    f : BoundedContinuousFunction X NNReal
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ↑(f a)) Top.top
  -/
  exact lintegral_lt_top_of_nnreal _ f
  /-
    🎉 no goals
  -/


theorem integral_eq_integral_nnrealPart_sub [OpensMeasurableSpace X] (f : X →ᵇ ℝ) :
    ∫ x, f x ∂μ = (∫ x, (f.nnrealPart x : ℝ) ∂μ) - ∫ x, ((-f).nnrealPart x : ℝ) ∂μ := by
  simp only [f.self_eq_nnrealPart_sub_nnrealPart_neg, Pi.sub_apply, integral_sub,
             integrable_of_nnreal]
  /-
    X : Type u_1
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : OpensMeasurableSpace X
    f : BoundedContinuousFunction X Real
    ⊢ Eq (HSub.hSub (MeasureTheory.integral μ fun a => Function.comp NNReal.toReal …
  -/
  simp only [Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem lintegral_of_real_lt_top (f : X →ᵇ ℝ) :
    ∫⁻ x, ENNReal.ofReal (f x) ∂μ < ∞ := lintegral_lt_top_of_nnreal _ f.nnrealPart


theorem toReal_lintegral_coe_eq_integral [OpensMeasurableSpace X] (f : X →ᵇ ℝ≥0) (μ : Measure X) :
    (∫⁻ x, (f x : ℝ≥0∞) ∂μ).toReal = ∫ x, (f x : ℝ) ∂μ := by
  rw [integral_eq_lintegral_of_nonneg_ae _ (by simpa [Function.comp_apply] using
        (NNReal.continuous_coe.comp f.continuous).measurable.aestronglyMeasurable)]
    /-
      X : Type u_1
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : OpensMeasurableSpace X
      f : BoundedContinuousFunction X NNReal
      μ : MeasureTheory.Measure X
      ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(f x)).toReal (MeasureTheory.lintegr …
    -/
  · simp only [ENNReal.ofReal_coe_nnreal]
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : OpensMeasurableSpace X
      f : BoundedContinuousFunction X NNReal
      μ : MeasureTheory.Measure X
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => ↑(f x)
    -/
  · exact Eventually.of_forall (by simp only [Pi.zero_apply, NNReal.zero_le_coe, imp_true_iff])
    /-
      🎉 no goals
    -/


lemma lintegral_nnnorm_le (f : X →ᵇ E) :
    ∫⁻ x, ‖f x‖₊ ∂μ ≤ ‖f‖₊ * (μ Set.univ) := by
  calc  ∫⁻ x, ‖f x‖₊ ∂μ
    _ ≤ ∫⁻ _, ‖f‖₊ ∂μ         := by gcongr; apply nnnorm_coe_le_nnnorm
    _ = ‖f‖₊ * (μ Set.univ)   := by rw [lintegral_const]


lemma integrable [IsFiniteMeasure μ] (f : X →ᵇ E) :
    Integrable f μ := by
  /-
    X : Type u_1
    inst✝⁷ : MeasurableSpace X
    inst✝⁶ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : OpensMeasurableSpace X
    inst✝³ : SecondCountableTopology E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X E
    ⊢ MeasureTheory.Integrable (⇑f) μ
  -/
  refine ⟨f.continuous.measurable.aestronglyMeasurable, (hasFiniteIntegral_def _ _).mp ?_⟩
  calc  ∫⁻ x, ‖f x‖₊ ∂μ
    _ ≤ ‖f‖₊ * (μ Set.univ)   := f.lintegral_nnnorm_le μ
    _ < ∞                     := ENNReal.mul_lt_top ENNReal.coe_lt_top (measure_lt_top μ Set.univ)


lemma norm_integral_le_mul_norm [IsFiniteMeasure μ] (f : X →ᵇ E) :
    ‖∫ x, f x ∂μ‖ ≤ ENNReal.toReal (μ Set.univ) * ‖f‖ := by
  calc  ‖∫ x, f x ∂μ‖
    _ ≤ ∫ x, ‖f x‖ ∂μ                       := by exact norm_integral_le_integral_norm _
    _ ≤ ∫ _, ‖f‖ ∂μ                         := ?_
    _ = ENNReal.toReal (μ Set.univ) • ‖f‖   := by rw [integral_const]
  /-
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X E
    ⊢ LE.le (MeasureTheory.integral μ fun x => Norm.norm (f x)) (MeasureTheory.int …
  -/
  apply integral_mono _ (integrable_const ‖f‖) (fun x ↦ f.norm_coe_le_norm x) -- NOTE: `gcongr`?
  /-
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X E
    ⊢ MeasureTheory.Integrable (fun x => Norm.norm (f x)) μ
  -/
  exact (integrable_norm_iff f.continuous.measurable.aestronglyMeasurable).mpr (f.integrable μ)
  /-
    🎉 no goals
  -/


lemma norm_integral_le_norm [IsProbabilityMeasure μ] (f : X →ᵇ E) :
    ‖∫ x, f x ∂μ‖ ≤ ‖f‖ := by
  /-
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    f : BoundedContinuousFunction X E
    ⊢ LE.le (Norm.norm (MeasureTheory.integral μ fun x => f x)) (Norm.norm f)
  -/
  convert f.norm_integral_le_mul_norm μ
  /-
    case h.e'_4
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    f : BoundedContinuousFunction X E
    ⊢ Eq (Norm.norm f) (HMul.hMul (μ Set.univ).toReal (Norm.norm f))
  -/
  simp only [measure_univ, ENNReal.one_toReal, one_mul]
  /-
    🎉 no goals
  -/


lemma isBounded_range_integral
    {ι : Type*} (μs : ι → Measure X) [∀ i, IsProbabilityMeasure (μs i)] (f : X →ᵇ E) :
    Bornology.IsBounded (Set.range (fun i ↦ ∫ x, f x ∂ (μs i))) := by
  /-
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    ι : Type u_3
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    f : BoundedContinuousFunction X E
    ⊢ Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fun x  …
  -/
  apply isBounded_iff_forall_norm_le.mpr ⟨‖f‖, fun v hv ↦ ?_⟩
  /-
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    ι : Type u_3
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    f : BoundedContinuousFunction X E
    v : E
    hv : Membership.mem (Set.range fun i => MeasureTheory.integral (μs i) fun x => …
    ⊢ LE.le (Norm.norm v) (Norm.norm f)
  -/
  obtain ⟨i, hi⟩ := hv
  /-
    case intro
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    ι : Type u_3
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    f : BoundedContinuousFunction X E
    v : E
    i : ι
    hi : Eq ((fun i => MeasureTheory.integral (μs i) fun x => f x) i) v
    ⊢ LE.le (Norm.norm v) (Norm.norm f)
  -/
  rw [← hi]
  /-
    case intro
    X : Type u_1
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : SecondCountableTopology E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : NormedSpace Real E
    ι : Type u_3
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    f : BoundedContinuousFunction X E
    v : E
    i : ι
    hi : Eq ((fun i => MeasureTheory.integral (μs i) fun x => f x) i) v
    ⊢ LE.le (Norm.norm ((fun i => MeasureTheory.integral (μs i) fun x => f x) i))  …
  -/
  apply f.norm_integral_le_norm (μs i)
  /-
    🎉 no goals
  -/


lemma integral_add_const (f : X →ᵇ ℝ) (c : ℝ) :
    ∫ x, (f + const X c) x ∂μ = ∫ x, f x ∂μ + ENNReal.toReal (μ (Set.univ)) • c := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : OpensMeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X Real
    c : Real
    ⊢ Eq (MeasureTheory.integral μ fun x => (HAdd.hAdd f (BoundedContinuousFunctio …
  -/
  simp [integral_add (f.integrable _) (integrable_const c)]
  /-
    🎉 no goals
  -/


lemma integral_const_sub (f : X →ᵇ ℝ) (c : ℝ) :
    ∫ x, (const X c - f) x ∂μ = ENNReal.toReal (μ (Set.univ)) • c - ∫ x, f x ∂μ := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : MeasurableSpace X
    inst✝¹ : OpensMeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : BoundedContinuousFunction X Real
    c : Real
    ⊢ Eq (MeasureTheory.integral μ fun x => (HSub.hSub (BoundedContinuousFunction. …
  -/
  simp [integral_sub (integrable_const c) (f.integrable _)]
  /-
    🎉 no goals
  -/


lemma tendsto_integral_of_forall_limsup_integral_le_integral {ι : Type*} {L : Filter ι}
    {μ : Measure X} [IsProbabilityMeasure μ] {μs : ι → Measure X} [∀ i, IsProbabilityMeasure (μs i)]
    (h : ∀ f : X →ᵇ ℝ, 0 ≤ f → L.limsup (fun i ↦ ∫ x, f x ∂ (μs i)) ≤ ∫ x, f x ∂μ)
    (f : X →ᵇ ℝ) :
    Tendsto (fun i ↦ ∫ x, f x ∂ (μs i)) L (𝓝 (∫ x, f x ∂μ)) := by
  /-
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
    f : BoundedContinuousFunction X Real
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  rcases eq_or_neBot L with rfl|hL
    /-
      case inl
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction X Real
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) Bot.bot …
    -/
  · simp only [tendsto_bot]
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  have obs := BoundedContinuousFunction.isBounded_range_integral μs f
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  have bdd_above := BddAbove.isBoundedUnder L.univ_mem (by simpa using obs.bddAbove)
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
    bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  have bdd_below := BddBelow.isBoundedUnder L.univ_mem (by simpa using obs.bddBelow)
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
    bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
    bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  apply tendsto_of_le_liminf_of_limsup_le _ _ bdd_above bdd_below
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
  · have key := h _ (f.norm_sub_nonneg)
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => (H …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    simp_rw [f.integral_const_sub ‖f‖] at key
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => HSub.hSub (HSMul.hSMul ((μs i) Set.univ). …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    simp only [measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul] at key
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => HSub.hSub (Norm.norm f) (MeasureTheory.in …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    have := limsup_const_sub L (fun i ↦ ∫ x, f x ∂ (μs i)) ‖f‖ bdd_above.isCobounded_ge bdd_below
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => HSub.hSub (Norm.norm f) (MeasureTheory.in …
      this : Eq (Filter.limsup (fun i => HSub.hSub (Norm.norm f) ((fun i => MeasureT …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    rwa [this, _root_.sub_le_sub_iff_left ‖f‖] at key
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
  · have key := h _ (f.add_norm_nonneg)
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => (H …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    simp_rw [f.integral_add_const ‖f‖] at key
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => HAdd.hAdd (MeasureTheory.integral (μs i)  …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    simp only [measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul] at key
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => HAdd.hAdd (MeasureTheory.integral (μs i)  …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    have := limsup_add_const L (fun i ↦ ∫ x, f x ∂ (μs i)) ‖f‖ bdd_above bdd_below.isCobounded_le
    /-
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (Filter.limsup …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (Filter.limsup (fun i => HAdd.hAdd (MeasureTheory.integral (μs i)  …
      this : Eq (Filter.limsup (fun i => HAdd.hAdd ((fun i => MeasureTheory.integral …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    rwa [this, add_le_add_iff_right] at key
    /-
      🎉 no goals
    -/


lemma tendsto_integral_of_forall_integral_le_liminf_integral {ι : Type*} {L : Filter ι}
    {μ : Measure X} [IsProbabilityMeasure μ] {μs : ι → Measure X} [∀ i, IsProbabilityMeasure (μs i)]
    (h : ∀ f : X →ᵇ ℝ, 0 ≤ f → ∫ x, f x ∂μ ≤ L.liminf (fun i ↦ ∫ x, f x ∂ (μs i)))
    (f : X →ᵇ ℝ) :
    Tendsto (fun i ↦ ∫ x, f x ∂ (μs i)) L (𝓝 (∫ x, f x ∂μ)) := by
  /-
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
    f : BoundedContinuousFunction X Real
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  rcases eq_or_neBot L with rfl|hL
    /-
      case inl
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction X Real
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) Bot.bot …
    -/
  · simp only [tendsto_bot]
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  have obs := BoundedContinuousFunction.isBounded_range_integral μs f
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  have bdd_above := BddAbove.isBoundedUnder L.univ_mem (by simpa using obs.bddAbove)
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
    bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  have bdd_below := BddBelow.isBoundedUnder L.univ_mem (by simpa using obs.bddBelow)
  /-
    case inr
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : MeasurableSpace X
    inst✝² : OpensMeasurableSpace X
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure X
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure X
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
    f : BoundedContinuousFunction X Real
    hL : L.NeBot
    obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
    bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
    bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μs i) fun x => f x) L (nhds …
  -/
  apply @tendsto_of_le_liminf_of_limsup_le ℝ ι _ _ _ L (fun i ↦ ∫ x, f x ∂ (μs i)) (∫ x, f x ∂μ)
    /-
      case inr.hinf
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
  · have key := h _ (f.add_norm_nonneg)
    /-
      case inr.hinf
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (MeasureTheory.integral μ fun x => (HAdd.hAdd f (BoundedContinuous …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    simp_rw [f.integral_add_const ‖f‖] at key
    /-
      case inr.hinf
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (HAdd.hAdd (MeasureTheory.integral μ fun x => f x) (HSMul.hSMul (μ …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    simp only [measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul] at key
    /-
      case inr.hinf
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (HAdd.hAdd (MeasureTheory.integral μ fun x => f x) (Norm.norm f))  …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    have := liminf_add_const L (fun i ↦ ∫ x, f x ∂ (μs i)) ‖f‖ bdd_above.isCobounded_ge bdd_below
    /-
      case inr.hinf
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (HAdd.hAdd (MeasureTheory.integral μ fun x => f x) (Norm.norm f))  …
      this : Eq (Filter.liminf (fun i => HAdd.hAdd ((fun i => MeasureTheory.integral …
      ⊢ LE.le (MeasureTheory.integral μ fun x => f x) (Filter.liminf (fun i => Measu …
    -/
    rwa [this, add_le_add_iff_right] at key
    /-
      🎉 no goals
    -/
    /-
      case inr.hsup
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
  · have key := h _ (f.norm_sub_nonneg)
    /-
      case inr.hsup
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (MeasureTheory.integral μ fun x => (HSub.hSub (BoundedContinuousFu …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    simp_rw [f.integral_const_sub ‖f‖] at key
    /-
      case inr.hsup
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (HSub.hSub (HSMul.hSMul (μ Set.univ).toReal (Norm.norm f)) (Measur …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    simp only [measure_univ, ENNReal.one_toReal, smul_eq_mul, one_mul] at key
    /-
      case inr.hsup
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (HSub.hSub (Norm.norm f) (MeasureTheory.integral μ fun x => f x))  …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    have := liminf_const_sub L (fun i ↦ ∫ x, f x ∂ (μs i)) ‖f‖ bdd_above bdd_below.isCobounded_le
    /-
      case inr.hsup
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      key : LE.le (HSub.hSub (Norm.norm f) (MeasureTheory.integral μ fun x => f x))  …
      this : Eq (Filter.liminf (fun i => HSub.hSub (Norm.norm f) ((fun i => MeasureT …
      ⊢ LE.le (Filter.limsup (fun i => MeasureTheory.integral (μs i) fun x => f x) L …
    -/
    rwa [this, sub_le_sub_iff_left] at key
    /-
      🎉 no goals
    -/
    /-
      case inr.h
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      ⊢ autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measu …
    -/
  · exact bdd_above
    /-
      🎉 no goals
    -/
    /-
      case inr.h'
      X : Type u_1
      inst✝⁴ : TopologicalSpace X
      inst✝³ : MeasurableSpace X
      inst✝² : OpensMeasurableSpace X
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure X
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      μs : ι → MeasureTheory.Measure X
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      h : ∀ (f : BoundedContinuousFunction X Real), LE.le 0 f → LE.le (MeasureTheory …
      f : BoundedContinuousFunction X Real
      hL : L.NeBot
      obs : Bornology.IsBounded (Set.range fun i => MeasureTheory.integral (μs i) fu …
      bdd_above : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => Measur …
      bdd_below : Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measur …
      ⊢ autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => Measu …
    -/
  · exact bdd_below
    /-
      🎉 no goals
    -/


