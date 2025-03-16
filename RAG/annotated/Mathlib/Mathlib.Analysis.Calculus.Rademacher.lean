theorem memℒp_lineDeriv (hf : LipschitzWith C f) (v : E) :
    Memℒp (fun x ↦ lineDeriv ℝ f x v) ∞ μ :=
  memℒp_top_of_bound (aestronglyMeasurable_lineDeriv hf.continuous μ)
    (C * ‖v‖) (.of_forall fun _x ↦ norm_lineDeriv_le_of_lipschitz ℝ hf)


theorem ae_lineDifferentiableAt
    (hf : LipschitzWith C f) (v : E) :
    ∀ᵐ p ∂μ, LineDifferentiableAt ℝ f p v := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    ⊢ Filter.Eventually (fun p => LineDifferentiableAt Real f p v) (MeasureTheory. …
  -/
  let L : ℝ →L[ℝ] E := ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) v
  suffices A : ∀ p, ∀ᵐ (t : ℝ) ∂volume, LineDifferentiableAt ℝ f (p + t • v) v from
    ae_mem_of_ae_add_linearMap_mem L.toLinearMap volume μ
      (measurableSet_lineDifferentiableAt hf.continuous) A
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    ⊢ ∀ (p : E), Filter.Eventually (fun t => LineDifferentiableAt Real f (HAdd.hAd …
  -/
  intro p
  have : ∀ᵐ (s : ℝ), DifferentiableAt ℝ (fun t ↦ f (p + t • v)) s :=
    (hf.comp ((LipschitzWith.const p).add L.lipschitz)).ae_differentiableAt_real
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    p : E
    this : Filter.Eventually (fun s => DifferentiableAt Real (fun t => f (HAdd.hAd …
    ⊢ Filter.Eventually (fun t => LineDifferentiableAt Real f (HAdd.hAdd p (HSMul. …
  -/
  filter_upwards [this] with s hs
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    p : E
    this : Filter.Eventually (fun s => DifferentiableAt Real (fun t => f (HAdd.hAd …
    s : Real
    hs : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) s
    ⊢ LineDifferentiableAt Real f (HAdd.hAdd p (HSMul.hSMul s v)) v
  -/
  have h's : DifferentiableAt ℝ (fun t ↦ f (p + t • v)) (s + 0) := by simpa using hs
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    p : E
    this : Filter.Eventually (fun s => DifferentiableAt Real (fun t => f (HAdd.hAd …
    s : Real
    hs : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) s
    h's : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) (HAdd …
    ⊢ LineDifferentiableAt Real f (HAdd.hAdd p (HSMul.hSMul s v)) v
  -/
  have : DifferentiableAt ℝ (fun t ↦ s + t) 0 := differentiableAt_id.const_add _
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    p : E
    this✝ : Filter.Eventually (fun s => DifferentiableAt Real (fun t => f (HAdd.hA …
    s : Real
    hs : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) s
    h's : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) (HAdd …
    this : DifferentiableAt Real (fun t => HAdd.hAdd s t) 0
    ⊢ LineDifferentiableAt Real f (HAdd.hAdd p (HSMul.hSMul s v)) v
  -/
  simp only [LineDifferentiableAt]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    p : E
    this✝ : Filter.Eventually (fun s => DifferentiableAt Real (fun t => f (HAdd.hA …
    s : Real
    hs : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) s
    h's : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) (HAdd …
    this : DifferentiableAt Real (fun t => HAdd.hAdd s t) 0
    ⊢ DifferentiableAt Real (fun t => f (HAdd.hAdd (HAdd.hAdd p (HSMul.hSMul s v)) …
  -/
  convert h's.comp 0 this with _ t
  /-
    case h.e'_11.h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    v : E
    L : ContinuousLinearMap (RingHom.id Real) Real E := ContinuousLinearMap.smulRi …
    p : E
    this✝ : Filter.Eventually (fun s => DifferentiableAt Real (fun t => f (HAdd.hA …
    s : Real
    hs : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) s
    h's : DifferentiableAt Real (fun t => f (HAdd.hAdd p (HSMul.hSMul t v))) (HAdd …
    this : DifferentiableAt Real (fun t => HAdd.hAdd s t) 0
    x✝ : Real
    ⊢ Eq (f (HAdd.hAdd (HAdd.hAdd p (HSMul.hSMul s v)) (HSMul.hSMul x✝ v))) (Funct …
  -/
  simp only [LineDifferentiableAt, add_assoc, Function.comp_apply, add_smul]
  /-
    🎉 no goals
  -/


theorem locallyIntegrable_lineDeriv (hf : LipschitzWith C f) (v : E) :
    LocallyIntegrable (fun x ↦ lineDeriv ℝ f x v) μ :=
  (hf.memℒp_lineDeriv v).locallyIntegrable le_top


theorem integral_inv_smul_sub_mul_tendsto_integral_lineDeriv_mul
    (hf : LipschitzWith C f) (hg : Integrable g μ) (v : E) :
    Tendsto (fun (t : ℝ) ↦ ∫ x, (t⁻¹ • (f (x + t • v) - f x)) * g x ∂μ) (𝓝[>] 0)
      (𝓝 (∫ x, lineDeriv ℝ f x v * g x ∂μ)) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f g : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    hg : MeasureTheory.Integrable g μ
    v : E
    ⊢ Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMul. …
  -/
  apply tendsto_integral_filter_of_dominated_convergence (fun x ↦ (C * ‖v‖) * ‖g x‖)
    /-
      case hF_meas
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      ⊢ Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fun a => HMu …
    -/
  · filter_upwards with t
    /-
      case hF_meas.h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HMul.hMul (HSMul.hSMul (Inv.inv …
    -/
    apply AEStronglyMeasurable.mul ?_ hg.aestronglyMeasurable
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HSMul.hSMul (Inv.inv t) (HSub.h …
    -/
    apply aestronglyMeasurable_const.smul
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSub.hSub (f (HAdd.hAdd x (HSMu …
    -/
    apply AEStronglyMeasurable.sub _ hf.continuous.measurable.aestronglyMeasurable
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => f (HAdd.hAdd x (HSMul.hSMul t v …
    -/
    apply AEMeasurable.aestronglyMeasurable
    /-
      case hf
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      t : Real
      ⊢ AEMeasurable (fun x => f (HAdd.hAdd x (HSMul.hSMul t v))) μ
    -/
    exact hf.continuous.measurable.comp_aemeasurable' (aemeasurable_id'.add_const _)
    /-
      🎉 no goals
    -/
    /-
      case h_bound
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm (HM …
    -/
  · filter_upwards [self_mem_nhdsWithin] with t (ht : 0 < t)
    /-
      case h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      t : Real
      ht : LT.lt 0 t
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HMul.hMul (HSMul.hSMul (Inv.in …
    -/
    filter_upwards with x
    calc ‖t⁻¹ • (f (x + t • v) - f x) * g x‖
      = (t⁻¹ * ‖f (x + t • v) - f x‖) * ‖g x‖ := by simp [norm_mul, ht.le]
    _ ≤ (t⁻¹ * (C * ‖(x + t • v) - x‖)) * ‖g x‖ := by
      gcongr; exact LipschitzWith.norm_sub_le hf (x + t • v) x
    _ = (C * ‖v‖) *‖g x‖ := by field_simp [norm_smul, abs_of_nonneg ht.le]; ring
    /-
      case bound_integrable
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (HMul.hMul (↑C) (Norm.norm v))  …
    -/
  · exact hg.norm.const_mul _
    /-
      🎉 no goals
    -/
    /-
      case h_lim
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v : E
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => HMul.hMul (HSMul.hSMul  …
    -/
  · filter_upwards [hf.ae_lineDifferentiableAt v] with x hx
    /-
      case h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : MeasureTheory.Integrable g μ
      v x : E
      hx : LineDifferentiableAt Real f x v
      ⊢ Filter.Tendsto (fun n => HMul.hMul (HSMul.hSMul (Inv.inv n) (HSub.hSub (f (H …
    -/
    exact hx.hasLineDerivAt.tendsto_slope_zero_right.mul tendsto_const_nhds
    /-
      🎉 no goals
    -/


theorem integral_inv_smul_sub_mul_tendsto_integral_lineDeriv_mul'
    (hf : LipschitzWith C f) (h'f : HasCompactSupport f) (hg : Continuous g) (v : E) :
    Tendsto (fun (t : ℝ) ↦ ∫ x, (t⁻¹ • (f (x + t • v) - f x)) * g x ∂μ) (𝓝[>] 0)
      (𝓝 (∫ x, lineDeriv ℝ f x v * g x ∂μ)) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f g : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    h'f : HasCompactSupport f
    hg : Continuous g
    v : E
    ⊢ Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMul. …
  -/
  let K := cthickening (‖v‖) (tsupport f)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f g : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    h'f : HasCompactSupport f
    hg : Continuous g
    v : E
    K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
    ⊢ Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMul. …
  -/
  have K_compact : IsCompact K := IsCompact.cthickening h'f
  apply tendsto_integral_filter_of_dominated_convergence
      (K.indicator (fun x ↦ (C * ‖v‖) * ‖g x‖))
    /-
      case hF_meas
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      ⊢ Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fun a => HMu …
    -/
  · filter_upwards with t
    /-
      case hF_meas.h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HMul.hMul (HSMul.hSMul (Inv.inv …
    -/
    apply AEStronglyMeasurable.mul ?_ hg.aestronglyMeasurable
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HSMul.hSMul (Inv.inv t) (HSub.h …
    -/
    apply aestronglyMeasurable_const.smul
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSub.hSub (f (HAdd.hAdd x (HSMu …
    -/
    apply AEStronglyMeasurable.sub _ hf.continuous.measurable.aestronglyMeasurable
    /-
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => f (HAdd.hAdd x (HSMul.hSMul t v …
    -/
    apply AEMeasurable.aestronglyMeasurable
    /-
      case hf
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ⊢ AEMeasurable (fun x => f (HAdd.hAdd x (HSMul.hSMul t v))) μ
    -/
    exact hf.continuous.measurable.comp_aemeasurable' (aemeasurable_id'.add_const _)
    /-
      🎉 no goals
    -/
    /-
      case h_bound
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm (HM …
    -/
  · filter_upwards [Ioc_mem_nhdsGT zero_lt_one] with t ht
    /-
      case h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ht : Membership.mem (Set.Ioc 0 1) t
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HMul.hMul (HSMul.hSMul (Inv.in …
    -/
    have t_pos : 0 < t := ht.1
    /-
      case h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ht : Membership.mem (Set.Ioc 0 1) t
      t_pos : LT.lt 0 t
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HMul.hMul (HSMul.hSMul (Inv.in …
    -/
    filter_upwards with x
    /-
      case h.h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      t : Real
      ht : Membership.mem (Set.Ioc 0 1) t
      t_pos : LT.lt 0 t
      x : E
      ⊢ LE.le (Norm.norm (HMul.hMul (HSMul.hSMul (Inv.inv t) (HSub.hSub (f (HAdd.hAd …
    -/
    by_cases hx : x ∈ K
    · calc ‖t⁻¹ • (f (x + t • v) - f x) * g x‖
        = (t⁻¹ * ‖f (x + t • v) - f x‖) * ‖g x‖ := by simp [norm_mul, t_pos.le]
      _ ≤ (t⁻¹ * (C * ‖(x + t • v) - x‖)) * ‖g x‖ := by
        gcongr; exact LipschitzWith.norm_sub_le hf (x + t • v) x
      _ = (C * ‖v‖) *‖g x‖ := by field_simp [norm_smul, abs_of_nonneg t_pos.le]; ring
      _ = K.indicator (fun x ↦ (C * ‖v‖) * ‖g x‖) x := by rw [indicator_of_mem hx]
    · have A : f x = 0 := by
        rw [← Function.nmem_support]
        contrapose! hx
        exact self_subset_cthickening _ (subset_tsupport _ hx)
      have B : f (x + t • v) = 0 := by
        rw [← Function.nmem_support]
        contrapose! hx
        apply mem_cthickening_of_dist_le _ _ (‖v‖) (tsupport f) (subset_tsupport _ hx)
        simp only [dist_eq_norm, sub_add_cancel_left, norm_neg, norm_smul, Real.norm_eq_abs,
          abs_of_nonneg t_pos.le, norm_pos_iff]
        exact mul_le_of_le_one_left (norm_nonneg v) ht.2
      /-
        case neg
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        C : NNReal
        f g : E → Real
        μ : MeasureTheory.Measure E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        hf : LipschitzWith C f
        h'f : HasCompactSupport f
        hg : Continuous g
        v : E
        K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
        K_compact : IsCompact K
        t : Real
        ht : Membership.mem (Set.Ioc 0 1) t
        t_pos : LT.lt 0 t
        x : E
        hx : Not (Membership.mem K x)
        A : Eq (f x) 0
        B : Eq (f (HAdd.hAdd x (HSMul.hSMul t v))) 0
        ⊢ LE.le (Norm.norm (HMul.hMul (HSMul.hSMul (Inv.inv t) (HSub.hSub (f (HAdd.hAd …
      -/
      simp only [B, A, _root_.sub_self, smul_eq_mul, mul_zero, zero_mul, norm_zero]
      /-
        case neg
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        C : NNReal
        f g : E → Real
        μ : MeasureTheory.Measure E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        hf : LipschitzWith C f
        h'f : HasCompactSupport f
        hg : Continuous g
        v : E
        K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
        K_compact : IsCompact K
        t : Real
        ht : Membership.mem (Set.Ioc 0 1) t
        t_pos : LT.lt 0 t
        x : E
        hx : Not (Membership.mem K x)
        A : Eq (f x) 0
        B : Eq (f (HAdd.hAdd x (HSMul.hSMul t v))) 0
        ⊢ LE.le 0 (K.indicator (fun x => HMul.hMul (HMul.hMul (↑C) (Norm.norm v)) (Nor …
      -/
      exact indicator_nonneg (fun y _hy ↦ by positivity) _
      /-
        🎉 no goals
      -/
    /-
      case bound_integrable
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      ⊢ MeasureTheory.Integrable (K.indicator fun x => HMul.hMul (HMul.hMul (↑C) (No …
    -/
  · rw [integrable_indicator_iff K_compact.measurableSet]
    /-
      case bound_integrable
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (HMul.hMul (↑C) (Norm.norm v) …
    -/
    apply ContinuousOn.integrableOn_compact K_compact
    /-
      case bound_integrable
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      ⊢ ContinuousOn (fun x => HMul.hMul (HMul.hMul (↑C) (Norm.norm v)) (Norm.norm ( …
    -/
    exact (Continuous.mul continuous_const hg.norm).continuousOn
    /-
      🎉 no goals
    -/
    /-
      case h_lim
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => HMul.hMul (HSMul.hSMul  …
    -/
  · filter_upwards [hf.ae_lineDifferentiableAt v] with x hx
    /-
      case h
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      h'f : HasCompactSupport f
      hg : Continuous g
      v : E
      K : Set E := Metric.cthickening (Norm.norm v) (tsupport f)
      K_compact : IsCompact K
      x : E
      hx : LineDifferentiableAt Real f x v
      ⊢ Filter.Tendsto (fun n => HMul.hMul (HSMul.hSMul (Inv.inv n) (HSub.hSub (f (H …
    -/
    exact hx.hasLineDerivAt.tendsto_slope_zero_right.mul tendsto_const_nhds
    /-
      🎉 no goals
    -/


/-- Integration by parts formula for the line derivative of Lipschitz functions, assuming one of
them is compactly supported. -/
theorem integral_lineDeriv_mul_eq
    (hf : LipschitzWith C f) (hg : LipschitzWith D g) (h'g : HasCompactSupport g) (v : E) :
    ∫ x, lineDeriv ℝ f x v * g x ∂μ = ∫ x, lineDeriv ℝ g x (-v) * f x ∂μ := by
  /- Write down the line derivative as the limit of `(f (x + t v) - f x) / t` and
  `(g (x - t v) - g x) / t`, and therefore the integrals as limits of the corresponding integrals
  thanks to the dominated convergence theorem. At fixed positive `t`, the integrals coincide
  (with the change of variables `y = x + t v`), so the limits also coincide. -/
  have A : Tendsto (fun (t : ℝ) ↦ ∫ x, (t⁻¹ • (f (x + t • v) - f x)) * g x ∂μ) (𝓝[>] 0)
              (𝓝 (∫ x, lineDeriv ℝ f x v * g x ∂μ)) :=
    integral_inv_smul_sub_mul_tendsto_integral_lineDeriv_mul
      hf (hg.continuous.integrable_of_hasCompactSupport h'g) v
  have B : Tendsto (fun (t : ℝ) ↦ ∫ x, (t⁻¹ • (g (x + t • (-v)) - g x)) * f x ∂μ) (𝓝[>] 0)
              (𝓝 (∫ x, lineDeriv ℝ g x (-v) * f x ∂μ)) :=
    integral_inv_smul_sub_mul_tendsto_integral_lineDeriv_mul' hg h'g hf.continuous (-v)
  suffices S1 : ∀ (t : ℝ), ∫ x, (t⁻¹ • (f (x + t • v) - f x)) * g x ∂μ =
                            ∫ x, (t⁻¹ • (g (x + t • (-v)) - g x)) * f x ∂μ by
    simp only [S1] at A; exact tendsto_nhds_unique A B
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C D : NNReal
    f g : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    hg : LipschitzWith D g
    h'g : HasCompactSupport g
    v : E
    A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
    B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
    ⊢ ∀ (t : Real), Eq (MeasureTheory.integral μ fun x => HMul.hMul (HSMul.hSMul ( …
  -/
  intro t
  suffices S2 : ∫ x, (f (x + t • v) - f x) * g x ∂μ = ∫ x, f x * (g (x + t • (-v)) - g x) ∂μ by
    simp only [smul_eq_mul, mul_assoc, integral_mul_left, S2, mul_neg, mul_comm (f _)]
  have S3 : ∫ x, f (x + t • v) * g x ∂μ = ∫ x, f x * g (x + t • (-v)) ∂μ := by
    rw [← integral_add_right_eq_self _ (t • (-v))]; simp
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C D : NNReal
    f g : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    hg : LipschitzWith D g
    h'g : HasCompactSupport g
    v : E
    A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
    B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
    t : Real
    S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
    ⊢ Eq (MeasureTheory.integral μ fun x => HMul.hMul (HSub.hSub (f (HAdd.hAdd x ( …
  -/
  simp_rw [_root_.sub_mul, _root_.mul_sub]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C D : NNReal
    f g : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    hg : LipschitzWith D g
    h'g : HasCompactSupport g
    v : E
    A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
    B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
    t : Real
    S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
    ⊢ Eq (MeasureTheory.integral μ fun x => HSub.hSub (HMul.hMul (f (HAdd.hAdd x ( …
  -/
  rw [integral_sub, integral_sub, S3]
    /-
      case hf
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C D : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : LipschitzWith D g
      h'g : HasCompactSupport g
      v : E
      A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      t : Real
      S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g (HAdd.hAdd x (HSMul.hS …
    -/
  · apply Continuous.integrable_of_hasCompactSupport
      /-
        case hf.hf
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        C D : NNReal
        f g : E → Real
        μ : MeasureTheory.Measure E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        hf : LipschitzWith C f
        hg : LipschitzWith D g
        h'g : HasCompactSupport g
        v : E
        A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        t : Real
        S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
        ⊢ Continuous fun x => HMul.hMul (f x) (g (HAdd.hAdd x (HSMul.hSMul t (Neg.neg  …
      -/
    · exact hf.continuous.mul (hg.continuous.comp (continuous_add_right _))
      /-
        🎉 no goals
      -/
      /-
        case hf.hcf
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        C D : NNReal
        f g : E → Real
        μ : MeasureTheory.Measure E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        hf : LipschitzWith C f
        hg : LipschitzWith D g
        h'g : HasCompactSupport g
        v : E
        A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        t : Real
        S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
        ⊢ HasCompactSupport fun x => HMul.hMul (f x) (g (HAdd.hAdd x (HSMul.hSMul t (N …
      -/
    · exact (h'g.comp_homeomorph (Homeomorph.addRight (t • (-v)))).mul_left
      /-
        🎉 no goals
      -/
    /-
      case hg
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C D : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : LipschitzWith D g
      h'g : HasCompactSupport g
      v : E
      A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      t : Real
      S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) μ
    -/
  · exact (hf.continuous.mul hg.continuous).integrable_of_hasCompactSupport h'g.mul_left
    /-
      🎉 no goals
    -/
    /-
      case hf
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C D : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : LipschitzWith D g
      h'g : HasCompactSupport g
      v : E
      A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      t : Real
      S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hSMul t  …
    -/
  · apply Continuous.integrable_of_hasCompactSupport
      /-
        case hf.hf
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        C D : NNReal
        f g : E → Real
        μ : MeasureTheory.Measure E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        hf : LipschitzWith C f
        hg : LipschitzWith D g
        h'g : HasCompactSupport g
        v : E
        A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        t : Real
        S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
        ⊢ Continuous fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hSMul t v))) (g x)
      -/
    · exact (hf.continuous.comp (continuous_add_right _)).mul hg.continuous
      /-
        🎉 no goals
      -/
      /-
        case hf.hcf
        E : Type u_1
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        C D : NNReal
        f g : E → Real
        μ : MeasureTheory.Measure E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        hf : LipschitzWith C f
        hg : LipschitzWith D g
        h'g : HasCompactSupport g
        v : E
        A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
        t : Real
        S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
        ⊢ HasCompactSupport fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hSMul t v))) (g x)
      -/
    · exact h'g.mul_left
      /-
        🎉 no goals
      -/
    /-
      case hg
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C D : NNReal
      f g : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      hg : LipschitzWith D g
      h'g : HasCompactSupport g
      v : E
      A : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      B : Filter.Tendsto (fun t => MeasureTheory.integral μ fun x => HMul.hMul (HSMu …
      t : Real
      S3 : Eq (MeasureTheory.integral μ fun x => HMul.hMul (f (HAdd.hAdd x (HSMul.hS …
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) μ
    -/
  · exact (hf.continuous.mul hg.continuous).integrable_of_hasCompactSupport h'g.mul_left
    /-
      🎉 no goals
    -/


/-- The line derivative of a Lipschitz function is almost everywhere linear with respect to fixed
coefficients. -/
theorem ae_lineDeriv_sum_eq
    (hf : LipschitzWith C f) {ι : Type*} (s : Finset ι) (a : ι → ℝ) (v : ι → E) :
    ∀ᵐ x ∂μ, lineDeriv ℝ f x (∑ i ∈ s, a i • v i) = ∑ i ∈ s, a i • lineDeriv ℝ f x (v i) := by
  /- Clever argument by Morrey: integrate against a smooth compactly supported function `g`, switch
  the derivative to `g` by integration by parts, and use the linearity of the derivative of `g` to
  conclude that the initial integrals coincide. -/
  apply ae_eq_of_integral_contDiff_smul_eq (hf.locallyIntegrable_lineDeriv _)
    (locallyIntegrable_finset_sum _ (fun i hi ↦ (hf.locallyIntegrable_lineDeriv (v i)).smul (a i)))
    (fun g g_smooth g_comp ↦ ?_)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    ι : Type u_3
    s : Finset ι
    a : ι → Real
    v : ι → E
    g : E → Real
    g_smooth : ContDiff Real (↑Top.top) g
    g_comp : HasCompactSupport g
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (g x) (lineDeriv Real f x  …
  -/
  simp_rw [Finset.smul_sum]
  have A : ∀ i ∈ s, Integrable (fun x ↦ g x • (a i • fun x ↦ lineDeriv ℝ f x (v i)) x) μ :=
    fun i hi ↦ (g_smooth.continuous.integrable_of_hasCompactSupport g_comp).smul_of_top_left
      ((hf.memℒp_lineDeriv (v i)).const_smul (a i))
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    ι : Type u_3
    s : Finset ι
    a : ι → Real
    v : ι → E
    g : E → Real
    g_smooth : ContDiff Real (↑Top.top) g
    g_comp : HasCompactSupport g
    A : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HSMul.h …
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (g x) (lineDeriv Real f x  …
  -/
  rw [integral_finset_sum _ A]
  suffices S1 : ∫ x, lineDeriv ℝ f x (∑ i ∈ s, a i • v i) * g x ∂μ
      = ∑ i ∈ s, a i * ∫ x, lineDeriv ℝ f x (v i) * g x ∂μ by
    dsimp only [smul_eq_mul, Pi.smul_apply]
    simp_rw [← mul_assoc, mul_comm _ (a _), mul_assoc, integral_mul_left, mul_comm (g _), S1]
  suffices S2 : ∫ x, (∑ i ∈ s, a i * fderiv ℝ g x (v i)) * f x ∂μ =
                  ∑ i ∈ s, a i * ∫ x, fderiv ℝ g x (v i) * f x ∂μ by
    obtain ⟨D, g_lip⟩ : ∃ D, LipschitzWith D g :=
      ContDiff.lipschitzWith_of_hasCompactSupport g_comp g_smooth (mod_cast le_top)
    simp_rw [integral_lineDeriv_mul_eq hf g_lip g_comp]
    simp_rw [(g_smooth.differentiable (mod_cast le_top)).differentiableAt.lineDeriv_eq_fderiv]
    simp only [map_neg, _root_.map_sum, _root_.map_smul, smul_eq_mul, neg_mul]
    simp only [integral_neg, mul_neg, Finset.sum_neg_distrib, neg_inj]
    exact S2
  suffices B : ∀ i ∈ s, Integrable (fun x ↦ a i * (fderiv ℝ g x (v i) * f x)) μ by
    simp_rw [Finset.sum_mul, mul_assoc, integral_finset_sum s B, integral_mul_left]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    ι : Type u_3
    s : Finset ι
    a : ι → Real
    v : ι → E
    g : E → Real
    g_smooth : ContDiff Real (↑Top.top) g
    g_comp : HasCompactSupport g
    A : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HSMul.h …
    ⊢ ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HMul.hMul …
  -/
  intro i _hi
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    ι : Type u_3
    s : Finset ι
    a : ι → Real
    v : ι → E
    g : E → Real
    g_smooth : ContDiff Real (↑Top.top) g
    g_comp : HasCompactSupport g
    A : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HSMul.h …
    i : ι
    _hi : Membership.mem s i
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (a i) (HMul.hMul ((fderiv Real  …
  -/
  let L : (E →L[ℝ] ℝ) → ℝ := fun f ↦ f (v i)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    ι : Type u_3
    s : Finset ι
    a : ι → Real
    v : ι → E
    g : E → Real
    g_smooth : ContDiff Real (↑Top.top) g
    g_comp : HasCompactSupport g
    A : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HSMul.h …
    i : ι
    _hi : Membership.mem s i
    L : ContinuousLinearMap (RingHom.id Real) E Real → Real := fun f => f (v i)
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (a i) (HMul.hMul ((fderiv Real  …
  -/
  change Integrable (fun x ↦ a i * ((L ∘ (fderiv ℝ g)) x * f x)) μ
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    ι : Type u_3
    s : Finset ι
    a : ι → Real
    v : ι → E
    g : E → Real
    g_smooth : ContDiff Real (↑Top.top) g
    g_comp : HasCompactSupport g
    A : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HSMul.h …
    i : ι
    _hi : Membership.mem s i
    L : ContinuousLinearMap (RingHom.id Real) E Real → Real := fun f => f (v i)
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (a i) (HMul.hMul (Function.comp …
  -/
  refine (Continuous.integrable_of_hasCompactSupport ?_ ?_).const_mul _
  · exact ((g_smooth.continuous_fderiv (mod_cast le_top)).clm_apply continuous_const).mul
      hf.continuous
    /-
      case refine_2
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      C : NNReal
      f : E → Real
      μ : MeasureTheory.Measure E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      hf : LipschitzWith C f
      ι : Type u_3
      s : Finset ι
      a : ι → Real
      v : ι → E
      g : E → Real
      g_smooth : ContDiff Real (↑Top.top) g
      g_comp : HasCompactSupport g
      A : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (fun x => HSMul.h …
      i : ι
      _hi : Membership.mem s i
      L : ContinuousLinearMap (RingHom.id Real) E Real → Real := fun f => f (v i)
      ⊢ HasCompactSupport fun x => HMul.hMul (Function.comp L (fderiv Real g) x) (f x)
    -/
  · exact ((g_comp.fderiv ℝ).comp_left rfl).mul_right
    /-
      🎉 no goals
    -/


theorem ae_exists_fderiv_of_countable
    (hf : LipschitzWith C f) {s : Set E} (hs : s.Countable) :
    ∀ᵐ x ∂μ, ∃ (L : E →L[ℝ] ℝ), ∀ v ∈ s, HasLineDerivAt ℝ f (L v) x v := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    hs : s.Countable
    ⊢ Filter.Eventually (fun x => Exists fun L => ∀ (v : E), Membership.mem s v →  …
  -/
  have B := Basis.ofVectorSpace ℝ E
  have I1 : ∀ᵐ (x : E) ∂μ, ∀ v ∈ s, lineDeriv ℝ f x (∑ i, (B.repr v i) • B i) =
                                  ∑ i, B.repr v i • lineDeriv ℝ f x (B i) :=
    (ae_ball_iff hs).2 (fun v _ ↦ hf.ae_lineDeriv_sum_eq _ _ _)
  have I2 : ∀ᵐ (x : E) ∂μ, ∀ v ∈ s, LineDifferentiableAt ℝ f x v :=
    (ae_ball_iff hs).2 (fun v _ ↦ hf.ae_lineDifferentiableAt v)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    hs : s.Countable
    B : Basis (↑(Basis.ofVectorSpaceIndex Real E)) Real E
    I1 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → Eq (lineDeriv …
    I2 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → LineDifferent …
    ⊢ Filter.Eventually (fun x => Exists fun L => ∀ (v : E), Membership.mem s v →  …
  -/
  filter_upwards [I1, I2] with x hx h'x
  let L : E →L[ℝ] ℝ :=
    LinearMap.toContinuousLinearMap (B.constr ℝ (fun i ↦ lineDeriv ℝ f x (B i)))
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    hs : s.Countable
    B : Basis (↑(Basis.ofVectorSpaceIndex Real E)) Real E
    I1 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → Eq (lineDeriv …
    I2 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → LineDifferent …
    x : E
    hx : ∀ (v : E), Membership.mem s v → Eq (lineDeriv Real f x (Finset.univ.sum f …
    h'x : ∀ (v : E), Membership.mem s v → LineDifferentiableAt Real f x v
    L : ContinuousLinearMap (RingHom.id Real) E Real := LinearMap.toContinuousLine …
    ⊢ Exists fun L => ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v)  …
  -/
  refine ⟨L, fun v hv ↦ ?_⟩
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    hs : s.Countable
    B : Basis (↑(Basis.ofVectorSpaceIndex Real E)) Real E
    I1 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → Eq (lineDeriv …
    I2 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → LineDifferent …
    x : E
    hx : ∀ (v : E), Membership.mem s v → Eq (lineDeriv Real f x (Finset.univ.sum f …
    h'x : ∀ (v : E), Membership.mem s v → LineDifferentiableAt Real f x v
    L : ContinuousLinearMap (RingHom.id Real) E Real := LinearMap.toContinuousLine …
    v : E
    hv : Membership.mem s v
    ⊢ HasLineDerivAt Real f (L v) x v
  -/
  have J : L v = lineDeriv ℝ f x v := by convert (hx v hv).symm <;> simp [L, B.sum_repr v]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    hs : s.Countable
    B : Basis (↑(Basis.ofVectorSpaceIndex Real E)) Real E
    I1 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → Eq (lineDeriv …
    I2 : Filter.Eventually (fun x => ∀ (v : E), Membership.mem s v → LineDifferent …
    x : E
    hx : ∀ (v : E), Membership.mem s v → Eq (lineDeriv Real f x (Finset.univ.sum f …
    h'x : ∀ (v : E), Membership.mem s v → LineDifferentiableAt Real f x v
    L : ContinuousLinearMap (RingHom.id Real) E Real := LinearMap.toContinuousLine …
    v : E
    hv : Membership.mem s v
    J : Eq (L v) (lineDeriv Real f x v)
    ⊢ HasLineDerivAt Real f (L v) x v
  -/
  simpa [J] using (h'x v hv).hasLineDerivAt
  /-
    🎉 no goals
  -/


/-- If a Lipschitz functions has line derivatives in a dense set of directions, all of them given by
a single continuous linear map `L`, then it admits `L` as Fréchet derivative. -/
-- We redeclare `E` here as we do not need the `[MeasurableSpace E]` instance
-- available in the rest of the file.
theorem hasFderivAt_of_hasLineDerivAt_of_closure
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [ProperSpace E] {f : E → F}
    (hf : LipschitzWith C f) {s : Set E} (hs : sphere 0 1 ⊆ closure s)
    {L : E →L[ℝ] F} {x : E} (hL : ∀ v ∈ s, HasLineDerivAt ℝ f (L v) x v) :
    HasFDerivAt f L x := by
  /-
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ⊢ HasFDerivAt f L x
  -/
  rw [hasFDerivAt_iff_isLittleO_nhds_zero, isLittleO_iff]
  /-
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm (HS …
  -/
  intro ε εpos
  obtain ⟨δ, δpos, hδ⟩ : ∃ δ, 0 < δ ∧ (C + ‖L‖ + 1) * δ = ε :=
    ⟨ε / (C + ‖L‖ + 1), by positivity, mul_div_cancel₀ ε (by positivity)⟩
  obtain ⟨q, hqs, q_fin, hq⟩ : ∃ q, q ⊆ s ∧ q.Finite ∧ sphere 0 1 ⊆ ⋃ y ∈ q, ball y δ := by
    have : sphere 0 1 ⊆ ⋃ y ∈ s, ball y δ := by
      apply hs.trans (fun z hz ↦ ?_)
      obtain ⟨y, ys, hy⟩ : ∃ y ∈ s, dist z y < δ := Metric.mem_closure_iff.1 hz δ δpos
      exact mem_biUnion ys hy
    exact (isCompact_sphere 0 1).elim_finite_subcover_image (fun y _hy ↦ isOpen_ball) this
  have I : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ v ∈ q, ‖f (x + t • v) - f x - t • L v‖ ≤ δ * ‖t‖ := by
    apply (Finite.eventually_all q_fin).2 (fun v hv ↦ ?_)
    apply Asymptotics.IsLittleO.def ?_ δpos
    exact hasLineDerivAt_iff_isLittleO_nhds_zero.1 (hL v (hqs hv))
  obtain ⟨r, r_pos, hr⟩ : ∃ (r : ℝ), 0 < r ∧ ∀ (t : ℝ), ‖t‖ < r →
      ∀ v ∈ q, ‖f (x + t • v) - f x - t • L v‖ ≤ δ * ‖t‖ := by
    rcases Metric.mem_nhds_iff.1 I with ⟨r, r_pos, hr⟩
    exact ⟨r, r_pos, fun t ht v hv ↦ hr (mem_ball_zero_iff.2 ht) v hv⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
    q : Set E
    hqs : HasSubset.Subset q s
    q_fin : q.Finite
    hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
    I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
    r : Real
    r_pos : LT.lt 0 r
    hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
    ⊢ Filter.Eventually (fun x_1 => LE.le (Norm.norm (HSub.hSub (HSub.hSub (f (HAd …
  -/
  apply Metric.mem_nhds_iff.2 ⟨r, r_pos, fun v hv ↦ ?_⟩
  /-
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
    q : Set E
    hqs : HasSubset.Subset q s
    q_fin : q.Finite
    hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
    I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
    r : Real
    r_pos : LT.lt 0 r
    hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
    v : E
    hv : Membership.mem (Metric.ball 0 r) v
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Norm.norm (HSub.hSub (HS …
  -/
  rcases eq_or_ne v 0 with rfl|v_ne
    /-
      case inl
      F : Type u_2
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      C : NNReal
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : ProperSpace E
      f : E → F
      hf : LipschitzWith C f
      s : Set E
      hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
      L : ContinuousLinearMap (RingHom.id Real) E F
      x : E
      hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
      ε : Real
      εpos : LT.lt 0 ε
      δ : Real
      δpos : LT.lt 0 δ
      hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
      q : Set E
      hqs : HasSubset.Subset q s
      q_fin : q.Finite
      hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
      I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
      r : Real
      r_pos : LT.lt 0 r
      hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
      hv : Membership.mem (Metric.ball 0 r) 0
      ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Norm.norm (HSub.hSub (HS …
    -/
  · simp
    /-
      🎉 no goals
    -/
  obtain ⟨w, ρ, w_mem, hvw, hρ⟩ : ∃ w ρ, w ∈ sphere 0 1 ∧ v = ρ • w ∧ ρ = ‖v‖ := by
    refine ⟨‖v‖⁻¹ • v, ‖v‖, by simp [norm_smul, inv_mul_cancel₀ (norm_ne_zero_iff.2 v_ne)], ?_, rfl⟩
    simp [smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.2 v_ne)]
  /-
    case inr.intro.intro.intro.intro
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
    q : Set E
    hqs : HasSubset.Subset q s
    q_fin : q.Finite
    hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
    I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
    r : Real
    r_pos : LT.lt 0 r
    hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
    v : E
    hv : Membership.mem (Metric.ball 0 r) v
    v_ne : Ne v 0
    w : E
    ρ : Real
    w_mem : Membership.mem (Metric.sphere 0 1) w
    hvw : Eq v (HSMul.hSMul ρ w)
    hρ : Eq ρ (Norm.norm v)
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Norm.norm (HSub.hSub (HS …
  -/
  have norm_rho : ‖ρ‖ = ρ := by rw [hρ, norm_norm]
  /-
    case inr.intro.intro.intro.intro
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
    q : Set E
    hqs : HasSubset.Subset q s
    q_fin : q.Finite
    hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
    I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
    r : Real
    r_pos : LT.lt 0 r
    hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
    v : E
    hv : Membership.mem (Metric.ball 0 r) v
    v_ne : Ne v 0
    w : E
    ρ : Real
    w_mem : Membership.mem (Metric.sphere 0 1) w
    hvw : Eq v (HSMul.hSMul ρ w)
    hρ : Eq ρ (Norm.norm v)
    norm_rho : Eq (Norm.norm ρ) ρ
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Norm.norm (HSub.hSub (HS …
  -/
  have rho_pos : 0 ≤ ρ := by simp [hρ]
  /-
    case inr.intro.intro.intro.intro
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
    q : Set E
    hqs : HasSubset.Subset q s
    q_fin : q.Finite
    hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
    I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
    r : Real
    r_pos : LT.lt 0 r
    hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
    v : E
    hv : Membership.mem (Metric.ball 0 r) v
    v_ne : Ne v 0
    w : E
    ρ : Real
    w_mem : Membership.mem (Metric.sphere 0 1) w
    hvw : Eq v (HSMul.hSMul ρ w)
    hρ : Eq ρ (Norm.norm v)
    norm_rho : Eq (Norm.norm ρ) ρ
    rho_pos : LE.le 0 ρ
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Norm.norm (HSub.hSub (HS …
  -/
  obtain ⟨y, yq, hy⟩ : ∃ y ∈ q, ‖w - y‖ < δ := by simpa [← dist_eq_norm] using hq w_mem
  /-
    case inr.intro.intro.intro.intro.intro.intro
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : ProperSpace E
    f : E → F
    hf : LipschitzWith C f
    s : Set E
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    L : ContinuousLinearMap (RingHom.id Real) E F
    x : E
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑C) (Norm.norm L)) 1) δ) ε
    q : Set E
    hqs : HasSubset.Subset q s
    q_fin : q.Finite
    hq : HasSubset.Subset (Metric.sphere 0 1) (Set.iUnion fun y => Set.iUnion fun  …
    I : Filter.Eventually (fun t => ∀ (v : E), Membership.mem q v → LE.le (Norm.no …
    r : Real
    r_pos : LT.lt 0 r
    hr : ∀ (t : Real), LT.lt (Norm.norm t) r → ∀ (v : E), Membership.mem q v → LE. …
    v : E
    hv : Membership.mem (Metric.ball 0 r) v
    v_ne : Ne v 0
    w : E
    ρ : Real
    w_mem : Membership.mem (Metric.sphere 0 1) w
    hvw : Eq v (HSMul.hSMul ρ w)
    hρ : Eq ρ (Norm.norm v)
    norm_rho : Eq (Norm.norm ρ) ρ
    rho_pos : LE.le 0 ρ
    y : E
    yq : Membership.mem q y
    hy : LT.lt (Norm.norm (HSub.hSub w y)) δ
    ⊢ Membership.mem (setOf fun x_1 => (fun x_2 => LE.le (Norm.norm (HSub.hSub (HS …
  -/
  have : ‖y - w‖ < δ := by rwa [norm_sub_rev]
  calc  ‖f (x + v) - f x - L v‖
      = ‖f (x + ρ • w) - f x - ρ • L w‖ := by simp [hvw]
    _ = ‖(f (x + ρ • w) - f (x + ρ • y)) + (ρ • L y - ρ • L w)
          + (f (x + ρ • y) - f x - ρ • L y)‖ := by congr; abel
    _ ≤ ‖f (x + ρ • w) - f (x + ρ • y)‖ + ‖ρ • L y - ρ • L w‖
          + ‖f (x + ρ • y) - f x - ρ • L y‖ := norm_add₃_le
    _ ≤ C * ‖(x + ρ • w) - (x + ρ • y)‖ + ρ * (‖L‖ * ‖y - w‖) + δ * ρ := by
      gcongr
      · exact hf.norm_sub_le _ _
      · rw [← smul_sub, norm_smul, norm_rho]
        gcongr
        exact L.lipschitz.norm_sub_le _ _
      · conv_rhs => rw [← norm_rho]
        apply hr _ _ _ yq
        simpa [norm_rho, hρ] using hv
    _ ≤ C * (ρ * δ) + ρ * (‖L‖ * δ) + δ * ρ := by
      simp only [add_sub_add_left_eq_sub, ← smul_sub, norm_smul, norm_rho]; gcongr
    _ = ((C + ‖L‖ + 1) * δ) * ρ := by ring
    _ = ε * ‖v‖ := by rw [hδ, hρ]


/-- A real-valued function on a finite-dimensional space which is Lipschitz is
differentiable almost everywere. Superseded by
`LipschitzWith.ae_differentiableAt` which works for functions taking value in any
finite-dimensional space. -/
theorem ae_differentiableAt_of_real (hf : LipschitzWith C f) :
    ∀ᵐ x ∂μ, DifferentiableAt ℝ f x := by
  obtain ⟨s, s_count, s_dense⟩ : ∃ (s : Set E), s.Countable ∧ Dense s :=
    TopologicalSpace.exists_countable_dense E
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    s_count : s.Countable
    s_dense : Dense s
    ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (MeasureTheory.ae μ)
  -/
  have hs : sphere 0 1 ⊆ closure s := by rw [s_dense.closure_eq]; exact subset_univ _
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    s_count : s.Countable
    s_dense : Dense s
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (MeasureTheory.ae μ)
  -/
  filter_upwards [hf.ae_exists_fderiv_of_countable s_count]
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    s_count : s.Countable
    s_dense : Dense s
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    ⊢ ∀ (a : E), (Exists fun L => ∀ (v : E), Membership.mem s v → HasLineDerivAt R …
  -/
  rintro x ⟨L, hL⟩
  /-
    case h.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzWith C f
    s : Set E
    s_count : s.Countable
    s_dense : Dense s
    hs : HasSubset.Subset (Metric.sphere 0 1) (closure s)
    x : E
    L : ContinuousLinearMap (RingHom.id Real) E Real
    hL : ∀ (v : E), Membership.mem s v → HasLineDerivAt Real f (L v) x v
    ⊢ DifferentiableAt Real f x
  -/
  exact (hf.hasFderivAt_of_hasLineDerivAt_of_closure hs hL).differentiableAt
  /-
    🎉 no goals
  -/


/-- A real-valued function on a finite-dimensional space which is Lipschitz on a set is
differentiable almost everywere in this set. Superseded by
`LipschitzOnWith.ae_differentiableWithinAt_of_mem` which works for functions taking value in any
finite-dimensional space. -/
theorem ae_differentiableWithinAt_of_mem_of_real (hf : LipschitzOnWith C f s) :
    ∀ᵐ x ∂μ, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzOnWith C f s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  obtain ⟨g, g_lip, hg⟩ : ∃ (g : E → ℝ), LipschitzWith C g ∧ EqOn f g s := hf.extend_real
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzOnWith C f s
    g : E → Real
    g_lip : LipschitzWith C g
    hg : Set.EqOn f g s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  filter_upwards [g_lip.ae_differentiableAt_of_real] with x hx xs
  /-
    case h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    C : NNReal
    f : E → Real
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    hf : LipschitzOnWith C f s
    g : E → Real
    g_lip : LipschitzWith C g
    hg : Set.EqOn f g s
    x : E
    hx : DifferentiableAt Real g x
    xs : Membership.mem s x
    ⊢ DifferentiableWithinAt Real f s x
  -/
  exact hx.differentiableWithinAt.congr hg (hg xs)
  /-
    🎉 no goals
  -/


/-- A function on a finite-dimensional space which is Lipschitz on a set and taking values in a
product space is differentiable almost everywere in this set. Superseded by
`LipschitzOnWith.ae_differentiableWithinAt_of_mem` which works for functions taking value in any
finite-dimensional space. -/
theorem ae_differentiableWithinAt_of_mem_pi
    {ι : Type*} [Fintype ι] {f : E → ι → ℝ} {s : Set E}
    (hf : LipschitzOnWith C f s) : ∀ᵐ x ∂μ, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    C : NNReal
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    ι : Type u_3
    inst✝ : Fintype ι
    f : E → ι → Real
    s : Set E
    hf : LipschitzOnWith C f s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  have A : ∀ i : ι, LipschitzWith 1 (fun x : ι → ℝ ↦ x i) := fun i => LipschitzWith.eval i
  have : ∀ i : ι, ∀ᵐ x ∂μ, x ∈ s → DifferentiableWithinAt ℝ (fun x : E ↦ f x i) s x := fun i ↦ by
    apply ae_differentiableWithinAt_of_mem_of_real
    exact LipschitzWith.comp_lipschitzOnWith (A i) hf
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    C : NNReal
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    ι : Type u_3
    inst✝ : Fintype ι
    f : E → ι → Real
    s : Set E
    hf : LipschitzOnWith C f s
    A : ∀ (i : ι), LipschitzWith 1 fun x => x i
    this : ∀ (i : ι), Filter.Eventually (fun x => Membership.mem s x → Differentia …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  filter_upwards [ae_all_iff.2 this] with x hx xs
  /-
    case h
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    C : NNReal
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : μ.IsAddHaarMeasure
    ι : Type u_3
    inst✝ : Fintype ι
    f : E → ι → Real
    s : Set E
    hf : LipschitzOnWith C f s
    A : ∀ (i : ι), LipschitzWith 1 fun x => x i
    this : ∀ (i : ι), Filter.Eventually (fun x => Membership.mem s x → Differentia …
    x : E
    hx : ∀ (i : ι), Membership.mem s x → DifferentiableWithinAt Real (fun x => f x …
    xs : Membership.mem s x
    ⊢ DifferentiableWithinAt Real f s x
  -/
  exact differentiableWithinAt_pi.2 (fun i ↦ hx i xs)
  /-
    🎉 no goals
  -/


/-- *Rademacher's theorem*: a function between finite-dimensional real vector spaces which is
Lipschitz on a set is differentiable almost everywere in this set. -/
theorem ae_differentiableWithinAt_of_mem {f : E → F} (hf : LipschitzOnWith C f s) :
    ∀ᵐ x ∂μ, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    hf : LipschitzOnWith C f s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  have A := (Basis.ofVectorSpace ℝ F).equivFun.toContinuousLinearEquiv
  suffices H : ∀ᵐ x ∂μ, x ∈ s → DifferentiableWithinAt ℝ (A ∘ f) s x by
    filter_upwards [H] with x hx xs
    have : f = (A.symm ∘ A) ∘ f := by
      simp only [ContinuousLinearEquiv.symm_comp_self, Function.id_comp]
    rw [this]
    exact A.symm.differentiableAt.comp_differentiableWithinAt x (hx xs)
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    hf : LipschitzOnWith C f s
    A : ContinuousLinearEquiv (RingHom.id Real) F (↑(Basis.ofVectorSpaceIndex Real …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  apply ae_differentiableWithinAt_of_mem_pi
  /-
    case hf
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    hf : LipschitzOnWith C f s
    A : ContinuousLinearEquiv (RingHom.id Real) F (↑(Basis.ofVectorSpaceIndex Real …
    ⊢ LipschitzOnWith ?C (Function.comp (⇑A) f) s
  -/
  exact A.lipschitz.comp_lipschitzOnWith hf
  /-
    🎉 no goals
  -/


/-- *Rademacher's theorem*: a function between finite-dimensional real vector spaces which is
Lipschitz on a set is differentiable almost everywere in this set. -/
theorem ae_differentiableWithinAt {f : E → F} (hf : LipschitzOnWith C f s)
    (hs : MeasurableSet s) :
    ∀ᵐ x ∂(μ.restrict s), DifferentiableWithinAt ℝ f s x := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    hf : LipschitzOnWith C f s
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => DifferentiableWithinAt Real f s x) (MeasureTheor …
  -/
  rw [ae_restrict_iff' hs]
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    s : Set E
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    hf : LipschitzOnWith C f s
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  exact hf.ae_differentiableWithinAt_of_mem
  /-
    🎉 no goals
  -/


/-- *Rademacher's theorem*: a Lipschitz function between finite-dimensional real vector spaces is
differentiable almost everywhere. -/
theorem LipschitzWith.ae_differentiableAt {f : E → F} (h : LipschitzWith C f) :
    ∀ᵐ x ∂μ, DifferentiableAt ℝ f x := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    h : LipschitzWith C f
    ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (MeasureTheory.ae μ)
  -/
  rw [← lipschitzOnWith_univ] at h
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    F : Type u_2
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    C : NNReal
    μ : MeasureTheory.Measure E
    inst✝² : FiniteDimensional Real E
    inst✝¹ : FiniteDimensional Real F
    inst✝ : μ.IsAddHaarMeasure
    f : E → F
    h : LipschitzOnWith C f Set.univ
    ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (MeasureTheory.ae μ)
  -/
  simpa [differentiableWithinAt_univ] using h.ae_differentiableWithinAt_of_mem
  /-
    🎉 no goals
  -/


/-- In a real finite-dimensional normed vector space,
  the norm is almost everywhere differentiable. -/
theorem ae_differentiableAt_norm :
    ∀ᵐ x ∂μ, DifferentiableAt ℝ (‖·‖) x := lipschitzWith_one_norm.ae_differentiableAt


omit [MeasurableSpace E] in
/-- In a real finite-dimensional normed vector space,
  the set of points where the norm is differentiable at is dense. -/
theorem dense_differentiableAt_norm :
    Dense {x : E | DifferentiableAt ℝ (‖·‖) x} :=
  let _ : MeasurableSpace E := borel E
  have _ : BorelSpace E := ⟨rfl⟩
  let w := Basis.ofVectorSpace ℝ E
  MeasureTheory.Measure.dense_of_ae (ae_differentiableAt_norm (μ := w.addHaar))

