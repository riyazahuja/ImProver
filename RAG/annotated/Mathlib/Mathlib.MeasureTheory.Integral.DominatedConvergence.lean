/-- **Lebesgue dominated convergence theorem** provides sufficient conditions under which almost
  everywhere convergence of a sequence of functions implies the convergence of their integrals.
  We could weaken the condition `bound_integrable` to require `HasFiniteIntegral bound μ` instead
  (i.e. not requiring that `bound` is measurable), but in all applications proving integrability
  is easier. -/
theorem tendsto_integral_of_dominated_convergence {F : ℕ → α → G} {f : α → G} (bound : α → ℝ)
    (F_measurable : ∀ n, AEStronglyMeasurable (F n) μ) (bound_integrable : Integrable bound μ)
    (h_bound : ∀ n, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound a)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop (𝓝 (f a))) :
    Tendsto (fun n => ∫ a, F n a ∂μ) atTop (𝓝 <| ∫ a, f a ∂μ) := by
  /-
    α : Type u_1
    G : Type u_3
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    F : Nat → α → G
    f : α → G
    bound : α → Real
    F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
    bound_integrable : MeasureTheory.Integrable bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun a => F n a) Filter.atT …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_3
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → G
      f : α → G
      bound : α → Real
      F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      hG : CompleteSpace G
      ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun a => F n a) Filter.atT …
    -/
  · simp only [integral, hG, L1.integral]
    exact tendsto_setToFun_of_dominated_convergence (dominatedFinMeasAdditive_weightedSMul μ)
      bound F_measurable bound_integrable h_bound h_lim
    /-
      case neg
      α : Type u_1
      G : Type u_3
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → G
      f : α → G
      bound : α → Real
      F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
      bound_integrable : MeasureTheory.Integrable bound μ
      h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      hG : Not (CompleteSpace G)
      ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun a => F n a) Filter.atT …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/


/-- Lebesgue dominated convergence theorem for filters with a countable basis -/
theorem tendsto_integral_filter_of_dominated_convergence {ι} {l : Filter ι} [l.IsCountablyGenerated]
    {F : ι → α → G} {f : α → G} (bound : α → ℝ) (hF_meas : ∀ᶠ n in l, AEStronglyMeasurable (F n) μ)
    (h_bound : ∀ᶠ n in l, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound a) (bound_integrable : Integrable bound μ)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) l (𝓝 (f a))) :
    Tendsto (fun n => ∫ a, F n a ∂μ) l (𝓝 <| ∫ a, f a ∂μ) := by
  /-
    α : Type u_1
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → G
    f : α → G
    bound : α → Real
    hF_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (F n) …
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
    bound_integrable : MeasureTheory.Integrable bound μ
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun a => F n a) l (nhds (M …
  -/
  by_cases hG : CompleteSpace G
    /-
      case pos
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → G
      f : α → G
      bound : α → Real
      hF_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (F n) …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      hG : CompleteSpace G
      ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun a => F n a) l (nhds (M …
    -/
  · simp only [integral, hG, L1.integral]
    exact tendsto_setToFun_filter_of_dominated_convergence (dominatedFinMeasAdditive_weightedSMul μ)
      bound hF_meas h_bound bound_integrable h_lim
    /-
      case neg
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → G
      f : α → G
      bound : α → Real
      hF_meas : Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (F n) …
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm. …
      bound_integrable : MeasureTheory.Integrable bound μ
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      hG : Not (CompleteSpace G)
      ⊢ Filter.Tendsto (fun n => MeasureTheory.integral μ fun a => F n a) l (nhds (M …
    -/
  · simp [integral, hG, tendsto_const_nhds]
    /-
      🎉 no goals
    -/


/-- Lebesgue dominated convergence theorem for series. -/
theorem hasSum_integral_of_dominated_convergence {ι} [Countable ι] {F : ι → α → G} {f : α → G}
    (bound : ι → α → ℝ) (hF_meas : ∀ n, AEStronglyMeasurable (F n) μ)
    (h_bound : ∀ n, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound n a)
    (bound_summable : ∀ᵐ a ∂μ, Summable fun n => bound n a)
    (bound_integrable : Integrable (fun a => ∑' n, bound n a) μ)
    (h_lim : ∀ᵐ a ∂μ, HasSum (fun n => F n a) (f a)) :
    HasSum (fun n => ∫ a, F n a ∂μ) (∫ a, f a ∂μ) := by
  have hb_nonneg : ∀ᵐ a ∂μ, ∀ n, 0 ≤ bound n a :=
    eventually_countable_forall.2 fun n => (h_bound n).mono fun a => (norm_nonneg _).trans
  have hb_le_tsum : ∀ n, bound n ≤ᵐ[μ] fun a => ∑' n, bound n a := by
    intro n
    filter_upwards [hb_nonneg, bound_summable]
      with _ ha0 ha_sum using le_tsum ha_sum _ fun i _ => ha0 i
  have hF_integrable : ∀ n, Integrable (F n) μ := by
    refine fun n => bound_integrable.mono' (hF_meas n) ?_
    exact EventuallyLE.trans (h_bound n) (hb_le_tsum n)
  /-
    α : Type u_1
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    inst✝ : Countable ι
    F : ι → α → G
    f : α → G
    bound : ι → α → Real
    hF_meas : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (F n) μ
    h_bound : ∀ (n : ι), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) (bo …
    bound_summable : Filter.Eventually (fun a => Summable fun n => bound n a) (Mea …
    bound_integrable : MeasureTheory.Integrable (fun a => tsum fun n => bound n a) μ
    h_lim : Filter.Eventually (fun a => HasSum (fun n => F n a) (f a)) (MeasureThe …
    hb_nonneg : Filter.Eventually (fun a => ∀ (n : ι), LE.le 0 (bound n a)) (Measu …
    hb_le_tsum : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyLE (bound n) fun a => t …
    hF_integrable : ∀ (n : ι), MeasureTheory.Integrable (F n) μ
    ⊢ HasSum (fun n => MeasureTheory.integral μ fun a => F n a) (MeasureTheory.int …
  -/
  simp only [HasSum, ← integral_finset_sum _ fun n _ => hF_integrable n]
  refine tendsto_integral_filter_of_dominated_convergence
      (fun a => ∑' n, bound n a) ?_ ?_ bound_integrable h_lim
    /-
      case refine_1
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → G
      f : α → G
      bound : ι → α → Real
      hF_meas : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (F n) μ
      h_bound : ∀ (n : ι), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) (bo …
      bound_summable : Filter.Eventually (fun a => Summable fun n => bound n a) (Mea …
      bound_integrable : MeasureTheory.Integrable (fun a => tsum fun n => bound n a) μ
      h_lim : Filter.Eventually (fun a => HasSum (fun n => F n a) (f a)) (MeasureThe …
      hb_nonneg : Filter.Eventually (fun a => ∀ (n : ι), LE.le 0 (bound n a)) (Measu …
      hb_le_tsum : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyLE (bound n) fun a => t …
      hF_integrable : ∀ (n : ι), MeasureTheory.Integrable (F n) μ
      ⊢ Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (fun a => n.s …
    -/
  · exact Eventually.of_forall fun s => s.aestronglyMeasurable_sum fun n _ => hF_meas n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → G
      f : α → G
      bound : ι → α → Real
      hF_meas : ∀ (n : ι), MeasureTheory.AEStronglyMeasurable (F n) μ
      h_bound : ∀ (n : ι), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) (bo …
      bound_summable : Filter.Eventually (fun a => Summable fun n => bound n a) (Mea …
      bound_integrable : MeasureTheory.Integrable (fun a => tsum fun n => bound n a) μ
      h_lim : Filter.Eventually (fun a => HasSum (fun n => F n a) (f a)) (MeasureThe …
      hb_nonneg : Filter.Eventually (fun a => ∀ (n : ι), LE.le 0 (bound n a)) (Measu …
      hb_le_tsum : ∀ (n : ι), (MeasureTheory.ae μ).EventuallyLE (bound n) fun a => t …
      hF_integrable : ∀ (n : ι), MeasureTheory.Integrable (F n) μ
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm (n. …
    -/
  · filter_upwards with s
    filter_upwards [eventually_countable_forall.2 h_bound, hb_nonneg, bound_summable]
      with a hFa ha0 has
    calc
      ‖∑ n ∈ s, F n a‖ ≤ ∑ n ∈ s, bound n a := norm_sum_le_of_le _ fun n _ => hFa n
      _ ≤ ∑' n, bound n a := sum_le_tsum _ (fun n _ => ha0 n) has


theorem integral_tsum {ι} [Countable ι] {f : ι → α → G} (hf : ∀ i, AEStronglyMeasurable (f i) μ)
    (hf' : ∑' i, ∫⁻ a : α, ‖f i a‖₊ ∂μ ≠ ∞) :
    ∫ a : α, ∑' i, f i a ∂μ = ∑' i, ∫ a : α, f i a ∂μ := by
  /-
    α : Type u_1
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    inst✝ : Countable ι
    f : ι → α → G
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
    ⊢ Eq (MeasureTheory.integral μ fun a => tsum fun i => f i a) (tsum fun i => Me …
  -/
  by_cases hG : CompleteSpace G; swap
    /-
      case neg
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral μ fun a => tsum fun i => f i a) (tsum fun i => Me …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    G : Type u_3
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    inst✝ : Countable ι
    f : ι → α → G
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
    hG : CompleteSpace G
    ⊢ Eq (MeasureTheory.integral μ fun a => tsum fun i => f i a) (tsum fun i => Me …
  -/
  have hf'' : ∀ i, AEMeasurable (fun x => (‖f i x‖₊ : ℝ≥0∞)) μ := fun i => (hf i).ennnorm
  have hhh : ∀ᵐ a : α ∂μ, Summable fun n => (‖f n a‖₊ : ℝ) := by
    rw [← lintegral_tsum hf''] at hf'
    refine (ae_lt_top' (AEMeasurable.ennreal_tsum hf'') hf').mono ?_
    intro x hx
    rw [← ENNReal.tsum_coe_ne_top_iff_summable_coe]
    exact hx.ne
  convert (MeasureTheory.hasSum_integral_of_dominated_convergence (fun i a => ‖f i a‖₊) hf _ hhh
          ⟨_, _⟩ _).tsum_eq.symm
    /-
      case pos.convert_2
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ ∀ (n : ι), Filter.Eventually (fun a => LE.le (Norm.norm (f n a)) ((fun i a = …
    -/
  · intro n
    /-
      case pos.convert_2
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      n : ι
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f n a)) ((fun i a => ↑(NNNorm. …
    -/
    filter_upwards with x
    /-
      case pos.convert_2.h
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      n : ι
      x : α
      ⊢ LE.le (Norm.norm (f n x)) ↑(NNNorm.nnnorm (f n x))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case pos.convert_3
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => tsum fun n => (fun i a => ↑(NNN …
    -/
  · simp_rw [← NNReal.coe_tsum]
    /-
      case pos.convert_3
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => ↑(tsum fun a_1 => NNNorm.nnnorm …
    -/
    rw [aestronglyMeasurable_iff_aemeasurable]
    /-
      case pos.convert_3
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ AEMeasurable (fun a => ↑(tsum fun a_1 => NNNorm.nnnorm (f a_1 a))) μ
    -/
    apply AEMeasurable.coe_nnreal_real
    /-
      case pos.convert_3.hf
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ AEMeasurable (fun x => tsum fun a => NNNorm.nnnorm (f a x)) μ
    -/
    apply AEMeasurable.nnreal_tsum
    /-
      case pos.convert_3.hf.h
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ ∀ (i : ι), AEMeasurable (fun x => NNNorm.nnnorm (f i x)) μ
    -/
    exact fun i => (hf i).nnnorm.aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case pos.convert_4
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ MeasureTheory.HasFiniteIntegral (fun a => tsum fun n => (fun i a => ↑(NNNorm …
    -/
  · dsimp [HasFiniteIntegral]
    /-
      case pos.convert_4
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ENorm.enorm (tsum fun n => Norm.no …
    -/
    have : ∫⁻ a, ∑' n, ‖f n a‖₊ ∂μ < ⊤ := by rwa [lintegral_tsum hf'', lt_top_iff_ne_top]
    /-
      case pos.convert_4
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      this : LT.lt (MeasureTheory.lintegral μ fun a => tsum fun n => ↑(NNNorm.nnnorm …
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ENorm.enorm (tsum fun n => Norm.no …
    -/
    convert this using 1
    /-
      case h.e'_3
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      this : LT.lt (MeasureTheory.lintegral μ fun a => tsum fun n => ↑(NNNorm.nnnorm …
      ⊢ Eq (MeasureTheory.lintegral μ fun a => ENorm.enorm (tsum fun n => Norm.norm  …
    -/
    apply lintegral_congr_ae
    /-
      case h.e'_3.h
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      this : LT.lt (MeasureTheory.lintegral μ fun a => tsum fun n => ↑(NNNorm.nnnorm …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ENorm.enorm (tsum fun n => Norm. …
    -/
    simp_rw [← coe_nnnorm, ← NNReal.coe_tsum, enorm_eq_nnnorm, NNReal.nnnorm_eq]
    /-
      case h.e'_3.h
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      this : LT.lt (MeasureTheory.lintegral μ fun a => tsum fun n => ↑(NNNorm.nnnorm …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ↑(tsum fun a_1 => NNNorm.nnnorm  …
    -/
    filter_upwards [hhh] with a ha
    /-
      case h
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      this : LT.lt (MeasureTheory.lintegral μ fun a => tsum fun n => ↑(NNNorm.nnnorm …
      a : α
      ha : Summable fun n => ↑(NNNorm.nnnorm (f n a))
      ⊢ Eq (↑(tsum fun a_1 => NNNorm.nnnorm (f a_1 a))) (tsum fun n => ↑(NNNorm.nnno …
    -/
    exact ENNReal.coe_tsum (NNReal.summable_coe.mp ha)
    /-
      🎉 no goals
    -/
    /-
      case pos.convert_5
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      ⊢ Filter.Eventually (fun a => HasSum (fun n => f n a) (tsum fun i => f i a)) ( …
    -/
  · filter_upwards [hhh] with x hx
    /-
      case h
      α : Type u_1
      G : Type u_3
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace Real G
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      f : ι → α → G
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      hf' : Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f  …
      hG : CompleteSpace G
      hf'' : ∀ (i : ι), AEMeasurable (fun x => ↑(NNNorm.nnnorm (f i x))) μ
      hhh : Filter.Eventually (fun a => Summable fun n => ↑(NNNorm.nnnorm (f n a)))  …
      x : α
      hx : Summable fun n => ↑(NNNorm.nnnorm (f n x))
      ⊢ HasSum (fun n => f n x) (tsum fun i => f i x)
    -/
    exact hx.of_norm.hasSum
    /-
      🎉 no goals
    -/


lemma hasSum_integral_of_summable_integral_norm {ι} [Countable ι] {F : ι → α → E}
    (hF_int : ∀ i : ι, Integrable (F i) μ) (hF_sum : Summable fun i ↦ ∫ a, ‖F i a‖ ∂μ) :
    HasSum (∫ a, F · a ∂μ) (∫ a, (∑' i, F i a) ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    inst✝ : Countable ι
    F : ι → α → E
    hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
    hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
    ⊢ HasSum (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheory.int …
  -/
  by_cases hE : CompleteSpace E; swap
    /-
      case neg
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → E
      hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
      hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
      hE : Not (CompleteSpace E)
      ⊢ HasSum (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheory.int …
    -/
  · simp [integral, hE, hasSum_zero]
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    inst✝ : Countable ι
    F : ι → α → E
    hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
    hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
    hE : CompleteSpace E
    ⊢ HasSum (fun x => MeasureTheory.integral μ fun a => F x a) (MeasureTheory.int …
  -/
  rw [integral_tsum (fun i ↦ (hF_int i).1)]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → E
      hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
      hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
      hE : CompleteSpace E
      ⊢ HasSum (fun x => MeasureTheory.integral μ fun a => F x a) (tsum fun i => Mea …
    -/
  · exact (hF_sum.of_norm_bounded _ fun i ↦ norm_integral_le_integral_norm _).hasSum
    /-
      🎉 no goals
    -/
  have (i : ι) : ∫⁻ (a : α), ‖F i a‖₊ ∂μ = ‖(∫ a : α, ‖F i a‖ ∂μ)‖₊ := by
    rw [lintegral_coe_eq_integral _ (hF_int i).norm, coe_nnreal_eq, coe_nnnorm,
      Real.norm_of_nonneg (integral_nonneg (fun a ↦ norm_nonneg (F i a)))]
    simp only [coe_nnnorm]
  /-
    case pos
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_4
    inst✝ : Countable ι
    F : ι → α → E
    hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
    hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
    hE : CompleteSpace E
    this : ∀ (i : ι), Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (F i  …
    ⊢ Ne (tsum fun i => MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (F i a) …
  -/
  rw [funext this, ← ENNReal.coe_tsum]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → E
      hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
      hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
      hE : CompleteSpace E
      this : ∀ (i : ι), Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (F i  …
      ⊢ Ne (↑(tsum fun x => NNNorm.nnnorm (MeasureTheory.integral μ fun a => Norm.no …
    -/
  · apply coe_ne_top
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → E
      hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
      hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
      hE : CompleteSpace E
      this : ∀ (i : ι), Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (F i  …
      ⊢ Summable fun x => NNNorm.nnnorm (MeasureTheory.integral μ fun a => Norm.norm …
    -/
  · simp_rw [← NNReal.summable_coe, coe_nnnorm]
    /-
      case pos
      α : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_4
      inst✝ : Countable ι
      F : ι → α → E
      hF_int : ∀ (i : ι), MeasureTheory.Integrable (F i) μ
      hF_sum : Summable fun i => MeasureTheory.integral μ fun a => Norm.norm (F i a)
      hE : CompleteSpace E
      this : ∀ (i : ι), Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (F i  …
      ⊢ Summable fun a => Norm.norm (MeasureTheory.integral μ fun a_1 => Norm.norm ( …
    -/
    exact hF_sum.abs
    /-
      🎉 no goals
    -/


lemma integral_tsum_of_summable_integral_norm {ι} [Countable ι] {F : ι → α → E}
    (hF_int : ∀ i : ι, Integrable (F i) μ) (hF_sum : Summable fun i ↦ ∫ a, ‖F i a‖ ∂μ) :
    ∑' i, (∫ a, F i a ∂μ) = ∫ a, (∑' i, F i a) ∂μ :=
  (hasSum_integral_of_summable_integral_norm hF_int hF_sum).tsum_eq


theorem _root_.Antitone.tendsto_setIntegral (hsm : ∀ i, MeasurableSet (s i)) (h_anti : Antitone s)
    (hfi : IntegrableOn f (s 0) μ) :
    Tendsto (fun i => ∫ a in s i, f a ∂μ) atTop (𝓝 (∫ a in ⋂ n, s n, f a ∂μ)) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Nat → Set α
    f : α → E
    hsm : ∀ (i : Nat), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : MeasureTheory.IntegrableOn f (s 0) μ
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a =>  …
  -/
  let bound : α → ℝ := indicator (s 0) fun a => ‖f a‖
  have h_int_eq : (fun i => ∫ a in s i, f a ∂μ) = fun i => ∫ a, (s i).indicator f a ∂μ :=
    funext fun i => (integral_indicator (hsm i)).symm
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Nat → Set α
    f : α → E
    hsm : ∀ (i : Nat), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : MeasureTheory.IntegrableOn f (s 0) μ
    bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
    h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a =>  …
  -/
  rw [h_int_eq]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Nat → Set α
    f : α → E
    hsm : ∀ (i : Nat), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : MeasureTheory.IntegrableOn f (s 0) μ
    bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
    h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => (s i).indicator f …
  -/
  rw [← integral_indicator (MeasurableSet.iInter hsm)]
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Nat → Set α
    f : α → E
    hsm : ∀ (i : Nat), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : MeasureTheory.IntegrableOn f (s 0) μ
    bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
    h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun a => (s i).indicator f …
  -/
  refine tendsto_integral_of_dominated_convergence bound ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      ⊢ ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable ((s n).indicator f) μ
    -/
  · intro n
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable ((s n).indicator f) μ
    -/
    rw [aestronglyMeasurable_indicator_iff (hsm n)]
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      n : Nat
      ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict (s n))
    -/
    exact (IntegrableOn.mono_set hfi (h_anti (zero_le n))).1
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      ⊢ MeasureTheory.Integrable bound μ
    -/
  · rw [integrable_indicator_iff (hsm 0)]
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      ⊢ MeasureTheory.IntegrableOn (fun a => Norm.norm (f a)) (s 0) μ
    -/
    exact hfi.norm
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      ⊢ ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm ((s n).indicator f …
    -/
  · simp_rw [norm_indicator_eq_indicator_norm]
    /-
      case refine_3
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      ⊢ ∀ (n : Nat), Filter.Eventually (fun a => LE.le ((s n).indicator (fun a => No …
    -/
    refine fun n => Eventually.of_forall fun x => ?_
    /-
      case refine_3
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      n : Nat
      x : α
      ⊢ LE.le ((s n).indicator (fun a => Norm.norm (f a)) x) (bound x)
    -/
    exact indicator_le_indicator_of_subset (h_anti (zero_le n)) (fun a => norm_nonneg _) _
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      E : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Nat → Set α
      f : α → E
      hsm : ∀ (i : Nat), MeasurableSet (s i)
      h_anti : Antitone s
      hfi : MeasureTheory.IntegrableOn f (s 0) μ
      bound : α → Real := (s 0).indicator fun a => Norm.norm (f a)
      h_int_eq : Eq (fun i => MeasureTheory.integral (μ.restrict (s i)) fun a => f a …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => (s n).indicator f a) Fi …
    -/
  · filter_upwards [] with a using le_trans (h_anti.tendsto_indicator _ _ _) (pure_le_nhds _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias _root_.Antitone.tendsto_set_integral :=  _root_.Antitone.tendsto_setIntegral


/-- Lebesgue dominated convergence theorem for filters with a countable basis -/
nonrec theorem tendsto_integral_filter_of_dominated_convergence {ι} {l : Filter ι}
    [l.IsCountablyGenerated] {F : ι → ℝ → E} (bound : ℝ → ℝ)
    (hF_meas : ∀ᶠ n in l, AEStronglyMeasurable (F n) (μ.restrict (Ι a b)))
    (h_bound : ∀ᶠ n in l, ∀ᵐ x ∂μ, x ∈ Ι a b → ‖F n x‖ ≤ bound x)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_lim : ∀ᵐ x ∂μ, x ∈ Ι a b → Tendsto (fun n => F n x) l (𝓝 (f x))) :
    Tendsto (fun n => ∫ x in a..b, F n x ∂μ) l (𝓝 <| ∫ x in a..b, f x ∂μ) := by
  simp only [intervalIntegrable_iff, intervalIntegral_eq_integral_uIoc,
    ← ae_restrict_iff' (α := ℝ) (μ := μ) measurableSet_uIoc] at *
  exact tendsto_const_nhds.smul <|
    tendsto_integral_filter_of_dominated_convergence bound hF_meas h_bound bound_integrable h_lim


/-- Lebesgue dominated convergence theorem for parametric interval integrals. -/
nonrec theorem hasSum_integral_of_dominated_convergence {ι} [Countable ι] {F : ι → ℝ → E}
    (bound : ι → ℝ → ℝ) (hF_meas : ∀ n, AEStronglyMeasurable (F n) (μ.restrict (Ι a b)))
    (h_bound : ∀ n, ∀ᵐ t ∂μ, t ∈ Ι a b → ‖F n t‖ ≤ bound n t)
    (bound_summable : ∀ᵐ t ∂μ, t ∈ Ι a b → Summable fun n => bound n t)
    (bound_integrable : IntervalIntegrable (fun t => ∑' n, bound n t) μ a b)
    (h_lim : ∀ᵐ t ∂μ, t ∈ Ι a b → HasSum (fun n => F n t) (f t)) :
    HasSum (fun n => ∫ t in a..b, F n t ∂μ) (∫ t in a..b, f t ∂μ) := by
  simp only [intervalIntegrable_iff, intervalIntegral_eq_integral_uIoc, ←
    ae_restrict_iff' (α := ℝ) (μ := μ) measurableSet_uIoc] at *
  exact
    (hasSum_integral_of_dominated_convergence bound hF_meas h_bound bound_summable bound_integrable
          h_lim).const_smul
      _


/-- Interval integrals commute with countable sums, when the supremum norms are summable (a
special case of the dominated convergence theorem). -/
theorem hasSum_intervalIntegral_of_summable_norm [Countable ι] {f : ι → C(ℝ, E)}
    (hf_sum : Summable fun i : ι => ‖(f i).restrict (⟨uIcc a b, isCompact_uIcc⟩ : Compacts ℝ)‖) :
    HasSum (fun i : ι => ∫ x in a..b, f i x) (∫ x in a..b, ∑' i : ι, f i x) := by
  /-
    ι : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    inst✝ : Countable ι
    f : ι → ContinuousMap Real E
    hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
    ⊢ HasSum (fun i => intervalIntegral (fun x => (f i) x) a b MeasureTheory.Measu …
  -/
  by_cases hE : CompleteSpace E; swap
    /-
      case neg
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : Not (CompleteSpace E)
      ⊢ HasSum (fun i => intervalIntegral (fun x => (f i) x) a b MeasureTheory.Measu …
    -/
  · simp [intervalIntegral, integral, hE, hasSum_zero]
    /-
      🎉 no goals
    -/
  apply hasSum_integral_of_dominated_convergence
    (fun i (x : ℝ) => ‖(f i).restrict ↑(⟨uIcc a b, isCompact_uIcc⟩ : Compacts ℝ)‖)
    (fun i => (map_continuous <| f i).aestronglyMeasurable)
    /-
      case pos.h_bound
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      ⊢ ∀ (n : ι), Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → LE. …
    -/
  · intro i; filter_upwards with x hx
    /-
      case pos.h_bound.h
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      i : ι
      x : Real
      hx : Membership.mem (Set.uIoc a b) x
      ⊢ LE.le (Norm.norm ((f i) x)) (Norm.norm (ContinuousMap.restrict (↑{ carrier : …
    -/
    apply ContinuousMap.norm_coe_le_norm ((f i).restrict _) ⟨x, _⟩
    /-
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      i : ι
      x : Real
      hx : Membership.mem (Set.uIoc a b) x
      ⊢ Membership.mem (↑{ carrier := Set.uIcc a b, isCompact' := ⋯ }) x
    -/
    exact ⟨hx.1.le, hx.2⟩
    /-
      🎉 no goals
    -/
    /-
      case pos.bound_summable
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      ⊢ Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → Summable fun n …
    -/
  · exact ae_of_all _ fun x _ => hf_sum
    /-
      🎉 no goals
    -/
    /-
      case pos.bound_integrable
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      ⊢ IntervalIntegrable (fun t => tsum fun n => Norm.norm (ContinuousMap.restrict …
    -/
  · exact intervalIntegrable_const
    /-
      🎉 no goals
    -/
    /-
      case pos.h_lim
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      ⊢ Filter.Eventually (fun t => Membership.mem (Set.uIoc a b) t → HasSum (fun n  …
    -/
  · refine ae_of_all _ fun x hx => Summable.hasSum ?_
    /-
      case pos.h_lim
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      x : Real
      hx : Membership.mem (Set.uIoc a b) x
      ⊢ Summable fun n => (f n) x
    -/
    let x : (⟨uIcc a b, isCompact_uIcc⟩ : Compacts ℝ) := ⟨x, ⟨hx.1.le, hx.2⟩⟩
    /-
      case pos.h_lim
      ι : Type u_1
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      inst✝ : Countable ι
      f : ι → ContinuousMap Real E
      hf_sum : Summable fun i => Norm.norm (ContinuousMap.restrict (↑{ carrier := Se …
      hE : CompleteSpace E
      x✝ : Real
      hx : Membership.mem (Set.uIoc a b) x✝
      x : Subtype fun x => Membership.mem { carrier := Set.uIcc a b, isCompact' := ⋯ …
      ⊢ Summable fun n => (f n) x✝
    -/
    have := hf_sum.of_norm
    simpa only [Compacts.coe_mk, ContinuousMap.restrict_apply]
      using ContinuousMap.summable_apply this x


theorem tsum_intervalIntegral_eq_of_summable_norm [Countable ι] {f : ι → C(ℝ, E)}
    (hf_sum : Summable fun i : ι => ‖(f i).restrict (⟨uIcc a b, isCompact_uIcc⟩ : Compacts ℝ)‖) :
    ∑' i : ι, ∫ x in a..b, f i x = ∫ x in a..b, ∑' i : ι, f i x :=
  (hasSum_intervalIntegral_of_summable_norm hf_sum).tsum_eq


/-- Continuity of interval integral with respect to a parameter, at a point within a set.
  Given `F : X → ℝ → E`, assume `F x` is ae-measurable on `[a, b]` for `x` in a
  neighborhood of `x₀` within `s` and at `x₀`, and assume it is bounded by a function integrable
  on `[a, b]` independent of `x` in a neighborhood of `x₀` within `s`. If `(fun x ↦ F x t)`
  is continuous at `x₀` within `s` for almost every `t` in `[a, b]`
  then the same holds for `(fun x ↦ ∫ t in a..b, F x t ∂μ) s x₀`. -/
theorem continuousWithinAt_of_dominated_interval {F : X → ℝ → E} {x₀ : X} {bound : ℝ → ℝ} {a b : ℝ}
    {s : Set X} (hF_meas : ∀ᶠ x in 𝓝[s] x₀, AEStronglyMeasurable (F x) (μ.restrict <| Ι a b))
    (h_bound : ∀ᶠ x in 𝓝[s] x₀, ∀ᵐ t ∂μ, t ∈ Ι a b → ‖F x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_cont : ∀ᵐ t ∂μ, t ∈ Ι a b → ContinuousWithinAt (fun x => F x t) s x₀) :
    ContinuousWithinAt (fun x => ∫ t in a..b, F x t ∂μ) s x₀ :=
  tendsto_integral_filter_of_dominated_convergence bound hF_meas h_bound bound_integrable h_cont


/-- Continuity of interval integral with respect to a parameter at a point.
  Given `F : X → ℝ → E`, assume `F x` is ae-measurable on `[a, b]` for `x` in a
  neighborhood of `x₀`, and assume it is bounded by a function integrable on
  `[a, b]` independent of `x` in a neighborhood of `x₀`. If `(fun x ↦ F x t)`
  is continuous at `x₀` for almost every `t` in `[a, b]`
  then the same holds for `(fun x ↦ ∫ t in a..b, F x t ∂μ) s x₀`. -/
theorem continuousAt_of_dominated_interval {F : X → ℝ → E} {x₀ : X} {bound : ℝ → ℝ} {a b : ℝ}
    (hF_meas : ∀ᶠ x in 𝓝 x₀, AEStronglyMeasurable (F x) (μ.restrict <| Ι a b))
    (h_bound : ∀ᶠ x in 𝓝 x₀, ∀ᵐ t ∂μ, t ∈ Ι a b → ‖F x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_cont : ∀ᵐ t ∂μ, t ∈ Ι a b → ContinuousAt (fun x => F x t) x₀) :
    ContinuousAt (fun x => ∫ t in a..b, F x t ∂μ) x₀ :=
  tendsto_integral_filter_of_dominated_convergence bound hF_meas h_bound bound_integrable h_cont


/-- Continuity of interval integral with respect to a parameter.
  Given `F : X → ℝ → E`, assume each `F x` is ae-measurable on `[a, b]`,
  and assume it is bounded by a function integrable on `[a, b]` independent of `x`.
  If `(fun x ↦ F x t)` is continuous for almost every `t` in `[a, b]`
  then the same holds for `(fun x ↦ ∫ t in a..b, F x t ∂μ) s x₀`. -/
theorem continuous_of_dominated_interval {F : X → ℝ → E} {bound : ℝ → ℝ} {a b : ℝ}
    (hF_meas : ∀ x, AEStronglyMeasurable (F x) <| μ.restrict <| Ι a b)
    (h_bound : ∀ x, ∀ᵐ t ∂μ, t ∈ Ι a b → ‖F x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_cont : ∀ᵐ t ∂μ, t ∈ Ι a b → Continuous fun x => F x t) :
    Continuous fun x => ∫ t in a..b, F x t ∂μ :=
  continuous_iff_continuousAt.mpr fun _ =>
    continuousAt_of_dominated_interval (Eventually.of_forall hF_meas) (Eventually.of_forall h_bound)
        bound_integrable <|
      h_cont.mono fun _ himp hx => (himp hx).continuousAt


theorem continuousWithinAt_primitive (hb₀ : μ {b₀} = 0)
    (h_int : IntervalIntegrable f μ (min a b₁) (max a b₂)) :
    ContinuousWithinAt (fun b => ∫ x in a..b, f x ∂μ) (Icc b₁ b₂) b₀ := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b₀ b₁ b₂ : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    hb₀ : Eq (μ (Singleton.singleton b₀)) 0
    h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
    ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.Icc …
  -/
  by_cases h₀ : b₀ ∈ Icc b₁ b₂
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
      ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.Icc …
    -/
  · have h₁₂ : b₁ ≤ b₂ := h₀.1.trans h₀.2
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
      h₁₂ : LE.le b₁ b₂
      ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.Icc …
    -/
    have min₁₂ : min b₁ b₂ = b₁ := min_eq_left h₁₂
    have h_int' : ∀ {x}, x ∈ Icc b₁ b₂ → IntervalIntegrable f μ b₁ x := by
      rintro x ⟨h₁, h₂⟩
      apply h_int.mono_set
      apply uIcc_subset_uIcc
      · exact ⟨min_le_of_left_le (min_le_right a b₁),
          h₁.trans (h₂.trans <| le_max_of_le_right <| le_max_right _ _)⟩
      · exact ⟨min_le_of_left_le <| (min_le_right _ _).trans h₁,
          le_max_of_le_right <| h₂.trans <| le_max_right _ _⟩
    have : ∀ b ∈ Icc b₁ b₂,
        ∫ x in a..b, f x ∂μ = (∫ x in a..b₁, f x ∂μ) + ∫ x in b₁..b, f x ∂μ := by
      rintro b ⟨h₁, h₂⟩
      rw [← integral_add_adjacent_intervals _ (h_int' ⟨h₁, h₂⟩)]
      apply h_int.mono_set
      apply uIcc_subset_uIcc
      · exact ⟨min_le_of_left_le (min_le_left a b₁), le_max_of_le_right (le_max_left _ _)⟩
      · exact ⟨min_le_of_left_le (min_le_right _ _),
          le_max_of_le_right (h₁.trans <| h₂.trans (le_max_right a b₂))⟩
    /-
      case pos
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
      h₁₂ : LE.le b₁ b₂
      min₁₂ : Eq (Min.min b₁ b₂) b₁
      h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
      this : ∀ (b : Real), Membership.mem (Set.Icc b₁ b₂) b → Eq (intervalIntegral ( …
      ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.Icc …
    -/
    apply ContinuousWithinAt.congr _ this (this _ h₀); clear this
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
      h₁₂ : LE.le b₁ b₂
      min₁₂ : Eq (Min.min b₁ b₂) b₁
      h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
      ⊢ ContinuousWithinAt (fun y => HAdd.hAdd (intervalIntegral (fun x => f x) a b₁ …
    -/
    refine continuousWithinAt_const.add ?_
    have :
      (fun b => ∫ x in b₁..b, f x ∂μ) =ᶠ[𝓝[Icc b₁ b₂] b₀] fun b =>
        ∫ x in b₁..b₂, indicator {x | x ≤ b} f x ∂μ := by
      apply eventuallyEq_of_mem self_mem_nhdsWithin
      exact fun b b_in => (integral_indicator b_in).symm
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
      h₁₂ : LE.le b₁ b₂
      min₁₂ : Eq (Min.min b₁ b₂) b₁
      h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
      this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
      ⊢ ContinuousWithinAt (fun y => intervalIntegral (fun x => f x) b₁ y μ) (Set.Ic …
    -/
    apply ContinuousWithinAt.congr_of_eventuallyEq _ this (integral_indicator h₀).symm
    have : IntervalIntegrable (fun x => ‖f x‖) μ b₁ b₂ :=
      IntervalIntegrable.norm (h_int' <| right_mem_Icc.mpr h₁₂)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
      h₁₂ : LE.le b₁ b₂
      min₁₂ : Eq (Min.min b₁ b₂) b₁
      h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
      this✝ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegra …
      this : IntervalIntegrable (fun x => Norm.norm (f x)) μ b₁ b₂
      ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => (setOf fun x => LE.l …
    -/
    refine continuousWithinAt_of_dominated_interval ?_ ?_ this ?_ <;> clear this
      /-
        case refine_1
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        ⊢ Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable ((setOf fun x …
      -/
    · filter_upwards [self_mem_nhdsWithin]
      /-
        case h
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        ⊢ ∀ (a : Real), Membership.mem (Set.Icc b₁ b₂) a → MeasureTheory.AEStronglyMea …
      -/
      intro x hx
      /-
        case h
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        x : Real
        hx : Membership.mem (Set.Icc b₁ b₂) x
        ⊢ MeasureTheory.AEStronglyMeasurable ((setOf fun x_1 => LE.le x_1 x).indicator …
      -/
      erw [aestronglyMeasurable_indicator_iff, Measure.restrict_restrict, Iic_inter_Ioc_of_le]
        /-
          case h
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
          x : Real
          hx : Membership.mem (Set.Icc b₁ b₂) x
          ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.Ioc (Min.min b₁ b₂) x))
        -/
      · rw [min₁₂]
        /-
          case h
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
          x : Real
          hx : Membership.mem (Set.Icc b₁ b₂) x
          ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.Ioc b₁ x))
        -/
        exact (h_int' hx).1.aestronglyMeasurable
        /-
          🎉 no goals
        -/
        /-
          case h
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
          x : Real
          hx : Membership.mem (Set.Icc b₁ b₂) x
          ⊢ LE.le x (Max.max b₁ b₂)
        -/
      · exact le_max_of_le_right hx.2
        /-
          🎉 no goals
        -/
      /-
        case h
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        x : Real
        hx : Membership.mem (Set.Icc b₁ b₂) x
        ⊢ MeasurableSet (setOf fun x_1 => LE.le x_1 x)
      -/
      exacts [measurableSet_Iic, measurableSet_Iic]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        ⊢ Filter.Eventually (fun x => Filter.Eventually (fun t => Membership.mem (Set. …
      -/
    · filter_upwards with x; filter_upwards with t
      /-
        case refine_2.h.h
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        x t : Real
        ⊢ Membership.mem (Set.uIoc b₁ b₂) t → LE.le (Norm.norm ((setOf fun x_1 => LE.l …
      -/
      dsimp [indicator]
      /-
        case refine_2.h.h
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegral …
        x t : Real
        ⊢ Membership.mem (Set.uIoc b₁ b₂) t → LE.le (Norm.norm (ite (LE.le t x) (f t)  …
      -/
                    /-
                      🎉 no goals
                    -/
      split_ifs <;> simp
                    /-
                      🎉 no goals
                    -/
    · have : ∀ᵐ t ∂μ, t < b₀ ∨ b₀ < t := by
        filter_upwards [compl_mem_ae_iff.mpr hb₀] with x hx using Ne.lt_or_lt hx
      /-
        case refine_3
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this✝ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegra …
        this : Filter.Eventually (fun t => Or (LT.lt t b₀) (LT.lt b₀ t)) (MeasureTheor …
        ⊢ Filter.Eventually (fun t => Membership.mem (Set.uIoc b₁ b₂) t → ContinuousWi …
      -/
      apply this.mono
      /-
        case refine_3
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b₀ b₁ b₂ : Real
        μ : MeasureTheory.Measure Real
        f : Real → E
        hb₀ : Eq (μ (Singleton.singleton b₀)) 0
        h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
        h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
        h₁₂ : LE.le b₁ b₂
        min₁₂ : Eq (Min.min b₁ b₂) b₁
        h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
        this✝ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegra …
        this : Filter.Eventually (fun t => Or (LT.lt t b₀) (LT.lt b₀ t)) (MeasureTheor …
        ⊢ ∀ (x : Real), Or (LT.lt x b₀) (LT.lt b₀ x) → Membership.mem (Set.uIoc b₁ b₂) …
      -/
      rintro x₀ (hx₀ | hx₀) -
      · have : ∀ᶠ x in 𝓝[Icc b₁ b₂] b₀, {t : ℝ | t ≤ x}.indicator f x₀ = f x₀ := by
          apply mem_nhdsWithin_of_mem_nhds
          apply Eventually.mono (Ioi_mem_nhds hx₀)
          intro x hx
          simp [hx.le]
        /-
          case refine_3.inl
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this✝¹ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegr …
          this✝ : Filter.Eventually (fun t => Or (LT.lt t b₀) (LT.lt b₀ t)) (MeasureTheo …
          x₀ : Real
          hx₀ : LT.lt x₀ b₀
          this : Filter.Eventually (fun x => Eq ((setOf fun t => LE.le t x).indicator f  …
          ⊢ ContinuousWithinAt (fun x => (setOf fun x_1 => LE.le x_1 x).indicator f x₀)  …
        -/
        apply continuousWithinAt_const.congr_of_eventuallyEq this
        /-
          case refine_3.inl
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this✝¹ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegr …
          this✝ : Filter.Eventually (fun t => Or (LT.lt t b₀) (LT.lt b₀ t)) (MeasureTheo …
          x₀ : Real
          hx₀ : LT.lt x₀ b₀
          this : Filter.Eventually (fun x => Eq ((setOf fun t => LE.le t x).indicator f  …
          ⊢ Eq ((setOf fun t => LE.le t b₀).indicator f x₀) (f x₀)
        -/
        simp [hx₀.le]
        /-
          🎉 no goals
        -/
      · have : ∀ᶠ x in 𝓝[Icc b₁ b₂] b₀, {t : ℝ | t ≤ x}.indicator f x₀ = 0 := by
          apply mem_nhdsWithin_of_mem_nhds
          apply Eventually.mono (Iio_mem_nhds hx₀)
          intro x hx
          simp [hx]
        /-
          case refine_3.inr
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this✝¹ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegr …
          this✝ : Filter.Eventually (fun t => Or (LT.lt t b₀) (LT.lt b₀ t)) (MeasureTheo …
          x₀ : Real
          hx₀ : LT.lt b₀ x₀
          this : Filter.Eventually (fun x => Eq ((setOf fun t => LE.le t x).indicator f  …
          ⊢ ContinuousWithinAt (fun x => (setOf fun x_1 => LE.le x_1 x).indicator f x₀)  …
        -/
        apply continuousWithinAt_const.congr_of_eventuallyEq this
        /-
          case refine_3.inr
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Real E
          a b₀ b₁ b₂ : Real
          μ : MeasureTheory.Measure Real
          f : Real → E
          hb₀ : Eq (μ (Singleton.singleton b₀)) 0
          h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
          h₀ : Membership.mem (Set.Icc b₁ b₂) b₀
          h₁₂ : LE.le b₁ b₂
          min₁₂ : Eq (Min.min b₁ b₂) b₁
          h_int' : ∀ {x : Real}, Membership.mem (Set.Icc b₁ b₂) x → IntervalIntegrable f …
          this✝¹ : (nhdsWithin b₀ (Set.Icc b₁ b₂)).EventuallyEq (fun b => intervalIntegr …
          this✝ : Filter.Eventually (fun t => Or (LT.lt t b₀) (LT.lt b₀ t)) (MeasureTheo …
          x₀ : Real
          hx₀ : LT.lt b₀ x₀
          this : Filter.Eventually (fun x => Eq ((setOf fun t => LE.le t x).indicator f  …
          ⊢ Eq ((setOf fun t => LE.le t b₀).indicator f x₀) 0
        -/
        simp [hx₀]
        /-
          🎉 no goals
        -/
    /-
      case neg
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Not (Membership.mem (Set.Icc b₁ b₂) b₀)
      ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.Icc …
    -/
  · apply continuousWithinAt_of_not_mem_closure
    /-
      case neg.hx
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b₀ b₁ b₂ : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      hb₀ : Eq (μ (Singleton.singleton b₀)) 0
      h_int : IntervalIntegrable f μ (Min.min a b₁) (Max.max a b₂)
      h₀ : Not (Membership.mem (Set.Icc b₁ b₂) b₀)
      ⊢ Not (Membership.mem (closure (Set.Icc b₁ b₂)) b₀)
    -/
    rwa [closure_Icc]
    /-
      🎉 no goals
    -/


theorem continuousAt_parametric_primitive_of_dominated [FirstCountableTopology X]
    {F : X → ℝ → E} (bound : ℝ → ℝ) (a b : ℝ)
    {a₀ b₀ : ℝ} {x₀ : X} (hF_meas : ∀ x, AEStronglyMeasurable (F x) (μ.restrict <| Ι a b))
    (h_bound : ∀ᶠ x in 𝓝 x₀, ∀ᵐ t ∂μ.restrict <| Ι a b, ‖F x t‖ ≤ bound t)
    (bound_integrable : IntervalIntegrable bound μ a b)
    (h_cont : ∀ᵐ t ∂μ.restrict <| Ι a b, ContinuousAt (fun x ↦ F x t) x₀) (ha₀ : a₀ ∈ Ioo a b)
    (hb₀ : b₀ ∈ Ioo a b) (hμb₀ : μ {b₀} = 0) :
    ContinuousAt (fun p : X × ℝ ↦ ∫ t : ℝ in a₀..p.2, F p.1 t ∂μ) (x₀, b₀) := by
  have hsub : ∀ {a₀ b₀}, a₀ ∈ Ioo a b → b₀ ∈ Ioo a b → Ι a₀ b₀ ⊆ Ι a b := fun ha₀ hb₀ ↦
    (ordConnected_Ioo.uIoc_subset ha₀ hb₀).trans (Ioo_subset_Ioc_self.trans Ioc_subset_uIoc)
  /-
    E : Type u_1
    X : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝ : FirstCountableTopology X
    F : X → Real → E
    bound : Real → Real
    a b a₀ b₀ : Real
    x₀ : X
    hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
    bound_integrable : IntervalIntegrable bound μ a b
    h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
    ha₀ : Membership.mem (Set.Ioo a b) a₀
    hb₀ : Membership.mem (Set.Ioo a b) b₀
    hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
    hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
    ⊢ ContinuousAt (fun p => intervalIntegral (fun t => F p.1 t) a₀ p.2 μ) { fst : …
  -/
  have Ioo_nhds : Ioo a b ∈ 𝓝 b₀ := Ioo_mem_nhds hb₀.1 hb₀.2
  /-
    E : Type u_1
    X : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝ : FirstCountableTopology X
    F : X → Real → E
    bound : Real → Real
    a b a₀ b₀ : Real
    x₀ : X
    hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
    bound_integrable : IntervalIntegrable bound μ a b
    h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
    ha₀ : Membership.mem (Set.Ioo a b) a₀
    hb₀ : Membership.mem (Set.Ioo a b) b₀
    hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
    hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
    Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
    ⊢ ContinuousAt (fun p => intervalIntegral (fun t => F p.1 t) a₀ p.2 μ) { fst : …
  -/
  have Icc_nhds : Icc a b ∈ 𝓝 b₀ := Icc_mem_nhds hb₀.1 hb₀.2
  /-
    E : Type u_1
    X : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝ : FirstCountableTopology X
    F : X → Real → E
    bound : Real → Real
    a b a₀ b₀ : Real
    x₀ : X
    hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
    bound_integrable : IntervalIntegrable bound μ a b
    h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
    ha₀ : Membership.mem (Set.Ioo a b) a₀
    hb₀ : Membership.mem (Set.Ioo a b) b₀
    hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
    hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
    Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
    Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
    ⊢ ContinuousAt (fun p => intervalIntegral (fun t => F p.1 t) a₀ p.2 μ) { fst : …
  -/
  have hx₀ : ∀ᵐ t : ℝ ∂μ.restrict (Ι a b), ‖F x₀ t‖ ≤ bound t := h_bound.self_of_nhds
  have : ∀ᶠ p : X × ℝ in 𝓝 (x₀, b₀),
      ∫ s in a₀..p.2, F p.1 s ∂μ =
        ∫ s in a₀..b₀, F p.1 s ∂μ + ∫ s in b₀..p.2, F x₀ s ∂μ +
          ∫ s in b₀..p.2, F p.1 s - F x₀ s ∂μ := by
    rw [nhds_prod_eq]
    refine (h_bound.prod_mk Ioo_nhds).mono ?_
    rintro ⟨x, t⟩ ⟨hx : ∀ᵐ t : ℝ ∂μ.restrict (Ι a b), ‖F x t‖ ≤ bound t, ht : t ∈ Ioo a b⟩
    dsimp
    have hiF : ∀ {x a₀ b₀},
        (∀ᵐ t : ℝ ∂μ.restrict (Ι a b), ‖F x t‖ ≤ bound t) → a₀ ∈ Ioo a b → b₀ ∈ Ioo a b →
          IntervalIntegrable (F x) μ a₀ b₀ := fun {x a₀ b₀} hx ha₀ hb₀ ↦
      (bound_integrable.mono_set_ae <| Eventually.of_forall <| hsub ha₀ hb₀).mono_fun'
        ((hF_meas x).mono_set <| hsub ha₀ hb₀)
        (ae_restrict_of_ae_restrict_of_subset (hsub ha₀ hb₀) hx)
    rw [intervalIntegral.integral_sub, add_assoc, add_sub_cancel,
      intervalIntegral.integral_add_adjacent_intervals]
    · exact hiF hx ha₀ hb₀
    · exact hiF hx hb₀ ht
    · exact hiF hx hb₀ ht
    · exact hiF hx₀ hb₀ ht
  /-
    E : Type u_1
    X : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝ : FirstCountableTopology X
    F : X → Real → E
    bound : Real → Real
    a b a₀ b₀ : Real
    x₀ : X
    hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
    bound_integrable : IntervalIntegrable bound μ a b
    h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
    ha₀ : Membership.mem (Set.Ioo a b) a₀
    hb₀ : Membership.mem (Set.Ioo a b) b₀
    hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
    hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
    Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
    Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
    hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
    this : Filter.Eventually (fun p => Eq (intervalIntegral (fun s => F p.1 s) a₀  …
    ⊢ ContinuousAt (fun p => intervalIntegral (fun t => F p.1 t) a₀ p.2 μ) { fst : …
  -/
  rw [continuousAt_congr this]; clear this
  /-
    E : Type u_1
    X : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝ : FirstCountableTopology X
    F : X → Real → E
    bound : Real → Real
    a b a₀ b₀ : Real
    x₀ : X
    hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
    h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
    bound_integrable : IntervalIntegrable bound μ a b
    h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
    ha₀ : Membership.mem (Set.Ioo a b) a₀
    hb₀ : Membership.mem (Set.Ioo a b) b₀
    hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
    hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
    Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
    Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
    hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
    ⊢ ContinuousAt (fun x => HAdd.hAdd (HAdd.hAdd (intervalIntegral (fun s => F x. …
  -/
  refine (ContinuousAt.add ?_ ?_).add ?_
  · exact (intervalIntegral.continuousAt_of_dominated_interval
        (Eventually.of_forall fun x ↦ (hF_meas x).mono_set <| hsub ha₀ hb₀)
          (h_bound.mono fun x hx ↦
            ae_imp_of_ae_restrict <| ae_restrict_of_ae_restrict_of_subset (hsub ha₀ hb₀) hx)
          (bound_integrable.mono_set_ae <| Eventually.of_forall <| hsub ha₀ hb₀) <|
          ae_imp_of_ae_restrict <| ae_restrict_of_ae_restrict_of_subset (hsub ha₀ hb₀) h_cont).fst'
    /-
      case refine_2
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      ⊢ ContinuousAt (fun x => intervalIntegral (fun s => F x₀ s) b₀ x.2 μ) { fst := …
    -/
  · refine (?_ : ContinuousAt (fun t ↦ ∫ s in b₀..t, F x₀ s ∂μ) b₀).snd'
    /-
      case refine_2
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      ⊢ ContinuousAt (fun t => intervalIntegral (fun s => F x₀ s) b₀ t μ) b₀
    -/
    apply ContinuousWithinAt.continuousAt _ (Icc_mem_nhds hb₀.1 hb₀.2)
    /-
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      ⊢ ContinuousWithinAt (fun t => intervalIntegral (fun s => F x₀ s) b₀ t μ) (Set …
    -/
    apply intervalIntegral.continuousWithinAt_primitive hμb₀
    /-
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      ⊢ IntervalIntegrable (F x₀) μ (Min.min b₀ a) (Max.max b₀ b)
    -/
    rw [min_eq_right hb₀.1.le, max_eq_right hb₀.2.le]
    /-
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      ⊢ IntervalIntegrable (F x₀) μ a b
    -/
    exact bound_integrable.mono_fun' (hF_meas x₀) hx₀
    /-
      🎉 no goals
    -/
  · suffices Tendsto (fun x : X × ℝ ↦ ∫ s in b₀..x.2, F x.1 s - F x₀ s ∂μ) (𝓝 (x₀, b₀)) (𝓝 0) by
      simpa [ContinuousAt]
    have : ∀ᶠ p : X × ℝ in 𝓝 (x₀, b₀),
        ‖∫ s in b₀..p.2, F p.1 s - F x₀ s ∂μ‖ ≤ |∫ s in b₀..p.2, 2 * bound s ∂μ| := by
      rw [nhds_prod_eq]
      refine (h_bound.prod_mk Ioo_nhds).mono ?_
      rintro ⟨x, t⟩ ⟨hx : ∀ᵐ t ∂μ.restrict (Ι a b), ‖F x t‖ ≤ bound t, ht : t ∈ Ioo a b⟩
      have H : ∀ᵐ t : ℝ ∂μ.restrict (Ι b₀ t), ‖F x t - F x₀ t‖ ≤ 2 * bound t := by
        apply (ae_restrict_of_ae_restrict_of_subset (hsub hb₀ ht) (hx.and hx₀)).mono
        rintro s ⟨hs₁, hs₂⟩
        calc
          ‖F x s - F x₀ s‖ ≤ ‖F x s‖ + ‖F x₀ s‖ := norm_sub_le _ _
          _ ≤ 2 * bound s := by linarith only [hs₁, hs₂]
      exact intervalIntegral.norm_integral_le_of_norm_le H
        ((bound_integrable.mono_set' <| hsub hb₀ ht).const_mul 2)
    /-
      case refine_3
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      this : Filter.Eventually (fun p => LE.le (Norm.norm (intervalIntegral (fun s = …
      ⊢ Filter.Tendsto (fun x => intervalIntegral (fun s => HSub.hSub (F x.1 s) (F x …
    -/
    apply squeeze_zero_norm' this
    have : Tendsto (fun t ↦ ∫ s in b₀..t, 2 * bound s ∂μ) (𝓝 b₀) (𝓝 0) := by
      suffices ContinuousAt (fun t ↦ ∫ s in b₀..t, 2 * bound s ∂μ) b₀ by
        simpa [ContinuousAt] using this
      apply ContinuousWithinAt.continuousAt _ Icc_nhds
      apply intervalIntegral.continuousWithinAt_primitive hμb₀
      apply IntervalIntegrable.const_mul
      apply bound_integrable.mono_set'
      rw [min_eq_right hb₀.1.le, max_eq_right hb₀.2.le]
    /-
      case refine_3
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      this✝ : Filter.Eventually (fun p => LE.le (Norm.norm (intervalIntegral (fun s  …
      this : Filter.Tendsto (fun t => intervalIntegral (fun s => HMul.hMul 2 (bound  …
      ⊢ Filter.Tendsto (fun n => abs (intervalIntegral (fun s => HMul.hMul 2 (bound  …
    -/
    rw [nhds_prod_eq]
    /-
      case refine_3
      E : Type u_1
      X : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : TopologicalSpace X
      μ : MeasureTheory.Measure Real
      inst✝ : FirstCountableTopology X
      F : X → Real → E
      bound : Real → Real
      a b a₀ b₀ : Real
      x₀ : X
      hF_meas : ∀ (x : X), MeasureTheory.AEStronglyMeasurable (F x) (μ.restrict (Set …
      h_bound : Filter.Eventually (fun x => Filter.Eventually (fun t => LE.le (Norm. …
      bound_integrable : IntervalIntegrable bound μ a b
      h_cont : Filter.Eventually (fun t => ContinuousAt (fun x => F x t) x₀) (Measur …
      ha₀ : Membership.mem (Set.Ioo a b) a₀
      hb₀ : Membership.mem (Set.Ioo a b) b₀
      hμb₀ : Eq (μ (Singleton.singleton b₀)) 0
      hsub : ∀ {a₀ b₀ : Real}, Membership.mem (Set.Ioo a b) a₀ → Membership.mem (Set …
      Ioo_nhds : Membership.mem (nhds b₀) (Set.Ioo a b)
      Icc_nhds : Membership.mem (nhds b₀) (Set.Icc a b)
      hx₀ : Filter.Eventually (fun t => LE.le (Norm.norm (F x₀ t)) (bound t)) (Measu …
      this✝ : Filter.Eventually (fun p => LE.le (Norm.norm (intervalIntegral (fun s  …
      this : Filter.Tendsto (fun t => intervalIntegral (fun s => HMul.hMul 2 (bound  …
      ⊢ Filter.Tendsto (fun n => abs (intervalIntegral (fun s => HMul.hMul 2 (bound  …
    -/
    exact (continuous_abs.tendsto' _ _ abs_zero).comp (this.comp tendsto_snd)
    /-
      🎉 no goals
    -/


theorem continuousOn_primitive (h_int : IntegrableOn f (Icc a b) μ) :
    ContinuousOn (fun x => ∫ t in Ioc a x, f t ∂μ) (Icc a b) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
    ⊢ ContinuousOn (fun x => MeasureTheory.integral (μ.restrict (Set.Ioc a x)) fun …
  -/
  by_cases h : a ≤ b
  · have : ∀ x ∈ Icc a b, ∫ t in Ioc a x, f t ∂μ = ∫ t in a..x, f t ∂μ := by
      intro x x_in
      simp_rw [integral_of_le x_in.1]
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      inst✝ : MeasureTheory.NoAtoms μ
      h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
      h : LE.le a b
      this : ∀ (x : Real), Membership.mem (Set.Icc a b) x → Eq (MeasureTheory.integr …
      ⊢ ContinuousOn (fun x => MeasureTheory.integral (μ.restrict (Set.Ioc a x)) fun …
    -/
    rw [continuousOn_congr this]
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      inst✝ : MeasureTheory.NoAtoms μ
      h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
      h : LE.le a b
      this : ∀ (x : Real), Membership.mem (Set.Icc a b) x → Eq (MeasureTheory.integr …
      ⊢ ContinuousOn (fun x => intervalIntegral (fun t => f t) a x μ) (Set.Icc a b)
    -/
    intro x₀ _
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      inst✝ : MeasureTheory.NoAtoms μ
      h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
      h : LE.le a b
      this : ∀ (x : Real), Membership.mem (Set.Icc a b) x → Eq (MeasureTheory.integr …
      x₀ : Real
      a✝ : Membership.mem (Set.Icc a b) x₀
      ⊢ ContinuousWithinAt (fun x => intervalIntegral (fun t => f t) a x μ) (Set.Icc …
    -/
    refine continuousWithinAt_primitive (measure_singleton x₀) ?_
    simp only [intervalIntegrable_iff_integrableOn_Ioc_of_le, min_eq_left, max_eq_right, h,
      min_self]
    /-
      case pos
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      inst✝ : MeasureTheory.NoAtoms μ
      h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
      h : LE.le a b
      this : ∀ (x : Real), Membership.mem (Set.Icc a b) x → Eq (MeasureTheory.integr …
      x₀ : Real
      a✝ : Membership.mem (Set.Icc a b) x₀
      ⊢ MeasureTheory.IntegrableOn f (Set.Ioc a b) μ
    -/
    exact h_int.mono Ioc_subset_Icc_self le_rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      inst✝ : MeasureTheory.NoAtoms μ
      h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
      h : Not (LE.le a b)
      ⊢ ContinuousOn (fun x => MeasureTheory.integral (μ.restrict (Set.Ioc a x)) fun …
    -/
  · rw [Icc_eq_empty h]
    /-
      case neg
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      f : Real → E
      inst✝ : MeasureTheory.NoAtoms μ
      h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
      h : Not (LE.le a b)
      ⊢ ContinuousOn (fun x => MeasureTheory.integral (μ.restrict (Set.Ioc a x)) fun …
    -/
    exact continuousOn_empty _
    /-
      🎉 no goals
    -/


theorem continuousOn_primitive_Icc (h_int : IntegrableOn f (Icc a b) μ) :
    ContinuousOn (fun x => ∫ t in Icc a x, f t ∂μ) (Icc a b) := by
  have aux : (fun x => ∫ t in Icc a x, f t ∂μ) = fun x => ∫ t in Ioc a x, f t ∂μ := by
    ext x
    exact integral_Icc_eq_integral_Ioc
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
    aux : Eq (fun x => MeasureTheory.integral (μ.restrict (Set.Icc a x)) fun t =>  …
    ⊢ ContinuousOn (fun x => MeasureTheory.integral (μ.restrict (Set.Icc a x)) fun …
  -/
  rw [aux]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : MeasureTheory.IntegrableOn f (Set.Icc a b) μ
    aux : Eq (fun x => MeasureTheory.integral (μ.restrict (Set.Icc a x)) fun t =>  …
    ⊢ ContinuousOn (fun x => MeasureTheory.integral (μ.restrict (Set.Ioc a x)) fun …
  -/
  exact continuousOn_primitive h_int
  /-
    🎉 no goals
  -/


/-- Note: this assumes that `f` is `IntervalIntegrable`, in contrast to some other lemmas here. -/
theorem continuousOn_primitive_interval' (h_int : IntervalIntegrable f μ b₁ b₂)
    (ha : a ∈ [[b₁, b₂]]) : ContinuousOn (fun b => ∫ x in a..b, f x ∂μ) [[b₁, b₂]] := fun _ _ ↦ by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b₁ b₂ : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : IntervalIntegrable f μ b₁ b₂
    ha : Membership.mem (Set.uIcc b₁ b₂) a
    x✝¹ : Real
    x✝ : Membership.mem (Set.uIcc b₁ b₂) x✝¹
    ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.uIc …
  -/
  refine continuousWithinAt_primitive (measure_singleton _) ?_
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b₁ b₂ : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : IntervalIntegrable f μ b₁ b₂
    ha : Membership.mem (Set.uIcc b₁ b₂) a
    x✝¹ : Real
    x✝ : Membership.mem (Set.uIcc b₁ b₂) x✝¹
    ⊢ IntervalIntegrable f μ (Min.min a (Min.min b₁ b₂)) (Max.max a (Max.max b₁ b₂))
  -/
  rw [min_eq_right ha.1, max_eq_right ha.2]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b₁ b₂ : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : IntervalIntegrable f μ b₁ b₂
    ha : Membership.mem (Set.uIcc b₁ b₂) a
    x✝¹ : Real
    x✝ : Membership.mem (Set.uIcc b₁ b₂) x✝¹
    ⊢ IntervalIntegrable f μ (Min.min b₁ b₂) (Max.max b₁ b₂)
  -/
  simpa [intervalIntegrable_iff, uIoc] using h_int
  /-
    🎉 no goals
  -/


theorem continuousOn_primitive_interval (h_int : IntegrableOn f (uIcc a b) μ) :
    ContinuousOn (fun x => ∫ t in a..x, f t ∂μ) (uIcc a b) :=
  continuousOn_primitive_interval' h_int.intervalIntegrable left_mem_uIcc


theorem continuousOn_primitive_interval_left (h_int : IntegrableOn f (uIcc a b) μ) :
    ContinuousOn (fun x => ∫ t in x..b, f t ∂μ) (uIcc a b) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : MeasureTheory.IntegrableOn f (Set.uIcc a b) μ
    ⊢ ContinuousOn (fun x => intervalIntegral (fun t => f t) x b μ) (Set.uIcc a b)
  -/
  rw [uIcc_comm a b] at h_int ⊢
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : MeasureTheory.IntegrableOn f (Set.uIcc b a) μ
    ⊢ ContinuousOn (fun x => intervalIntegral (fun t => f t) x b μ) (Set.uIcc b a)
  -/
  simp only [integral_symm b]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : MeasureTheory.IntegrableOn f (Set.uIcc b a) μ
    ⊢ ContinuousOn (fun x => Neg.neg (intervalIntegral (fun t => f t) b x μ)) (Set …
  -/
  exact (continuousOn_primitive_interval h_int).neg
  /-
    🎉 no goals
  -/


theorem continuous_primitive (h_int : ∀ a b, IntervalIntegrable f μ a b) (a : ℝ) :
    Continuous fun b => ∫ x in a..b, f x ∂μ := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : ∀ (a b : Real), IntervalIntegrable f μ a b
    a : Real
    ⊢ Continuous fun b => intervalIntegral (fun x => f x) a b μ
  -/
  rw [continuous_iff_continuousAt]
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : ∀ (a b : Real), IntervalIntegrable f μ a b
    a : Real
    ⊢ ∀ (x : Real), ContinuousAt (fun b => intervalIntegral (fun x => f x) a b μ) x
  -/
  intro b₀
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : ∀ (a b : Real), IntervalIntegrable f μ a b
    a b₀ : Real
    ⊢ ContinuousAt (fun b => intervalIntegral (fun x => f x) a b μ) b₀
  -/
  cases' exists_lt b₀ with b₁ hb₁
  /-
    case intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : ∀ (a b : Real), IntervalIntegrable f μ a b
    a b₀ b₁ : Real
    hb₁ : LT.lt b₁ b₀
    ⊢ ContinuousAt (fun b => intervalIntegral (fun x => f x) a b μ) b₀
  -/
  cases' exists_gt b₀ with b₂ hb₂
  /-
    case intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : ∀ (a b : Real), IntervalIntegrable f μ a b
    a b₀ b₁ : Real
    hb₁ : LT.lt b₁ b₀
    b₂ : Real
    hb₂ : LT.lt b₀ b₂
    ⊢ ContinuousAt (fun b => intervalIntegral (fun x => f x) a b μ) b₀
  -/
  apply ContinuousWithinAt.continuousAt _ (Icc_mem_nhds hb₁ hb₂)
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝ : MeasureTheory.NoAtoms μ
    h_int : ∀ (a b : Real), IntervalIntegrable f μ a b
    a b₀ b₁ : Real
    hb₁ : LT.lt b₁ b₀
    b₂ : Real
    hb₂ : LT.lt b₀ b₂
    ⊢ ContinuousWithinAt (fun b => intervalIntegral (fun x => f x) a b μ) (Set.Icc …
  -/
  exact continuousWithinAt_primitive (measure_singleton b₀) (h_int _ _)
  /-
    🎉 no goals
  -/


nonrec theorem _root_.MeasureTheory.Integrable.continuous_primitive (h_int : Integrable f μ)
    (a : ℝ) : Continuous fun b => ∫ x in a..b, f x ∂μ :=
  continuous_primitive (fun _ _ => h_int.intervalIntegrable) a


theorem continuous_parametric_primitive_of_continuous
    {a₀ : ℝ} (hf : Continuous f.uncurry) :
    Continuous fun p : X × ℝ ↦ ∫ t in a₀..p.2, f p.1 t ∂μ := by
  -- We will prove continuity at a point `(q, b₀)`.
  /-
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    ⊢ Continuous fun p => intervalIntegral (fun t => f p.1 t) a₀ p.2 μ
  -/
  rw [continuous_iff_continuousAt]
  /-
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    ⊢ ∀ (x : Prod X Real), ContinuousAt (fun p => intervalIntegral (fun t => f p.1 …
  -/
  rintro ⟨q, b₀⟩
  /-
    case mk
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ : Real
    ⊢ ContinuousAt (fun p => intervalIntegral (fun t => f p.1 t) a₀ p.2 μ) { fst : …
  -/
  apply Metric.continuousAt_iff'.2 (fun ε εpos ↦ ?_)
  -- choose `a` and `b` such that `(a, b)` contains both `a₀` and `b₀`. We will use uniform
  -- estimates on a neighborhood of the compact set `{q} × [a, b]`.
  /-
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  cases' exists_lt (min a₀ b₀) with a a_lt
  /-
    case intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : LT.lt a (Min.min a₀ b₀)
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  cases' exists_gt (max a₀ b₀) with b lt_b
  /-
    case intro.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : LT.lt a (Min.min a₀ b₀)
    b : Real
    lt_b : LT.lt (Max.max a₀ b₀) b
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  rw [lt_min_iff] at a_lt
  /-
    case intro.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : LT.lt (Max.max a₀ b₀) b
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  rw [max_lt_iff] at lt_b
  /-
    case intro.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : And (LT.lt a₀ b) (LT.lt b₀ b)
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  have : IsCompact ({q} ×ˢ (Icc a b)) := isCompact_singleton.prod isCompact_Icc
  -- let `M` be a bound for `f` on the compact set `{q} × [a, b]`.
  /-
    case intro.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : And (LT.lt a₀ b) (LT.lt b₀ b)
    this : IsCompact (SProd.sprod (Singleton.singleton q) (Set.Icc a b))
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  obtain ⟨M, hM⟩ := this.bddAbove_image hf.norm.continuousOn
  -- let `δ` be small enough to satisfy several properties that will show up later.
  obtain ⟨δ, δpos, hδ, h'δ, h''δ⟩ : ∃ (δ : ℝ), 0 < δ ∧ δ < 1 ∧ Icc (b₀ - δ) (b₀ + δ) ⊆ Icc a b ∧
      (M + 1) * (μ (Icc (b₀ - δ) (b₀ + δ))).toReal + δ * (μ (Icc a b)).toReal < ε := by
    have A : ∀ᶠ δ in 𝓝[>] (0 : ℝ), δ ∈ Ioo 0 1 := Ioo_mem_nhdsGT zero_lt_one
    have B : ∀ᶠ δ in 𝓝 0, Icc (b₀ - δ) (b₀ + δ) ⊆ Icc a b := by
      have I : Tendsto (fun δ ↦ b₀ - δ) (𝓝 0) (𝓝 (b₀ - 0)) := tendsto_const_nhds.sub tendsto_id
      have J : Tendsto (fun δ ↦ b₀ + δ) (𝓝 0) (𝓝 (b₀ + 0)) := tendsto_const_nhds.add tendsto_id
      simp only [sub_zero, add_zero] at I J
      filter_upwards [(tendsto_order.1 I).1 _ a_lt.2, (tendsto_order.1 J).2 _ lt_b.2] with δ hδ h'δ
      exact Icc_subset_Icc hδ.le h'δ.le
    have C : ∀ᶠ δ in 𝓝 0,
        (M + 1) * (μ (Icc (b₀ - δ) (b₀ + δ))).toReal + δ * (μ (Icc a b)).toReal < ε := by
      suffices Tendsto
        (fun δ ↦ (M + 1) * (μ (Icc (b₀ - δ) (b₀ + δ))).toReal + δ * (μ (Icc a b)).toReal)
          (𝓝 0) (𝓝 ((M + 1) * (0 : ℝ≥0∞).toReal + 0 * (μ (Icc a b)).toReal)) by
        simp only [zero_toReal, mul_zero, zero_mul, add_zero] at this
        exact (tendsto_order.1 this).2 _ εpos
      apply Tendsto.add (Tendsto.mul tendsto_const_nhds _)
        (Tendsto.mul tendsto_id tendsto_const_nhds)
      exact (tendsto_toReal zero_ne_top).comp (tendsto_measure_Icc _ _)
    rcases (A.and ((B.and C).filter_mono nhdsWithin_le_nhds)).exists with ⟨δ, hδ, h'δ, h''δ⟩
    exact ⟨δ, hδ.1, hδ.2, h'δ, h''δ⟩
  -- By compactness of `[a, b]` and continuity of `f` there, if `p` is close enough to `q`
  -- then `f p x` is `δ`-close to `f q x`, uniformly in `x ∈ [a, b]`.
  -- (Note in particular that this implies a bound `M + δ ≤ M + 1` for `f p x`).
  obtain ⟨v, v_mem, hv⟩ : ∃ v ∈ 𝓝[univ] q, ∀ p ∈ v, ∀ x ∈ Icc a b, dist (f p x) (f q x) < δ :=
    IsCompact.mem_uniformity_of_prod isCompact_Icc hf.continuousOn (mem_univ _)
      (dist_mem_uniformity δpos)
  -- for `p` in this neighborhood and `s` which is `δ`-close to `b₀`, we will show that the
  -- integrals are `ε`-close.
  have : v ×ˢ (Ioo (b₀ - δ) (b₀ + δ)) ∈ 𝓝 (q, b₀) := by
    rw [nhdsWithin_univ] at v_mem
    simp only [prod_mem_nhds_iff, v_mem, true_and]
    apply Ioo_mem_nhds <;> linarith
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : And (LT.lt a₀ b) (LT.lt b₀ b)
    this✝ : IsCompact (SProd.sprod (Singleton.singleton q) (Set.Icc a b))
    M : Real
    hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt δ 1
    h'δ : HasSubset.Subset (Set.Icc (HSub.hSub b₀ δ) (HAdd.hAdd b₀ δ)) (Set.Icc a b)
    h''δ : LT.lt (HAdd.hAdd (HMul.hMul (HAdd.hAdd M 1) (μ (Set.Icc (HSub.hSub b₀ δ …
    v : Set X
    v_mem : Membership.mem (nhdsWithin q Set.univ) v
    hv : ∀ (p : X), Membership.mem v p → ∀ (x : Real), Membership.mem (Set.Icc a b …
    this : Membership.mem (nhds { fst := q, snd := b₀ }) (SProd.sprod v (Set.Ioo ( …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (intervalIntegral (fun t => f x …
  -/
  filter_upwards [this]
  /-
    case h
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : And (LT.lt a₀ b) (LT.lt b₀ b)
    this✝ : IsCompact (SProd.sprod (Singleton.singleton q) (Set.Icc a b))
    M : Real
    hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt δ 1
    h'δ : HasSubset.Subset (Set.Icc (HSub.hSub b₀ δ) (HAdd.hAdd b₀ δ)) (Set.Icc a b)
    h''δ : LT.lt (HAdd.hAdd (HMul.hMul (HAdd.hAdd M 1) (μ (Set.Icc (HSub.hSub b₀ δ …
    v : Set X
    v_mem : Membership.mem (nhdsWithin q Set.univ) v
    hv : ∀ (p : X), Membership.mem v p → ∀ (x : Real), Membership.mem (Set.Icc a b …
    this : Membership.mem (nhds { fst := q, snd := b₀ }) (SProd.sprod v (Set.Ioo ( …
    ⊢ ∀ (a : Prod X Real), Membership.mem (SProd.sprod v (Set.Ioo (HSub.hSub b₀ δ) …
  -/
  rintro ⟨p, s⟩ ⟨hp : p ∈ v, hs : s ∈ Ioo (b₀ - δ) (b₀ + δ)⟩
  /-
    case h.mk.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : And (LT.lt a₀ b) (LT.lt b₀ b)
    this✝ : IsCompact (SProd.sprod (Singleton.singleton q) (Set.Icc a b))
    M : Real
    hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt δ 1
    h'δ : HasSubset.Subset (Set.Icc (HSub.hSub b₀ δ) (HAdd.hAdd b₀ δ)) (Set.Icc a b)
    h''δ : LT.lt (HAdd.hAdd (HMul.hMul (HAdd.hAdd M 1) (μ (Set.Icc (HSub.hSub b₀ δ …
    v : Set X
    v_mem : Membership.mem (nhdsWithin q Set.univ) v
    hv : ∀ (p : X), Membership.mem v p → ∀ (x : Real), Membership.mem (Set.Icc a b …
    this : Membership.mem (nhds { fst := q, snd := b₀ }) (SProd.sprod v (Set.Ioo ( …
    p : X
    s : Real
    hp : Membership.mem v p
    hs : Membership.mem (Set.Ioo (HSub.hSub b₀ δ) (HAdd.hAdd b₀ δ)) s
    ⊢ LT.lt (Dist.dist (intervalIntegral (fun t => f { fst := p, snd := s }.1 t) a …
  -/
  simp only [dist_eq_norm] at hv ⊢
  /-
    case h.mk.intro
    E : Type u_1
    X : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : TopologicalSpace X
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.NoAtoms μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Real → E
    a₀ : Real
    hf : Continuous (Function.uncurry f)
    q : X
    b₀ ε : Real
    εpos : GT.gt ε 0
    a : Real
    a_lt : And (LT.lt a a₀) (LT.lt a b₀)
    b : Real
    lt_b : And (LT.lt a₀ b) (LT.lt b₀ b)
    this✝ : IsCompact (SProd.sprod (Singleton.singleton q) (Set.Icc a b))
    M : Real
    hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt δ 1
    h'δ : HasSubset.Subset (Set.Icc (HSub.hSub b₀ δ) (HAdd.hAdd b₀ δ)) (Set.Icc a b)
    h''δ : LT.lt (HAdd.hAdd (HMul.hMul (HAdd.hAdd M 1) (μ (Set.Icc (HSub.hSub b₀ δ …
    v : Set X
    v_mem : Membership.mem (nhdsWithin q Set.univ) v
    this : Membership.mem (nhds { fst := q, snd := b₀ }) (SProd.sprod v (Set.Ioo ( …
    p : X
    s : Real
    hp : Membership.mem v p
    hs : Membership.mem (Set.Ioo (HSub.hSub b₀ δ) (HAdd.hAdd b₀ δ)) s
    hv : ∀ (p : X), Membership.mem v p → ∀ (x : Real), Membership.mem (Set.Icc a b …
    ⊢ LT.lt (Norm.norm (HSub.hSub (intervalIntegral (fun t => f p t) a₀ s μ) (inte …
  -/
  have J r u v : IntervalIntegrable (f r) μ u v := (hf.uncurry_left _).intervalIntegrable _ _
  /- we compute the difference between the integrals by splitting the contribution of the change
  from `b₀` to `s` (which gives a contribution controlled by the measure of `(b₀ - δ, b₀ + δ)`,
  small enough thanks to our choice of `δ`) and the change from `q` to `p`, which is small as
  `f p x` and `f q x` are uniformly close by design. -/
  calc
  ‖∫ t in a₀..s, f p t ∂μ - ∫ t in a₀..b₀, f q t ∂μ‖
    = ‖(∫ t in a₀..s, f p t ∂μ - ∫ t in a₀..b₀, f p t ∂μ)
        + (∫ t in a₀..b₀, f p t ∂μ - ∫ t in a₀..b₀, f q t ∂μ)‖ := by congr 1; abel
  _ ≤ ‖∫ t in a₀..s, f p t ∂μ - ∫ t in a₀..b₀, f p t ∂μ‖
        + ‖∫ t in a₀..b₀, f p t ∂μ - ∫ t in a₀..b₀, f q t ∂μ‖ := norm_add_le _ _
  _ = ‖∫ t in b₀..s, f p t ∂μ‖ + ‖∫ t in a₀..b₀, (f p t - f q t) ∂μ‖ := by
      congr 2
      · rw [integral_interval_sub_left (J _ _ _) (J _ _ _)]
      · rw [integral_sub (J _ _ _) (J _ _ _)]
  _ ≤ ∫ t in Ι b₀ s, ‖f p t‖ ∂μ + ∫ t in Ι a₀ b₀, ‖f p t - f q t‖ ∂μ := by
      gcongr
      · exact norm_integral_le_integral_norm_Ioc
      · exact norm_integral_le_integral_norm_Ioc
  _ ≤ ∫ t in Icc (b₀ - δ) (b₀ + δ), ‖f p t‖ ∂μ + ∫ t in Icc a b, ‖f p t - f q t‖ ∂μ := by
      gcongr
      · apply setIntegral_mono_set
        · exact (hf.uncurry_left _).norm.integrableOn_Icc
        · exact Eventually.of_forall (fun x ↦ norm_nonneg _)
        · have : Ι b₀ s ⊆ Icc (b₀ - δ) (b₀ + δ) := by
            apply uIoc_subset_uIcc.trans (uIcc_subset_Icc ?_ ⟨hs.1.le, hs.2.le⟩ )
            simp [δpos.le]
          exact Eventually.of_forall this
      · apply setIntegral_mono_set
        · exact ((hf.uncurry_left _).sub (hf.uncurry_left _)).norm.integrableOn_Icc
        · exact Eventually.of_forall (fun x ↦ norm_nonneg _)
        · have : Ι a₀ b₀ ⊆ Icc a b := uIoc_subset_uIcc.trans
            (uIcc_subset_Icc ⟨a_lt.1.le, lt_b.1.le⟩ ⟨a_lt.2.le, lt_b.2.le⟩)
          exact Eventually.of_forall this
  _ ≤ ∫ t in Icc (b₀ - δ) (b₀ + δ), M + 1 ∂μ + ∫ _t in Icc a b, δ ∂μ := by
      gcongr ?_ + ?_
      · apply setIntegral_mono_on
        · exact (hf.uncurry_left _).norm.integrableOn_Icc
        · exact continuous_const.integrableOn_Icc
        · exact measurableSet_Icc
        · intro x hx
          calc ‖f p x‖ = ‖f q x + (f p x - f q x)‖ := by congr; abel
          _ ≤ ‖f q x‖ + ‖f p x - f q x‖ := norm_add_le _ _
          _ ≤ M + δ := by
              gcongr
              · apply hM
                change (fun x ↦ ‖Function.uncurry f x‖) (q, x) ∈ _
                apply mem_image_of_mem
                simp only [singleton_prod, mem_image, Prod.mk.injEq, true_and, exists_eq_right]
                exact h'δ hx
              · exact le_of_lt (hv _ hp _ (h'δ hx))
          _ ≤ M + 1 := by linarith
      · apply setIntegral_mono_on
        · exact ((hf.uncurry_left _).sub (hf.uncurry_left _)).norm.integrableOn_Icc
        · exact continuous_const.integrableOn_Icc
        · exact measurableSet_Icc
        · intro x hx
          exact le_of_lt (hv _ hp _ hx)
  _ = (M + 1) * (μ (Icc (b₀ - δ) (b₀ + δ))).toReal + δ * (μ (Icc a b)).toReal := by simp [mul_comm]
  _ < ε := h''δ


@[fun_prop]
theorem continuous_parametric_intervalIntegral_of_continuous {a₀ : ℝ}
    (hf : Continuous f.uncurry) {s : X → ℝ} (hs : Continuous s) :
    Continuous fun x ↦ ∫ t in a₀..s x, f x t ∂μ :=
  show Continuous ((fun p : X × ℝ ↦ ∫ t in a₀..p.2, f p.1 t ∂μ) ∘ fun x ↦ (x, s x)) from
    (continuous_parametric_primitive_of_continuous hf).comp₂ continuous_id hs


theorem continuous_parametric_intervalIntegral_of_continuous'
    (hf : Continuous f.uncurry) (a₀ b₀ : ℝ) :
                                                     /-
                                                       E : Type u_1
                                                       X : Type u_2
                                                       inst✝⁴ : NormedAddCommGroup E
                                                       inst✝³ : NormedSpace Real E
                                                       inst✝² : TopologicalSpace X
                                                       μ : MeasureTheory.Measure Real
                                                       inst✝¹ : MeasureTheory.NoAtoms μ
                                                       inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
                                                       f : X → Real → E
                                                       hf : Continuous (Function.uncurry f)
                                                       a₀ b₀ : Real
                                                       ⊢ Continuous fun x => intervalIntegral (fun t => f x t) a₀ b₀ μ
                                                     -/
    Continuous fun x ↦ ∫ t in a₀..b₀, f x t ∂μ := by fun_prop
                                                     /-
                                                       🎉 no goals
                                                     -/


