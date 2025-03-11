/-- If a locally integrable function `f` on a finite-dimensional real manifold has zero integral
when multiplied by any smooth compactly supported function, then `f` vanishes almost everywhere. -/
theorem ae_eq_zero_of_integral_smooth_smul_eq_zero [SigmaCompactSpace M]
    (hf : LocallyIntegrable f μ)
    (h : ∀ g : M → ℝ, ContMDiff I 𝓘(ℝ) ⊤ g → HasCompactSupport g → ∫ x, g x • f x ∂μ = 0) :
    ∀ᵐ x ∂μ, f x = 0 := by
  -- record topological properties of `M`
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  have := I.locallyCompactSpace
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this : LocallyCompactSpace H
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  have := ChartedSpace.locallyCompactSpace H M
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  have := I.secondCountableTopology
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝¹ : LocallyCompactSpace H
    this✝ : LocallyCompactSpace M
    this : SecondCountableTopology H
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  have := ChartedSpace.secondCountable_of_sigmaCompact H M
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝² : LocallyCompactSpace H
    this✝¹ : LocallyCompactSpace M
    this✝ : SecondCountableTopology H
    this : SecondCountableTopology M
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  have := Manifold.metrizableSpace I M
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝³ : LocallyCompactSpace H
    this✝² : LocallyCompactSpace M
    this✝¹ : SecondCountableTopology H
    this✝ : SecondCountableTopology M
    this : TopologicalSpace.MetrizableSpace M
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  let _ : MetricSpace M := TopologicalSpace.metrizableSpaceMetric M
  -- it suffices to show that the integral of the function vanishes on any compact set `s`
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝³ : LocallyCompactSpace H
    this✝² : LocallyCompactSpace M
    this✝¹ : SecondCountableTopology H
    this✝ : SecondCountableTopology M
    this : TopologicalSpace.MetrizableSpace M
    x✝ : MetricSpace M := TopologicalSpace.metrizableSpaceMetric M
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae μ)
  -/
  apply ae_eq_zero_of_forall_setIntegral_isCompact_eq_zero' hf (fun s hs ↦ Eq.symm ?_)
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝³ : LocallyCompactSpace H
    this✝² : LocallyCompactSpace M
    this✝¹ : SecondCountableTopology H
    this✝ : SecondCountableTopology M
    this : TopologicalSpace.MetrizableSpace M
    x✝ : MetricSpace M := TopologicalSpace.metrizableSpaceMetric M
    s : Set M
    hs : IsCompact s
    ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => f x)
  -/
  obtain ⟨δ, δpos, hδ⟩ : ∃ δ, 0 < δ ∧ IsCompact (cthickening δ s) := hs.exists_isCompact_cthickening
  -- choose a sequence of smooth functions `gₙ` equal to `1` on `s` and vanishing outside of the
  -- `uₙ`-neighborhood of `s`, where `uₙ` tends to zero. Then each integral `∫ gₙ f` vanishes,
  -- and by dominated convergence these integrals converge to `∫ x in s, f`.
  obtain ⟨u, -, u_pos, u_lim⟩ : ∃ u, StrictAnti u ∧ (∀ (n : ℕ), u n ∈ Ioo 0 δ)
    ∧ Tendsto u atTop (𝓝 0) := exists_seq_strictAnti_tendsto' δpos
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝³ : LocallyCompactSpace H
    this✝² : LocallyCompactSpace M
    this✝¹ : SecondCountableTopology H
    this✝ : SecondCountableTopology M
    this : TopologicalSpace.MetrizableSpace M
    x✝ : MetricSpace M := TopologicalSpace.metrizableSpaceMetric M
    s : Set M
    hs : IsCompact s
    δ : Real
    δpos : LT.lt 0 δ
    hδ : IsCompact (Metric.cthickening δ s)
    u : Nat → Real
    u_pos : ∀ (n : Nat), Membership.mem (Set.Ioo 0 δ) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => f x)
  -/
  let v : ℕ → Set M := fun n ↦ thickening (u n) s
  obtain ⟨K, K_compact, vK⟩ : ∃ K, IsCompact K ∧ ∀ n, v n ⊆ K :=
    ⟨_, hδ, fun n ↦ thickening_subset_cthickening_of_le (u_pos n).2.le _⟩
  have : ∀ n, ∃ (g : M → ℝ), support g = v n ∧ ContMDiff I 𝓘(ℝ) ⊤ g ∧ Set.range g ⊆ Set.Icc 0 1
          ∧ ∀ x ∈ s, g x = 1 := by
    intro n
    rcases exists_msmooth_support_eq_eq_one_iff I isOpen_thickening hs.isClosed
      (self_subset_thickening (u_pos n).1 s) with ⟨g, g_smooth, g_range, g_supp, hg⟩
    exact ⟨g, g_supp, g_smooth, g_range, fun x hx ↦ (hg x).1 hx⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝⁴ : LocallyCompactSpace H
    this✝³ : LocallyCompactSpace M
    this✝² : SecondCountableTopology H
    this✝¹ : SecondCountableTopology M
    this✝ : TopologicalSpace.MetrizableSpace M
    x✝ : MetricSpace M := TopologicalSpace.metrizableSpaceMetric M
    s : Set M
    hs : IsCompact s
    δ : Real
    δpos : LT.lt 0 δ
    hδ : IsCompact (Metric.cthickening δ s)
    u : Nat → Real
    u_pos : ∀ (n : Nat), Membership.mem (Set.Ioo 0 δ) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set M := fun n => Metric.thickening (u n) s
    K : Set M
    K_compact : IsCompact K
    vK : ∀ (n : Nat), HasSubset.Subset (v n) K
    this : ∀ (n : Nat), Exists fun g => And (Eq (Function.support g) (v n)) (And ( …
    ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => f x)
  -/
  choose g g_supp g_diff g_range hg using this
  -- main fact: the integral of `∫ gₙ f` tends to `∫ x in s, f`.
  have L : Tendsto (fun n ↦ ∫ x, g n x • f x ∂μ) atTop (𝓝 (∫ x in s, f x ∂μ)) := by
    rw [← integral_indicator hs.measurableSet]
    let bound : M → ℝ := K.indicator (fun x ↦ ‖f x‖)
    have A : ∀ n, AEStronglyMeasurable (fun x ↦ g n x • f x) μ :=
      fun n ↦ (g_diff n).continuous.aestronglyMeasurable.smul hf.aestronglyMeasurable
    have B : Integrable bound μ := by
      rw [integrable_indicator_iff K_compact.measurableSet]
      exact (hf.integrableOn_isCompact K_compact).norm
    have C : ∀ n, ∀ᵐ x ∂μ, ‖g n x • f x‖ ≤ bound x := by
      intro n
      filter_upwards with x
      rw [norm_smul]
      refine le_indicator_apply (fun _ ↦ ?_) (fun hxK ↦ ?_)
      · have : ‖g n x‖ ≤ 1 := by
          have := g_range n (mem_range_self (f := g n) x)
          rw [Real.norm_of_nonneg this.1]
          exact this.2
        exact mul_le_of_le_one_left (norm_nonneg _) this
      · have : g n x = 0 := by rw [← nmem_support, g_supp]; contrapose! hxK; exact vK n hxK
        simp [this]
    have D : ∀ᵐ x ∂μ, Tendsto (fun n => g n x • f x) atTop (𝓝 (s.indicator f x)) := by
      filter_upwards with x
      by_cases hxs : x ∈ s
      · have : ∀ n, g n x = 1 := fun n ↦ hg n x hxs
        simp [this, indicator_of_mem hxs f]
      · simp_rw [indicator_of_not_mem hxs f]
        apply tendsto_const_nhds.congr'
        suffices H : ∀ᶠ n in atTop, g n x = 0 by
          filter_upwards [H] with n hn using by simp [hn]
        obtain ⟨ε, εpos, hε⟩ : ∃ ε, 0 < ε ∧ x ∉ thickening ε s := by
          rw [← hs.isClosed.closure_eq, closure_eq_iInter_thickening s] at hxs
          simpa using hxs
        filter_upwards [(tendsto_order.1 u_lim).2 _ εpos] with n hn
        rw [← nmem_support, g_supp]
        contrapose! hε
        exact thickening_mono hn.le s hε
    exact tendsto_integral_of_dominated_convergence bound A B C D
  -- deduce that `∫ x in s, f = 0` as each integral `∫ gₙ f` vanishes by assumption
  have : ∀ n, ∫ x, g n x • f x ∂μ = 0 := by
    refine fun n ↦ h _ (g_diff n) ?_
    apply HasCompactSupport.of_support_subset_isCompact K_compact
    simpa [g_supp] using vK n
  /-
    case intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this✝⁴ : LocallyCompactSpace H
    this✝³ : LocallyCompactSpace M
    this✝² : SecondCountableTopology H
    this✝¹ : SecondCountableTopology M
    this✝ : TopologicalSpace.MetrizableSpace M
    x✝ : MetricSpace M := TopologicalSpace.metrizableSpaceMetric M
    s : Set M
    hs : IsCompact s
    δ : Real
    δpos : LT.lt 0 δ
    hδ : IsCompact (Metric.cthickening δ s)
    u : Nat → Real
    u_pos : ∀ (n : Nat), Membership.mem (Set.Ioo 0 δ) (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    v : Nat → Set M := fun n => Metric.thickening (u n) s
    K : Set M
    K_compact : IsCompact K
    vK : ∀ (n : Nat), HasSubset.Subset (v n) K
    g : Nat → M → Real
    g_supp : ∀ (n : Nat), Eq (Function.support (g n)) (v n)
    g_diff : ∀ (n : Nat), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g n)
    g_range : ∀ (n : Nat), HasSubset.Subset (Set.range (g n)) (Set.Icc 0 1)
    hg : ∀ (n : Nat) (x : M), Membership.mem s x → Eq (g n x) 1
    L : Filter.Tendsto (fun n => MeasureTheory.integral μ fun x => HSMul.hSMul (g  …
    this : ∀ (n : Nat), Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (g n x)  …
    ⊢ Eq 0 (MeasureTheory.integral (μ.restrict s) fun x => f x)
  -/
  simpa [this] using L
  /-
    🎉 no goals
  -/

-- An instance with keys containing `Opens`

instance (U : Opens M) : BorelSpace U := inferInstanceAs (BorelSpace (U : Set M))


/-- If a function `f` locally integrable on an open subset `U` of a finite-dimensional real
  manifold has zero integral when multiplied by any smooth function compactly supported
  in `U`, then `f` vanishes almost everywhere in `U`. -/
nonrec theorem IsOpen.ae_eq_zero_of_integral_smooth_smul_eq_zero' {U : Set M} (hU : IsOpen U)
    (hSig : IsSigmaCompact U) (hf : LocallyIntegrableOn f U μ)
    (h : ∀ g : M → ℝ,
      ContMDiff I 𝓘(ℝ) ⊤ g → HasCompactSupport g → tsupport g ⊆ U → ∫ x, g x • f x ∂μ = 0) :
    ∀ᵐ x ∂μ, x ∈ U → f x = 0 := by
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U : Set M
    hU : IsOpen U
    hSig : IsSigmaCompact U
    hf : MeasureTheory.LocallyIntegrableOn f U μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    ⊢ Filter.Eventually (fun x => Membership.mem U x → Eq (f x) 0) (MeasureTheory. …
  -/
  have meas_U := hU.measurableSet
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U : Set M
    hU : IsOpen U
    hSig : IsSigmaCompact U
    hf : MeasureTheory.LocallyIntegrableOn f U μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    meas_U : MeasurableSet U
    ⊢ Filter.Eventually (fun x => Membership.mem U x → Eq (f x) 0) (MeasureTheory. …
  -/
  rw [← ae_restrict_iff' meas_U, ae_restrict_iff_subtype meas_U]
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U : Set M
    hU : IsOpen U
    hSig : IsSigmaCompact U
    hf : MeasureTheory.LocallyIntegrableOn f U μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    meas_U : MeasurableSet U
    ⊢ Filter.Eventually (fun x => Eq (f ↑x) 0) (MeasureTheory.ae (MeasureTheory.Me …
  -/
  let U : Opens M := ⟨U, hU⟩
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U✝ : Set M
    hU : IsOpen U✝
    hSig : IsSigmaCompact U✝
    hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    meas_U : MeasurableSet U✝
    U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
    ⊢ Filter.Eventually (fun x => Eq (f ↑x) 0) (MeasureTheory.ae (MeasureTheory.Me …
  -/
  change ∀ᵐ (x : U) ∂_, _
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U✝ : Set M
    hU : IsOpen U✝
    hSig : IsSigmaCompact U✝
    hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    meas_U : MeasurableSet U✝
    U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
    ⊢ Filter.Eventually (fun x => Eq (f ↑x) 0) (MeasureTheory.ae (MeasureTheory.Me …
  -/
  haveI : SigmaCompactSpace U := isSigmaCompact_iff_sigmaCompactSpace.mp hSig
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U✝ : Set M
    hU : IsOpen U✝
    hSig : IsSigmaCompact U✝
    hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    meas_U : MeasurableSet U✝
    U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
    this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
    ⊢ Filter.Eventually (fun x => Eq (f ↑x) 0) (MeasureTheory.ae (MeasureTheory.Me …
  -/
  refine ae_eq_zero_of_integral_smooth_smul_eq_zero I ?_ fun g g_smth g_supp ↦ ?_
    /-
      case refine_1
      E : Type u_1
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : FiniteDimensional Real E
      F : Type u_2
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : CompleteSpace F
      H : Type u_3
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_4
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      inst✝² : MeasurableSpace M
      inst✝¹ : BorelSpace M
      inst✝ : T2Space M
      f : M → F
      μ : MeasureTheory.Measure M
      U✝ : Set M
      hU : IsOpen U✝
      hSig : IsSigmaCompact U✝
      hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
      h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
      meas_U : MeasurableSet U✝
      U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
      this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
      ⊢ MeasureTheory.LocallyIntegrable (fun x => f ↑x) (MeasureTheory.Measure.comap …
    -/
  · exact (locallyIntegrable_comap meas_U).mpr hf
    /-
      🎉 no goals
    -/
  specialize h (Subtype.val.extend g 0) (g_smth.extend_zero g_supp)
    (g_supp.extend_zero continuous_subtype_val) ((g_supp.tsupport_extend_zero_subset
      continuous_subtype_val).trans <| Subtype.coe_image_subset _ _)
  /-
    case refine_2
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U✝ : Set M
    hU : IsOpen U✝
    hSig : IsSigmaCompact U✝
    hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
    meas_U : MeasurableSet U✝
    U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
    this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
    g : (Subtype fun x => Membership.mem U x) → Real
    g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
    g_supp : HasCompactSupport g
    h : Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (Function.extend Subtype …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val μ) fun x …
  -/
  rw [← setIntegral_eq_integral_of_forall_compl_eq_zero (s := U) fun x hx ↦ ?_] at h
    /-
      case refine_2
      E : Type u_1
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : FiniteDimensional Real E
      F : Type u_2
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : CompleteSpace F
      H : Type u_3
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_4
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      inst✝² : MeasurableSpace M
      inst✝¹ : BorelSpace M
      inst✝ : T2Space M
      f : M → F
      μ : MeasureTheory.Measure M
      U✝ : Set M
      hU : IsOpen U✝
      hSig : IsSigmaCompact U✝
      hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
      meas_U : MeasurableSet U✝
      U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
      this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
      g : (Subtype fun x => Membership.mem U x) → Real
      g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_supp : HasCompactSupport g
      h : Eq (MeasureTheory.integral (μ.restrict ↑U) fun x => HSMul.hSMul (Function. …
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val μ) fun x …
    -/
  · rw [← integral_subtype_comap] at h
      /-
        case refine_2
        E : Type u_1
        inst✝¹² : NormedAddCommGroup E
        inst✝¹¹ : NormedSpace Real E
        inst✝¹⁰ : FiniteDimensional Real E
        F : Type u_2
        inst✝⁹ : NormedAddCommGroup F
        inst✝⁸ : NormedSpace Real F
        inst✝⁷ : CompleteSpace F
        H : Type u_3
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_4
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : SmoothManifoldWithCorners I M
        inst✝² : MeasurableSpace M
        inst✝¹ : BorelSpace M
        inst✝ : T2Space M
        f : M → F
        μ : MeasureTheory.Measure M
        U✝ : Set M
        hU : IsOpen U✝
        hSig : IsSigmaCompact U✝
        hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
        meas_U : MeasurableSet U✝
        U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
        this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
        g : (Subtype fun x => Membership.mem U x) → Real
        g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
        g_supp : HasCompactSupport g
        h : Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val μ) fun …
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.comap Subtype.val μ) fun x …
      -/
    · simp_rw [Subtype.val_injective.extend_apply] at h; exact h
                                                         /-
                                                           🎉 no goals
                                                         -/
      /-
        case refine_2.hs
        E : Type u_1
        inst✝¹² : NormedAddCommGroup E
        inst✝¹¹ : NormedSpace Real E
        inst✝¹⁰ : FiniteDimensional Real E
        F : Type u_2
        inst✝⁹ : NormedAddCommGroup F
        inst✝⁸ : NormedSpace Real F
        inst✝⁷ : CompleteSpace F
        H : Type u_3
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type u_4
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : SmoothManifoldWithCorners I M
        inst✝² : MeasurableSpace M
        inst✝¹ : BorelSpace M
        inst✝ : T2Space M
        f : M → F
        μ : MeasureTheory.Measure M
        U✝ : Set M
        hU : IsOpen U✝
        hSig : IsSigmaCompact U✝
        hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
        meas_U : MeasurableSet U✝
        U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
        this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
        g : (Subtype fun x => Membership.mem U x) → Real
        g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
        g_supp : HasCompactSupport g
        h : Eq (MeasureTheory.integral (μ.restrict ↑U) fun x => HSMul.hSMul (Function. …
        ⊢ MeasurableSet ↑U
      -/
    · exact meas_U
      /-
        🎉 no goals
      -/
  /-
    E : Type u_1
    inst✝¹² : NormedAddCommGroup E
    inst✝¹¹ : NormedSpace Real E
    inst✝¹⁰ : FiniteDimensional Real E
    F : Type u_2
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : CompleteSpace F
    H : Type u_3
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : MeasurableSpace M
    inst✝¹ : BorelSpace M
    inst✝ : T2Space M
    f : M → F
    μ : MeasureTheory.Measure M
    U✝ : Set M
    hU : IsOpen U✝
    hSig : IsSigmaCompact U✝
    hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
    meas_U : MeasurableSet U✝
    U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
    this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
    g : (Subtype fun x => Membership.mem U x) → Real
    g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
    g_supp : HasCompactSupport g
    h : Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (Function.extend Subtype …
    x : M
    hx : Not (Membership.mem (↑U) x)
    ⊢ Eq (HSMul.hSMul (Function.extend Subtype.val g 0 x) (f x)) 0
  -/
  rw [Function.extend_apply' _ _ _ (mt _ hx)]
    /-
      E : Type u_1
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : FiniteDimensional Real E
      F : Type u_2
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : CompleteSpace F
      H : Type u_3
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_4
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      inst✝² : MeasurableSpace M
      inst✝¹ : BorelSpace M
      inst✝ : T2Space M
      f : M → F
      μ : MeasureTheory.Measure M
      U✝ : Set M
      hU : IsOpen U✝
      hSig : IsSigmaCompact U✝
      hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
      meas_U : MeasurableSet U✝
      U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
      this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
      g : (Subtype fun x => Membership.mem U x) → Real
      g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_supp : HasCompactSupport g
      h : Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (Function.extend Subtype …
      x : M
      hx : Not (Membership.mem (↑U) x)
      ⊢ Eq (HSMul.hSMul (0 x) (f x)) 0
    -/
  · apply zero_smul
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝¹² : NormedAddCommGroup E
      inst✝¹¹ : NormedSpace Real E
      inst✝¹⁰ : FiniteDimensional Real E
      F : Type u_2
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : CompleteSpace F
      H : Type u_3
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_4
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      inst✝² : MeasurableSpace M
      inst✝¹ : BorelSpace M
      inst✝ : T2Space M
      f : M → F
      μ : MeasureTheory.Measure M
      U✝ : Set M
      hU : IsOpen U✝
      hSig : IsSigmaCompact U✝
      hf : MeasureTheory.LocallyIntegrableOn f U✝ μ
      meas_U : MeasurableSet U✝
      U : TopologicalSpace.Opens M := { carrier := U✝, is_open' := hU }
      this : SigmaCompactSpace (Subtype fun x => Membership.mem U x)
      g : (Subtype fun x => Membership.mem U x) → Real
      g_smth : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_supp : HasCompactSupport g
      h : Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (Function.extend Subtype …
      x : M
      hx : Not (Membership.mem (↑U) x)
      ⊢ (Exists fun a => Eq (↑a) x) → Membership.mem (↑U) x
    -/
  · rintro ⟨x, rfl⟩; exact x.2
                     /-
                       🎉 no goals
                     -/


theorem IsOpen.ae_eq_zero_of_integral_smooth_smul_eq_zero {U : Set M} (hU : IsOpen U)
    (hf : LocallyIntegrableOn f U μ)
    (h : ∀ g : M → ℝ,
      ContMDiff I 𝓘(ℝ) ⊤ g → HasCompactSupport g → tsupport g ⊆ U → ∫ x, g x • f x ∂μ = 0) :
    ∀ᵐ x ∂μ, x ∈ U → f x = 0 :=
  haveI := I.locallyCompactSpace
  haveI := ChartedSpace.locallyCompactSpace H M
  haveI := hU.locallyCompactSpace
  haveI := I.secondCountableTopology
  haveI := ChartedSpace.secondCountable_of_sigmaCompact H M
  hU.ae_eq_zero_of_integral_smooth_smul_eq_zero' _
    (isSigmaCompact_iff_sigmaCompactSpace.mpr inferInstance) hf h


/-- If two locally integrable functions on a finite-dimensional real manifold have the same integral
when multiplied by any smooth compactly supported function, then they coincide almost everywhere. -/
theorem ae_eq_of_integral_smooth_smul_eq
    (hf : LocallyIntegrable f μ) (hf' : LocallyIntegrable f' μ) (h : ∀ (g : M → ℝ),
      ContMDiff I 𝓘(ℝ) ⊤ g → HasCompactSupport g → ∫ x, g x • f x ∂μ = ∫ x, g x • f' x ∂μ) :
    ∀ᵐ x ∂μ, f x = f' x := by
  have : ∀ᵐ x ∂μ, (f - f') x = 0 := by
    apply ae_eq_zero_of_integral_smooth_smul_eq_zero I (hf.sub hf')
    intro g g_diff g_supp
    simp only [Pi.sub_apply, smul_sub]
    rw [integral_sub, sub_eq_zero]
    · exact h g g_diff g_supp
    · exact hf.integrable_smul_left_of_hasCompactSupport g_diff.continuous g_supp
    · exact hf'.integrable_smul_left_of_hasCompactSupport g_diff.continuous g_supp
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f f' : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    hf' : MeasureTheory.LocallyIntegrable f' μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this : Filter.Eventually (fun x => Eq (HSub.hSub f f' x) 0) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun x => Eq (f x) (f' x)) (MeasureTheory.ae μ)
  -/
  filter_upwards [this] with x hx
  /-
    case h
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    F : Type u_2
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace Real F
    inst✝⁸ : CompleteSpace F
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : SmoothManifoldWithCorners I M
    inst✝³ : MeasurableSpace M
    inst✝² : BorelSpace M
    inst✝¹ : T2Space M
    f f' : M → F
    μ : MeasureTheory.Measure M
    inst✝ : SigmaCompactSpace M
    hf : MeasureTheory.LocallyIntegrable f μ
    hf' : MeasureTheory.LocallyIntegrable f' μ
    h : ∀ (g : M → Real), ContMDiff I (modelWithCornersSelf Real Real) Top.top g → …
    this : Filter.Eventually (fun x => Eq (HSub.hSub f f' x) 0) (MeasureTheory.ae μ)
    x : M
    hx : Eq (HSub.hSub f f' x) 0
    ⊢ Eq (f x) (f' x)
  -/
  simpa [sub_eq_zero] using hx
  /-
    🎉 no goals
  -/


/-- If a locally integrable function `f` on a finite-dimensional real vector space has zero integral
when multiplied by any smooth compactly supported function, then `f` vanishes almost everywhere. -/
theorem ae_eq_zero_of_integral_contDiff_smul_eq_zero (hf : LocallyIntegrable f μ)
    (h : ∀ (g : E → ℝ), ContDiff ℝ ∞ g → HasCompactSupport g → ∫ x, g x • f x ∂μ = 0) :
    ∀ᵐ x ∂μ, f x = 0 :=
  ae_eq_zero_of_integral_smooth_smul_eq_zero 𝓘(ℝ, E) hf
    (fun g g_diff g_supp ↦ h g g_diff.contDiff g_supp)


/-- If two locally integrable functions on a finite-dimensional real vector space have the same
integral when multiplied by any smooth compactly supported function, then they coincide almost
everywhere. -/
theorem ae_eq_of_integral_contDiff_smul_eq
    (hf : LocallyIntegrable f μ) (hf' : LocallyIntegrable f' μ) (h : ∀ (g : E → ℝ),
      ContDiff ℝ ∞ g → HasCompactSupport g → ∫ x, g x • f x ∂μ = ∫ x, g x • f' x ∂μ) :
    ∀ᵐ x ∂μ, f x = f' x :=
  ae_eq_of_integral_smooth_smul_eq 𝓘(ℝ, E) hf hf'
    (fun g g_diff g_supp ↦ h g g_diff.contDiff g_supp)


/-- If a function `f` locally integrable on an open subset `U` of a finite-dimensional real
  manifold has zero integral when multiplied by any smooth function compactly supported
  in an open set `U`, then `f` vanishes almost everywhere in `U`. -/
theorem IsOpen.ae_eq_zero_of_integral_contDiff_smul_eq_zero {U : Set E} (hU : IsOpen U)
    (hf : LocallyIntegrableOn f U μ)
    (h : ∀ (g : E → ℝ), ContDiff ℝ ∞ g → HasCompactSupport g → tsupport g ⊆ U →
        ∫ x, g x • f x ∂μ = 0) :
    ∀ᵐ x ∂μ, x ∈ U → f x = 0 :=
  hU.ae_eq_zero_of_integral_smooth_smul_eq_zero 𝓘(ℝ, E) hf
    (fun g g_diff g_supp ↦ h g g_diff.contDiff g_supp)


