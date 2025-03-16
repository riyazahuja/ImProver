/-- If `φ` is a bump function, compute `(φ ⋆ g) x₀`
if `g` is constant on `Metric.ball x₀ φ.rOut`. -/
theorem convolution_eq_right {x₀ : G} (hg : ∀ x ∈ ball x₀ φ.rOut, g x = g x₀) :
    (φ ⋆[lsmul ℝ ℝ, μ] g : G → E') x₀ = integral μ φ • g x₀ := by
  /-
    G : Type uG
    E' : Type uE'
    inst✝⁶ : NormedAddCommGroup E'
    g : G → E'
    inst✝⁵ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁴ : NormedSpace Real E'
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace Real G
    inst✝¹ : HasContDiffBump G
    inst✝ : CompleteSpace E'
    φ : ContDiffBump 0
    x₀ : G
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ φ.rOut) x → Eq (g x) (g x₀)
    ⊢ Eq (MeasureTheory.convolution (↑φ) g (ContinuousLinearMap.lsmul Real Real) μ …
  -/
  simp_rw [convolution_eq_right' _ φ.support_eq.subset hg, lsmul_apply, integral_smul_const]
  /-
    🎉 no goals
  -/


/-- If `φ` is a normed bump function, compute `φ ⋆ g`
if `g` is constant on `Metric.ball x₀ φ.rOut`. -/
theorem normed_convolution_eq_right {x₀ : G} (hg : ∀ x ∈ ball x₀ φ.rOut, g x = g x₀) :
    (φ.normed μ ⋆[lsmul ℝ ℝ, μ] g : G → E') x₀ = g x₀ := by
  /-
    G : Type uG
    E' : Type uE'
    inst✝¹⁰ : NormedAddCommGroup E'
    g : G → E'
    inst✝⁹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁸ : NormedSpace Real E'
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : HasContDiffBump G
    inst✝⁴ : CompleteSpace E'
    φ : ContDiffBump 0
    inst✝³ : BorelSpace G
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : μ.IsOpenPosMeasure
    inst✝ : FiniteDimensional Real G
    x₀ : G
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ φ.rOut) x → Eq (g x) (g x₀)
    ⊢ Eq (MeasureTheory.convolution (φ.normed μ) g (ContinuousLinearMap.lsmul Real …
  -/
  rw [convolution_eq_right' _ φ.support_normed_eq.subset hg]
  /-
    G : Type uG
    E' : Type uE'
    inst✝¹⁰ : NormedAddCommGroup E'
    g : G → E'
    inst✝⁹ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁸ : NormedSpace Real E'
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : HasContDiffBump G
    inst✝⁴ : CompleteSpace E'
    φ : ContDiffBump 0
    inst✝³ : BorelSpace G
    inst✝² : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝¹ : μ.IsOpenPosMeasure
    inst✝ : FiniteDimensional Real G
    x₀ : G
    hg : ∀ (x : G), Membership.mem (Metric.ball x₀ φ.rOut) x → Eq (g x) (g x₀)
    ⊢ Eq (MeasureTheory.integral μ fun t => ((ContinuousLinearMap.lsmul Real Real) …
  -/
  exact integral_normed_smul φ μ (g x₀)
  /-
    🎉 no goals
  -/


/-- If `φ` is a normed bump function, approximate `(φ ⋆ g) x₀`
if `g` is near `g x₀` on a ball with radius `φ.rOut` around `x₀`. -/
theorem dist_normed_convolution_le {x₀ : G} {ε : ℝ} (hmg : AEStronglyMeasurable g μ)
    (hg : ∀ x ∈ ball x₀ φ.rOut, dist (g x) (g x₀) ≤ ε) :
    dist ((φ.normed μ ⋆[lsmul ℝ ℝ, μ] g : G → E') x₀) (g x₀) ≤ ε :=
                          /-
                            G : Type uG
                            E' : Type uE'
                            inst✝¹¹ : NormedAddCommGroup E'
                            g : G → E'
                            inst✝¹⁰ : MeasurableSpace G
                            μ : MeasureTheory.Measure G
                            inst✝⁹ : NormedSpace Real E'
                            inst✝⁸ : NormedAddCommGroup G
                            inst✝⁷ : NormedSpace Real G
                            inst✝⁶ : HasContDiffBump G
                            inst✝⁵ : CompleteSpace E'
                            φ : ContDiffBump 0
                            inst✝⁴ : BorelSpace G
                            inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
                            inst✝² : μ.IsOpenPosMeasure
                            inst✝¹ : FiniteDimensional Real G
                            inst✝ : μ.IsAddLeftInvariant
                            x₀ : G
                            ε : Real
                            hmg : MeasureTheory.AEStronglyMeasurable g μ
                            hg : ∀ (x : G), Membership.mem (Metric.ball x₀ φ.rOut) x → LE.le (Dist.dist (g …
                            ⊢ LE.le 0 ε
                          -/
  dist_convolution_le (by simp_rw [← dist_self (g x₀), hg x₀ (mem_ball_self φ.rOut_pos)])
                          /-
                            🎉 no goals
                          -/
    φ.support_normed_eq.subset φ.nonneg_normed φ.integral_normed hmg hg


/-- `(φ i ⋆ g i) (k i)` tends to `z₀` as `i` tends to some filter `l` if
* `φ` is a sequence of normed bump functions
  such that `(φ i).rOut` tends to `0` as `i` tends to `l`;
* `g i` is `μ`-a.e. strongly measurable as `i` tends to `l`;
* `g i x` tends to `z₀` as `(i, x)` tends to `l ×ˢ 𝓝 x₀`;
* `k i` tends to `x₀`. -/
nonrec theorem convolution_tendsto_right {ι} {φ : ι → ContDiffBump (0 : G)} {g : ι → G → E'}
    {k : ι → G} {x₀ : G} {z₀ : E'} {l : Filter ι} (hφ : Tendsto (fun i => (φ i).rOut) l (𝓝 0))
    (hig : ∀ᶠ i in l, AEStronglyMeasurable (g i) μ) (hcg : Tendsto (uncurry g) (l ×ˢ 𝓝 x₀) (𝓝 z₀))
    (hk : Tendsto k l (𝓝 x₀)) :
    Tendsto (fun i => ((φ i).normed μ ⋆[lsmul ℝ ℝ, μ] g i) (k i)) l (𝓝 z₀) :=
  convolution_tendsto_right (Eventually.of_forall fun i => (φ i).nonneg_normed)
    (Eventually.of_forall fun i => (φ i).integral_normed) (tendsto_support_normed_smallSets hφ) hig
    hcg hk


/-- Special case of `ContDiffBump.convolution_tendsto_right` where `g` is continuous,
  and the limit is taken only in the first function. -/
theorem convolution_tendsto_right_of_continuous {ι} {φ : ι → ContDiffBump (0 : G)} {l : Filter ι}
    (hφ : Tendsto (fun i => (φ i).rOut) l (𝓝 0)) (hg : Continuous g) (x₀ : G) :
    Tendsto (fun i => ((φ i).normed μ ⋆[lsmul ℝ ℝ, μ] g) x₀) l (𝓝 (g x₀)) :=
  convolution_tendsto_right hφ (Eventually.of_forall fun _ => hg.aestronglyMeasurable)
    ((hg.tendsto x₀).comp tendsto_snd) tendsto_const_nhds


/-- If a function `g` is locally integrable, then the convolution `φ i * g` converges almost
everywhere to `g` if `φ i` is a sequence of bump functions with support tending to `0`, provided
that the ratio between the inner and outer radii of `φ i` remains bounded. -/
theorem ae_convolution_tendsto_right_of_locallyIntegrable
    {ι} {φ : ι → ContDiffBump (0 : G)} {l : Filter ι} {K : ℝ}
    (hφ : Tendsto (fun i ↦ (φ i).rOut) l (𝓝 0))
    (h'φ : ∀ᶠ i in l, (φ i).rOut ≤ K * (φ i).rIn) (hg : LocallyIntegrable g μ) : ∀ᵐ x₀ ∂μ,
    Tendsto (fun i ↦ ((φ i).normed μ ⋆[lsmul ℝ ℝ, μ] g) x₀) l (𝓝 (g x₀)) := by
  /-
    G : Type uG
    E' : Type uE'
    inst✝¹¹ : NormedAddCommGroup E'
    g : G → E'
    inst✝¹⁰ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁹ : NormedSpace Real E'
    inst✝⁸ : NormedAddCommGroup G
    inst✝⁷ : NormedSpace Real G
    inst✝⁶ : HasContDiffBump G
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : BorelSpace G
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : FiniteDimensional Real G
    inst✝ : μ.IsAddLeftInvariant
    ι : Type u_1
    φ : ι → ContDiffBump 0
    l : Filter ι
    K : Real
    hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
    h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
    hg : MeasureTheory.LocallyIntegrable g μ
    ⊢ Filter.Eventually (fun x₀ => Filter.Tendsto (fun i => MeasureTheory.convolut …
  -/
  have : IsAddHaarMeasure μ := ⟨⟩
  -- By Lebesgue differentiation theorem, the average of `g` on a small ball converges
  -- almost everywhere to the value of `g` as the radius shrinks to zero.
  -- We will see that this set of points satisfies the desired conclusion.
  /-
    G : Type uG
    E' : Type uE'
    inst✝¹¹ : NormedAddCommGroup E'
    g : G → E'
    inst✝¹⁰ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁹ : NormedSpace Real E'
    inst✝⁸ : NormedAddCommGroup G
    inst✝⁷ : NormedSpace Real G
    inst✝⁶ : HasContDiffBump G
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : BorelSpace G
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : FiniteDimensional Real G
    inst✝ : μ.IsAddLeftInvariant
    ι : Type u_1
    φ : ι → ContDiffBump 0
    l : Filter ι
    K : Real
    hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
    h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
    hg : MeasureTheory.LocallyIntegrable g μ
    this : μ.IsAddHaarMeasure
    ⊢ Filter.Eventually (fun x₀ => Filter.Tendsto (fun i => MeasureTheory.convolut …
  -/
  filter_upwards [(Besicovitch.vitaliFamily μ).ae_tendsto_average_norm_sub hg] with x₀ h₀
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝¹¹ : NormedAddCommGroup E'
    g : G → E'
    inst✝¹⁰ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁹ : NormedSpace Real E'
    inst✝⁸ : NormedAddCommGroup G
    inst✝⁷ : NormedSpace Real G
    inst✝⁶ : HasContDiffBump G
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : BorelSpace G
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : FiniteDimensional Real G
    inst✝ : μ.IsAddLeftInvariant
    ι : Type u_1
    φ : ι → ContDiffBump 0
    l : Filter ι
    K : Real
    hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
    h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
    hg : MeasureTheory.LocallyIntegrable g μ
    this : μ.IsAddHaarMeasure
    x₀ : G
    h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.convolution ((φ i).normed μ) g (Conti …
  -/
  simp only [convolution_eq_swap, lsmul_apply]
  have hφ' : Tendsto (fun i ↦ (φ i).rOut) l (𝓝[>] 0) :=
    tendsto_nhdsWithin_iff.2 ⟨hφ, Eventually.of_forall (fun i ↦ (φ i).rOut_pos)⟩
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝¹¹ : NormedAddCommGroup E'
    g : G → E'
    inst✝¹⁰ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁹ : NormedSpace Real E'
    inst✝⁸ : NormedAddCommGroup G
    inst✝⁷ : NormedSpace Real G
    inst✝⁶ : HasContDiffBump G
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : BorelSpace G
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : FiniteDimensional Real G
    inst✝ : μ.IsAddLeftInvariant
    ι : Type u_1
    φ : ι → ContDiffBump 0
    l : Filter ι
    K : Real
    hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
    h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
    hg : MeasureTheory.LocallyIntegrable g μ
    this : μ.IsAddHaarMeasure
    x₀ : G
    h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun t => HSMul.hSMul ((φ i …
  -/
  have := (h₀.comp (Besicovitch.tendsto_filterAt μ x₀)).comp hφ'
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝¹¹ : NormedAddCommGroup E'
    g : G → E'
    inst✝¹⁰ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁹ : NormedSpace Real E'
    inst✝⁸ : NormedAddCommGroup G
    inst✝⁷ : NormedSpace Real G
    inst✝⁶ : HasContDiffBump G
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : BorelSpace G
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : FiniteDimensional Real G
    inst✝ : μ.IsAddLeftInvariant
    ι : Type u_1
    φ : ι → ContDiffBump 0
    l : Filter ι
    K : Real
    hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
    h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
    hg : MeasureTheory.LocallyIntegrable g μ
    this✝ : μ.IsAddHaarMeasure
    x₀ : G
    h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
    this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun t => HSMul.hSMul ((φ i …
  -/
  simp only [Function.comp] at this
  /-
    case h
    G : Type uG
    E' : Type uE'
    inst✝¹¹ : NormedAddCommGroup E'
    g : G → E'
    inst✝¹⁰ : MeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝⁹ : NormedSpace Real E'
    inst✝⁸ : NormedAddCommGroup G
    inst✝⁷ : NormedSpace Real G
    inst✝⁶ : HasContDiffBump G
    inst✝⁵ : CompleteSpace E'
    inst✝⁴ : BorelSpace G
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : μ.IsOpenPosMeasure
    inst✝¹ : FiniteDimensional Real G
    inst✝ : μ.IsAddLeftInvariant
    ι : Type u_1
    φ : ι → ContDiffBump 0
    l : Filter ι
    K : Real
    hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
    h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
    hg : MeasureTheory.LocallyIntegrable g μ
    this✝ : μ.IsAddHaarMeasure
    x₀ : G
    h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
    hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
    this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun t => HSMul.hSMul ((φ i …
  -/
  apply tendsto_integral_smul_of_tendsto_average_norm_sub (K ^ (Module.finrank ℝ G)) this
  · filter_upwards with i using
      hg.integrableOn_isCompact (isCompact_closedBall _ _)
    /-
      case h.hg
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun y => (φ i).normed μ (H …
    -/
  · apply tendsto_const_nhds.congr (fun i ↦ ?_)
    /-
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      ⊢ Eq 1 (MeasureTheory.integral μ fun y => (φ i).normed μ (HSub.hSub x₀ y))
    -/
    rw [← integral_neg_eq_self]
    /-
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      ⊢ Eq 1 (MeasureTheory.integral μ fun x => (φ i).normed μ (HSub.hSub x₀ (Neg.ne …
    -/
    simp only [sub_neg_eq_add, integral_add_left_eq_self, integral_normed]
    /-
      🎉 no goals
    -/
    /-
      case h.g_supp
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      ⊢ Filter.Eventually (fun i => HasSubset.Subset (Function.support fun y => (φ i …
    -/
  · filter_upwards with i
    /-
      case h.g_supp.h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      ⊢ HasSubset.Subset (Function.support fun y => (φ i).normed μ (HSub.hSub x₀ y)) …
    -/
    change support ((ContDiffBump.normed (φ i) μ) ∘ (fun y ↦ x₀ - y)) ⊆ closedBall x₀ (φ i).rOut
    /-
      case h.g_supp.h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      ⊢ HasSubset.Subset (Function.support (Function.comp ((φ i).normed μ) fun y =>  …
    -/
    simp only [support_comp_eq_preimage, support_normed_eq]
    /-
      case h.g_supp.h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      ⊢ HasSubset.Subset (Set.preimage (fun y => HSub.hSub x₀ y) (Metric.ball 0 (φ i …
    -/
    intro x hx
    /-
      case h.g_supp.h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      x : G
      hx : Membership.mem (Set.preimage (fun y => HSub.hSub x₀ y) (Metric.ball 0 (φ  …
      ⊢ Membership.mem (Metric.closedBall x₀ (φ i).rOut) x
    -/
    simp only [mem_preimage, mem_ball, dist_zero_right] at hx
    /-
      case h.g_supp.h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      x : G
      hx : LT.lt (Norm.norm (HSub.hSub x₀ x)) (φ i).rOut
      ⊢ Membership.mem (Metric.closedBall x₀ (φ i).rOut) x
    -/
    simpa [dist_eq_norm_sub'] using hx.le
    /-
      🎉 no goals
    -/
    /-
      case h.g_bound
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      ⊢ Filter.Eventually (fun i => ∀ (x : G), LE.le (abs ((φ i).normed μ (HSub.hSub …
    -/
  · filter_upwards [h'φ] with i hi x
    /-
      case h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      hi : LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)
      x : G
      ⊢ LE.le (abs ((φ i).normed μ (HSub.hSub x₀ x))) (HDiv.hDiv (HPow.hPow K (Modul …
    -/
    rw [abs_of_nonneg (nonneg_normed _ _), addHaar_closedBall_center]
    /-
      case h
      G : Type uG
      E' : Type uE'
      inst✝¹¹ : NormedAddCommGroup E'
      g : G → E'
      inst✝¹⁰ : MeasurableSpace G
      μ : MeasureTheory.Measure G
      inst✝⁹ : NormedSpace Real E'
      inst✝⁸ : NormedAddCommGroup G
      inst✝⁷ : NormedSpace Real G
      inst✝⁶ : HasContDiffBump G
      inst✝⁵ : CompleteSpace E'
      inst✝⁴ : BorelSpace G
      inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
      inst✝² : μ.IsOpenPosMeasure
      inst✝¹ : FiniteDimensional Real G
      inst✝ : μ.IsAddLeftInvariant
      ι : Type u_1
      φ : ι → ContDiffBump 0
      l : Filter ι
      K : Real
      hφ : Filter.Tendsto (fun i => (φ i).rOut) l (nhds 0)
      h'φ : Filter.Eventually (fun i => LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)) l
      hg : MeasureTheory.LocallyIntegrable g μ
      this✝ : μ.IsAddHaarMeasure
      x₀ : G
      h₀ : Filter.Tendsto (fun a => MeasureTheory.average (μ.restrict a) fun y => No …
      hφ' : Filter.Tendsto (fun i => (φ i).rOut) l (nhdsWithin 0 (Set.Ioi 0))
      this : Filter.Tendsto (Function.comp (Function.comp (fun a => MeasureTheory.av …
      i : ι
      hi : LE.le (φ i).rOut (HMul.hMul K (φ i).rIn)
      x : G
      ⊢ LE.le ((φ i).normed μ (HSub.hSub x₀ x)) (HDiv.hDiv (HPow.hPow K (Module.finr …
    -/
    exact (φ i).normed_le_div_measure_closedBall_rOut _ _ hi _
    /-
      🎉 no goals
    -/


