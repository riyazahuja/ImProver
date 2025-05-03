/-- If a sequence of peak functions `φᵢ` converges uniformly to zero away from a point `x₀`, and
`g` is integrable and has a limit at `x₀`, then `φᵢ • g` is eventually integrable. -/
theorem integrableOn_peak_smul_of_integrableOn_of_tendsto
    (hs : MeasurableSet s) (h'st : t ∈ 𝓝[s] x₀)
    (hlφ : ∀ u : Set α, IsOpen u → x₀ ∈ u → TendstoUniformlyOn φ 0 l (s \ u))
    (hiφ : Tendsto (fun i ↦ ∫ x in t, φ i x ∂μ) l (𝓝 1))
    (h'iφ : ∀ᶠ i in l, AEStronglyMeasurable (φ i) (μ.restrict s))
    (hmg : IntegrableOn g s μ) (hcg : Tendsto g (𝓝[s] x₀) (𝓝 a)) :
    ∀ᶠ i in l, IntegrableOn (fun x => φ i x • g x) s μ := by
  obtain ⟨u, u_open, x₀u, ut, hu⟩ :
      ∃ u, IsOpen u ∧ x₀ ∈ u ∧ s ∩ u ⊆ t ∧ ∀ x ∈ u ∩ s, g x ∈ ball a 1 := by
    rcases mem_nhdsWithin.1 (Filter.inter_mem h'st (hcg (ball_mem_nhds _ zero_lt_one)))
      with ⟨u, u_open, x₀u, hu⟩
    refine ⟨u, u_open, x₀u, ?_, hu.trans inter_subset_right⟩
    rw [inter_comm]
    exact hu.trans inter_subset_left
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s t : Set α
    φ : ι → α → Real
    a : E
    hs : MeasurableSet s
    h'st : Membership.mem (nhdsWithin x₀ s) t
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    u : Set α
    u_open : IsOpen u
    x₀u : Membership.mem u x₀
    ut : HasSubset.Subset (Inter.inter s u) t
    hu : ∀ (x : α), Membership.mem (Inter.inter u s) x → Membership.mem (Metric.ba …
    ⊢ Filter.Eventually (fun i => MeasureTheory.IntegrableOn (fun x => HSMul.hSMul …
  -/
  rw [tendsto_iff_norm_sub_tendsto_zero] at hiφ
  filter_upwards [tendstoUniformlyOn_iff.1 (hlφ u u_open x₀u) 1 zero_lt_one,
    (tendsto_order.1 hiφ).2 1 zero_lt_one, h'iφ] with i hi h'i h''i
  /-
    case h
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s t : Set α
    φ : ι → α → Real
    a : E
    hs : MeasurableSet s
    h'st : Membership.mem (nhdsWithin x₀ s) t
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun e => Norm.norm (HSub.hSub (MeasureTheory.integral (μ …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    u : Set α
    u_open : IsOpen u
    x₀u : Membership.mem u x₀
    ut : HasSubset.Subset (Inter.inter s u) t
    hu : ∀ (x : α), Membership.mem (Inter.inter u s) x → Membership.mem (Metric.ba …
    i : ι
    hi : ∀ (x : α), Membership.mem (SDiff.sdiff s u) x → LT.lt (Dist.dist (0 x) (φ …
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    h''i : MeasureTheory.AEStronglyMeasurable (φ i) (μ.restrict s)
    ⊢ MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
  -/
  have I : IntegrableOn (φ i) t μ := .of_integral_ne_zero (fun h ↦ by simp [h] at h'i)
  have A : IntegrableOn (fun x => φ i x • g x) (s \ u) μ := by
    refine Integrable.smul_of_top_right (hmg.mono diff_subset le_rfl) ?_
    apply memℒp_top_of_bound (h''i.mono_set diff_subset) 1
    filter_upwards [self_mem_ae_restrict (hs.diff u_open.measurableSet)] with x hx
    simpa only [Pi.zero_apply, dist_zero_left] using (hi x hx).le
  have B : IntegrableOn (fun x => φ i x • g x) (s ∩ u) μ := by
    apply Integrable.smul_of_top_left
    · exact IntegrableOn.mono_set I ut
    · apply
        memℒp_top_of_bound (hmg.mono_set inter_subset_left).aestronglyMeasurable (‖a‖ + 1)
      filter_upwards [self_mem_ae_restrict (hs.inter u_open.measurableSet)] with x hx
      rw [inter_comm] at hx
      exact (norm_lt_of_mem_ball (hu x hx)).le
  /-
    case h
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s t : Set α
    φ : ι → α → Real
    a : E
    hs : MeasurableSet s
    h'st : Membership.mem (nhdsWithin x₀ s) t
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun e => Norm.norm (HSub.hSub (MeasureTheory.integral (μ …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    u : Set α
    u_open : IsOpen u
    x₀u : Membership.mem u x₀
    ut : HasSubset.Subset (Inter.inter s u) t
    hu : ∀ (x : α), Membership.mem (Inter.inter u s) x → Membership.mem (Metric.ba …
    i : ι
    hi : ∀ (x : α), Membership.mem (SDiff.sdiff s u) x → LT.lt (Dist.dist (0 x) (φ …
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    h''i : MeasureTheory.AEStronglyMeasurable (φ i) (μ.restrict s)
    I : MeasureTheory.IntegrableOn (φ i) t μ
    A : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) (SDiff.sdi …
    B : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) (Inter.int …
    ⊢ MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
  -/
  convert A.union B
  /-
    case h.e'_7
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s t : Set α
    φ : ι → α → Real
    a : E
    hs : MeasurableSet s
    h'st : Membership.mem (nhdsWithin x₀ s) t
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun e => Norm.norm (HSub.hSub (MeasureTheory.integral (μ …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    u : Set α
    u_open : IsOpen u
    x₀u : Membership.mem u x₀
    ut : HasSubset.Subset (Inter.inter s u) t
    hu : ∀ (x : α), Membership.mem (Inter.inter u s) x → Membership.mem (Metric.ba …
    i : ι
    hi : ∀ (x : α), Membership.mem (SDiff.sdiff s u) x → LT.lt (Dist.dist (0 x) (φ …
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    h''i : MeasureTheory.AEStronglyMeasurable (φ i) (μ.restrict s)
    I : MeasureTheory.IntegrableOn (φ i) t μ
    A : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) (SDiff.sdi …
    B : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) (Inter.int …
    ⊢ Eq s (Union.union (SDiff.sdiff s u) (Inter.inter s u))
  -/
  simp only [diff_union_inter]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-20")]
alias integrableOn_peak_smul_of_integrableOn_of_continuousWithinAt :=
  integrableOn_peak_smul_of_integrableOn_of_tendsto


/-- If a sequence of peak functions `φᵢ` converges uniformly to zero away from a point `x₀` and its
integral on some finite-measure neighborhood of `x₀` converges to `1`, and `g` is integrable and
has a limit `a` at `x₀`, then `∫ φᵢ • g` converges to `a`.
Auxiliary lemma where one assumes additionally `a = 0`. -/
theorem tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto_aux
    (hs : MeasurableSet s) (ht : MeasurableSet t) (hts : t ⊆ s) (h'ts : t ∈ 𝓝[s] x₀)
    (hnφ : ∀ᶠ i in l, ∀ x ∈ s, 0 ≤ φ i x)
    (hlφ : ∀ u : Set α, IsOpen u → x₀ ∈ u → TendstoUniformlyOn φ 0 l (s \ u))
    (hiφ : Tendsto (fun i ↦ ∫ x in t, φ i x ∂μ) l (𝓝 1))
    (h'iφ : ∀ᶠ i in l, AEStronglyMeasurable (φ i) (μ.restrict s))
    (hmg : IntegrableOn g s μ) (hcg : Tendsto g (𝓝[s] x₀) (𝓝 0)) :
    Tendsto (fun i : ι => ∫ x in s, φ i x • g x ∂μ) l (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace α
    inst✝² : BorelSpace α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s t : Set α
    φ : ι → α → Real
    hs : MeasurableSet s
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds 0)
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => HSMu …
  -/
  refine Metric.tendsto_nhds.2 fun ε εpos => ?_
  obtain ⟨δ, hδ, δpos, δone⟩ : ∃ δ, (δ * ∫ x in s, ‖g x‖ ∂μ) + 2 * δ < ε ∧ 0 < δ ∧ δ < 1 := by
    have A :
      Tendsto (fun δ => (δ * ∫ x in s, ‖g x‖ ∂μ) + 2 * δ) (𝓝[>] 0)
        (𝓝 ((0 * ∫ x in s, ‖g x‖ ∂μ) + 2 * 0)) := by
      apply Tendsto.mono_left _ nhdsWithin_le_nhds
      exact (tendsto_id.mul tendsto_const_nhds).add (tendsto_id.const_mul _)
    rw [zero_mul, zero_add, mul_zero] at A
    have : Ioo (0 : ℝ) 1 ∈ 𝓝[>] 0 := Ioo_mem_nhdsGT zero_lt_one
    rcases (((tendsto_order.1 A).2 ε εpos).and this).exists with ⟨δ, hδ, h'δ⟩
    exact ⟨δ, hδ, h'δ.1, h'δ.2⟩
  suffices ∀ᶠ i in l, ‖∫ x in s, φ i x • g x ∂μ‖ ≤ (δ * ∫ x in s, ‖g x‖ ∂μ) + 2 * δ by
    filter_upwards [this] with i hi
    simp only [dist_zero_right]
    exact hi.trans_lt hδ
  obtain ⟨u, u_open, x₀u, ut, hu⟩ :
      ∃ u, IsOpen u ∧ x₀ ∈ u ∧ s ∩ u ⊆ t ∧ ∀ x ∈ u ∩ s, g x ∈ ball 0 δ := by
    rcases mem_nhdsWithin.1 (Filter.inter_mem h'ts (hcg (ball_mem_nhds _ δpos)))
      with ⟨u, u_open, x₀u, hu⟩
    refine ⟨u, u_open, x₀u, ?_, hu.trans inter_subset_right⟩
    rw [inter_comm]
    exact hu.trans inter_subset_left
  filter_upwards [tendstoUniformlyOn_iff.1 (hlφ u u_open x₀u) δ δpos,
    (tendsto_order.1 (tendsto_iff_norm_sub_tendsto_zero.1 hiφ)).2 δ δpos, hnφ,
    integrableOn_peak_smul_of_integrableOn_of_tendsto hs h'ts hlφ hiφ h'iφ hmg hcg]
    with i hi h'i hφpos h''i
  have I : IntegrableOn (φ i) t μ := by
    apply Integrable.of_integral_ne_zero (fun h ↦ ?_)
    simp [h] at h'i
    linarith
  have B : ‖∫ x in s ∩ u, φ i x • g x ∂μ‖ ≤ 2 * δ :=
    calc
      ‖∫ x in s ∩ u, φ i x • g x ∂μ‖ ≤ ∫ x in s ∩ u, ‖φ i x • g x‖ ∂μ :=
        norm_integral_le_integral_norm _
      _ ≤ ∫ x in s ∩ u, ‖φ i x‖ * δ ∂μ := by
        refine setIntegral_mono_on ?_ ?_ (hs.inter u_open.measurableSet) fun x hx => ?_
        · exact IntegrableOn.mono_set h''i.norm inter_subset_left
        · exact IntegrableOn.mono_set (I.norm.mul_const _) ut
        rw [norm_smul]
        apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
        rw [inter_comm] at hu
        exact (mem_ball_zero_iff.1 (hu x hx)).le
      _ ≤ ∫ x in t, ‖φ i x‖ * δ ∂μ := by
        apply setIntegral_mono_set
        · exact I.norm.mul_const _
        · exact Eventually.of_forall fun x => mul_nonneg (norm_nonneg _) δpos.le
        · exact Eventually.of_forall ut
      _ = ∫ x in t, φ i x * δ ∂μ := by
        apply setIntegral_congr_fun ht fun x hx => ?_
        rw [Real.norm_of_nonneg (hφpos _ (hts hx))]
      _ = (∫ x in t, φ i x ∂μ) * δ := by rw [integral_mul_right]
      _ ≤ 2 * δ := by gcongr; linarith [(le_abs_self _).trans h'i.le]
  have C : ‖∫ x in s \ u, φ i x • g x ∂μ‖ ≤ δ * ∫ x in s, ‖g x‖ ∂μ :=
    calc
      ‖∫ x in s \ u, φ i x • g x ∂μ‖ ≤ ∫ x in s \ u, ‖φ i x • g x‖ ∂μ :=
        norm_integral_le_integral_norm _
      _ ≤ ∫ x in s \ u, δ * ‖g x‖ ∂μ := by
        refine setIntegral_mono_on ?_ ?_ (hs.diff u_open.measurableSet) fun x hx => ?_
        · exact IntegrableOn.mono_set h''i.norm diff_subset
        · exact IntegrableOn.mono_set (hmg.norm.const_mul _) diff_subset
        rw [norm_smul]
        apply mul_le_mul_of_nonneg_right _ (norm_nonneg _)
        simpa only [Pi.zero_apply, dist_zero_left] using (hi x hx).le
      _ ≤ δ * ∫ x in s, ‖g x‖ ∂μ := by
        rw [integral_mul_left]
        apply mul_le_mul_of_nonneg_left (setIntegral_mono_set hmg.norm _ _) δpos.le
        · filter_upwards with x using norm_nonneg _
        · filter_upwards using diff_subset (s := s) (t := u)
  calc
    ‖∫ x in s, φ i x • g x ∂μ‖ =
      ‖(∫ x in s \ u, φ i x • g x ∂μ) + ∫ x in s ∩ u, φ i x • g x ∂μ‖ := by
      conv_lhs => rw [← diff_union_inter s u]
      rw [setIntegral_union disjoint_sdiff_inter (hs.inter u_open.measurableSet)
          (h''i.mono_set diff_subset) (h''i.mono_set inter_subset_left)]
    _ ≤ ‖∫ x in s \ u, φ i x • g x ∂μ‖ + ‖∫ x in s ∩ u, φ i x • g x ∂μ‖ := norm_add_le _ _
    _ ≤ (δ * ∫ x in s, ‖g x‖ ∂μ) + 2 * δ := add_le_add C B


@[deprecated (since := "2024-02-20")]
alias tendsto_setIntegral_peak_smul_of_integrableOn_of_continuousWithinAt_aux :=
  tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto_aux


/-- If a sequence of peak functions `φᵢ` converges uniformly to zero away from a point `x₀` and its
integral on some finite-measure neighborhood of `x₀` converges to `1`, and `g` is integrable and
has a limit `a` at `x₀`, then `∫ φᵢ • g` converges to `a`. Version localized to a subset. -/
theorem tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto
    (hs : MeasurableSet s) {t : Set α} (ht : MeasurableSet t) (hts : t ⊆ s) (h'ts : t ∈ 𝓝[s] x₀)
    (h't : μ t ≠ ∞) (hnφ : ∀ᶠ i in l, ∀ x ∈ s, 0 ≤ φ i x)
    (hlφ : ∀ u : Set α, IsOpen u → x₀ ∈ u → TendstoUniformlyOn φ 0 l (s \ u))
    (hiφ : Tendsto (fun i ↦ ∫ x in t, φ i x ∂μ) l (𝓝 1))
    (h'iφ : ∀ᶠ i in l, AEStronglyMeasurable (φ i) (μ.restrict s))
    (hmg : IntegrableOn g s μ) (hcg : Tendsto g (𝓝[s] x₀) (𝓝 a)) :
    Tendsto (fun i : ι ↦ ∫ x in s, φ i x • g x ∂μ) l (𝓝 a) := by
  /-
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => HSMu …
  -/
  let h := g - t.indicator (fun _ ↦ a)
  have A : Tendsto (fun i : ι => (∫ x in s, φ i x • h x ∂μ) + (∫ x in t, φ i x ∂μ) • a) l
      (𝓝 (0 + (1 : ℝ) • a)) := by
    refine Tendsto.add ?_ (Tendsto.smul hiφ tendsto_const_nhds)
    apply tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto_aux hs ht hts h'ts
        hnφ hlφ hiφ h'iφ
    · apply hmg.sub
      simp only [integrable_indicator_iff ht, integrableOn_const, ht, Measure.restrict_apply]
      right
      exact lt_of_le_of_lt (measure_mono inter_subset_left) (h't.lt_top)
    · rw [← sub_self a]
      apply Tendsto.sub hcg
      apply tendsto_const_nhds.congr'
      filter_upwards [h'ts] with x hx using by simp [hx]
  /-
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => HSMu …
  -/
  simp only [one_smul, zero_add] at A
  /-
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => HSMu …
  -/
  refine Tendsto.congr' ?_ A
  filter_upwards [integrableOn_peak_smul_of_integrableOn_of_tendsto hs h'ts
    hlφ hiφ h'iφ hmg hcg,
    (tendsto_order.1 (tendsto_iff_norm_sub_tendsto_zero.1 hiφ)).2 1 zero_lt_one] with i hi h'i
  /-
    case h
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    i : ι
    hi : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (φ …
  -/
  simp only [h, Pi.sub_apply, smul_sub, ← indicator_smul_apply]
  rw [integral_sub hi, setIntegral_indicator ht, inter_eq_right.mpr hts,
    integral_smul_const, sub_add_cancel]
  /-
    case h
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    i : ι
    hi : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    ⊢ MeasureTheory.Integrable (t.indicator fun a_1 => HSMul.hSMul (φ i a_1) a) (μ …
  -/
  rw [integrable_indicator_iff ht]
  /-
    case h
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    i : ι
    hi : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    ⊢ MeasureTheory.IntegrableOn (fun a_1 => HSMul.hSMul (φ i a_1) a) t (μ.restric …
  -/
  apply Integrable.smul_const
  /-
    case h.hf
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    i : ι
    hi : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    ⊢ MeasureTheory.Integrable (φ i) ((μ.restrict s).restrict t)
  -/
  rw [restrict_restrict ht, inter_eq_left.mpr hts]
  /-
    case h.hf
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    s : Set α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    hs : MeasurableSet s
    t : Set α
    ht : MeasurableSet t
    hts : HasSubset.Subset t s
    h'ts : Membership.mem (nhdsWithin x₀ s) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → LE.le 0 (φ i …
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) (μ …
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : Filter.Tendsto g (nhdsWithin x₀ s) (nhds a)
    h : α → E := HSub.hSub g (t.indicator fun x => a)
    A : Filter.Tendsto (fun i => HAdd.hAdd (MeasureTheory.integral (μ.restrict s)  …
    i : ι
    hi : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (φ i x) (g x)) s μ
    h'i : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral (μ.restrict t) fun x …
    ⊢ MeasureTheory.Integrable (φ i) (μ.restrict t)
  -/
  exact .of_integral_ne_zero (fun h ↦ by simp [h] at h'i)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-20")]
alias tendsto_setIntegral_peak_smul_of_integrableOn_of_continuousWithinAt :=
  tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto


/-- If a sequence of peak functions `φᵢ` converges uniformly to zero away from a point `x₀` and its
integral on some finite-measure neighborhood of `x₀` converges to `1`, and `g` is integrable and
has a limit `a` at `x₀`, then `∫ φᵢ • g` converges to `a`. -/
theorem tendsto_integral_peak_smul_of_integrable_of_tendsto
    {t : Set α} (ht : MeasurableSet t) (h'ts : t ∈ 𝓝 x₀)
    (h't : μ t ≠ ∞) (hnφ : ∀ᶠ i in l, ∀ x, 0 ≤ φ i x)
    (hlφ : ∀ u : Set α, IsOpen u → x₀ ∈ u → TendstoUniformlyOn φ 0 l uᶜ)
    (hiφ : Tendsto (fun i ↦ ∫ x in t, φ i x ∂μ) l (𝓝 1))
    (h'iφ : ∀ᶠ i in l, AEStronglyMeasurable (φ i) μ)
    (hmg : Integrable g μ) (hcg : Tendsto g (𝓝 x₀) (𝓝 a)) :
    Tendsto (fun i : ι ↦ ∫ x, φ i x • g x ∂μ) l (𝓝 a) := by
  /-
    α : Type u_1
    E : Type u_2
    ι : Type u_3
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : BorelSpace α
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    g : α → E
    l : Filter ι
    x₀ : α
    φ : ι → α → Real
    a : E
    inst✝ : CompleteSpace E
    t : Set α
    ht : MeasurableSet t
    h'ts : Membership.mem (nhds x₀) t
    h't : Ne (μ t) Top.top
    hnφ : Filter.Eventually (fun i => ∀ (x : α), LE.le 0 (φ i x)) l
    hlφ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 l …
    hiφ : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict t) fun x =>  …
    h'iφ : Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (φ i) μ) l
    hmg : MeasureTheory.Integrable g μ
    hcg : Filter.Tendsto g (nhds x₀) (nhds a)
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral μ fun x => HSMul.hSMul (φ i  …
  -/
  suffices Tendsto (fun i : ι ↦ ∫ x in univ, φ i x • g x ∂μ) l (𝓝 a) by simpa
  exact tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto MeasurableSet.univ ht (x₀ := x₀)
    (subset_univ _) (by simpa [nhdsWithin_univ]) h't (by simpa)
    (by simpa [← compl_eq_univ_diff] using hlφ) hiφ
    (by simpa) (by simpa) (by simpa [nhdsWithin_univ])


/-- If a continuous function `c` realizes its maximum at a unique point `x₀` in a compact set `s`,
then the sequence of functions `(c x) ^ n / ∫ (c x) ^ n` is a sequence of peak functions
concentrating around `x₀`. Therefore, `∫ (c x) ^ n * g / ∫ (c x) ^ n` converges to `g x₀` if `g` is
integrable on `s` and continuous at `x₀`.

Version assuming that `μ` gives positive mass to all neighborhoods of `x₀` within `s`.
For a less precise but more usable version, see
`tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_continuousOn`.
 -/
theorem tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_measure_nhdsWithin_pos
    [MetrizableSpace α] [IsLocallyFiniteMeasure μ] (hs : IsCompact s)
    (hμ : ∀ u, IsOpen u → x₀ ∈ u → 0 < μ (u ∩ s)) {c : α → ℝ} (hc : ContinuousOn c s)
    (h'c : ∀ y ∈ s, y ≠ x₀ → c y < c x₀) (hnc : ∀ x ∈ s, 0 ≤ c x) (hnc₀ : 0 < c x₀) (h₀ : x₀ ∈ s)
    (hmg : IntegrableOn g s μ) (hcg : ContinuousWithinAt g s x₀) :
    Tendsto (fun n : ℕ => (∫ x in s, c x ^ n ∂μ)⁻¹ • ∫ x in s, c x ^ n • g x ∂μ)
      atTop (𝓝 (g x₀)) := by
  /- We apply the general result
    `tendsto_setIntegral_peak_smul_of_integrableOn_of_continuousWithinAt` to the sequence of
    peak functions `φₙ = (c x) ^ n / ∫ (c x) ^ n`. The only nontrivial bit is to check that this
    sequence converges uniformly to zero on any set `s \ u` away from `x₀`. By compactness, the
    function `c` is bounded by `t < c x₀` there. Consider `t' ∈ (t, c x₀)`, and a neighborhood `v`
    of `x₀` where `c x ≥ t'`, by continuity. Then `∫ (c x) ^ n` is bounded below by `t' ^ n μ v`.
    It follows that, on `s \ u`, then `φₙ x ≤ t ^ n / (t' ^ n μ v)`,
    which tends (exponentially fast) to zero with `n`. -/
  /-
    α : Type u_1
    E : Type u_2
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : BorelSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    g : α → E
    x₀ : α
    s : Set α
    inst✝² : CompleteSpace E
    inst✝¹ : TopologicalSpace.MetrizableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hs : IsCompact s
    hμ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → LT.lt 0 (μ (Inter.inter u …
    c : α → Real
    hc : ContinuousOn c s
    h'c : ∀ (y : α), Membership.mem s y → Ne y x₀ → LT.lt (c y) (c x₀)
    hnc : ∀ (x : α), Membership.mem s x → LE.le 0 (c x)
    hnc₀ : LT.lt 0 (c x₀)
    h₀ : Membership.mem s x₀
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : ContinuousWithinAt g s x₀
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv (MeasureTheory.integral (μ.res …
  -/
  let φ : ℕ → α → ℝ := fun n x => (∫ x in s, c x ^ n ∂μ)⁻¹ * c x ^ n
  have hnφ : ∀ n, ∀ x ∈ s, 0 ≤ φ n x := by
    intro n x hx
    apply mul_nonneg (inv_nonneg.2 _) (pow_nonneg (hnc x hx) _)
    exact setIntegral_nonneg hs.measurableSet fun x hx => pow_nonneg (hnc x hx) _
  have I : ∀ n, IntegrableOn (fun x => c x ^ n) s μ := fun n =>
    ContinuousOn.integrableOn_compact hs (hc.pow n)
  have J : ∀ n, 0 ≤ᵐ[μ.restrict s] fun x : α => c x ^ n := by
    intro n
    filter_upwards [ae_restrict_mem hs.measurableSet] with x hx
    exact pow_nonneg (hnc x hx) n
  have P : ∀ n, (0 : ℝ) < ∫ x in s, c x ^ n ∂μ := by
    intro n
    refine (setIntegral_pos_iff_support_of_nonneg_ae (J n) (I n)).2 ?_
    obtain ⟨u, u_open, x₀_u, hu⟩ : ∃ u : Set α, IsOpen u ∧ x₀ ∈ u ∧ u ∩ s ⊆ c ⁻¹' Ioi 0 :=
      _root_.continuousOn_iff.1 hc x₀ h₀ (Ioi (0 : ℝ)) isOpen_Ioi hnc₀
    apply (hμ u u_open x₀_u).trans_le
    exact measure_mono fun x hx => ⟨ne_of_gt (pow_pos (a := c x) (hu hx) _), hx.2⟩
  have hiφ : ∀ n, ∫ x in s, φ n x ∂μ = 1 := fun n => by
    rw [integral_mul_left, inv_mul_cancel₀ (P n).ne']
  have A : ∀ u : Set α, IsOpen u → x₀ ∈ u → TendstoUniformlyOn φ 0 atTop (s \ u) := by
    intro u u_open x₀u
    obtain ⟨t, t_pos, tx₀, ht⟩ : ∃ t, 0 ≤ t ∧ t < c x₀ ∧ ∀ x ∈ s \ u, c x ≤ t := by
      rcases eq_empty_or_nonempty (s \ u) with (h | h)
      · exact
          ⟨0, le_rfl, hnc₀, by simp only [h, mem_empty_iff_false, IsEmpty.forall_iff, imp_true_iff]⟩
      obtain ⟨x, hx, h'x⟩ : ∃ x ∈ s \ u, ∀ y ∈ s \ u, c y ≤ c x :=
        IsCompact.exists_isMaxOn (hs.diff u_open) h (hc.mono diff_subset)
      refine ⟨c x, hnc x hx.1, h'c x hx.1 ?_, h'x⟩
      rintro rfl
      exact hx.2 x₀u
    obtain ⟨t', tt', t'x₀⟩ : ∃ t', t < t' ∧ t' < c x₀ := exists_between tx₀
    have t'_pos : 0 < t' := t_pos.trans_lt tt'
    obtain ⟨v, v_open, x₀_v, hv⟩ : ∃ v : Set α, IsOpen v ∧ x₀ ∈ v ∧ v ∩ s ⊆ c ⁻¹' Ioi t' :=
      _root_.continuousOn_iff.1 hc x₀ h₀ (Ioi t') isOpen_Ioi t'x₀
    have M : ∀ n, ∀ x ∈ s \ u, φ n x ≤ (μ (v ∩ s)).toReal⁻¹ * (t / t') ^ n := by
      intro n x hx
      have B : t' ^ n * (μ (v ∩ s)).toReal ≤ ∫ y in s, c y ^ n ∂μ :=
        calc
          t' ^ n * (μ (v ∩ s)).toReal = ∫ _ in v ∩ s, t' ^ n ∂μ := by
            simp only [integral_const, Measure.restrict_apply, MeasurableSet.univ, univ_inter,
              Algebra.id.smul_eq_mul, mul_comm]
          _ ≤ ∫ y in v ∩ s, c y ^ n ∂μ := by
            apply setIntegral_mono_on _ _ (v_open.measurableSet.inter hs.measurableSet) _
            · apply integrableOn_const.2 (Or.inr _)
              exact lt_of_le_of_lt (measure_mono inter_subset_right) hs.measure_lt_top
            · exact (I n).mono inter_subset_right le_rfl
            · intro x hx
              exact pow_le_pow_left₀ t'_pos.le (hv hx).le _
          _ ≤ ∫ y in s, c y ^ n ∂μ :=
            setIntegral_mono_set (I n) (J n) (Eventually.of_forall inter_subset_right)
      simp_rw [φ, ← div_eq_inv_mul, div_pow, div_div]
      have := ENNReal.toReal_pos (hμ v v_open x₀_v).ne'
        ((measure_mono inter_subset_right).trans_lt hs.measure_lt_top).ne
      gcongr
      · exact hnc _ hx.1
      · exact ht x hx
    have N :
      Tendsto (fun n => (μ (v ∩ s)).toReal⁻¹ * (t / t') ^ n) atTop
        (𝓝 ((μ (v ∩ s)).toReal⁻¹ * 0)) := by
      apply Tendsto.mul tendsto_const_nhds _
      apply tendsto_pow_atTop_nhds_zero_of_lt_one (div_nonneg t_pos t'_pos.le)
      exact (div_lt_one t'_pos).2 tt'
    rw [mul_zero] at N
    refine tendstoUniformlyOn_iff.2 fun ε εpos => ?_
    filter_upwards [(tendsto_order.1 N).2 ε εpos] with n hn x hx
    simp only [Pi.zero_apply, dist_zero_left, Real.norm_of_nonneg (hnφ n x hx.1)]
    exact (M n x hx).trans_lt hn
  have : Tendsto (fun i : ℕ => ∫ x : α in s, φ i x • g x ∂μ) atTop (𝓝 (g x₀)) := by
    have B : Tendsto (fun i ↦ ∫ (x : α) in s, φ i x ∂μ) atTop (𝓝 1) :=
      tendsto_const_nhds.congr (fun n ↦ (hiφ n).symm)
    have C : ∀ᶠ (i : ℕ) in atTop, AEStronglyMeasurable (fun x ↦ φ i x) (μ.restrict s) := by
      apply Eventually.of_forall (fun n ↦ ((I n).const_mul _).aestronglyMeasurable)
    exact tendsto_setIntegral_peak_smul_of_integrableOn_of_tendsto hs.measurableSet
      hs.measurableSet (Subset.rfl) (self_mem_nhdsWithin)
      hs.measure_lt_top.ne (Eventually.of_forall hnφ) A B C hmg hcg
  /-
    α : Type u_1
    E : Type u_2
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : BorelSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    g : α → E
    x₀ : α
    s : Set α
    inst✝² : CompleteSpace E
    inst✝¹ : TopologicalSpace.MetrizableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hs : IsCompact s
    hμ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → LT.lt 0 (μ (Inter.inter u …
    c : α → Real
    hc : ContinuousOn c s
    h'c : ∀ (y : α), Membership.mem s y → Ne y x₀ → LT.lt (c y) (c x₀)
    hnc : ∀ (x : α), Membership.mem s x → LE.le 0 (c x)
    hnc₀ : LT.lt 0 (c x₀)
    h₀ : Membership.mem s x₀
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : ContinuousWithinAt g s x₀
    φ : Nat → α → Real := fun n x => HMul.hMul (Inv.inv (MeasureTheory.integral (μ …
    hnφ : ∀ (n : Nat) (x : α), Membership.mem s x → LE.le 0 (φ n x)
    I : ∀ (n : Nat), MeasureTheory.IntegrableOn (fun x => HPow.hPow (c x) n) s μ
    J : ∀ (n : Nat), (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 fun x => HPo …
    P : ∀ (n : Nat), LT.lt 0 (MeasureTheory.integral (μ.restrict s) fun x => HPow. …
    hiφ : ∀ (n : Nat), Eq (MeasureTheory.integral (μ.restrict s) fun x => φ n x) 1
    A : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 Fil …
    this : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv (MeasureTheory.integral (μ.res …
  -/
  convert this
  /-
    case h.e'_3.h
    α : Type u_1
    E : Type u_2
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : BorelSpace α
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    g : α → E
    x₀ : α
    s : Set α
    inst✝² : CompleteSpace E
    inst✝¹ : TopologicalSpace.MetrizableSpace α
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hs : IsCompact s
    hμ : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → LT.lt 0 (μ (Inter.inter u …
    c : α → Real
    hc : ContinuousOn c s
    h'c : ∀ (y : α), Membership.mem s y → Ne y x₀ → LT.lt (c y) (c x₀)
    hnc : ∀ (x : α), Membership.mem s x → LE.le 0 (c x)
    hnc₀ : LT.lt 0 (c x₀)
    h₀ : Membership.mem s x₀
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : ContinuousWithinAt g s x₀
    φ : Nat → α → Real := fun n x => HMul.hMul (Inv.inv (MeasureTheory.integral (μ …
    hnφ : ∀ (n : Nat) (x : α), Membership.mem s x → LE.le 0 (φ n x)
    I : ∀ (n : Nat), MeasureTheory.IntegrableOn (fun x => HPow.hPow (c x) n) s μ
    J : ∀ (n : Nat), (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 fun x => HPo …
    P : ∀ (n : Nat), LT.lt 0 (MeasureTheory.integral (μ.restrict s) fun x => HPow. …
    hiφ : ∀ (n : Nat), Eq (MeasureTheory.integral (μ.restrict s) fun x => φ n x) 1
    A : ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → TendstoUniformlyOn φ 0 Fil …
    this : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict s) fun x => …
    x✝ : Nat
    ⊢ Eq (HSMul.hSMul (Inv.inv (MeasureTheory.integral (μ.restrict s) fun x => HPo …
  -/
  simp_rw [φ, ← smul_smul, integral_smul]
  /-
    🎉 no goals
  -/


/-- If a continuous function `c` realizes its maximum at a unique point `x₀` in a compact set `s`,
then the sequence of functions `(c x) ^ n / ∫ (c x) ^ n` is a sequence of peak functions
concentrating around `x₀`. Therefore, `∫ (c x) ^ n * g / ∫ (c x) ^ n` converges to `g x₀` if `g` is
integrable on `s` and continuous at `x₀`.

Version assuming that `μ` gives positive mass to all open sets.
For a less precise but more usable version, see
`tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_continuousOn`.
-/
theorem tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_integrableOn
    [MetrizableSpace α] [IsLocallyFiniteMeasure μ] [IsOpenPosMeasure μ] (hs : IsCompact s)
    {c : α → ℝ} (hc : ContinuousOn c s) (h'c : ∀ y ∈ s, y ≠ x₀ → c y < c x₀)
    (hnc : ∀ x ∈ s, 0 ≤ c x) (hnc₀ : 0 < c x₀) (h₀ : x₀ ∈ closure (interior s))
    (hmg : IntegrableOn g s μ) (hcg : ContinuousWithinAt g s x₀) :
    Tendsto (fun n : ℕ => (∫ x in s, c x ^ n ∂μ)⁻¹ • ∫ x in s, c x ^ n • g x ∂μ) atTop
      (𝓝 (g x₀)) := by
  /-
    α : Type u_1
    E : Type u_2
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    g : α → E
    x₀ : α
    s : Set α
    inst✝³ : CompleteSpace E
    inst✝² : TopologicalSpace.MetrizableSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    hs : IsCompact s
    c : α → Real
    hc : ContinuousOn c s
    h'c : ∀ (y : α), Membership.mem s y → Ne y x₀ → LT.lt (c y) (c x₀)
    hnc : ∀ (x : α), Membership.mem s x → LE.le 0 (c x)
    hnc₀ : LT.lt 0 (c x₀)
    h₀ : Membership.mem (closure (interior s)) x₀
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : ContinuousWithinAt g s x₀
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv (MeasureTheory.integral (μ.res …
  -/
  have : x₀ ∈ s := by rw [← hs.isClosed.closure_eq]; exact closure_mono interior_subset h₀
  apply
    tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_measure_nhdsWithin_pos hs _ hc
      h'c hnc hnc₀ this hmg hcg
  /-
    α : Type u_1
    E : Type u_2
    hm : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁷ : TopologicalSpace α
    inst✝⁶ : BorelSpace α
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    g : α → E
    x₀ : α
    s : Set α
    inst✝³ : CompleteSpace E
    inst✝² : TopologicalSpace.MetrizableSpace α
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : μ.IsOpenPosMeasure
    hs : IsCompact s
    c : α → Real
    hc : ContinuousOn c s
    h'c : ∀ (y : α), Membership.mem s y → Ne y x₀ → LT.lt (c y) (c x₀)
    hnc : ∀ (x : α), Membership.mem s x → LE.le 0 (c x)
    hnc₀ : LT.lt 0 (c x₀)
    h₀ : Membership.mem (closure (interior s)) x₀
    hmg : MeasureTheory.IntegrableOn g s μ
    hcg : ContinuousWithinAt g s x₀
    this : Membership.mem s x₀
    ⊢ ∀ (u : Set α), IsOpen u → Membership.mem u x₀ → LT.lt 0 (μ (Inter.inter u s))
  -/
  intro u u_open x₀_u
  calc
    0 < μ (u ∩ interior s) :=
      (u_open.inter isOpen_interior).measure_pos μ (_root_.mem_closure_iff.1 h₀ u u_open x₀_u)
    _ ≤ μ (u ∩ s) := by gcongr; apply interior_subset


/-- If a continuous function `c` realizes its maximum at a unique point `x₀` in a compact set `s`,
then the sequence of functions `(c x) ^ n / ∫ (c x) ^ n` is a sequence of peak functions
concentrating around `x₀`. Therefore, `∫ (c x) ^ n * g / ∫ (c x) ^ n` converges to `g x₀` if `g` is
continuous on `s`. -/
theorem tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_continuousOn
    [MetrizableSpace α] [IsLocallyFiniteMeasure μ] [IsOpenPosMeasure μ] (hs : IsCompact s)
    {c : α → ℝ} (hc : ContinuousOn c s) (h'c : ∀ y ∈ s, y ≠ x₀ → c y < c x₀)
    (hnc : ∀ x ∈ s, 0 ≤ c x) (hnc₀ : 0 < c x₀) (h₀ : x₀ ∈ closure (interior s))
    (hmg : ContinuousOn g s) :
    Tendsto (fun n : ℕ => (∫ x in s, c x ^ n ∂μ)⁻¹ • ∫ x in s, c x ^ n • g x ∂μ) atTop (𝓝 (g x₀)) :=
                       /-
                         α : Type u_1
                         E : Type u_2
                         hm : MeasurableSpace α
                         μ : MeasureTheory.Measure α
                         inst✝⁷ : TopologicalSpace α
                         inst✝⁶ : BorelSpace α
                         inst✝⁵ : NormedAddCommGroup E
                         inst✝⁴ : NormedSpace Real E
                         g : α → E
                         x₀ : α
                         s : Set α
                         inst✝³ : CompleteSpace E
                         inst✝² : TopologicalSpace.MetrizableSpace α
                         inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
                         inst✝ : μ.IsOpenPosMeasure
                         hs : IsCompact s
                         c : α → Real
                         hc : ContinuousOn c s
                         h'c : ∀ (y : α), Membership.mem s y → Ne y x₀ → LT.lt (c y) (c x₀)
                         hnc : ∀ (x : α), Membership.mem s x → LE.le 0 (c x)
                         hnc₀ : LT.lt 0 (c x₀)
                         h₀ : Membership.mem (closure (interior s)) x₀
                         hmg : ContinuousOn g s
                         ⊢ Membership.mem s x₀
                       -/
  haveI : x₀ ∈ s := by rw [← hs.isClosed.closure_eq]; exact closure_mono interior_subset h₀
                                                      /-
                                                        🎉 no goals
                                                      -/
  tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_integrableOn hs hc h'c hnc hnc₀ h₀
    (hmg.integrableOn_compact hs) (hmg x₀ this)


/-- Consider a nonnegative function `φ` with integral one, decaying quickly enough at infinity.
Then suitable renormalizations of `φ` form a sequence of peak functions around the origin:
`∫ (c ^ d * φ (c • x)) • g x` converges to `g 0` as `c → ∞` if `g` is continuous at `0`
and integrable. -/
theorem tendsto_integral_comp_smul_smul_of_integrable
    {φ : F → ℝ} (hφ : ∀ x, 0 ≤ φ x) (h'φ : ∫ x, φ x ∂μ = 1)
    (h : Tendsto (fun x ↦ ‖x‖ ^ finrank ℝ F * φ x) (cobounded F) (𝓝 0))
    {g : F → E} (hg : Integrable g μ) (h'g : ContinuousAt g 0) :
    Tendsto (fun (c : ℝ) ↦ ∫ x, (c ^ (finrank ℝ F) * φ (c • x)) • g x ∂μ) atTop (𝓝 (g 0)) := by
  /-
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g 0
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul (HMul …
  -/
  have I : Integrable φ μ := integrable_of_integral_eq_one h'φ
  /-
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g 0
    I : MeasureTheory.Integrable φ μ
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul (HMul …
  -/
  apply tendsto_integral_peak_smul_of_integrable_of_tendsto (t := closedBall 0 1) (x₀ := 0)
    /-
      case ht
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ MeasurableSet (Metric.closedBall 0 1)
    -/
  · exact isClosed_ball.measurableSet
    /-
      🎉 no goals
    -/
    /-
      case h'ts
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ Membership.mem (nhds 0) (Metric.closedBall 0 1)
    -/
  · exact closedBall_mem_nhds _ zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case h't
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ Ne (μ (Metric.closedBall 0 1)) Top.top
    -/
  · exact (isCompact_closedBall 0 1).measure_ne_top
    /-
      🎉 no goals
    -/
    /-
      case hnφ
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ Filter.Eventually (fun i => ∀ (x : F), LE.le 0 (HMul.hMul (HPow.hPow i (Modu …
    -/
  · filter_upwards [Ici_mem_atTop 0] with c (hc : 0 ≤ c) x using mul_nonneg (by positivity) (hφ _)
    /-
      🎉 no goals
    -/
    /-
      case hlφ
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ ∀ (u : Set F), IsOpen u → Membership.mem u 0 → TendstoUniformlyOn (fun i x = …
    -/
  · intro u u_open hu
    /-
      case hlφ
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      u : Set F
      u_open : IsOpen u
      hu : Membership.mem u 0
      ⊢ TendstoUniformlyOn (fun i x => HMul.hMul (HPow.hPow i (Module.finrank Real F …
    -/
    apply tendstoUniformlyOn_iff.2 (fun ε εpos ↦ ?_)
    /-
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      u : Set F
      u_open : IsOpen u
      hu : Membership.mem u 0
      ε : Real
      εpos : GT.gt ε 0
      ⊢ Filter.Eventually (fun n => ∀ (x : F), Membership.mem (HasCompl.compl u) x → …
    -/
    obtain ⟨δ, δpos, h'u⟩ : ∃ δ > 0, ball 0 δ ⊆ u := Metric.isOpen_iff.1 u_open _ hu
    obtain ⟨M, Mpos, hM⟩ : ∃ M > 0, ∀ ⦃x : F⦄, x ∈ (closedBall 0 M)ᶜ →
        ‖x‖ ^ finrank ℝ F * φ x < δ ^ finrank ℝ F * ε := by
      rcases (hasBasis_cobounded_compl_closedBall (0 : F)).eventually_iff.1
        ((tendsto_order.1 h).2 (δ ^ finrank ℝ F * ε) (by positivity)) with ⟨M, -, hM⟩
      refine ⟨max M 1, zero_lt_one.trans_le (le_max_right _ _), fun x hx ↦ hM ?_⟩
      simp only [mem_compl_iff, mem_closedBall, dist_zero_right, le_max_iff, not_or, not_le] at hx
      simpa using hx.1
    /-
      case intro.intro.intro.intro
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      u : Set F
      u_open : IsOpen u
      hu : Membership.mem u 0
      ε : Real
      εpos : GT.gt ε 0
      δ : Real
      δpos : GT.gt δ 0
      h'u : HasSubset.Subset (Metric.ball 0 δ) u
      M : Real
      Mpos : GT.gt M 0
      hM : ∀ ⦃x : F⦄, Membership.mem (HasCompl.compl (Metric.closedBall 0 M)) x → LT …
      ⊢ Filter.Eventually (fun n => ∀ (x : F), Membership.mem (HasCompl.compl u) x → …
    -/
    filter_upwards [Ioi_mem_atTop (M / δ)] with c (hc : M / δ < c) x hx
    /-
      case h
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      u : Set F
      u_open : IsOpen u
      hu : Membership.mem u 0
      ε : Real
      εpos : GT.gt ε 0
      δ : Real
      δpos : GT.gt δ 0
      h'u : HasSubset.Subset (Metric.ball 0 δ) u
      M : Real
      Mpos : GT.gt M 0
      hM : ∀ ⦃x : F⦄, Membership.mem (HasCompl.compl (Metric.closedBall 0 M)) x → LT …
      c : Real
      hc : LT.lt (HDiv.hDiv M δ) c
      x : F
      hx : Membership.mem (HasCompl.compl u) x
      ⊢ LT.lt (Dist.dist (0 x) (HMul.hMul (HPow.hPow c (Module.finrank Real F)) (φ ( …
    -/
    have cpos : 0 < c := lt_trans (by positivity) hc
    /-
      case h
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      u : Set F
      u_open : IsOpen u
      hu : Membership.mem u 0
      ε : Real
      εpos : GT.gt ε 0
      δ : Real
      δpos : GT.gt δ 0
      h'u : HasSubset.Subset (Metric.ball 0 δ) u
      M : Real
      Mpos : GT.gt M 0
      hM : ∀ ⦃x : F⦄, Membership.mem (HasCompl.compl (Metric.closedBall 0 M)) x → LT …
      c : Real
      hc : LT.lt (HDiv.hDiv M δ) c
      x : F
      hx : Membership.mem (HasCompl.compl u) x
      cpos : LT.lt 0 c
      ⊢ LT.lt (Dist.dist (0 x) (HMul.hMul (HPow.hPow c (Module.finrank Real F)) (φ ( …
    -/
    suffices c ^ finrank ℝ F * φ (c • x) < ε by simpa [abs_of_nonneg (hφ _), abs_of_nonneg cpos.le]
    have hδx : δ ≤ ‖x‖ := by
      have : x ∈ (ball 0 δ)ᶜ := fun h ↦ hx (h'u h)
      simpa only [mem_compl_iff, mem_ball, dist_zero_right, not_lt]
    suffices δ ^ finrank ℝ F * (c ^ finrank ℝ F * φ (c • x)) < δ ^ finrank ℝ F * ε by
      rwa [mul_lt_mul_iff_of_pos_left (by positivity)] at this
    calc
      δ ^ finrank ℝ F * (c ^ finrank ℝ F * φ (c • x))
      _ ≤ ‖x‖ ^ finrank ℝ F * (c ^ finrank ℝ F * φ (c • x)) := by
        gcongr; exact mul_nonneg (by positivity) (hφ _)
      _ = ‖c • x‖ ^ finrank ℝ F * φ (c • x) := by
        simp [norm_smul, abs_of_pos cpos, mul_pow]; ring
      _ < δ ^ finrank ℝ F * ε := by
        apply hM
        rw [div_lt_iff₀ δpos] at hc
        simp only [mem_compl_iff, mem_closedBall, dist_zero_right, norm_smul, Real.norm_eq_abs,
          abs_of_nonneg cpos.le, not_le, gt_iff_lt]
        exact hc.trans_le (by gcongr)
  · have : Tendsto (fun c ↦ ∫ (x : F) in closedBall 0 c, φ x ∂μ) atTop (𝓝 1) := by
      rw [← h'φ]
      exact (aecover_closedBall tendsto_id).integral_tendsto_of_countably_generated I
    /-
      case hiφ
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      this : Filter.Tendsto (fun c => MeasureTheory.integral (μ.restrict (Metric.clo …
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (Metric.closedBa …
    -/
    apply this.congr'
    /-
      case hiφ
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      this : Filter.Tendsto (fun c => MeasureTheory.integral (μ.restrict (Metric.clo …
      ⊢ Filter.atTop.EventuallyEq (fun c => MeasureTheory.integral (μ.restrict (Metr …
    -/
    filter_upwards [Ioi_mem_atTop 0] with c (hc : 0 < c)
    rw [integral_mul_left, setIntegral_comp_smul_of_pos _ _ _ hc, smul_eq_mul, ← mul_assoc,
      mul_inv_cancel₀ (by positivity), _root_.smul_closedBall _ _ zero_le_one]
    /-
      case h
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      this : Filter.Tendsto (fun c => MeasureTheory.integral (μ.restrict (Metric.clo …
      c : Real
      hc : LT.lt 0 c
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Metric.closedBall 0 c)) fun x => φ x …
    -/
    simp [abs_of_nonneg hc.le]
    /-
      🎉 no goals
    -/
    /-
      case h'iφ
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ Filter.Eventually (fun i => MeasureTheory.AEStronglyMeasurable (fun x => HMu …
    -/
  · filter_upwards [Ioi_mem_atTop 0] with c (hc : 0 < c)
    /-
      case h
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      c : Real
      hc : LT.lt 0 c
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HMul.hMul (HPow.hPow c (Module. …
    -/
    exact (I.comp_smul hc.ne').aestronglyMeasurable.const_mul _
    /-
      🎉 no goals
    -/
    /-
      case hmg
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ MeasureTheory.Integrable g μ
    -/
  · exact hg
    /-
      🎉 no goals
    -/
    /-
      case hcg
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace Real F
      inst✝³ : FiniteDimensional Real F
      inst✝² : MeasurableSpace F
      inst✝¹ : BorelSpace F
      μ : MeasureTheory.Measure F
      inst✝ : μ.IsAddHaarMeasure
      φ : F → Real
      hφ : ∀ (x : F), LE.le 0 (φ x)
      h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
      h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
      g : F → E
      hg : MeasureTheory.Integrable g μ
      h'g : ContinuousAt g 0
      I : MeasureTheory.Integrable φ μ
      ⊢ Filter.Tendsto g (nhds 0) (nhds (g 0))
    -/
  · exact h'g
    /-
      🎉 no goals
    -/


/-- Consider a nonnegative function `φ` with integral one, decaying quickly enough at infinity.
Then suitable renormalizations of `φ` form a sequence of peak functions around any point:
`∫ (c ^ d * φ (c • (x₀ - x)) • g x` converges to `g x₀` as `c → ∞` if `g` is continuous at `x₀`
and integrable. -/
theorem tendsto_integral_comp_smul_smul_of_integrable'
    {φ : F → ℝ} (hφ : ∀ x, 0 ≤ φ x) (h'φ : ∫ x, φ x ∂μ = 1)
    (h : Tendsto (fun x ↦ ‖x‖ ^ finrank ℝ F * φ x) (cobounded F) (𝓝 0))
    {g : F → E} {x₀ : F} (hg : Integrable g μ) (h'g : ContinuousAt g x₀) :
    Tendsto (fun (c : ℝ) ↦ ∫ x, (c ^ (finrank ℝ F) * φ (c • (x₀ - x))) • g x ∂μ)
      atTop (𝓝 (g x₀)) := by
  /-
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    x₀ : F
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g x₀
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul (HMul …
  -/
  let f := fun x ↦ g (x₀ - x)
  /-
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    x₀ : F
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g x₀
    f : F → E := fun x => g (HSub.hSub x₀ x)
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul (HMul …
  -/
  have If : Integrable f μ := by simpa [f, sub_eq_add_neg] using (hg.comp_add_left x₀).comp_neg
  have : Tendsto (fun (c : ℝ) ↦ ∫ x, (c ^ (finrank ℝ F) * φ (c • x)) • f x ∂μ)
      atTop (𝓝 (f 0)) := by
    apply tendsto_integral_comp_smul_smul_of_integrable hφ h'φ h If
    have A : ContinuousAt g (x₀ - 0) := by simpa using h'g
    exact A.comp <| by fun_prop
  /-
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    x₀ : F
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g x₀
    f : F → E := fun x => g (HSub.hSub x₀ x)
    If : MeasureTheory.Integrable f μ
    this : Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul  …
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul (HMul …
  -/
  simp only [f, sub_zero] at this
  /-
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    x₀ : F
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g x₀
    f : F → E := fun x => g (HSub.hSub x₀ x)
    If : MeasureTheory.Integrable f μ
    this : Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul  …
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul (HMul …
  -/
  convert this using 2 with c
  conv_rhs => rw [← integral_add_left_eq_self x₀ (μ := μ)
    (f := fun x ↦ (c ^ finrank ℝ F * φ (c • x)) • g (x₀ - x)), ← integral_neg_eq_self]
  /-
    case h.e'_3.h
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    μ : MeasureTheory.Measure F
    inst✝ : μ.IsAddHaarMeasure
    φ : F → Real
    hφ : ∀ (x : F), LE.le 0 (φ x)
    h'φ : Eq (MeasureTheory.integral μ fun x => φ x) 1
    h : Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Norm.norm x) (Module.finran …
    g : F → E
    x₀ : F
    hg : MeasureTheory.Integrable g μ
    h'g : ContinuousAt g x₀
    f : F → E := fun x => g (HSub.hSub x₀ x)
    If : MeasureTheory.Integrable f μ
    this : Filter.Tendsto (fun c => MeasureTheory.integral μ fun x => HSMul.hSMul  …
    c : Real
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (HMul.hMul (HPow.hPow c (M …
  -/
  simp [smul_sub, sub_eq_add_neg]
  /-
    🎉 no goals
  -/

