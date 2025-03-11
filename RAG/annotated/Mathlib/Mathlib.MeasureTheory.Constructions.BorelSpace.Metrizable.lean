/-- A limit (over a general filter) of measurable functions valued in a (pseudo) metrizable space is
measurable. -/
theorem measurable_of_tendsto_metrizable' {ι} {f : ι → α → β} {g : α → β} (u : Filter ι) [NeBot u]
    [IsCountablyGenerated u] (hf : ∀ i, Measurable (f i)) (lim : Tendsto f u (𝓝 g)) :
    Measurable g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    ι : Type u_3
    f : ι → α → β
    g : α → β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    ⊢ Measurable g
  -/
  letI : PseudoMetricSpace β := pseudoMetrizableSpacePseudoMetric β
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    ι : Type u_3
    f : ι → α → β
    g : α → β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ Measurable g
  -/
  apply measurable_of_isClosed'
  /-
    case hf
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    ι : Type u_3
    f : ι → α → β
    g : α → β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ ∀ (s : Set β), IsClosed s → s.Nonempty → Ne s Set.univ → MeasurableSet (Set. …
  -/
  intro s h1s h2s h3s
  have : Measurable fun x => infNndist (g x) s := by
    suffices Tendsto (fun i x => infNndist (f i x) s) u (𝓝 fun x => infNndist (g x) s) from
      NNReal.measurable_of_tendsto' u (fun i => (hf i).infNndist) this
    rw [tendsto_pi_nhds] at lim ⊢
    intro x
    exact ((continuous_infNndist_pt s).tendsto (g x)).comp (lim x)
  have h4s : g ⁻¹' s = (fun x => infNndist (g x) s) ⁻¹' {0} := by
    ext x
    simp [h1s, ← h1s.mem_iff_infDist_zero h2s, ← NNReal.coe_eq_zero]
  /-
    case hf
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    ι : Type u_3
    f : ι → α → β
    g : α → β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    this✝ : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMet …
    s : Set β
    h1s : IsClosed s
    h2s : s.Nonempty
    h3s : Ne s Set.univ
    this : Measurable fun x => Metric.infNndist (g x) s
    h4s : Eq (Set.preimage g s) (Set.preimage (fun x => Metric.infNndist (g x) s)  …
    ⊢ MeasurableSet (Set.preimage g s)
  -/
  rw [h4s]
  /-
    case hf
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝³ : MeasurableSpace β
    inst✝² : BorelSpace β
    ι : Type u_3
    f : ι → α → β
    g : α → β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    this✝ : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMet …
    s : Set β
    h1s : IsClosed s
    h2s : s.Nonempty
    h3s : Ne s Set.univ
    this : Measurable fun x => Metric.infNndist (g x) s
    h4s : Eq (Set.preimage g s) (Set.preimage (fun x => Metric.infNndist (g x) s)  …
    ⊢ MeasurableSet (Set.preimage (fun x => Metric.infNndist (g x) s) (Singleton.s …
  -/
  exact this (measurableSet_singleton 0)
  /-
    🎉 no goals
  -/


/-- A sequential limit of measurable functions valued in a (pseudo) metrizable space is
measurable. -/
theorem measurable_of_tendsto_metrizable {f : ℕ → α → β} {g : α → β} (hf : ∀ i, Measurable (f i))
    (lim : Tendsto f atTop (𝓝 g)) : Measurable g :=
  measurable_of_tendsto_metrizable' atTop hf lim


theorem aemeasurable_of_tendsto_metrizable_ae {ι} {μ : Measure α} {f : ι → α → β} {g : α → β}
    (u : Filter ι) [hu : NeBot u] [IsCountablyGenerated u] (hf : ∀ n, AEMeasurable (f n) μ)
    (h_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) u (𝓝 (g x))) : AEMeasurable g μ := by
  classical
  rcases u.exists_seq_tendsto with ⟨v, hv⟩
  have h'f : ∀ n, AEMeasurable (f (v n)) μ := fun n => hf (v n)
  set p : α → (ℕ → β) → Prop := fun x f' => Tendsto (fun n => f' n) atTop (𝓝 (g x))
  have hp : ∀ᵐ x ∂μ, p x fun n => f (v n) x := by
    filter_upwards [h_tendsto] with x hx using hx.comp hv
  set aeSeqLim := fun x => ite (x ∈ aeSeqSet h'f p) (g x) (⟨f (v 0) x⟩ : Nonempty β).some
  refine
    ⟨aeSeqLim,
      measurable_of_tendsto_metrizable' atTop (aeSeq.measurable h'f p)
        (tendsto_pi_nhds.mpr fun x => ?_),
      ?_⟩
  · simp_rw [aeSeqLim, aeSeq]
    split_ifs with hx
    · simp_rw [aeSeq.mk_eq_fun_of_mem_aeSeqSet h'f hx]
      exact @aeSeq.fun_prop_of_mem_aeSeqSet _ α β _ _ _ _ _ h'f x hx
    · exact tendsto_const_nhds
  · exact
      (ite_ae_eq_of_measure_compl_zero g (fun x => (⟨f (v 0) x⟩ : Nonempty β).some) (aeSeqSet h'f p)
          (aeSeq.measure_compl_aeSeqSet_eq_zero h'f hp)).symm


theorem aemeasurable_of_tendsto_metrizable_ae' {μ : Measure α} {f : ℕ → α → β} {g : α → β}
    (hf : ∀ n, AEMeasurable (f n) μ)
    (h_ae_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) : AEMeasurable g μ :=
  aemeasurable_of_tendsto_metrizable_ae atTop hf h_ae_tendsto


theorem aemeasurable_of_unif_approx {β} [MeasurableSpace β] [PseudoMetricSpace β] [BorelSpace β]
    {μ : Measure α} {g : α → β}
    (hf : ∀ ε > (0 : ℝ), ∃ f : α → β, AEMeasurable f μ ∧ ∀ᵐ x ∂μ, dist (f x) (g x) ≤ ε) :
    AEMeasurable g μ := by
  obtain ⟨u, -, u_pos, u_lim⟩ :
    ∃ u : ℕ → ℝ, StrictAnti u ∧ (∀ n : ℕ, 0 < u n) ∧ Tendsto u atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_3
    inst✝² : MeasurableSpace β
    inst✝¹ : PseudoMetricSpace β
    inst✝ : BorelSpace β
    μ : MeasureTheory.Measure α
    g : α → β
    hf : ∀ (ε : Real), GT.gt ε 0 → Exists fun f => And (AEMeasurable f μ) (Filter. …
    u : Nat → Real
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ AEMeasurable g μ
  -/
  choose f Hf using fun n : ℕ => hf (u n) (u_pos n)
  have : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x)) := by
    have : ∀ᵐ x ∂μ, ∀ n, dist (f n x) (g x) ≤ u n := ae_all_iff.2 fun n => (Hf n).2
    filter_upwards [this]
    intro x hx
    rw [tendsto_iff_dist_tendsto_zero]
    exact squeeze_zero (fun n => dist_nonneg) hx u_lim
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : MeasurableSpace α
    β : Type u_3
    inst✝² : MeasurableSpace β
    inst✝¹ : PseudoMetricSpace β
    inst✝ : BorelSpace β
    μ : MeasureTheory.Measure α
    g : α → β
    hf : ∀ (ε : Real), GT.gt ε 0 → Exists fun f => And (AEMeasurable f μ) (Filter. …
    u : Nat → Real
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    f : Nat → α → β
    Hf : ∀ (n : Nat), And (AEMeasurable (f n) μ) (Filter.Eventually (fun x => LE.l …
    this : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTo …
    ⊢ AEMeasurable g μ
  -/
  exact aemeasurable_of_tendsto_metrizable_ae' (fun n => (Hf n).1) this
  /-
    🎉 no goals
  -/


theorem measurable_of_tendsto_metrizable_ae {μ : Measure α} [μ.IsComplete] {f : ℕ → α → β}
    {g : α → β} (hf : ∀ n, Measurable (f n))
    (h_ae_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x))) : Measurable g :=
  aemeasurable_iff_measurable.mp
    (aemeasurable_of_tendsto_metrizable_ae' (fun i => (hf i).aemeasurable) h_ae_tendsto)


theorem measurable_limit_of_tendsto_metrizable_ae {ι} [Countable ι] [Nonempty ι] {μ : Measure α}
    {f : ι → α → β} {L : Filter ι} [L.IsCountablyGenerated] (hf : ∀ n, AEMeasurable (f n) μ)
    (h_ae_tendsto : ∀ᵐ x ∂μ, ∃ l : β, Tendsto (fun n => f n x) L (𝓝 l)) :
    ∃ f_lim : α → β, Measurable f_lim ∧ ∀ᵐ x ∂μ, Tendsto (fun n => f n x) L (𝓝 (f_lim x)) := by
  classical
  inhabit ι
  rcases eq_or_neBot L with (rfl | hL)
  · exact ⟨(hf default).mk _, (hf default).measurable_mk, Eventually.of_forall fun x => tendsto_bot⟩
  let p : α → (ι → β) → Prop := fun x f' => ∃ l : β, Tendsto (fun n => f' n) L (𝓝 l)
  have hp_mem : ∀ x ∈ aeSeqSet hf p, p x fun n => f n x := fun x hx =>
    aeSeq.fun_prop_of_mem_aeSeqSet hf hx
  have h_ae_eq : ∀ᵐ x ∂μ, ∀ n, aeSeq hf p n x = f n x := aeSeq.aeSeq_eq_fun_ae hf h_ae_tendsto
  set f_lim : α → β := fun x => dite (x ∈ aeSeqSet hf p) (fun h => (hp_mem x h).choose)
    fun _ => (⟨f default x⟩ : Nonempty β).some
  have hf_lim : ∀ x, Tendsto (fun n => aeSeq hf p n x) L (𝓝 (f_lim x)) := by
    intro x
    simp only [aeSeq, f_lim]
    split_ifs with h
    · refine (hp_mem x h).choose_spec.congr fun n => ?_
      exact (aeSeq.mk_eq_fun_of_mem_aeSeqSet hf h n).symm
    · exact tendsto_const_nhds
  have h_ae_tendsto_f_lim : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) L (𝓝 (f_lim x)) :=
    h_ae_eq.mono fun x hx => (hf_lim x).congr hx
  have h_f_lim_meas : Measurable f_lim :=
    measurable_of_tendsto_metrizable' L (aeSeq.measurable hf p)
      (tendsto_pi_nhds.mpr fun x => hf_lim x)
  exact ⟨f_lim, h_f_lim_meas, h_ae_tendsto_f_lim⟩


/-- If the indicator functions of measurable sets `Aᵢ` converge to the indicator function of
a set `A` along a nontrivial countably generated filter, then `A` is also measurable. -/
lemma measurableSet_of_tendsto_indicator [NeBot L] (As_mble : ∀ i, MeasurableSet (As i))
    (h_lim : ∀ x, ∀ᶠ i in L, x ∈ As i ↔ x ∈ A) :
    MeasurableSet A := by
  /-
    α : Type u_3
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_4
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    inst✝ : L.NeBot
    As_mble : ∀ (i : ι), MeasurableSet (As i)
    h_lim : ∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) ( …
    ⊢ MeasurableSet A
  -/
  simp_rw [← measurable_indicator_const_iff (1 : ℝ≥0∞)] at As_mble ⊢
  exact ENNReal.measurable_of_tendsto' L As_mble
    ((tendsto_indicator_const_iff_forall_eventually L (1 : ℝ≥0∞)).mpr h_lim)


/-- If the indicator functions of a.e.-measurable sets `Aᵢ` converge a.e. to the indicator function
of a set `A` along a nontrivial countably generated filter, then `A` is also a.e.-measurable. -/
lemma nullMeasurableSet_of_tendsto_indicator [NeBot L] {μ : Measure α}
    (As_mble : ∀ i, NullMeasurableSet (As i) μ)
    (h_lim : ∀ᵐ x ∂μ, ∀ᶠ i in L, x ∈ As i ↔ x ∈ A) :
    NullMeasurableSet A μ := by
  /-
    α : Type u_3
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_4
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    inst✝ : L.NeBot
    μ : MeasureTheory.Measure α
    As_mble : ∀ (i : ι), MeasureTheory.NullMeasurableSet (As i) μ
    h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
    ⊢ MeasureTheory.NullMeasurableSet A μ
  -/
  simp_rw [← aemeasurable_indicator_const_iff (1 : ℝ≥0∞)] at As_mble ⊢
  /-
    α : Type u_3
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_4
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    inst✝ : L.NeBot
    μ : MeasureTheory.Measure α
    h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
    As_mble : ∀ (i : ι), AEMeasurable ((As i).indicator fun x => 1) μ
    ⊢ AEMeasurable (A.indicator fun x => 1) μ
  -/
  apply aemeasurable_of_tendsto_metrizable_ae L As_mble
  /-
    α : Type u_3
    inst✝² : MeasurableSpace α
    A : Set α
    ι : Type u_4
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    As : ι → Set α
    inst✝ : L.NeBot
    μ : MeasureTheory.Measure α
    h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
    As_mble : ∀ (i : ι), AEMeasurable ((As i).indicator fun x => 1) μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => (As n).indicator (fun x …
  -/
  simpa [tendsto_indicator_const_apply_iff_eventually] using h_lim
  /-
    🎉 no goals
  -/


