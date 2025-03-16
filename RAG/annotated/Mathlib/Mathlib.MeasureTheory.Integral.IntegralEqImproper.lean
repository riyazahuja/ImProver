/-- A sequence `φ` of subsets of `α` is a `MeasureTheory.AECover` w.r.t. a measure `μ` and a filter
    `l` if almost every point (w.r.t. `μ`) of `α` eventually belongs to `φ n` (w.r.t. `l`), and if
    each `φ n` is measurable.  This definition is a technical way to avoid duplicating a lot of
    proofs.  It should be thought of as a sufficient condition for being able to interpret
    `∫ x, f x ∂μ` (if it exists) as the limit of `∫ x in φ n, f x ∂μ` as `n` tends to `l`.

    See for example `MeasureTheory.AECover.lintegral_tendsto_of_countably_generated`,
    `MeasureTheory.AECover.integrable_of_integral_norm_tendsto` and
    `MeasureTheory.AECover.integral_tendsto_of_countably_generated`. -/
structure AECover (φ : ι → Set α) : Prop where
  ae_eventually_mem : ∀ᵐ x ∂μ, ∀ᶠ i in l, x ∈ φ i
  protected measurableSet : ∀ i, MeasurableSet <| φ i


/-- Elementwise intersection of two `AECover`s is an `AECover`. -/
theorem inter {φ ψ : ι → Set α} (hφ : AECover μ l φ) (hψ : AECover μ l ψ) :
    AECover μ l (fun i ↦ φ i ∩ ψ i) where
  ae_eventually_mem := hψ.1.mp <| hφ.1.mono fun _ ↦ Eventually.and
  measurableSet _ := (hφ.2 _).inter (hψ.2 _)


theorem superset {φ ψ : ι → Set α} (hφ : AECover μ l φ) (hsub : ∀ i, φ i ⊆ ψ i)
    (hmeas : ∀ i, MeasurableSet (ψ i)) : AECover μ l ψ :=
  ⟨hφ.1.mono fun _x hx ↦ hx.mono fun i hi ↦ hsub i hi, hmeas⟩


theorem mono_ac {ν : Measure α} {φ : ι → Set α} (hφ : AECover μ l φ) (hle : ν ≪ μ) :
    AECover ν l φ := ⟨hle hφ.1, hφ.2⟩


theorem mono {ν : Measure α} {φ : ι → Set α} (hφ : AECover μ l φ) (hle : ν ≤ μ) :
    AECover ν l φ := hφ.mono_ac hle.absolutelyContinuous


theorem aecover_ball {x : α} {r : ι → ℝ} (hr : Tendsto r l atTop) :
    AECover μ l (fun i ↦ Metric.ball x (r i)) where
  measurableSet _ := Metric.isOpen_ball.measurableSet
  ae_eventually_mem := by
    /-
      α : Type u_1
      ι : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      l : Filter ι
      inst✝¹ : PseudoMetricSpace α
      inst✝ : OpensMeasurableSpace α
      x : α
      r : ι → Real
      hr : Filter.Tendsto r l Filter.atTop
      ⊢ Filter.Eventually (fun x_1 => Filter.Eventually (fun i => Membership.mem (Me …
    -/
    filter_upwards with y
    /-
      case h
      α : Type u_1
      ι : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      l : Filter ι
      inst✝¹ : PseudoMetricSpace α
      inst✝ : OpensMeasurableSpace α
      x : α
      r : ι → Real
      hr : Filter.Tendsto r l Filter.atTop
      y : α
      ⊢ Filter.Eventually (fun i => Membership.mem (Metric.ball x (r i)) y) l
    -/
    filter_upwards [hr (Ioi_mem_atTop (dist x y))] with a ha using by simpa [dist_comm] using ha
    /-
      🎉 no goals
    -/


theorem aecover_closedBall {x : α} {r : ι → ℝ} (hr : Tendsto r l atTop) :
    AECover μ l (fun i ↦ Metric.closedBall x (r i)) where
  measurableSet _ := Metric.isClosed_ball.measurableSet
  ae_eventually_mem := by
    /-
      α : Type u_1
      ι : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      l : Filter ι
      inst✝¹ : PseudoMetricSpace α
      inst✝ : OpensMeasurableSpace α
      x : α
      r : ι → Real
      hr : Filter.Tendsto r l Filter.atTop
      ⊢ Filter.Eventually (fun x_1 => Filter.Eventually (fun i => Membership.mem (Me …
    -/
    filter_upwards with y
    /-
      case h
      α : Type u_1
      ι : Type u_2
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      l : Filter ι
      inst✝¹ : PseudoMetricSpace α
      inst✝ : OpensMeasurableSpace α
      x : α
      r : ι → Real
      hr : Filter.Tendsto r l Filter.atTop
      y : α
      ⊢ Filter.Eventually (fun i => Membership.mem (Metric.closedBall x (r i)) y) l
    -/
    filter_upwards [hr (Ici_mem_atTop (dist x y))] with a ha using by simpa [dist_comm] using ha
    /-
      🎉 no goals
    -/


theorem aecover_Ici (ha : Tendsto a l atBot) : AECover μ l fun i => Ici (a i) where
  ae_eventually_mem := ae_of_all μ ha.eventually_le_atBot
  measurableSet _ := measurableSet_Ici


theorem aecover_Iic (hb : Tendsto b l atTop) : AECover μ l fun i => Iic <| b i :=
  aecover_Ici (α := αᵒᵈ) hb


theorem aecover_Icc (ha : Tendsto a l atBot) (hb : Tendsto b l atTop) :
    AECover μ l fun i => Icc (a i) (b i) :=
  (aecover_Ici ha).inter (aecover_Iic hb)


include ha in
theorem aecover_Ioi [NoMinOrder α] : AECover μ l fun i => Ioi (a i) where
  ae_eventually_mem := ae_of_all μ ha.eventually_lt_atBot
  measurableSet _ := measurableSet_Ioi


include hb in
theorem aecover_Iio [NoMaxOrder α] : AECover μ l fun i => Iio (b i) := aecover_Ioi (α := αᵒᵈ) hb


theorem aecover_Ioo [NoMinOrder α] [NoMaxOrder α] : AECover μ l fun i => Ioo (a i) (b i) :=
  (aecover_Ioi ha).inter (aecover_Iio hb)


theorem aecover_Ioc [NoMinOrder α] : AECover μ l fun i => Ioc (a i) (b i) :=
  (aecover_Ioi ha).inter (aecover_Iic hb)


theorem aecover_Ico [NoMaxOrder α] : AECover μ l fun i => Ico (a i) (b i) :=
  (aecover_Ici ha).inter (aecover_Iio hb)


include ha in
theorem aecover_Ioi_of_Ioi : AECover (μ.restrict (Ioi A)) l fun i ↦ Ioi (a i) where
  ae_eventually_mem := (ae_restrict_mem measurableSet_Ioi).mono fun _x hx ↦ ha.eventually <|
    eventually_lt_nhds hx
  measurableSet _ := measurableSet_Ioi


include hb in
theorem aecover_Iio_of_Iio : AECover (μ.restrict (Iio B)) l fun i ↦ Iio (b i) :=
  aecover_Ioi_of_Ioi (α := αᵒᵈ) hb


include ha in
theorem aecover_Ioi_of_Ici : AECover (μ.restrict (Ioi A)) l fun i ↦ Ici (a i) :=
  (aecover_Ioi_of_Ioi ha).superset (fun _ ↦ Ioi_subset_Ici_self) fun _ ↦ measurableSet_Ici


include hb in
theorem aecover_Iio_of_Iic : AECover (μ.restrict (Iio B)) l fun i ↦ Iic (b i) :=
  aecover_Ioi_of_Ici (α := αᵒᵈ) hb


include ha hb in
theorem aecover_Ioo_of_Ioo : AECover (μ.restrict <| Ioo A B) l fun i => Ioo (a i) (b i) :=
  ((aecover_Ioi_of_Ioi ha).mono <| Measure.restrict_mono Ioo_subset_Ioi_self le_rfl).inter
    ((aecover_Iio_of_Iio hb).mono <| Measure.restrict_mono Ioo_subset_Iio_self le_rfl)


include ha hb in
theorem aecover_Ioo_of_Icc : AECover (μ.restrict <| Ioo A B) l fun i => Icc (a i) (b i) :=
  (aecover_Ioo_of_Ioo ha hb).superset (fun _ ↦ Ioo_subset_Icc_self) fun _ ↦ measurableSet_Icc


include ha hb in
theorem aecover_Ioo_of_Ico : AECover (μ.restrict <| Ioo A B) l fun i => Ico (a i) (b i) :=
  (aecover_Ioo_of_Ioo ha hb).superset (fun _ ↦ Ioo_subset_Ico_self) fun _ ↦ measurableSet_Ico


include ha hb in
theorem aecover_Ioo_of_Ioc : AECover (μ.restrict <| Ioo A B) l fun i => Ioc (a i) (b i) :=
  (aecover_Ioo_of_Ioo ha hb).superset (fun _ ↦ Ioo_subset_Ioc_self) fun _ ↦ measurableSet_Ioc


theorem aecover_Ioc_of_Icc (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ioc A B) l fun i => Icc (a i) (b i) :=
  (aecover_Ioo_of_Icc ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ioc).ge


theorem aecover_Ioc_of_Ico (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ioc A B) l fun i => Ico (a i) (b i) :=
  (aecover_Ioo_of_Ico ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ioc).ge


theorem aecover_Ioc_of_Ioc (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ioc A B) l fun i => Ioc (a i) (b i) :=
  (aecover_Ioo_of_Ioc ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ioc).ge


theorem aecover_Ioc_of_Ioo (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ioc A B) l fun i => Ioo (a i) (b i) :=
  (aecover_Ioo_of_Ioo ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ioc).ge


theorem aecover_Ico_of_Icc (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ico A B) l fun i => Icc (a i) (b i) :=
  (aecover_Ioo_of_Icc ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ico).ge


theorem aecover_Ico_of_Ico (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ico A B) l fun i => Ico (a i) (b i) :=
  (aecover_Ioo_of_Ico ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ico).ge


theorem aecover_Ico_of_Ioc (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ico A B) l fun i => Ioc (a i) (b i) :=
  (aecover_Ioo_of_Ioc ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ico).ge


theorem aecover_Ico_of_Ioo (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Ico A B) l fun i => Ioo (a i) (b i) :=
  (aecover_Ioo_of_Ioo ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Ico).ge


theorem aecover_Icc_of_Icc (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Icc A B) l fun i => Icc (a i) (b i) :=
  (aecover_Ioo_of_Icc ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Icc).ge


theorem aecover_Icc_of_Ico (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Icc A B) l fun i => Ico (a i) (b i) :=
  (aecover_Ioo_of_Ico ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Icc).ge


theorem aecover_Icc_of_Ioc (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Icc A B) l fun i => Ioc (a i) (b i) :=
  (aecover_Ioo_of_Ioc ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Icc).ge


theorem aecover_Icc_of_Ioo (ha : Tendsto a l (𝓝 A)) (hb : Tendsto b l (𝓝 B)) :
    AECover (μ.restrict <| Icc A B) l fun i => Ioo (a i) (b i) :=
  (aecover_Ioo_of_Ioo ha hb).mono (Measure.restrict_congr_set Ioo_ae_eq_Icc).ge


protected theorem AECover.restrict {φ : ι → Set α} (hφ : AECover μ l φ) {s : Set α} :
    AECover (μ.restrict s) l φ :=
  hφ.mono Measure.restrict_le_self


theorem aecover_restrict_of_ae_imp {s : Set α} {φ : ι → Set α} (hs : MeasurableSet s)
    (ae_eventually_mem : ∀ᵐ x ∂μ, x ∈ s → ∀ᶠ n in l, x ∈ φ n)
    (measurable : ∀ n, MeasurableSet <| φ n) : AECover (μ.restrict s) l φ where
                          /-
                            α : Type u_1
                            ι : Type u_2
                            inst✝ : MeasurableSpace α
                            μ : MeasureTheory.Measure α
                            l : Filter ι
                            s : Set α
                            φ : ι → Set α
                            hs : MeasurableSet s
                            ae_eventually_mem : Filter.Eventually (fun x => Membership.mem s x → Filter.Ev …
                            measurable : ∀ (n : ι), MeasurableSet (φ n)
                            ⊢ Filter.Eventually (fun x => Filter.Eventually (fun i => Membership.mem (φ i) …
                          -/
  ae_eventually_mem := by rwa [ae_restrict_iff' hs]
                          /-
                            🎉 no goals
                          -/
  measurableSet := measurable


theorem AECover.inter_restrict {φ : ι → Set α} (hφ : AECover μ l φ) {s : Set α}
    (hs : MeasurableSet s) : AECover (μ.restrict s) l fun i => φ i ∩ s :=
  aecover_restrict_of_ae_imp hs
    (hφ.ae_eventually_mem.mono fun _x hx hxs => hx.mono fun _i hi => ⟨hi, hxs⟩) fun i =>
    (hφ.measurableSet i).inter hs


theorem AECover.ae_tendsto_indicator {β : Type*} [Zero β] [TopologicalSpace β] (f : α → β)
    {φ : ι → Set α} (hφ : AECover μ l φ) :
    ∀ᵐ x ∂μ, Tendsto (fun i => (φ i).indicator f x) l (𝓝 <| f x) :=
  hφ.ae_eventually_mem.mono fun _x hx =>
    tendsto_const_nhds.congr' <| hx.mono fun _n hn => (indicator_of_mem hn _).symm


theorem AECover.aemeasurable {β : Type*} [MeasurableSpace β] [l.IsCountablyGenerated] [l.NeBot]
    {f : α → β} {φ : ι → Set α} (hφ : AECover μ l φ)
    (hfm : ∀ i, AEMeasurable f (μ.restrict <| φ i)) : AEMeasurable f μ := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝² : MeasurableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), AEMeasurable f (μ.restrict (φ i))
    ⊢ AEMeasurable f μ
  -/
  obtain ⟨u, hu⟩ := l.exists_seq_tendsto
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝² : MeasurableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), AEMeasurable f (μ.restrict (φ i))
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    ⊢ AEMeasurable f μ
  -/
  have := aemeasurable_iUnion_iff.mpr fun n : ℕ => hfm (u n)
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝² : MeasurableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), AEMeasurable f (μ.restrict (φ i))
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    this : AEMeasurable f (μ.restrict (Set.iUnion fun i => φ (u i)))
    ⊢ AEMeasurable f μ
  -/
  rwa [Measure.restrict_eq_self_of_ae_mem] at this
  filter_upwards [hφ.ae_eventually_mem] with x hx using
    mem_iUnion.mpr (hu.eventually hx).exists


theorem AECover.aestronglyMeasurable {β : Type*} [TopologicalSpace β] [PseudoMetrizableSpace β]
    [l.IsCountablyGenerated] [l.NeBot] {f : α → β} {φ : ι → Set α} (hφ : AECover μ l φ)
    (hfm : ∀ i, AEStronglyMeasurable f (μ.restrict <| φ i)) : AEStronglyMeasurable f μ := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ.restrict (φ i))
    ⊢ MeasureTheory.AEStronglyMeasurable f μ
  -/
  obtain ⟨u, hu⟩ := l.exists_seq_tendsto
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ.restrict (φ i))
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    ⊢ MeasureTheory.AEStronglyMeasurable f μ
  -/
  have := aestronglyMeasurable_iUnion_iff.mpr fun n : ℕ => hfm (u n)
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ.restrict (φ i))
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    this : MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.iUnion fun i => φ …
    ⊢ MeasureTheory.AEStronglyMeasurable f μ
  -/
  rwa [Measure.restrict_eq_self_of_ae_mem] at this
  /-
    case intro.hs
    α : Type u_1
    ι : Type u_2
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    β : Type u_3
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : l.NeBot
    f : α → β
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    hfm : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ.restrict (φ i))
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    this : MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.iUnion fun i => φ …
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.iUnion fun i => φ (u i)) x)  …
  -/
  filter_upwards [hφ.ae_eventually_mem] with x hx using mem_iUnion.mpr (hu.eventually hx).exists
  /-
    🎉 no goals
  -/


theorem AECover.comp_tendsto {α ι ι' : Type*} [MeasurableSpace α] {μ : Measure α} {l : Filter ι}
    {l' : Filter ι'} {φ : ι → Set α} (hφ : AECover μ l φ) {u : ι' → ι} (hu : Tendsto u l' l) :
    AECover μ l' (φ ∘ u) where
  ae_eventually_mem := hφ.ae_eventually_mem.mono fun _x hx => hu.eventually hx
  measurableSet i := hφ.measurableSet (u i)


theorem AECover.biUnion_Iic_aecover [Preorder ι] {φ : ι → Set α} (hφ : AECover μ atTop φ) :
    AECover μ atTop fun n : ι => ⋃ (k) (_h : k ∈ Iic n), φ k :=
  hφ.superset (fun _ ↦ subset_biUnion_of_mem right_mem_Iic) fun _ ↦ .biUnion (to_countable _)
    fun _ _ ↦ (hφ.2 _)

-- Porting note: generalized from `[SemilatticeSup ι] [Nonempty ι]` to `[Preorder ι]`

theorem AECover.biInter_Ici_aecover [Preorder ι] {φ : ι → Set α}
    (hφ : AECover μ atTop φ) : AECover μ atTop fun n : ι => ⋂ (k) (_h : k ∈ Ici n), φ k where
  ae_eventually_mem := hφ.ae_eventually_mem.mono fun x h ↦ by
    /-
      α : Type u_1
      ι : Type u_2
      inst✝² : Countable ι
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Preorder ι
      φ : ι → Set α
      hφ : MeasureTheory.AECover μ Filter.atTop φ
      x : α
      h : Filter.Eventually (fun i => Membership.mem (φ i) x) Filter.atTop
      ⊢ Filter.Eventually (fun i => Membership.mem (Set.iInter fun k => Set.iInter f …
    -/
    simpa only [mem_iInter, mem_Ici, eventually_forall_ge_atTop]
    /-
      🎉 no goals
    -/
  measurableSet _ := .biInter (to_countable _) fun n _ => hφ.measurableSet n


private theorem lintegral_tendsto_of_monotone_of_nat {φ : ℕ → Set α} (hφ : AECover μ atTop φ)
    (hmono : Monotone φ) {f : α → ℝ≥0∞} (hfm : AEMeasurable f μ) :
    Tendsto (fun i => ∫⁻ x in φ i, f x ∂μ) atTop (𝓝 <| ∫⁻ x, f x ∂μ) :=
  let F n := (φ n).indicator f
  have key₁ : ∀ n, AEMeasurable (F n) μ := fun n => hfm.indicator (hφ.measurableSet n)
  have key₂ : ∀ᵐ x : α ∂μ, Monotone fun n => F n x := ae_of_all _ fun x _i _j hij =>
    indicator_le_indicator_of_subset (hmono hij) (fun x => zero_le <| f x) x
  have key₃ : ∀ᵐ x : α ∂μ, Tendsto (fun n => F n x) atTop (𝓝 (f x)) := hφ.ae_tendsto_indicator f
  (lintegral_tendsto_of_tendsto_of_monotone key₁ key₂ key₃).congr fun n =>
    lintegral_indicator (hφ.measurableSet n) _


theorem AECover.lintegral_tendsto_of_nat {φ : ℕ → Set α} (hφ : AECover μ atTop φ) {f : α → ℝ≥0∞}
    (hfm : AEMeasurable f μ) : Tendsto (∫⁻ x in φ ·, f x ∂μ) atTop (𝓝 <| ∫⁻ x, f x ∂μ) := by
  have lim₁ := lintegral_tendsto_of_monotone_of_nat hφ.biInter_Ici_aecover
    (fun i j hij => biInter_subset_biInter_left (Ici_subset_Ici.mpr hij)) hfm
  have lim₂ := lintegral_tendsto_of_monotone_of_nat hφ.biUnion_Iic_aecover
    (fun i j hij => biUnion_subset_biUnion_left (Iic_subset_Iic.mpr hij)) hfm
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    φ : Nat → Set α
    hφ : MeasureTheory.AECover μ Filter.atTop φ
    f : α → ENNReal
    hfm : AEMeasurable f μ
    lim₁ : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (Set.iInte …
    lim₂ : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (Set.iUnio …
    ⊢ Filter.Tendsto (fun x => MeasureTheory.lintegral (μ.restrict (φ x)) fun x => …
  -/
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le lim₁ lim₂ (fun n ↦ ?_) fun n ↦ ?_
  exacts [lintegral_mono_set (biInter_subset_of_mem left_mem_Ici),
    lintegral_mono_set (subset_biUnion_of_mem right_mem_Iic)]


theorem AECover.lintegral_tendsto_of_countably_generated [l.IsCountablyGenerated] {φ : ι → Set α}
    (hφ : AECover μ l φ) {f : α → ℝ≥0∞} (hfm : AEMeasurable f μ) :
    Tendsto (fun i => ∫⁻ x in φ i, f x ∂μ) l (𝓝 <| ∫⁻ x, f x ∂μ) :=
  tendsto_of_seq_tendsto fun _u hu => (hφ.comp_tendsto hu).lintegral_tendsto_of_nat hfm


theorem AECover.lintegral_eq_of_tendsto [l.NeBot] [l.IsCountablyGenerated] {φ : ι → Set α}
    (hφ : AECover μ l φ) {f : α → ℝ≥0∞} (I : ℝ≥0∞) (hfm : AEMeasurable f μ)
    (htendsto : Tendsto (fun i => ∫⁻ x in φ i, f x ∂μ) l (𝓝 I)) : ∫⁻ x, f x ∂μ = I :=
  tendsto_nhds_unique (hφ.lintegral_tendsto_of_countably_generated hfm) htendsto


theorem AECover.iSup_lintegral_eq_of_countably_generated [Nonempty ι] [l.NeBot]
    [l.IsCountablyGenerated] {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → ℝ≥0∞}
    (hfm : AEMeasurable f μ) : ⨆ i : ι, ∫⁻ x in φ i, f x ∂μ = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : Nonempty ι
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → ENNReal
    hfm : AEMeasurable f μ
    ⊢ Eq (iSup fun i => MeasureTheory.lintegral (μ.restrict (φ i)) fun x => f x) ( …
  -/
  have := hφ.lintegral_tendsto_of_countably_generated hfm
  refine ciSup_eq_of_forall_le_of_forall_lt_exists_gt
    (fun i => lintegral_mono' Measure.restrict_le_self le_rfl) fun w hw => ?_
  /-
    α : Type u_1
    ι : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : Nonempty ι
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → ENNReal
    hfm : AEMeasurable f μ
    this : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) fun …
    w : ENNReal
    hw : LT.lt w (MeasureTheory.lintegral μ fun x => f x)
    ⊢ Exists fun i => LT.lt w (MeasureTheory.lintegral (μ.restrict (φ i)) fun x => …
  -/
  exact (this.eventually_const_lt hw).exists
  /-
    🎉 no goals
  -/


theorem AECover.integrable_of_lintegral_nnnorm_bounded [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → E} (I : ℝ) (hfm : AEStronglyMeasurable f μ)
    (hbounded : ∀ᶠ i in l, (∫⁻ x in φ i, ‖f x‖₊ ∂μ) ≤ ENNReal.ofReal I) : Integrable f μ := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restr …
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine ⟨hfm, (le_of_tendsto ?_ hbounded).trans_lt ENNReal.ofReal_lt_top⟩
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restr …
    ⊢ Filter.Tendsto (fun c => MeasureTheory.lintegral (μ.restrict (φ c)) fun x => …
  -/
  exact hφ.lintegral_tendsto_of_countably_generated hfm.ennnorm
  /-
    🎉 no goals
  -/


theorem AECover.integrable_of_lintegral_nnnorm_tendsto [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → E} (I : ℝ) (hfm : AEStronglyMeasurable f μ)
    (htendsto : Tendsto (fun i => ∫⁻ x in φ i, ‖f x‖₊ ∂μ) l (𝓝 <| ENNReal.ofReal I)) :
    Integrable f μ := by
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    htendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) …
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine hφ.integrable_of_lintegral_nnnorm_bounded (max 1 (I + 1)) hfm ?_
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    htendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) …
    ⊢ Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restrict (φ i) …
  -/
  refine htendsto.eventually (ge_mem_nhds ?_)
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    htendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) …
    ⊢ LT.lt (ENNReal.ofReal I) (ENNReal.ofReal (Max.max 1 (HAdd.hAdd I 1)))
  -/
  refine (ENNReal.ofReal_lt_ofReal_iff (lt_max_of_lt_left zero_lt_one)).2 ?_
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    htendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) …
    ⊢ LT.lt I (Max.max 1 (HAdd.hAdd I 1))
  -/
  exact lt_max_of_lt_right (lt_add_one I)
  /-
    🎉 no goals
  -/


theorem AECover.integrable_of_lintegral_nnnorm_bounded' [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → E} (I : ℝ≥0) (hfm : AEStronglyMeasurable f μ)
    (hbounded : ∀ᶠ i in l, (∫⁻ x in φ i, ‖f x‖₊ ∂μ) ≤ I) : Integrable f μ :=
  hφ.integrable_of_lintegral_nnnorm_bounded I hfm
        /-
          α : Type u_1
          ι : Type u_2
          E : Type u_3
          inst✝³ : MeasurableSpace α
          μ : MeasureTheory.Measure α
          l : Filter ι
          inst✝² : NormedAddCommGroup E
          inst✝¹ : l.NeBot
          inst✝ : l.IsCountablyGenerated
          φ : ι → Set α
          hφ : MeasureTheory.AECover μ l φ
          f : α → E
          I : NNReal
          hfm : MeasureTheory.AEStronglyMeasurable f μ
          hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restr …
          ⊢ Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restrict (φ i) …
        -/
    (by simpa only [ENNReal.ofReal_coe_nnreal] using hbounded)
        /-
          🎉 no goals
        -/


theorem AECover.integrable_of_lintegral_nnnorm_tendsto' [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → E} (I : ℝ≥0) (hfm : AEStronglyMeasurable f μ)
    (htendsto : Tendsto (fun i => ∫⁻ x in φ i, ‖f x‖₊ ∂μ) l (𝓝 I)) : Integrable f μ :=
  hφ.integrable_of_lintegral_nnnorm_tendsto I hfm
        /-
          α : Type u_1
          ι : Type u_2
          E : Type u_3
          inst✝³ : MeasurableSpace α
          μ : MeasureTheory.Measure α
          l : Filter ι
          inst✝² : NormedAddCommGroup E
          inst✝¹ : l.NeBot
          inst✝ : l.IsCountablyGenerated
          φ : ι → Set α
          hφ : MeasureTheory.AECover μ l φ
          f : α → E
          I : NNReal
          hfm : MeasureTheory.AEStronglyMeasurable f μ
          htendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) …
          ⊢ Filter.Tendsto (fun i => MeasureTheory.lintegral (μ.restrict (φ i)) fun x => …
        -/
    (by simpa only [ENNReal.ofReal_coe_nnreal] using htendsto)
        /-
          🎉 no goals
        -/


theorem AECover.integrable_of_integral_norm_bounded [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → E} (I : ℝ) (hfi : ∀ i, IntegrableOn f (φ i) μ)
    (hbounded : ∀ᶠ i in l, (∫ x in φ i, ‖f x‖ ∂μ) ≤ I) : Integrable f μ := by
  have hfm : AEStronglyMeasurable f μ :=
    hφ.aestronglyMeasurable fun i => (hfi i).aestronglyMeasurable
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (φ i) μ
    hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (μ.restri …
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine hφ.integrable_of_lintegral_nnnorm_bounded I hfm ?_
  conv at hbounded in integral _ _ =>
    rw [integral_eq_lintegral_of_nonneg_ae (ae_of_all _ fun x => @norm_nonneg E _ (f x))
        hfm.norm.restrict]
  conv at hbounded in ENNReal.ofReal _ =>
    rw [← coe_nnnorm, ENNReal.ofReal_coe_nnreal]
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (φ i) μ
    hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restr …
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restrict (φ i) …
  -/
  refine hbounded.mono fun i hi => ?_
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (φ i) μ
    hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restr …
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    i : ι
    hi : LE.le (MeasureTheory.lintegral (μ.restrict (φ i)) fun a => ↑(NNNorm.nnnor …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (φ i)) fun x => ↑(NNNorm.nnnorm ( …
  -/
  rw [← ENNReal.ofReal_toReal <| ne_top_of_lt <| hasFiniteIntegral_iff_nnnorm.mp (hfi i).2]
  /-
    α : Type u_1
    ι : Type u_2
    E : Type u_3
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Filter ι
    inst✝² : NormedAddCommGroup E
    inst✝¹ : l.NeBot
    inst✝ : l.IsCountablyGenerated
    φ : ι → Set α
    hφ : MeasureTheory.AECover μ l φ
    f : α → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (φ i) μ
    hbounded : Filter.Eventually (fun i => LE.le (MeasureTheory.lintegral (μ.restr …
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    i : ι
    hi : LE.le (MeasureTheory.lintegral (μ.restrict (φ i)) fun a => ↑(NNNorm.nnnor …
    ⊢ LE.le (ENNReal.ofReal (MeasureTheory.lintegral (μ.restrict (φ i)) fun a => ↑ …
  -/
  apply ENNReal.ofReal_le_ofReal hi
  /-
    🎉 no goals
  -/


theorem AECover.integrable_of_integral_norm_tendsto [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → E} (I : ℝ) (hfi : ∀ i, IntegrableOn f (φ i) μ)
    (htendsto : Tendsto (fun i => ∫ x in φ i, ‖f x‖ ∂μ) l (𝓝 I)) : Integrable f μ :=
  let ⟨I', hI'⟩ := htendsto.isBoundedUnder_le
  hφ.integrable_of_integral_norm_bounded I' hfi hI'


theorem AECover.integrable_of_integral_bounded_of_nonneg_ae [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → ℝ} (I : ℝ) (hfi : ∀ i, IntegrableOn f (φ i) μ)
    (hnng : ∀ᵐ x ∂μ, 0 ≤ f x) (hbounded : ∀ᶠ i in l, (∫ x in φ i, f x ∂μ) ≤ I) : Integrable f μ :=
  hφ.integrable_of_integral_norm_bounded I hfi <| hbounded.mono fun _i hi =>
    (integral_congr_ae <| ae_restrict_of_ae <| hnng.mono fun _ => Real.norm_of_nonneg).le.trans hi


theorem AECover.integrable_of_integral_tendsto_of_nonneg_ae [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → ℝ} (I : ℝ) (hfi : ∀ i, IntegrableOn f (φ i) μ)
    (hnng : ∀ᵐ x ∂μ, 0 ≤ f x) (htendsto : Tendsto (fun i => ∫ x in φ i, f x ∂μ) l (𝓝 I)) :
    Integrable f μ :=
  let ⟨I', hI'⟩ := htendsto.isBoundedUnder_le
  hφ.integrable_of_integral_bounded_of_nonneg_ae I' hfi hnng hI'


theorem AECover.integral_tendsto_of_countably_generated [l.IsCountablyGenerated] {φ : ι → Set α}
    (hφ : AECover μ l φ) {f : α → E} (hfi : Integrable f μ) :
    Tendsto (fun i => ∫ x in φ i, f x ∂μ) l (𝓝 <| ∫ x, f x ∂μ) :=
  suffices h : Tendsto (fun i => ∫ x : α, (φ i).indicator f x ∂μ) l (𝓝 (∫ x : α, f x ∂μ)) from by
    /-
      α : Type u_1
      ι : Type u_2
      E : Type u_3
      inst✝³ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      l : Filter ι
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : l.IsCountablyGenerated
      φ : ι → Set α
      hφ : MeasureTheory.AECover μ l φ
      f : α → E
      hfi : MeasureTheory.Integrable f μ
      h : Filter.Tendsto (fun i => MeasureTheory.integral μ fun x => (φ i).indicator …
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (φ i)) fun x =>  …
    -/
    convert h using 2; rw [integral_indicator (hφ.measurableSet _)]
                       /-
                         🎉 no goals
                       -/
  tendsto_integral_filter_of_dominated_convergence (fun x => ‖f x‖)
    (Eventually.of_forall fun i => hfi.aestronglyMeasurable.indicator <| hφ.measurableSet i)
    (Eventually.of_forall fun _ => ae_of_all _ fun _ => norm_indicator_le_norm_self _ _) hfi.norm
    (hφ.ae_tendsto_indicator f)


/-- Slight reformulation of
    `MeasureTheory.AECover.integral_tendsto_of_countably_generated`. -/
theorem AECover.integral_eq_of_tendsto [l.NeBot] [l.IsCountablyGenerated] {φ : ι → Set α}
    (hφ : AECover μ l φ) {f : α → E} (I : E) (hfi : Integrable f μ)
    (h : Tendsto (fun n => ∫ x in φ n, f x ∂μ) l (𝓝 I)) : ∫ x, f x ∂μ = I :=
  tendsto_nhds_unique (hφ.integral_tendsto_of_countably_generated hfi) h


theorem AECover.integral_eq_of_tendsto_of_nonneg_ae [l.NeBot] [l.IsCountablyGenerated]
    {φ : ι → Set α} (hφ : AECover μ l φ) {f : α → ℝ} (I : ℝ) (hnng : 0 ≤ᵐ[μ] f)
    (hfi : ∀ n, IntegrableOn f (φ n) μ) (htendsto : Tendsto (fun n => ∫ x in φ n, f x ∂μ) l (𝓝 I)) :
    ∫ x, f x ∂μ = I :=
  have hfi' : Integrable f μ := hφ.integrable_of_integral_tendsto_of_nonneg_ae I hfi hnng htendsto
  hφ.integral_eq_of_tendsto I hfi' htendsto


theorem integrable_of_intervalIntegral_norm_bounded (I : ℝ)
    (hfi : ∀ i, IntegrableOn f (Ioc (a i) (b i)) μ) (ha : Tendsto a l atBot)
    (hb : Tendsto b l atTop) (h : ∀ᶠ i in l, (∫ x in a i..b i, ‖f x‖ ∂μ) ≤ I) : Integrable f μ := by
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a b : ι → Real
    f : Real → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    ⊢ MeasureTheory.Integrable f μ
  -/
  have hφ : AECover μ l _ := aecover_Ioc ha hb
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a b : ι → Real
    f : Real → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover μ l fun i => Set.Ioc (a i) (b i)
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine hφ.integrable_of_integral_norm_bounded I hfi (h.mp ?_)
  filter_upwards [ha.eventually (eventually_le_atBot 0),
    hb.eventually (eventually_ge_atTop 0)] with i hai hbi ht
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a b : ι → Real
    f : Real → E
    I : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover μ l fun i => Set.Ioc (a i) (b i)
    i : ι
    hai : LE.le (a i) 0
    hbi : LE.le 0 (b i)
    ht : LE.le (intervalIntegral (fun x => Norm.norm (f x)) (a i) (b i) μ) I
    ⊢ LE.le (MeasureTheory.integral (μ.restrict (Set.Ioc (a i) (b i))) fun x => No …
  -/
  rwa [← intervalIntegral.integral_of_le (hai.trans hbi)]
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on intervals `Ioc (a i) (b i)`,
where `a i` tends to -∞ and `b i` tends to ∞, and
`∫ x in a i .. b i, ‖f x‖ ∂μ` converges to `I : ℝ` along a filter `l`,
then `f` is integrable on the interval (-∞, ∞) -/
theorem integrable_of_intervalIntegral_norm_tendsto (I : ℝ)
    (hfi : ∀ i, IntegrableOn f (Ioc (a i) (b i)) μ) (ha : Tendsto a l atBot)
    (hb : Tendsto b l atTop) (h : Tendsto (fun i => ∫ x in a i..b i, ‖f x‖ ∂μ) l (𝓝 I)) :
    Integrable f μ :=
  let ⟨I', hI'⟩ := h.isBoundedUnder_le
  integrable_of_intervalIntegral_norm_bounded I' hfi ha hb hI'


theorem integrableOn_Iic_of_intervalIntegral_norm_bounded (I b : ℝ)
    (hfi : ∀ i, IntegrableOn f (Ioc (a i) b) μ) (ha : Tendsto a l atBot)
    (h : ∀ᶠ i in l, (∫ x in a i..b, ‖f x‖ ∂μ) ≤ I) : IntegrableOn f (Iic b) μ := by
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a : ι → Real
    f : Real → E
    I b : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) b) μ
    ha : Filter.Tendsto a l Filter.atBot
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic b) μ
  -/
  have hφ : AECover (μ.restrict <| Iic b) l _ := aecover_Ioi ha
  have hfi : ∀ i, IntegrableOn f (Ioi (a i)) (μ.restrict <| Iic b) := by
    intro i
    rw [IntegrableOn, Measure.restrict_restrict (hφ.measurableSet i)]
    exact hfi i
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a : ι → Real
    f : Real → E
    I b : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) b) μ
    ha : Filter.Tendsto a l Filter.atBot
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l fun i => Set.Ioi (a i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioi (a i)) (μ.restrict (Set …
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic b) μ
  -/
  refine hφ.integrable_of_integral_norm_bounded I hfi (h.mp ?_)
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a : ι → Real
    f : Real → E
    I b : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) b) μ
    ha : Filter.Tendsto a l Filter.atBot
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l fun i => Set.Ioi (a i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioi (a i)) (μ.restrict (Set …
    ⊢ Filter.Eventually (fun x => LE.le (intervalIntegral (fun x => Norm.norm (f x …
  -/
  filter_upwards [ha.eventually (eventually_le_atBot b)] with i hai
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a : ι → Real
    f : Real → E
    I b : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) b) μ
    ha : Filter.Tendsto a l Filter.atBot
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l fun i => Set.Ioi (a i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioi (a i)) (μ.restrict (Set …
    i : ι
    hai : LE.le (a i) b
    ⊢ LE.le (intervalIntegral (fun x => Norm.norm (f x)) (a i) b μ) I → LE.le (Mea …
  -/
  rw [intervalIntegral.integral_of_le hai, Measure.restrict_restrict (hφ.measurableSet i)]
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a : ι → Real
    f : Real → E
    I b : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) b) μ
    ha : Filter.Tendsto a l Filter.atBot
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l fun i => Set.Ioi (a i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioi (a i)) (μ.restrict (Set …
    i : ι
    hai : LE.le (a i) b
    ⊢ LE.le (MeasureTheory.integral (μ.restrict (Set.Ioc (a i) b)) fun x => Norm.n …
  -/
  exact id
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on intervals `Ioc (a i) b`,
where `a i` tends to -∞, and
`∫ x in a i .. b, ‖f x‖ ∂μ` converges to `I : ℝ` along a filter `l`,
then `f` is integrable on the interval (-∞, b) -/
theorem integrableOn_Iic_of_intervalIntegral_norm_tendsto (I b : ℝ)
    (hfi : ∀ i, IntegrableOn f (Ioc (a i) b) μ) (ha : Tendsto a l atBot)
    (h : Tendsto (fun i => ∫ x in a i..b, ‖f x‖ ∂μ) l (𝓝 I)) : IntegrableOn f (Iic b) μ :=
  let ⟨I', hI'⟩ := h.isBoundedUnder_le
  integrableOn_Iic_of_intervalIntegral_norm_bounded I' b hfi ha hI'


theorem integrableOn_Ioi_of_intervalIntegral_norm_bounded (I a : ℝ)
    (hfi : ∀ i, IntegrableOn f (Ioc a (b i)) μ) (hb : Tendsto b l atTop)
    (h : ∀ᶠ i in l, (∫ x in a..b i, ‖f x‖ ∂μ) ≤ I) : IntegrableOn f (Ioi a) μ := by
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    b : ι → Real
    f : Real → E
    I a : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc a (b i)) μ
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    ⊢ MeasureTheory.IntegrableOn f (Set.Ioi a) μ
  -/
  have hφ : AECover (μ.restrict <| Ioi a) l _ := aecover_Iic hb
  have hfi : ∀ i, IntegrableOn f (Iic (b i)) (μ.restrict <| Ioi a) := by
    intro i
    rw [IntegrableOn, Measure.restrict_restrict (hφ.measurableSet i), inter_comm]
    exact hfi i
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    b : ι → Real
    f : Real → E
    I a : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc a (b i)) μ
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Ioi a)) l fun i => Set.Iic (b i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Iic (b i)) (μ.restrict (Set …
    ⊢ MeasureTheory.IntegrableOn f (Set.Ioi a) μ
  -/
  refine hφ.integrable_of_integral_norm_bounded I hfi (h.mp ?_)
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    b : ι → Real
    f : Real → E
    I a : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc a (b i)) μ
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Ioi a)) l fun i => Set.Iic (b i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Iic (b i)) (μ.restrict (Set …
    ⊢ Filter.Eventually (fun x => LE.le (intervalIntegral (fun x => Norm.norm (f x …
  -/
  filter_upwards [hb.eventually (eventually_ge_atTop a)] with i hbi
  rw [intervalIntegral.integral_of_le hbi, Measure.restrict_restrict (hφ.measurableSet i),
    inter_comm]
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    b : ι → Real
    f : Real → E
    I a : Real
    hfi✝ : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc a (b i)) μ
    hb : Filter.Tendsto b l Filter.atTop
    h : Filter.Eventually (fun i => LE.le (intervalIntegral (fun x => Norm.norm (f …
    hφ : MeasureTheory.AECover (μ.restrict (Set.Ioi a)) l fun i => Set.Iic (b i)
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Iic (b i)) (μ.restrict (Set …
    i : ι
    hbi : LE.le a (b i)
    ⊢ LE.le (MeasureTheory.integral (μ.restrict (Set.Ioc a (b i))) fun x => Norm.n …
  -/
  exact id
  /-
    🎉 no goals
  -/


/-- If `f` is integrable on intervals `Ioc a (b i)`,
where `b i` tends to ∞, and
`∫ x in a .. b i, ‖f x‖ ∂μ` converges to `I : ℝ` along a filter `l`,
then `f` is integrable on the interval (a, ∞) -/
theorem integrableOn_Ioi_of_intervalIntegral_norm_tendsto (I a : ℝ)
    (hfi : ∀ i, IntegrableOn f (Ioc a (b i)) μ) (hb : Tendsto b l atTop)
    (h : Tendsto (fun i => ∫ x in a..b i, ‖f x‖ ∂μ) l (𝓝 <| I)) : IntegrableOn f (Ioi a) μ :=
  let ⟨I', hI'⟩ := h.isBoundedUnder_le
  integrableOn_Ioi_of_intervalIntegral_norm_bounded I' a hfi hb hI'


theorem integrableOn_Ioc_of_intervalIntegral_norm_bounded {I a₀ b₀ : ℝ}
    (hfi : ∀ i, IntegrableOn f <| Ioc (a i) (b i)) (ha : Tendsto a l <| 𝓝 a₀)
    (hb : Tendsto b l <| 𝓝 b₀) (h : ∀ᶠ i in l, (∫ x in Ioc (a i) (b i), ‖f x‖) ≤ I) :
    /-
      ι : Type u_1
      E : Type u_2
      μ : MeasureTheory.Measure Real
      l : Filter ι
      inst✝² : l.NeBot
      inst✝¹ : l.IsCountablyGenerated
      inst✝ : NormedAddCommGroup E
      a b : ι → Real
      f : Real → E
      I a₀ b₀ : Real
      hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) MeasureThe …
      ha : Filter.Tendsto a l (nhds a₀)
      hb : Filter.Tendsto b l (nhds b₀)
      h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn f (Ioc a₀ b₀) := by
    /-
      🎉 no goals
    -/
  refine (aecover_Ioc_of_Ioc ha hb).integrable_of_integral_norm_bounded I
    (fun i => (hfi i).restrict measurableSet_Ioc) (h.mono fun i hi ↦ ?_)
  /-
    ι : Type u_1
    E : Type u_2
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a b : ι → Real
    f : Real → E
    I a₀ b₀ : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) MeasureThe …
    ha : Filter.Tendsto a l (nhds a₀)
    hb : Filter.Tendsto b l (nhds b₀)
    h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
    i : ι
    hi : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict …
    ⊢ LE.le (MeasureTheory.integral ((MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  rw [Measure.restrict_restrict measurableSet_Ioc]
  /-
    ι : Type u_1
    E : Type u_2
    l : Filter ι
    inst✝² : l.NeBot
    inst✝¹ : l.IsCountablyGenerated
    inst✝ : NormedAddCommGroup E
    a b : ι → Real
    f : Real → E
    I a₀ b₀ : Real
    hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) MeasureThe …
    ha : Filter.Tendsto a l (nhds a₀)
    hb : Filter.Tendsto b l (nhds b₀)
    h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
    i : ι
    hi : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict …
    ⊢ LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (I …
  -/
  refine le_trans (setIntegral_mono_set (hfi i).norm ?_ ?_) hi <;> apply ae_of_all
    /-
      case refine_1.a
      ι : Type u_1
      E : Type u_2
      l : Filter ι
      inst✝² : l.NeBot
      inst✝¹ : l.IsCountablyGenerated
      inst✝ : NormedAddCommGroup E
      a b : ι → Real
      f : Real → E
      I a₀ b₀ : Real
      hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) MeasureThe …
      ha : Filter.Tendsto a l (nhds a₀)
      hb : Filter.Tendsto b l (nhds b₀)
      h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
      i : ι
      hi : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict …
      ⊢ ∀ (a : Real), LE.le (0 a) ((fun x => Norm.norm (f x)) a)
    -/
  · simp only [Pi.zero_apply, norm_nonneg, forall_const]
    /-
      🎉 no goals
    -/
    /-
      case refine_2.a
      ι : Type u_1
      E : Type u_2
      l : Filter ι
      inst✝² : l.NeBot
      inst✝¹ : l.IsCountablyGenerated
      inst✝ : NormedAddCommGroup E
      a b : ι → Real
      f : Real → E
      I a₀ b₀ : Real
      hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) (b i)) MeasureThe …
      ha : Filter.Tendsto a l (nhds a₀)
      hb : Filter.Tendsto b l (nhds b₀)
      h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
      i : ι
      hi : LE.le (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict …
      ⊢ ∀ (a_1 : Real), LE.le (Inter.inter (Set.Ioc (a i) (b i)) (Set.Ioc a₀ b₀) a_1 …
    -/
  · intro c hc; exact hc.1
                /-
                  🎉 no goals
                -/


theorem integrableOn_Ioc_of_intervalIntegral_norm_bounded_left {I a₀ b : ℝ}
    (hfi : ∀ i, IntegrableOn f <| Ioc (a i) b) (ha : Tendsto a l <| 𝓝 a₀)
                                                       /-
                                                         ι : Type u_1
                                                         E : Type u_2
                                                         μ : MeasureTheory.Measure Real
                                                         l : Filter ι
                                                         inst✝² : l.NeBot
                                                         inst✝¹ : l.IsCountablyGenerated
                                                         inst✝ : NormedAddCommGroup E
                                                         a b✝ : ι → Real
                                                         f : Real → E
                                                         I a₀ b : Real
                                                         hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc (a i) b) MeasureTheory. …
                                                         ha : Filter.Tendsto a l (nhds a₀)
                                                         h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
                                                         ⊢ MeasureTheory.Measure Real
                                                       -/
    (h : ∀ᶠ i in l, (∫ x in Ioc (a i) b, ‖f x‖) ≤ I) : IntegrableOn f (Ioc a₀ b) :=
                                                       /-
                                                         🎉 no goals
                                                       -/
  integrableOn_Ioc_of_intervalIntegral_norm_bounded hfi ha tendsto_const_nhds h


theorem integrableOn_Ioc_of_intervalIntegral_norm_bounded_right {I a b₀ : ℝ}
    (hfi : ∀ i, IntegrableOn f <| Ioc a (b i)) (hb : Tendsto b l <| 𝓝 b₀)
                                                       /-
                                                         ι : Type u_1
                                                         E : Type u_2
                                                         μ : MeasureTheory.Measure Real
                                                         l : Filter ι
                                                         inst✝² : l.NeBot
                                                         inst✝¹ : l.IsCountablyGenerated
                                                         inst✝ : NormedAddCommGroup E
                                                         a✝ b : ι → Real
                                                         f : Real → E
                                                         I a b₀ : Real
                                                         hfi : ∀ (i : ι), MeasureTheory.IntegrableOn f (Set.Ioc a (b i)) MeasureTheory. …
                                                         hb : Filter.Tendsto b l (nhds b₀)
                                                         h : Filter.Eventually (fun i => LE.le (MeasureTheory.integral (MeasureTheory.M …
                                                         ⊢ MeasureTheory.Measure Real
                                                       -/
    (h : ∀ᶠ i in l, (∫ x in Ioc a (b i), ‖f x‖) ≤ I) : IntegrableOn f (Ioc a b₀) :=
                                                       /-
                                                         🎉 no goals
                                                       -/
  integrableOn_Ioc_of_intervalIntegral_norm_bounded hfi tendsto_const_nhds hb h


@[deprecated (since := "2024-04-06")]
alias integrableOn_Ioc_of_interval_integral_norm_bounded :=
  integrableOn_Ioc_of_intervalIntegral_norm_bounded

@[deprecated (since := "2024-04-06")]
alias integrableOn_Ioc_of_interval_integral_norm_bounded_left :=
  integrableOn_Ioc_of_intervalIntegral_norm_bounded_left

@[deprecated (since := "2024-04-06")]
alias integrableOn_Ioc_of_interval_integral_norm_bounded_right :=
  integrableOn_Ioc_of_intervalIntegral_norm_bounded_right


theorem intervalIntegral_tendsto_integral (hfi : Integrable f μ) (ha : Tendsto a l atBot)
    (hb : Tendsto b l atTop) : Tendsto (fun i => ∫ x in a i..b i, f x ∂μ) l (𝓝 <| ∫ x, f x ∂μ) := by
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : ι → Real
    f : Real → E
    hfi : MeasureTheory.Integrable f μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) (a i) (b i) μ) l (n …
  -/
  let φ i := Ioc (a i) (b i)
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : ι → Real
    f : Real → E
    hfi : MeasureTheory.Integrable f μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Ioc (a i) (b i)
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) (a i) (b i) μ) l (n …
  -/
  have hφ : AECover μ l φ := aecover_Ioc ha hb
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : ι → Real
    f : Real → E
    hfi : MeasureTheory.Integrable f μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Ioc (a i) (b i)
    hφ : MeasureTheory.AECover μ l φ
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) (a i) (b i) μ) l (n …
  -/
  refine (hφ.integral_tendsto_of_countably_generated hfi).congr' ?_
  filter_upwards [ha.eventually (eventually_le_atBot 0),
    hb.eventually (eventually_ge_atTop 0)] with i hai hbi
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : ι → Real
    f : Real → E
    hfi : MeasureTheory.Integrable f μ
    ha : Filter.Tendsto a l Filter.atBot
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Ioc (a i) (b i)
    hφ : MeasureTheory.AECover μ l φ
    i : ι
    hai : LE.le (a i) 0
    hbi : LE.le 0 (b i)
    ⊢ Eq (MeasureTheory.integral (μ.restrict (φ i)) fun x => f x) (intervalIntegra …
  -/
  exact (intervalIntegral.integral_of_le (hai.trans hbi)).symm
  /-
    🎉 no goals
  -/


theorem intervalIntegral_tendsto_integral_Iic (b : ℝ) (hfi : IntegrableOn f (Iic b) μ)
    (ha : Tendsto a l atBot) :
    Tendsto (fun i => ∫ x in a i..b, f x ∂μ) l (𝓝 <| ∫ x in Iic b, f x ∂μ) := by
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : ι → Real
    f : Real → E
    b : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ha : Filter.Tendsto a l Filter.atBot
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) (a i) b μ) l (nhds  …
  -/
  let φ i := Ioi (a i)
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : ι → Real
    f : Real → E
    b : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ha : Filter.Tendsto a l Filter.atBot
    φ : ι → Set Real := fun i => Set.Ioi (a i)
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) (a i) b μ) l (nhds  …
  -/
  have hφ : AECover (μ.restrict <| Iic b) l φ := aecover_Ioi ha
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : ι → Real
    f : Real → E
    b : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ha : Filter.Tendsto a l Filter.atBot
    φ : ι → Set Real := fun i => Set.Ioi (a i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l φ
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) (a i) b μ) l (nhds  …
  -/
  refine (hφ.integral_tendsto_of_countably_generated hfi).congr' ?_
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : ι → Real
    f : Real → E
    b : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ha : Filter.Tendsto a l Filter.atBot
    φ : ι → Set Real := fun i => Set.Ioi (a i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l φ
    ⊢ l.EventuallyEq (fun i => MeasureTheory.integral ((μ.restrict (Set.Iic b)).re …
  -/
  filter_upwards [ha.eventually (eventually_le_atBot <| b)] with i hai
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : ι → Real
    f : Real → E
    b : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ha : Filter.Tendsto a l Filter.atBot
    φ : ι → Set Real := fun i => Set.Ioi (a i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l φ
    i : ι
    hai : LE.le (a i) b
    ⊢ Eq (MeasureTheory.integral ((μ.restrict (Set.Iic b)).restrict (φ i)) fun x = …
  -/
  rw [intervalIntegral.integral_of_le hai, Measure.restrict_restrict (hφ.measurableSet i)]
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : ι → Real
    f : Real → E
    b : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ha : Filter.Tendsto a l Filter.atBot
    φ : ι → Set Real := fun i => Set.Ioi (a i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Iic b)) l φ
    i : ι
    hai : LE.le (a i) b
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Inter.inter (φ i) (Set.Iic b))) fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem intervalIntegral_tendsto_integral_Ioi (a : ℝ) (hfi : IntegrableOn f (Ioi a) μ)
    (hb : Tendsto b l atTop) :
    Tendsto (fun i => ∫ x in a..b i, f x ∂μ) l (𝓝 <| ∫ x in Ioi a, f x ∂μ) := by
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : ι → Real
    f : Real → E
    a : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Ioi a) μ
    hb : Filter.Tendsto b l Filter.atTop
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) a (b i) μ) l (nhds  …
  -/
  let φ i := Iic (b i)
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : ι → Real
    f : Real → E
    a : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Ioi a) μ
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Iic (b i)
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) a (b i) μ) l (nhds  …
  -/
  have hφ : AECover (μ.restrict <| Ioi a) l φ := aecover_Iic hb
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : ι → Real
    f : Real → E
    a : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Ioi a) μ
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Iic (b i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Ioi a)) l φ
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f x) a (b i) μ) l (nhds  …
  -/
  refine (hφ.integral_tendsto_of_countably_generated hfi).congr' ?_
  /-
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : ι → Real
    f : Real → E
    a : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Ioi a) μ
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Iic (b i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Ioi a)) l φ
    ⊢ l.EventuallyEq (fun i => MeasureTheory.integral ((μ.restrict (Set.Ioi a)).re …
  -/
  filter_upwards [hb.eventually (eventually_ge_atTop <| a)] with i hbi
  rw [intervalIntegral.integral_of_le hbi, Measure.restrict_restrict (hφ.measurableSet i),
    inter_comm]
  /-
    case h
    ι : Type u_1
    E : Type u_2
    μ : MeasureTheory.Measure Real
    l : Filter ι
    inst✝² : l.IsCountablyGenerated
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : ι → Real
    f : Real → E
    a : Real
    hfi : MeasureTheory.IntegrableOn f (Set.Ioi a) μ
    hb : Filter.Tendsto b l Filter.atTop
    φ : ι → Set Real := fun i => Set.Iic (b i)
    hφ : MeasureTheory.AECover (μ.restrict (Set.Ioi a)) l φ
    i : ι
    hbi : LE.le a (b i)
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Inter.inter (Set.Ioi a) (φ i))) fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If the derivative of a function defined on the real line is integrable close to `+∞`, then
the function has a limit at `+∞`. -/
theorem tendsto_limUnder_of_hasDerivAt_of_integrableOn_Ioi [CompleteSpace E]
                                                           /-
                                                             E : Type u_1
                                                             f f' : Real → E
                                                             g g' : Real → Real
                                                             a l : Real
                                                             m : E
                                                             inst✝² : NormedAddCommGroup E
                                                             inst✝¹ : NormedSpace Real E
                                                             inst✝ : CompleteSpace E
                                                             hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
                                                             ⊢ MeasureTheory.Measure Real
                                                           -/
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt f (f' x) x) (f'int : IntegrableOn f' (Ioi a)) :
                                                           /-
                                                             🎉 no goals
                                                           -/
    Tendsto f atTop (𝓝 (limUnder atTop f)) := by
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    ⊢ Filter.Tendsto f Filter.atTop (nhds (limUnder Filter.atTop f))
  -/
  suffices ∃ a, Tendsto f atTop (𝓝 a) from tendsto_nhds_limUnder this
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  suffices CauchySeq f from cauchySeq_tendsto_of_complete this
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    ⊢ CauchySeq f
  -/
  apply Metric.cauchySeq_iff'.2 (fun ε εpos ↦ ?_)
  have A : ∀ᶠ (n : ℕ) in atTop, ∫ (x : ℝ) in Ici ↑n, ‖f' x‖ < ε := by
    have L : Tendsto (fun (n : ℕ) ↦ ∫ x in Ici (n : ℝ), ‖f' x‖) atTop
        (𝓝 (∫ x in ⋂ (n : ℕ), Ici (n : ℝ), ‖f' x‖)) := by
      apply tendsto_setIntegral_of_antitone (fun n ↦ measurableSet_Ici)
      · intro m n hmn
        exact Ici_subset_Ici.2 (Nat.cast_le.mpr hmn)
      · rcases exists_nat_gt a with ⟨n, hn⟩
        exact ⟨n, IntegrableOn.mono_set f'int.norm (Ici_subset_Ioi.2 hn)⟩
    have B : ⋂ (n : ℕ), Ici (n : ℝ) = ∅ := by
      apply eq_empty_of_forall_not_mem (fun x ↦ ?_)
      simpa only [mem_iInter, mem_Ici, not_forall, not_le] using exists_nat_gt x
    simp only [B, Measure.restrict_empty, integral_zero_measure] at L
    exact (tendsto_order.1 L).2 _ εpos
  have B : ∀ᶠ (n : ℕ) in atTop, a < n := by
    rcases exists_nat_gt a with ⟨n, hn⟩
    filter_upwards [Ioi_mem_atTop n] with m (hm : n < m) using hn.trans (Nat.cast_lt.mpr hm)
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    ε : Real
    εpos : GT.gt ε 0
    A : Filter.Eventually (fun n => LT.lt (MeasureTheory.integral (MeasureTheory.M …
    B : Filter.Eventually (fun n => LT.lt a ↑n) Filter.atTop
    ⊢ Exists fun N => ∀ (n : Real), GE.ge n N → LT.lt (Dist.dist (f n) (f N)) ε
  -/
  rcases (A.and B).exists with ⟨N, hN, h'N⟩
  /-
    case intro.intro
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    ε : Real
    εpos : GT.gt ε 0
    A : Filter.Eventually (fun n => LT.lt (MeasureTheory.integral (MeasureTheory.M …
    B : Filter.Eventually (fun n => LT.lt a ↑n) Filter.atTop
    N : Nat
    hN : LT.lt (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict …
    h'N : LT.lt a ↑N
    ⊢ Exists fun N => ∀ (n : Real), GE.ge n N → LT.lt (Dist.dist (f n) (f N)) ε
  -/
  refine ⟨N, fun x hx ↦ ?_⟩
  calc
  dist (f x) (f ↑N)
    = ‖f x - f N‖ := dist_eq_norm _ _
  _ = ‖∫ t in Ioc ↑N x, f' t‖ := by
      rw [← intervalIntegral.integral_of_le hx, intervalIntegral.integral_eq_sub_of_hasDerivAt]
      · intro y hy
        simp only [hx, uIcc_of_le, mem_Icc] at hy
        exact hderiv _ (h'N.trans_le hy.1)
      · rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hx]
        exact f'int.mono_set (Ioc_subset_Ioi_self.trans (Ioi_subset_Ioi h'N.le))
  _ ≤ ∫ t in Ioc ↑N x, ‖f' t‖ := norm_integral_le_integral_norm fun a ↦ f' a
  _ ≤ ∫ t in Ici ↑N, ‖f' t‖ := by
      apply setIntegral_mono_set
      · apply IntegrableOn.mono_set f'int.norm (Ici_subset_Ioi.2 h'N)
      · filter_upwards with x using norm_nonneg _
      · have : Ioc (↑N) x ⊆ Ici ↑N := Ioc_subset_Ioi_self.trans Ioi_subset_Ici_self
        exact this.eventuallyLE
  _ < ε := hN


open UniformSpace in
/-- If a function and its derivative are integrable on `(a, +∞)`, then the function tends to zero
at `+∞`. -/
theorem tendsto_zero_of_hasDerivAt_of_integrableOn_Ioi
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt f (f' x) x)
             /-
               E : Type u_1
               f f' : Real → E
               g g' : Real → Real
               a l : Real
               m : E
               inst✝¹ : NormedAddCommGroup E
               inst✝ : NormedSpace Real E
               hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
               ⊢ MeasureTheory.Measure Real
             -/
             /-
               🎉 no goals
             -/
    (f'int : IntegrableOn f' (Ioi a)) (fint : IntegrableOn f (Ioi a)) :
                                              /-
                                                🎉 no goals
                                              -/
    Tendsto f atTop (𝓝 0) := by
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Ioi a) MeasureTheory.MeasureSpace.vol …
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  let F : E →L[ℝ] Completion E := Completion.toComplL
  have Fderiv : ∀ x ∈ Ioi a, HasDerivAt (F ∘ f) (F (f' x)) x :=
    fun x hx ↦ F.hasFDerivAt.comp_hasDerivAt _ (hderiv x hx)
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Ioi a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt (Function.com …
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  have Fint : IntegrableOn (F ∘ f) (Ioi a) := by apply F.integrable_comp fint
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Ioi a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt (Function.com …
    Fint : MeasureTheory.IntegrableOn (Function.comp (⇑F) f) (Set.Ioi a) MeasureTh …
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  have F'int : IntegrableOn (F ∘ f') (Ioi a) := by apply F.integrable_comp f'int
  have A : Tendsto (F ∘ f) atTop (𝓝 (limUnder atTop (F ∘ f))) := by
    apply tendsto_limUnder_of_hasDerivAt_of_integrableOn_Ioi Fderiv F'int
  have B : limUnder atTop (F ∘ f) = F 0 := by
    have : IntegrableAtFilter (F ∘ f) atTop := by exact ⟨Ioi a, Ioi_mem_atTop _, Fint⟩
    apply IntegrableAtFilter.eq_zero_of_tendsto this ?_ A
    intro s hs
    rcases mem_atTop_sets.1 hs with ⟨b, hb⟩
    rw [← top_le_iff, ← volume_Ici (a := b)]
    exact measure_mono hb
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Ioi a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt (Function.com …
    Fint : MeasureTheory.IntegrableOn (Function.comp (⇑F) f) (Set.Ioi a) MeasureTh …
    F'int : MeasureTheory.IntegrableOn (Function.comp (⇑F) f') (Set.Ioi a) Measure …
    A : Filter.Tendsto (Function.comp (⇑F) f) Filter.atTop (nhds (limUnder Filter. …
    B : Eq (limUnder Filter.atTop (Function.comp (⇑F) f)) (F 0)
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  rwa [B, ← IsEmbedding.tendsto_nhds_iff] at A
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Ioi a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt (Function.com …
    Fint : MeasureTheory.IntegrableOn (Function.comp (⇑F) f) (Set.Ioi a) MeasureTh …
    F'int : MeasureTheory.IntegrableOn (Function.comp (⇑F) f') (Set.Ioi a) Measure …
    A : Filter.Tendsto (Function.comp (⇑F) f) Filter.atTop (nhds (F 0))
    B : Eq (limUnder Filter.atTop (Function.comp (⇑F) f)) (F 0)
    ⊢ Topology.IsEmbedding ⇑F
  -/
  exact (Completion.isUniformEmbedding_coe E).isEmbedding
  /-
    🎉 no goals
  -/


/-- **Fundamental theorem of calculus-2**, on semi-infinite intervals `(a, +∞)`.
When a function has a limit at infinity `m`, and its derivative is integrable, then the
integral of the derivative on `(a, +∞)` is `m - f a`. Version assuming differentiability
on `(a, +∞)` and continuity at `a⁺`.

Note that such a function always has a limit at infinity,
see `tendsto_limUnder_of_hasDerivAt_of_integrableOn_Ioi`. -/
theorem integral_Ioi_of_hasDerivAt_of_tendsto (hcont : ContinuousWithinAt f (Ici a) a)
                                                           /-
                                                             E : Type u_1
                                                             f f' : Real → E
                                                             g g' : Real → Real
                                                             a l : Real
                                                             m : E
                                                             inst✝² : NormedAddCommGroup E
                                                             inst✝¹ : NormedSpace Real E
                                                             inst✝ : CompleteSpace E
                                                             hcont : ContinuousWithinAt f (Set.Ici a) a
                                                             hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
                                                             ⊢ MeasureTheory.Measure Real
                                                           -/
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt f (f' x) x) (f'int : IntegrableOn f' (Ioi a))
                                                           /-
                                                             🎉 no goals
                                                           -/
    (hf : Tendsto f atTop (𝓝 m)) : ∫ x in Ioi a, f' x = m - f a := by
  have hcont : ContinuousOn f (Ici a) := by
    intro x hx
    rcases hx.out.eq_or_lt with rfl|hx
    · exact hcont
    · exact (hderiv x hx).continuousAt.continuousWithinAt
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine tendsto_nhds_unique (intervalIntegral_tendsto_integral_Ioi a f'int tendsto_id) ?_
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f' x) a (id i) MeasureTh …
  -/
  apply Tendsto.congr' _ (hf.sub_const _)
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    ⊢ Filter.atTop.EventuallyEq (fun x => HSub.hSub (f x) (f a)) fun i => interval …
  -/
  filter_upwards [Ioi_mem_atTop a] with x hx
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    x : Real
    hx : Membership.mem (Set.Ioi a) x
    ⊢ Eq (HSub.hSub (f x) (f a)) (intervalIntegral (fun x => f' x) a (id x) Measur …
  -/
  have h'x : a ≤ id x := le_of_lt hx
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    x : Real
    hx : Membership.mem (Set.Ioi a) x
    h'x : LE.le a (id x)
    ⊢ Eq (HSub.hSub (f x) (f a)) (intervalIntegral (fun x => f' x) a (id x) Measur …
  -/
  symm
  apply
    intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le h'x (hcont.mono Icc_subset_Ici_self)
      fun y hy => hderiv y hy.1
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    x : Real
    hx : Membership.mem (Set.Ioi a) x
    h'x : LE.le a (id x)
    ⊢ IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a (id x)
  -/
  rw [intervalIntegrable_iff_integrableOn_Ioc_of_le h'x]
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    hcont : ContinuousOn f (Set.Ici a)
    x : Real
    hx : Membership.mem (Set.Ioi a) x
    h'x : LE.le a (id x)
    ⊢ MeasureTheory.IntegrableOn f' (Set.Ioc a (id x)) MeasureTheory.MeasureSpace. …
  -/
  exact f'int.mono (fun y hy => hy.1) le_rfl
  /-
    🎉 no goals
  -/


/-- **Fundamental theorem of calculus-2**, on semi-infinite intervals `(a, +∞)`.
When a function has a limit at infinity `m`, and its derivative is integrable, then the
integral of the derivative on `(a, +∞)` is `m - f a`. Version assuming differentiability
on `[a, +∞)`.

Note that such a function always has a limit at infinity,
see `tendsto_limUnder_of_hasDerivAt_of_integrableOn_Ioi`. -/
theorem integral_Ioi_of_hasDerivAt_of_tendsto' (hderiv : ∀ x ∈ Ici a, HasDerivAt f (f' x) x)
             /-
               E : Type u_1
               f f' : Real → E
               g g' : Real → Real
               a l : Real
               m : E
               inst✝² : NormedAddCommGroup E
               inst✝¹ : NormedSpace Real E
               inst✝ : CompleteSpace E
               hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt f (f' x) x
               ⊢ MeasureTheory.Measure Real
             -/
    (f'int : IntegrableOn f' (Ioi a)) (hf : Tendsto f atTop (𝓝 m)) :
             /-
               🎉 no goals
             -/
    ∫ x in Ioi a, f' x = m - f a := by
  refine integral_Ioi_of_hasDerivAt_of_tendsto ?_ (fun x hx => hderiv x hx.out.le)
    f'int hf
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Ioi a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atTop (nhds m)
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  exact (hderiv a left_mem_Ici).continuousAt.continuousWithinAt
  /-
    🎉 no goals
  -/


/-- A special case of `integral_Ioi_of_hasDerivAt_of_tendsto` where we assume that `f` is C^1 with
compact support. -/
theorem _root_.HasCompactSupport.integral_Ioi_deriv_eq (hf : ContDiff ℝ 1 f)
    (h2f : HasCompactSupport f) (b : ℝ) : ∫ x in Ioi b, deriv f x = - f b := by
  /-
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : HasCompactSupport f
    b : Real
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have := fun x (_ : x ∈ Ioi b) ↦ hf.differentiable le_rfl x |>.hasDerivAt
  /-
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : HasCompactSupport f
    b : Real
    this : ∀ (x : Real), Membership.mem (Set.Ioi b) x → HasDerivAt f (deriv f x) x
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [integral_Ioi_of_hasDerivAt_of_tendsto hf.continuous.continuousWithinAt this, zero_sub]
    /-
      case f'int
      E : Type u_1
      f : Real → E
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hf : ContDiff Real 1 f
      h2f : HasCompactSupport f
      b : Real
      this : ∀ (x : Real), Membership.mem (Set.Ioi b) x → HasDerivAt f (deriv f x) x
      ⊢ MeasureTheory.IntegrableOn (deriv f) (Set.Ioi b) MeasureTheory.MeasureSpace. …
    -/
  · refine hf.continuous_deriv le_rfl |>.integrable_of_hasCompactSupport h2f.deriv |>.integrableOn
    /-
      🎉 no goals
    -/
  /-
    case hf
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : HasCompactSupport f
    b : Real
    this : ∀ (x : Real), Membership.mem (Set.Ioi b) x → HasDerivAt f (deriv f x) x
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  rw [hasCompactSupport_iff_eventuallyEq, Filter.coclosedCompact_eq_cocompact] at h2f
  /-
    case hf
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : (Filter.cocompact Real).EventuallyEq f 0
    b : Real
    this : ∀ (x : Real), Membership.mem (Set.Ioi b) x → HasDerivAt f (deriv f x) x
    ⊢ Filter.Tendsto f Filter.atTop (nhds 0)
  -/
  exact h2f.filter_mono _root_.atTop_le_cocompact |>.tendsto
  /-
    🎉 no goals
  -/


/-- When a function has a limit at infinity, and its derivative is nonnegative, then the derivative
is automatically integrable on `(a, +∞)`. Version assuming differentiability
on `(a, +∞)` and continuity at `a⁺`. -/
theorem integrableOn_Ioi_deriv_of_nonneg (hcont : ContinuousWithinAt g (Ici a) a)
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt g (g' x) x) (g'pos : ∀ x ∈ Ioi a, 0 ≤ g' x)
                                   /-
                                     E : Type u_1
                                     f f' : Real → E
                                     g g' : Real → Real
                                     a l : Real
                                     m : E
                                     inst✝² : NormedAddCommGroup E
                                     inst✝¹ : NormedSpace Real E
                                     inst✝ : CompleteSpace E
                                     hcont : ContinuousWithinAt g (Set.Ici a) a
                                     hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
                                     g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
                                     hg : Filter.Tendsto g Filter.atTop (nhds l)
                                     ⊢ MeasureTheory.Measure Real
                                   -/
    (hg : Tendsto g atTop (𝓝 l)) : IntegrableOn g' (Ioi a) := by
                                   /-
                                     🎉 no goals
                                   -/
  have hcont : ContinuousOn g (Ici a) := by
    intro x hx
    rcases hx.out.eq_or_lt with rfl|hx
    · exact hcont
    · exact (hderiv x hx).continuousAt.continuousWithinAt
  /-
    g g' : Real → Real
    a l : Real
    hcont✝ : ContinuousWithinAt g (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    hcont : ContinuousOn g (Set.Ici a)
    ⊢ MeasureTheory.IntegrableOn g' (Set.Ioi a) MeasureTheory.MeasureSpace.volume
  -/
  refine integrableOn_Ioi_of_intervalIntegral_norm_tendsto (l - g a) a (fun x => ?_) tendsto_id ?_
  · exact intervalIntegral.integrableOn_deriv_of_nonneg (hcont.mono Icc_subset_Ici_self)
      (fun y hy => hderiv y hy.1) fun y hy => g'pos y hy.1
  /-
    case refine_2
    g g' : Real → Real
    a l : Real
    hcont✝ : ContinuousWithinAt g (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    hcont : ContinuousOn g (Set.Ici a)
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Norm.norm (g' x)) a (id  …
  -/
  apply Tendsto.congr' _ (hg.sub_const _)
  /-
    g g' : Real → Real
    a l : Real
    hcont✝ : ContinuousWithinAt g (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    hcont : ContinuousOn g (Set.Ici a)
    ⊢ Filter.atTop.EventuallyEq (fun x => HSub.hSub (g x) (g a)) fun i => interval …
  -/
  filter_upwards [Ioi_mem_atTop a] with x hx
  /-
    case h
    g g' : Real → Real
    a l : Real
    hcont✝ : ContinuousWithinAt g (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    hcont : ContinuousOn g (Set.Ici a)
    x : Real
    hx : Membership.mem (Set.Ioi a) x
    ⊢ Eq (HSub.hSub (g x) (g a)) (intervalIntegral (fun x => Norm.norm (g' x)) a ( …
  -/
  have h'x : a ≤ id x := le_of_lt hx
  calc
    g x - g a = ∫ y in a..id x, g' y := by
      symm
      apply intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le h'x
        (hcont.mono Icc_subset_Ici_self) fun y hy => hderiv y hy.1
      rw [intervalIntegrable_iff_integrableOn_Ioc_of_le h'x]
      exact intervalIntegral.integrableOn_deriv_of_nonneg (hcont.mono Icc_subset_Ici_self)
        (fun y hy => hderiv y hy.1) fun y hy => g'pos y hy.1
    _ = ∫ y in a..id x, ‖g' y‖ := by
      simp_rw [intervalIntegral.integral_of_le h'x]
      refine setIntegral_congr_fun measurableSet_Ioc fun y hy => ?_
      dsimp
      rw [abs_of_nonneg]
      exact g'pos _ hy.1


/-- When a function has a limit at infinity, and its derivative is nonnegative, then the derivative
is automatically integrable on `(a, +∞)`. Version assuming differentiability
on `[a, +∞)`. -/
theorem integrableOn_Ioi_deriv_of_nonneg' (hderiv : ∀ x ∈ Ici a, HasDerivAt g (g' x) x)
                                                                   /-
                                                                     E : Type u_1
                                                                     f f' : Real → E
                                                                     g g' : Real → Real
                                                                     a l : Real
                                                                     m : E
                                                                     inst✝² : NormedAddCommGroup E
                                                                     inst✝¹ : NormedSpace Real E
                                                                     inst✝ : CompleteSpace E
                                                                     hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt g (g' x) x
                                                                     g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
                                                                     hg : Filter.Tendsto g Filter.atTop (nhds l)
                                                                     ⊢ MeasureTheory.Measure Real
                                                                   -/
    (g'pos : ∀ x ∈ Ioi a, 0 ≤ g' x) (hg : Tendsto g atTop (𝓝 l)) : IntegrableOn g' (Ioi a) := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    g g' : Real → Real
    a l : Real
    hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt g (g' x) x
    g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    ⊢ MeasureTheory.IntegrableOn g' (Set.Ioi a) MeasureTheory.MeasureSpace.volume
  -/
  refine integrableOn_Ioi_deriv_of_nonneg ?_ (fun x hx => hderiv x hx.out.le) g'pos hg
  /-
    g g' : Real → Real
    a l : Real
    hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt g (g' x) x
    g'pos : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le 0 (g' x)
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    ⊢ ContinuousWithinAt g (Set.Ici a) a
  -/
  exact (hderiv a left_mem_Ici).continuousAt.continuousWithinAt
  /-
    🎉 no goals
  -/


/-- When a function has a limit at infinity `l`, and its derivative is nonnegative, then the
integral of the derivative on `(a, +∞)` is `l - g a` (and the derivative is integrable, see
`integrable_on_Ioi_deriv_of_nonneg`). Version assuming differentiability on `(a, +∞)` and
continuity at `a⁺`. -/
theorem integral_Ioi_of_hasDerivAt_of_nonneg (hcont : ContinuousWithinAt g (Ici a) a)
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt g (g' x) x) (g'pos : ∀ x ∈ Ioi a, 0 ≤ g' x)
    (hg : Tendsto g atTop (𝓝 l)) : ∫ x in Ioi a, g' x = l - g a :=
  integral_Ioi_of_hasDerivAt_of_tendsto hcont hderiv
    (integrableOn_Ioi_deriv_of_nonneg hcont hderiv g'pos hg) hg


/-- When a function has a limit at infinity `l`, and its derivative is nonnegative, then the
integral of the derivative on `(a, +∞)` is `l - g a` (and the derivative is integrable, see
`integrable_on_Ioi_deriv_of_nonneg'`). Version assuming differentiability on `[a, +∞)`. -/
theorem integral_Ioi_of_hasDerivAt_of_nonneg' (hderiv : ∀ x ∈ Ici a, HasDerivAt g (g' x) x)
    (g'pos : ∀ x ∈ Ioi a, 0 ≤ g' x) (hg : Tendsto g atTop (𝓝 l)) : ∫ x in Ioi a, g' x = l - g a :=
  integral_Ioi_of_hasDerivAt_of_tendsto' hderiv (integrableOn_Ioi_deriv_of_nonneg' hderiv g'pos hg)
    hg


/-- When a function has a limit at infinity, and its derivative is nonpositive, then the derivative
is automatically integrable on `(a, +∞)`. Version assuming differentiability
on `(a, +∞)` and continuity at `a⁺`. -/
theorem integrableOn_Ioi_deriv_of_nonpos (hcont : ContinuousWithinAt g (Ici a) a)
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt g (g' x) x) (g'neg : ∀ x ∈ Ioi a, g' x ≤ 0)
                                   /-
                                     E : Type u_1
                                     f f' : Real → E
                                     g g' : Real → Real
                                     a l : Real
                                     m : E
                                     inst✝² : NormedAddCommGroup E
                                     inst✝¹ : NormedSpace Real E
                                     inst✝ : CompleteSpace E
                                     hcont : ContinuousWithinAt g (Set.Ici a) a
                                     hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
                                     g'neg : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le (g' x) 0
                                     hg : Filter.Tendsto g Filter.atTop (nhds l)
                                     ⊢ MeasureTheory.Measure Real
                                   -/
    (hg : Tendsto g atTop (𝓝 l)) : IntegrableOn g' (Ioi a) := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    g g' : Real → Real
    a l : Real
    hcont : ContinuousWithinAt g (Set.Ici a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    g'neg : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le (g' x) 0
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    ⊢ MeasureTheory.IntegrableOn g' (Set.Ioi a) MeasureTheory.MeasureSpace.volume
  -/
  apply integrable_neg_iff.1
  exact integrableOn_Ioi_deriv_of_nonneg hcont.neg (fun x hx => (hderiv x hx).neg)
    (fun x hx => neg_nonneg_of_nonpos (g'neg x hx)) hg.neg


/-- When a function has a limit at infinity, and its derivative is nonpositive, then the derivative
is automatically integrable on `(a, +∞)`. Version assuming differentiability
on `[a, +∞)`. -/
theorem integrableOn_Ioi_deriv_of_nonpos' (hderiv : ∀ x ∈ Ici a, HasDerivAt g (g' x) x)
                                                                   /-
                                                                     E : Type u_1
                                                                     f f' : Real → E
                                                                     g g' : Real → Real
                                                                     a l : Real
                                                                     m : E
                                                                     inst✝² : NormedAddCommGroup E
                                                                     inst✝¹ : NormedSpace Real E
                                                                     inst✝ : CompleteSpace E
                                                                     hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt g (g' x) x
                                                                     g'neg : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le (g' x) 0
                                                                     hg : Filter.Tendsto g Filter.atTop (nhds l)
                                                                     ⊢ MeasureTheory.Measure Real
                                                                   -/
    (g'neg : ∀ x ∈ Ioi a, g' x ≤ 0) (hg : Tendsto g atTop (𝓝 l)) : IntegrableOn g' (Ioi a) := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  /-
    g g' : Real → Real
    a l : Real
    hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt g (g' x) x
    g'neg : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le (g' x) 0
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    ⊢ MeasureTheory.IntegrableOn g' (Set.Ioi a) MeasureTheory.MeasureSpace.volume
  -/
  refine integrableOn_Ioi_deriv_of_nonpos ?_ (fun x hx ↦ hderiv x hx.out.le) g'neg hg
  /-
    g g' : Real → Real
    a l : Real
    hderiv : ∀ (x : Real), Membership.mem (Set.Ici a) x → HasDerivAt g (g' x) x
    g'neg : ∀ (x : Real), Membership.mem (Set.Ioi a) x → LE.le (g' x) 0
    hg : Filter.Tendsto g Filter.atTop (nhds l)
    ⊢ ContinuousWithinAt g (Set.Ici a) a
  -/
  exact (hderiv a left_mem_Ici).continuousAt.continuousWithinAt
  /-
    🎉 no goals
  -/


/-- When a function has a limit at infinity `l`, and its derivative is nonpositive, then the
integral of the derivative on `(a, +∞)` is `l - g a` (and the derivative is integrable, see
`integrable_on_Ioi_deriv_of_nonneg`). Version assuming differentiability on `(a, +∞)` and
continuity at `a⁺`. -/
theorem integral_Ioi_of_hasDerivAt_of_nonpos (hcont : ContinuousWithinAt g (Ici a) a)
    (hderiv : ∀ x ∈ Ioi a, HasDerivAt g (g' x) x) (g'neg : ∀ x ∈ Ioi a, g' x ≤ 0)
    (hg : Tendsto g atTop (𝓝 l)) : ∫ x in Ioi a, g' x = l - g a :=
  integral_Ioi_of_hasDerivAt_of_tendsto hcont hderiv
    (integrableOn_Ioi_deriv_of_nonpos hcont hderiv g'neg hg) hg


/-- When a function has a limit at infinity `l`, and its derivative is nonpositive, then the
integral of the derivative on `(a, +∞)` is `l - g a` (and the derivative is integrable, see
`integrable_on_Ioi_deriv_of_nonneg'`). Version assuming differentiability on `[a, +∞)`. -/
theorem integral_Ioi_of_hasDerivAt_of_nonpos' (hderiv : ∀ x ∈ Ici a, HasDerivAt g (g' x) x)
    (g'neg : ∀ x ∈ Ioi a, g' x ≤ 0) (hg : Tendsto g atTop (𝓝 l)) : ∫ x in Ioi a, g' x = l - g a :=
  integral_Ioi_of_hasDerivAt_of_tendsto' hderiv (integrableOn_Ioi_deriv_of_nonpos' hderiv g'neg hg)
    hg


/-- If the derivative of a function defined on the real line is integrable close to `-∞`, then
the function has a limit at `-∞`. -/
theorem tendsto_limUnder_of_hasDerivAt_of_integrableOn_Iic [CompleteSpace E]
                                                           /-
                                                             E : Type u_1
                                                             f f' : Real → E
                                                             a : Real
                                                             m : E
                                                             inst✝² : NormedAddCommGroup E
                                                             inst✝¹ : NormedSpace Real E
                                                             inst✝ : CompleteSpace E
                                                             hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
                                                             ⊢ MeasureTheory.Measure Real
                                                           -/
    (hderiv : ∀ x ∈ Iic a, HasDerivAt f (f' x) x) (f'int : IntegrableOn f' (Iic a)) :
                                                           /-
                                                             🎉 no goals
                                                           -/
    Tendsto f atBot (𝓝 (limUnder atBot f)) := by
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    ⊢ Filter.Tendsto f Filter.atBot (nhds (limUnder Filter.atBot f))
  -/
  suffices ∃ a, Tendsto f atBot (𝓝 a) from tendsto_nhds_limUnder this
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    ⊢ Exists fun a => Filter.Tendsto f Filter.atBot (nhds a)
  -/
  let g := f ∘ (fun x ↦ -x)
  have hdg : ∀ x ∈ Ioi (-a), HasDerivAt g (-f' (-x)) x := by
    intro x hx
    have : -x ∈ Iic a := by simp only [mem_Iic, mem_Ioi, neg_le] at *; exact hx.le
    simpa using HasDerivAt.scomp x (hderiv (-x) this) (hasDerivAt_neg' x)
  have L : Tendsto g atTop (𝓝 (limUnder atTop g)) := by
    apply tendsto_limUnder_of_hasDerivAt_of_integrableOn_Ioi hdg
    exact ((MeasurePreserving.integrableOn_comp_preimage (Measure.measurePreserving_neg _)
      (Homeomorph.neg ℝ).measurableEmbedding).2 f'int.neg).mono_set (by simp)
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    g : Real → E := Function.comp f fun x => Neg.neg x
    hdg : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt g (Neg …
    L : Filter.Tendsto g Filter.atTop (nhds (limUnder Filter.atTop g))
    ⊢ Exists fun a => Filter.Tendsto f Filter.atBot (nhds a)
  -/
  refine ⟨limUnder atTop g, ?_⟩
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    g : Real → E := Function.comp f fun x => Neg.neg x
    hdg : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt g (Neg …
    L : Filter.Tendsto g Filter.atTop (nhds (limUnder Filter.atTop g))
    ⊢ Filter.Tendsto f Filter.atBot (nhds (limUnder Filter.atTop g))
  -/
  have : Tendsto (fun x ↦ g (-x)) atBot (𝓝 (limUnder atTop g)) := L.comp tendsto_neg_atBot_atTop
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    g : Real → E := Function.comp f fun x => Neg.neg x
    hdg : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt g (Neg …
    L : Filter.Tendsto g Filter.atTop (nhds (limUnder Filter.atTop g))
    this : Filter.Tendsto (fun x => g (Neg.neg x)) Filter.atBot (nhds (limUnder Fi …
    ⊢ Filter.Tendsto f Filter.atBot (nhds (limUnder Filter.atTop g))
  -/
  simpa [g] using this
  /-
    🎉 no goals
  -/


open UniformSpace in
/-- If a function and its derivative are integrable on `(-∞, a]`, then the function tends to zero
at `-∞`. -/
theorem tendsto_zero_of_hasDerivAt_of_integrableOn_Iic
    (hderiv : ∀ x ∈ Iic a, HasDerivAt f (f' x) x)
             /-
               E : Type u_1
               f f' : Real → E
               a : Real
               m : E
               inst✝¹ : NormedAddCommGroup E
               inst✝ : NormedSpace Real E
               hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
               ⊢ MeasureTheory.Measure Real
             -/
             /-
               🎉 no goals
             -/
    (f'int : IntegrableOn f' (Iic a)) (fint : IntegrableOn f (Iic a)) :
                                              /-
                                                🎉 no goals
                                              -/
    Tendsto f atBot (𝓝 0) := by
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Iic a) MeasureTheory.MeasureSpace.vol …
    ⊢ Filter.Tendsto f Filter.atBot (nhds 0)
  -/
  let F : E →L[ℝ] Completion E := Completion.toComplL
  have Fderiv : ∀ x ∈ Iic a, HasDerivAt (F ∘ f) (F (f' x)) x :=
    fun x hx ↦ F.hasFDerivAt.comp_hasDerivAt _ (hderiv x hx)
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Iic a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt (Function.com …
    ⊢ Filter.Tendsto f Filter.atBot (nhds 0)
  -/
  have Fint : IntegrableOn (F ∘ f) (Iic a) := by apply F.integrable_comp fint
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Iic a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt (Function.com …
    Fint : MeasureTheory.IntegrableOn (Function.comp (⇑F) f) (Set.Iic a) MeasureTh …
    ⊢ Filter.Tendsto f Filter.atBot (nhds 0)
  -/
  have F'int : IntegrableOn (F ∘ f') (Iic a) := by apply F.integrable_comp f'int
  have A : Tendsto (F ∘ f) atBot (𝓝 (limUnder atBot (F ∘ f))) := by
    apply tendsto_limUnder_of_hasDerivAt_of_integrableOn_Iic Fderiv F'int
  have B : limUnder atBot (F ∘ f) = F 0 := by
    have : IntegrableAtFilter (F ∘ f) atBot := by exact ⟨Iic a, Iic_mem_atBot _, Fint⟩
    apply IntegrableAtFilter.eq_zero_of_tendsto this ?_ A
    intro s hs
    rcases mem_atBot_sets.1 hs with ⟨b, hb⟩
    apply le_antisymm (le_top)
    rw [← volume_Iic (a := b)]
    exact measure_mono hb
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Iic a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt (Function.com …
    Fint : MeasureTheory.IntegrableOn (Function.comp (⇑F) f) (Set.Iic a) MeasureTh …
    F'int : MeasureTheory.IntegrableOn (Function.comp (⇑F) f') (Set.Iic a) Measure …
    A : Filter.Tendsto (Function.comp (⇑F) f) Filter.atBot (nhds (limUnder Filter. …
    B : Eq (limUnder Filter.atBot (Function.comp (⇑F) f)) (F 0)
    ⊢ Filter.Tendsto f Filter.atBot (nhds 0)
  -/
  rwa [B, ← IsEmbedding.tendsto_nhds_iff] at A
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    fint : MeasureTheory.IntegrableOn f (Set.Iic a) MeasureTheory.MeasureSpace.vol …
    F : ContinuousLinearMap (RingHom.id Real) E (UniformSpace.Completion E) := Uni …
    Fderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt (Function.com …
    Fint : MeasureTheory.IntegrableOn (Function.comp (⇑F) f) (Set.Iic a) MeasureTh …
    F'int : MeasureTheory.IntegrableOn (Function.comp (⇑F) f') (Set.Iic a) Measure …
    A : Filter.Tendsto (Function.comp (⇑F) f) Filter.atBot (nhds (F 0))
    B : Eq (limUnder Filter.atBot (Function.comp (⇑F) f)) (F 0)
    ⊢ Topology.IsEmbedding ⇑F
  -/
  exact (Completion.isUniformEmbedding_coe E).isEmbedding
  /-
    🎉 no goals
  -/


/-- **Fundamental theorem of calculus-2**, on semi-infinite intervals `(-∞, a)`.
When a function has a limit `m` at `-∞`, and its derivative is integrable, then the
integral of the derivative on `(-∞, a)` is `f a - m`. Version assuming differentiability
on `(-∞, a)` and continuity at `a⁻`.

Note that such a function always has a limit at minus infinity,
see `tendsto_limUnder_of_hasDerivAt_of_integrableOn_Iic`. -/
theorem integral_Iic_of_hasDerivAt_of_tendsto (hcont : ContinuousWithinAt f (Iic a) a)
                                                           /-
                                                             E : Type u_1
                                                             f f' : Real → E
                                                             a : Real
                                                             m : E
                                                             inst✝² : NormedAddCommGroup E
                                                             inst✝¹ : NormedSpace Real E
                                                             inst✝ : CompleteSpace E
                                                             hcont : ContinuousWithinAt f (Set.Iic a) a
                                                             hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
                                                             ⊢ MeasureTheory.Measure Real
                                                           -/
    (hderiv : ∀ x ∈ Iio a, HasDerivAt f (f' x) x) (f'int : IntegrableOn f' (Iic a))
                                                           /-
                                                             🎉 no goals
                                                           -/
    (hf : Tendsto f atBot (𝓝 m)) : ∫ x in Iic a, f' x = f a - m := by
  have hcont : ContinuousOn f (Iic a) := by
    intro x hx
    rcases hx.out.eq_or_lt with rfl|hx
    · exact hcont
    · exact (hderiv x hx).continuousAt.continuousWithinAt
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Iic a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    hcont : ContinuousOn f (Set.Iic a)
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine tendsto_nhds_unique (intervalIntegral_tendsto_integral_Iic a f'int tendsto_id) ?_
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Iic a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    hcont : ContinuousOn f (Set.Iic a)
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => f' x) (id i) a MeasureTh …
  -/
  apply Tendsto.congr' _ (hf.const_sub _)
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Iic a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    hcont : ContinuousOn f (Set.Iic a)
    ⊢ Filter.atBot.EventuallyEq (fun k => HSub.hSub (f a) (f k)) fun i => interval …
  -/
  filter_upwards [Iic_mem_atBot a] with x hx
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Iic a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    hcont : ContinuousOn f (Set.Iic a)
    x : Real
    hx : Membership.mem (Set.Iic a) x
    ⊢ Eq (HSub.hSub (f a) (f x)) (intervalIntegral (fun x => f' x) (id x) a Measur …
  -/
  symm
  apply intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hx
    (hcont.mono Icc_subset_Iic_self) fun y hy => hderiv y hy.2
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Iic a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    hcont : ContinuousOn f (Set.Iic a)
    x : Real
    hx : Membership.mem (Set.Iic a) x
    ⊢ IntervalIntegrable f' MeasureTheory.MeasureSpace.volume x a
  -/
  rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hx]
  /-
    case h
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hcont✝ : ContinuousWithinAt f (Set.Iic a) a
    hderiv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    hcont : ContinuousOn f (Set.Iic a)
    x : Real
    hx : Membership.mem (Set.Iic a) x
    ⊢ MeasureTheory.IntegrableOn f' (Set.Ioc x a) MeasureTheory.MeasureSpace.volume
  -/
  exact f'int.mono (fun y hy => hy.2) le_rfl
  /-
    🎉 no goals
  -/


/-- **Fundamental theorem of calculus-2**, on semi-infinite intervals `(-∞, a)`.
When a function has a limit `m` at `-∞`, and its derivative is integrable, then the
integral of the derivative on `(-∞, a)` is `f a - m`. Version assuming differentiability
on `(-∞, a]`.

Note that such a function always has a limit at minus infinity,
see `tendsto_limUnder_of_hasDerivAt_of_integrableOn_Iic`. -/
theorem integral_Iic_of_hasDerivAt_of_tendsto'
                                                           /-
                                                             E : Type u_1
                                                             f f' : Real → E
                                                             a : Real
                                                             m : E
                                                             inst✝² : NormedAddCommGroup E
                                                             inst✝¹ : NormedSpace Real E
                                                             inst✝ : CompleteSpace E
                                                             hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
                                                             ⊢ MeasureTheory.Measure Real
                                                           -/
    (hderiv : ∀ x ∈ Iic a, HasDerivAt f (f' x) x) (f'int : IntegrableOn f' (Iic a))
                                                           /-
                                                             🎉 no goals
                                                           -/
    (hf : Tendsto f atBot (𝓝 m)) : ∫ x in Iic a, f' x = f a - m := by
  refine integral_Iic_of_hasDerivAt_of_tendsto ?_ (fun x hx => hderiv x hx.out.le)
    f'int hf
  /-
    E : Type u_1
    f f' : Real → E
    a : Real
    m : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), Membership.mem (Set.Iic a) x → HasDerivAt f (f' x) x
    f'int : MeasureTheory.IntegrableOn f' (Set.Iic a) MeasureTheory.MeasureSpace.v …
    hf : Filter.Tendsto f Filter.atBot (nhds m)
    ⊢ ContinuousWithinAt f (Set.Iic a) a
  -/
  exact (hderiv a right_mem_Iic).continuousAt.continuousWithinAt
  /-
    🎉 no goals
  -/


/-- A special case of `integral_Iic_of_hasDerivAt_of_tendsto` where we assume that `f` is C^1 with
compact support. -/
theorem _root_.HasCompactSupport.integral_Iic_deriv_eq (hf : ContDiff ℝ 1 f)
    (h2f : HasCompactSupport f) (b : ℝ) : ∫ x in Iic b, deriv f x = f b := by
  /-
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : HasCompactSupport f
    b : Real
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have := fun x (_ : x ∈ Iio b) ↦ hf.differentiable le_rfl x |>.hasDerivAt
  /-
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : HasCompactSupport f
    b : Real
    this : ∀ (x : Real), Membership.mem (Set.Iio b) x → HasDerivAt f (deriv f x) x
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [integral_Iic_of_hasDerivAt_of_tendsto hf.continuous.continuousWithinAt this, sub_zero]
    /-
      case f'int
      E : Type u_1
      f : Real → E
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      hf : ContDiff Real 1 f
      h2f : HasCompactSupport f
      b : Real
      this : ∀ (x : Real), Membership.mem (Set.Iio b) x → HasDerivAt f (deriv f x) x
      ⊢ MeasureTheory.IntegrableOn (deriv f) (Set.Iic b) MeasureTheory.MeasureSpace. …
    -/
  · refine hf.continuous_deriv le_rfl |>.integrable_of_hasCompactSupport h2f.deriv |>.integrableOn
    /-
      🎉 no goals
    -/
  /-
    case hf
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : HasCompactSupport f
    b : Real
    this : ∀ (x : Real), Membership.mem (Set.Iio b) x → HasDerivAt f (deriv f x) x
    ⊢ Filter.Tendsto f Filter.atBot (nhds 0)
  -/
  rw [hasCompactSupport_iff_eventuallyEq, Filter.coclosedCompact_eq_cocompact] at h2f
  /-
    case hf
    E : Type u_1
    f : Real → E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hf : ContDiff Real 1 f
    h2f : (Filter.cocompact Real).EventuallyEq f 0
    b : Real
    this : ∀ (x : Real), Membership.mem (Set.Iio b) x → HasDerivAt f (deriv f x) x
    ⊢ Filter.Tendsto f Filter.atBot (nhds 0)
  -/
  exact h2f.filter_mono _root_.atBot_le_cocompact |>.tendsto
  /-
    🎉 no goals
  -/


open UniformSpace in
lemma _root_.HasCompactSupport.ennnorm_le_lintegral_Ici_deriv
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    {f : ℝ → F} (hf : ContDiff ℝ 1 f) (h'f : HasCompactSupport f) (x : ℝ) :
    (‖f x‖₊ : ℝ≥0∞) ≤ ∫⁻ y in Iic x, ‖deriv f y‖₊ := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    hf : ContDiff Real 1 f
    h'f : HasCompactSupport f
    x : Real
    ⊢ LE.le (↑(NNNorm.nnnorm (f x))) (MeasureTheory.lintegral (MeasureTheory.Measu …
  -/
  let I : F →L[ℝ] Completion F := Completion.toComplL
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    hf : ContDiff Real 1 f
    h'f : HasCompactSupport f
    x : Real
    I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
    ⊢ LE.le (↑(NNNorm.nnnorm (f x))) (MeasureTheory.lintegral (MeasureTheory.Measu …
  -/
  let f' : ℝ → Completion F := I ∘ f
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    hf : ContDiff Real 1 f
    h'f : HasCompactSupport f
    x : Real
    I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
    f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
    ⊢ LE.le (↑(NNNorm.nnnorm (f x))) (MeasureTheory.lintegral (MeasureTheory.Measu …
  -/
  have hf' : ContDiff ℝ 1 f' := hf.continuousLinearMap_comp I
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    hf : ContDiff Real 1 f
    h'f : HasCompactSupport f
    x : Real
    I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
    f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
    hf' : ContDiff Real 1 f'
    ⊢ LE.le (↑(NNNorm.nnnorm (f x))) (MeasureTheory.lintegral (MeasureTheory.Measu …
  -/
  have h'f' : HasCompactSupport f' := h'f.comp_left rfl
  have : (‖f' x‖₊ : ℝ≥0∞) ≤ ∫⁻ y in Iic x, ‖deriv f' y‖₊ := by
    rw [← HasCompactSupport.integral_Iic_deriv_eq hf' h'f' x]
    exact ennnorm_integral_le_lintegral_ennnorm _
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    f : Real → F
    hf : ContDiff Real 1 f
    h'f : HasCompactSupport f
    x : Real
    I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
    f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
    hf' : ContDiff Real 1 f'
    h'f' : HasCompactSupport f'
    this : LE.le (↑(NNNorm.nnnorm (f' x))) (MeasureTheory.lintegral (MeasureTheory …
    ⊢ LE.le (↑(NNNorm.nnnorm (f x))) (MeasureTheory.lintegral (MeasureTheory.Measu …
  -/
  convert this with y
    /-
      case h.e'_3.h.e'_1
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : Real → F
      hf : ContDiff Real 1 f
      h'f : HasCompactSupport f
      x : Real
      I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
      f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
      hf' : ContDiff Real 1 f'
      h'f' : HasCompactSupport f'
      this : LE.le (↑(NNNorm.nnnorm (f' x))) (MeasureTheory.lintegral (MeasureTheory …
      ⊢ Eq (NNNorm.nnnorm (f x)) (NNNorm.nnnorm (f' x))
    -/
  · simp [f', I, Completion.nnnorm_coe]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_4.h.h.e'_1
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : Real → F
      hf : ContDiff Real 1 f
      h'f : HasCompactSupport f
      x : Real
      I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
      f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
      hf' : ContDiff Real 1 f'
      h'f' : HasCompactSupport f'
      this : LE.le (↑(NNNorm.nnnorm (f' x))) (MeasureTheory.lintegral (MeasureTheory …
      y : Real
      ⊢ Eq (NNNorm.nnnorm (deriv f y)) (NNNorm.nnnorm (deriv f' y))
    -/
  · rw [fderiv_comp_deriv _ I.differentiableAt (hf.differentiable le_rfl _)]
    /-
      case h.e'_4.h.e'_4.h.h.e'_1
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : Real → F
      hf : ContDiff Real 1 f
      h'f : HasCompactSupport f
      x : Real
      I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
      f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
      hf' : ContDiff Real 1 f'
      h'f' : HasCompactSupport f'
      this : LE.le (↑(NNNorm.nnnorm (f' x))) (MeasureTheory.lintegral (MeasureTheory …
      y : Real
      ⊢ Eq (NNNorm.nnnorm (deriv f y)) (NNNorm.nnnorm ((fderiv Real (⇑I) (f y)) (der …
    -/
    simp only [ContinuousLinearMap.fderiv]
    /-
      case h.e'_4.h.e'_4.h.h.e'_1
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      f : Real → F
      hf : ContDiff Real 1 f
      h'f : HasCompactSupport f
      x : Real
      I : ContinuousLinearMap (RingHom.id Real) F (UniformSpace.Completion F) := Uni …
      f' : Real → UniformSpace.Completion F := Function.comp (⇑I) f
      hf' : ContDiff Real 1 f'
      h'f' : HasCompactSupport f'
      this : LE.le (↑(NNNorm.nnnorm (f' x))) (MeasureTheory.lintegral (MeasureTheory …
      y : Real
      ⊢ Eq (NNNorm.nnnorm (deriv f y)) (NNNorm.nnnorm (I (deriv f y)))
    -/
    simp [I]
    /-
      🎉 no goals
    -/


/-- **Fundamental theorem of calculus-2**, on the whole real line
When a function has a limit `m` at `-∞` and `n` at `+∞`, and its derivative is integrable, then the
integral of the derivative is `n - m`.

Note that such a function always has a limit at `-∞` and `+∞`,
see `tendsto_limUnder_of_hasDerivAt_of_integrableOn_Iic` and
`tendsto_limUnder_of_hasDerivAt_of_integrableOn_Ioi`. -/
theorem integral_of_hasDerivAt_of_tendsto [CompleteSpace E]
                                                 /-
                                                   E : Type u_1
                                                   f f' : Real → E
                                                   m n : E
                                                   inst✝² : NormedAddCommGroup E
                                                   inst✝¹ : NormedSpace Real E
                                                   inst✝ : CompleteSpace E
                                                   hderiv : ∀ (x : Real), HasDerivAt f (f' x) x
                                                   ⊢ MeasureTheory.Measure Real
                                                 -/
    (hderiv : ∀ x, HasDerivAt f (f' x) x) (hf' : Integrable f')
                                                 /-
                                                   🎉 no goals
                                                 -/
    (hbot : Tendsto f atBot (𝓝 m)) (htop : Tendsto f atTop (𝓝 n)) : ∫ x, f' x = n - m := by
  rw [← setIntegral_univ, ← Set.Iic_union_Ioi (a := 0),
    setIntegral_union (Iic_disjoint_Ioi le_rfl) measurableSet_Ioi hf'.integrableOn hf'.integrableOn,
    integral_Iic_of_hasDerivAt_of_tendsto' (fun x _ ↦ hderiv x) hf'.integrableOn hbot,
    integral_Ioi_of_hasDerivAt_of_tendsto' (fun x _ ↦ hderiv x) hf'.integrableOn htop]
  /-
    E : Type u_1
    f f' : Real → E
    m n : E
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    hderiv : ∀ (x : Real), HasDerivAt f (f' x) x
    hf' : MeasureTheory.Integrable f' MeasureTheory.MeasureSpace.volume
    hbot : Filter.Tendsto f Filter.atBot (nhds m)
    htop : Filter.Tendsto f Filter.atTop (nhds n)
    ⊢ Eq (HAdd.hAdd (HSub.hSub (f 0) m) (HSub.hSub n (f 0))) (HSub.hSub n m)
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- If a function and its derivative are integrable on the real line, then the integral of the
derivative is zero. -/
theorem integral_eq_zero_of_hasDerivAt_of_integrable
                                                 /-
                                                   E : Type u_1
                                                   f f' : Real → E
                                                   m n : E
                                                   inst✝¹ : NormedAddCommGroup E
                                                   inst✝ : NormedSpace Real E
                                                   hderiv : ∀ (x : Real), HasDerivAt f (f' x) x
                                                   ⊢ MeasureTheory.Measure Real
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    (hderiv : ∀ x, HasDerivAt f (f' x) x) (hf' : Integrable f') (hf : Integrable f) :
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    ∫ x, f' x = 0 := by
  /-
    E : Type u_1
    f f' : Real → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), HasDerivAt f (f' x) x
    hf' : MeasureTheory.Integrable f' MeasureTheory.MeasureSpace.volume
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => f' x) 0
  -/
  by_cases hE : CompleteSpace E; swap
    /-
      case neg
      E : Type u_1
      f f' : Real → E
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      hderiv : ∀ (x : Real), HasDerivAt f (f' x) x
      hf' : MeasureTheory.Integrable f' MeasureTheory.MeasureSpace.volume
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      hE : Not (CompleteSpace E)
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => f' x) 0
    -/
  · simp [integral, hE]
    /-
      🎉 no goals
    -/
  have A : Tendsto f atBot (𝓝 0) :=
    tendsto_zero_of_hasDerivAt_of_integrableOn_Iic (a := 0) (fun x _hx ↦ hderiv x)
      hf'.integrableOn hf.integrableOn
  have B : Tendsto f atTop (𝓝 0) :=
    tendsto_zero_of_hasDerivAt_of_integrableOn_Ioi (a := 0) (fun x _hx ↦ hderiv x)
      hf'.integrableOn hf.integrableOn
  /-
    case pos
    E : Type u_1
    f f' : Real → E
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    hderiv : ∀ (x : Real), HasDerivAt f (f' x) x
    hf' : MeasureTheory.Integrable f' MeasureTheory.MeasureSpace.volume
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hE : CompleteSpace E
    A : Filter.Tendsto f Filter.atBot (nhds 0)
    B : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => f' x) 0
  -/
  simpa using integral_of_hasDerivAt_of_tendsto hderiv hf' A B
  /-
    🎉 no goals
  -/


/-- Change-of-variables formula for `Ioi` integrals of vector-valued functions, proved by taking
limits from the result for finite intervals. -/
theorem integral_comp_smul_deriv_Ioi {f f' : ℝ → ℝ} {g : ℝ → E} {a : ℝ}
    (hf : ContinuousOn f <| Ici a) (hft : Tendsto f atTop atTop)
    (hff' : ∀ x ∈ Ioi a, HasDerivWithinAt f (f' x) (Ioi x) x)
    (hg_cont : ContinuousOn g <| f '' Ioi a) (hg1 : IntegrableOn g <| f '' Ici a)
           /-
             E : Type u_1
             inst✝¹ : NormedAddCommGroup E
             inst✝ : NormedSpace Real E
             f f' : Real → Real
             g : Real → E
             a : Real
             hf : ContinuousOn f (Set.Ici a)
             hft : Filter.Tendsto f Filter.atTop Filter.atTop
             hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
             hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
             hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
             ⊢ MeasureTheory.Measure Real
           -/
    (hg2 : IntegrableOn (fun x => f' x • (g ∘ f) x) (Ici a)) :
           /-
             🎉 no goals
           -/
    (∫ x in Ioi a, f' x • (g ∘ f) x) = ∫ u in Ioi (f a), g u := by
  have eq : ∀ b : ℝ, a < b → (∫ x in a..b, f' x • (g ∘ f) x) = ∫ u in f a..f b, g u := fun b hb ↦ by
    have i1 : Ioo (min a b) (max a b) ⊆ Ioi a := by
      rw [min_eq_left hb.le]
      exact Ioo_subset_Ioi_self
    have i2 : [[a, b]] ⊆ Ici a := by rw [uIcc_of_le hb.le]; exact Icc_subset_Ici_self
    refine
      intervalIntegral.integral_comp_smul_deriv''' (hf.mono i2)
        (fun x hx => hff' x <| mem_of_mem_of_subset hx i1) (hg_cont.mono <| image_subset _ ?_)
        (hg1.mono_set <| image_subset _ ?_) (hg2.mono_set i2)
    · rw [min_eq_left hb.le]; exact Ioo_subset_Ioi_self
    · rw [uIcc_of_le hb.le]; exact Icc_subset_Ici_self
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f f' : Real → Real
    g : Real → E
    a : Real
    hf : ContinuousOn f (Set.Ici a)
    hft : Filter.Tendsto f Filter.atTop Filter.atTop
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
    hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
    hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
    hg2 : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f' x) (Function.comp g …
    eq : ∀ (b : Real), LT.lt a b → Eq (intervalIntegral (fun x => HSMul.hSMul (f'  …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [integrableOn_Ici_iff_integrableOn_Ioi] at hg2
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f f' : Real → Real
    g : Real → E
    a : Real
    hf : ContinuousOn f (Set.Ici a)
    hft : Filter.Tendsto f Filter.atTop Filter.atTop
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
    hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
    hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
    hg2 : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f' x) (Function.comp g …
    eq : ∀ (b : Real), LT.lt a b → Eq (intervalIntegral (fun x => HSMul.hSMul (f'  …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have t2 := intervalIntegral_tendsto_integral_Ioi _ hg2 tendsto_id
  have : Ioi (f a) ⊆ f '' Ici a :=
    Ioi_subset_Ici_self.trans <|
      IsPreconnected.intermediate_value_Ici isPreconnected_Ici left_mem_Ici
        (le_principal_iff.mpr <| Ici_mem_atTop _) hf hft
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f f' : Real → Real
    g : Real → E
    a : Real
    hf : ContinuousOn f (Set.Ici a)
    hft : Filter.Tendsto f Filter.atTop Filter.atTop
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
    hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
    hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
    hg2 : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f' x) (Function.comp g …
    eq : ∀ (b : Real), LT.lt a b → Eq (intervalIntegral (fun x => HSMul.hSMul (f'  …
    t2 : Filter.Tendsto (fun i => intervalIntegral (fun x => HSMul.hSMul (f' x) (F …
    this : HasSubset.Subset (Set.Ioi (f a)) (Set.image f (Set.Ici a))
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have t1 := (intervalIntegral_tendsto_integral_Ioi _ (hg1.mono_set this) tendsto_id).comp hft
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f f' : Real → Real
    g : Real → E
    a : Real
    hf : ContinuousOn f (Set.Ici a)
    hft : Filter.Tendsto f Filter.atTop Filter.atTop
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
    hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
    hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
    hg2 : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f' x) (Function.comp g …
    eq : ∀ (b : Real), LT.lt a b → Eq (intervalIntegral (fun x => HSMul.hSMul (f'  …
    t2 : Filter.Tendsto (fun i => intervalIntegral (fun x => HSMul.hSMul (f' x) (F …
    this : HasSubset.Subset (Set.Ioi (f a)) (Set.image f (Set.Ici a))
    t1 : Filter.Tendsto (Function.comp (fun i => intervalIntegral (fun x => g x) ( …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  exact tendsto_nhds_unique (Tendsto.congr' (eventuallyEq_of_mem (Ioi_mem_atTop a) eq) t2) t1
  /-
    🎉 no goals
  -/


/-- Change-of-variables formula for `Ioi` integrals of scalar-valued functions -/
theorem integral_comp_mul_deriv_Ioi {f f' : ℝ → ℝ} {g : ℝ → ℝ} {a : ℝ}
    (hf : ContinuousOn f <| Ici a) (hft : Tendsto f atTop atTop)
    (hff' : ∀ x ∈ Ioi a, HasDerivWithinAt f (f' x) (Ioi x) x)
    (hg_cont : ContinuousOn g <| f '' Ioi a) (hg1 : IntegrableOn g <| f '' Ici a)
           /-
             E : Type u_1
             inst✝¹ : NormedAddCommGroup E
             inst✝ : NormedSpace Real E
             f f' g : Real → Real
             a : Real
             hf : ContinuousOn f (Set.Ici a)
             hft : Filter.Tendsto f Filter.atTop Filter.atTop
             hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
             hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
             hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
             ⊢ MeasureTheory.Measure Real
           -/
    (hg2 : IntegrableOn (fun x => (g ∘ f) x * f' x) (Ici a)) :
           /-
             🎉 no goals
           -/
    (∫ x in Ioi a, (g ∘ f) x * f' x) = ∫ u in Ioi (f a), g u := by
  /-
    f f' g : Real → Real
    a : Real
    hf : ContinuousOn f (Set.Ici a)
    hft : Filter.Tendsto f Filter.atTop Filter.atTop
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
    hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
    hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
    hg2 : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Function.comp g f x) (f' …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have hg2' : IntegrableOn (fun x => f' x • (g ∘ f) x) (Ici a) := by simpa [mul_comm] using hg2
  /-
    f f' g : Real → Real
    a : Real
    hf : ContinuousOn f (Set.Ici a)
    hft : Filter.Tendsto f Filter.atTop Filter.atTop
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivWithinAt f (f' x)  …
    hg_cont : ContinuousOn g (Set.image f (Set.Ioi a))
    hg1 : MeasureTheory.IntegrableOn g (Set.image f (Set.Ici a)) MeasureTheory.Mea …
    hg2 : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Function.comp g f x) (f' …
    hg2' : MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f' x) (Function.comp  …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simpa [mul_comm] using integral_comp_smul_deriv_Ioi hf hft hff' hg_cont hg1 hg2'
  /-
    🎉 no goals
  -/


/-- Substitution `y = x ^ p` in integrals over `Ioi 0` -/
theorem integral_comp_rpow_Ioi (g : ℝ → E) {p : ℝ} (hp : p ≠ 0) :
    (∫ x in Ioi 0, (|p| * x ^ (p - 1)) • g (x ^ p)) = ∫ y in Ioi 0, g y := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : Ne p 0
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  let S := Ioi (0 : ℝ)
  have a1 : ∀ x : ℝ, x ∈ S → HasDerivWithinAt (fun t : ℝ => t ^ p) (p * x ^ (p - 1)) S x :=
    fun x hx => (hasDerivAt_rpow_const (Or.inl (mem_Ioi.mp hx).ne')).hasDerivWithinAt
  have a2 : InjOn (fun x : ℝ => x ^ p) S := by
    rcases lt_or_gt_of_ne hp with (h | h)
    · apply StrictAntiOn.injOn
      intro x hx y hy hxy
      rw [← inv_lt_inv₀ (rpow_pos_of_pos hx p) (rpow_pos_of_pos hy p), ← rpow_neg (le_of_lt hx),
        ← rpow_neg (le_of_lt hy)]
      exact rpow_lt_rpow (le_of_lt hx) hxy (neg_pos.mpr h)
    exact StrictMonoOn.injOn fun x hx y _ hxy => rpow_lt_rpow (mem_Ioi.mp hx).le hxy h
  have a3 : (fun t : ℝ => t ^ p) '' S = S := by
    ext1 x; rw [mem_image]; constructor
    · rintro ⟨y, hy, rfl⟩; exact rpow_pos_of_pos hy p
    · intro hx; refine ⟨x ^ (1 / p), rpow_pos_of_pos hx _, ?_⟩
      rw [← rpow_mul (le_of_lt hx), one_div_mul_cancel hp, rpow_one]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have := integral_image_eq_integral_abs_deriv_smul measurableSet_Ioi a1 a2 g
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [a3] at this; rw [this]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioi ?_
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
    ⊢ Set.EqOn (fun x => HSMul.hSMul (HMul.hMul (abs p) (HPow.hPow x (HSub.hSub p  …
  -/
  intro x hx; dsimp only
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HSMul.hSMul (HMul.hMul (abs p) (HPow.hPow x (HSub.hSub p 1))) (g (HPow.h …
  -/
  rw [abs_mul, abs_of_nonneg (rpow_nonneg (le_of_lt hx) _)]
  /-
    🎉 no goals
  -/


theorem integral_comp_rpow_Ioi_of_pos {g : ℝ → E} {p : ℝ} (hp : 0 < p) :
    (∫ x in Ioi 0, (p * x ^ (p - 1)) • g (x ^ p)) = ∫ y in Ioi 0, g y := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : LT.lt 0 p
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  convert integral_comp_rpow_Ioi g hp.ne'
  /-
    case h.e'_2.h.e'_7.h.h.e'_5.h.e'_5
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    p : Real
    hp : LT.lt 0 p
    x✝ : Real
    ⊢ Eq p (abs p)
  -/
  rw [abs_of_nonneg hp.le]
  /-
    🎉 no goals
  -/


theorem integral_comp_mul_left_Ioi (g : ℝ → E) (a : ℝ) {b : ℝ} (hb : 0 < b) :
    (∫ x in Ioi a, g (b * x)) = b⁻¹ • ∫ x in Ioi (b * a), g x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    a b : Real
    hb : LT.lt 0 b
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  have : ∀ c : ℝ, MeasurableSet (Ioi c) := fun c => measurableSet_Ioi
  rw [← integral_indicator (this a), ← integral_indicator (this (b * a)),
    ← abs_of_pos (inv_pos.mpr hb), ← Measure.integral_comp_mul_left]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    a b : Real
    hb : LT.lt 0 b
    this : ∀ (c : Real), MeasurableSet (Set.Ioi c)
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (Set.I …
  -/
  congr
  /-
    case e_f
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    a b : Real
    hb : LT.lt 0 b
    this : ∀ (c : Real), MeasurableSet (Set.Ioi c)
    ⊢ Eq (fun x => (Set.Ioi a).indicator (fun x => g (HMul.hMul b x)) x) fun x =>  …
  -/
  ext1 x
  /-
    case e_f.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    a b : Real
    hb : LT.lt 0 b
    this : ∀ (c : Real), MeasurableSet (Set.Ioi c)
    x : Real
    ⊢ Eq ((Set.Ioi a).indicator (fun x => g (HMul.hMul b x)) x) ((Set.Ioi (HMul.hM …
  -/
  rw [← indicator_comp_right, preimage_const_mul_Ioi _ hb, mul_div_cancel_left₀ _ hb.ne']
  /-
    case e_f.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    a b : Real
    hb : LT.lt 0 b
    this : ∀ (c : Real), MeasurableSet (Set.Ioi c)
    x : Real
    ⊢ Eq ((Set.Ioi a).indicator (fun x => g (HMul.hMul b x)) x) ((Set.Ioi a).indic …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem integral_comp_mul_right_Ioi (g : ℝ → E) (a : ℝ) {b : ℝ} (hb : 0 < b) :
    (∫ x in Ioi a, g (x * b)) = b⁻¹ • ∫ x in Ioi (a * b), g x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    g : Real → E
    a b : Real
    hb : LT.lt 0 b
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simpa only [mul_comm] using integral_comp_mul_left_Ioi g a hb
  /-
    🎉 no goals
  -/


/-- The substitution `y = x ^ p` in integrals over `Ioi 0` preserves integrability. -/
theorem integrableOn_Ioi_comp_rpow_iff [NormedSpace ℝ E] (f : ℝ → E) {p : ℝ} (hp : p ≠ 0) :
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      p : Real
      hp : Ne p 0
      ⊢ MeasureTheory.Measure Real
    -/
    /-
      🎉 no goals
    -/
    IntegrableOn (fun x => (|p| * x ^ (p - 1)) • f (x ^ p)) (Ioi 0) ↔ IntegrableOn f (Ioi 0) := by
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    p : Real
    hp : Ne p 0
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (HMul.hMul (abs p) (HP …
  -/
  let S := Ioi (0 : ℝ)
  have a1 : ∀ x : ℝ, x ∈ S → HasDerivWithinAt (fun t : ℝ => t ^ p) (p * x ^ (p - 1)) S x :=
    fun x hx => (hasDerivAt_rpow_const (Or.inl (mem_Ioi.mp hx).ne')).hasDerivWithinAt
  have a2 : InjOn (fun x : ℝ => x ^ p) S := by
    rcases lt_or_gt_of_ne hp with (h | h)
    · apply StrictAntiOn.injOn
      intro x hx y hy hxy
      rw [← inv_lt_inv₀ (rpow_pos_of_pos hx p) (rpow_pos_of_pos hy p), ← rpow_neg (le_of_lt hx), ←
        rpow_neg (le_of_lt hy)]
      exact rpow_lt_rpow (le_of_lt hx) hxy (neg_pos.mpr h)
    exact StrictMonoOn.injOn fun x hx y _hy hxy => rpow_lt_rpow (mem_Ioi.mp hx).le hxy h
  have a3 : (fun t : ℝ => t ^ p) '' S = S := by
    ext1 x; rw [mem_image]; constructor
    · rintro ⟨y, hy, rfl⟩; exact rpow_pos_of_pos hy p
    · intro hx; refine ⟨x ^ (1 / p), rpow_pos_of_pos hx _, ?_⟩
      rw [← rpow_mul (le_of_lt hx), one_div_mul_cancel hp, rpow_one]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (HMul.hMul (abs p) (HP …
  -/
  have := integrableOn_image_iff_integrableOn_abs_deriv_smul measurableSet_Ioi a1 a2 f
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Iff (MeasureTheory.IntegrableOn f (Set.image (fun t => HPow.hPow t p) ( …
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (HMul.hMul (abs p) (HP …
  -/
  rw [a3] at this
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Iff (MeasureTheory.IntegrableOn f S MeasureTheory.MeasureSpace.volume)  …
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (HMul.hMul (abs p) (HP …
  -/
  rw [this]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Iff (MeasureTheory.IntegrableOn f S MeasureTheory.MeasureSpace.volume)  …
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (HMul.hMul (abs p) (HP …
  -/
  refine integrableOn_congr_fun (fun x hx => ?_) measurableSet_Ioi
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    p : Real
    hp : Ne p 0
    S : Set Real := Set.Ioi 0
    a1 : ∀ (x : Real), Membership.mem S x → HasDerivWithinAt (fun t => HPow.hPow t …
    a2 : Set.InjOn (fun x => HPow.hPow x p) S
    a3 : Eq (Set.image (fun t => HPow.hPow t p) S) S
    this : Iff (MeasureTheory.IntegrableOn f S MeasureTheory.MeasureSpace.volume)  …
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HSMul.hSMul (HMul.hMul (abs p) (HPow.hPow x (HSub.hSub p 1))) (f (HPow.h …
  -/
  simp_rw [abs_mul, abs_of_nonneg (rpow_nonneg (le_of_lt hx) _)]
  /-
    🎉 no goals
  -/


/-- The substitution `y = x ^ p` in integrals over `Ioi 0` preserves integrability (version
without `|p|` factor) -/
theorem integrableOn_Ioi_comp_rpow_iff' [NormedSpace ℝ E] (f : ℝ → E) {p : ℝ} (hp : p ≠ 0) :
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      p : Real
      hp : Ne p 0
      ⊢ MeasureTheory.Measure Real
    -/
    /-
      🎉 no goals
    -/
    IntegrableOn (fun x => x ^ (p - 1) • f (x ^ p)) (Ioi 0) ↔ IntegrableOn f (Ioi 0) := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  simpa only [← integrableOn_Ioi_comp_rpow_iff f hp, mul_smul] using
    (integrable_smul_iff (abs_pos.mpr hp).ne' _).symm


theorem integrableOn_Ioi_comp_mul_left_iff (f : ℝ → E) (c : ℝ) {a : ℝ} (ha : 0 < a) :
    /-
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Real → E
      c a : Real
      ha : LT.lt 0 a
      ⊢ MeasureTheory.Measure Real
    -/
    /-
      🎉 no goals
    -/
    IntegrableOn (fun x => f (a * x)) (Ioi c) ↔ IntegrableOn f (Ioi <| a * c) := by
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Real → E
    c a : Real
    ha : LT.lt 0 a
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => f (HMul.hMul a x)) (Set.Ioi c) Mea …
  -/
  rw [← integrable_indicator_iff (measurableSet_Ioi : MeasurableSet <| Ioi c)]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Real → E
    c a : Real
    ha : LT.lt 0 a
    ⊢ Iff (MeasureTheory.Integrable ((Set.Ioi c).indicator fun x => f (HMul.hMul a …
  -/
  rw [← integrable_indicator_iff (measurableSet_Ioi : MeasurableSet <| Ioi <| a * c)]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Real → E
    c a : Real
    ha : LT.lt 0 a
    ⊢ Iff (MeasureTheory.Integrable ((Set.Ioi c).indicator fun x => f (HMul.hMul a …
  -/
  convert integrable_comp_mul_left_iff ((Ioi (a * c)).indicator f) ha.ne' using 2
  /-
    case h.e'_1.h.e'_6
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Real → E
    c a : Real
    ha : LT.lt 0 a
    ⊢ Eq ((Set.Ioi c).indicator fun x => f (HMul.hMul a x)) fun x => (Set.Ioi (HMu …
  -/
  ext1 x
  rw [← indicator_comp_right, preimage_const_mul_Ioi _ ha, mul_comm a c,
    mul_div_cancel_right₀ _ ha.ne']
  /-
    case h.e'_1.h.e'_6.h
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Real → E
    c a : Real
    ha : LT.lt 0 a
    x : Real
    ⊢ Eq ((Set.Ioi c).indicator (fun x => f (HMul.hMul a x)) x) ((Set.Ioi c).indic …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem integrableOn_Ioi_comp_mul_right_iff (f : ℝ → E) (c : ℝ) {a : ℝ} (ha : 0 < a) :
    /-
      E : Type u_1
      inst✝ : NormedAddCommGroup E
      f : Real → E
      c a : Real
      ha : LT.lt 0 a
      ⊢ MeasureTheory.Measure Real
    -/
    /-
      🎉 no goals
    -/
    IntegrableOn (fun x => f (x * a)) (Ioi c) ↔ IntegrableOn f (Ioi <| c * a) := by
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    f : Real → E
    c a : Real
    ha : LT.lt 0 a
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => f (HMul.hMul x a)) (Set.Ioi c) Mea …
  -/
  simpa only [mul_comm, mul_zero] using integrableOn_Ioi_comp_mul_left_iff f c ha
  /-
    🎉 no goals
  -/


theorem integral_bilinear_hasDerivAt_eq_sub [CompleteSpace G]
    (hu : ∀ x, HasDerivAt u (u' x) x) (hv : ∀ x, HasDerivAt v (v' x) x)
           /-
             E : Type u_1
             F : Type u_2
             G : Type u_3
             inst✝⁶ : NormedAddCommGroup E
             inst✝⁵ : NormedSpace Real E
             inst✝⁴ : NormedAddCommGroup F
             inst✝³ : NormedSpace Real F
             inst✝² : NormedAddCommGroup G
             inst✝¹ : NormedSpace Real G
             L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
             u : Real → E
             v : Real → F
             u' : Real → E
             v' : Real → F
             m n : G
             inst✝ : CompleteSpace G
             hu : ∀ (x : Real), HasDerivAt u (u' x) x
             hv : ∀ (x : Real), HasDerivAt v (v' x) x
             ⊢ MeasureTheory.Measure Real
           -/
    (huv : Integrable (fun x ↦ L (u x) (v' x) + L (u' x) (v x)))
           /-
             🎉 no goals
           -/
    (h_bot : Tendsto (fun x ↦ L (u x) (v x)) atBot (𝓝 m))
    (h_top : Tendsto (fun x ↦ L (u x) (v x)) atTop (𝓝 n)) :
    ∫ (x : ℝ), L (u x) (v' x) + L (u' x) (v x) = n - m :=
  integral_of_hasDerivAt_of_tendsto (fun x ↦ L.hasDerivAt_of_bilinear (hu x) (hv x))
    huv h_bot h_top


/-- **Integration by parts on (-∞, ∞).**
With respect to a general bilinear form. For the specific case of multiplication, see
`integral_mul_deriv_eq_deriv_mul`. -/
theorem integral_bilinear_hasDerivAt_right_eq_sub [CompleteSpace G]
    (hu : ∀ x, HasDerivAt u (u' x) x) (hv : ∀ x, HasDerivAt v (v' x) x)
            /-
              E : Type u_1
              F : Type u_2
              G : Type u_3
              inst✝⁶ : NormedAddCommGroup E
              inst✝⁵ : NormedSpace Real E
              inst✝⁴ : NormedAddCommGroup F
              inst✝³ : NormedSpace Real F
              inst✝² : NormedAddCommGroup G
              inst✝¹ : NormedSpace Real G
              L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
              u : Real → E
              v : Real → F
              u' : Real → E
              v' : Real → F
              m n : G
              inst✝ : CompleteSpace G
              hu : ∀ (x : Real), HasDerivAt u (u' x) x
              hv : ∀ (x : Real), HasDerivAt v (v' x) x
              ⊢ MeasureTheory.Measure Real
            -/
            /-
              🎉 no goals
            -/
    (huv' : Integrable (fun x ↦ L (u x) (v' x))) (hu'v : Integrable (fun x ↦ L (u' x) (v x)))
                                                         /-
                                                           🎉 no goals
                                                         -/
    (h_bot : Tendsto (fun x ↦ L (u x) (v x)) atBot (𝓝 m))
    (h_top : Tendsto (fun x ↦ L (u x) (v x)) atTop (𝓝 n)) :
    ∫ (x : ℝ), L (u x) (v' x) = n - m - ∫ (x : ℝ), L (u' x) (v x) := by
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    u : Real → E
    v : Real → F
    u' : Real → E
    v' : Real → F
    m n : G
    inst✝ : CompleteSpace G
    hu : ∀ (x : Real), HasDerivAt u (u' x) x
    hv : ∀ (x : Real), HasDerivAt v (v' x) x
    huv' : MeasureTheory.Integrable (fun x => (L (u x)) (v' x)) MeasureTheory.Meas …
    hu'v : MeasureTheory.Integrable (fun x => (L (u' x)) (v x)) MeasureTheory.Meas …
    h_bot : Filter.Tendsto (fun x => (L (u x)) (v x)) Filter.atBot (nhds m)
    h_top : Filter.Tendsto (fun x => (L (u x)) (v x)) Filter.atTop (nhds n)
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (L (u  …
  -/
  rw [eq_sub_iff_add_eq, ← integral_add huv' hu'v]
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace Real G
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    u : Real → E
    v : Real → F
    u' : Real → E
    v' : Real → F
    m n : G
    inst✝ : CompleteSpace G
    hu : ∀ (x : Real), HasDerivAt u (u' x) x
    hv : ∀ (x : Real), HasDerivAt v (v' x) x
    huv' : MeasureTheory.Integrable (fun x => (L (u x)) (v' x)) MeasureTheory.Meas …
    hu'v : MeasureTheory.Integrable (fun x => (L (u' x)) (v x)) MeasureTheory.Meas …
    h_bot : Filter.Tendsto (fun x => (L (u x)) (v x)) Filter.atBot (nhds m)
    h_top : Filter.Tendsto (fun x => (L (u x)) (v x)) Filter.atTop (nhds n)
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun a => HAdd.h …
  -/
  exact integral_bilinear_hasDerivAt_eq_sub hu hv (huv'.add hu'v) h_bot h_top
  /-
    🎉 no goals
  -/


/-- **Integration by parts on (-∞, ∞).**
With respect to a general bilinear form, assuming moreover that the total function is integrable.
-/
theorem integral_bilinear_hasDerivAt_right_eq_neg_left_of_integrable
    (hu : ∀ x, HasDerivAt u (u' x) x) (hv : ∀ x, HasDerivAt v (v' x) x)
            /-
              E : Type u_1
              F : Type u_2
              G : Type u_3
              inst✝⁵ : NormedAddCommGroup E
              inst✝⁴ : NormedSpace Real E
              inst✝³ : NormedAddCommGroup F
              inst✝² : NormedSpace Real F
              inst✝¹ : NormedAddCommGroup G
              inst✝ : NormedSpace Real G
              L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
              u : Real → E
              v : Real → F
              u' : Real → E
              v' : Real → F
              m n : G
              hu : ∀ (x : Real), HasDerivAt u (u' x) x
              hv : ∀ (x : Real), HasDerivAt v (v' x) x
              ⊢ MeasureTheory.Measure Real
            -/
            /-
              🎉 no goals
            -/
    (huv' : Integrable (fun x ↦ L (u x) (v' x))) (hu'v : Integrable (fun x ↦ L (u' x) (v x)))
                                                         /-
                                                           🎉 no goals
                                                         -/
           /-
             E : Type u_1
             F : Type u_2
             G : Type u_3
             inst✝⁵ : NormedAddCommGroup E
             inst✝⁴ : NormedSpace Real E
             inst✝³ : NormedAddCommGroup F
             inst✝² : NormedSpace Real F
             inst✝¹ : NormedAddCommGroup G
             inst✝ : NormedSpace Real G
             L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
             u : Real → E
             v : Real → F
             u' : Real → E
             v' : Real → F
             m n : G
             hu : ∀ (x : Real), HasDerivAt u (u' x) x
             hv : ∀ (x : Real), HasDerivAt v (v' x) x
             huv' : MeasureTheory.Integrable (fun x => (L (u x)) (v' x)) MeasureTheory.Meas …
             hu'v : MeasureTheory.Integrable (fun x => (L (u' x)) (v x)) MeasureTheory.Meas …
             ⊢ MeasureTheory.Measure Real
           -/
    (huv : Integrable (fun x ↦ L (u x) (v x))) :
           /-
             🎉 no goals
           -/
    ∫ (x : ℝ), L (u x) (v' x) = - ∫ (x : ℝ), L (u' x) (v x) := by
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    u : Real → E
    v : Real → F
    u' : Real → E
    v' : Real → F
    hu : ∀ (x : Real), HasDerivAt u (u' x) x
    hv : ∀ (x : Real), HasDerivAt v (v' x) x
    huv' : MeasureTheory.Integrable (fun x => (L (u x)) (v' x)) MeasureTheory.Meas …
    hu'v : MeasureTheory.Integrable (fun x => (L (u' x)) (v x)) MeasureTheory.Meas …
    huv : MeasureTheory.Integrable (fun x => (L (u x)) (v x)) MeasureTheory.Measur …
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (L (u  …
  -/
  by_cases hG : CompleteSpace G; swap
    /-
      case neg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace Real G
      L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
      u : Real → E
      v : Real → F
      u' : Real → E
      v' : Real → F
      hu : ∀ (x : Real), HasDerivAt u (u' x) x
      hv : ∀ (x : Real), HasDerivAt v (v' x) x
      huv' : MeasureTheory.Integrable (fun x => (L (u x)) (v' x)) MeasureTheory.Meas …
      hu'v : MeasureTheory.Integrable (fun x => (L (u' x)) (v x)) MeasureTheory.Meas …
      huv : MeasureTheory.Integrable (fun x => (L (u x)) (v x)) MeasureTheory.Measur …
      hG : Not (CompleteSpace G)
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (L (u  …
    -/
  · simp [integral, hG]
    /-
      🎉 no goals
    -/
  have I : Tendsto (fun x ↦ L (u x) (v x)) atBot (𝓝 0) :=
    tendsto_zero_of_hasDerivAt_of_integrableOn_Iic (a := 0)
      (fun x _hx ↦ L.hasDerivAt_of_bilinear (hu x) (hv x))
      (huv'.add hu'v).integrableOn huv.integrableOn
  have J : Tendsto (fun x ↦ L (u x) (v x)) atTop (𝓝 0) :=
    tendsto_zero_of_hasDerivAt_of_integrableOn_Ioi (a := 0)
      (fun x _hx ↦ L.hasDerivAt_of_bilinear (hu x) (hv x))
      (huv'.add hu'v).integrableOn huv.integrableOn
  /-
    case pos
    E : Type u_1
    F : Type u_2
    G : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace Real G
    L : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id R …
    u : Real → E
    v : Real → F
    u' : Real → E
    v' : Real → F
    hu : ∀ (x : Real), HasDerivAt u (u' x) x
    hv : ∀ (x : Real), HasDerivAt v (v' x) x
    huv' : MeasureTheory.Integrable (fun x => (L (u x)) (v' x)) MeasureTheory.Meas …
    hu'v : MeasureTheory.Integrable (fun x => (L (u' x)) (v x)) MeasureTheory.Meas …
    huv : MeasureTheory.Integrable (fun x => (L (u x)) (v x)) MeasureTheory.Measur …
    hG : CompleteSpace G
    I : Filter.Tendsto (fun x => (L (u x)) (v x)) Filter.atBot (nhds 0)
    J : Filter.Tendsto (fun x => (L (u x)) (v x)) Filter.atTop (nhds 0)
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun x => (L (u  …
  -/
  simp [integral_bilinear_hasDerivAt_right_eq_sub hu hv huv' hu'v I J]
  /-
    🎉 no goals
  -/


/-- For finite intervals, see: `intervalIntegral.integral_deriv_mul_eq_sub`. -/
theorem integral_deriv_mul_eq_sub [CompleteSpace A]
    (hu : ∀ x, HasDerivAt u (u' x) x) (hv : ∀ x, HasDerivAt v (v' x) x)
           /-
             A : Type u_1
             inst✝² : NormedRing A
             inst✝¹ : NormedAlgebra Real A
             a : Real
             a' b' : A
             u v u' v' : Real → A
             inst✝ : CompleteSpace A
             hu : ∀ (x : Real), HasDerivAt u (u' x) x
             hv : ∀ (x : Real), HasDerivAt v (v' x) x
             ⊢ MeasureTheory.Measure Real
           -/
    (huv : Integrable (u' * v + u * v'))
           /-
             🎉 no goals
           -/
    (h_bot : Tendsto (u * v) atBot (𝓝 a')) (h_top : Tendsto (u * v) atTop (𝓝 b')) :
    ∫ (x : ℝ), u' x * v x + u x * v' x = b' - a' :=
  integral_of_hasDerivAt_of_tendsto (fun x ↦ (hu x).mul (hv x)) huv h_bot h_top


/-- **Integration by parts on (-∞, ∞).**
For finite intervals, see: `intervalIntegral.integral_mul_deriv_eq_deriv_mul`. -/
theorem integral_mul_deriv_eq_deriv_mul [CompleteSpace A]
    (hu : ∀ x, HasDerivAt u (u' x) x) (hv : ∀ x, HasDerivAt v (v' x) x)
            /-
              A : Type u_1
              inst✝² : NormedRing A
              inst✝¹ : NormedAlgebra Real A
              a : Real
              a' b' : A
              u v u' v' : Real → A
              inst✝ : CompleteSpace A
              hu : ∀ (x : Real), HasDerivAt u (u' x) x
              hv : ∀ (x : Real), HasDerivAt v (v' x) x
              ⊢ MeasureTheory.Measure Real
            -/
            /-
              🎉 no goals
            -/
    (huv' : Integrable (u * v')) (hu'v : Integrable (u' * v))
                                         /-
                                           🎉 no goals
                                         -/
    (h_bot : Tendsto (u * v) atBot (𝓝 a')) (h_top : Tendsto (u * v) atTop (𝓝 b')) :
    ∫ (x : ℝ), u x * v' x = b' - a' - ∫ (x : ℝ), u' x * v x :=
  integral_bilinear_hasDerivAt_right_eq_sub (L := ContinuousLinearMap.mul ℝ A)
    hu hv huv' hu'v h_bot h_top


/-- **Integration by parts on (-∞, ∞).**
Version assuming that the total function is integrable -/
theorem integral_mul_deriv_eq_deriv_mul_of_integrable
    (hu : ∀ x, HasDerivAt u (u' x) x) (hv : ∀ x, HasDerivAt v (v' x) x)
            /-
              A : Type u_1
              inst✝¹ : NormedRing A
              inst✝ : NormedAlgebra Real A
              a : Real
              a' b' : A
              u v u' v' : Real → A
              hu : ∀ (x : Real), HasDerivAt u (u' x) x
              hv : ∀ (x : Real), HasDerivAt v (v' x) x
              ⊢ MeasureTheory.Measure Real
            -/
            /-
              🎉 no goals
            -/
                                         /-
                                           🎉 no goals
                                         -/
    (huv' : Integrable (u * v')) (hu'v : Integrable (u' * v)) (huv : Integrable (u * v)) :
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    ∫ (x : ℝ), u x * v' x = - ∫ (x : ℝ), u' x * v x :=
  integral_bilinear_hasDerivAt_right_eq_neg_left_of_integrable (L := ContinuousLinearMap.mul ℝ A)
    hu hv huv' hu'v huv


/-- For finite intervals, see: `intervalIntegral.integral_deriv_mul_eq_sub`. -/
theorem integral_Ioi_deriv_mul_eq_sub
    (hu : ∀ x ∈ Ioi a, HasDerivAt u (u' x) x) (hv : ∀ x ∈ Ioi a, HasDerivAt v (v' x) x)
           /-
             A : Type u_1
             inst✝² : NormedRing A
             inst✝¹ : NormedAlgebra Real A
             a : Real
             a' b' : A
             u v u' v' : Real → A
             inst✝ : CompleteSpace A
             hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
             hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
             ⊢ MeasureTheory.Measure Real
           -/
    (huv : IntegrableOn (u' * v + u * v') (Ioi a))
           /-
             🎉 no goals
           -/
    (h_zero : Tendsto (u * v) (𝓝[>] a) (𝓝 a')) (h_infty : Tendsto (u * v) atTop (𝓝 b')) :
    ∫ (x : ℝ) in Ioi a, u' x * v x + u x * v' x = b' - a' := by
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
    huv : MeasureTheory.IntegrableOn (HAdd.hAdd (HMul.hMul u' v) (HMul.hMul u v')) …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Ioi a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atTop (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [← Ici_diff_left] at h_zero
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
    huv : MeasureTheory.IntegrableOn (HAdd.hAdd (HMul.hMul u' v) (HMul.hMul u v')) …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (SDiff.sdiff (Set.Ici a) …
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atTop (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  let f := Function.update (u * v) a a'
  have hderiv : ∀ x ∈ Ioi a, HasDerivAt f (u' x * v x + u x * v' x) x := by
    intro x (hx : a < x)
    apply ((hu x hx).mul (hv x hx)).congr_of_eventuallyEq
    filter_upwards [eventually_ne_nhds hx.ne.symm] with y hy
    exact Function.update_of_ne hy a' (u * v)
  have htendsto : Tendsto f atTop (𝓝 b') := by
    apply h_infty.congr'
    filter_upwards [eventually_ne_atTop a] with x hx
    exact (Function.update_of_ne hx a' (u * v)).symm
  simpa using integral_Ioi_of_hasDerivAt_of_tendsto
    (continuousWithinAt_update_same.mpr h_zero) hderiv huv htendsto


/-- **Integration by parts on (a, ∞).**
For finite intervals, see: `intervalIntegral.integral_mul_deriv_eq_deriv_mul`. -/
theorem integral_Ioi_mul_deriv_eq_deriv_mul
    (hu : ∀ x ∈ Ioi a, HasDerivAt u (u' x) x) (hv : ∀ x ∈ Ioi a, HasDerivAt v (v' x) x)
            /-
              A : Type u_1
              inst✝² : NormedRing A
              inst✝¹ : NormedAlgebra Real A
              a : Real
              a' b' : A
              u v u' v' : Real → A
              inst✝ : CompleteSpace A
              hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
              hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
              ⊢ MeasureTheory.Measure Real
            -/
            /-
              🎉 no goals
            -/
    (huv' : IntegrableOn (u * v') (Ioi a)) (hu'v : IntegrableOn (u' * v) (Ioi a))
                                                   /-
                                                     🎉 no goals
                                                   -/
    (h_zero : Tendsto (u * v) (𝓝[>] a) (𝓝 a')) (h_infty : Tendsto (u * v) atTop (𝓝 b')) :
    ∫ (x : ℝ) in Ioi a, u x * v' x = b' - a' - ∫ (x : ℝ) in Ioi a, u' x * v x := by
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
    huv' : MeasureTheory.IntegrableOn (HMul.hMul u v') (Set.Ioi a) MeasureTheory.M …
    hu'v : MeasureTheory.IntegrableOn (HMul.hMul u' v) (Set.Ioi a) MeasureTheory.M …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Ioi a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atTop (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [Pi.mul_def] at huv' hu'v
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
    huv' : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u i) (v' i)) (Set.Ioi a …
    hu'v : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u' i) (v i)) (Set.Ioi a …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Ioi a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atTop (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [eq_sub_iff_add_eq, ← integral_add huv' hu'v]
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt v (v' x) x
    huv' : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u i) (v' i)) (Set.Ioi a …
    hu'v : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u' i) (v i)) (Set.Ioi a …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Ioi a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atTop (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simpa only [add_comm] using integral_Ioi_deriv_mul_eq_sub hu hv (hu'v.add huv') h_zero h_infty
  /-
    🎉 no goals
  -/


/-- For finite intervals, see: `intervalIntegral.integral_deriv_mul_eq_sub`. -/
theorem integral_Iic_deriv_mul_eq_sub
    (hu : ∀ x ∈ Iio a, HasDerivAt u (u' x) x) (hv : ∀ x ∈ Iio a, HasDerivAt v (v' x) x)
           /-
             A : Type u_1
             inst✝² : NormedRing A
             inst✝¹ : NormedAlgebra Real A
             a : Real
             a' b' : A
             u v u' v' : Real → A
             inst✝ : CompleteSpace A
             hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
             hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
             ⊢ MeasureTheory.Measure Real
           -/
    (huv : IntegrableOn (u' * v + u * v') (Iic a))
           /-
             🎉 no goals
           -/
    (h_zero : Tendsto (u * v) (𝓝[<] a) (𝓝 a')) (h_infty : Tendsto (u * v) atBot (𝓝 b')) :
    ∫ (x : ℝ) in Iic a, u' x * v x + u x * v' x = a' - b' := by
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
    huv : MeasureTheory.IntegrableOn (HAdd.hAdd (HMul.hMul u' v) (HMul.hMul u v')) …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Iio a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atBot (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [← Iic_diff_right] at h_zero
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
    huv : MeasureTheory.IntegrableOn (HAdd.hAdd (HMul.hMul u' v) (HMul.hMul u v')) …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (SDiff.sdiff (Set.Iic a) …
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atBot (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  let f := Function.update (u * v) a a'
  have hderiv : ∀ x ∈ Iio a, HasDerivAt f (u' x * v x + u x * v' x) x := by
    intro x hx
    apply ((hu x hx).mul (hv x hx)).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hx] with x (hx : x < a)
    exact Function.update_of_ne (ne_of_lt hx) a' (u * v)
  have htendsto : Tendsto f atBot (𝓝 b') := by
    apply h_infty.congr'
    filter_upwards [Iio_mem_atBot a] with x (hx : x < a)
    exact (Function.update_of_ne (ne_of_lt hx) a' (u * v)).symm
  simpa using integral_Iic_of_hasDerivAt_of_tendsto
    (continuousWithinAt_update_same.mpr h_zero) hderiv huv htendsto


/-- **Integration by parts on (∞, a].**
For finite intervals, see: `intervalIntegral.integral_mul_deriv_eq_deriv_mul`. -/
theorem integral_Iic_mul_deriv_eq_deriv_mul
    (hu : ∀ x ∈ Iio a, HasDerivAt u (u' x) x) (hv : ∀ x ∈ Iio a, HasDerivAt v (v' x) x)
            /-
              A : Type u_1
              inst✝² : NormedRing A
              inst✝¹ : NormedAlgebra Real A
              a : Real
              a' b' : A
              u v u' v' : Real → A
              inst✝ : CompleteSpace A
              hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
              hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
              ⊢ MeasureTheory.Measure Real
            -/
            /-
              🎉 no goals
            -/
    (huv' : IntegrableOn (u * v') (Iic a)) (hu'v : IntegrableOn (u' * v) (Iic a))
                                                   /-
                                                     🎉 no goals
                                                   -/
    (h_zero : Tendsto (u * v) (𝓝[<] a) (𝓝 a')) (h_infty : Tendsto (u * v) atBot (𝓝 b')) :
    ∫ (x : ℝ) in Iic a, u x * v' x = a' - b' - ∫ (x : ℝ) in Iic a, u' x * v x := by
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
    huv' : MeasureTheory.IntegrableOn (HMul.hMul u v') (Set.Iic a) MeasureTheory.M …
    hu'v : MeasureTheory.IntegrableOn (HMul.hMul u' v) (Set.Iic a) MeasureTheory.M …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Iio a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atBot (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [Pi.mul_def] at huv' hu'v
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
    huv' : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u i) (v' i)) (Set.Iic a …
    hu'v : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u' i) (v i)) (Set.Iic a …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Iio a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atBot (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [eq_sub_iff_add_eq, ← integral_add huv' hu'v]
  /-
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : Real
    a' b' : A
    u v u' v' : Real → A
    inst✝ : CompleteSpace A
    hu : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt u (u' x) x
    hv : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt v (v' x) x
    huv' : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u i) (v' i)) (Set.Iic a …
    hu'v : MeasureTheory.IntegrableOn (fun i => HMul.hMul (u' i) (v i)) (Set.Iic a …
    h_zero : Filter.Tendsto (HMul.hMul u v) (nhdsWithin a (Set.Iio a)) (nhds a')
    h_infty : Filter.Tendsto (HMul.hMul u v) Filter.atBot (nhds b')
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simpa only [add_comm] using integral_Iic_deriv_mul_eq_sub hu hv (hu'v.add huv') h_zero h_infty
  /-
    🎉 no goals
  -/


