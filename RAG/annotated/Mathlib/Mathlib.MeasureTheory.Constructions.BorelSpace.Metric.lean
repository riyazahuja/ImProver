@[measurability]
theorem measurableSet_ball : MeasurableSet (Metric.ball x ε) :=
  Metric.isOpen_ball.measurableSet


@[measurability]
theorem measurableSet_closedBall : MeasurableSet (Metric.closedBall x ε) :=
  Metric.isClosed_ball.measurableSet


@[measurability]
theorem measurable_infDist {s : Set α} : Measurable fun x => infDist x s :=
  (continuous_infDist_pt s).measurable


@[measurability, fun_prop]
theorem Measurable.infDist {f : β → α} (hf : Measurable f) {s : Set α} :
    Measurable fun x => infDist (f x) s :=
  measurable_infDist.comp hf


@[measurability]
theorem measurable_infNndist {s : Set α} : Measurable fun x => infNndist x s :=
  (continuous_infNndist_pt s).measurable


@[measurability, fun_prop]
theorem Measurable.infNndist {f : β → α} (hf : Measurable f) {s : Set α} :
    Measurable fun x => infNndist (f x) s :=
  measurable_infNndist.comp hf


@[measurability]
theorem measurable_dist : Measurable fun p : α × α => dist p.1 p.2 :=
  continuous_dist.measurable


@[measurability, fun_prop]
theorem Measurable.dist {f g : β → α} (hf : Measurable f) (hg : Measurable g) :
    Measurable fun b => dist (f b) (g b) :=
  (@continuous_dist α _).measurable2 hf hg


@[measurability]
theorem measurable_nndist : Measurable fun p : α × α => nndist p.1 p.2 :=
  continuous_nndist.measurable


@[measurability, fun_prop]
theorem Measurable.nndist {f g : β → α} (hf : Measurable f) (hg : Measurable g) :
    Measurable fun b => nndist (f b) (g b) :=
  (@continuous_nndist α _).measurable2 hf hg


@[measurability]
theorem measurableSet_eball : MeasurableSet (EMetric.ball x ε) :=
  EMetric.isOpen_ball.measurableSet


@[measurability, fun_prop]
theorem measurable_edist_right : Measurable (edist x) :=
  (continuous_const.edist continuous_id).measurable


@[measurability, fun_prop]
theorem measurable_edist_left : Measurable fun y => edist y x :=
  (continuous_id.edist continuous_const).measurable


@[measurability]
theorem measurable_infEdist {s : Set α} : Measurable fun x => infEdist x s :=
  continuous_infEdist.measurable


@[measurability, fun_prop]
theorem Measurable.infEdist {f : β → α} (hf : Measurable f) {s : Set α} :
    Measurable fun x => infEdist (f x) s :=
  measurable_infEdist.comp hf


/-- If a set has a closed thickening with finite measure, then the measure of its `r`-closed
thickenings converges to the measure of its closure as `r` tends to `0`. -/
theorem tendsto_measure_cthickening {μ : Measure α} {s : Set α}
    (hs : ∃ R > 0, μ (cthickening R s) ≠ ∞) :
    Tendsto (fun r => μ (cthickening r s)) (𝓝 0) (𝓝 (μ (closure s))) := by
  have A : Tendsto (fun r => μ (cthickening r s)) (𝓝[Ioi 0] 0) (𝓝 (μ (closure s))) := by
    rw [closure_eq_iInter_cthickening]
    exact
      tendsto_measure_biInter_gt (fun r _ => isClosed_cthickening.nullMeasurableSet)
        (fun i j _ ij => cthickening_mono ij _) hs
  have B : Tendsto (fun r => μ (cthickening r s)) (𝓝[Iic 0] 0) (𝓝 (μ (closure s))) := by
    apply Tendsto.congr' _ tendsto_const_nhds
    filter_upwards [self_mem_nhdsWithin (α := ℝ)] with _ hr
    rw [cthickening_of_nonpos hr]
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.cthickening R s)) Top.top)
    A : Filter.Tendsto (fun r => μ (Metric.cthickening r s)) (nhdsWithin 0 (Set.Io …
    B : Filter.Tendsto (fun r => μ (Metric.cthickening r s)) (nhdsWithin 0 (Set.Ii …
    ⊢ Filter.Tendsto (fun r => μ (Metric.cthickening r s)) (nhds 0) (nhds (μ (clos …
  -/
  convert B.sup A
  /-
    case h.e'_4
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.cthickening R s)) Top.top)
    A : Filter.Tendsto (fun r => μ (Metric.cthickening r s)) (nhdsWithin 0 (Set.Io …
    B : Filter.Tendsto (fun r => μ (Metric.cthickening r s)) (nhdsWithin 0 (Set.Ii …
    ⊢ Eq (nhds 0) (Max.max (nhdsWithin 0 (Set.Iic 0)) (nhdsWithin 0 (Set.Ioi 0)))
  -/
  exact (nhdsLE_sup_nhdsGT 0).symm
  /-
    🎉 no goals
  -/


/-- If a closed set has a closed thickening with finite measure, then the measure of its closed
`r`-thickenings converge to its measure as `r` tends to `0`. -/
theorem tendsto_measure_cthickening_of_isClosed {μ : Measure α} {s : Set α}
    (hs : ∃ R > 0, μ (cthickening R s) ≠ ∞) (h's : IsClosed s) :
    Tendsto (fun r => μ (cthickening r s)) (𝓝 0) (𝓝 (μ s)) := by
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.cthickening R s)) Top.top)
    h's : IsClosed s
    ⊢ Filter.Tendsto (fun r => μ (Metric.cthickening r s)) (nhds 0) (nhds (μ s))
  -/
  convert tendsto_measure_cthickening hs
  /-
    case h.e'_5.h.e'_3.h.e'_6
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.cthickening R s)) Top.top)
    h's : IsClosed s
    ⊢ Eq s (closure s)
  -/
  exact h's.closure_eq.symm
  /-
    🎉 no goals
  -/


/-- If a set has a thickening with finite measure, then the measures of its `r`-thickenings
converge to the measure of its closure as `r > 0` tends to `0`. -/
theorem tendsto_measure_thickening {μ : Measure α} {s : Set α}
    (hs : ∃ R > 0, μ (thickening R s) ≠ ∞) :
    Tendsto (fun r => μ (thickening r s)) (𝓝[>] 0) (𝓝 (μ (closure s))) := by
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.thickening R s)) Top.top)
    ⊢ Filter.Tendsto (fun r => μ (Metric.thickening r s)) (nhdsWithin 0 (Set.Ioi 0 …
  -/
  rw [closure_eq_iInter_thickening]
  exact tendsto_measure_biInter_gt (fun r _ => isOpen_thickening.nullMeasurableSet)
      (fun i j _ ij => thickening_mono ij _) hs


/-- If a closed set has a thickening with finite measure, then the measure of its
`r`-thickenings converge to its measure as `r > 0` tends to `0`. -/
theorem tendsto_measure_thickening_of_isClosed {μ : Measure α} {s : Set α}
    (hs : ∃ R > 0, μ (thickening R s) ≠ ∞) (h's : IsClosed s) :
    Tendsto (fun r => μ (thickening r s)) (𝓝[>] 0) (𝓝 (μ s)) := by
  /-
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.thickening R s)) Top.top)
    h's : IsClosed s
    ⊢ Filter.Tendsto (fun r => μ (Metric.thickening r s)) (nhdsWithin 0 (Set.Ioi 0 …
  -/
  convert tendsto_measure_thickening hs
  /-
    case h.e'_5.h.e'_3.h.e'_6
    α : Type u_1
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Exists fun R => And (GT.gt R 0) (Ne (μ (Metric.thickening R s)) Top.top)
    h's : IsClosed s
    ⊢ Eq s (closure s)
  -/
  exact h's.closure_eq.symm
  /-
    🎉 no goals
  -/


@[measurability]
theorem measurable_edist : Measurable fun p : α × α => edist p.1 p.2 :=
  continuous_edist.measurable


@[measurability, fun_prop]
theorem Measurable.edist {f g : β → α} (hf : Measurable f) (hg : Measurable g) :
    Measurable fun b => edist (f b) (g b) :=
  (@continuous_edist α _).measurable2 hf hg


@[measurability, fun_prop]
theorem AEMeasurable.edist {f g : β → α} {μ : Measure β} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) : AEMeasurable (fun a => edist (f a) (g a)) μ :=
  (@continuous_edist α _).aemeasurable2 hf hg


/-- Given a compact set in a proper space, the measure of its `r`-closed thickenings converges to
its measure as `r` tends to `0`. -/
theorem tendsto_measure_cthickening_of_isCompact [MetricSpace α] [MeasurableSpace α]
    [OpensMeasurableSpace α] [ProperSpace α] {μ : Measure α} [IsFiniteMeasureOnCompacts μ]
    {s : Set α} (hs : IsCompact s) :
    Tendsto (fun r => μ (Metric.cthickening r s)) (𝓝 0) (𝓝 (μ s)) :=
  tendsto_measure_cthickening_of_isClosed
    ⟨1, zero_lt_one, hs.isBounded.cthickening.measure_lt_top.ne⟩ hs.isClosed


/-- If a measurable space is countably generated and separates points, it arises as
the borel sets of some second countable t4 topology (i.e. a separable metrizable one). -/
theorem exists_borelSpace_of_countablyGenerated_of_separatesPoints (α : Type*)
    [m : MeasurableSpace α] [CountablyGenerated α] [SeparatesPoints α] :
    ∃ _ : TopologicalSpace α, SecondCountableTopology α ∧ T4Space α ∧ BorelSpace α := by
  /-
    α : Type u_5
    m : MeasurableSpace α
    inst✝¹ : MeasurableSpace.CountablyGenerated α
    inst✝ : MeasurableSpace.SeparatesPoints α
    ⊢ Exists fun x => And (SecondCountableTopology α) (And (T4Space α) (BorelSpace …
  -/
  rcases measurableEquiv_nat_bool_of_countablyGenerated α with ⟨s, ⟨f⟩⟩
  /-
    case intro.intro
    α : Type u_5
    m : MeasurableSpace α
    inst✝¹ : MeasurableSpace.CountablyGenerated α
    inst✝ : MeasurableSpace.SeparatesPoints α
    s : Set (Nat → Bool)
    f : MeasurableEquiv α ↑s
    ⊢ Exists fun x => And (SecondCountableTopology α) (And (T4Space α) (BorelSpace …
  -/
  letI := induced f inferInstance
  /-
    case intro.intro
    α : Type u_5
    m : MeasurableSpace α
    inst✝¹ : MeasurableSpace.CountablyGenerated α
    inst✝ : MeasurableSpace.SeparatesPoints α
    s : Set (Nat → Bool)
    f : MeasurableEquiv α ↑s
    this : TopologicalSpace α := TopologicalSpace.induced (⇑f) inferInstance
    ⊢ Exists fun x => And (SecondCountableTopology α) (And (T4Space α) (BorelSpace …
  -/
  let F := f.toEquiv.toHomeomorphOfIsInducing <| .induced _
  exact ⟨inferInstance, F.secondCountableTopology, F.symm.t4Space,
    f.measurableEmbedding.borelSpace F.isInducing⟩


/-- If a measurable space on `α` is countably generated and separates points, there is some
second countable t4 topology on `α` (i.e. a separable metrizable one) for which every
open set is measurable. -/
theorem exists_opensMeasurableSpace_of_countablySeparated (α : Type*)
    [m : MeasurableSpace α] [CountablySeparated α] :
    ∃ _ : TopologicalSpace α, SecondCountableTopology α ∧ T4Space α ∧ OpensMeasurableSpace α := by
  /-
    α : Type u_5
    m : MeasurableSpace α
    inst✝ : MeasurableSpace.CountablySeparated α
    ⊢ Exists fun x => And (SecondCountableTopology α) (And (T4Space α) (OpensMeasu …
  -/
  rcases exists_countablyGenerated_le_of_countablySeparated α with ⟨m', _, _, m'le⟩
  /-
    case intro.intro.intro
    α : Type u_5
    m : MeasurableSpace α
    inst✝ : MeasurableSpace.CountablySeparated α
    m' : MeasurableSpace α
    left✝¹ : MeasurableSpace.CountablyGenerated α
    left✝ : MeasurableSpace.SeparatesPoints α
    m'le : LE.le m' m
    ⊢ Exists fun x => And (SecondCountableTopology α) (And (T4Space α) (OpensMeasu …
  -/
  rcases exists_borelSpace_of_countablyGenerated_of_separatesPoints (m := m') with ⟨τ, _, _, τm'⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_5
    m : MeasurableSpace α
    inst✝ : MeasurableSpace.CountablySeparated α
    m' : MeasurableSpace α
    left✝³ : MeasurableSpace.CountablyGenerated α
    left✝² : MeasurableSpace.SeparatesPoints α
    m'le : LE.le m' m
    τ : TopologicalSpace α
    left✝¹ : SecondCountableTopology α
    left✝ : T4Space α
    τm' : BorelSpace α
    ⊢ Exists fun x => And (SecondCountableTopology α) (And (T4Space α) (OpensMeasu …
  -/
  exact ⟨τ, ‹_›, ‹_›, @OpensMeasurableSpace.mk _ _ m (τm'.measurable_eq.symm.le.trans m'le)⟩
  /-
    🎉 no goals
  -/


@[fun_prop, measurability]
theorem measurable_norm : Measurable (norm : α → ℝ) :=
  continuous_norm.measurable


@[measurability, fun_prop]
theorem Measurable.norm {f : β → α} (hf : Measurable f) : Measurable fun a => norm (f a) :=
  measurable_norm.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.norm {f : β → α} {μ : Measure β} (hf : AEMeasurable f μ) :
    AEMeasurable (fun a => norm (f a)) μ :=
  measurable_norm.comp_aemeasurable hf


@[measurability]
theorem measurable_nnnorm : Measurable (nnnorm : α → ℝ≥0) :=
  continuous_nnnorm.measurable


@[measurability, fun_prop]
theorem Measurable.nnnorm {f : β → α} (hf : Measurable f) : Measurable fun a => ‖f a‖₊ :=
  measurable_nnnorm.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.nnnorm {f : β → α} {μ : Measure β} (hf : AEMeasurable f μ) :
    AEMeasurable (fun a => ‖f a‖₊) μ :=
  measurable_nnnorm.comp_aemeasurable hf


@[measurability]
theorem measurable_ennnorm : Measurable fun x : α => (‖x‖₊ : ℝ≥0∞) :=
  measurable_nnnorm.coe_nnreal_ennreal


@[measurability, fun_prop]
theorem Measurable.ennnorm {f : β → α} (hf : Measurable f) : Measurable fun a => (‖f a‖₊ : ℝ≥0∞) :=
  hf.nnnorm.coe_nnreal_ennreal


@[measurability, fun_prop]
theorem AEMeasurable.ennnorm {f : β → α} {μ : Measure β} (hf : AEMeasurable f μ) :
    AEMeasurable (fun a => (‖f a‖₊ : ℝ≥0∞)) μ :=
  measurable_ennnorm.comp_aemeasurable hf


