theorem le_measure_compl_liminf_of_limsup_measure_le {ι : Type*} {L : Filter ι} {μ : Measure Ω}
    {μs : ι → Measure Ω} [IsProbabilityMeasure μ] [∀ i, IsProbabilityMeasure (μs i)] {E : Set Ω}
    (E_mble : MeasurableSet E) (h : (L.limsup fun i ↦ μs i E) ≤ μ E) :
    μ Eᶜ ≤ L.liminf fun i ↦ μs i Eᶜ := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    E : Set Ω
    E_mble : MeasurableSet E
    h : LE.le (Filter.limsup (fun i => (μs i) E) L) (μ E)
    ⊢ LE.le (μ (HasCompl.compl E)) (Filter.liminf (fun i => (μs i) (HasCompl.compl …
  -/
  rcases L.eq_or_neBot with rfl | hne
    /-
      case inl
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      ι : Type u_2
      μ : MeasureTheory.Measure Ω
      μs : ι → MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      E : Set Ω
      E_mble : MeasurableSet E
      h : LE.le (Filter.limsup (fun i => (μs i) E) Bot.bot) (μ E)
      ⊢ LE.le (μ (HasCompl.compl E)) (Filter.liminf (fun i => (μs i) (HasCompl.compl …
    -/
  · simp only [liminf_bot, le_top]
    /-
      🎉 no goals
    -/
  have meas_Ec : μ Eᶜ = 1 - μ E := by
    simpa only [measure_univ] using measure_compl E_mble (measure_lt_top μ E).ne
  have meas_i_Ec : ∀ i, μs i Eᶜ = 1 - μs i E := by
    intro i
    simpa only [measure_univ] using measure_compl E_mble (measure_lt_top (μs i) E).ne
  /-
    case inr
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    E : Set Ω
    E_mble : MeasurableSet E
    h : LE.le (Filter.limsup (fun i => (μs i) E) L) (μ E)
    hne : L.NeBot
    meas_Ec : Eq (μ (HasCompl.compl E)) (HSub.hSub 1 (μ E))
    meas_i_Ec : ∀ (i : ι), Eq ((μs i) (HasCompl.compl E)) (HSub.hSub 1 ((μs i) E))
    ⊢ LE.le (μ (HasCompl.compl E)) (Filter.liminf (fun i => (μs i) (HasCompl.compl …
  -/
  simp_rw [meas_Ec, meas_i_Ec]
  rw [show (L.liminf fun i : ι ↦ 1 - μs i E) = L.liminf ((fun x ↦ 1 - x) ∘ fun i : ι ↦ μs i E)
      from rfl]
  have key := antitone_const_tsub.map_limsup_of_continuousAt (F := L)
    (fun i ↦ μs i E) (ENNReal.continuous_sub_left ENNReal.one_ne_top).continuousAt
  /-
    case inr
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    E : Set Ω
    E_mble : MeasurableSet E
    h : LE.le (Filter.limsup (fun i => (μs i) E) L) (μ E)
    hne : L.NeBot
    meas_Ec : Eq (μ (HasCompl.compl E)) (HSub.hSub 1 (μ E))
    meas_i_Ec : ∀ (i : ι), Eq ((μs i) (HasCompl.compl E)) (HSub.hSub 1 ((μs i) E))
    key : Eq (HSub.hSub 1 (Filter.limsup (fun i => (μs i) E) L)) (Filter.liminf (F …
    ⊢ LE.le (HSub.hSub 1 (μ E)) (Filter.liminf (Function.comp (fun x => HSub.hSub  …
  -/
  simpa [← key] using antitone_const_tsub h
  /-
    🎉 no goals
  -/


theorem le_measure_liminf_of_limsup_measure_compl_le {ι : Type*} {L : Filter ι} {μ : Measure Ω}
    {μs : ι → Measure Ω} [IsProbabilityMeasure μ] [∀ i, IsProbabilityMeasure (μs i)] {E : Set Ω}
    (E_mble : MeasurableSet E) (h : (L.limsup fun i ↦ μs i Eᶜ) ≤ μ Eᶜ) :
    μ E ≤ L.liminf fun i ↦ μs i E :=
  compl_compl E ▸ le_measure_compl_liminf_of_limsup_measure_le (MeasurableSet.compl E_mble) h


theorem limsup_measure_compl_le_of_le_liminf_measure {ι : Type*} {L : Filter ι} {μ : Measure Ω}
    {μs : ι → Measure Ω} [IsProbabilityMeasure μ] [∀ i, IsProbabilityMeasure (μs i)] {E : Set Ω}
    (E_mble : MeasurableSet E) (h : μ E ≤ L.liminf fun i ↦ μs i E) :
    (L.limsup fun i ↦ μs i Eᶜ) ≤ μ Eᶜ := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    E : Set Ω
    E_mble : MeasurableSet E
    h : LE.le (μ E) (Filter.liminf (fun i => (μs i) E) L)
    ⊢ LE.le (Filter.limsup (fun i => (μs i) (HasCompl.compl E)) L) (μ (HasCompl.co …
  -/
  rcases L.eq_or_neBot with rfl | hne
    /-
      case inl
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      ι : Type u_2
      μ : MeasureTheory.Measure Ω
      μs : ι → MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      E : Set Ω
      E_mble : MeasurableSet E
      h : LE.le (μ E) (Filter.liminf (fun i => (μs i) E) Bot.bot)
      ⊢ LE.le (Filter.limsup (fun i => (μs i) (HasCompl.compl E)) Bot.bot) (μ (HasCo …
    -/
  · simp only [limsup_bot, bot_le]
    /-
      🎉 no goals
    -/
  have meas_Ec : μ Eᶜ = 1 - μ E := by
    simpa only [measure_univ] using measure_compl E_mble (measure_lt_top μ E).ne
  have meas_i_Ec : ∀ i, μs i Eᶜ = 1 - μs i E := by
    intro i
    simpa only [measure_univ] using measure_compl E_mble (measure_lt_top (μs i) E).ne
  /-
    case inr
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    E : Set Ω
    E_mble : MeasurableSet E
    h : LE.le (μ E) (Filter.liminf (fun i => (μs i) E) L)
    hne : L.NeBot
    meas_Ec : Eq (μ (HasCompl.compl E)) (HSub.hSub 1 (μ E))
    meas_i_Ec : ∀ (i : ι), Eq ((μs i) (HasCompl.compl E)) (HSub.hSub 1 ((μs i) E))
    ⊢ LE.le (Filter.limsup (fun i => (μs i) (HasCompl.compl E)) L) (μ (HasCompl.co …
  -/
  simp_rw [meas_Ec, meas_i_Ec]
  rw [show (L.limsup fun i : ι ↦ 1 - μs i E) = L.limsup ((fun x ↦ 1 - x) ∘ fun i : ι ↦ μs i E)
      from rfl]
  have key := antitone_const_tsub.map_liminf_of_continuousAt (F := L)
    (fun i ↦ μs i E) (ENNReal.continuous_sub_left ENNReal.one_ne_top).continuousAt
  /-
    case inr
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    E : Set Ω
    E_mble : MeasurableSet E
    h : LE.le (μ E) (Filter.liminf (fun i => (μs i) E) L)
    hne : L.NeBot
    meas_Ec : Eq (μ (HasCompl.compl E)) (HSub.hSub 1 (μ E))
    meas_i_Ec : ∀ (i : ι), Eq ((μs i) (HasCompl.compl E)) (HSub.hSub 1 ((μs i) E))
    key : Eq (HSub.hSub 1 (Filter.liminf (fun i => (μs i) E) L)) (Filter.limsup (F …
    ⊢ LE.le (Filter.limsup (Function.comp (fun x => HSub.hSub 1 x) fun i => (μs i) …
  -/
  simpa [← key] using antitone_const_tsub h
  /-
    🎉 no goals
  -/


theorem limsup_measure_le_of_le_liminf_measure_compl {ι : Type*} {L : Filter ι} {μ : Measure Ω}
    {μs : ι → Measure Ω} [IsProbabilityMeasure μ] [∀ i, IsProbabilityMeasure (μs i)] {E : Set Ω}
    (E_mble : MeasurableSet E) (h : μ Eᶜ ≤ L.liminf fun i ↦ μs i Eᶜ) :
    (L.limsup fun i ↦ μs i E) ≤ μ E :=
  compl_compl E ▸ limsup_measure_compl_le_of_le_liminf_measure (MeasurableSet.compl E_mble) h


/-- One pair of implications of the portmanteau theorem:
For a sequence of Borel probability measures, the following two are equivalent:

(C) The limsup of the measures of any closed set is at most the measure of the closed set
under a candidate limit measure.

(O) The liminf of the measures of any open set is at least the measure of the open set
under a candidate limit measure.
-/
theorem limsup_measure_closed_le_iff_liminf_measure_open_ge {ι : Type*} {L : Filter ι}
    {μ : Measure Ω} {μs : ι → Measure Ω} [IsProbabilityMeasure μ]
    [∀ i, IsProbabilityMeasure (μs i)] :
    (∀ F, IsClosed F → (L.limsup fun i ↦ μs i F) ≤ μ F) ↔
      ∀ G, IsOpen G → μ G ≤ L.liminf fun i ↦ μs i G := by
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    ⊢ Iff (∀ (F : Set Ω), IsClosed F → LE.le (Filter.limsup (fun i => (μs i) F) L) …
  -/
  constructor
    /-
      case mp
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : TopologicalSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure Ω
      μs : ι → MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      ⊢ (∀ (F : Set Ω), IsClosed F → LE.le (Filter.limsup (fun i => (μs i) F) L) (μ  …
    -/
  · intro h G G_open
    exact le_measure_liminf_of_limsup_measure_compl_le
      G_open.measurableSet (h Gᶜ (isClosed_compl_iff.mpr G_open))
    /-
      case mpr
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : TopologicalSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure Ω
      μs : ι → MeasureTheory.Measure Ω
      inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
      inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
      ⊢ (∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i) G) L) …
    -/
  · intro h F F_closed
    exact limsup_measure_le_of_le_liminf_measure_compl
      F_closed.measurableSet (h Fᶜ (isOpen_compl_iff.mpr F_closed))


theorem tendsto_measure_of_le_liminf_measure_of_limsup_measure_le {ι : Type*} {L : Filter ι}
    {μ : Measure Ω} {μs : ι → Measure Ω} {E₀ E E₁ : Set Ω} (E₀_subset : E₀ ⊆ E) (subset_E₁ : E ⊆ E₁)
    (nulldiff : μ (E₁ \ E₀) = 0) (h_E₀ : μ E₀ ≤ L.liminf fun i ↦ μs i E₀)
    (h_E₁ : (L.limsup fun i ↦ μs i E₁) ≤ μ E₁) : L.Tendsto (fun i ↦ μs i E) (𝓝 (μ E)) := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    μ : MeasureTheory.Measure Ω
    μs : ι → MeasureTheory.Measure Ω
    E₀ E E₁ : Set Ω
    E₀_subset : HasSubset.Subset E₀ E
    subset_E₁ : HasSubset.Subset E E₁
    nulldiff : Eq (μ (SDiff.sdiff E₁ E₀)) 0
    h_E₀ : LE.le (μ E₀) (Filter.liminf (fun i => (μs i) E₀) L)
    h_E₁ : LE.le (Filter.limsup (fun i => (μs i) E₁) L) (μ E₁)
    ⊢ Filter.Tendsto (fun i => (μs i) E) L (nhds (μ E))
  -/
  apply tendsto_of_le_liminf_of_limsup_le
  · have E₀_ae_eq_E : E₀ =ᵐ[μ] E :=
      EventuallyLE.antisymm E₀_subset.eventuallyLE
        (subset_E₁.eventuallyLE.trans (ae_le_set.mpr nulldiff))
    calc
      μ E = μ E₀ := measure_congr E₀_ae_eq_E.symm
      _ ≤ L.liminf fun i ↦ μs i E₀ := h_E₀
      _ ≤ L.liminf fun i ↦ μs i E :=
        liminf_le_liminf (.of_forall fun _ ↦ measure_mono E₀_subset)
  · have E_ae_eq_E₁ : E =ᵐ[μ] E₁ :=
      EventuallyLE.antisymm subset_E₁.eventuallyLE
        ((ae_le_set.mpr nulldiff).trans E₀_subset.eventuallyLE)
    calc
      (L.limsup fun i ↦ μs i E) ≤ L.limsup fun i ↦ μs i E₁ :=
        limsup_le_limsup (.of_forall fun _ ↦ measure_mono subset_E₁)
      _ ≤ μ E₁ := h_E₁
      _ = μ E := measure_congr E_ae_eq_E₁.symm
    /-
      case h
      Ω : Type u_1
      inst✝ : MeasurableSpace Ω
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure Ω
      μs : ι → MeasureTheory.Measure Ω
      E₀ E E₁ : Set Ω
      E₀_subset : HasSubset.Subset E₀ E
      subset_E₁ : HasSubset.Subset E E₁
      nulldiff : Eq (μ (SDiff.sdiff E₁ E₀)) 0
      h_E₀ : LE.le (μ E₀) (Filter.liminf (fun i => (μs i) E₀) L)
      h_E₁ : LE.le (Filter.limsup (fun i => (μs i) E₁) L) (μ E₁)
      ⊢ autoParam (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) L fun i => (μs i …
    -/
  · infer_param
    /-
      🎉 no goals
    -/
    /-
      case h'
      Ω : Type u_1
      inst✝ : MeasurableSpace Ω
      ι : Type u_2
      L : Filter ι
      μ : MeasureTheory.Measure Ω
      μs : ι → MeasureTheory.Measure Ω
      E₀ E E₁ : Set Ω
      E₀_subset : HasSubset.Subset E₀ E
      subset_E₁ : HasSubset.Subset E E₁
      nulldiff : Eq (μ (SDiff.sdiff E₁ E₀)) 0
      h_E₀ : LE.le (μ E₀) (Filter.liminf (fun i => (μs i) E₀) L)
      h_E₁ : LE.le (Filter.limsup (fun i => (μs i) E₁) L) (μ E₁)
      ⊢ autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) L fun i => (μs i …
    -/
  · infer_param
    /-
      🎉 no goals
    -/


/-- One implication of the portmanteau theorem:
For a sequence of Borel probability measures, if the liminf of the measures of any open set is at
least the measure of the open set under a candidate limit measure, then for any set whose
boundary carries no probability mass under the candidate limit measure, then its measures under the
sequence converge to its measure under the candidate limit measure.
-/
theorem tendsto_measure_of_null_frontier {ι : Type*} {L : Filter ι} {μ : Measure Ω}
    {μs : ι → Measure Ω} [IsProbabilityMeasure μ] [∀ i, IsProbabilityMeasure (μs i)]
    (h_opens : ∀ G, IsOpen G → μ G ≤ L.liminf fun i ↦ μs i G) {E : Set Ω}
    (E_nullbdry : μ (frontier E) = 0) : L.Tendsto (fun i ↦ μs i E) (𝓝 (μ E)) :=
  haveI h_closeds : ∀ F, IsClosed F → (L.limsup fun i ↦ μs i F) ≤ μ F :=
    limsup_measure_closed_le_iff_liminf_measure_open_ge.mpr h_opens
  tendsto_measure_of_le_liminf_measure_of_limsup_measure_le interior_subset subset_closure
    E_nullbdry (h_opens _ isOpen_interior) (h_closeds _ isClosed_closure)


/-- One implication of the portmanteau theorem:
Weak convergence of finite measures implies that the limsup of the measures of any closed set is
at most the measure of the closed set under the limit measure.
-/
theorem FiniteMeasure.limsup_measure_closed_le_of_tendsto {Ω ι : Type*} {L : Filter ι}
    [MeasurableSpace Ω] [TopologicalSpace Ω] [HasOuterApproxClosed Ω]
    [OpensMeasurableSpace Ω] {μ : FiniteMeasure Ω}
    {μs : ι → FiniteMeasure Ω} (μs_lim : Tendsto μs L (𝓝 μ)) {F : Set Ω} (F_closed : IsClosed F) :
    (L.limsup fun i ↦ (μs i : Measure Ω) F) ≤ (μ : Measure Ω) F := by
  /-
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (↑μ F)
  -/
  rcases L.eq_or_neBot with rfl | hne
    /-
      case inl
      Ω : Type u_1
      ι : Type u_2
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : HasOuterApproxClosed Ω
      inst✝ : OpensMeasurableSpace Ω
      μ : MeasureTheory.FiniteMeasure Ω
      μs : ι → MeasureTheory.FiniteMeasure Ω
      F : Set Ω
      F_closed : IsClosed F
      μs_lim : Filter.Tendsto μs Bot.bot (nhds μ)
      ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) Bot.bot) (↑μ F)
    -/
  · simp only [limsup_bot, bot_le]
    /-
      🎉 no goals
    -/
  /-
    case inr
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (↑μ F)
  -/
  apply ENNReal.le_of_forall_pos_le_add
  /-
    case inr.h
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ⊢ ∀ (ε : NNReal), LT.lt 0 ε → LT.lt (↑μ F) Top.top → LE.le (Filter.limsup (fun …
  -/
  intro ε ε_pos _
  /-
    case inr.h
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (HAdd.hAdd (↑μ F) ↑ε)
  -/
  have ε_pos' := (ENNReal.half_pos (ENNReal.coe_ne_zero.mpr ε_pos.ne.symm)).ne.symm
  /-
    case inr.h
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (HAdd.hAdd (↑μ F) ↑ε)
  -/
  let fs := F_closed.apprSeq
  have key₁ : Tendsto (fun n ↦ ∫⁻  ω, (fs n ω : ℝ≥0∞) ∂μ) atTop (𝓝 ((μ : Measure Ω) F)) :=
    HasOuterApproxClosed.tendsto_lintegral_apprSeq F_closed (μ : Measure Ω)
  have room₁ : (μ : Measure Ω) F < (μ : Measure Ω) F + ε / 2 :=
    ENNReal.lt_add_right (measure_lt_top (μ : Measure Ω) F).ne ε_pos'
  /-
    case inr.h
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (HAdd.hAdd (↑μ F) ↑ε)
  -/
  obtain ⟨M, hM⟩ := eventually_atTop.mp <| key₁.eventually_lt_const room₁
  /-
    case inr.h.intro
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    M : Nat
    hM : ∀ (b : Nat), GE.ge b M → LT.lt (MeasureTheory.lintegral ↑μ fun ω => ↑((fs …
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (HAdd.hAdd (↑μ F) ↑ε)
  -/
  have key₂ := FiniteMeasure.tendsto_iff_forall_lintegral_tendsto.mp μs_lim (fs M)
  have room₂ :
    (lintegral (μ : Measure Ω) fun a ↦ fs M a) <
      (lintegral (μ : Measure Ω) fun a ↦ fs M a) + ε / 2 :=
    ENNReal.lt_add_right (ne_of_lt ((fs M).lintegral_lt_top_of_nnreal _)) ε_pos'
  /-
    case inr.h.intro
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    M : Nat
    hM : ∀ (b : Nat), GE.ge b M → LT.lt (MeasureTheory.lintegral ↑μ fun ω => ↑((fs …
    key₂ : Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(μs i) fun x => ↑((fs …
    room₂ : LT.lt (MeasureTheory.lintegral ↑μ fun a => ↑((fs M) a)) (HAdd.hAdd (Me …
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (HAdd.hAdd (↑μ F) ↑ε)
  -/
  have ev_near := key₂.eventually_le_const room₂
  have ev_near' := ev_near.mono
    (fun n ↦ le_trans (HasOuterApproxClosed.measure_le_lintegral F_closed (μs n) M))
  /-
    case inr.h.intro
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    M : Nat
    hM : ∀ (b : Nat), GE.ge b M → LT.lt (MeasureTheory.lintegral ↑μ fun ω => ↑((fs …
    key₂ : Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(μs i) fun x => ↑((fs …
    room₂ : LT.lt (MeasureTheory.lintegral ↑μ fun a => ↑((fs M) a)) (HAdd.hAdd (Me …
    ev_near : Filter.Eventually (fun a => LE.le (MeasureTheory.lintegral ↑(μs a) f …
    ev_near' : Filter.Eventually (fun x => LE.le (↑(μs x) F) (HAdd.hAdd (MeasureTh …
    ⊢ LE.le (Filter.limsup (fun i => ↑(μs i) F) L) (HAdd.hAdd (↑μ F) ↑ε)
  -/
  apply (Filter.limsup_le_limsup ev_near').trans
  /-
    case inr.h.intro
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    M : Nat
    hM : ∀ (b : Nat), GE.ge b M → LT.lt (MeasureTheory.lintegral ↑μ fun ω => ↑((fs …
    key₂ : Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(μs i) fun x => ↑((fs …
    room₂ : LT.lt (MeasureTheory.lintegral ↑μ fun a => ↑((fs M) a)) (HAdd.hAdd (Me …
    ev_near : Filter.Eventually (fun a => LE.le (MeasureTheory.lintegral ↑(μs a) f …
    ev_near' : Filter.Eventually (fun x => LE.le (↑(μs x) F) (HAdd.hAdd (MeasureTh …
    ⊢ LE.le (Filter.limsup (fun x => HAdd.hAdd (MeasureTheory.lintegral ↑μ fun a = …
  -/
  rw [limsup_const]
  /-
    case inr.h.intro
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    M : Nat
    hM : ∀ (b : Nat), GE.ge b M → LT.lt (MeasureTheory.lintegral ↑μ fun ω => ↑((fs …
    key₂ : Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(μs i) fun x => ↑((fs …
    room₂ : LT.lt (MeasureTheory.lintegral ↑μ fun a => ↑((fs M) a)) (HAdd.hAdd (Me …
    ev_near : Filter.Eventually (fun a => LE.le (MeasureTheory.lintegral ↑(μs a) f …
    ev_near' : Filter.Eventually (fun x => LE.le (↑(μs x) F) (HAdd.hAdd (MeasureTh …
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral ↑μ fun a => ↑((fs M) a)) (HDiv.hDi …
  -/
  apply le_trans (add_le_add (hM M rfl.le).le (le_refl (ε / 2 : ℝ≥0∞)))
  /-
    case inr.h.intro
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : HasOuterApproxClosed Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.FiniteMeasure Ω
    μs : ι → MeasureTheory.FiniteMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    F : Set Ω
    F_closed : IsClosed F
    hne : L.NeBot
    ε : NNReal
    ε_pos : LT.lt 0 ε
    a✝ : LT.lt (↑μ F) Top.top
    ε_pos' : Ne (HDiv.hDiv (↑ε) 2) 0
    fs : Nat → BoundedContinuousFunction Ω NNReal := F_closed.apprSeq
    key₁ : Filter.Tendsto (fun n => MeasureTheory.lintegral ↑μ fun ω => ↑((fs n) ω …
    room₁ : LT.lt (↑μ F) (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2))
    M : Nat
    hM : ∀ (b : Nat), GE.ge b M → LT.lt (MeasureTheory.lintegral ↑μ fun ω => ↑((fs …
    key₂ : Filter.Tendsto (fun i => MeasureTheory.lintegral ↑(μs i) fun x => ↑((fs …
    room₂ : LT.lt (MeasureTheory.lintegral ↑μ fun a => ↑((fs M) a)) (HAdd.hAdd (Me …
    ev_near : Filter.Eventually (fun a => LE.le (MeasureTheory.lintegral ↑(μs a) f …
    ev_near' : Filter.Eventually (fun x => LE.le (↑(μs x) F) (HAdd.hAdd (MeasureTh …
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (↑μ F) (HDiv.hDiv (↑ε) 2)) (HDiv.hDiv (↑ε) 2)) ( …
  -/
  simp only [add_assoc, ENNReal.add_halves, le_refl]
  /-
    🎉 no goals
  -/


/-- One implication of the portmanteau theorem:
Weak convergence of probability measures implies that the limsup of the measures of any closed
set is at most the measure of the closed set under the limit probability measure.
-/
theorem ProbabilityMeasure.limsup_measure_closed_le_of_tendsto {Ω ι : Type*} {L : Filter ι}
    [MeasurableSpace Ω] [TopologicalSpace Ω] [OpensMeasurableSpace Ω] [HasOuterApproxClosed Ω]
    {μ : ProbabilityMeasure Ω} {μs : ι → ProbabilityMeasure Ω} (μs_lim : Tendsto μs L (𝓝 μ))
    {F : Set Ω} (F_closed : IsClosed F) :
    (L.limsup fun i ↦ (μs i : Measure Ω) F) ≤ (μ : Measure Ω) F := by
  apply FiniteMeasure.limsup_measure_closed_le_of_tendsto
    ((tendsto_nhds_iff_toFiniteMeasure_tendsto_nhds L).mp μs_lim) F_closed


/-- One implication of the portmanteau theorem:
Weak convergence of probability measures implies that the liminf of the measures of any open set
is at least the measure of the open set under the limit probability measure.
-/
theorem ProbabilityMeasure.le_liminf_measure_open_of_tendsto {Ω ι : Type*} {L : Filter ι}
    [MeasurableSpace Ω] [PseudoEMetricSpace Ω] [OpensMeasurableSpace Ω] [HasOuterApproxClosed Ω]
    {μ : ProbabilityMeasure Ω} {μs : ι → ProbabilityMeasure Ω} (μs_lim : Tendsto μs L (𝓝 μ))
    {G : Set Ω} (G_open : IsOpen G) :
    (μ : Measure Ω) G ≤ L.liminf fun i ↦ (μs i : Measure Ω) G :=
  haveI h_closeds : ∀ F, IsClosed F → (L.limsup fun i ↦ (μs i : Measure Ω) F) ≤ (μ : Measure Ω) F :=
    fun _ F_closed ↦ limsup_measure_closed_le_of_tendsto μs_lim F_closed
  le_measure_liminf_of_limsup_measure_compl_le G_open.measurableSet
    (h_closeds _ (isClosed_compl_iff.mpr G_open))


theorem ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto' {Ω ι : Type*}
    {L : Filter ι} [MeasurableSpace Ω] [PseudoEMetricSpace Ω] [OpensMeasurableSpace Ω]
    [HasOuterApproxClosed Ω] {μ : ProbabilityMeasure Ω} {μs : ι → ProbabilityMeasure Ω}
    (μs_lim : Tendsto μs L (𝓝 μ)) {E : Set Ω} (E_nullbdry : (μ : Measure Ω) (frontier E) = 0) :
    Tendsto (fun i ↦ (μs i : Measure Ω) E) L (𝓝 ((μ : Measure Ω) E)) :=
  haveI h_opens : ∀ G, IsOpen G → (μ : Measure Ω) G ≤ L.liminf fun i ↦ (μs i : Measure Ω) G :=
    fun _ G_open ↦ le_liminf_measure_open_of_tendsto μs_lim G_open
  tendsto_measure_of_null_frontier h_opens E_nullbdry


/-- One implication of the portmanteau theorem:
Weak convergence of probability measures implies that if the boundary of a Borel set
carries no probability mass under the limit measure, then the limit of the measures of the set
equals the measure of the set under the limit probability measure.

A version with coercions to ordinary `ℝ≥0∞`-valued measures is
`MeasureTheory.ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto'`.
-/
theorem ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto {Ω ι : Type*} {L : Filter ι}
    [MeasurableSpace Ω] [PseudoEMetricSpace Ω] [OpensMeasurableSpace Ω] [HasOuterApproxClosed Ω]
    {μ : ProbabilityMeasure Ω} {μs : ι → ProbabilityMeasure Ω} (μs_lim : Tendsto μs L (𝓝 μ))
    {E : Set Ω} (E_nullbdry : μ (frontier E) = 0) : Tendsto (fun i ↦ μs i E) L (𝓝 (μ E)) := by
  /-
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : HasOuterApproxClosed Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : ι → MeasureTheory.ProbabilityMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    E : Set Ω
    E_nullbdry : Eq (μ (frontier E)) 0
    ⊢ Filter.Tendsto (fun i => (μs i) E) L (nhds (μ E))
  -/
  have key := tendsto_measure_of_null_frontier_of_tendsto' μs_lim (by simpa using E_nullbdry)
  /-
    Ω : Type u_1
    ι : Type u_2
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    inst✝ : HasOuterApproxClosed Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : ι → MeasureTheory.ProbabilityMeasure Ω
    μs_lim : Filter.Tendsto μs L (nhds μ)
    E : Set Ω
    E_nullbdry : Eq (μ (frontier E)) 0
    key : Filter.Tendsto (fun i => ↑(μs i) E) L (nhds (↑μ E))
    ⊢ Filter.Tendsto (fun i => (μs i) E) L (nhds (μ E))
  -/
  exact (ENNReal.tendsto_toNNReal (measure_ne_top (↑μ) E)).comp key
  /-
    🎉 no goals
  -/


theorem exists_null_frontier_thickening (μ : Measure Ω) [SFinite μ] (s : Set Ω) {a b : ℝ}
    (hab : a < b) : ∃ r ∈ Ioo a b, μ (frontier (Metric.thickening r s)) = 0 := by
  have mbles : ∀ r : ℝ, MeasurableSet (frontier (Metric.thickening r s)) :=
    fun r ↦ isClosed_frontier.measurableSet
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  have disjs := Metric.frontier_thickening_disjoint s
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  have key := Measure.countable_meas_pos_of_disjoint_iUnion (μ := μ) mbles disjs
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    key : (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickening i s)))).Countable
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  have aux := measure_diff_null (s := Ioo a b) (Set.Countable.measure_zero key volume)
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    key : (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickening i s)))).Countable
    aux : Eq (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b) (setOf  …
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  have len_pos : 0 < ENNReal.ofReal (b - a) := by simp only [hab, ENNReal.ofReal_pos, sub_pos]
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    key : (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickening i s)))).Countable
    aux : Eq (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b) (setOf  …
    len_pos : LT.lt 0 (ENNReal.ofReal (HSub.hSub b a))
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  rw [← Real.volume_Ioo, ← aux] at len_pos
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    key : (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickening i s)))).Countable
    aux : Eq (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b) (setOf  …
    len_pos : LT.lt 0 (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b …
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  rcases nonempty_of_measure_ne_zero len_pos.ne.symm with ⟨r, ⟨r_in_Ioo, hr⟩⟩
  /-
    case intro.intro
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    key : (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickening i s)))).Countable
    aux : Eq (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b) (setOf  …
    len_pos : LT.lt 0 (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b …
    r : Real
    r_in_Ioo : Membership.mem (Set.Ioo a b) r
    hr : Not (Membership.mem (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickeni …
    ⊢ Exists fun r => And (Membership.mem (Set.Ioo a b) r) (Eq (μ (frontier (Metri …
  -/
  refine ⟨r, r_in_Ioo, ?_⟩
  /-
    case intro.intro
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    a b : Real
    hab : LT.lt a b
    mbles : ∀ (r : Real), MeasurableSet (frontier (Metric.thickening r s))
    disjs : Pairwise (Function.onFun Disjoint fun r => frontier (Metric.thickening …
    key : (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickening i s)))).Countable
    aux : Eq (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b) (setOf  …
    len_pos : LT.lt 0 (MeasureTheory.MeasureSpace.volume (SDiff.sdiff (Set.Ioo a b …
    r : Real
    r_in_Ioo : Membership.mem (Set.Ioo a b) r
    hr : Not (Membership.mem (setOf fun i => LT.lt 0 (μ (frontier (Metric.thickeni …
    ⊢ Eq (μ (frontier (Metric.thickening r s))) 0
  -/
  simpa only [mem_setOf_eq, not_lt, le_zero_iff] using hr
  /-
    🎉 no goals
  -/


theorem exists_null_frontiers_thickening (μ : Measure Ω) [SFinite μ] (s : Set Ω) :
    ∃ rs : ℕ → ℝ,
      Tendsto rs atTop (𝓝 0) ∧ ∀ n, 0 < rs n ∧ μ (frontier (Metric.thickening (rs n) s)) = 0 := by
  /-
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    ⊢ Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Nat), …
  -/
  rcases exists_seq_strictAnti_tendsto (0 : ℝ) with ⟨Rs, ⟨_, ⟨Rs_pos, Rs_lim⟩⟩⟩
  /-
    case intro.intro.intro
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    Rs : Nat → Real
    left✝ : StrictAnti Rs
    Rs_pos : ∀ (n : Nat), LT.lt 0 (Rs n)
    Rs_lim : Filter.Tendsto Rs Filter.atTop (nhds 0)
    ⊢ Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Nat), …
  -/
  have obs := fun n : ℕ => exists_null_frontier_thickening μ s (Rs_pos n)
  /-
    case intro.intro.intro
    Ω : Type u_1
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : MeasurableSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.SFinite μ
    s : Set Ω
    Rs : Nat → Real
    left✝ : StrictAnti Rs
    Rs_pos : ∀ (n : Nat), LT.lt 0 (Rs n)
    Rs_lim : Filter.Tendsto Rs Filter.atTop (nhds 0)
    obs : ∀ (n : Nat), Exists fun r => And (Membership.mem (Set.Ioo 0 (Rs n)) r) ( …
    ⊢ Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Nat), …
  -/
  refine ⟨fun n : ℕ => (obs n).choose, ⟨?_, ?_⟩⟩
  · exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds Rs_lim
      (fun n ↦ (obs n).choose_spec.1.1.le) fun n ↦ (obs n).choose_spec.1.2.le
    /-
      case intro.intro.intro.refine_2
      Ω : Type u_1
      inst✝³ : PseudoEMetricSpace Ω
      inst✝² : MeasurableSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.SFinite μ
      s : Set Ω
      Rs : Nat → Real
      left✝ : StrictAnti Rs
      Rs_pos : ∀ (n : Nat), LT.lt 0 (Rs n)
      Rs_lim : Filter.Tendsto Rs Filter.atTop (nhds 0)
      obs : ∀ (n : Nat), Exists fun r => And (Membership.mem (Set.Ioo 0 (Rs n)) r) ( …
      ⊢ ∀ (n : Nat), And (LT.lt 0 ((fun n => ⋯.choose) n)) (Eq (μ (frontier (Metric. …
    -/
  · exact fun n ↦ ⟨(obs n).choose_spec.1.1, (obs n).choose_spec.2⟩
    /-
      🎉 no goals
    -/


/-- One implication of the portmanteau theorem:
Assuming that for all Borel sets E whose boundary ∂E carries no probability mass under a
candidate limit probability measure μ we have convergence of the measures μsᵢ(E) to μ(E),
then for all closed sets F we have the limsup condition limsup μsᵢ(F) ≤ μ(F). -/
lemma limsup_measure_closed_le_of_forall_tendsto_measure
    {Ω ι : Type*} {L : Filter ι} [MeasurableSpace Ω] [PseudoEMetricSpace Ω] [OpensMeasurableSpace Ω]
    {μ : Measure Ω} [IsFiniteMeasure μ] {μs : ι → Measure Ω}
    (h : ∀ {E : Set Ω}, MeasurableSet E → μ (frontier E) = 0 →
            Tendsto (fun i ↦ μs i E) L (𝓝 (μ E)))
    (F : Set Ω) (F_closed : IsClosed F) :
    L.limsup (fun i ↦ μs i F) ≤ μ F := by
  /-
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  rcases L.eq_or_neBot with rfl | _
    /-
      case inl
      Ω : Type u_2
      ι : Type u_3
      inst✝³ : MeasurableSpace Ω
      inst✝² : PseudoEMetricSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      μs : ι → MeasureTheory.Measure Ω
      F : Set Ω
      F_closed : IsClosed F
      h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
      ⊢ LE.le (Filter.limsup (fun i => (μs i) F) Bot.bot) (μ F)
    -/
  · simp only [limsup_bot, bot_eq_zero', zero_le]
    /-
      🎉 no goals
    -/
  /-
    case inr
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  have ex := exists_null_frontiers_thickening μ F
  /-
    case inr
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  let rs := Classical.choose ex
  /-
    case inr
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  have rs_lim : Tendsto rs atTop (𝓝 0) := (Classical.choose_spec ex).1
  /-
    case inr
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_lim : Filter.Tendsto rs Filter.atTop (nhds 0)
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  have rs_pos : ∀ n, 0 < rs n := fun n ↦ ((Classical.choose_spec ex).2 n).1
  have rs_null : ∀ n, μ (frontier (Metric.thickening (rs n) F)) = 0 :=
    fun n ↦ ((Classical.choose_spec ex).2 n).2
  have Fthicks_open : ∀ n, IsOpen (Metric.thickening (rs n) F) :=
    fun n ↦ Metric.isOpen_thickening
  /-
    case inr
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_lim : Filter.Tendsto rs Filter.atTop (nhds 0)
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  have key := fun (n : ℕ) ↦ h (Fthicks_open n).measurableSet (rs_null n)
  /-
    case inr
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_lim : Filter.Tendsto rs Filter.atTop (nhds 0)
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (μ F)
  -/
  apply ENNReal.le_of_forall_pos_le_add
  /-
    case inr.h
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_lim : Filter.Tendsto rs Filter.atTop (nhds 0)
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ⊢ ∀ (ε : NNReal), LT.lt 0 ε → LT.lt (μ F) Top.top → LE.le (Filter.limsup (fun  …
  -/
  intros ε ε_pos μF_finite
  have keyB := tendsto_measure_cthickening_of_isClosed (μ := μ) (s := F)
                ⟨1, ⟨by simp only [gt_iff_lt, zero_lt_one], measure_ne_top _ _⟩⟩ F_closed
  have nhd : Iio (μ F + ε) ∈ 𝓝 (μ F) :=
    Iio_mem_nhds <| ENNReal.lt_add_right μF_finite.ne (ENNReal.coe_pos.mpr ε_pos).ne'
  /-
    case inr.h
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_lim : Filter.Tendsto rs Filter.atTop (nhds 0)
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ε : NNReal
    ε_pos : LT.lt 0 ε
    μF_finite : LT.lt (μ F) Top.top
    keyB : Filter.Tendsto (fun r => μ (Metric.cthickening r F)) (nhds 0) (nhds (μ  …
    nhd : Membership.mem (nhds (μ F)) (Set.Iio (HAdd.hAdd (μ F) ↑ε))
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (HAdd.hAdd (μ F) ↑ε)
  -/
  specialize rs_lim (keyB nhd)
  /-
    case inr.h
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ε : NNReal
    ε_pos : LT.lt 0 ε
    μF_finite : LT.lt (μ F) Top.top
    keyB : Filter.Tendsto (fun r => μ (Metric.cthickening r F)) (nhds 0) (nhds (μ  …
    nhd : Membership.mem (nhds (μ F)) (Set.Iio (HAdd.hAdd (μ F) ↑ε))
    rs_lim : Membership.mem (Filter.map rs Filter.atTop) (Set.preimage (fun r => μ …
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (HAdd.hAdd (μ F) ↑ε)
  -/
  simp only [mem_map, mem_atTop_sets, ge_iff_le, mem_preimage, mem_Iio] at rs_lim
  /-
    case inr.h
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ε : NNReal
    ε_pos : LT.lt 0 ε
    μF_finite : LT.lt (μ F) Top.top
    keyB : Filter.Tendsto (fun r => μ (Metric.cthickening r F)) (nhds 0) (nhds (μ  …
    nhd : Membership.mem (nhds (μ F)) (Set.Iio (HAdd.hAdd (μ F) ↑ε))
    rs_lim : Exists fun a => ∀ (b : Nat), LE.le a b → LT.lt (μ (Metric.cthickening …
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (HAdd.hAdd (μ F) ↑ε)
  -/
  obtain ⟨m, hm⟩ := rs_lim
  have aux : (fun i ↦ (μs i F)) ≤ᶠ[L] (fun i ↦ μs i (Metric.thickening (rs m) F)) :=
    .of_forall <| fun i ↦ measure_mono (Metric.self_subset_thickening (rs_pos m) F)
  /-
    case inr.h.intro
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ε : NNReal
    ε_pos : LT.lt 0 ε
    μF_finite : LT.lt (μ F) Top.top
    keyB : Filter.Tendsto (fun r => μ (Metric.cthickening r F)) (nhds 0) (nhds (μ  …
    nhd : Membership.mem (nhds (μ F)) (Set.Iio (HAdd.hAdd (μ F) ↑ε))
    m : Nat
    hm : ∀ (b : Nat), LE.le m b → LT.lt (μ (Metric.cthickening (rs b) F)) (HAdd.hA …
    aux : L.EventuallyLE (fun i => (μs i) F) fun i => (μs i) (Metric.thickening (r …
    ⊢ LE.le (Filter.limsup (fun i => (μs i) F) L) (HAdd.hAdd (μ F) ↑ε)
  -/
  refine (limsup_le_limsup aux).trans ?_
  /-
    case inr.h.intro
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ε : NNReal
    ε_pos : LT.lt 0 ε
    μF_finite : LT.lt (μ F) Top.top
    keyB : Filter.Tendsto (fun r => μ (Metric.cthickening r F)) (nhds 0) (nhds (μ  …
    nhd : Membership.mem (nhds (μ F)) (Set.Iio (HAdd.hAdd (μ F) ↑ε))
    m : Nat
    hm : ∀ (b : Nat), LE.le m b → LT.lt (μ (Metric.cthickening (rs b) F)) (HAdd.hA …
    aux : L.EventuallyLE (fun i => (μs i) F) fun i => (μs i) (Metric.thickening (r …
    ⊢ LE.le (Filter.limsup (fun i => (μs i) (Metric.thickening (rs m) F)) L) (HAdd …
  -/
  rw [Tendsto.limsup_eq (key m)]
  /-
    case inr.h.intro
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    F : Set Ω
    F_closed : IsClosed F
    h✝ : L.NeBot
    ex : Exists fun rs => And (Filter.Tendsto rs Filter.atTop (nhds 0)) (∀ (n : Na …
    rs : Nat → Real := Classical.choose ex
    rs_pos : ∀ (n : Nat), LT.lt 0 (rs n)
    rs_null : ∀ (n : Nat), Eq (μ (frontier (Metric.thickening (rs n) F))) 0
    Fthicks_open : ∀ (n : Nat), IsOpen (Metric.thickening (rs n) F)
    key : ∀ (n : Nat), Filter.Tendsto (fun i => (μs i) (Metric.thickening (rs n) F …
    ε : NNReal
    ε_pos : LT.lt 0 ε
    μF_finite : LT.lt (μ F) Top.top
    keyB : Filter.Tendsto (fun r => μ (Metric.cthickening r F)) (nhds 0) (nhds (μ  …
    nhd : Membership.mem (nhds (μ F)) (Set.Iio (HAdd.hAdd (μ F) ↑ε))
    m : Nat
    hm : ∀ (b : Nat), LE.le m b → LT.lt (μ (Metric.cthickening (rs b) F)) (HAdd.hA …
    aux : L.EventuallyLE (fun i => (μs i) F) fun i => (μs i) (Metric.thickening (r …
    ⊢ LE.le (μ (Metric.thickening (rs m) F)) (HAdd.hAdd (μ F) ↑ε)
  -/
  apply (measure_mono (Metric.thickening_subset_cthickening (rs m) F)).trans (hm m rfl.le).le
  /-
    🎉 no goals
  -/


/-- One implication of the portmanteau theorem:
Assuming that for all Borel sets E whose boundary ∂E carries no probability mass under a
candidate limit probability measure μ we have convergence of the measures μsᵢ(E) to μ(E),
then for all open sets G we have the limsup condition μ(G) ≤ liminf μsᵢ(G). -/
lemma le_liminf_measure_open_of_forall_tendsto_measure
    {Ω ι : Type*} {L : Filter ι}
    [MeasurableSpace Ω] [PseudoEMetricSpace Ω] [OpensMeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {μs : ι → Measure Ω} [∀ i, IsProbabilityMeasure (μs i)]
    (h : ∀ {E}, MeasurableSet E → μ (frontier E) = 0 → Tendsto (fun i ↦ μs i E) L (𝓝 (μ E)))
    (G : Set Ω) (G_open : IsOpen G) :
    μ G ≤ L.liminf (fun i ↦ μs i G) := by
  /-
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    G : Set Ω
    G_open : IsOpen G
    ⊢ LE.le (μ G) (Filter.liminf (fun i => (μs i) G) L)
  -/
  apply le_measure_liminf_of_limsup_measure_compl_le G_open.measurableSet
  /-
    Ω : Type u_2
    ι : Type u_3
    L : Filter ι
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : PseudoEMetricSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    μs : ι → MeasureTheory.Measure Ω
    inst✝ : ∀ (i : ι), MeasureTheory.IsProbabilityMeasure (μs i)
    h : ∀ {E : Set Ω}, MeasurableSet E → Eq (μ (frontier E)) 0 → Filter.Tendsto (f …
    G : Set Ω
    G_open : IsOpen G
    ⊢ LE.le (Filter.limsup (fun i => (μs i) (HasCompl.compl G)) L) (μ (HasCompl.co …
  -/
  exact limsup_measure_closed_le_of_forall_tendsto_measure h _ (isClosed_compl_iff.mpr G_open)
  /-
    🎉 no goals
  -/


lemma lintegral_le_liminf_lintegral_of_forall_isOpen_measure_le_liminf_measure
    {μ : Measure Ω} {μs : ℕ → Measure Ω} {f : Ω → ℝ} (f_cont : Continuous f) (f_nn : 0 ≤ f)
    (h_opens : ∀ G, IsOpen G → μ G ≤ atTop.liminf (fun i ↦ μs i G)) :
    ∫⁻ x, ENNReal.ofReal (f x) ∂μ ≤ atTop.liminf (fun i ↦ ∫⁻ x, ENNReal.ofReal (f x) ∂ (μs i)) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    μs : Nat → MeasureTheory.Measure Ω
    f : Ω → Real
    f_cont : Continuous f
    f_nn : LE.le 0 f
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter.limi …
  -/
  simp_rw [lintegral_eq_lintegral_meas_lt _ (Eventually.of_forall f_nn) f_cont.aemeasurable]
  calc  ∫⁻ (t : ℝ) in Set.Ioi 0, μ {a | t < f a}
      ≤ ∫⁻ (t : ℝ) in Set.Ioi 0, atTop.liminf (fun i ↦ (μs i) {a | t < f a}) := ?_ -- (i)
    _ ≤ atTop.liminf (fun i ↦ ∫⁻ (t : ℝ) in Set.Ioi 0, (μs i) {a | t < f a}) := ?_ -- (ii)
  · -- (i)
    exact (lintegral_mono (fun t ↦ h_opens _ (continuous_def.mp f_cont _ isOpen_Ioi))).trans
            (le_refl _)
  · -- (ii)
    exact lintegral_liminf_le (fun n ↦ Antitone.measurable (fun s t hst ↦
            measure_mono (fun ω hω ↦ lt_of_le_of_lt hst hω)))


lemma integral_le_liminf_integral_of_forall_isOpen_measure_le_liminf_measure
    {μ : Measure Ω} {μs : ℕ → Measure Ω} [∀ i, IsProbabilityMeasure (μs i)]
    {f : Ω →ᵇ ℝ} (f_nn : 0 ≤ f)
    (h_opens : ∀ G, IsOpen G → μ G ≤ atTop.liminf (fun i ↦ μs i G)) :
    ∫ x, (f x) ∂μ ≤ atTop.liminf (fun i ↦ ∫ x, (f x) ∂ (μs i)) := by
  have same := lintegral_le_liminf_lintegral_of_forall_isOpen_measure_le_liminf_measure
                  f.continuous f_nn h_opens
  rw [@integral_eq_lintegral_of_nonneg_ae Ω _ μ f (Eventually.of_forall f_nn)
        f.continuous.measurable.aestronglyMeasurable]
  /-
    Ω : Type u_1
    inst✝³ : MeasurableSpace Ω
    inst✝² : TopologicalSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    μs : Nat → MeasureTheory.Measure Ω
    inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)).toReal (Filt …
  -/
  convert ENNReal.toReal_mono ?_ same
  · simp only [fun i ↦ @integral_eq_lintegral_of_nonneg_ae Ω _ (μs i) f (Eventually.of_forall f_nn)
                        f.continuous.measurable.aestronglyMeasurable]
    /-
      case h.e'_4
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      ⊢ Eq (Filter.liminf (fun i => (MeasureTheory.lintegral (μs i) fun a => ENNReal …
    -/
    let g := BoundedContinuousFunction.comp _ Real.lipschitzWith_toNNReal f
    have bound : ∀ i, ∫⁻ x, ENNReal.ofReal (f x) ∂(μs i) ≤ nndist 0 g := fun i ↦ by
      simpa only [coe_nnreal_ennreal_nndist, measure_univ, mul_one, ge_iff_le] using
            BoundedContinuousFunction.lintegral_le_edist_mul (μ := μs i) g
    /-
      case h.e'_4
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      g : BoundedContinuousFunction Ω NNReal := BoundedContinuousFunction.comp Real. …
      bound : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ENNReal.of …
      ⊢ Eq (Filter.liminf (fun i => (MeasureTheory.lintegral (μs i) fun a => ENNReal …
    -/
    apply ENNReal.liminf_toReal_eq ENNReal.coe_ne_top (Eventually.of_forall bound)
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      ⊢ Ne (Filter.liminf (fun i => MeasureTheory.lintegral (μs i) fun x => ENNReal. …
    -/
  · apply ne_of_lt
    /-
      case h
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      ⊢ LT.lt (Filter.liminf (fun i => MeasureTheory.lintegral (μs i) fun x => ENNRe …
    -/
    have obs := fun (i : ℕ) ↦ @BoundedContinuousFunction.lintegral_nnnorm_le Ω _ _ (μs i) ℝ _ f
    /-
      case h
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
      ⊢ LT.lt (Filter.liminf (fun i => MeasureTheory.lintegral (μs i) fun x => ENNRe …
    -/
    simp only [measure_univ, mul_one] at obs
    /-
      case h
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
      ⊢ LT.lt (Filter.liminf (fun i => MeasureTheory.lintegral (μs i) fun x => ENNRe …
    -/
    apply lt_of_le_of_lt _ (show (‖f‖₊ : ℝ≥0∞) < ∞ from ENNReal.coe_lt_top)
    /-
      Ω : Type u_1
      inst✝³ : MeasurableSpace Ω
      inst✝² : TopologicalSpace Ω
      inst✝¹ : OpensMeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      μs : Nat → MeasureTheory.Measure Ω
      inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
      f : BoundedContinuousFunction Ω Real
      f_nn : LE.le 0 f
      h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
      same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
      obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
      ⊢ LE.le (Filter.liminf (fun i => MeasureTheory.lintegral (μs i) fun x => ENNRe …
    -/
    apply liminf_le_of_le
      /-
        case hf
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        ⊢ autoParam (Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Filter.atTop fun …
      -/
    · refine ⟨0, .of_forall (by simp only [ge_iff_le, zero_le, forall_const])⟩
      /-
        🎉 no goals
      -/
      /-
        case h
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        ⊢ ∀ (b : ENNReal), Filter.Eventually (fun n => LE.le b (MeasureTheory.lintegra …
      -/
    · intro x hx
      /-
        case h
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x : ENNReal
        hx : Filter.Eventually (fun n => LE.le x (MeasureTheory.lintegral (μs n) fun x …
        ⊢ LE.le x ↑(NNNorm.nnnorm f)
      -/
      obtain ⟨i, hi⟩ := hx.exists
      /-
        case h.intro
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x : ENNReal
        hx : Filter.Eventually (fun n => LE.le x (MeasureTheory.lintegral (μs n) fun x …
        i : Nat
        hi : LE.le x (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        ⊢ LE.le x ↑(NNNorm.nnnorm f)
      -/
      apply le_trans hi
      /-
        case h.intro
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x : ENNReal
        hx : Filter.Eventually (fun n => LE.le x (MeasureTheory.lintegral (μs n) fun x …
        i : Nat
        hi : LE.le x (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        ⊢ LE.le (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x)) ↑(NNNor …
      -/
      convert obs i with x
      /-
        case h.e'_3.h.e'_4.h
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x✝ : ENNReal
        hx : Filter.Eventually (fun n => LE.le x✝ (MeasureTheory.lintegral (μs n) fun  …
        i : Nat
        hi : LE.le x✝ (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        x : Ω
        ⊢ Eq (ENNReal.ofReal (f x)) ↑(NNNorm.nnnorm (f x))
      -/
      have aux := ENNReal.ofReal_eq_coe_nnreal (f_nn x)
      /-
        case h.e'_3.h.e'_4.h
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x✝ : ENNReal
        hx : Filter.Eventually (fun n => LE.le x✝ (MeasureTheory.lintegral (μs n) fun  …
        i : Nat
        hi : LE.le x✝ (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        x : Ω
        aux : Eq (ENNReal.ofReal ((fun f => f.toFun) f x)) ↑⟨(fun f => f.toFun) f x, ⋯⟩
        ⊢ Eq (ENNReal.ofReal (f x)) ↑(NNNorm.nnnorm (f x))
      -/
      simp only [ContinuousMap.toFun_eq_coe, BoundedContinuousFunction.coe_toContinuousMap] at aux
      /-
        case h.e'_3.h.e'_4.h
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x✝ : ENNReal
        hx : Filter.Eventually (fun n => LE.le x✝ (MeasureTheory.lintegral (μs n) fun  …
        i : Nat
        hi : LE.le x✝ (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        x : Ω
        aux : Eq (ENNReal.ofReal (f x)) ↑⟨f x, ⋯⟩
        ⊢ Eq (ENNReal.ofReal (f x)) ↑(NNNorm.nnnorm (f x))
      -/
      rw [aux]
      /-
        case h.e'_3.h.e'_4.h
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x✝ : ENNReal
        hx : Filter.Eventually (fun n => LE.le x✝ (MeasureTheory.lintegral (μs n) fun  …
        i : Nat
        hi : LE.le x✝ (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        x : Ω
        aux : Eq (ENNReal.ofReal (f x)) ↑⟨f x, ⋯⟩
        ⊢ Eq ↑⟨f x, ⋯⟩ ↑(NNNorm.nnnorm (f x))
      -/
      congr
      /-
        case h.e'_3.h.e'_4.h.e_a.e_val
        Ω : Type u_1
        inst✝³ : MeasurableSpace Ω
        inst✝² : TopologicalSpace Ω
        inst✝¹ : OpensMeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        μs : Nat → MeasureTheory.Measure Ω
        inst✝ : ∀ (i : Nat), MeasureTheory.IsProbabilityMeasure (μs i)
        f : BoundedContinuousFunction Ω Real
        f_nn : LE.le 0 f
        h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
        same : LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (Filter …
        obs : ∀ (i : Nat), LE.le (MeasureTheory.lintegral (μs i) fun x => ↑(NNNorm.nnn …
        x✝ : ENNReal
        hx : Filter.Eventually (fun n => LE.le x✝ (MeasureTheory.lintegral (μs n) fun  …
        i : Nat
        hi : LE.le x✝ (MeasureTheory.lintegral (μs i) fun x => ENNReal.ofReal (f x))
        x : Ω
        aux : Eq (ENNReal.ofReal (f x)) ↑⟨f x, ⋯⟩
        ⊢ Eq (f x) (Norm.norm (f x))
      -/
      exact (Real.norm_of_nonneg (f_nn x)).symm
      /-
        🎉 no goals
      -/


/-- One implication of the portmanteau theorem:
If for all open sets G we have the liminf condition `μ(G) ≤ liminf μsₙ(G)`, then the measures
μsₙ converge weakly to the measure μ. -/
theorem tendsto_of_forall_isOpen_le_liminf {μ : ProbabilityMeasure Ω}
    {μs : ℕ → ProbabilityMeasure Ω}
    (h_opens : ∀ G, IsOpen G → μ G ≤ atTop.liminf (fun i ↦ μs i G)) :
    atTop.Tendsto (fun i ↦ μs i) (𝓝 μ) := by
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    ⊢ Filter.Tendsto (fun i => μs i) Filter.atTop (nhds μ)
  -/
  refine ProbabilityMeasure.tendsto_iff_forall_integral_tendsto.mpr ?_
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    ⊢ ∀ (f : BoundedContinuousFunction Ω Real), Filter.Tendsto (fun i => MeasureTh …
  -/
  apply tendsto_integral_of_forall_integral_le_liminf_integral
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    ⊢ ∀ (f : BoundedContinuousFunction Ω Real), LE.le 0 f → LE.le (MeasureTheory.i …
  -/
  intro f f_nn
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    ⊢ LE.le (MeasureTheory.integral ↑μ fun x => f x) (Filter.liminf (fun i => Meas …
  -/
  apply integral_le_liminf_integral_of_forall_isOpen_measure_le_liminf_measure (f := f) f_nn
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    ⊢ ∀ (G : Set Ω), IsOpen G → LE.le (↑μ G) (Filter.liminf (fun i => ↑(μs i) G) F …
  -/
  intro G G_open
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    h_opens : ∀ (G : Set Ω), IsOpen G → LE.le (μ G) (Filter.liminf (fun i => (μs i …
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    G : Set Ω
    G_open : IsOpen G
    ⊢ LE.le (↑μ G) (Filter.liminf (fun i => ↑(μs i) G) Filter.atTop)
  -/
  specialize h_opens G G_open
  have aux : ENNReal.ofNNReal (liminf (fun i ↦ μs i G) atTop) =
          liminf (ENNReal.ofNNReal ∘ fun i ↦ μs i G) atTop := by
    refine Monotone.map_liminf_of_continuousAt (F := atTop) ENNReal.coe_mono (μs · G) ?_ ?_ ?_
    · exact ENNReal.continuous_coe.continuousAt
    · exact IsBoundedUnder.isCoboundedUnder_ge ⟨1, by simp⟩
    · exact ⟨0, by simp⟩
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    G : Set Ω
    G_open : IsOpen G
    h_opens : LE.le (μ G) (Filter.liminf (fun i => (μs i) G) Filter.atTop)
    aux : Eq (↑(Filter.liminf (fun i => (μs i) G) Filter.atTop)) (Filter.liminf (F …
    ⊢ LE.le (↑μ G) (Filter.liminf (fun i => ↑(μs i) G) Filter.atTop)
  -/
  have obs := ENNReal.coe_mono h_opens
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    G : Set Ω
    G_open : IsOpen G
    h_opens : LE.le (μ G) (Filter.liminf (fun i => (μs i) G) Filter.atTop)
    aux : Eq (↑(Filter.liminf (fun i => (μs i) G) Filter.atTop)) (Filter.liminf (F …
    obs : LE.le ↑(μ G) ↑(Filter.liminf (fun i => (μs i) G) Filter.atTop)
    ⊢ LE.le (↑μ G) (Filter.liminf (fun i => ↑(μs i) G) Filter.atTop)
  -/
  simp only [ne_eq, ProbabilityMeasure.ennreal_coeFn_eq_coeFn_toMeasure, aux] at obs
  /-
    case h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    G : Set Ω
    G_open : IsOpen G
    h_opens : LE.le (μ G) (Filter.liminf (fun i => (μs i) G) Filter.atTop)
    aux : Eq (↑(Filter.liminf (fun i => (μs i) G) Filter.atTop)) (Filter.liminf (F …
    obs : LE.le (↑μ G) (Filter.liminf (Function.comp ENNReal.ofNNReal fun i => (μs …
    ⊢ LE.le (↑μ G) (Filter.liminf (fun i => ↑(μs i) G) Filter.atTop)
  -/
  convert obs
  /-
    case h.e'_4.h.e'_4.h
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : TopologicalSpace Ω
    inst✝ : OpensMeasurableSpace Ω
    μ : MeasureTheory.ProbabilityMeasure Ω
    μs : Nat → MeasureTheory.ProbabilityMeasure Ω
    f : BoundedContinuousFunction Ω Real
    f_nn : LE.le 0 f
    G : Set Ω
    G_open : IsOpen G
    h_opens : LE.le (μ G) (Filter.liminf (fun i => (μs i) G) Filter.atTop)
    aux : Eq (↑(Filter.liminf (fun i => (μs i) G) Filter.atTop)) (Filter.liminf (F …
    obs : LE.le (↑μ G) (Filter.liminf (Function.comp ENNReal.ofNNReal fun i => (μs …
    x✝ : Nat
    ⊢ Eq (↑(μs x✝) G) (Function.comp ENNReal.ofNNReal (fun i => (μs i) G) x✝)
  -/
  simp only [Function.comp_apply, ne_eq, ProbabilityMeasure.ennreal_coeFn_eq_coeFn_toMeasure]
  /-
    🎉 no goals
  -/


