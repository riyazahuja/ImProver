/-- A bounded convergence theorem for a finite measure:
If bounded continuous non-negative functions are uniformly bounded by a constant and tend to a
limit, then their integrals against the finite measure tend to the integral of the limit.
This formulation assumes:
 * the functions tend to a limit along a countably generated filter;
 * the limit is in the almost everywhere sense;
 * boundedness holds almost everywhere;
 * integration is `MeasureTheory.lintegral`, i.e., the functions and their integrals are
   `ℝ≥0∞`-valued.
-/
theorem tendsto_lintegral_nn_filter_of_le_const {ι : Type*} {L : Filter ι} [L.IsCountablyGenerated]
    (μ : Measure Ω) [IsFiniteMeasure μ] {fs : ι → Ω →ᵇ ℝ≥0} {c : ℝ≥0}
    (fs_le_const : ∀ᶠ i in L, ∀ᵐ ω : Ω ∂μ, fs i ω ≤ c) {f : Ω → ℝ≥0}
    (fs_lim : ∀ᵐ ω : Ω ∂μ, Tendsto (fun i ↦ fs i ω) L (𝓝 (f ω))) :
    Tendsto (fun i ↦ ∫⁻ ω, fs i ω ∂μ) L (𝓝 (∫⁻ ω, f ω ∂μ)) := by
  refine tendsto_lintegral_filter_of_dominated_convergence (fun _ ↦ c)
    (Eventually.of_forall fun i ↦ (ENNReal.continuous_coe.comp (fs i).continuous).measurable) ?_
    (@lintegral_const_lt_top _ _ μ _ _ (@ENNReal.coe_ne_top c)).ne ?_
    /-
      case refine_1
      Ω : Type u_1
      inst✝⁴ : TopologicalSpace Ω
      inst✝³ : MeasurableSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      ι : Type u_2
      L : Filter ι
      inst✝¹ : L.IsCountablyGenerated
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      fs : ι → BoundedContinuousFunction Ω NNReal
      c : NNReal
      fs_le_const : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le (( …
      f : Ω → NNReal
      fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (↑((fs n) a))  …
    -/
  · simpa only [Function.comp_apply, ENNReal.coe_le_coe] using fs_le_const
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Ω : Type u_1
      inst✝⁴ : TopologicalSpace Ω
      inst✝³ : MeasurableSpace Ω
      inst✝² : OpensMeasurableSpace Ω
      ι : Type u_2
      L : Filter ι
      inst✝¹ : L.IsCountablyGenerated
      μ : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      fs : ι → BoundedContinuousFunction Ω NNReal
      c : NNReal
      fs_le_const : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le (( …
      f : Ω → NNReal
      fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => ↑((fs n) a)) L (nhds ↑( …
    -/
  · simpa only [Function.comp_apply, ENNReal.tendsto_coe] using fs_lim
    /-
      🎉 no goals
    -/


/-- If bounded continuous functions tend to the indicator of a measurable set and are
uniformly bounded, then their integrals against a finite measure tend to the measure of the set.
This formulation assumes:
 * the functions tend to a limit along a countably generated filter;
 * the limit is in the almost everywhere sense;
 * boundedness holds almost everywhere.
-/
theorem measure_of_cont_bdd_of_tendsto_filter_indicator {ι : Type*} {L : Filter ι}
    [L.IsCountablyGenerated] (μ : Measure Ω)
    [IsFiniteMeasure μ] {c : ℝ≥0} {E : Set Ω} (E_mble : MeasurableSet E) (fs : ι → Ω →ᵇ ℝ≥0)
    (fs_bdd : ∀ᶠ i in L, ∀ᵐ ω : Ω ∂μ, fs i ω ≤ c)
    (fs_lim : ∀ᵐ ω ∂μ, Tendsto (fun i ↦ fs i ω) L (𝓝 (indicator E (fun _ ↦ (1 : ℝ≥0)) ω))) :
    Tendsto (fun n ↦ lintegral μ fun ω ↦ fs n ω) L (𝓝 (μ E)) := by
  /-
    Ω : Type u_1
    inst✝⁴ : TopologicalSpace Ω
    inst✝³ : MeasurableSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : NNReal
    E : Set Ω
    E_mble : MeasurableSet E
    fs : ι → BoundedContinuousFunction Ω NNReal
    fs_bdd : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le ((fs i) …
    fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun ω => ↑((fs n) ω)) L ( …
  -/
  convert tendsto_lintegral_nn_filter_of_le_const μ fs_bdd fs_lim
  have aux : ∀ ω, indicator E (fun _ ↦ (1 : ℝ≥0∞)) ω = ↑(indicator E (fun _ ↦ (1 : ℝ≥0)) ω) :=
    fun ω ↦ by simp only [ENNReal.coe_indicator, ENNReal.coe_one]
  /-
    case h.e'_5.h.e'_3
    Ω : Type u_1
    inst✝⁴ : TopologicalSpace Ω
    inst✝³ : MeasurableSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : NNReal
    E : Set Ω
    E_mble : MeasurableSet E
    fs : ι → BoundedContinuousFunction Ω NNReal
    fs_bdd : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le ((fs i) …
    fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
    aux : ∀ (ω : Ω), Eq (E.indicator (fun x => 1) ω) ↑(E.indicator (fun x => 1) ω)
    ⊢ Eq (μ E) (MeasureTheory.lintegral μ fun ω => ↑(E.indicator (fun x => 1) ω))
  -/
  simp_rw [← aux, lintegral_indicator E_mble]
  /-
    case h.e'_5.h.e'_3
    Ω : Type u_1
    inst✝⁴ : TopologicalSpace Ω
    inst✝³ : MeasurableSpace Ω
    inst✝² : OpensMeasurableSpace Ω
    ι : Type u_2
    L : Filter ι
    inst✝¹ : L.IsCountablyGenerated
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : NNReal
    E : Set Ω
    E_mble : MeasurableSet E
    fs : ι → BoundedContinuousFunction Ω NNReal
    fs_bdd : Filter.Eventually (fun i => Filter.Eventually (fun ω => LE.le ((fs i) …
    fs_lim : Filter.Eventually (fun ω => Filter.Tendsto (fun i => (fs i) ω) L (nhd …
    aux : ∀ (ω : Ω), Eq (E.indicator (fun x => 1) ω) ↑(E.indicator (fun x => 1) ω)
    ⊢ Eq (μ E) (MeasureTheory.lintegral (μ.restrict E) fun x => 1)
  -/
  simp only [lintegral_one, Measure.restrict_apply, MeasurableSet.univ, univ_inter]
  /-
    🎉 no goals
  -/


/-- If a sequence of bounded continuous functions tends to the indicator of a measurable set and
the functions are uniformly bounded, then their integrals against a finite measure tend to the
measure of the set.

A similar result with more general assumptions is
`MeasureTheory.measure_of_cont_bdd_of_tendsto_filter_indicator`.
-/
theorem measure_of_cont_bdd_of_tendsto_indicator
    (μ : Measure Ω) [IsFiniteMeasure μ] {c : ℝ≥0} {E : Set Ω} (E_mble : MeasurableSet E)
    (fs : ℕ → Ω →ᵇ ℝ≥0) (fs_bdd : ∀ n ω, fs n ω ≤ c)
    (fs_lim : Tendsto (fun n ω ↦ fs n ω) atTop (𝓝 (indicator E fun _ ↦ (1 : ℝ≥0)))) :
    Tendsto (fun n ↦ lintegral μ fun ω ↦ fs n ω) atTop (𝓝 (μ E)) := by
  have fs_lim' :
    ∀ ω, Tendsto (fun n : ℕ ↦ (fs n ω : ℝ≥0)) atTop (𝓝 (indicator E (fun _ ↦ (1 : ℝ≥0)) ω)) := by
    rw [tendsto_pi_nhds] at fs_lim
    exact fun ω ↦ fs_lim ω
  apply measure_of_cont_bdd_of_tendsto_filter_indicator μ E_mble fs
    (Eventually.of_forall fun n ↦ Eventually.of_forall (fs_bdd n)) (Eventually.of_forall fs_lim')


/-- The integrals of thickened indicators of a closed set against a finite measure tend to the
measure of the closed set if the thickening radii tend to zero. -/
theorem tendsto_lintegral_thickenedIndicator_of_isClosed {Ω : Type*} [MeasurableSpace Ω]
    [PseudoEMetricSpace Ω] [OpensMeasurableSpace Ω] (μ : Measure Ω) [IsFiniteMeasure μ] {F : Set Ω}
    (F_closed : IsClosed F) {δs : ℕ → ℝ} (δs_pos : ∀ n, 0 < δs n)
    (δs_lim : Tendsto δs atTop (𝓝 0)) :
    Tendsto (fun n ↦ lintegral μ fun ω ↦ (thickenedIndicator (δs_pos n) F ω : ℝ≥0∞)) atTop
      (𝓝 (μ F)) := by
  apply measure_of_cont_bdd_of_tendsto_indicator μ F_closed.measurableSet
    (fun n ↦ thickenedIndicator (δs_pos n) F) fun n ω ↦ thickenedIndicator_le_one (δs_pos n) F ω
  /-
    Ω : Type u_2
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    F : Set Ω
    F_closed : IsClosed F
    δs : Nat → Real
    δs_pos : ∀ (n : Nat), LT.lt 0 (δs n)
    δs_lim : Filter.Tendsto δs Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun n ω => (thickenedIndicator ⋯ F) ω) Filter.atTop (nhds (F …
  -/
  have key := thickenedIndicator_tendsto_indicator_closure δs_pos δs_lim F
  /-
    Ω : Type u_2
    inst✝³ : MeasurableSpace Ω
    inst✝² : PseudoEMetricSpace Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    F : Set Ω
    F_closed : IsClosed F
    δs : Nat → Real
    δs_pos : ∀ (n : Nat), LT.lt 0 (δs n)
    δs_lim : Filter.Tendsto δs Filter.atTop (nhds 0)
    key : Filter.Tendsto (fun n => ⇑(thickenedIndicator ⋯ F)) Filter.atTop (nhds ( …
    ⊢ Filter.Tendsto (fun n ω => (thickenedIndicator ⋯ F) ω) Filter.atTop (nhds (F …
  -/
  rwa [F_closed.closure_eq] at key
  /-
    🎉 no goals
  -/


/-- A type class for topological spaces in which the indicator functions of closed sets can be
approximated pointwise from above by a sequence of bounded continuous functions. -/
class HasOuterApproxClosed (X : Type*) [TopologicalSpace X] : Prop where
  exAppr : ∀ (F : Set X), IsClosed F → ∃ (fseq : ℕ → (X →ᵇ ℝ≥0)),
    (∀ n x, fseq n x ≤ 1) ∧ (∀ n x, x ∈ F → 1 ≤ fseq n x) ∧
    Tendsto (fun n : ℕ ↦ (fun x ↦ fseq n x)) atTop (𝓝 (indicator F fun _ ↦ (1 : ℝ≥0)))


/-- A sequence of continuous functions `X → [0,1]` tending to the indicator of a closed set. -/
noncomputable def _root_.IsClosed.apprSeq : ℕ → (X →ᵇ ℝ≥0) :=
  Exists.choose (HasOuterApproxClosed.exAppr F hF)


lemma apprSeq_apply_le_one (n : ℕ) (x : X) :
    hF.apprSeq n x ≤ 1 :=
  (Exists.choose_spec (HasOuterApproxClosed.exAppr F hF)).1 n x


lemma apprSeq_apply_eq_one (n : ℕ) {x : X} (hxF : x ∈ F) :
    hF.apprSeq n x = 1 :=
  le_antisymm (apprSeq_apply_le_one _ _ _)
    ((Exists.choose_spec (HasOuterApproxClosed.exAppr F hF)).2.1 n x hxF)


lemma tendsto_apprSeq :
    Tendsto (fun n : ℕ ↦ (fun x ↦ hF.apprSeq n x)) atTop (𝓝 (indicator F fun _ ↦ (1 : ℝ≥0))) :=
  (Exists.choose_spec (HasOuterApproxClosed.exAppr F hF)).2.2


lemma indicator_le_apprSeq (n : ℕ) :
    indicator F (fun _ ↦ 1) ≤ hF.apprSeq n := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : HasOuterApproxClosed X
    F : Set X
    hF : IsClosed F
    n : Nat
    ⊢ LE.le (F.indicator fun x => 1) ⇑(hF.apprSeq n)
  -/
  intro x
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : HasOuterApproxClosed X
    F : Set X
    hF : IsClosed F
    n : Nat
    x : X
    ⊢ LE.le (F.indicator (fun x => 1) x) ((hF.apprSeq n) x)
  -/
  by_cases hxF : x ∈ F
    /-
      case pos
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      n : Nat
      x : X
      hxF : Membership.mem F x
      ⊢ LE.le (F.indicator (fun x => 1) x) ((hF.apprSeq n) x)
    -/
  · simp only [hxF, indicator_of_mem, apprSeq_apply_eq_one hF n, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      n : Nat
      x : X
      hxF : Not (Membership.mem F x)
      ⊢ LE.le (F.indicator (fun x => 1) x) ((hF.apprSeq n) x)
    -/
  · simp only [hxF, not_false_eq_true, indicator_of_not_mem, zero_le]
    /-
      🎉 no goals
    -/


/-- The measure of a closed set is at most the integral of any function in a decreasing
approximating sequence to the indicator of the set. -/
theorem measure_le_lintegral [MeasurableSpace X] [OpensMeasurableSpace X] (μ : Measure X) (n : ℕ) :
    μ F ≤ ∫⁻ x, (hF.apprSeq n x : ℝ≥0∞) ∂μ := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : HasOuterApproxClosed X
    F : Set X
    hF : IsClosed F
    inst✝¹ : MeasurableSpace X
    inst✝ : OpensMeasurableSpace X
    μ : MeasureTheory.Measure X
    n : Nat
    ⊢ LE.le (μ F) (MeasureTheory.lintegral μ fun x => ↑((hF.apprSeq n) x))
  -/
  convert_to ∫⁻ x, (F.indicator (fun _ ↦ (1 : ℝ≥0∞))) x ∂μ ≤ ∫⁻ x, hF.apprSeq n x ∂μ
    /-
      case h.e'_3
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      inst✝¹ : MeasurableSpace X
      inst✝ : OpensMeasurableSpace X
      μ : MeasureTheory.Measure X
      n : Nat
      ⊢ Eq (μ F) (MeasureTheory.lintegral μ fun x => F.indicator (fun x => 1) x)
    -/
  · rw [lintegral_indicator hF.measurableSet]
    /-
      case h.e'_3
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      inst✝¹ : MeasurableSpace X
      inst✝ : OpensMeasurableSpace X
      μ : MeasureTheory.Measure X
      n : Nat
      ⊢ Eq (μ F) (MeasureTheory.lintegral (μ.restrict F) fun a => 1)
    -/
    simp only [lintegral_one, MeasurableSet.univ, Measure.restrict_apply, univ_inter]
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      inst✝¹ : MeasurableSpace X
      inst✝ : OpensMeasurableSpace X
      μ : MeasureTheory.Measure X
      n : Nat
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => F.indicator (fun x => 1) x) (Measu …
    -/
  · apply lintegral_mono
    /-
      case hfg
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      inst✝¹ : MeasurableSpace X
      inst✝ : OpensMeasurableSpace X
      μ : MeasureTheory.Measure X
      n : Nat
      ⊢ LE.le (F.indicator fun x => 1) fun a => ↑((hF.apprSeq n) a)
    -/
    intro x
    /-
      case hfg
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : HasOuterApproxClosed X
      F : Set X
      hF : IsClosed F
      inst✝¹ : MeasurableSpace X
      inst✝ : OpensMeasurableSpace X
      μ : MeasureTheory.Measure X
      n : Nat
      x : X
      ⊢ LE.le (F.indicator (fun x => 1) x) ((fun a => ↑((hF.apprSeq n) a)) x)
    -/
    by_cases hxF : x ∈ F
      /-
        case pos
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : HasOuterApproxClosed X
        F : Set X
        hF : IsClosed F
        inst✝¹ : MeasurableSpace X
        inst✝ : OpensMeasurableSpace X
        μ : MeasureTheory.Measure X
        n : Nat
        x : X
        hxF : Membership.mem F x
        ⊢ LE.le (F.indicator (fun x => 1) x) ((fun a => ↑((hF.apprSeq n) a)) x)
      -/
    · simp only [hxF, indicator_of_mem, apprSeq_apply_eq_one hF n hxF, ENNReal.coe_one, le_refl]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : HasOuterApproxClosed X
        F : Set X
        hF : IsClosed F
        inst✝¹ : MeasurableSpace X
        inst✝ : OpensMeasurableSpace X
        μ : MeasureTheory.Measure X
        n : Nat
        x : X
        hxF : Not (Membership.mem F x)
        ⊢ LE.le (F.indicator (fun x => 1) x) ((fun a => ↑((hF.apprSeq n) a)) x)
      -/
    · simp only [hxF, not_false_eq_true, indicator_of_not_mem, zero_le]
      /-
        🎉 no goals
      -/


/-- The integrals along a decreasing approximating sequence to the indicator of a closed set
tend to the measure of the closed set. -/
lemma tendsto_lintegral_apprSeq [MeasurableSpace X] [OpensMeasurableSpace X]
    (μ : Measure X) [IsFiniteMeasure μ] :
    Tendsto (fun n ↦ ∫⁻ x, hF.apprSeq n x ∂μ) atTop (𝓝 ((μ : Measure X) F)) :=
  measure_of_cont_bdd_of_tendsto_indicator μ hF.measurableSet hF.apprSeq
    (apprSeq_apply_le_one hF) (tendsto_apprSeq hF)


noncomputable instance (X : Type*) [TopologicalSpace X]
    [TopologicalSpace.PseudoMetrizableSpace X] : HasOuterApproxClosed X := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    ⊢ HasOuterApproxClosed X
  -/
  letI : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetric X
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ HasOuterApproxClosed X
  -/
  refine ⟨fun F hF ↦ ?_⟩
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    F : Set X
    hF : IsClosed F
    ⊢ Exists fun fseq => And (∀ (n : Nat) (x : X), LE.le ((fseq n) x) 1) (And (∀ ( …
  -/
  use fun n ↦ thickenedIndicator (δ := (1 : ℝ) / (n + 1)) Nat.one_div_pos_of_nat F
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace.PseudoMetrizableSpace X
    this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    F : Set X
    hF : IsClosed F
    ⊢ And (∀ (n : Nat) (x : X), LE.le (((fun n => thickenedIndicator ⋯ F) n) x) 1) …
  -/
  refine ⟨?_, ⟨?_, ?_⟩⟩
    /-
      case h.refine_1
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace.PseudoMetrizableSpace X
      this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      F : Set X
      hF : IsClosed F
      ⊢ ∀ (n : Nat) (x : X), LE.le (((fun n => thickenedIndicator ⋯ F) n) x) 1
    -/
  · exact fun n x ↦ thickenedIndicator_le_one Nat.one_div_pos_of_nat F x
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace.PseudoMetrizableSpace X
      this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      F : Set X
      hF : IsClosed F
      ⊢ ∀ (n : Nat) (x : X), Membership.mem F x → LE.le 1 (((fun n => thickenedIndic …
    -/
  · exact fun n x hxF ↦ one_le_thickenedIndicator_apply X Nat.one_div_pos_of_nat hxF
    /-
      🎉 no goals
    -/
  · have key := thickenedIndicator_tendsto_indicator_closure
              (δseq := fun (n : ℕ) ↦ (1 : ℝ) / (n + 1))
              (fun _ ↦ Nat.one_div_pos_of_nat) tendsto_one_div_add_atTop_nhds_zero_nat F
    /-
      case h.refine_3
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace.PseudoMetrizableSpace X
      this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      F : Set X
      hF : IsClosed F
      key : Filter.Tendsto (fun n => ⇑(thickenedIndicator ⋯ F)) Filter.atTop (nhds ( …
      ⊢ Filter.Tendsto (fun n x => ((fun n => thickenedIndicator ⋯ F) n) x) Filter.a …
    -/
    rw [tendsto_pi_nhds] at *
    /-
      case h.refine_3
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace.PseudoMetrizableSpace X
      this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      F : Set X
      hF : IsClosed F
      key : ∀ (x : X), Filter.Tendsto (fun i => (thickenedIndicator ⋯ F) x) Filter.a …
      ⊢ ∀ (x : X), Filter.Tendsto (fun i => ((fun n => thickenedIndicator ⋯ F) i) x) …
    -/
    intro x
    /-
      case h.refine_3
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace.PseudoMetrizableSpace X
      this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      F : Set X
      hF : IsClosed F
      key : ∀ (x : X), Filter.Tendsto (fun i => (thickenedIndicator ⋯ F) x) Filter.a …
      x : X
      ⊢ Filter.Tendsto (fun i => ((fun n => thickenedIndicator ⋯ F) i) x) Filter.atT …
    -/
    nth_rw 2 [← IsClosed.closure_eq hF]
    /-
      case h.refine_3
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace.PseudoMetrizableSpace X
      this : PseudoMetricSpace X := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      F : Set X
      hF : IsClosed F
      key : ∀ (x : X), Filter.Tendsto (fun i => (thickenedIndicator ⋯ F) x) Filter.a …
      x : X
      ⊢ Filter.Tendsto (fun i => ((fun n => thickenedIndicator ⋯ F) i) x) Filter.atT …
    -/
    exact key x
    /-
      🎉 no goals
    -/


/-- Two finite measures give equal values to all closed sets if the integrals of all bounded
continuous functions with respect to the two measures agree. -/
theorem measure_isClosed_eq_of_forall_lintegral_eq_of_isFiniteMeasure {Ω : Type*}
    [MeasurableSpace Ω] [TopologicalSpace Ω] [HasOuterApproxClosed Ω]
    [OpensMeasurableSpace Ω] {μ ν : Measure Ω} [IsFiniteMeasure μ]
    (h : ∀ (f : Ω →ᵇ ℝ≥0), ∫⁻ x, f x ∂μ = ∫⁻ x, f x ∂ν) {F : Set Ω} (F_closed : IsClosed F) :
    μ F = ν F := by
  have ν_finite : IsFiniteMeasure ν := by
    constructor
    have whole := h 1
    simp only [BoundedContinuousFunction.coe_one, Pi.one_apply, ENNReal.coe_one, lintegral_const,
      one_mul] at whole
    simp [← whole]
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : HasOuterApproxClosed Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
    F : Set Ω
    F_closed : IsClosed F
    ν_finite : MeasureTheory.IsFiniteMeasure ν
    ⊢ Eq (μ F) (ν F)
  -/
  have obs_μ := HasOuterApproxClosed.tendsto_lintegral_apprSeq F_closed μ
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : HasOuterApproxClosed Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
    F : Set Ω
    F_closed : IsClosed F
    ν_finite : MeasureTheory.IsFiniteMeasure ν
    obs_μ : Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun x => ↑((F_close …
    ⊢ Eq (μ F) (ν F)
  -/
  have obs_ν := HasOuterApproxClosed.tendsto_lintegral_apprSeq F_closed ν
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : HasOuterApproxClosed Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
    F : Set Ω
    F_closed : IsClosed F
    ν_finite : MeasureTheory.IsFiniteMeasure ν
    obs_μ : Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun x => ↑((F_close …
    obs_ν : Filter.Tendsto (fun n => MeasureTheory.lintegral ν fun x => ↑((F_close …
    ⊢ Eq (μ F) (ν F)
  -/
  simp_rw [h] at obs_μ
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : HasOuterApproxClosed Ω
    inst✝¹ : OpensMeasurableSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
    F : Set Ω
    F_closed : IsClosed F
    ν_finite : MeasureTheory.IsFiniteMeasure ν
    obs_ν : Filter.Tendsto (fun n => MeasureTheory.lintegral ν fun x => ↑((F_close …
    obs_μ : Filter.Tendsto (fun n => MeasureTheory.lintegral ν fun x => ↑((F_close …
    ⊢ Eq (μ F) (ν F)
  -/
  exact tendsto_nhds_unique obs_μ obs_ν
  /-
    🎉 no goals
  -/


/-- Two finite Borel measures are equal if the integrals of all bounded continuous functions with
respect to both agree. -/
theorem ext_of_forall_lintegral_eq_of_IsFiniteMeasure {Ω : Type*}
    [MeasurableSpace Ω] [TopologicalSpace Ω] [HasOuterApproxClosed Ω]
    [BorelSpace Ω] {μ ν : Measure Ω} [IsFiniteMeasure μ]
    (h : ∀ (f : Ω →ᵇ ℝ≥0), ∫⁻ x, f x ∂μ = ∫⁻ x, f x ∂ν) :
    μ = ν := by
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : HasOuterApproxClosed Ω
    inst✝¹ : BorelSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
    ⊢ Eq μ ν
  -/
  have key := @measure_isClosed_eq_of_forall_lintegral_eq_of_isFiniteMeasure Ω _ _ _ _ μ ν _ h
  /-
    Ω : Type u_1
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : TopologicalSpace Ω
    inst✝² : HasOuterApproxClosed Ω
    inst✝¹ : BorelSpace Ω
    μ ν : MeasureTheory.Measure Ω
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
    key : ∀ {F : Set Ω}, IsClosed F → Eq (μ F) (ν F)
    ⊢ Eq μ ν
  -/
  apply ext_of_generate_finite _ ?_ isPiSystem_isClosed
    /-
      case hμν
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : TopologicalSpace Ω
      inst✝² : HasOuterApproxClosed Ω
      inst✝¹ : BorelSpace Ω
      μ ν : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
      key : ∀ {F : Set Ω}, IsClosed F → Eq (μ F) (ν F)
      ⊢ ∀ (s : Set Ω), Membership.mem (setOf fun s => IsClosed s) s → Eq (μ s) (ν s)
    -/
  · exact fun F F_closed ↦ key F_closed
    /-
      🎉 no goals
    -/
    /-
      case h_univ
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : TopologicalSpace Ω
      inst✝² : HasOuterApproxClosed Ω
      inst✝¹ : BorelSpace Ω
      μ ν : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
      key : ∀ {F : Set Ω}, IsClosed F → Eq (μ F) (ν F)
      ⊢ Eq (μ Set.univ) (ν Set.univ)
    -/
  · exact key isClosed_univ
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : TopologicalSpace Ω
      inst✝² : HasOuterApproxClosed Ω
      inst✝¹ : BorelSpace Ω
      μ ν : MeasureTheory.Measure Ω
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (f : BoundedContinuousFunction Ω NNReal), Eq (MeasureTheory.lintegral μ  …
      key : ∀ {F : Set Ω}, IsClosed F → Eq (μ F) (ν F)
      ⊢ Eq inst✝⁴ (MeasurableSpace.generateFrom (setOf fun s => IsClosed s))
    -/
  · rw [BorelSpace.measurable_eq (α := Ω), borel_eq_generateFrom_isClosed]
    /-
      🎉 no goals
    -/


