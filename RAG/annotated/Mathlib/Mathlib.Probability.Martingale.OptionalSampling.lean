theorem condexp_stopping_time_ae_eq_restrict_eq_const
    [(Filter.atTop : Filter ι).IsCountablyGenerated] (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) [SigmaFinite (μ.trim hτ.measurableSpace_le)] (hin : i ≤ n) :
    μ[f n|hτ.measurableSpace] =ᵐ[μ.restrict {x | τ x = i}] f i := by
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    inst✝⁷ : CompleteSpace E
    ι : Type u_3
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : OrderTopology ι
    inst✝³ : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝² : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    i n : ι
    inst✝¹ : Filter.atTop.IsCountablyGenerated
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hin : LE.le i n
    ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => Eq (τ x) i))).EventuallyEq (Me …
  -/
  refine Filter.EventuallyEq.trans ?_ (ae_restrict_of_ae (h.condexp_ae_eq hin))
  refine condexp_ae_eq_restrict_of_measurableSpace_eq_on hτ.measurableSpace_le (ℱ.le i)
    (hτ.measurableSet_eq' i) fun t => ?_
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    inst✝⁷ : CompleteSpace E
    ι : Type u_3
    inst✝⁶ : LinearOrder ι
    inst✝⁵ : TopologicalSpace ι
    inst✝⁴ : OrderTopology ι
    inst✝³ : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝² : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    i n : ι
    inst✝¹ : Filter.atTop.IsCountablyGenerated
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hin : LE.le i n
    t : Set Ω
    ⊢ Iff (MeasurableSet (Inter.inter (setOf fun x => Eq (τ x) i) t)) (MeasurableS …
  -/
  rw [Set.inter_comm _ t, IsStoppingTime.measurableSet_inter_eq_iff]
  /-
    🎉 no goals
  -/


theorem condexp_stopping_time_ae_eq_restrict_eq_const_of_le_const (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hτ_le : ∀ x, τ x ≤ n)
    [SigmaFinite (μ.trim (hτ.measurableSpace_le_of_le hτ_le))] (i : ι) :
    μ[f n|hτ.measurableSpace] =ᵐ[μ.restrict {x | τ x = i}] f i := by
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => Eq (τ x) i))).EventuallyEq (Me …
  -/
  by_cases hin : i ≤ n
    /-
      case pos
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      i : ι
      hin : LE.le i n
      ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => Eq (τ x) i))).EventuallyEq (Me …
    -/
  · refine Filter.EventuallyEq.trans ?_ (ae_restrict_of_ae (h.condexp_ae_eq hin))
    refine condexp_ae_eq_restrict_of_measurableSpace_eq_on (hτ.measurableSpace_le_of_le hτ_le)
      (ℱ.le i) (hτ.measurableSet_eq' i) fun t => ?_
    /-
      case pos
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      i : ι
      hin : LE.le i n
      t : Set Ω
      ⊢ Iff (MeasurableSet (Inter.inter (setOf fun x => Eq (τ x) i) t)) (MeasurableS …
    -/
    rw [Set.inter_comm _ t, IsStoppingTime.measurableSet_inter_eq_iff]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      i : ι
      hin : Not (LE.le i n)
      ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => Eq (τ x) i))).EventuallyEq (Me …
    -/
  · suffices {x : Ω | τ x = i} = ∅ by simp [this]; norm_cast
    /-
      case neg
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      i : ι
      hin : Not (LE.le i n)
      ⊢ Eq (setOf fun x => Eq (τ x) i) EmptyCollection.emptyCollection
    -/
    ext1 x
    /-
      case neg.h
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      i : ι
      hin : Not (LE.le i n)
      x : Ω
      ⊢ Iff (Membership.mem (setOf fun x => Eq (τ x) i) x) (Membership.mem EmptyColl …
    -/
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    /-
      case neg.h
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      i : ι
      hin : Not (LE.le i n)
      x : Ω
      ⊢ Not (Eq (τ x) i)
    -/
    rintro rfl
    /-
      case neg.h
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      x : Ω
      hin : Not (LE.le (τ x) n)
      ⊢ False
    -/
    exact hin (hτ_le x)
    /-
      🎉 no goals
    -/


theorem stoppedValue_ae_eq_restrict_eq (h : Martingale f ℱ μ) (hτ : IsStoppingTime ℱ τ)
    (hτ_le : ∀ x, τ x ≤ n) [SigmaFinite (μ.trim (hτ.measurableSpace_le_of_le hτ_le))] (i : ι) :
    stoppedValue f τ =ᵐ[μ.restrict {x | τ x = i}] μ[f n|hτ.measurableSpace] := by
  refine Filter.EventuallyEq.trans ?_
    (condexp_stopping_time_ae_eq_restrict_eq_const_of_le_const h hτ hτ_le i).symm
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => Eq (τ x) i))).EventuallyEq (Me …
  -/
  rw [Filter.EventuallyEq, ae_restrict_iff' (ℱ.le _ _ (hτ.measurableSet_eq i))]
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun ω => Eq (τ ω) i) x → E …
  -/
  refine Filter.Eventually.of_forall fun x hx => ?_
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    x : Ω
    hx : Membership.mem (setOf fun ω => Eq (τ ω) i) x
    ⊢ Eq (MeasureTheory.stoppedValue f τ x) (f i x)
  -/
  rw [Set.mem_setOf_eq] at hx
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    i : ι
    x : Ω
    hx : Eq (τ x) i
    ⊢ Eq (MeasureTheory.stoppedValue f τ x) (f i x)
  -/
  simp_rw [stoppedValue, hx]
  /-
    🎉 no goals
  -/


/-- The value of a martingale `f` at a stopping time `τ` bounded by `n` is the conditional
expectation of `f n` with respect to the σ-algebra generated by `τ`. -/
theorem stoppedValue_ae_eq_condexp_of_le_const_of_countable_range (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hτ_le : ∀ x, τ x ≤ n) (h_countable_range : (Set.range τ).Countable)
    [SigmaFinite (μ.trim (hτ.measurableSpace_le_of_le hτ_le))] :
    stoppedValue f τ =ᵐ[μ] μ[f n|hτ.measurableSpace] := by
  have : Set.univ = ⋃ i ∈ Set.range τ, {x | τ x = i} := by
    ext1 x
    simp only [Set.mem_univ, Set.mem_range, Set.iUnion_exists, Set.iUnion_iUnion_eq',
      Set.mem_iUnion, Set.mem_setOf_eq, exists_apply_eq_apply']
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    h_countable_range : (Set.range τ).Countable
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    this : Eq Set.univ (Set.iUnion fun i => Set.iUnion fun h => setOf fun x => Eq  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.stoppedValue f τ) (MeasureT …
  -/
  nth_rw 1 [← @Measure.restrict_univ Ω _ μ]
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    h_countable_range : (Set.range τ).Countable
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    this : Eq Set.univ (Set.iUnion fun i => Set.iUnion fun h => setOf fun x => Eq  …
    ⊢ (MeasureTheory.ae (μ.restrict Set.univ)).EventuallyEq (MeasureTheory.stopped …
  -/
  rw [this, ae_eq_restrict_biUnion_iff _ h_countable_range]
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    inst✝⁶ : CompleteSpace E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : TopologicalSpace ι
    inst✝³ : OrderTopology ι
    inst✝² : FirstCountableTopology ι
    ℱ : MeasureTheory.Filtration ι m
    inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    τ : Ω → ι
    f : ι → Ω → E
    n : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    h_countable_range : (Set.range τ).Countable
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    this : Eq Set.univ (Set.iUnion fun i => Set.iUnion fun h => setOf fun x => Eq  …
    ⊢ ∀ (i : ι), Membership.mem (Set.range τ) i → (MeasureTheory.ae (μ.restrict (s …
  -/
  exact fun i _ => stoppedValue_ae_eq_restrict_eq h _ hτ_le i
  /-
    🎉 no goals
  -/


/-- The value of a martingale `f` at a stopping time `τ` bounded by `n` is the conditional
expectation of `f n` with respect to the σ-algebra generated by `τ`. -/
theorem stoppedValue_ae_eq_condexp_of_le_const [Countable ι] (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hτ_le : ∀ x, τ x ≤ n)
    [SigmaFinite (μ.trim (hτ.measurableSpace_le_of_le hτ_le))] :
    stoppedValue f τ =ᵐ[μ] μ[f n|hτ.measurableSpace] :=
  h.stoppedValue_ae_eq_condexp_of_le_const_of_countable_range hτ hτ_le (Set.to_countable _)


/-- If `τ` and `σ` are two stopping times with `σ ≤ τ` and `τ` is bounded, then the value of a
martingale `f` at `σ` is the conditional expectation of its value at `τ` with respect to the
σ-algebra generated by `σ`. -/
theorem stoppedValue_ae_eq_condexp_of_le_of_countable_range (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hσ : IsStoppingTime ℱ σ) (hσ_le_τ : σ ≤ τ) (hτ_le : ∀ x, τ x ≤ n)
    (hτ_countable_range : (Set.range τ).Countable) (hσ_countable_range : (Set.range σ).Countable)
    [SigmaFinite (μ.trim (hσ.measurableSpace_le_of_le fun x => (hσ_le_τ x).trans (hτ_le x)))] :
    stoppedValue f σ =ᵐ[μ] μ[stoppedValue f τ|hσ.measurableSpace] := by
  have : SigmaFinite (μ.trim (hτ.measurableSpace_le_of_le hτ_le)) :=
    sigmaFiniteTrim_mono _ (IsStoppingTime.measurableSpace_mono hσ hτ hσ_le_τ)
  have : μ[stoppedValue f τ|hσ.measurableSpace] =ᵐ[μ]
      μ[μ[f n|hτ.measurableSpace]|hσ.measurableSpace] := condexp_congr_ae
    (h.stoppedValue_ae_eq_condexp_of_le_const_of_countable_range hτ hτ_le hτ_countable_range)
  refine (Filter.EventuallyEq.trans ?_
    (condexp_condexp_of_le ?_ (hτ.measurableSpace_le_of_le hτ_le)).symm).trans this.symm
  · exact h.stoppedValue_ae_eq_condexp_of_le_const_of_countable_range hσ
      (fun x => (hσ_le_τ x).trans (hτ_le x)) hσ_countable_range
    /-
      case refine_2
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      inst✝⁶ : CompleteSpace E
      ι : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : TopologicalSpace ι
      inst✝³ : OrderTopology ι
      inst✝² : FirstCountableTopology ι
      ℱ : MeasureTheory.Filtration ι m
      inst✝¹ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      τ σ : Ω → ι
      f : ι → Ω → E
      n : ι
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      hσ_le_τ : LE.le σ τ
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      hτ_countable_range : (Set.range τ).Countable
      hσ_countable_range : (Set.range σ).Countable
      inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      this✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
      this : (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp hσ.measurableS …
      ⊢ LE.le hσ.measurableSpace hτ.measurableSpace
    -/
  · exact hσ.measurableSpace_mono hτ hσ_le_τ
    /-
      🎉 no goals
    -/


/-- If `τ` and `σ` are two stopping times with `σ ≤ τ` and `τ` is bounded, then the value of a
martingale `f` at `σ` is the conditional expectation of its value at `τ` with respect to the
σ-algebra generated by `σ`. -/
theorem stoppedValue_ae_eq_condexp_of_le [Countable ι] (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hσ : IsStoppingTime ℱ σ) (hσ_le_τ : σ ≤ τ) (hτ_le : ∀ x, τ x ≤ n)
    [SigmaFinite (μ.trim hσ.measurableSpace_le)] :
    stoppedValue f σ =ᵐ[μ] μ[stoppedValue f τ|hσ.measurableSpace] :=
  h.stoppedValue_ae_eq_condexp_of_le_of_countable_range hτ hσ hσ_le_τ hτ_le (Set.to_countable _)
    (Set.to_countable _)


theorem condexp_stoppedValue_stopping_time_ae_eq_restrict_le (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hσ : IsStoppingTime ℱ σ) [SigmaFinite (μ.trim hσ.measurableSpace_le)]
    (hτ_le : ∀ x, τ x ≤ i) :
    μ[stoppedValue f τ|hσ.measurableSpace] =ᵐ[μ.restrict {x : Ω | τ x ≤ σ x}] stoppedValue f τ := by
  rw [ae_eq_restrict_iff_indicator_ae_eq
    (hτ.measurableSpace_le _ (hτ.measurableSet_le_stopping_time hσ))]
  refine (condexp_indicator (integrable_stoppedValue ι hτ h.integrable hτ_le)
    (hτ.measurableSet_stopping_time_le hσ)).symm.trans ?_
  have h_int :
      Integrable ({ω : Ω | τ ω ≤ σ ω}.indicator (stoppedValue (fun n : ι => f n) τ)) μ := by
    refine (integrable_stoppedValue ι hτ h.integrable hτ_le).indicator ?_
    exact hτ.measurableSpace_le _ (hτ.measurableSet_le_stopping_time hσ)
  have h_meas : AEStronglyMeasurable' hσ.measurableSpace
      ({ω : Ω | τ ω ≤ σ ω}.indicator (stoppedValue (fun n : ι => f n) τ)) μ := by
    refine StronglyMeasurable.aeStronglyMeasurable' ?_
    refine StronglyMeasurable.stronglyMeasurable_of_measurableSpace_le_on
      (hτ.measurableSet_le_stopping_time hσ) ?_ ?_ ?_
    · intro t ht
      rw [Set.inter_comm _ t] at ht ⊢
      rw [hτ.measurableSet_inter_le_iff hσ, IsStoppingTime.measurableSet_min_iff hτ hσ] at ht
      exact ht.2
    · refine StronglyMeasurable.indicator ?_ (hτ.measurableSet_le_stopping_time hσ)
      refine Measurable.stronglyMeasurable ?_
      exact measurable_stoppedValue h.adapted.progMeasurable_of_discrete hτ
    · intro x hx
      simp only [hx, Set.indicator_of_not_mem, not_false_iff]
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : CompleteSpace E
    ι : Type u_3
    inst✝¹⁰ : LinearOrder ι
    inst✝⁹ : LocallyFiniteOrder ι
    inst✝⁸ : OrderBot ι
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : DiscreteTopology ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : SecondCountableTopology E
    ℱ : MeasureTheory.Filtration ι m
    τ σ : Ω → ι
    f : ι → Ω → E
    i : ι
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hσ : MeasureTheory.IsStoppingTime ℱ σ
    inst✝ : MeasureTheory.SigmaFinite (μ.trim ⋯)
    hτ_le : ∀ (x : Ω), LE.le (τ x) i
    h_int : MeasureTheory.Integrable ((setOf fun ω => LE.le (τ ω) (σ ω)).indicator …
    h_meas : MeasureTheory.AEStronglyMeasurable' hσ.measurableSpace ((setOf fun ω  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp hσ.measurableSpace  …
  -/
  exact condexp_of_aestronglyMeasurable' hσ.measurableSpace_le h_meas h_int
  /-
    🎉 no goals
  -/


/-- **Optional Sampling theorem**. If `τ` is a bounded stopping time and `σ` is another stopping
time, then the value of a martingale `f` at the stopping time `min τ σ` is almost everywhere equal
to the conditional expectation of `f` stopped at `τ` with respect to the σ-algebra generated
by `σ`. -/
theorem stoppedValue_min_ae_eq_condexp [SigmaFiniteFiltration μ ℱ] (h : Martingale f ℱ μ)
    (hτ : IsStoppingTime ℱ τ) (hσ : IsStoppingTime ℱ σ) {n : ι} (hτ_le : ∀ x, τ x ≤ n)
    [h_sf_min : SigmaFinite (μ.trim (hτ.min hσ).measurableSpace_le)] :
    (stoppedValue f fun x => min (σ x) (τ x)) =ᵐ[μ] μ[stoppedValue f τ|hσ.measurableSpace] := by
  refine
    (h.stoppedValue_ae_eq_condexp_of_le hτ (hσ.min hτ) (fun x => min_le_right _ _) hτ_le).trans ?_
  /-
    Ω : Type u_1
    E : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : CompleteSpace E
    ι : Type u_3
    inst✝¹⁰ : LinearOrder ι
    inst✝⁹ : LocallyFiniteOrder ι
    inst✝⁸ : OrderBot ι
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : DiscreteTopology ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : BorelSpace ι
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : SecondCountableTopology E
    ℱ : MeasureTheory.Filtration ι m
    τ σ : Ω → ι
    f : ι → Ω → E
    inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
    h : MeasureTheory.Martingale f ℱ μ
    hτ : MeasureTheory.IsStoppingTime ℱ τ
    hσ : MeasureTheory.IsStoppingTime ℱ σ
    n : ι
    hτ_le : ∀ (x : Ω), LE.le (τ x) n
    h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp ⋯.measurableSpace μ …
  -/
  refine ae_of_ae_restrict_of_ae_restrict_compl {x | σ x ≤ τ x} ?_ ?_
    /-
      case refine_1
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace Real E
      inst✝¹¹ : CompleteSpace E
      ι : Type u_3
      inst✝¹⁰ : LinearOrder ι
      inst✝⁹ : LocallyFiniteOrder ι
      inst✝⁸ : OrderBot ι
      inst✝⁷ : TopologicalSpace ι
      inst✝⁶ : DiscreteTopology ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : BorelSpace ι
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : SecondCountableTopology E
      ℱ : MeasureTheory.Filtration ι m
      τ σ : Ω → ι
      f : ι → Ω → E
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      n : ι
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
      ⊢ Filter.Eventually (fun x => Eq (MeasureTheory.condexp ⋯.measurableSpace μ (M …
    -/
  · exact condexp_min_stopping_time_ae_eq_restrict_le hσ hτ
    /-
      🎉 no goals
    -/
  · suffices μ[stoppedValue f τ|(hσ.min hτ).measurableSpace] =ᵐ[μ.restrict {x | τ x ≤ σ x}]
        μ[stoppedValue f τ|hσ.measurableSpace] by
      rw [ae_restrict_iff' (hσ.measurableSpace_le _ (hσ.measurableSet_le_stopping_time hτ).compl)]
      rw [Filter.EventuallyEq, ae_restrict_iff'] at this
      swap; · exact hτ.measurableSpace_le _ (hτ.measurableSet_le_stopping_time hσ)
      filter_upwards [this] with x hx hx_mem
      simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hx_mem
      exact hx hx_mem.le
    /-
      case refine_2
      Ω : Type u_1
      E : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      inst✝¹³ : NormedAddCommGroup E
      inst✝¹² : NormedSpace Real E
      inst✝¹¹ : CompleteSpace E
      ι : Type u_3
      inst✝¹⁰ : LinearOrder ι
      inst✝⁹ : LocallyFiniteOrder ι
      inst✝⁸ : OrderBot ι
      inst✝⁷ : TopologicalSpace ι
      inst✝⁶ : DiscreteTopology ι
      inst✝⁵ : MeasurableSpace ι
      inst✝⁴ : BorelSpace ι
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      inst✝¹ : SecondCountableTopology E
      ℱ : MeasureTheory.Filtration ι m
      τ σ : Ω → ι
      f : ι → Ω → E
      inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
      h : MeasureTheory.Martingale f ℱ μ
      hτ : MeasureTheory.IsStoppingTime ℱ τ
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      n : ι
      hτ_le : ∀ (x : Ω), LE.le (τ x) n
      h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
      ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => LE.le (τ x) (σ x)))).Eventuall …
    -/
    apply Filter.EventuallyEq.trans _ ((condexp_min_stopping_time_ae_eq_restrict_le hτ hσ).trans _)
      /-
        Ω : Type u_1
        E : Type u_2
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace Real E
        inst✝¹¹ : CompleteSpace E
        ι : Type u_3
        inst✝¹⁰ : LinearOrder ι
        inst✝⁹ : LocallyFiniteOrder ι
        inst✝⁸ : OrderBot ι
        inst✝⁷ : TopologicalSpace ι
        inst✝⁶ : DiscreteTopology ι
        inst✝⁵ : MeasurableSpace ι
        inst✝⁴ : BorelSpace ι
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : SecondCountableTopology E
        ℱ : MeasureTheory.Filtration ι m
        τ σ : Ω → ι
        f : ι → Ω → E
        inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
        h : MeasureTheory.Martingale f ℱ μ
        hτ : MeasureTheory.IsStoppingTime ℱ τ
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        n : ι
        hτ_le : ∀ (x : Ω), LE.le (τ x) n
        h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
        ⊢ Ω → E
      -/
    · exact stoppedValue f τ
      /-
        🎉 no goals
      -/
      /-
        Ω : Type u_1
        E : Type u_2
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace Real E
        inst✝¹¹ : CompleteSpace E
        ι : Type u_3
        inst✝¹⁰ : LinearOrder ι
        inst✝⁹ : LocallyFiniteOrder ι
        inst✝⁸ : OrderBot ι
        inst✝⁷ : TopologicalSpace ι
        inst✝⁶ : DiscreteTopology ι
        inst✝⁵ : MeasurableSpace ι
        inst✝⁴ : BorelSpace ι
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : SecondCountableTopology E
        ℱ : MeasureTheory.Filtration ι m
        τ σ : Ω → ι
        f : ι → Ω → E
        inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
        h : MeasureTheory.Martingale f ℱ μ
        hτ : MeasureTheory.IsStoppingTime ℱ τ
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        n : ι
        hτ_le : ∀ (x : Ω), LE.le (τ x) n
        h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
        ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => LE.le (τ x) (σ x)))).Eventuall …
      -/
    · rw [IsStoppingTime.measurableSpace_min hσ, IsStoppingTime.measurableSpace_min hτ, inf_comm]
      /-
        🎉 no goals
      -/
    · have h1 : μ[stoppedValue f τ|hτ.measurableSpace] = stoppedValue f τ := by
        apply condexp_of_stronglyMeasurable hτ.measurableSpace_le
        · exact Measurable.stronglyMeasurable <|
            measurable_stoppedValue h.adapted.progMeasurable_of_discrete hτ
        · exact integrable_stoppedValue ι hτ h.integrable hτ_le
      /-
        Ω : Type u_1
        E : Type u_2
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace Real E
        inst✝¹¹ : CompleteSpace E
        ι : Type u_3
        inst✝¹⁰ : LinearOrder ι
        inst✝⁹ : LocallyFiniteOrder ι
        inst✝⁸ : OrderBot ι
        inst✝⁷ : TopologicalSpace ι
        inst✝⁶ : DiscreteTopology ι
        inst✝⁵ : MeasurableSpace ι
        inst✝⁴ : BorelSpace ι
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : SecondCountableTopology E
        ℱ : MeasureTheory.Filtration ι m
        τ σ : Ω → ι
        f : ι → Ω → E
        inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
        h : MeasureTheory.Martingale f ℱ μ
        hτ : MeasureTheory.IsStoppingTime ℱ τ
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        n : ι
        hτ_le : ∀ (x : Ω), LE.le (τ x) n
        h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
        h1 : Eq (MeasureTheory.condexp hτ.measurableSpace μ (MeasureTheory.stoppedValu …
        ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => LE.le (τ x) (σ x)))).Eventuall …
      -/
      rw [h1]
      /-
        Ω : Type u_1
        E : Type u_2
        m : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        inst✝¹³ : NormedAddCommGroup E
        inst✝¹² : NormedSpace Real E
        inst✝¹¹ : CompleteSpace E
        ι : Type u_3
        inst✝¹⁰ : LinearOrder ι
        inst✝⁹ : LocallyFiniteOrder ι
        inst✝⁸ : OrderBot ι
        inst✝⁷ : TopologicalSpace ι
        inst✝⁶ : DiscreteTopology ι
        inst✝⁵ : MeasurableSpace ι
        inst✝⁴ : BorelSpace ι
        inst✝³ : MeasurableSpace E
        inst✝² : BorelSpace E
        inst✝¹ : SecondCountableTopology E
        ℱ : MeasureTheory.Filtration ι m
        τ σ : Ω → ι
        f : ι → Ω → E
        inst✝ : MeasureTheory.SigmaFiniteFiltration μ ℱ
        h : MeasureTheory.Martingale f ℱ μ
        hτ : MeasureTheory.IsStoppingTime ℱ τ
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        n : ι
        hτ_le : ∀ (x : Ω), LE.le (τ x) n
        h_sf_min : MeasureTheory.SigmaFinite (μ.trim ⋯)
        h1 : Eq (MeasureTheory.condexp hτ.measurableSpace μ (MeasureTheory.stoppedValu …
        ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => LE.le (τ x) (σ x)))).Eventuall …
      -/
      exact (condexp_stoppedValue_stopping_time_ae_eq_restrict_le h hτ hσ hτ_le).symm
      /-
        🎉 no goals
      -/


