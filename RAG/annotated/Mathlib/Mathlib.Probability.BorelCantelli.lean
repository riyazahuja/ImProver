theorem iIndepFun.indep_comap_natural_of_lt (hf : ∀ i, StronglyMeasurable (f i))
    (hfi : iIndepFun (fun _ => mβ) f μ) (hij : i < j) :
    Indep (MeasurableSpace.comap (f j) mβ) (Filtration.natural f hf i) μ := by
  suffices Indep (⨆ k ∈ ({j} : Set ι), MeasurableSpace.comap (f k) mβ)
      (⨆ k ∈ {k | k ≤ i}, MeasurableSpace.comap (f k) mβ) μ by rwa [iSup_singleton] at this
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ι : Type u_2
    β : Type u_3
    inst✝² : LinearOrder ι
    mβ : MeasurableSpace β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : BorelSpace β
    f : ι → Ω → β
    i j : ι
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hfi : ProbabilityTheory.iIndepFun (fun x => mβ) f μ
    hij : LT.lt i j
    ⊢ ProbabilityTheory.Indep (iSup fun k => iSup fun h => MeasurableSpace.comap ( …
  -/
  exact indep_iSup_of_disjoint (fun k => (hf k).measurable.comap_le) hfi (by simpa)
  /-
    🎉 no goals
  -/


theorem iIndepFun.condexp_natural_ae_eq_of_lt [SecondCountableTopology β] [CompleteSpace β]
    [NormedSpace ℝ β] (hf : ∀ i, StronglyMeasurable (f i)) (hfi : iIndepFun (fun _ => mβ) f μ)
    (hij : i < j) : μ[f j|Filtration.natural f hf i] =ᵐ[μ] fun _ => μ[f j] := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ι : Type u_2
    β : Type u_3
    inst✝⁵ : LinearOrder ι
    mβ : MeasurableSpace β
    inst✝⁴ : NormedAddCommGroup β
    inst✝³ : BorelSpace β
    f : ι → Ω → β
    i j : ι
    inst✝² : SecondCountableTopology β
    inst✝¹ : CompleteSpace β
    inst✝ : NormedSpace Real β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    hfi : ProbabilityTheory.iIndepFun (fun x => mβ) f μ
    hij : LT.lt i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑(MeasureTheory.Fi …
  -/
  have : IsProbabilityMeasure μ := hfi.isProbabilityMeasure
  exact condexp_indep_eq (hf j).measurable.comap_le (Filtration.le _ _)
    (comap_measurable <| f j).stronglyMeasurable (hfi.indep_comap_natural_of_lt hf hij)


theorem iIndepSet.condexp_indicator_filtrationOfSet_ae_eq (hsm : ∀ n, MeasurableSet (s n))
    (hs : iIndepSet s μ) (hij : i < j) :
    μ[(s j).indicator (fun _ => 1 : Ω → ℝ)|filtrationOfSet hsm i] =ᵐ[μ]
    fun _ => (μ (s j)).toReal := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ι : Type u_2
    inst✝ : LinearOrder ι
    i j : ι
    s : ι → Set Ω
    hsm : ∀ (n : ι), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hij : LT.lt i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑(MeasureTheory.fi …
  -/
  rw [Filtration.filtrationOfSet_eq_natural (β := ℝ) hsm]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ι : Type u_2
    inst✝ : LinearOrder ι
    i j : ι
    s : ι → Set Ω
    hsm : ∀ (n : ι), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hij : LT.lt i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.condexp (↑(MeasureTheory.Fi …
  -/
  refine (iIndepFun.condexp_natural_ae_eq_of_lt _ hs.iIndepFun_indicator hij).trans ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ι : Type u_2
    inst✝ : LinearOrder ι
    i j : ι
    s : ι → Set Ω
    hsm : ∀ (n : ι), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hij : LT.lt i j
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => MeasureTheory.integral μ fun x = …
  -/
  simp only [integral_indicator_const _ (hsm _), Algebra.id.smul_eq_mul, mul_one]; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- **The second Borel-Cantelli lemma**: Given a sequence of independent sets `(sₙ)` such that
`∑ n, μ sₙ = ∞`, `limsup sₙ` has measure 1. -/
theorem measure_limsup_eq_one {s : ℕ → Set Ω} (hsm : ∀ n, MeasurableSet (s n)) (hs : iIndepSet s μ)
    (hs' : (∑' n, μ (s n)) = ∞) : μ (limsup s atTop) = 1 := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hs' : Eq (tsum fun n => μ (s n)) Top.top
    ⊢ Eq (μ (Filter.limsup s Filter.atTop)) 1
  -/
  have : IsProbabilityMeasure μ := hs.isProbabilityMeasure
  rw [measure_congr (eventuallyEq_set.2 (ae_mem_limsup_atTop_iff μ <|
    measurableSet_filtrationOfSet' hsm) : (limsup s atTop : Set Ω) =ᵐ[μ]
      {ω | Tendsto (fun n => ∑ k ∈ Finset.range n,
        (μ[(s (k + 1)).indicator (1 : Ω → ℝ)|filtrationOfSet hsm k]) ω) atTop atTop})]
  suffices {ω | Tendsto (fun n => ∑ k ∈ Finset.range n,
      (μ[(s (k + 1)).indicator (1 : Ω → ℝ)|filtrationOfSet hsm k]) ω) atTop atTop} =ᵐ[μ] Set.univ by
    rw [measure_congr this, measure_univ]
  have : ∀ᵐ ω ∂μ, ∀ n, (μ[(s (n + 1)).indicator (1 : Ω → ℝ)|filtrationOfSet hsm n]) ω = _ :=
    ae_all_iff.2 fun n => hs.condexp_indicator_filtrationOfSet_ae_eq hsm n.lt_succ_self
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hs' : Eq (tsum fun n => μ (s n)) Top.top
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (setOf fun ω => Filter.Tendsto (fun n => ( …
  -/
  filter_upwards [this] with ω hω
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hs' : Eq (tsum fun n => μ (s n)) Top.top
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
    ω : Ω
    hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
    ⊢ Eq (setOf (fun ω => Filter.Tendsto (fun n => (Finset.range n).sum fun k => M …
  -/
  refine eq_true (?_ : Tendsto _ _ _)
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hs' : Eq (tsum fun n => μ (s n)) Top.top
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
    ω : Ω
    hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun k => MeasureTheory.condexp …
  -/
  simp_rw [hω]
  have htends : Tendsto (fun n => ∑ k ∈ Finset.range n, μ (s (k + 1))) atTop (𝓝 ∞) := by
    rw [← ENNReal.tsum_add_one_eq_top hs' (measure_ne_top _ _)]
    exact ENNReal.tendsto_nat_tsum _
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hs' : Eq (tsum fun n => μ (s n)) Top.top
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
    ω : Ω
    hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
    htends : Filter.Tendsto (fun n => (Finset.range n).sum fun k => μ (s (HAdd.hAd …
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun x => (μ (s (HAdd.hAdd x 1) …
  -/
  rw [ENNReal.tendsto_nhds_top_iff_nnreal] at htends
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    hsm : ∀ (n : Nat), MeasurableSet (s n)
    hs : ProbabilityTheory.iIndepSet s μ
    hs' : Eq (tsum fun n => μ (s n)) Top.top
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
    ω : Ω
    hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
    htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
    ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun x => (μ (s (HAdd.hAdd x 1) …
  -/
  refine tendsto_atTop_atTop_of_monotone' ?_ ?_
    /-
      case h.refine_1
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      ⊢ Monotone fun n => (Finset.range n).sum fun x => (μ (s (HAdd.hAdd x 1))).toReal
    -/
  · refine monotone_nat_of_le_succ fun n => ?_
    /-
      case h.refine_1
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      n : Nat
      ⊢ LE.le ((Finset.range n).sum fun x => (μ (s (HAdd.hAdd x 1))).toReal) ((Finse …
    -/
    rw [← sub_nonneg, Finset.sum_range_succ_sub_sum]
    /-
      case h.refine_1
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      n : Nat
      ⊢ LE.le 0 (μ (s (HAdd.hAdd n 1))).toReal
    -/
    exact ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      ⊢ Not (BddAbove (Set.range fun n => (Finset.range n).sum fun x => (μ (s (HAdd. …
    -/
  · rintro ⟨B, hB⟩
    /-
      case h.refine_2.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      B : Real
      hB : Membership.mem (upperBounds (Set.range fun n => (Finset.range n).sum fun  …
      ⊢ False
    -/
    refine not_eventually.2 (Frequently.of_forall fun n => ?_) (htends B.toNNReal)
    /-
      case h.refine_2.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      B : Real
      hB : Membership.mem (upperBounds (Set.range fun n => (Finset.range n).sum fun  …
      n : Nat
      ⊢ Not (LT.lt (↑B.toNNReal) ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1) …
    -/
    rw [mem_upperBounds] at hB
    /-
      case h.refine_2.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      s : Nat → Set Ω
      hsm : ∀ (n : Nat), MeasurableSet (s n)
      hs : ProbabilityTheory.iIndepSet s μ
      hs' : Eq (tsum fun n => μ (s n)) Top.top
      this✝ : MeasureTheory.IsProbabilityMeasure μ
      this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
      ω : Ω
      hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
      htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
      B : Real
      hB : ∀ (x : Real), Membership.mem (Set.range fun n => (Finset.range n).sum fun …
      n : Nat
      ⊢ Not (LT.lt (↑B.toNNReal) ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1) …
    -/
    specialize hB (∑ k ∈ Finset.range n, μ (s (k + 1))).toReal _
      /-
        case h.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        s : Nat → Set Ω
        hsm : ∀ (n : Nat), MeasurableSet (s n)
        hs : ProbabilityTheory.iIndepSet s μ
        hs' : Eq (tsum fun n => μ (s n)) Top.top
        this✝ : MeasureTheory.IsProbabilityMeasure μ
        this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
        ω : Ω
        hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
        htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
        B : Real
        hB : ∀ (x : Real), Membership.mem (Set.range fun n => (Finset.range n).sum fun …
        n : Nat
        ⊢ Membership.mem (Set.range fun n => (Finset.range n).sum fun x => (μ (s (HAdd …
      -/
    · refine ⟨n, ?_⟩
      /-
        case h.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        s : Nat → Set Ω
        hsm : ∀ (n : Nat), MeasurableSet (s n)
        hs : ProbabilityTheory.iIndepSet s μ
        hs' : Eq (tsum fun n => μ (s n)) Top.top
        this✝ : MeasureTheory.IsProbabilityMeasure μ
        this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
        ω : Ω
        hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
        htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
        B : Real
        hB : ∀ (x : Real), Membership.mem (Set.range fun n => (Finset.range n).sum fun …
        n : Nat
        ⊢ Eq ((fun n => (Finset.range n).sum fun x => (μ (s (HAdd.hAdd x 1))).toReal)  …
      -/
      rw [ENNReal.toReal_sum]
      /-
        case h.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        s : Nat → Set Ω
        hsm : ∀ (n : Nat), MeasurableSet (s n)
        hs : ProbabilityTheory.iIndepSet s μ
        hs' : Eq (tsum fun n => μ (s n)) Top.top
        this✝ : MeasureTheory.IsProbabilityMeasure μ
        this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
        ω : Ω
        hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
        htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
        B : Real
        hB : ∀ (x : Real), Membership.mem (Set.range fun n => (Finset.range n).sum fun …
        n : Nat
        ⊢ ∀ (a : Nat), Membership.mem (Finset.range n) a → Ne (μ (s (HAdd.hAdd a 1)))  …
      -/
      exact fun _ _ => measure_ne_top _ _
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        s : Nat → Set Ω
        hsm : ∀ (n : Nat), MeasurableSet (s n)
        hs : ProbabilityTheory.iIndepSet s μ
        hs' : Eq (tsum fun n => μ (s n)) Top.top
        this✝ : MeasureTheory.IsProbabilityMeasure μ
        this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
        ω : Ω
        hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
        htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
        B : Real
        n : Nat
        hB : LE.le ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1))).toReal B
        ⊢ Not (LT.lt (↑B.toNNReal) ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1) …
      -/
    · rwa [not_lt, ENNReal.ofNNReal_toNNReal, ENNReal.le_ofReal_iff_toReal_le]
        /-
          case h.refine_2.intro.ha
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          s : Nat → Set Ω
          hsm : ∀ (n : Nat), MeasurableSet (s n)
          hs : ProbabilityTheory.iIndepSet s μ
          hs' : Eq (tsum fun n => μ (s n)) Top.top
          this✝ : MeasureTheory.IsProbabilityMeasure μ
          this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
          ω : Ω
          hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
          htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
          B : Real
          n : Nat
          hB : LE.le ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1))).toReal B
          ⊢ Ne ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1))) Top.top
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case h.refine_2.intro.hb
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          s : Nat → Set Ω
          hsm : ∀ (n : Nat), MeasurableSet (s n)
          hs : ProbabilityTheory.iIndepSet s μ
          hs' : Eq (tsum fun n => μ (s n)) Top.top
          this✝ : MeasureTheory.IsProbabilityMeasure μ
          this : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(M …
          ω : Ω
          hω : ∀ (n : Nat), Eq (MeasureTheory.condexp (↑(MeasureTheory.filtrationOfSet h …
          htends : ∀ (x : NNReal), Filter.Eventually (fun a => LT.lt (↑x) ((Finset.range …
          B : Real
          n : Nat
          hB : LE.le ((Finset.range n).sum fun k => μ (s (HAdd.hAdd k 1))).toReal B
          ⊢ LE.le 0 B
        -/
      · exact le_trans (by positivity) hB
        /-
          🎉 no goals
        -/


