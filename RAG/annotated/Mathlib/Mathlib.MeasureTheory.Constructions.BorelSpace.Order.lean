theorem borel_eq_generateFrom_Iio : borel α = .generateFrom (range Iio) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : SecondCountableTopology α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    ⊢ Eq (borel α) (MeasurableSpace.generateFrom (Set.range Set.Iio))
  -/
  refine le_antisymm ?_ (generateFrom_le ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      ⊢ LE.le (borel α) (MeasurableSpace.generateFrom (Set.range Set.Iio))
    -/
  · rw [borel_eq_generateFrom_of_subbasis (@OrderTopology.topology_eq_generate_intervals α _ _ _)]
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun s => Exists fun a => Or (Eq s …
    -/
    letI : MeasurableSpace α := MeasurableSpace.generateFrom (range Iio)
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
      ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun s => Exists fun a => Or (Eq s …
    -/
    have H : ∀ a : α, MeasurableSet (Iio a) := fun a => GenerateMeasurable.basic _ ⟨_, rfl⟩
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
      H : ∀ (a : α), MeasurableSet (Set.Iio a)
      ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun s => Exists fun a => Or (Eq s …
    -/
    refine generateFrom_le ?_
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
      H : ∀ (a : α), MeasurableSet (Set.Iio a)
      ⊢ ∀ (t : Set α), Membership.mem (setOf fun s => Exists fun a => Or (Eq s (Set. …
    -/
    rintro _ ⟨a, rfl | rfl⟩
      /-
        case refine_1.intro.inl
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : SecondCountableTopology α
        inst✝¹ : LinearOrder α
        inst✝ : OrderTopology α
        this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
        H : ∀ (a : α), MeasurableSet (Set.Iio a)
        a : α
        ⊢ MeasurableSet (Set.Ioi a)
      -/
    · rcases em (∃ b, a ⋖ b) with ⟨b, hb⟩ | hcovBy
        /-
          case refine_1.intro.inl.inl.intro
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : SecondCountableTopology α
          inst✝¹ : LinearOrder α
          inst✝ : OrderTopology α
          this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
          H : ∀ (a : α), MeasurableSet (Set.Iio a)
          a b : α
          hb : CovBy a b
          ⊢ MeasurableSet (Set.Ioi a)
        -/
      · rw [hb.Ioi_eq, ← compl_Iio]
        /-
          case refine_1.intro.inl.inl.intro
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : SecondCountableTopology α
          inst✝¹ : LinearOrder α
          inst✝ : OrderTopology α
          this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
          H : ∀ (a : α), MeasurableSet (Set.Iio a)
          a b : α
          hb : CovBy a b
          ⊢ MeasurableSet (HasCompl.compl (Set.Iio b))
        -/
        exact (H _).compl
        /-
          🎉 no goals
        -/
        /-
          case refine_1.intro.inl.inr
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : SecondCountableTopology α
          inst✝¹ : LinearOrder α
          inst✝ : OrderTopology α
          this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
          H : ∀ (a : α), MeasurableSet (Set.Iio a)
          a : α
          hcovBy : Not (Exists fun b => CovBy a b)
          ⊢ MeasurableSet (Set.Ioi a)
        -/
      · rcases isOpen_biUnion_countable (Ioi a) Ioi fun _ _ ↦ isOpen_Ioi with ⟨t, hat, htc, htU⟩
        have : Ioi a = ⋃ b ∈ t, Ici b := by
          refine Subset.antisymm ?_ <| iUnion₂_subset fun b hb ↦ Ici_subset_Ioi.2 (hat hb)
          refine Subset.trans ?_ <| iUnion₂_mono fun _ _ ↦ Ioi_subset_Ici_self
          simpa [CovBy, htU, subset_def] using hcovBy
        /-
          case refine_1.intro.inl.inr.intro.intro.intro
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : SecondCountableTopology α
          inst✝¹ : LinearOrder α
          inst✝ : OrderTopology α
          this✝ : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
          H : ∀ (a : α), MeasurableSet (Set.Iio a)
          a : α
          hcovBy : Not (Exists fun b => CovBy a b)
          t : Set α
          hat : HasSubset.Subset t (Set.Ioi a)
          htc : t.Countable
          htU : Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioi i) (Set.iUnion fun i …
          this : Eq (Set.Ioi a) (Set.iUnion fun b => Set.iUnion fun h => Set.Ici b)
          ⊢ MeasurableSet (Set.Ioi a)
        -/
        simp only [this, ← compl_Iio]
        /-
          case refine_1.intro.inl.inr.intro.intro.intro
          α : Type u_1
          inst✝³ : TopologicalSpace α
          inst✝² : SecondCountableTopology α
          inst✝¹ : LinearOrder α
          inst✝ : OrderTopology α
          this✝ : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
          H : ∀ (a : α), MeasurableSet (Set.Iio a)
          a : α
          hcovBy : Not (Exists fun b => CovBy a b)
          t : Set α
          hat : HasSubset.Subset t (Set.Ioi a)
          htc : t.Countable
          htU : Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioi i) (Set.iUnion fun i …
          this : Eq (Set.Ioi a) (Set.iUnion fun b => Set.iUnion fun h => Set.Ici b)
          ⊢ MeasurableSet (Set.iUnion fun b => Set.iUnion fun x => HasCompl.compl (Set.I …
        -/
        exact .biUnion htc <| fun _ _ ↦ (H _).compl
        /-
          🎉 no goals
        -/
      /-
        case refine_1.intro.inr
        α : Type u_1
        inst✝³ : TopologicalSpace α
        inst✝² : SecondCountableTopology α
        inst✝¹ : LinearOrder α
        inst✝ : OrderTopology α
        this : MeasurableSpace α := MeasurableSpace.generateFrom (Set.range Set.Iio)
        H : ∀ (a : α), MeasurableSet (Set.Iio a)
        a : α
        ⊢ MeasurableSet (Set.Iio a)
      -/
    · apply H
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      ⊢ ∀ (t : Set α), Membership.mem (Set.range Set.Iio) t → MeasurableSet t
    -/
  · rw [forall_mem_range]
    /-
      case refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      ⊢ ∀ (i : α), MeasurableSet (Set.Iio i)
    -/
    intro a
    /-
      case refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      ⊢ MeasurableSet (Set.Iio a)
    -/
    exact GenerateMeasurable.basic _ isOpen_Iio
    /-
      🎉 no goals
    -/


theorem borel_eq_generateFrom_Ioi : borel α = .generateFrom (range Ioi) :=
                                       /-
                                         α : Type u_1
                                         inst✝³ : TopologicalSpace α
                                         inst✝² : SecondCountableTopology α
                                         inst✝¹ : LinearOrder α
                                         inst✝ : OrderTopology α
                                         ⊢ SecondCountableTopology α
                                       -/
  @borel_eq_generateFrom_Iio αᵒᵈ _ (by infer_instance : SecondCountableTopology α) _ _
                                       /-
                                         🎉 no goals
                                       -/


theorem borel_eq_generateFrom_Iic :
    borel α = MeasurableSpace.generateFrom (range Iic) := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : SecondCountableTopology α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    ⊢ Eq (borel α) (MeasurableSpace.generateFrom (Set.range Set.Iic))
  -/
  rw [borel_eq_generateFrom_Ioi]
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : SecondCountableTopology α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    ⊢ Eq (MeasurableSpace.generateFrom (Set.range Set.Ioi)) (MeasurableSpace.gener …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      ⊢ LE.le (MeasurableSpace.generateFrom (Set.range Set.Ioi)) (MeasurableSpace.ge …
    -/
  · refine MeasurableSpace.generateFrom_le fun t ht => ?_
    /-
      case refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      t : Set α
      ht : Membership.mem (Set.range Set.Ioi) t
      ⊢ MeasurableSet t
    -/
    obtain ⟨u, rfl⟩ := ht
    /-
      case refine_1.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      u : α
      ⊢ MeasurableSet (Set.Ioi u)
    -/
    rw [← compl_Iic]
    /-
      case refine_1.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      u : α
      ⊢ MeasurableSet (HasCompl.compl (Set.Iic u))
    -/
    exact (MeasurableSpace.measurableSet_generateFrom (mem_range.mpr ⟨u, rfl⟩)).compl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      ⊢ LE.le (MeasurableSpace.generateFrom (Set.range Set.Iic)) (MeasurableSpace.ge …
    -/
  · refine MeasurableSpace.generateFrom_le fun t ht => ?_
    /-
      case refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      t : Set α
      ht : Membership.mem (Set.range Set.Iic) t
      ⊢ MeasurableSet t
    -/
    obtain ⟨u, rfl⟩ := ht
    /-
      case refine_2.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      u : α
      ⊢ MeasurableSet (Set.Iic u)
    -/
    rw [← compl_Ioi]
    /-
      case refine_2.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : SecondCountableTopology α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      u : α
      ⊢ MeasurableSet (HasCompl.compl (Set.Ioi u))
    -/
    exact (MeasurableSpace.measurableSet_generateFrom (mem_range.mpr ⟨u, rfl⟩)).compl
    /-
      🎉 no goals
    -/


theorem borel_eq_generateFrom_Ici : borel α = MeasurableSpace.generateFrom (range Ici) :=
  @borel_eq_generateFrom_Iic αᵒᵈ _ _ _ _


@[simp, measurability]
theorem measurableSet_Ici : MeasurableSet (Ici a) :=
  isClosed_Ici.measurableSet


theorem nullMeasurableSet_Ici : NullMeasurableSet (Ici a) μ :=
  measurableSet_Ici.nullMeasurableSet


@[simp, measurability]
theorem measurableSet_Iic : MeasurableSet (Iic a) :=
  isClosed_Iic.measurableSet


theorem nullMeasurableSet_Iic : NullMeasurableSet (Iic a) μ :=
  measurableSet_Iic.nullMeasurableSet


@[simp, measurability]
theorem measurableSet_Icc : MeasurableSet (Icc a b) :=
  isClosed_Icc.measurableSet


theorem nullMeasurableSet_Icc : NullMeasurableSet (Icc a b) μ :=
  measurableSet_Icc.nullMeasurableSet


instance nhdsWithin_Ici_isMeasurablyGenerated : (𝓝[Ici b] a).IsMeasurablyGenerated :=
  measurableSet_Ici.nhdsWithin_isMeasurablyGenerated _


instance nhdsWithin_Iic_isMeasurablyGenerated : (𝓝[Iic b] a).IsMeasurablyGenerated :=
  measurableSet_Iic.nhdsWithin_isMeasurablyGenerated _


instance nhdsWithin_Icc_isMeasurablyGenerated : IsMeasurablyGenerated (𝓝[Icc a b] x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Sort y
    s t u : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    mδ : MeasurableSpace δ
    inst✝¹ : Preorder α
    inst✝ : OrderClosedTopology α
    a b x : α
    μ : MeasureTheory.Measure α
    ⊢ (nhdsWithin x (Set.Icc a b)).IsMeasurablyGenerated
  -/
  rw [← Ici_inter_Iic, nhdsWithin_inter]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Sort y
    s t u : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    mδ : MeasurableSpace δ
    inst✝¹ : Preorder α
    inst✝ : OrderClosedTopology α
    a b x : α
    μ : MeasureTheory.Measure α
    ⊢ (Min.min (nhdsWithin x (Set.Ici a)) (nhdsWithin x (Set.Iic b))).IsMeasurably …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance atTop_isMeasurablyGenerated : (Filter.atTop : Filter α).IsMeasurablyGenerated :=
  @Filter.iInf_isMeasurablyGenerated _ _ _ _ fun a =>
    (measurableSet_Ici : MeasurableSet (Ici a)).principal_isMeasurablyGenerated


instance atBot_isMeasurablyGenerated : (Filter.atBot : Filter α).IsMeasurablyGenerated :=
  @Filter.iInf_isMeasurablyGenerated _ _ _ _ fun a =>
    (measurableSet_Iic : MeasurableSet (Iic a)).principal_isMeasurablyGenerated


instance [R1Space α] : IsMeasurablyGenerated (cocompact α) where
  exists_measurable_subset := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      mδ : MeasurableSpace δ
      inst✝² : Preorder α
      inst✝¹ : OrderClosedTopology α
      a b x : α
      μ : MeasureTheory.Measure α
      inst✝ : R1Space α
      ⊢ ∀ ⦃s : Set α⦄, Membership.mem (Filter.cocompact α) s → Exists fun t => And ( …
    -/
    intro _ hs
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      mδ : MeasurableSpace δ
      inst✝² : Preorder α
      inst✝¹ : OrderClosedTopology α
      a b x : α
      μ : MeasureTheory.Measure α
      inst✝ : R1Space α
      s✝ : Set α
      hs : Membership.mem (Filter.cocompact α) s✝
      ⊢ Exists fun t => And (Membership.mem (Filter.cocompact α) t) (And (Measurable …
    -/
    obtain ⟨t, ht, hts⟩ := mem_cocompact.mp hs
    exact ⟨(closure t)ᶜ, ht.closure.compl_mem_cocompact, isClosed_closure.measurableSet.compl,
      (compl_subset_compl.2 subset_closure).trans hts⟩


@[measurability]
theorem measurableSet_le' : MeasurableSet { p : α × α | p.1 ≤ p.2 } :=
  OrderClosedTopology.isClosed_le'.measurableSet


@[measurability]
theorem measurableSet_le {f g : δ → α} (hf : Measurable f) (hg : Measurable g) :
    MeasurableSet { a | f a ≤ g a } :=
  hf.prod_mk hg measurableSet_le'


@[simp, measurability]
theorem measurableSet_Iio : MeasurableSet (Iio a) :=
  isOpen_Iio.measurableSet


theorem nullMeasurableSet_Iio : NullMeasurableSet (Iio a) μ :=
  measurableSet_Iio.nullMeasurableSet


@[simp, measurability]
theorem measurableSet_Ioi : MeasurableSet (Ioi a) :=
  isOpen_Ioi.measurableSet


theorem nullMeasurableSet_Ioi : NullMeasurableSet (Ioi a) μ :=
  measurableSet_Ioi.nullMeasurableSet


@[simp, measurability]
theorem measurableSet_Ioo : MeasurableSet (Ioo a b) :=
  isOpen_Ioo.measurableSet


theorem nullMeasurableSet_Ioo : NullMeasurableSet (Ioo a b) μ :=
  measurableSet_Ioo.nullMeasurableSet


@[simp, measurability]
theorem measurableSet_Ioc : MeasurableSet (Ioc a b) :=
  measurableSet_Ioi.inter measurableSet_Iic


theorem nullMeasurableSet_Ioc : NullMeasurableSet (Ioc a b) μ :=
  measurableSet_Ioc.nullMeasurableSet


@[simp, measurability]
theorem measurableSet_Ico : MeasurableSet (Ico a b) :=
  measurableSet_Ici.inter measurableSet_Iio


theorem nullMeasurableSet_Ico : NullMeasurableSet (Ico a b) μ :=
  measurableSet_Ico.nullMeasurableSet


instance nhdsWithin_Ioi_isMeasurablyGenerated : (𝓝[Ioi b] a).IsMeasurablyGenerated :=
  measurableSet_Ioi.nhdsWithin_isMeasurablyGenerated _


instance nhdsWithin_Iio_isMeasurablyGenerated : (𝓝[Iio b] a).IsMeasurablyGenerated :=
  measurableSet_Iio.nhdsWithin_isMeasurablyGenerated _


instance nhdsWithin_uIcc_isMeasurablyGenerated : IsMeasurablyGenerated (𝓝[[[a, b]]] x) :=
  nhdsWithin_Icc_isMeasurablyGenerated


@[measurability]
theorem measurableSet_lt' [SecondCountableTopology α] : MeasurableSet { p : α × α | p.1 < p.2 } :=
  (isOpen_lt continuous_fst continuous_snd).measurableSet


@[measurability]
theorem measurableSet_lt [SecondCountableTopology α] {f g : δ → α} (hf : Measurable f)
    (hg : Measurable g) : MeasurableSet { a | f a < g a } :=
  hf.prod_mk hg measurableSet_lt'


theorem nullMeasurableSet_lt [SecondCountableTopology α] {μ : Measure δ} {f g : δ → α}
    (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) : NullMeasurableSet { a | f a < g a } μ :=
  (hf.prod_mk hg).nullMeasurable measurableSet_lt'


theorem nullMeasurableSet_lt' [SecondCountableTopology α] {μ : Measure (α × α)} :
    NullMeasurableSet { p : α × α | p.1 < p.2 } μ :=
  measurableSet_lt'.nullMeasurableSet


theorem nullMeasurableSet_le [SecondCountableTopology α] {μ : Measure δ}
    {f g : δ → α} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    NullMeasurableSet { a | f a ≤ g a } μ :=
  (hf.prod_mk hg).nullMeasurable measurableSet_le'


theorem Set.OrdConnected.measurableSet (h : OrdConnected s) : MeasurableSet s := by
  /-
    α : Type u_1
    s : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    h : s.OrdConnected
    ⊢ MeasurableSet s
  -/
  let u := ⋃ (x ∈ s) (y ∈ s), Ioo x y
  /-
    α : Type u_1
    s : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    h : s.OrdConnected
    u : Set α := Set.iUnion fun x => Set.iUnion fun h => Set.iUnion fun y => Set.i …
    ⊢ MeasurableSet s
  -/
  have huopen : IsOpen u := isOpen_biUnion fun _ _ => isOpen_biUnion fun _ _ => isOpen_Ioo
  /-
    α : Type u_1
    s : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    h : s.OrdConnected
    u : Set α := Set.iUnion fun x => Set.iUnion fun h => Set.iUnion fun y => Set.i …
    huopen : IsOpen u
    ⊢ MeasurableSet s
  -/
  have humeas : MeasurableSet u := huopen.measurableSet
  /-
    α : Type u_1
    s : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    h : s.OrdConnected
    u : Set α := Set.iUnion fun x => Set.iUnion fun h => Set.iUnion fun y => Set.i …
    huopen : IsOpen u
    humeas : MeasurableSet u
    ⊢ MeasurableSet s
  -/
  have hfinite : (s \ u).Finite := s.finite_diff_iUnion_Ioo
  have : u ⊆ s := iUnion₂_subset fun x hx => iUnion₂_subset fun y hy =>
    Ioo_subset_Icc_self.trans (h.out hx hy)
  /-
    α : Type u_1
    s : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    h : s.OrdConnected
    u : Set α := Set.iUnion fun x => Set.iUnion fun h => Set.iUnion fun y => Set.i …
    huopen : IsOpen u
    humeas : MeasurableSet u
    hfinite : (SDiff.sdiff s u).Finite
    this : HasSubset.Subset u s
    ⊢ MeasurableSet s
  -/
  rw [← union_diff_cancel this]
  /-
    α : Type u_1
    s : Set α
    inst✝³ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    h : s.OrdConnected
    u : Set α := Set.iUnion fun x => Set.iUnion fun h => Set.iUnion fun y => Set.i …
    huopen : IsOpen u
    humeas : MeasurableSet u
    hfinite : (SDiff.sdiff s u).Finite
    this : HasSubset.Subset u s
    ⊢ MeasurableSet (Union.union u (SDiff.sdiff s u))
  -/
  exact humeas.union hfinite.measurableSet
  /-
    🎉 no goals
  -/


theorem IsPreconnected.measurableSet (h : IsPreconnected s) : MeasurableSet s :=
  h.ordConnected.measurableSet


theorem generateFrom_Ico_mem_le_borel {α : Type*} [TopologicalSpace α] [LinearOrder α]
    [OrderClosedTopology α] (s t : Set α) :
    MeasurableSpace.generateFrom { S | ∃ l ∈ s, ∃ u ∈ t, l < u ∧ Ico l u = S }
      ≤ borel α := by
  /-
    α : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    s t : Set α
    ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun S => Exists fun l => And (Mem …
  -/
  apply generateFrom_le
  /-
    case h
    α : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    s t : Set α
    ⊢ ∀ (t_1 : Set α), Membership.mem (setOf fun S => Exists fun l => And (Members …
  -/
  borelize α
  /-
    case h
    α : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    s t : Set α
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    ⊢ ∀ (t_1 : Set α), Membership.mem (setOf fun S => Exists fun l => And (Members …
  -/
  rintro _ ⟨a, -, b, -, -, rfl⟩
  /-
    case h.intro.intro.intro.intro.intro
    α : Type u_5
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderClosedTopology α
    s t : Set α
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    a b : α
    ⊢ MeasurableSet (Set.Ico a b)
  -/
  exact measurableSet_Ico
  /-
    🎉 no goals
  -/


theorem Dense.borel_eq_generateFrom_Ico_mem_aux {α : Type*} [TopologicalSpace α] [LinearOrder α]
    [OrderTopology α] [SecondCountableTopology α] {s : Set α} (hd : Dense s)
    (hbot : ∀ x, IsBot x → x ∈ s) (hIoo : ∀ x y : α, x < y → Ioo x y = ∅ → y ∈ s) :
    borel α = .generateFrom { S : Set α | ∃ l ∈ s, ∃ u ∈ s, l < u ∧ Ico l u = S } := by
  /-
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    ⊢ Eq (borel α) (MeasurableSpace.generateFrom (setOf fun S => Exists fun l => A …
  -/
  set S : Set (Set α) := { S | ∃ l ∈ s, ∃ u ∈ s, l < u ∧ Ico l u = S }
  /-
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
    ⊢ Eq (borel α) (MeasurableSpace.generateFrom S)
  -/
  refine le_antisymm ?_ (generateFrom_Ico_mem_le_borel _ _)
  /-
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
    ⊢ LE.le (borel α) (MeasurableSpace.generateFrom S)
  -/
  letI : MeasurableSpace α := generateFrom S
  /-
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
    this : MeasurableSpace α := MeasurableSpace.generateFrom S
    ⊢ LE.le (borel α) (MeasurableSpace.generateFrom S)
  -/
  rw [borel_eq_generateFrom_Iio]
  /-
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
    this : MeasurableSpace α := MeasurableSpace.generateFrom S
    ⊢ LE.le (MeasurableSpace.generateFrom (Set.range Set.Iio)) (MeasurableSpace.ge …
  -/
  refine generateFrom_le (forall_mem_range.2 fun a => ?_)
  /-
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
    this : MeasurableSpace α := MeasurableSpace.generateFrom S
    a : α
    ⊢ MeasurableSet (Set.Iio a)
  -/
  rcases hd.exists_countable_dense_subset_bot_top with ⟨t, hts, hc, htd, htb, -⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_5
    inst✝³ : TopologicalSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    hd : Dense s
    hbot : ∀ (x : α), IsBot x → Membership.mem s x
    hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
    S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
    this : MeasurableSpace α := MeasurableSpace.generateFrom S
    a : α
    t : Set α
    hts : HasSubset.Subset t s
    hc : t.Countable
    htd : Dense t
    htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
    ⊢ MeasurableSet (Set.Iio a)
  -/
  by_cases ha : ∀ b < a, (Ioo b a).Nonempty
    /-
      case pos
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsBot x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
      this : MeasurableSpace α := MeasurableSpace.generateFrom S
      a : α
      t : Set α
      hts : HasSubset.Subset t s
      hc : t.Countable
      htd : Dense t
      htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
      ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
      ⊢ MeasurableSet (Set.Iio a)
    -/
  · convert_to MeasurableSet (⋃ (l ∈ t) (u ∈ t) (_ : l < u) (_ : u ≤ a), Ico l u)
      /-
        case h.e'_3
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
        ⊢ Eq (Set.Iio a) (Set.iUnion fun l => Set.iUnion fun h => Set.iUnion fun u =>  …
      -/
    · ext y
      /-
        case h.e'_3.h
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
        y : α
        ⊢ Iff (Membership.mem (Set.Iio a) y) (Membership.mem (Set.iUnion fun l => Set. …
      -/
      simp only [mem_iUnion, mem_Iio, mem_Ico]
      /-
        case h.e'_3.h
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
        y : α
        ⊢ Iff (LT.lt y a) (Exists fun i => Exists fun h => Exists fun i_1 => Exists fu …
      -/
      constructor
        /-
          case h.e'_3.h.mp
          α : Type u_5
          inst✝³ : TopologicalSpace α
          inst✝² : LinearOrder α
          inst✝¹ : OrderTopology α
          inst✝ : SecondCountableTopology α
          s : Set α
          hd : Dense s
          hbot : ∀ (x : α), IsBot x → Membership.mem s x
          hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
          S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
          this : MeasurableSpace α := MeasurableSpace.generateFrom S
          a : α
          t : Set α
          hts : HasSubset.Subset t s
          hc : t.Countable
          htd : Dense t
          htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
          ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
          y : α
          ⊢ LT.lt y a → Exists fun i => Exists fun h => Exists fun i_1 => Exists fun h = …
        -/
      · intro hy
        /-
          case h.e'_3.h.mp
          α : Type u_5
          inst✝³ : TopologicalSpace α
          inst✝² : LinearOrder α
          inst✝¹ : OrderTopology α
          inst✝ : SecondCountableTopology α
          s : Set α
          hd : Dense s
          hbot : ∀ (x : α), IsBot x → Membership.mem s x
          hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
          S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
          this : MeasurableSpace α := MeasurableSpace.generateFrom S
          a : α
          t : Set α
          hts : HasSubset.Subset t s
          hc : t.Countable
          htd : Dense t
          htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
          ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
          y : α
          hy : LT.lt y a
          ⊢ Exists fun i => Exists fun h => Exists fun i_1 => Exists fun h => Exists fun …
        -/
        rcases htd.exists_le' (fun b hb => htb _ hb (hbot b hb)) y with ⟨l, hlt, hly⟩
        /-
          case h.e'_3.h.mp.intro.intro
          α : Type u_5
          inst✝³ : TopologicalSpace α
          inst✝² : LinearOrder α
          inst✝¹ : OrderTopology α
          inst✝ : SecondCountableTopology α
          s : Set α
          hd : Dense s
          hbot : ∀ (x : α), IsBot x → Membership.mem s x
          hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
          S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
          this : MeasurableSpace α := MeasurableSpace.generateFrom S
          a : α
          t : Set α
          hts : HasSubset.Subset t s
          hc : t.Countable
          htd : Dense t
          htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
          ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
          y : α
          hy : LT.lt y a
          l : α
          hlt : Membership.mem t l
          hly : LE.le l y
          ⊢ Exists fun i => Exists fun h => Exists fun i_1 => Exists fun h => Exists fun …
        -/
        rcases htd.exists_mem_open isOpen_Ioo (ha y hy) with ⟨u, hut, hyu, hua⟩
        /-
          case h.e'_3.h.mp.intro.intro.intro.intro.intro
          α : Type u_5
          inst✝³ : TopologicalSpace α
          inst✝² : LinearOrder α
          inst✝¹ : OrderTopology α
          inst✝ : SecondCountableTopology α
          s : Set α
          hd : Dense s
          hbot : ∀ (x : α), IsBot x → Membership.mem s x
          hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
          S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
          this : MeasurableSpace α := MeasurableSpace.generateFrom S
          a : α
          t : Set α
          hts : HasSubset.Subset t s
          hc : t.Countable
          htd : Dense t
          htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
          ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
          y : α
          hy : LT.lt y a
          l : α
          hlt : Membership.mem t l
          hly : LE.le l y
          u : α
          hut : Membership.mem t u
          hyu : LT.lt y u
          hua : LT.lt u a
          ⊢ Exists fun i => Exists fun h => Exists fun i_1 => Exists fun h => Exists fun …
        -/
        exact ⟨l, hlt, u, hut, hly.trans_lt hyu, hua.le, hly, hyu⟩
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3.h.mpr
          α : Type u_5
          inst✝³ : TopologicalSpace α
          inst✝² : LinearOrder α
          inst✝¹ : OrderTopology α
          inst✝ : SecondCountableTopology α
          s : Set α
          hd : Dense s
          hbot : ∀ (x : α), IsBot x → Membership.mem s x
          hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
          S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
          this : MeasurableSpace α := MeasurableSpace.generateFrom S
          a : α
          t : Set α
          hts : HasSubset.Subset t s
          hc : t.Countable
          htd : Dense t
          htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
          ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
          y : α
          ⊢ (Exists fun i => Exists fun h => Exists fun i_1 => Exists fun h => Exists fu …
        -/
      · rintro ⟨l, -, u, -, -, hua, -, hyu⟩
        /-
          case h.e'_3.h.mpr.intro.intro.intro.intro.intro.intro.intro
          α : Type u_5
          inst✝³ : TopologicalSpace α
          inst✝² : LinearOrder α
          inst✝¹ : OrderTopology α
          inst✝ : SecondCountableTopology α
          s : Set α
          hd : Dense s
          hbot : ∀ (x : α), IsBot x → Membership.mem s x
          hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
          S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
          this : MeasurableSpace α := MeasurableSpace.generateFrom S
          a : α
          t : Set α
          hts : HasSubset.Subset t s
          hc : t.Countable
          htd : Dense t
          htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
          ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
          y l u : α
          hua : LE.le u a
          hyu : LT.lt y u
          ⊢ LT.lt y a
        -/
        exact hyu.trans_le hua
        /-
          🎉 no goals
        -/
      /-
        case pos
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : ∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty
        ⊢ MeasurableSet (Set.iUnion fun l => Set.iUnion fun h => Set.iUnion fun u => S …
      -/
    · refine MeasurableSet.biUnion hc fun a ha => MeasurableSet.biUnion hc fun b hb => ?_
      /-
        case pos
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a✝ : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha✝ : ∀ (b : α), LT.lt b a✝ → (Set.Ioo b a✝).Nonempty
        a : α
        ha : Membership.mem t a
        b : α
        hb : Membership.mem t b
        ⊢ MeasurableSet (Set.iUnion fun x => Set.iUnion fun x => Set.Ico a b)
      -/
      refine MeasurableSet.iUnion fun hab => MeasurableSet.iUnion fun _ => ?_
      /-
        case pos
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a✝ : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha✝ : ∀ (b : α), LT.lt b a✝ → (Set.Ioo b a✝).Nonempty
        a : α
        ha : Membership.mem t a
        b : α
        hb : Membership.mem t b
        hab : LT.lt a b
        x✝ : LE.le b a✝
        ⊢ MeasurableSet (Set.Ico a b)
      -/
      exact .basic _ ⟨a, hts ha, b, hts hb, hab, mem_singleton _⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsBot x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
      this : MeasurableSpace α := MeasurableSpace.generateFrom S
      a : α
      t : Set α
      hts : HasSubset.Subset t s
      hc : t.Countable
      htd : Dense t
      htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
      ha : Not (∀ (b : α), LT.lt b a → (Set.Ioo b a).Nonempty)
      ⊢ MeasurableSet (Set.Iio a)
    -/
  · simp only [not_forall, not_nonempty_iff_eq_empty] at ha
    /-
      case neg
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsBot x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
      this : MeasurableSpace α := MeasurableSpace.generateFrom S
      a : α
      t : Set α
      hts : HasSubset.Subset t s
      hc : t.Countable
      htd : Dense t
      htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
      ha : Exists fun x => Exists fun h => Eq (Set.Ioo x a) EmptyCollection.emptyCol …
      ⊢ MeasurableSet (Set.Iio a)
    -/
    replace ha : a ∈ s := hIoo ha.choose a ha.choose_spec.fst ha.choose_spec.snd
    /-
      case neg
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsBot x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
      this : MeasurableSpace α := MeasurableSpace.generateFrom S
      a : α
      t : Set α
      hts : HasSubset.Subset t s
      hc : t.Countable
      htd : Dense t
      htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
      ha : Membership.mem s a
      ⊢ MeasurableSet (Set.Iio a)
    -/
    convert_to MeasurableSet (⋃ (l ∈ t) (_ : l < a), Ico l a)
      /-
        case h.e'_3
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : Membership.mem s a
        ⊢ Eq (Set.Iio a) (Set.iUnion fun l => Set.iUnion fun h => Set.iUnion fun x =>  …
      -/
    · symm
      simp only [← Ici_inter_Iio, ← iUnion_inter, inter_eq_right, subset_def, mem_iUnion,
        mem_Ici, mem_Iio]
      /-
        case h.e'_3
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : Membership.mem s a
        ⊢ ∀ (x : α), LT.lt x a → Exists fun i => Exists fun h => Exists fun h => LE.le …
      -/
      intro x hx
      /-
        case h.e'_3
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : Membership.mem s a
        x : α
        hx : LT.lt x a
        ⊢ Exists fun i => Exists fun h => Exists fun h => LE.le i x
      -/
      rcases htd.exists_le' (fun b hb => htb _ hb (hbot b hb)) x with ⟨z, hzt, hzx⟩
      /-
        case h.e'_3.intro.intro
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : Membership.mem s a
        x : α
        hx : LT.lt x a
        z : α
        hzt : Membership.mem t z
        hzx : LE.le z x
        ⊢ Exists fun i => Exists fun h => Exists fun h => LE.le i x
      -/
      exact ⟨z, hzt, hzx.trans_lt hx, hzx⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : Membership.mem s a
        ⊢ MeasurableSet (Set.iUnion fun l => Set.iUnion fun h => Set.iUnion fun x => S …
      -/
    · refine .biUnion hc fun x hx => MeasurableSet.iUnion fun hlt => ?_
      /-
        case neg
        α : Type u_5
        inst✝³ : TopologicalSpace α
        inst✝² : LinearOrder α
        inst✝¹ : OrderTopology α
        inst✝ : SecondCountableTopology α
        s : Set α
        hd : Dense s
        hbot : ∀ (x : α), IsBot x → Membership.mem s x
        hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
        S : Set (Set α) := setOf fun S => Exists fun l => And (Membership.mem s l) (Ex …
        this : MeasurableSpace α := MeasurableSpace.generateFrom S
        a : α
        t : Set α
        hts : HasSubset.Subset t s
        hc : t.Countable
        htd : Dense t
        htb : ∀ (x : α), IsBot x → Membership.mem s x → Membership.mem t x
        ha : Membership.mem s a
        x : α
        hx : Membership.mem t x
        hlt : LT.lt x a
        ⊢ MeasurableSet (Set.Ico x a)
      -/
      exact .basic _ ⟨x, hts hx, a, ha, hlt, mem_singleton _⟩
      /-
        🎉 no goals
      -/


theorem Dense.borel_eq_generateFrom_Ico_mem {α : Type*} [TopologicalSpace α] [LinearOrder α]
    [OrderTopology α] [SecondCountableTopology α] [DenselyOrdered α] [NoMinOrder α] {s : Set α}
    (hd : Dense s) :
    borel α = .generateFrom { S : Set α | ∃ l ∈ s, ∃ u ∈ s, l < u ∧ Ico l u = S } :=
                                           /-
                                             α : Type u_5
                                             inst✝⁵ : TopologicalSpace α
                                             inst✝⁴ : LinearOrder α
                                             inst✝³ : OrderTopology α
                                             inst✝² : SecondCountableTopology α
                                             inst✝¹ : DenselyOrdered α
                                             inst✝ : NoMinOrder α
                                             s : Set α
                                             hd : Dense s
                                             ⊢ ∀ (x : α), IsBot x → Membership.mem s x
                                           -/
  hd.borel_eq_generateFrom_Ico_mem_aux (by simp) fun _ _ hxy H =>
                                           /-
                                             🎉 no goals
                                           -/
    ((nonempty_Ioo.2 hxy).ne_empty H).elim


theorem borel_eq_generateFrom_Ico (α : Type*) [TopologicalSpace α] [SecondCountableTopology α]
    [LinearOrder α] [OrderTopology α] :
    borel α = .generateFrom { S : Set α | ∃ (l u : α), l < u ∧ Ico l u = S } := by
  simpa only [exists_prop, mem_univ, true_and] using
    (@dense_univ α _).borel_eq_generateFrom_Ico_mem_aux (fun _ _ => mem_univ _) fun _ _ _ _ =>
      mem_univ _


theorem Dense.borel_eq_generateFrom_Ioc_mem_aux {α : Type*} [TopologicalSpace α] [LinearOrder α]
    [OrderTopology α] [SecondCountableTopology α] {s : Set α} (hd : Dense s)
    (hbot : ∀ x, IsTop x → x ∈ s) (hIoo : ∀ x y : α, x < y → Ioo x y = ∅ → x ∈ s) :
    borel α = .generateFrom { S : Set α | ∃ l ∈ s, ∃ u ∈ s, l < u ∧ Ioc l u = S } := by
  convert hd.orderDual.borel_eq_generateFrom_Ico_mem_aux hbot fun x y hlt he => hIoo y x hlt _
    using 2
    /-
      case h.e'_3.h.h.e'_2.h
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsTop x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      e_1✝¹ : Eq (MeasurableSpace α) (MeasurableSpace (OrderDual α))
      e_1✝ : Eq α (OrderDual α)
      ⊢ Eq (setOf fun S => Exists fun l => And (Membership.mem s l) (Exists fun u => …
    -/
  · ext s
    /-
      case h.e'_3.h.h.e'_2.h.h
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s✝ : Set α
      hd : Dense s✝
      hbot : ∀ (x : α), IsTop x → Membership.mem s✝ x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      e_1✝¹ : Eq (MeasurableSpace α) (MeasurableSpace (OrderDual α))
      e_1✝ : Eq α (OrderDual α)
      s : Set α
      ⊢ Iff (Membership.mem (setOf fun S => Exists fun l => And (Membership.mem s✝ l …
    -/
    constructor <;> rintro ⟨l, hl, u, hu, hlt, rfl⟩
    /-
      case h.e'_3.h.h.e'_2.h.h.mp.intro.intro.intro.intro.intro
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsTop x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      e_1✝¹ : Eq (MeasurableSpace α) (MeasurableSpace (OrderDual α))
      e_1✝ : Eq α (OrderDual α)
      l : α
      hl : Membership.mem s l
      u : α
      hu : Membership.mem s u
      hlt : LT.lt l u
      ⊢ Membership.mem (setOf fun S => Exists fun l => And (Membership.mem (Set.prei …
    -/
    exacts [⟨u, hu, l, hl, hlt, dual_Ico⟩, ⟨u, hu, l, hl, hlt, dual_Ioc⟩]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsTop x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      x y : OrderDual α
      hlt : LT.lt x y
      he : Eq (Set.Ioo x y) EmptyCollection.emptyCollection
      ⊢ Eq (Set.Ioo y x) EmptyCollection.emptyCollection
    -/
  · erw [dual_Ioo]
    /-
      α : Type u_5
      inst✝³ : TopologicalSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      hd : Dense s
      hbot : ∀ (x : α), IsTop x → Membership.mem s x
      hIoo : ∀ (x y : α), LT.lt x y → Eq (Set.Ioo x y) EmptyCollection.emptyCollecti …
      x y : OrderDual α
      hlt : LT.lt x y
      he : Eq (Set.Ioo x y) EmptyCollection.emptyCollection
      ⊢ Eq (Set.preimage (⇑OrderDual.ofDual) (Set.Ioo x y)) EmptyCollection.emptyCol …
    -/
    exact he
    /-
      🎉 no goals
    -/


theorem Dense.borel_eq_generateFrom_Ioc_mem {α : Type*} [TopologicalSpace α] [LinearOrder α]
    [OrderTopology α] [SecondCountableTopology α] [DenselyOrdered α] [NoMaxOrder α] {s : Set α}
    (hd : Dense s) :
    borel α = .generateFrom { S : Set α | ∃ l ∈ s, ∃ u ∈ s, l < u ∧ Ioc l u = S } :=
                                           /-
                                             α : Type u_5
                                             inst✝⁵ : TopologicalSpace α
                                             inst✝⁴ : LinearOrder α
                                             inst✝³ : OrderTopology α
                                             inst✝² : SecondCountableTopology α
                                             inst✝¹ : DenselyOrdered α
                                             inst✝ : NoMaxOrder α
                                             s : Set α
                                             hd : Dense s
                                             ⊢ ∀ (x : α), IsTop x → Membership.mem s x
                                           -/
  hd.borel_eq_generateFrom_Ioc_mem_aux (by simp) fun _ _ hxy H =>
                                           /-
                                             🎉 no goals
                                           -/
    ((nonempty_Ioo.2 hxy).ne_empty H).elim


theorem borel_eq_generateFrom_Ioc (α : Type*) [TopologicalSpace α] [SecondCountableTopology α]
    [LinearOrder α] [OrderTopology α] :
    borel α = .generateFrom { S : Set α | ∃ l u, l < u ∧ Ioc l u = S } := by
  simpa only [exists_prop, mem_univ, true_and] using
    (@dense_univ α _).borel_eq_generateFrom_Ioc_mem_aux (fun _ _ => mem_univ _) fun _ _ _ _ =>
      mem_univ _


/-- Two finite measures on a Borel space are equal if they agree on all closed-open intervals.  If
`α` is a conditionally complete linear order with no top element,
`MeasureTheory.Measure.ext_of_Ico` is an extensionality lemma with weaker assumptions on `μ` and
`ν`. -/
theorem ext_of_Ico_finite {α : Type*} [TopologicalSpace α] {m : MeasurableSpace α}
    [SecondCountableTopology α] [LinearOrder α] [OrderTopology α] [BorelSpace α] (μ ν : Measure α)
    [IsFiniteMeasure μ] (hμν : μ univ = ν univ) (h : ∀ ⦃a b⦄, a < b → μ (Ico a b) = ν (Ico a b)) :
    μ = ν := by
  refine
    ext_of_generate_finite _ (BorelSpace.measurable_eq.trans (borel_eq_generateFrom_Ico α))
      (isPiSystem_Ico (id : α → α) id) ?_ hμν
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμν : Eq (μ Set.univ) (ν Set.univ)
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
    ⊢ ∀ (s : Set α), Membership.mem (setOf fun S => Exists fun l => Exists fun u = …
  -/
  rintro - ⟨a, b, hlt, rfl⟩
  /-
    case intro.intro.intro
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμν : Eq (μ Set.univ) (ν Set.univ)
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
    a b : α
    hlt : LT.lt a b
    ⊢ Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
  -/
  exact h hlt
  /-
    🎉 no goals
  -/


/-- Two finite measures on a Borel space are equal if they agree on all open-closed intervals.  If
`α` is a conditionally complete linear order with no top element,
`MeasureTheory.Measure.ext_of_Ioc` is an extensionality lemma with weaker assumptions on `μ` and
`ν`. -/
theorem ext_of_Ioc_finite {α : Type*} [TopologicalSpace α] {m : MeasurableSpace α}
    [SecondCountableTopology α] [LinearOrder α] [OrderTopology α] [BorelSpace α] (μ ν : Measure α)
    [IsFiniteMeasure μ] (hμν : μ univ = ν univ) (h : ∀ ⦃a b⦄, a < b → μ (Ioc a b) = ν (Ioc a b)) :
    μ = ν := by
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμν : Eq (μ Set.univ) (ν Set.univ)
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ioc a b)) (ν (Set.Ioc a b))
    ⊢ Eq μ ν
  -/
  refine @ext_of_Ico_finite αᵒᵈ _ _ _ _ _ ‹_› μ ν _ hμν fun a b hab => ?_
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμν : Eq (μ Set.univ) (ν Set.univ)
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ioc a b)) (ν (Set.Ioc a b))
    a b : OrderDual α
    hab : LT.lt a b
    ⊢ Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
  -/
  erw [dual_Ico (α := α)]
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hμν : Eq (μ Set.univ) (ν Set.univ)
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ioc a b)) (ν (Set.Ioc a b))
    a b : OrderDual α
    hab : LT.lt a b
    ⊢ Eq (μ (Set.preimage (⇑OrderDual.ofDual) (Set.Ioc b a))) (ν (Set.preimage (⇑O …
  -/
  exact h hab
  /-
    🎉 no goals
  -/


/-- Two measures which are finite on closed-open intervals are equal if they agree on all
closed-open intervals. -/
theorem ext_of_Ico' {α : Type*} [TopologicalSpace α] {m : MeasurableSpace α}
    [SecondCountableTopology α] [LinearOrder α] [OrderTopology α] [BorelSpace α] [NoMaxOrder α]
    (μ ν : Measure α) (hμ : ∀ ⦃a b⦄, a < b → μ (Ico a b) ≠ ∞)
    (h : ∀ ⦃a b⦄, a < b → μ (Ico a b) = ν (Ico a b)) : μ = ν := by
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    inst✝ : NoMaxOrder α
    μ ν : MeasureTheory.Measure α
    hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
    ⊢ Eq μ ν
  -/
  rcases exists_countable_dense_bot_top α with ⟨s, hsc, hsd, hsb, _⟩
  have : (⋃ (l ∈ s) (u ∈ s) (_ : l < u), {Ico l u} : Set (Set α)).Countable :=
    hsc.biUnion fun l _ => hsc.biUnion fun u _ => countable_iUnion fun _ => countable_singleton _
  /-
    case intro.intro.intro.intro
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    inst✝ : NoMaxOrder α
    μ ν : MeasureTheory.Measure α
    hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
    s : Set α
    hsc : s.Countable
    hsd : Dense s
    hsb : ∀ (x : α), IsBot x → Membership.mem s x
    right✝ : ∀ (x : α), IsTop x → Membership.mem s x
    this : (Set.iUnion fun l => Set.iUnion fun h => Set.iUnion fun u => Set.iUnion …
    ⊢ Eq μ ν
  -/
  simp only [← setOf_eq_eq_singleton, ← setOf_exists] at this
  refine
    Measure.ext_of_generateFrom_of_cover_subset
      (BorelSpace.measurable_eq.trans (borel_eq_generateFrom_Ico α)) (isPiSystem_Ico id id) ?_ this
      ?_ ?_ ?_
    /-
      case intro.intro.intro.intro.refine_1
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      ⊢ HasSubset.Subset (setOf fun x => Exists fun i => Exists fun i_1 => Exists fu …
    -/
  · rintro _ ⟨l, -, u, -, h, rfl⟩
    /-
      case intro.intro.intro.intro.refine_1.intro.intro.intro.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h✝ : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      l u : α
      h : LT.lt l u
      ⊢ Membership.mem (setOf fun S => Exists fun l => Exists fun u => And (LT.lt l  …
    -/
    exact ⟨l, u, h, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      ⊢ Eq (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exist …
    -/
  · refine sUnion_eq_univ_iff.2 fun x => ?_
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      x : α
      ⊢ Exists fun b => And (Membership.mem (setOf fun x => Exists fun i => Exists f …
    -/
    rcases hsd.exists_le' hsb x with ⟨l, hls, hlx⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      x l : α
      hls : Membership.mem s l
      hlx : LE.le l x
      ⊢ Exists fun b => And (Membership.mem (setOf fun x => Exists fun i => Exists f …
    -/
    rcases hsd.exists_gt x with ⟨u, hus, hxu⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      x l : α
      hls : Membership.mem s l
      hlx : LE.le l x
      u : α
      hus : Membership.mem s u
      hxu : LT.lt x u
      ⊢ Exists fun b => And (Membership.mem (setOf fun x => Exists fun i => Exists f …
    -/
    exact ⟨_, ⟨l, hls, u, hus, hlx.trans_lt hxu, rfl⟩, hlx, hxu⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_3
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      ⊢ ∀ (s_1 : Set α), Membership.mem (setOf fun x => Exists fun i => Exists fun i …
    -/
  · rintro _ ⟨l, -, u, -, hlt, rfl⟩
    /-
      case intro.intro.intro.intro.refine_3.intro.intro.intro.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      l u : α
      hlt : LT.lt l u
      ⊢ Ne (μ (Set.Ico l u)) Top.top
    -/
    exact hμ hlt
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_4
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      ⊢ ∀ (s : Set α), Membership.mem (setOf fun S => Exists fun l => Exists fun u = …
    -/
  · rintro _ ⟨l, u, hlt, rfl⟩
    /-
      case intro.intro.intro.intro.refine_4.intro.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      inst✝ : NoMaxOrder α
      μ ν : MeasureTheory.Measure α
      hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ico a b)) Top.top
      h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ico a b)) (ν (Set.Ico a b))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hsb : ∀ (x : α), IsBot x → Membership.mem s x
      right✝ : ∀ (x : α), IsTop x → Membership.mem s x
      this : (setOf fun x => Exists fun i => Exists fun i_1 => Exists fun i_2 => Exi …
      l u : α
      hlt : LT.lt l u
      ⊢ Eq (μ (Set.Ico l u)) (ν (Set.Ico l u))
    -/
    exact h hlt
    /-
      🎉 no goals
    -/


/-- Two measures which are finite on closed-open intervals are equal if they agree on all
open-closed intervals. -/
theorem ext_of_Ioc' {α : Type*} [TopologicalSpace α] {m : MeasurableSpace α}
    [SecondCountableTopology α] [LinearOrder α] [OrderTopology α] [BorelSpace α] [NoMinOrder α]
    (μ ν : Measure α) (hμ : ∀ ⦃a b⦄, a < b → μ (Ioc a b) ≠ ∞)
    (h : ∀ ⦃a b⦄, a < b → μ (Ioc a b) = ν (Ioc a b)) : μ = ν := by
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    inst✝ : NoMinOrder α
    μ ν : MeasureTheory.Measure α
    hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ioc a b)) Top.top
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ioc a b)) (ν (Set.Ioc a b))
    ⊢ Eq μ ν
  -/
  refine @ext_of_Ico' αᵒᵈ _ _ _ _ _ ‹_› _ μ ν ?_ ?_ <;> intro a b hab <;> erw [dual_Ico (α := α)]
  /-
    case refine_1
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    inst✝ : NoMinOrder α
    μ ν : MeasureTheory.Measure α
    hμ : ∀ ⦃a b : α⦄, LT.lt a b → Ne (μ (Set.Ioc a b)) Top.top
    h : ∀ ⦃a b : α⦄, LT.lt a b → Eq (μ (Set.Ioc a b)) (ν (Set.Ioc a b))
    a b : OrderDual α
    hab : LT.lt a b
    ⊢ Ne (μ (Set.preimage (⇑OrderDual.ofDual) (Set.Ioc b a))) Top.top
  -/
  exacts [hμ hab, h hab]
  /-
    🎉 no goals
  -/


/-- Two measures which are finite on closed-open intervals are equal if they agree on all
closed-open intervals. -/
theorem ext_of_Ico {α : Type*} [TopologicalSpace α] {_m : MeasurableSpace α}
    [SecondCountableTopology α] [ConditionallyCompleteLinearOrder α] [OrderTopology α]
    [BorelSpace α] [NoMaxOrder α] (μ ν : Measure α) [IsLocallyFiniteMeasure μ]
    (h : ∀ ⦃a b⦄, a < b → μ (Ico a b) = ν (Ico a b)) : μ = ν :=
  μ.ext_of_Ico' ν (fun _ _ _ => measure_Ico_lt_top.ne) h


/-- Two measures which are finite on closed-open intervals are equal if they agree on all
open-closed intervals. -/
theorem ext_of_Ioc {α : Type*} [TopologicalSpace α] {_m : MeasurableSpace α}
    [SecondCountableTopology α] [ConditionallyCompleteLinearOrder α] [OrderTopology α]
    [BorelSpace α] [NoMinOrder α] (μ ν : Measure α) [IsLocallyFiniteMeasure μ]
    (h : ∀ ⦃a b⦄, a < b → μ (Ioc a b) = ν (Ioc a b)) : μ = ν :=
  μ.ext_of_Ioc' ν (fun _ _ _ => measure_Ioc_lt_top.ne) h


/-- Two finite measures on a Borel space are equal if they agree on all left-infinite right-closed
intervals. -/
theorem ext_of_Iic {α : Type*} [TopologicalSpace α] {m : MeasurableSpace α}
    [SecondCountableTopology α] [LinearOrder α] [OrderTopology α] [BorelSpace α] (μ ν : Measure α)
    [IsFiniteMeasure μ] (h : ∀ a, μ (Iic a) = ν (Iic a)) : μ = ν := by
  /-
    α : Type u_5
    inst✝⁵ : TopologicalSpace α
    m : MeasurableSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : BorelSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
    ⊢ Eq μ ν
  -/
  refine ext_of_Ioc_finite μ ν ?_ fun a b hlt => ?_
    /-
      case refine_1
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
      ⊢ Eq (μ Set.univ) (ν Set.univ)
    -/
  · rcases exists_countable_dense_bot_top α with ⟨s, hsc, hsd, -, hst⟩
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hst : ∀ (x : α), IsTop x → Membership.mem s x
      ⊢ Eq (μ Set.univ) (ν Set.univ)
    -/
    have : DirectedOn (· ≤ ·) s := directedOn_iff_directed.2 (Subtype.mono_coe _).directed_le
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
      s : Set α
      hsc : s.Countable
      hsd : Dense s
      hst : ∀ (x : α), IsTop x → Membership.mem s x
      this : DirectedOn (fun x1 x2 => LE.le x1 x2) s
      ⊢ Eq (μ Set.univ) (ν Set.univ)
    -/
    simp only [← biSup_measure_Iic hsc (hsd.exists_ge' hst) this, h]
    /-
      🎉 no goals
    -/
  rw [← Iic_diff_Iic, measure_diff (Iic_subset_Iic.2 hlt.le) nullMeasurableSet_Iic,
    measure_diff (Iic_subset_Iic.2 hlt.le) nullMeasurableSet_Iic, h a, h b]
    /-
      case refine_2
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
      a b : α
      hlt : LT.lt a b
      ⊢ Ne (ν (Set.Iic a)) Top.top
    -/
  · rw [← h a]
    /-
      case refine_2
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
      a b : α
      hlt : LT.lt a b
      ⊢ Ne (μ (Set.Iic a)) Top.top
    -/
    exact measure_ne_top μ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_5
      inst✝⁵ : TopologicalSpace α
      m : MeasurableSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : BorelSpace α
      μ ν : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      h : ∀ (a : α), Eq (μ (Set.Iic a)) (ν (Set.Iic a))
      a b : α
      hlt : LT.lt a b
      ⊢ Ne (μ (Set.Iic a)) Top.top
    -/
  · exact measure_ne_top μ _
    /-
      🎉 no goals
    -/


/-- Two finite measures on a Borel space are equal if they agree on all left-closed right-infinite
intervals. -/
theorem ext_of_Ici {α : Type*} [TopologicalSpace α] {_ : MeasurableSpace α}
    [SecondCountableTopology α] [LinearOrder α] [OrderTopology α] [BorelSpace α] (μ ν : Measure α)
    [IsFiniteMeasure μ] (h : ∀ a, μ (Ici a) = ν (Ici a)) : μ = ν :=
  @ext_of_Iic αᵒᵈ _ _ _ _ _ ‹_› _ _ _ h


@[measurability]
theorem measurableSet_uIcc : MeasurableSet (uIcc a b) :=
  measurableSet_Icc


@[measurability]
theorem measurableSet_uIoc : MeasurableSet (uIoc a b) :=
  measurableSet_Ioc


@[measurability]
theorem Measurable.max {f g : δ → α} (hf : Measurable f) (hg : Measurable g) :
    Measurable fun a => max (f a) (g a) := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    inst✝ : SecondCountableTopology α
    f g : δ → α
    hf : Measurable f
    hg : Measurable g
    ⊢ Measurable fun a => Max.max (f a) (g a)
  -/
  simpa only [max_def'] using hf.piecewise (measurableSet_le hg hf) hg
  /-
    🎉 no goals
  -/


@[measurability]
nonrec theorem AEMeasurable.max {f g : δ → α} {μ : Measure δ} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) : AEMeasurable (fun a => max (f a) (g a)) μ :=
  ⟨fun a => max (hf.mk f a) (hg.mk g a), hf.measurable_mk.max hg.measurable_mk,
    EventuallyEq.comp₂ hf.ae_eq_mk _ hg.ae_eq_mk⟩


@[measurability]
theorem Measurable.min {f g : δ → α} (hf : Measurable f) (hg : Measurable g) :
    Measurable fun a => min (f a) (g a) := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderClosedTopology α
    inst✝ : SecondCountableTopology α
    f g : δ → α
    hf : Measurable f
    hg : Measurable g
    ⊢ Measurable fun a => Min.min (f a) (g a)
  -/
  simpa only [min_def] using hf.piecewise (measurableSet_le hf hg) hg
  /-
    🎉 no goals
  -/


@[measurability]
nonrec theorem AEMeasurable.min {f g : δ → α} {μ : Measure δ} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) : AEMeasurable (fun a => min (f a) (g a)) μ :=
  ⟨fun a => min (hf.mk f a) (hg.mk g a), hf.measurable_mk.min hg.measurable_mk,
    EventuallyEq.comp₂ hf.ae_eq_mk _ hg.ae_eq_mk⟩


instance (priority := 100) ContinuousSup.measurableSup [Max γ] [ContinuousSup γ] :
    MeasurableSup γ where
  measurable_const_sup _ := (continuous_const.sup continuous_id).measurable
  measurable_sup_const _ := (continuous_id.sup continuous_const).measurable


instance (priority := 100) ContinuousSup.measurableSup₂ [SecondCountableTopology γ] [Max γ]
    [ContinuousSup γ] : MeasurableSup₂ γ :=
  ⟨continuous_sup.measurable⟩


instance (priority := 100) ContinuousInf.measurableInf [Min γ] [ContinuousInf γ] :
    MeasurableInf γ where
  measurable_const_inf _ := (continuous_const.inf continuous_id).measurable
  measurable_inf_const _ := (continuous_id.inf continuous_const).measurable


instance (priority := 100) ContinuousInf.measurableInf₂ [SecondCountableTopology γ] [Min γ]
    [ContinuousInf γ] : MeasurableInf₂ γ :=
  ⟨continuous_inf.measurable⟩


theorem measurable_of_Iio {f : δ → α} (hf : ∀ x, MeasurableSet (f ⁻¹' Iio x)) : Measurable f := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iio x))
    ⊢ Measurable f
  -/
  convert measurable_generateFrom (α := δ) _
    /-
      case h.e'_4
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      f : δ → α
      hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iio x))
      ⊢ Eq mα (MeasurableSpace.generateFrom ?convert_2)
    -/
  · exact BorelSpace.measurable_eq.trans (borel_eq_generateFrom_Iio _)
    /-
      🎉 no goals
    -/
    /-
      case convert_4
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      f : δ → α
      hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iio x))
      ⊢ ∀ (t : Set α), Membership.mem (Set.range Set.Iio) t → MeasurableSet (Set.pre …
    -/
  · rintro _ ⟨x, rfl⟩; exact hf x
                       /-
                         🎉 no goals
                       -/


theorem UpperSemicontinuous.measurable [TopologicalSpace δ] [OpensMeasurableSpace δ] {f : δ → α}
    (hf : UpperSemicontinuous f) : Measurable f :=
  measurable_of_Iio fun y => (hf.isOpen_preimage y).measurableSet


theorem measurable_of_Ioi {f : δ → α} (hf : ∀ x, MeasurableSet (f ⁻¹' Ioi x)) : Measurable f := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ioi x))
    ⊢ Measurable f
  -/
  convert measurable_generateFrom (α := δ) _
    /-
      case h.e'_4
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      f : δ → α
      hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ioi x))
      ⊢ Eq mα (MeasurableSpace.generateFrom ?convert_2)
    -/
  · exact BorelSpace.measurable_eq.trans (borel_eq_generateFrom_Ioi _)
    /-
      🎉 no goals
    -/
    /-
      case convert_4
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      f : δ → α
      hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ioi x))
      ⊢ ∀ (t : Set α), Membership.mem (Set.range Set.Ioi) t → MeasurableSet (Set.pre …
    -/
  · rintro _ ⟨x, rfl⟩; exact hf x
                       /-
                         🎉 no goals
                       -/


theorem LowerSemicontinuous.measurable [TopologicalSpace δ] [OpensMeasurableSpace δ] {f : δ → α}
    (hf : LowerSemicontinuous f) : Measurable f :=
  measurable_of_Ioi fun y => (hf.isOpen_preimage y).measurableSet


theorem measurable_of_Iic {f : δ → α} (hf : ∀ x, MeasurableSet (f ⁻¹' Iic x)) : Measurable f := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iic x))
    ⊢ Measurable f
  -/
  apply measurable_of_Ioi
  /-
    case hf
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iic x))
    ⊢ ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ioi x))
  -/
  simp_rw [← compl_Iic, preimage_compl, MeasurableSet.compl_iff]
  /-
    case hf
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iic x))
    ⊢ ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iic x))
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem measurable_of_Ici {f : δ → α} (hf : ∀ x, MeasurableSet (f ⁻¹' Ici x)) : Measurable f := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ici x))
    ⊢ Measurable f
  -/
  apply measurable_of_Iio
  /-
    case hf
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ici x))
    ⊢ ∀ (x : α), MeasurableSet (Set.preimage f (Set.Iio x))
  -/
  simp_rw [← compl_Ici, preimage_compl, MeasurableSet.compl_iff]
  /-
    case hf
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    f : δ → α
    hf : ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ici x))
    ⊢ ∀ (x : α), MeasurableSet (Set.preimage f (Set.Ici x))
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- If a function is the least upper bound of countably many measurable functions,
then it is measurable. -/
theorem Measurable.isLUB {ι} [Countable ι] {f : ι → δ → α} {g : δ → α} (hf : ∀ i, Measurable (f i))
    (hg : ∀ b, IsLUB { a | ∃ i, f i b = a } (g b)) : Measurable g := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    g : δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hg : ∀ (b : δ), IsLUB (_root_.setOf fun a => Exists fun i => Eq (f i b) a) (g b)
    ⊢ Measurable g
  -/
  change ∀ b, IsLUB (range fun i => f i b) (g b) at hg
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    g : δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hg : ∀ (b : δ), IsLUB (Set.range fun i => f i b) (g b)
    ⊢ Measurable g
  -/
  rw [‹BorelSpace α›.measurable_eq, borel_eq_generateFrom_Ioi α]
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    g : δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hg : ∀ (b : δ), IsLUB (Set.range fun i => f i b) (g b)
    ⊢ Measurable g
  -/
  apply measurable_generateFrom
  /-
    case h
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    g : δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hg : ∀ (b : δ), IsLUB (Set.range fun i => f i b) (g b)
    ⊢ ∀ (t : Set α), Membership.mem (Set.range Set.Ioi) t → MeasurableSet (Set.pre …
  -/
  rintro _ ⟨a, rfl⟩
  /-
    case h.intro
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    g : δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hg : ∀ (b : δ), IsLUB (Set.range fun i => f i b) (g b)
    a : α
    ⊢ MeasurableSet (Set.preimage g (Set.Ioi a))
  -/
  simp_rw [Set.preimage, mem_Ioi, lt_isLUB_iff (hg _), exists_range_iff, setOf_exists]
  /-
    case h.intro
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    g : δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hg : ∀ (b : δ), IsLUB (Set.range fun i => f i b) (g b)
    a : α
    ⊢ MeasurableSet (Set.iUnion fun i => _root_.setOf fun x => LT.lt a (f i x))
  -/
  exact MeasurableSet.iUnion fun i => hf i (isOpen_lt' _).measurableSet
  /-
    🎉 no goals
  -/


/-- If a function is the least upper bound of countably many measurable functions on a measurable
set `s`, and coincides with a measurable function outside of `s`, then it is measurable. -/
theorem Measurable.isLUB_of_mem {ι} [Countable ι] {f : ι → δ → α} {g g' : δ → α}
    (hf : ∀ i, Measurable (f i))
    {s : Set δ} (hs : MeasurableSet s) (hg : ∀ b ∈ s, IsLUB { a | ∃ i, f i b = a } (g b))
    (hg' : EqOn g g' sᶜ) (g'_meas : Measurable g') : Measurable g := by
  classical
  rcases isEmpty_or_nonempty ι with hι|⟨⟨i⟩⟩
  · rcases eq_empty_or_nonempty s with rfl|⟨x, hx⟩
    · convert g'_meas
      rwa [compl_empty, eqOn_univ] at hg'
    · have A : ∀ b ∈ s, IsBot (g b) := by simpa using hg
      have B : ∀ b ∈ s, g b = g x := by
        intro b hb
        apply le_antisymm (A b hb (g x)) (A x hx (g b))
      have : g = s.piecewise (fun _y ↦ g x) g' := by
        ext b
        by_cases hb : b ∈ s
        · simp [hb, B]
        · simp [hb, hg' hb]
      rw [this]
      exact Measurable.piecewise hs measurable_const g'_meas
  · have : Nonempty ι := ⟨i⟩
    let f' : ι → δ → α := fun i ↦ s.piecewise (f i) g'
    suffices ∀ b, IsLUB { a | ∃ i, f' i b = a } (g b) from
      Measurable.isLUB (fun i ↦ Measurable.piecewise hs (hf i) g'_meas) this
    intro b
    by_cases hb : b ∈ s
    · have A : ∀ i, f' i b = f i b := fun i ↦ by simp [f', hb]
      simpa [A] using hg b hb
    · have A : ∀ i, f' i b = g' b := fun i ↦ by simp [f', hb]
      simp [A, hg' hb, isLUB_singleton]


theorem AEMeasurable.isLUB {ι} {μ : Measure δ} [Countable ι] {f : ι → δ → α} {g : δ → α}
    (hf : ∀ i, AEMeasurable (f i) μ) (hg : ∀ᵐ b ∂μ, IsLUB { a | ∃ i, f i b = a } (g b)) :
    AEMeasurable g μ := by
  classical
  nontriviality α
  haveI hα : Nonempty α := inferInstance
  cases' isEmpty_or_nonempty ι with hι hι
  · simp only [IsEmpty.exists_iff, setOf_false, isLUB_empty_iff] at hg
    exact aemeasurable_const' (hg.mono fun a ha => hg.mono fun b hb => (ha _).antisymm (hb _))
  let p : δ → (ι → α) → Prop := fun x f' => IsLUB { a | ∃ i, f' i = a } (g x)
  let g_seq := (aeSeqSet hf p).piecewise g fun _ => hα.some
  have hg_seq : ∀ b, IsLUB { a | ∃ i, aeSeq hf p i b = a } (g_seq b) := by
    intro b
    simp only [g_seq, aeSeq, Set.piecewise]
    split_ifs with h
    · have h_set_eq : { a : α | ∃ i : ι, (hf i).mk (f i) b = a } =
        { a : α | ∃ i : ι, f i b = a } := by
        ext x
        simp_rw [Set.mem_setOf_eq, aeSeq.mk_eq_fun_of_mem_aeSeqSet hf h]
      rw [h_set_eq]
      exact aeSeq.fun_prop_of_mem_aeSeqSet hf h
    · exact IsGreatest.isLUB ⟨(@exists_const (hα.some = hα.some) ι _).2 rfl, fun x ⟨i, hi⟩ => hi.ge⟩
  refine ⟨g_seq, Measurable.isLUB (aeSeq.measurable hf p) hg_seq, ?_⟩
  exact
    (ite_ae_eq_of_measure_compl_zero g (fun _ => hα.some) (aeSeqSet hf p)
        (aeSeq.measure_compl_aeSeqSet_eq_zero hf hg)).symm


/-- If a function is the greatest lower bound of countably many measurable functions,
then it is measurable. -/
theorem Measurable.isGLB {ι} [Countable ι] {f : ι → δ → α} {g : δ → α} (hf : ∀ i, Measurable (f i))
    (hg : ∀ b, IsGLB { a | ∃ i, f i b = a } (g b)) : Measurable g :=
  Measurable.isLUB (α := αᵒᵈ) hf hg


/-- If a function is the greatest lower bound of countably many measurable functions on a measurable
set `s`, and coincides with a measurable function outside of `s`, then it is measurable. -/
theorem Measurable.isGLB_of_mem {ι} [Countable ι] {f : ι → δ → α} {g g' : δ → α}
    (hf : ∀ i, Measurable (f i))
    {s : Set δ} (hs : MeasurableSet s) (hg : ∀ b ∈ s, IsGLB { a | ∃ i, f i b = a } (g b))
    (hg' : EqOn g g' sᶜ) (g'_meas : Measurable g') : Measurable g :=
  Measurable.isLUB_of_mem (α := αᵒᵈ) hf hs hg hg'  g'_meas


theorem AEMeasurable.isGLB {ι} {μ : Measure δ} [Countable ι] {f : ι → δ → α} {g : δ → α}
    (hf : ∀ i, AEMeasurable (f i) μ) (hg : ∀ᵐ b ∂μ, IsGLB { a | ∃ i, f i b = a } (g b)) :
    AEMeasurable g μ :=
  AEMeasurable.isLUB (α := αᵒᵈ) hf hg


protected theorem Monotone.measurable [LinearOrder β] [OrderClosedTopology β] {f : β → α}
    (hf : Monotone f) : Measurable f :=
  suffices h : ∀ x, OrdConnected (f ⁻¹' Ioi x) from measurable_of_Ioi fun x => (h x).measurableSet
  fun _ => ordConnected_def.mpr fun _a ha _ _ _c hc => lt_of_lt_of_le ha (hf hc.1)


theorem aemeasurable_restrict_of_monotoneOn [LinearOrder β] [OrderClosedTopology β] {μ : Measure β}
    {s : Set β} (hs : MeasurableSet s) {f : β → α} (hf : MonotoneOn f s) :
    AEMeasurable f (μ.restrict s) :=
  have : Monotone (f ∘ (↑) : s → α) := fun ⟨x, hx⟩ ⟨y, hy⟩ => fun (hxy : x ≤ y) => hf hx hy hxy
  aemeasurable_restrict_of_measurable_subtype hs this.measurable


protected theorem Antitone.measurable [LinearOrder β] [OrderClosedTopology β] {f : β → α}
    (hf : Antitone f) : Measurable f :=
  @Monotone.measurable αᵒᵈ β _ _ ‹_› _ _ _ _ _ ‹_› _ _ _ hf


theorem aemeasurable_restrict_of_antitoneOn [LinearOrder β] [OrderClosedTopology β] {μ : Measure β}
    {s : Set β} (hs : MeasurableSet s) {f : β → α} (hf : AntitoneOn f s) :
    AEMeasurable f (μ.restrict s) :=
  @aemeasurable_restrict_of_monotoneOn αᵒᵈ β _ _ ‹_› _ _ _ _ _ ‹_› _ _ _ _ hs _ hf


theorem MeasurableSet.of_mem_nhdsGT_aux {s : Set α} (h : ∀ x ∈ s, s ∈ 𝓝[>] x)
    (h' : ∀ x ∈ s, ∃ y, x < y) : MeasurableSet s := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
    h' : ∀ (x : α), Membership.mem s x → Exists fun y => LT.lt x y
    ⊢ MeasurableSet s
  -/
  choose! M hM using h'
  suffices H : (s \ interior s).Countable by
    have : s = interior s ∪ s \ interior s := by rw [union_diff_cancel interior_subset]
    rw [this]
    exact isOpen_interior.measurableSet.union H.measurableSet
  have A : ∀ x ∈ s, ∃ y ∈ Ioi x, Ioo x y ⊆ s := fun x hx =>
    (mem_nhdsGT_iff_exists_Ioo_subset' (hM x hx)).1 (h x hx)
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
    M : α → α
    hM : ∀ (x : α), Membership.mem s x → LT.lt x (M x)
    A : ∀ (x : α), Membership.mem s x → Exists fun y => And (Membership.mem (Set.I …
    ⊢ (SDiff.sdiff s (interior s)).Countable
  -/
  choose! y hy h'y using A
  have B : Set.PairwiseDisjoint (s \ interior s) fun x => Ioo x (y x) := by
    intro x hx x' hx' hxx'
    rcases lt_or_gt_of_ne hxx' with (h' | h')
    · refine disjoint_left.2 fun z hz h'z => ?_
      have : x' ∈ interior s :=
        mem_interior.2 ⟨Ioo x (y x), h'y _ hx.1, isOpen_Ioo, ⟨h', h'z.1.trans hz.2⟩⟩
      exact False.elim (hx'.2 this)
    · refine disjoint_left.2 fun z hz h'z => ?_
      have : x ∈ interior s :=
        mem_interior.2 ⟨Ioo x' (y x'), h'y _ hx'.1, isOpen_Ioo, ⟨h', hz.1.trans h'z.2⟩⟩
      exact False.elim (hx.2 this)
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
    M : α → α
    hM : ∀ (x : α), Membership.mem s x → LT.lt x (M x)
    y : α → α
    hy : ∀ (x : α), Membership.mem s x → Membership.mem (Set.Ioi x) (y x)
    h'y : ∀ (x : α), Membership.mem s x → HasSubset.Subset (Set.Ioo x (y x)) s
    B : (SDiff.sdiff s (interior s)).PairwiseDisjoint fun x => Set.Ioo x (y x)
    ⊢ (SDiff.sdiff s (interior s)).Countable
  -/
  exact B.countable_of_Ioo fun x hx => hy x hx.1
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias measurableSet_of_mem_nhdsWithin_Ioi_aux := MeasurableSet.of_mem_nhdsGT_aux


/-- If a set is a right-neighborhood of all of its points, then it is measurable. -/
theorem MeasurableSet.of_mem_nhdsGT {s : Set α} (h : ∀ x ∈ s, s ∈ 𝓝[>] x) : MeasurableSet s := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : LinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    s : Set α
    h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
    ⊢ MeasurableSet s
  -/
  by_cases H : ∃ x ∈ s, IsTop x
    /-
      case pos
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      H : Exists fun x => And (Membership.mem s x) (IsTop x)
      ⊢ MeasurableSet s
    -/
  · rcases H with ⟨x₀, x₀s, h₀⟩
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      ⊢ MeasurableSet s
    -/
    have : s = { x₀ } ∪ s \ { x₀ } := by rw [union_diff_cancel (singleton_subset_iff.2 x₀s)]
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      this : Eq s (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleton.si …
      ⊢ MeasurableSet s
    -/
    rw [this]
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      this : Eq s (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleton.si …
      ⊢ MeasurableSet (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleto …
    -/
    refine (measurableSet_singleton _).union ?_
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      this : Eq s (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleton.si …
      ⊢ MeasurableSet (SDiff.sdiff s (Singleton.singleton x₀))
    -/
    have A : ∀ x ∈ s \ { x₀ }, x < x₀ := fun x hx => lt_of_le_of_ne (h₀ _) (by simpa using hx.2)
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      this : Eq s (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleton.si …
      A : ∀ (x : α), Membership.mem (SDiff.sdiff s (Singleton.singleton x₀)) x → LT. …
      ⊢ MeasurableSet (SDiff.sdiff s (Singleton.singleton x₀))
    -/
    refine .of_mem_nhdsGT_aux (fun x hx => ?_) fun x hx => ⟨x₀, A x hx⟩
    obtain ⟨u, hu, us⟩ : ∃ (u : α), u ∈ Ioi x ∧ Ioo x u ⊆ s :=
      (mem_nhdsGT_iff_exists_Ioo_subset' (A x hx)).1 (h x hx.1)
    /-
      case pos.intro.intro.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      this : Eq s (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleton.si …
      A : ∀ (x : α), Membership.mem (SDiff.sdiff s (Singleton.singleton x₀)) x → LT. …
      x : α
      hx : Membership.mem (SDiff.sdiff s (Singleton.singleton x₀)) x
      u : α
      hu : Membership.mem (Set.Ioi x) u
      us : HasSubset.Subset (Set.Ioo x u) s
      ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (SDiff.sdiff s (Singleton.singleto …
    -/
    refine (mem_nhdsGT_iff_exists_Ioo_subset' (A x hx)).2 ⟨u, hu, fun y hy => ⟨us hy, ?_⟩⟩
    /-
      case pos.intro.intro.intro.intro
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      x₀ : α
      x₀s : Membership.mem s x₀
      h₀ : IsTop x₀
      this : Eq s (Union.union (Singleton.singleton x₀) (SDiff.sdiff s (Singleton.si …
      A : ∀ (x : α), Membership.mem (SDiff.sdiff s (Singleton.singleton x₀)) x → LT. …
      x : α
      hx : Membership.mem (SDiff.sdiff s (Singleton.singleton x₀)) x
      u : α
      hu : Membership.mem (Set.Ioi x) u
      us : HasSubset.Subset (Set.Ioo x u) s
      y : α
      hy : Membership.mem (Set.Ioo x u) y
      ⊢ Not (Membership.mem (Singleton.singleton x₀) y)
    -/
    exact ne_of_lt (hy.2.trans_le (h₀ _))
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      H : Not (Exists fun x => And (Membership.mem s x) (IsTop x))
      ⊢ MeasurableSet s
    -/
  · refine .of_mem_nhdsGT_aux h ?_
    /-
      case neg
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      H : Not (Exists fun x => And (Membership.mem s x) (IsTop x))
      ⊢ ∀ (x : α), Membership.mem s x → Exists fun y => LT.lt x y
    -/
    simp only [IsTop] at H
    /-
      case neg
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      H : Not (Exists fun x => And (Membership.mem s x) (∀ (b : α), LE.le b x))
      ⊢ ∀ (x : α), Membership.mem s x → Exists fun y => LT.lt x y
    -/
    push_neg at H
    /-
      case neg
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      inst✝² : LinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Membership.mem (nhdsWithin x (Set.Ioi x)) s
      H : ∀ (x : α), Membership.mem s x → Exists fun b => LT.lt x b
      ⊢ ∀ (x : α), Membership.mem s x → Exists fun y => LT.lt x y
    -/
    exact H
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias measurableSet_of_mem_nhdsWithin_Ioi := MeasurableSet.of_mem_nhdsGT


lemma measurableSet_bddAbove_range {ι} [Countable ι] {f : ι → δ → α} (hf : ∀ i, Measurable (f i)) :
    MeasurableSet {b | BddAbove (range (fun i ↦ f i b))} := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    ⊢ MeasurableSet (setOf fun b => BddAbove (Set.range fun i => f i b))
  -/
  rcases isEmpty_or_nonempty α with hα|hα
    /-
      case inl
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hα : IsEmpty α
      ⊢ MeasurableSet (setOf fun b => BddAbove (Set.range fun i => f i b))
    -/
  · have : ∀ b, range (fun i ↦ f i b) = ∅ := fun b ↦ eq_empty_of_isEmpty _
    /-
      case inl
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : LinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hα : IsEmpty α
      this : ∀ (b : δ), Eq (Set.range fun i => f i b) EmptyCollection.emptyCollection
      ⊢ MeasurableSet (setOf fun b => BddAbove (Set.range fun i => f i b))
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  have A : ∀ (i : ι) (c : α), MeasurableSet {x | f i x ≤ c} := by
    intro i c
    exact measurableSet_le (hf i) measurable_const
  have B : ∀ (c : α), MeasurableSet {x | ∀ i, f i x ≤ c} := by
    intro c
    rw [setOf_forall]
    exact MeasurableSet.iInter (fun i ↦ A i c)
  /-
    case inr
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hα : Nonempty α
    A : ∀ (i : ι) (c : α), MeasurableSet (setOf fun x => LE.le (f i x) c)
    B : ∀ (c : α), MeasurableSet (setOf fun x => ∀ (i : ι), LE.le (f i x) c)
    ⊢ MeasurableSet (setOf fun b => BddAbove (Set.range fun i => f i b))
  -/
  obtain ⟨u, hu⟩ : ∃ (u : ℕ → α), Tendsto u atTop atTop := exists_seq_tendsto (atTop : Filter α)
  have : {b | BddAbove (range (fun i ↦ f i b))} = {x | ∃ n, ∀ i, f i x ≤ u n} := by
    apply Subset.antisymm
    · rintro x ⟨c, hc⟩
      obtain ⟨n, hn⟩ : ∃ n, c ≤ u n := (tendsto_atTop.1 hu c).exists
      exact ⟨n, fun i ↦ (hc ((mem_range_self i))).trans hn⟩
    · rintro x ⟨n, hn⟩
      refine ⟨u n, ?_⟩
      rintro - ⟨i, rfl⟩
      exact hn i
  /-
    case inr.intro
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hα : Nonempty α
    A : ∀ (i : ι) (c : α), MeasurableSet (setOf fun x => LE.le (f i x) c)
    B : ∀ (c : α), MeasurableSet (setOf fun x => ∀ (i : ι), LE.le (f i x) c)
    u : Nat → α
    hu : Filter.Tendsto u Filter.atTop Filter.atTop
    this : Eq (setOf fun b => BddAbove (Set.range fun i => f i b)) (setOf fun x => …
    ⊢ MeasurableSet (setOf fun b => BddAbove (Set.range fun i => f i b))
  -/
  rw [this, setOf_exists]
  /-
    case inr.intro
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : LinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hα : Nonempty α
    A : ∀ (i : ι) (c : α), MeasurableSet (setOf fun x => LE.le (f i x) c)
    B : ∀ (c : α), MeasurableSet (setOf fun x => ∀ (i : ι), LE.le (f i x) c)
    u : Nat → α
    hu : Filter.Tendsto u Filter.atTop Filter.atTop
    this : Eq (setOf fun b => BddAbove (Set.range fun i => f i b)) (setOf fun x => …
    ⊢ MeasurableSet (Set.iUnion fun i => setOf fun x => ∀ (i_1 : ι), LE.le (f i_1  …
  -/
  exact MeasurableSet.iUnion (fun n ↦ B (u n))
  /-
    🎉 no goals
  -/


lemma measurableSet_bddBelow_range {ι} [Countable ι] {f : ι → δ → α} (hf : ∀ i, Measurable (f i)) :
    MeasurableSet {b | BddBelow (range (fun i ↦ f i b))} :=
  measurableSet_bddAbove_range (α := αᵒᵈ) hf


@[measurability]
theorem Measurable.iSup_Prop {α} {mα : MeasurableSpace α} [ConditionallyCompleteLattice α]
    (p : Prop) {f : δ → α} (hf : Measurable f) : Measurable fun b => ⨆ _ : p, f b := by
  classical
  simp_rw [ciSup_eq_ite]
  split_ifs with h
  · exact hf
  · exact measurable_const


@[measurability]
theorem Measurable.iInf_Prop {α} {mα : MeasurableSpace α} [ConditionallyCompleteLattice α]
    (p : Prop) {f : δ → α} (hf : Measurable f) : Measurable fun b => ⨅ _ : p, f b := by
  classical
  simp_rw [ciInf_eq_ite]
  split_ifs with h
  · exact hf
  · exact measurable_const


@[measurability, fun_prop]
protected theorem Measurable.iSup {ι} [Countable ι] {f : ι → δ → α} (hf : ∀ i, Measurable (f i)) :
    Measurable (fun b ↦ ⨆ i, f i b) := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    ⊢ Measurable fun b => iSup fun i => f i b
  -/
  rcases isEmpty_or_nonempty ι with hι|hι
    /-
      case inl
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : IsEmpty ι
      ⊢ Measurable fun b => iSup fun i => f i b
    -/
  · simp [iSup_of_empty']
    /-
      🎉 no goals
    -/
  have A : MeasurableSet {b | BddAbove (range (fun i ↦ f i b))} :=
    measurableSet_bddAbove_range hf
  /-
    case inr
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hι : Nonempty ι
    A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
    ⊢ Measurable fun b => iSup fun i => f i b
  -/
  have : Measurable (fun (_b : δ) ↦ sSup (∅ : Set α)) := measurable_const
  /-
    case inr
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), Measurable (f i)
    hι : Nonempty ι
    A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
    this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
    ⊢ Measurable fun b => iSup fun i => f i b
  -/
  apply Measurable.isLUB_of_mem hf A _ _ this
    /-
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      ⊢ ∀ (b : δ), Membership.mem (_root_.setOf fun b => BddAbove (Set.range fun i = …
    -/
  · rintro b ⟨c, hc⟩
    /-
      case intro
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      b : δ
      c : α
      hc : Membership.mem (upperBounds (Set.range fun i => f i b)) c
      ⊢ IsLUB (_root_.setOf fun a => Exists fun i => Eq (f i b) a) (iSup fun i => f  …
    -/
    apply isLUB_ciSup
    /-
      case intro.H
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      b : δ
      c : α
      hc : Membership.mem (upperBounds (Set.range fun i => f i b)) c
      ⊢ BddAbove (Set.range fun y => f y b)
    -/
    refine ⟨c, ?_⟩
    /-
      case intro.H
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      b : δ
      c : α
      hc : Membership.mem (upperBounds (Set.range fun i => f i b)) c
      ⊢ Membership.mem (upperBounds (Set.range fun y => f y b)) c
    -/
    rintro d ⟨i, rfl⟩
    /-
      case intro.H.intro
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      b : δ
      c : α
      hc : Membership.mem (upperBounds (Set.range fun i => f i b)) c
      i : ι
      ⊢ LE.le ((fun y => f y b) i) c
    -/
    exact hc (mem_range_self i)
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      ⊢ Set.EqOn (fun b => iSup fun i => f i b) (fun _b => SupSet.sSup EmptyCollecti …
    -/
  · intro b hb
    /-
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      b : δ
      hb : Membership.mem (HasCompl.compl (_root_.setOf fun b => BddAbove (Set.range …
      ⊢ Eq ((fun b => iSup fun i => f i b) b) ((fun _b => SupSet.sSup EmptyCollectio …
    -/
    apply csSup_of_not_bddAbove
    /-
      case hs
      α : Type u_1
      δ : Type u_4
      inst✝⁵ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝⁴ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : OrderTopology α
      inst✝¹ : SecondCountableTopology α
      ι : Sort u_5
      inst✝ : Countable ι
      f : ι → δ → α
      hf : ∀ (i : ι), Measurable (f i)
      hι : Nonempty ι
      A : MeasurableSet (_root_.setOf fun b => BddAbove (Set.range fun i => f i b))
      this : Measurable fun _b => SupSet.sSup EmptyCollection.emptyCollection
      b : δ
      hb : Membership.mem (HasCompl.compl (_root_.setOf fun b => BddAbove (Set.range …
      ⊢ Not (BddAbove (Set.range fun i => f i b))
    -/
    exact hb
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-21")]
alias measurable_iSup := Measurable.iSup

-- TODO: Why does this error?
-- /-- Compositional version of `Measurable.iSup` for use by `fun_prop`. -/
-- @[fun_prop]
-- protected lemma Measurable.iSup'' {_ : MeasurableSpace γ} {ι : Sort*} [Countable ι]
--     {f : ι → γ → δ → α} {h : γ → δ} (hf : ∀ i, Measurable ↿(f i)) (hh : Measurable h) :
--     Measurable fun a ↦ (⨆ i, f i a) (h a) := by
--   simp_rw [iSup_apply]
--   exact .iSup fun i ↦ by fun_prop


@[measurability]
protected theorem AEMeasurable.iSup {ι} {μ : Measure δ} [Countable ι] {f : ι → δ → α}
    (hf : ∀ i, AEMeasurable (f i) μ) : AEMeasurable (fun b => ⨆ i, f i b) μ := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    μ : MeasureTheory.Measure δ
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    ⊢ AEMeasurable (fun b => iSup fun i => f i b) μ
  -/
  refine ⟨fun b ↦ ⨆ i, (hf i).mk (f i) b, .iSup (fun i ↦ (hf i).measurable_mk), ?_⟩
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁵ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝⁴ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : OrderTopology α
    inst✝¹ : SecondCountableTopology α
    ι : Sort u_5
    μ : MeasureTheory.Measure δ
    inst✝ : Countable ι
    f : ι → δ → α
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun b => iSup fun i => f i b) fun b => iS …
  -/
  filter_upwards [ae_all_iff.2 (fun i ↦ (hf i).ae_eq_mk)] with b hb using by simp [hb]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-21")]
alias aemeasurable_iSup := AEMeasurable.iSup


@[measurability, fun_prop]
protected theorem Measurable.iInf {ι} [Countable ι] {f : ι → δ → α} (hf : ∀ i, Measurable (f i)) :
    Measurable fun b => ⨅ i, f i b :=
  .iSup (α := αᵒᵈ) hf


@[deprecated (since := "2024-10-21")]
alias measurable_iInf := Measurable.iInf


@[measurability]
protected theorem AEMeasurable.iInf {ι} {μ : Measure δ} [Countable ι] {f : ι → δ → α}
    (hf : ∀ i, AEMeasurable (f i) μ) : AEMeasurable (fun b => ⨅ i, f i b) μ :=
  .iSup (α := αᵒᵈ) hf


@[deprecated (since := "2024-10-21")]
alias aemeasurable_iInf := AEMeasurable.iInf


protected theorem Measurable.sSup {ι} {f : ι → δ → α} {s : Set ι} (hs : s.Countable)
    (hf : ∀ i ∈ s, Measurable (f i)) :
    Measurable fun x => sSup ((fun i => f i x) '' s) := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ι : Type u_5
    f : ι → δ → α
    s : Set ι
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    ⊢ Measurable fun x => SupSet.sSup (Set.image (fun i => f i x) s)
  -/
  simp_rw [image_eq_range]
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ι : Type u_5
    f : ι → δ → α
    s : Set ι
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    ⊢ Measurable fun x => SupSet.sSup (Set.range fun x_1 => f (↑x_1) x)
  -/
  have : Countable s := hs.to_subtype
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ι : Type u_5
    f : ι → δ → α
    s : Set ι
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    this : Countable ↑s
    ⊢ Measurable fun x => SupSet.sSup (Set.range fun x_1 => f (↑x_1) x)
  -/
  exact .iSup fun i ↦ hf i i.2
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-21")]
alias measurable_sSup := Measurable.sSup


protected theorem Measurable.sInf {ι} {f : ι → δ → α} {s : Set ι} (hs : s.Countable)
    (hf : ∀ i ∈ s, Measurable (f i)) :
    Measurable fun x => sInf ((fun i => f i x) '' s) :=
  .sSup (α := αᵒᵈ) hs hf


@[deprecated (since := "2024-10-21")]
alias measurable_sInf := Measurable.sInf


theorem Measurable.biSup {ι} (s : Set ι) {f : ι → δ → α} (hs : s.Countable)
    (hf : ∀ i ∈ s, Measurable (f i)) : Measurable fun b => ⨆ i ∈ s, f i b := by
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ι : Type u_5
    s : Set ι
    f : ι → δ → α
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    ⊢ Measurable fun b => iSup fun i => iSup fun h => f i b
  -/
  haveI : Encodable s := hs.toEncodable
  /-
    α : Type u_1
    δ : Type u_4
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    mδ : MeasurableSpace δ
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : OrderTopology α
    inst✝ : SecondCountableTopology α
    ι : Type u_5
    s : Set ι
    f : ι → δ → α
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    this : Encodable ↑s
    ⊢ Measurable fun b => iSup fun i => iSup fun h => f i b
  -/
  by_cases H : ∀ i, i ∈ s
  · have : ∀ b, ⨆ i ∈ s, f i b = ⨆ (i : s), f i b :=
      fun b ↦ cbiSup_eq_of_forall (f := fun i ↦ f i b) H
    /-
      case pos
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      ι : Type u_5
      s : Set ι
      f : ι → δ → α
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
      this✝ : Encodable ↑s
      H : ∀ (i : ι), Membership.mem s i
      this : ∀ (b : δ), Eq (iSup fun i => iSup fun h => f i b) (iSup fun i => f (↑i) …
      ⊢ Measurable fun b => iSup fun i => iSup fun h => f i b
    -/
    simp only [this]
    /-
      case pos
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      ι : Type u_5
      s : Set ι
      f : ι → δ → α
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
      this✝ : Encodable ↑s
      H : ∀ (i : ι), Membership.mem s i
      this : ∀ (b : δ), Eq (iSup fun i => iSup fun h => f i b) (iSup fun i => f (↑i) …
      ⊢ Measurable fun b => iSup fun i => f (↑i) b
    -/
    exact .iSup (fun (i : s) ↦ hf i i.2)
    /-
      🎉 no goals
    -/
  · have : ∀ b, ⨆ i ∈ s, f i b = (⨆ (i : s), f i b) ⊔ sSup ∅ :=
      fun b ↦ cbiSup_eq_of_not_forall (f := fun i ↦ f i b) H
    /-
      case neg
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      ι : Type u_5
      s : Set ι
      f : ι → δ → α
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
      this✝ : Encodable ↑s
      H : Not (∀ (i : ι), Membership.mem s i)
      this : ∀ (b : δ), Eq (iSup fun i => iSup fun h => f i b) (Max.max (iSup fun i  …
      ⊢ Measurable fun b => iSup fun i => iSup fun h => f i b
    -/
    simp only [this]
    /-
      case neg
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      ι : Type u_5
      s : Set ι
      f : ι → δ → α
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
      this✝ : Encodable ↑s
      H : Not (∀ (i : ι), Membership.mem s i)
      this : ∀ (b : δ), Eq (iSup fun i => iSup fun h => f i b) (Max.max (iSup fun i  …
      ⊢ Measurable fun b => Max.max (iSup fun i => f (↑i) b) (SupSet.sSup EmptyColle …
    -/
    apply Measurable.sup _ measurable_const
    /-
      α : Type u_1
      δ : Type u_4
      inst✝⁴ : TopologicalSpace α
      mα : MeasurableSpace α
      inst✝³ : BorelSpace α
      mδ : MeasurableSpace δ
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : OrderTopology α
      inst✝ : SecondCountableTopology α
      ι : Type u_5
      s : Set ι
      f : ι → δ → α
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
      this✝ : Encodable ↑s
      H : Not (∀ (i : ι), Membership.mem s i)
      this : ∀ (b : δ), Eq (iSup fun i => iSup fun h => f i b) (Max.max (iSup fun i  …
      ⊢ Measurable fun a => iSup fun i => f (↑i) a
    -/
    exact .iSup (fun (i : s) ↦ hf i i.2)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-21")]
alias measurable_biSup := Measurable.biSup


theorem AEMeasurable.biSup {ι} {μ : Measure δ} (s : Set ι) {f : ι → δ → α} (hs : s.Countable)
    (hf : ∀ i ∈ s, AEMeasurable (f i) μ) : AEMeasurable (fun b => ⨆ i ∈ s, f i b) μ := by
  classical
  let g : ι → δ → α := fun i ↦ if hi : i ∈ s then (hf i hi).mk (f i) else fun _b ↦ sSup ∅
  have : ∀ i ∈ s, Measurable (g i) := by
    intro i hi
    simpa [g, hi] using (hf i hi).measurable_mk
  refine ⟨fun b ↦ ⨆ (i) (_ : i ∈ s), g i b, .biSup s hs this, ?_⟩
  have : ∀ i ∈ s, ∀ᵐ b ∂μ, f i b = g i b :=
    fun i hi ↦ by simpa [g, hi] using (hf i hi).ae_eq_mk
  filter_upwards [(ae_ball_iff hs).2 this] with b hb
  exact iSup_congr fun i => iSup_congr (hb i)


@[deprecated (since := "2024-10-21")]
alias aemeasurable_biSup := AEMeasurable.biSup


theorem Measurable.biInf {ι} (s : Set ι) {f : ι → δ → α} (hs : s.Countable)
    (hf : ∀ i ∈ s, Measurable (f i)) : Measurable fun b => ⨅ i ∈ s, f i b :=
  .biSup (α := αᵒᵈ) s hs hf


@[deprecated (since := "2024-10-21")]
alias measurable_biInf := Measurable.biInf


theorem AEMeasurable.biInf {ι} {μ : Measure δ} (s : Set ι) {f : ι → δ → α} (hs : s.Countable)
    (hf : ∀ i ∈ s, AEMeasurable (f i) μ) : AEMeasurable (fun b => ⨅ i ∈ s, f i b) μ :=
  .biSup (α := αᵒᵈ) s hs hf


@[deprecated (since := "2024-10-21")]
alias aemeasurable_biInf := AEMeasurable.biInf


/-- `liminf` over a general filter is measurable. See `Measurable.liminf` for the version over `ℕ`.
-/
theorem Measurable.liminf' {ι ι'} {f : ι → δ → α} {v : Filter ι} (hf : ∀ i, Measurable (f i))
    {p : ι' → Prop} {s : ι' → Set ι} (hv : v.HasCountableBasis p s) (hs : ∀ j, (s j).Countable) :
    Measurable fun x => liminf (f · x) v := by
  classical
  /- We would like to write the liminf as `⨆ (j : Subtype p), ⨅ (i : s j), f i x`, as the
  measurability would follow from the measurability of infs and sups. Unfortunately, this is not
  true in general conditionally complete linear orders because of issues with empty sets or sets
  which are not bounded above or below. A slightly more complicated expression for the liminf,
  valid in general, is given in `Filter.HasBasis.liminf_eq_ite`. This expression, built from
  `if ... then ... else` and infs and sups, can be readily checked to be measurable. -/
  have : Countable (Subtype p) := hv.countable
  rcases isEmpty_or_nonempty (Subtype p) with hp|hp
  · simp [hv.liminf_eq_sSup_iUnion_iInter]
  by_cases H : ∃ (j : Subtype p), s j = ∅
  · simp_rw [hv.liminf_eq_ite, if_pos H, measurable_const]
  simp_rw [hv.liminf_eq_ite, if_neg H]
  have : ∀ i, Countable (s i) := fun i ↦ countable_coe_iff.2 (hs i)
  let m : Subtype p → Set δ := fun j ↦ {x | BddBelow (range (fun (i : s j) ↦ f i x))}
  have m_meas : ∀ j, MeasurableSet (m j) :=
    fun j ↦ measurableSet_bddBelow_range (fun (i : s j) ↦ hf i)
  have mc_meas : MeasurableSet {x | ∀ (j : Subtype p), x ∉ m j} := by
    rw [setOf_forall]
    exact MeasurableSet.iInter (fun j ↦ (m_meas j).compl)
  refine measurable_const.piecewise mc_meas <| .iSup fun j ↦ ?_
  let reparam : δ → Subtype p → Subtype p := fun x ↦ liminf_reparam (fun i ↦ f i x) s p
  let F0 : Subtype p → δ → α := fun j x ↦ ⨅ (i : s j), f i x
  have F0_meas : ∀ j, Measurable (F0 j) := fun j ↦ .iInf (fun (i : s j) ↦ hf i)
  set F1 : δ → α := fun x ↦ F0 (reparam x j) x with hF1
  change Measurable F1
  let g : ℕ → Subtype p := Classical.choose (exists_surjective_nat (Subtype p))
  have Z : ∀ x, ∃ n, x ∈ m (g n) ∨ ∀ k, x ∉ m k := by
    intro x
    by_cases H : ∃ k, x ∈ m k
    · rcases H with ⟨k, hk⟩
      rcases Classical.choose_spec (exists_surjective_nat (Subtype p)) k with ⟨n, rfl⟩
      exact ⟨n, Or.inl hk⟩
    · push_neg at H
      exact ⟨0, Or.inr H⟩
  have : F1 = fun x ↦ if x ∈ m j then F0 j x else F0 (g (Nat.find (Z x))) x := by
    ext x
    have A : reparam x j = if x ∈ m j then j else g (Nat.find (Z x)) := rfl
    split_ifs with hjx
    · have : reparam x j = j := by rw [A, if_pos hjx]
      simp only [hF1, this]
    · have : reparam x j = g (Nat.find (Z x)) := by rw [A, if_neg hjx]
      simp only [hF1, this]
  rw [this]
  apply Measurable.piecewise (m_meas j) (F0_meas j)
  apply Measurable.find (fun n ↦ F0_meas (g n)) (fun n ↦ ?_)
  exact (m_meas (g n)).union mc_meas


@[deprecated (since := "2024-10-21")]
alias measurable_liminf' := Measurable.liminf'


/-- `limsup` over a general filter is measurable. See `Measurable.limsup` for the version over `ℕ`.
-/
theorem Measurable.limsup' {ι ι'} {f : ι → δ → α} {u : Filter ι} (hf : ∀ i, Measurable (f i))
    {p : ι' → Prop} {s : ι' → Set ι} (hu : u.HasCountableBasis p s) (hs : ∀ i, (s i).Countable) :
    Measurable fun x => limsup (fun i => f i x) u :=
  .liminf' (α := αᵒᵈ) hf hu hs


@[deprecated (since := "2024-10-21")]
alias measurable_limsup' := Measurable.limsup'


/-- `liminf` over `ℕ` is measurable. See `Measurable.liminf'` for a version with a general filter.
-/
@[measurability]
theorem Measurable.liminf {f : ℕ → δ → α} (hf : ∀ i, Measurable (f i)) :
    Measurable fun x => liminf (fun i => f i x) atTop :=
  .liminf' hf atTop_countable_basis fun _ => to_countable _


@[deprecated (since := "2024-10-21")]
alias measurable_liminf := Measurable.liminf


/-- `limsup` over `ℕ` is measurable. See `Measurable.limsup'` for a version with a general filter.
-/
@[measurability]
theorem Measurable.limsup {f : ℕ → δ → α} (hf : ∀ i, Measurable (f i)) :
    Measurable fun x => limsup (fun i => f i x) atTop :=
  .limsup' hf atTop_countable_basis fun _ => to_countable _


@[deprecated (since := "2024-10-21")]
alias measurable_limsup := Measurable.limsup


/-- Convert a `Homeomorph` to a `MeasurableEquiv`. -/
def Homemorph.toMeasurableEquiv (h : α ≃ₜ β) : α ≃ᵐ β where
  toEquiv := h.toEquiv
  measurable_toFun := h.continuous_toFun.measurable
  measurable_invFun := h.continuous_invFun.measurable


protected theorem IsFiniteMeasureOnCompacts.map (μ : Measure α) [IsFiniteMeasureOnCompacts μ]
    (f : α ≃ₜ β) : IsFiniteMeasureOnCompacts (Measure.map f μ) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : TopologicalSpace β
    mβ : MeasurableSpace β
    inst✝¹ : BorelSpace β
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    f : Homeomorph α β
    ⊢ MeasureTheory.IsFiniteMeasureOnCompacts (MeasureTheory.Measure.map (⇑f) μ)
  -/
  refine ⟨fun K hK ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : TopologicalSpace β
    mβ : MeasurableSpace β
    inst✝¹ : BorelSpace β
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    f : Homeomorph α β
    K : Set β
    hK : IsCompact K
    ⊢ LT.lt ((MeasureTheory.Measure.map (⇑f) μ) K) Top.top
  -/
  rw [← Homeomorph.toMeasurableEquiv_coe, MeasurableEquiv.map_apply]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : TopologicalSpace α
    mα : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : TopologicalSpace β
    mβ : MeasurableSpace β
    inst✝¹ : BorelSpace β
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    f : Homeomorph α β
    K : Set β
    hK : IsCompact K
    ⊢ LT.lt (μ (Set.preimage (⇑f.toMeasurableEquiv) K)) Top.top
  -/
  exact IsCompact.measure_lt_top (f.isCompact_preimage.2 hK)
  /-
    🎉 no goals
  -/


/-- One can cut out `ℝ≥0∞` into the sets `{0}`, `Ico (t^n) (t^(n+1))` for `n : ℤ` and `{∞}`. This
gives a way to compute the measure of a set in terms of sets on which a given function `f` does not
fluctuate by more than `t`. -/
theorem measure_eq_measure_preimage_add_measure_tsum_Ico_zpow {α : Type*} {mα : MeasurableSpace α}
    (μ : Measure α) {f : α → ℝ≥0∞} (hf : Measurable f) {s : Set α} (hs : MeasurableSet s)
    {t : ℝ≥0} (ht : 1 < t) :
    μ s =
      μ (s ∩ f ⁻¹' {0}) + μ (s ∩ f ⁻¹' {∞}) +
      ∑' n : ℤ, μ (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) := by
  have A : μ s = μ (s ∩ f ⁻¹' {0}) + μ (s ∩ f ⁻¹' Ioi 0) := by
    rw [← measure_union]
    · rw [← inter_union_distrib_left, ← preimage_union, singleton_union, Ioi_insert,
        ← _root_.bot_eq_zero, Ici_bot, preimage_univ, inter_univ]
    · exact disjoint_singleton_left.mpr not_mem_Ioi_self
        |>.preimage f |>.inter_right' s |>.inter_left' s
    · exact hs.inter (hf measurableSet_Ioi)
  have B : μ (s ∩ f ⁻¹' Ioi 0) = μ (s ∩ f ⁻¹' {∞}) + μ (s ∩ f ⁻¹' Ioo 0 ∞) := by
    rw [← measure_union]
    · rw [← inter_union_distrib_left]
      congr
      ext x
      simp only [mem_singleton_iff, mem_union, mem_Ioo, mem_Ioi, mem_preimage]
      obtain (H | H) : f x = ∞ ∨ f x < ∞ := eq_or_lt_of_le le_top
      · simp only [H, eq_self_iff_true, or_false, ENNReal.zero_lt_top, not_top_lt, and_false]
      · simp only [H, H.ne, and_true, false_or]
    · refine disjoint_left.2 fun x hx h'x => ?_
      have : f x < ∞ := h'x.2.2
      exact lt_irrefl _ (this.trans_le (le_of_eq hx.2.symm))
    · exact hs.inter (hf measurableSet_Ioo)
  have C : μ (s ∩ f ⁻¹' Ioo 0 ∞) =
      ∑' n : ℤ, μ (s ∩ f ⁻¹' Ico ((t : ℝ≥0∞) ^ n) ((t : ℝ≥0∞) ^ (n + 1))) := by
    rw [← measure_iUnion,
      ENNReal.Ioo_zero_top_eq_iUnion_Ico_zpow (ENNReal.one_lt_coe_iff.2 ht) ENNReal.coe_ne_top,
      preimage_iUnion, inter_iUnion]
    · intro i j hij
      wlog h : i < j generalizing i j
      · exact (this hij.symm (hij.lt_or_lt.resolve_left h)).symm
      refine disjoint_left.2 fun x hx h'x => lt_irrefl (f x) ?_
      calc
        f x < (t : ℝ≥0∞) ^ (i + 1) := hx.2.2
        _ ≤ (t : ℝ≥0∞) ^ j := ENNReal.zpow_le_of_le (ENNReal.one_le_coe_iff.2 ht.le) h
        _ ≤ f x := h'x.2.1
    · intro n
      exact hs.inter (hf measurableSet_Ico)
  /-
    α : Type u_5
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    s : Set α
    hs : MeasurableSet s
    t : NNReal
    ht : LT.lt 1 t
    A : Eq (μ s) (HAdd.hAdd (μ (Inter.inter s (Set.preimage f (Singleton.singleton …
    B : Eq (μ (Inter.inter s (Set.preimage f (Set.Ioi 0)))) (HAdd.hAdd (μ (Inter.i …
    C : Eq (μ (Inter.inter s (Set.preimage f (Set.Ioo 0 Top.top)))) (tsum fun n => …
    ⊢ Eq (μ s) (HAdd.hAdd (HAdd.hAdd (μ (Inter.inter s (Set.preimage f (Singleton. …
  -/
  rw [A, B, C, add_assoc]
  /-
    🎉 no goals
  -/


