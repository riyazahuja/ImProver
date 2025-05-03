/-- A (topological) fiber bundle with fiber `F` over a base `B` is a space projecting on `B`
for which the fibers are all homeomorphic to `F`, such that the local situation around each point
is a direct product. -/
class FiberBundle where
  totalSpaceMk_isInducing' : ∀ b : B, IsInducing (@TotalSpace.mk B F E b)
  trivializationAtlas' : Set (Trivialization F (π F E))
  trivializationAt' : B → Trivialization F (π F E)
  mem_baseSet_trivializationAt' : ∀ b : B, b ∈ (trivializationAt' b).baseSet
  trivialization_mem_atlas' : ∀ b : B, trivializationAt' b ∈ trivializationAtlas'


theorem totalSpaceMk_isInducing : IsInducing (@TotalSpace.mk B F E b) := totalSpaceMk_isInducing' b


@[deprecated (since := "2024-10-28")] alias totalSpaceMk_inducing := totalSpaceMk_isInducing


/-- Atlas of a fiber bundle. -/
abbrev trivializationAtlas : Set (Trivialization F (π F E)) := trivializationAtlas'


/-- Trivialization of a fiber bundle at a point. -/
abbrev trivializationAt : Trivialization F (π F E) := trivializationAt' b


theorem mem_baseSet_trivializationAt : b ∈ (trivializationAt F E b).baseSet :=
  mem_baseSet_trivializationAt' b


theorem trivialization_mem_atlas : trivializationAt F E b ∈ trivializationAtlas F E :=
  trivialization_mem_atlas' b


/-- Given a type `E` equipped with a fiber bundle structure, this is a `Prop` typeclass
for trivializations of `E`, expressing that a trivialization is in the designated atlas for the
bundle.  This is needed because lemmas about the linearity of trivializations or the continuity (as
functions to `F →L[R] F`, where `F` is the model fiber) of the transition functions are only
expected to hold for trivializations in the designated atlas. -/
@[mk_iff]
class MemTrivializationAtlas [FiberBundle F E] (e : Trivialization F (π F E)) : Prop where
  out : e ∈ trivializationAtlas F E


instance [FiberBundle F E] (b : B) : MemTrivializationAtlas (trivializationAt F E b) where
  out := trivialization_mem_atlas F E b


theorem map_proj_nhds (x : TotalSpace F E) : map (π F E) (𝓝 x) = 𝓝 x.proj :=
  (trivializationAt F E x.proj).map_proj_nhds <|
    (trivializationAt F E x.proj).mem_source.2 <| mem_baseSet_trivializationAt F E x.proj


/-- The projection from a fiber bundle to its base is continuous. -/
@[continuity]
theorem continuous_proj : Continuous (π F E) :=
  continuous_iff_continuousAt.2 fun x => (map_proj_nhds F x).le


/-- The projection from a fiber bundle to its base is an open map. -/
theorem isOpenMap_proj : IsOpenMap (π F E) :=
  IsOpenMap.of_nhds_le fun x => (map_proj_nhds F x).ge


/-- The projection from a fiber bundle with a nonempty fiber to its base is a surjective
map. -/
theorem surjective_proj [Nonempty F] : Function.Surjective (π F E) := fun b =>
  let ⟨p, _, hpb⟩ :=
    (trivializationAt F E b).proj_surjOn_baseSet (mem_baseSet_trivializationAt F E b)
  ⟨p, hpb⟩


/-- The projection from a fiber bundle with a nonempty fiber to its base is a quotient
map. -/
theorem isQuotientMap_proj [Nonempty F] : IsQuotientMap (π F E) :=
  (isOpenMap_proj F E).isQuotientMap (continuous_proj F E) (surjective_proj F E)


@[deprecated (since := "2024-10-22")]
alias quotientMap_proj := isQuotientMap_proj


theorem continuous_totalSpaceMk (x : B) : Continuous (@TotalSpace.mk B F E x) :=
  (totalSpaceMk_isInducing F E x).continuous


theorem totalSpaceMk_isEmbedding (x : B) : IsEmbedding (@TotalSpace.mk B F E x) :=
  ⟨totalSpaceMk_isInducing F E x, TotalSpace.mk_injective x⟩


@[deprecated (since := "2024-10-26")]
alias totalSpaceMk_embedding := totalSpaceMk_isEmbedding


theorem totalSpaceMk_isClosedEmbedding [T1Space B] (x : B) :
    IsClosedEmbedding (@TotalSpace.mk B F E x) :=
  ⟨totalSpaceMk_isEmbedding F E x, by
    /-
      B : Type u_2
      F : Type u_3
      inst✝⁵ : TopologicalSpace B
      inst✝⁴ : TopologicalSpace F
      E : B → Type u_5
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (b : B) → TopologicalSpace (E b)
      inst✝¹ : FiberBundle F E
      inst✝ : T1Space B
      x : B
      ⊢ IsClosed (Set.range (Bundle.TotalSpace.mk x))
    -/
    rw [TotalSpace.range_mk]
    /-
      B : Type u_2
      F : Type u_3
      inst✝⁵ : TopologicalSpace B
      inst✝⁴ : TopologicalSpace F
      E : B → Type u_5
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (b : B) → TopologicalSpace (E b)
      inst✝¹ : FiberBundle F E
      inst✝ : T1Space B
      x : B
      ⊢ IsClosed (Set.preimage Bundle.TotalSpace.proj (Singleton.singleton x))
    -/
    exact isClosed_singleton.preimage <| continuous_proj F E⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias totalSpaceMk_closedEmbedding := totalSpaceMk_isClosedEmbedding


@[simp, mfld_simps]
theorem mem_trivializationAt_proj_source {x : TotalSpace F E} :
    x ∈ (trivializationAt F E x.proj).source :=
  (Trivialization.mem_source _).mpr <| mem_baseSet_trivializationAt F E x.proj

-- Porting note: removed `@[simp, mfld_simps]` because `simp` could already prove this

theorem trivializationAt_proj_fst {x : TotalSpace F E} :
    ((trivializationAt F E x.proj) x).1 = x.proj :=
  Trivialization.coe_fst' _ <| mem_baseSet_trivializationAt F E x.proj


/-- Characterization of continuous functions (at a point, within a set) into a fiber bundle. -/
theorem continuousWithinAt_totalSpace (f : X → TotalSpace F E) {s : Set X} {x₀ : X} :
    ContinuousWithinAt f s x₀ ↔
      ContinuousWithinAt (fun x => (f x).proj) s x₀ ∧
        ContinuousWithinAt (fun x => ((trivializationAt F E (f x₀).proj) (f x)).2) s x₀ :=
  (trivializationAt F E (f x₀).proj).tendsto_nhds_iff mem_trivializationAt_proj_source


/-- Characterization of continuous functions (at a point) into a fiber bundle. -/
theorem continuousAt_totalSpace (f : X → TotalSpace F E) {x₀ : X} :
    ContinuousAt f x₀ ↔
      ContinuousAt (fun x => (f x).proj) x₀ ∧
        ContinuousAt (fun x => ((trivializationAt F E (f x₀).proj) (f x)).2) x₀ :=
  (trivializationAt F E (f x₀).proj).tendsto_nhds_iff mem_trivializationAt_proj_source


/-- If `E` is a fiber bundle over a conditionally complete linear order,
then it is trivial over any closed interval. -/
theorem FiberBundle.exists_trivialization_Icc_subset [ConditionallyCompleteLinearOrder B]
    [OrderTopology B] [FiberBundle F E] (a b : B) :
    ∃ e : Trivialization F (π F E), Icc a b ⊆ e.baseSet := by
  obtain ⟨ea, hea⟩ : ∃ ea : Trivialization F (π F E), a ∈ ea.baseSet :=
    ⟨trivializationAt F E a, mem_baseSet_trivializationAt F E a⟩
  -- If `a < b`, then `[a, b] = ∅`, and the statement is trivial
  /-
    case intro
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  cases' lt_or_le b a with hab hab
    /-
      case intro.inl
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LT.lt b a
      ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
    -/
  · exact ⟨ea, by simp [*]⟩
    /-
      🎉 no goals
    -/
  /- Let `s` be the set of points `x ∈ [a, b]` such that `E` is trivializable over `[a, x]`.
    We need to show that `b ∈ s`. Let `c = Sup s`. We will show that `c ∈ s` and `c = b`. -/
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  set s : Set B := { x ∈ Icc a b | ∃ e : Trivialization F (π F E), Icc a x ⊆ e.baseSet }
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  have ha : a ∈ s := ⟨left_mem_Icc.2 hab, ea, by simp [hea]⟩
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  have sne : s.Nonempty := ⟨a, ha⟩
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  have hsb : b ∈ upperBounds s := fun x hx => hx.1.2
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  have sbd : BddAbove s := ⟨b, hsb⟩
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  set c := sSup s
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    c : B := SupSet.sSup s
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  have hsc : IsLUB s c := isLUB_csSup sne sbd
  /-
    case intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    c : B := SupSet.sSup s
    hsc : IsLUB s c
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  have hc : c ∈ Icc a b := ⟨hsc.1 ha, hsc.2 hsb⟩
  obtain ⟨-, ec : Trivialization F (π F E), hec : Icc a c ⊆ ec.baseSet⟩ : c ∈ s := by
    rcases hc.1.eq_or_lt with heq | hlt
    · rwa [← heq]
    refine ⟨hc, ?_⟩
    /- In order to show that `c ∈ s`, consider a trivialization `ec` of `proj` over a neighborhood
      of `c`. Its base set includes `(c', c]` for some `c' ∈ [a, c)`. -/
    obtain ⟨ec, hc⟩ : ∃ ec : Trivialization F (π F E), c ∈ ec.baseSet :=
      ⟨trivializationAt F E c, mem_baseSet_trivializationAt F E c⟩
    obtain ⟨c', hc', hc'e⟩ : ∃ c' ∈ Ico a c, Ioc c' c ⊆ ec.baseSet :=
      (mem_nhdsLE_iff_exists_mem_Ico_Ioc_subset hlt).1
        (mem_nhdsWithin_of_mem_nhds <| IsOpen.mem_nhds ec.open_baseSet hc)
    /- Since `c' < c = Sup s`, there exists `d ∈ s ∩ (c', c]`. Let `ead` be a trivialization of
      `proj` over `[a, d]`. Then we can glue `ead` and `ec` into a trivialization over `[a, c]`. -/
    obtain ⟨d, ⟨hdab, ead, had⟩, hd⟩ : ∃ d ∈ s, d ∈ Ioc c' c := hsc.exists_between hc'.2
    refine ⟨ead.piecewiseLe ec d (had ⟨hdab.1, le_rfl⟩) (hc'e hd), subset_ite.2 ?_⟩
    exact ⟨fun x hx => had ⟨hx.1.1, hx.2⟩, fun x hx => hc'e ⟨hd.1.trans (not_le.1 hx.2), hx.1.2⟩⟩
  /- So, `c ∈ s`. Let `ec` be a trivialization of `proj` over `[a, c]`.  If `c = b`, then we are
    done. Otherwise we show that `proj` can be trivialized over a larger interval `[a, d]`,
    `d ∈ (c, b]`, hence `c` is not an upper bound of `s`. -/
  /-
    case intro.inr.intro.intro
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    c : B := SupSet.sSup s
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    ec : Trivialization F Bundle.TotalSpace.proj
    hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  rcases hc.2.eq_or_lt with heq | hlt
    /-
      case intro.inr.intro.intro.inl
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      heq : Eq c b
      ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
    -/
  · exact ⟨ec, heq ▸ hec⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.inr.intro.intro.inr
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    c : B := SupSet.sSup s
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    ec : Trivialization F Bundle.TotalSpace.proj
    hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
    hlt : LT.lt c b
    ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
  -/
  rsuffices ⟨d, hdcb, hd⟩ : ∃ d ∈ Ioc c b, ∃ e : Trivialization F (π F E), Icc a d ⊆ e.baseSet
    /-
      case intro.inr.intro.intro.inr.intro.intro
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      d : B
      hdcb : Membership.mem (Set.Ioc c b) d
      hd : Exists fun e => HasSubset.Subset (Set.Icc a d) e.baseSet
      ⊢ Exists fun e => HasSubset.Subset (Set.Icc a b) e.baseSet
    -/
  · exact ((hsc.1 ⟨⟨hc.1.trans hdcb.1.le, hdcb.2⟩, hd⟩).not_lt hdcb.1).elim
    /-
      🎉 no goals
    -/
  /- Since the base set of `ec` is open, it includes `[c, d)` (hence, `[a, d)`) for some
    `d ∈ (c, b]`. -/
  obtain ⟨d, hdcb, hd⟩ : ∃ d ∈ Ioc c b, Ico c d ⊆ ec.baseSet :=
    (mem_nhdsGE_iff_exists_mem_Ioc_Ico_subset hlt).1
      (mem_nhdsWithin_of_mem_nhds <| IsOpen.mem_nhds ec.open_baseSet (hec ⟨hc.1, le_rfl⟩))
  /-
    case intro.intro
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    c : B := SupSet.sSup s
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    ec : Trivialization F Bundle.TotalSpace.proj
    hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
    hlt : LT.lt c b
    d : B
    hdcb : Membership.mem (Set.Ioc c b) d
    hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
    ⊢ Exists fun d => And (Membership.mem (Set.Ioc c b) d) (Exists fun e => HasSub …
  -/
  have had : Ico a d ⊆ ec.baseSet := Ico_subset_Icc_union_Ico.trans (union_subset hec hd)
  /-
    case intro.intro
    B : Type u_2
    F : Type u_3
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace F
    E : B → Type u_5
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (b : B) → TopologicalSpace (E b)
    inst✝² : ConditionallyCompleteLinearOrder B
    inst✝¹ : OrderTopology B
    inst✝ : FiberBundle F E
    a b : B
    ea : Trivialization F Bundle.TotalSpace.proj
    hea : Membership.mem ea.baseSet a
    hab : LE.le a b
    s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
    ha : Membership.mem s a
    sne : s.Nonempty
    hsb : Membership.mem (upperBounds s) b
    sbd : BddAbove s
    c : B := SupSet.sSup s
    hsc : IsLUB s c
    hc : Membership.mem (Set.Icc a b) c
    ec : Trivialization F Bundle.TotalSpace.proj
    hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
    hlt : LT.lt c b
    d : B
    hdcb : Membership.mem (Set.Ioc c b) d
    hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
    had : HasSubset.Subset (Set.Ico a d) ec.baseSet
    ⊢ Exists fun d => And (Membership.mem (Set.Ioc c b) d) (Exists fun e => HasSub …
  -/
  by_cases he : Disjoint (Iio d) (Ioi c)
  · /- If `(c, d) = ∅`, then let `ed` be a trivialization of `proj` over a neighborhood of `d`.
      Then the disjoint union of `ec` restricted to `(-∞, d)` and `ed` restricted to `(c, ∞)` is
      a trivialization over `[a, d]`. -/
    obtain ⟨ed, hed⟩ : ∃ ed : Trivialization F (π F E), d ∈ ed.baseSet :=
      ⟨trivializationAt F E d, mem_baseSet_trivializationAt F E d⟩
    refine ⟨d, hdcb,
      (ec.restrOpen (Iio d) isOpen_Iio).disjointUnion (ed.restrOpen (Ioi c) isOpen_Ioi)
        (he.mono inter_subset_right inter_subset_right), fun x hx => ?_⟩
    /-
      case pos.intro
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      d : B
      hdcb : Membership.mem (Set.Ioc c b) d
      hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
      had : HasSubset.Subset (Set.Ico a d) ec.baseSet
      he : Disjoint (Set.Iio d) (Set.Ioi c)
      ed : Trivialization F Bundle.TotalSpace.proj
      hed : Membership.mem ed.baseSet d
      x : B
      hx : Membership.mem (Set.Icc a d) x
      ⊢ Membership.mem ((ec.restrOpen (Set.Iio d) ⋯).disjointUnion (ed.restrOpen (Se …
    -/
    rcases hx.2.eq_or_lt with (rfl | hxd)
    /-
      case pos.intro.inl
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      ed : Trivialization F Bundle.TotalSpace.proj
      x : B
      hdcb : Membership.mem (Set.Ioc c b) x
      hd : HasSubset.Subset (Set.Ico c x) ec.baseSet
      had : HasSubset.Subset (Set.Ico a x) ec.baseSet
      he : Disjoint (Set.Iio x) (Set.Ioi c)
      hed : Membership.mem ed.baseSet x
      hx : Membership.mem (Set.Icc a x) x
      ⊢ Membership.mem ((ec.restrOpen (Set.Iio x) ⋯).disjointUnion (ed.restrOpen (Se …
    -/
    exacts [Or.inr ⟨hed, hdcb.1⟩, Or.inl ⟨had ⟨hx.1, hxd⟩, hxd⟩]
    /-
      🎉 no goals
    -/
  · /- If `(c, d)` is nonempty, then take `d' ∈ (c, d)`. Since the base set of `ec` includes
          `[a, d)`, it includes `[a, d'] ⊆ [a, d)` as well. -/
    /-
      case neg
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      d : B
      hdcb : Membership.mem (Set.Ioc c b) d
      hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
      had : HasSubset.Subset (Set.Ico a d) ec.baseSet
      he : Not (Disjoint (Set.Iio d) (Set.Ioi c))
      ⊢ Exists fun d => And (Membership.mem (Set.Ioc c b) d) (Exists fun e => HasSub …
    -/
    rw [disjoint_left] at he
    /-
      case neg
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      d : B
      hdcb : Membership.mem (Set.Ioc c b) d
      hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
      had : HasSubset.Subset (Set.Ico a d) ec.baseSet
      he : Not (∀ ⦃a : B⦄, Membership.mem (Set.Iio d) a → Not (Membership.mem (Set.I …
      ⊢ Exists fun d => And (Membership.mem (Set.Ioc c b) d) (Exists fun e => HasSub …
    -/
    push_neg at he
    /-
      case neg
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      d : B
      hdcb : Membership.mem (Set.Ioc c b) d
      hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
      had : HasSubset.Subset (Set.Ico a d) ec.baseSet
      he : Exists fun ⦃a⦄ => And (Membership.mem (Set.Iio d) a) (Membership.mem (Set …
      ⊢ Exists fun d => And (Membership.mem (Set.Ioc c b) d) (Exists fun e => HasSub …
    -/
    rcases he with ⟨d', hdd' : d' < d, hd'c⟩
    /-
      case neg.intro.intro
      B : Type u_2
      F : Type u_3
      inst✝⁶ : TopologicalSpace B
      inst✝⁵ : TopologicalSpace F
      E : B → Type u_5
      inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝³ : (b : B) → TopologicalSpace (E b)
      inst✝² : ConditionallyCompleteLinearOrder B
      inst✝¹ : OrderTopology B
      inst✝ : FiberBundle F E
      a b : B
      ea : Trivialization F Bundle.TotalSpace.proj
      hea : Membership.mem ea.baseSet a
      hab : LE.le a b
      s : Set B := setOf fun x => And (Membership.mem (Set.Icc a b) x) (Exists fun e …
      ha : Membership.mem s a
      sne : s.Nonempty
      hsb : Membership.mem (upperBounds s) b
      sbd : BddAbove s
      c : B := SupSet.sSup s
      hsc : IsLUB s c
      hc : Membership.mem (Set.Icc a b) c
      ec : Trivialization F Bundle.TotalSpace.proj
      hec : HasSubset.Subset (Set.Icc a c) ec.baseSet
      hlt : LT.lt c b
      d : B
      hdcb : Membership.mem (Set.Ioc c b) d
      hd : HasSubset.Subset (Set.Ico c d) ec.baseSet
      had : HasSubset.Subset (Set.Ico a d) ec.baseSet
      d' : B
      hdd' : LT.lt d' d
      hd'c : Membership.mem (Set.Ioi c) d'
      ⊢ Exists fun d => And (Membership.mem (Set.Ioc c b) d) (Exists fun e => HasSub …
    -/
    exact ⟨d', ⟨hd'c, hdd'.le.trans hdcb.2⟩, ec, (Icc_subset_Ico_right hdd').trans had⟩
    /-
      🎉 no goals
    -/


/-- Core data defining a locally trivial bundle with fiber `F` over a topological
space `B`. Note that "bundle" is used in its mathematical sense. This is the (computer science)
bundled version, i.e., all the relevant data is contained in the following structure. A family of
local trivializations is indexed by a type `ι`, on open subsets `baseSet i` for each `i : ι`.
Trivialization changes from `i` to `j` are given by continuous maps `coordChange i j` from
`baseSet i ∩ baseSet j` to the set of homeomorphisms of `F`, but we express them as maps
`B → F → F` and require continuity on `(baseSet i ∩ baseSet j) × F` to avoid the topology on the
space of continuous maps on `F`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure FiberBundleCore (ι : Type*) (B : Type*) [TopologicalSpace B] (F : Type*)
    [TopologicalSpace F] where
  baseSet : ι → Set B
  isOpen_baseSet : ∀ i, IsOpen (baseSet i)
  indexAt : B → ι
  mem_baseSet_at : ∀ x, x ∈ baseSet (indexAt x)
  coordChange : ι → ι → B → F → F
  coordChange_self : ∀ i, ∀ x ∈ baseSet i, ∀ v, coordChange i i x v = v
  continuousOn_coordChange : ∀ i j,
    ContinuousOn (fun p : B × F => coordChange i j p.1 p.2) ((baseSet i ∩ baseSet j) ×ˢ univ)
  coordChange_comp : ∀ i j k, ∀ x ∈ baseSet i ∩ baseSet j ∩ baseSet k, ∀ v,
    (coordChange j k x) (coordChange i j x v) = coordChange i k x v


/-- The index set of a fiber bundle core, as a convenience function for dot notation -/
@[nolint unusedArguments] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was has_nonempty_instance
def Index (_Z : FiberBundleCore ι B F) := ι


/-- The base space of a fiber bundle core, as a convenience function for dot notation -/
@[nolint unusedArguments, reducible]
def Base (_Z : FiberBundleCore ι B F) := B


/-- The fiber of a fiber bundle core, as a convenience function for dot notation and
typeclass inference -/
@[nolint unusedArguments] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was has_nonempty_instance
def Fiber (_ : FiberBundleCore ι B F) (_x : B) := F


instance topologicalSpaceFiber (x : B) : TopologicalSpace (Z.Fiber x) := ‹_›


/-- The total space of the fiber bundle, as a convenience function for dot notation.
It is by definition equal to `Bundle.TotalSpace F Z.Fiber`. -/
abbrev TotalSpace := Bundle.TotalSpace F Z.Fiber


/-- The projection from the total space of a fiber bundle core, on its base. -/
@[reducible, simp, mfld_simps]
def proj : Z.TotalSpace → B :=
  Bundle.TotalSpace.proj


/-- Local homeomorphism version of the trivialization change. -/
def trivChange (i j : ι) : PartialHomeomorph (B × F) (B × F) where
  source := (Z.baseSet i ∩ Z.baseSet j) ×ˢ univ
  target := (Z.baseSet i ∩ Z.baseSet j) ×ˢ univ
  toFun p := ⟨p.1, Z.coordChange i j p.1 p.2⟩
  invFun p := ⟨p.1, Z.coordChange j i p.1 p.2⟩
                         /-
                           ι : Type u_1
                           B : Type u_2
                           F : Type u_3
                           X : Type u_4
                           inst✝² : TopologicalSpace X
                           inst✝¹ : TopologicalSpace B
                           inst✝ : TopologicalSpace F
                           Z : FiberBundleCore ι B F
                           i j : ι
                           p : Prod B F
                           hp : Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z.baseSet j)) Set …
                           ⊢ Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z.baseSet j)) Set.un …
                         -/
  map_source' p hp := by simpa using hp
                         /-
                           🎉 no goals
                         -/
                         /-
                           ι : Type u_1
                           B : Type u_2
                           F : Type u_3
                           X : Type u_4
                           inst✝² : TopologicalSpace X
                           inst✝¹ : TopologicalSpace B
                           inst✝ : TopologicalSpace F
                           Z : FiberBundleCore ι B F
                           i j : ι
                           p : Prod B F
                           hp : Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z.baseSet j)) Set …
                           ⊢ Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z.baseSet j)) Set.un …
                         -/
  map_target' p hp := by simpa using hp
                         /-
                           🎉 no goals
                         -/
  left_inv' := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      ⊢ ∀ ⦃x : Prod B F⦄, Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z. …
    -/
    rintro ⟨x, v⟩ hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z.baseSet j)) Set …
      ⊢ Eq ((fun p => { fst := p.1, snd := Z.coordChange j i p.1 p.2 }) ((fun p => { …
    -/
    simp only [prod_mk_mem_set_prod_eq, mem_inter_iff, and_true, mem_univ] at hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
      ⊢ Eq ((fun p => { fst := p.1, snd := Z.coordChange j i p.1 p.2 }) ((fun p => { …
    -/
    dsimp only
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
      ⊢ Eq { fst := x, snd := Z.coordChange j i x (Z.coordChange i j x v) } { fst := …
    -/
    rw [coordChange_comp, Z.coordChange_self]
    /-
      case mk.a
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
      ⊢ Membership.mem (Z.baseSet i) x
    -/
    exacts [hx.1, ⟨⟨hx.1, hx.2⟩, hx.1⟩]
    /-
      🎉 no goals
    -/
  right_inv' := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      ⊢ ∀ ⦃x : Prod B F⦄, Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z. …
    -/
    rintro ⟨x, v⟩ hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : Membership.mem (SProd.sprod (Inter.inter (Z.baseSet i) (Z.baseSet j)) Set …
      ⊢ Eq ((fun p => { fst := p.1, snd := Z.coordChange i j p.1 p.2 }) ((fun p => { …
    -/
    simp only [prod_mk_mem_set_prod_eq, mem_inter_iff, and_true, mem_univ] at hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
      ⊢ Eq ((fun p => { fst := p.1, snd := Z.coordChange i j p.1 p.2 }) ((fun p => { …
    -/
    dsimp only
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
      ⊢ Eq { fst := x, snd := Z.coordChange i j x (Z.coordChange j i x v) } { fst := …
    -/
    rw [Z.coordChange_comp, Z.coordChange_self]
      /-
        case mk.a
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace B
        inst✝ : TopologicalSpace F
        Z : FiberBundleCore ι B F
        i j : ι
        x : B
        v : F
        hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
        ⊢ Membership.mem (Z.baseSet j) x
      -/
    · exact hx.2
      /-
        🎉 no goals
      -/
      /-
        case mk.a
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace B
        inst✝ : TopologicalSpace F
        Z : FiberBundleCore ι B F
        i j : ι
        x : B
        v : F
        hx : And (Membership.mem (Z.baseSet i) x) (Membership.mem (Z.baseSet j) x)
        ⊢ Membership.mem (Inter.inter (Inter.inter (Z.baseSet j) (Z.baseSet i)) (Z.bas …
      -/
    · simp [hx]
      /-
        🎉 no goals
      -/
  open_source := ((Z.isOpen_baseSet i).inter (Z.isOpen_baseSet j)).prod isOpen_univ
  open_target := ((Z.isOpen_baseSet i).inter (Z.isOpen_baseSet j)).prod isOpen_univ
  continuousOn_toFun := continuous_fst.continuousOn.prod (Z.continuousOn_coordChange i j)
  continuousOn_invFun := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      ⊢ ContinuousOn { toFun := fun p => { fst := p.1, snd := Z.coordChange i j p.1  …
    -/
    simpa [inter_comm] using continuous_fst.continuousOn.prod (Z.continuousOn_coordChange j i)
    /-
      🎉 no goals
    -/


@[simp, mfld_simps]
theorem mem_trivChange_source (i j : ι) (p : B × F) :
    p ∈ (Z.trivChange i j).source ↔ p.1 ∈ Z.baseSet i ∩ Z.baseSet j := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i j : ι
    p : Prod B F
    ⊢ Iff (Membership.mem (Z.trivChange i j).source p) (Membership.mem (Inter.inte …
  -/
  erw [mem_prod]
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i j : ι
    p : Prod B F
    ⊢ Iff (And (Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet j)) p.1) (Mem …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Associate to a trivialization index `i : ι` the corresponding trivialization, i.e., a bijection
between `proj ⁻¹ (baseSet i)` and `baseSet i × F`. As the fiber above `x` is `F` but read in the
chart with index `index_at x`, the trivialization in the fiber above x is by definition the
coordinate change from i to `index_at x`, so it depends on `x`.
The local trivialization will ultimately be a partial homeomorphism. For now, we only introduce the
partial equivalence version, denoted with a prime.
In further developments, avoid this auxiliary version, and use `Z.local_triv` instead. -/
def localTrivAsPartialEquiv (i : ι) : PartialEquiv Z.TotalSpace (B × F) where
  source := Z.proj ⁻¹' Z.baseSet i
  target := Z.baseSet i ×ˢ univ
  invFun p := ⟨p.1, Z.coordChange i (Z.indexAt p.1) p.1 p.2⟩
  toFun p := ⟨p.1, Z.coordChange (Z.indexAt p.1) i p.1 p.2⟩
  map_source' p hp := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      p : Z.TotalSpace
      hp : Membership.mem (Set.preimage Z.proj (Z.baseSet i)) p
      ⊢ Membership.mem (SProd.sprod (Z.baseSet i) Set.univ) ((fun p => { fst := p.pr …
    -/
    simpa only [Set.mem_preimage, and_true, Set.mem_univ, Set.prod_mk_mem_set_prod_eq] using hp
    /-
      🎉 no goals
    -/
  map_target' p hp := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      p : Prod B F
      hp : Membership.mem (SProd.sprod (Z.baseSet i) Set.univ) p
      ⊢ Membership.mem (Set.preimage Z.proj (Z.baseSet i)) ((fun p => { proj := p.1, …
    -/
    simpa only [Set.mem_preimage, and_true, Set.mem_univ, Set.mem_prod] using hp
    /-
      🎉 no goals
    -/
  left_inv' := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      ⊢ ∀ ⦃x : Z.TotalSpace⦄, Membership.mem (Set.preimage Z.proj (Z.baseSet i)) x → …
    -/
    rintro ⟨x, v⟩ hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : Z.Fiber x
      hx : Membership.mem (Set.preimage Z.proj (Z.baseSet i)) { proj := x, snd := v }
      ⊢ Eq ((fun p => { proj := p.1, snd := Z.coordChange i (Z.indexAt p.1) p.1 p.2  …
    -/
    replace hx : x ∈ Z.baseSet i := hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : Z.Fiber x
      hx : Membership.mem (Z.baseSet i) x
      ⊢ Eq ((fun p => { proj := p.1, snd := Z.coordChange i (Z.indexAt p.1) p.1 p.2  …
    -/
    dsimp only
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : Z.Fiber x
      hx : Membership.mem (Z.baseSet i) x
      ⊢ Eq { proj := x, snd := Z.coordChange i (Z.indexAt x) x (Z.coordChange (Z.ind …
    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    rw [Z.coordChange_comp, Z.coordChange_self] <;> apply_rules [mem_baseSet_at, mem_inter]
                                                    /-
                                                      🎉 no goals
                                                    -/
  right_inv' := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      ⊢ ∀ ⦃x : Prod B F⦄, Membership.mem (SProd.sprod (Z.baseSet i) Set.univ) x → Eq …
    -/
    rintro ⟨x, v⟩ hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : F
      hx : Membership.mem (SProd.sprod (Z.baseSet i) Set.univ) { fst := x, snd := v }
      ⊢ Eq ((fun p => { fst := p.proj, snd := Z.coordChange (Z.indexAt p.proj) i p.p …
    -/
    simp only [prod_mk_mem_set_prod_eq, and_true, mem_univ] at hx
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : F
      hx : Membership.mem (Z.baseSet i) x
      ⊢ Eq ((fun p => { fst := p.proj, snd := Z.coordChange (Z.indexAt p.proj) i p.p …
    -/
    dsimp only
    /-
      case mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : F
      hx : Membership.mem (Z.baseSet i) x
      ⊢ Eq { fst := x, snd := Z.coordChange (Z.indexAt x) i x (Z.coordChange i (Z.in …
    -/
    rw [Z.coordChange_comp, Z.coordChange_self]
    /-
      case mk.a
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      x : B
      v : F
      hx : Membership.mem (Z.baseSet i) x
      ⊢ Membership.mem (Z.baseSet i) x
    -/
    exacts [hx, ⟨⟨hx, Z.mem_baseSet_at _⟩, hx⟩]
    /-
      🎉 no goals
    -/


theorem mem_localTrivAsPartialEquiv_source (p : Z.TotalSpace) :
    p ∈ (Z.localTrivAsPartialEquiv i).source ↔ p.1 ∈ Z.baseSet i :=
  Iff.rfl


theorem mem_localTrivAsPartialEquiv_target (p : B × F) :
    p ∈ (Z.localTrivAsPartialEquiv i).target ↔ p.1 ∈ Z.baseSet i := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i : ι
    p : Prod B F
    ⊢ Iff (Membership.mem (Z.localTrivAsPartialEquiv i).target p) (Membership.mem  …
  -/
  erw [mem_prod]
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i : ι
    p : Prod B F
    ⊢ Iff (And (Membership.mem (Z.baseSet i) p.1) (Membership.mem Set.univ p.2)) ( …
  -/
  simp only [and_true, mem_univ]
  /-
    🎉 no goals
  -/


theorem localTrivAsPartialEquiv_apply (p : Z.TotalSpace) :
    (Z.localTrivAsPartialEquiv i) p = ⟨p.1, Z.coordChange (Z.indexAt p.1) i p.1 p.2⟩ :=
  rfl


/-- The composition of two local trivializations is the trivialization change Z.triv_change i j. -/
theorem localTrivAsPartialEquiv_trans (i j : ι) :
    (Z.localTrivAsPartialEquiv i).symm.trans (Z.localTrivAsPartialEquiv j) ≈
      (Z.trivChange i j).toPartialEquiv := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i j : ι
    ⊢ HasEquiv.Equiv ((Z.localTrivAsPartialEquiv i).symm.trans (Z.localTrivAsParti …
  -/
  constructor
    /-
      case left
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      ⊢ Eq ((Z.localTrivAsPartialEquiv i).symm.trans (Z.localTrivAsPartialEquiv j)). …
    -/
  · ext x
    /-
      case left.h
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : Prod B F
      ⊢ Iff (Membership.mem ((Z.localTrivAsPartialEquiv i).symm.trans (Z.localTrivAs …
    -/
    simp only [mem_localTrivAsPartialEquiv_target, mfld_simps]
    /-
      case left.h
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : Prod B F
      ⊢ Iff (And (Membership.mem (Z.baseSet i) x.1) (Membership.mem (Z.localTrivAsPa …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case right
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      ⊢ Set.EqOn (↑((Z.localTrivAsPartialEquiv i).symm.trans (Z.localTrivAsPartialEq …
    -/
  · rintro ⟨x, v⟩ hx
    simp only [trivChange, localTrivAsPartialEquiv, PartialEquiv.symm,
      Prod.mk.inj_iff, prod_mk_mem_set_prod_eq, PartialEquiv.trans_source, mem_inter_iff,
      mem_preimage, proj, mem_univ, eq_self_iff_true, (· ∘ ·),
      PartialEquiv.coe_trans, TotalSpace.proj] at hx ⊢
    /-
      case right.mk
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i j : ι
      x : B
      v : F
      hx : And (And (Membership.mem (Z.baseSet i) x) True) (Membership.mem (Z.baseSe …
      ⊢ And True (Eq (Z.coordChange (Z.indexAt x) j x (Z.coordChange i (Z.indexAt x) …
    -/
    simp only [Z.coordChange_comp, hx, mem_inter_iff, and_self_iff, mem_baseSet_at]
    /-
      🎉 no goals
    -/


/-- Topological structure on the total space of a fiber bundle created from core, designed so
that all the local trivialization are continuous. -/
instance toTopologicalSpace : TopologicalSpace (Bundle.TotalSpace F Z.Fiber) :=
  TopologicalSpace.generateFrom <| ⋃ (i : ι) (s : Set (B × F)) (_ : IsOpen s),
    {(Z.localTrivAsPartialEquiv i).source ∩ Z.localTrivAsPartialEquiv i ⁻¹' s}


theorem open_source' (i : ι) : IsOpen (Z.localTrivAsPartialEquiv i).source := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i : ι
    ⊢ IsOpen (Z.localTrivAsPartialEquiv i).source
  -/
  apply TopologicalSpace.GenerateOpen.basic
  /-
    case a
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i : ι
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun s => Set.iUnion fun x =>  …
  -/
  simp only [exists_prop, mem_iUnion, mem_singleton_iff]
  /-
    case a
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i : ι
    ⊢ Exists fun i_1 => Exists fun i_2 => And (IsOpen i_2) (Eq (Z.localTrivAsParti …
  -/
  refine ⟨i, Z.baseSet i ×ˢ univ, (Z.isOpen_baseSet i).prod isOpen_univ, ?_⟩
  /-
    case a
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    i : ι
    ⊢ Eq (Z.localTrivAsPartialEquiv i).source (Inter.inter (Z.localTrivAsPartialEq …
  -/
  ext p
  simp only [localTrivAsPartialEquiv_apply, prod_mk_mem_set_prod_eq, mem_inter_iff, and_self_iff,
    mem_localTrivAsPartialEquiv_source, and_true, mem_univ, mem_preimage]


/-- Extended version of the local trivialization of a fiber bundle constructed from core,
registering additionally in its type that it is a local bundle trivialization. -/
def localTriv (i : ι) : Trivialization F Z.proj where
  baseSet := Z.baseSet i
  open_baseSet := Z.isOpen_baseSet i
  source_eq := rfl
  target_eq := rfl
  proj_toFun p _ := by
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      p : Z.TotalSpace
      x✝ : Membership.mem { toPartialEquiv := Z.localTrivAsPartialEquiv i, open_sour …
      ⊢ Eq (↑{ toPartialEquiv := Z.localTrivAsPartialEquiv i, open_source := ⋯, open …
    -/
    simp only [mfld_simps]
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      p : Z.TotalSpace
      x✝ : Membership.mem { toPartialEquiv := Z.localTrivAsPartialEquiv i, open_sour …
      ⊢ Eq (↑(Z.localTrivAsPartialEquiv i) p).1 p.proj
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      ⊢ ContinuousOn (↑(Z.localTrivAsPartialEquiv i)) (Z.localTrivAsPartialEquiv i). …
    -/
  open_source := Z.open_source' i
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      ⊢ ∀ (t : Set (Prod B F)), IsOpen t → IsOpen (Inter.inter (Z.localTrivAsPartial …
    -/
  open_target := (Z.isOpen_baseSet i).prod isOpen_univ
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).source (Set.preimage (↑(Z. …
    -/
  continuousOn_toFun := by
    /-
      case a
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun s => Set.iUnion fun x =>  …
    -/
    rw [continuousOn_open_iff (Z.open_source' i)]
    /-
      case a
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ⊢ Exists fun i_1 => Exists fun i_2 => And (IsOpen i_2) (Eq (Inter.inter (Z.loc …
    -/
    intro s s_open
    /-
      🎉 no goals
    -/
    apply TopologicalSpace.GenerateOpen.basic
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      ⊢ ContinuousOn (Z.localTrivAsPartialEquiv i).invFun (Z.localTrivAsPartialEquiv …
    -/
    simp only [exists_prop, mem_iUnion, mem_singleton_iff]
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      ht : Membership.mem (Set.iUnion fun i => Set.iUnion fun s => Set.iUnion fun x  …
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Set.preimage (Z.lo …
    -/
    exact ⟨i, s, s_open, rfl⟩
  continuousOn_invFun := by
    refine continuousOn_isOpen_of_generateFrom fun t ht ↦ ?_
    /-
      case intro.intro.intro
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Set.preimage (Z.lo …
    -/
    simp only [exists_prop, mem_iUnion, mem_singleton_iff] at ht
    /-
      case intro.intro.intro
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Set.preimage (Z.lo …
    -/
    obtain ⟨j, s, s_open, ts⟩ : ∃ j s, IsOpen s ∧
    /-
      case intro.intro.intro
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Inter.inter (Set.p …
    -/
      t = (localTrivAsPartialEquiv Z j).source ∩ localTrivAsPartialEquiv Z j ⁻¹' s := ht
    /-
      case intro.intro.intro
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      e : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv i
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Inter.inter (Set.p …
    -/
    rw [ts]
    /-
      case intro.intro.intro
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      e : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv i
      e' : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv j
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Inter.inter (Set.p …
    -/
    simp only [PartialEquiv.right_inv, preimage_inter, PartialEquiv.left_inv]
    let e := Z.localTrivAsPartialEquiv i
    let e' := Z.localTrivAsPartialEquiv j
    let f := e.symm.trans e'
    have : IsOpen (f.source ∩ f ⁻¹' s) := by
    /-
      case intro.intro.intro
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      e : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv i
      e' : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv j
      f : PartialEquiv (Prod B F) (Prod B F) := e.symm.trans e'
      this : IsOpen (Inter.inter f.source (Set.preimage (↑f) s))
      ⊢ IsOpen (Inter.inter (Z.localTrivAsPartialEquiv i).target (Inter.inter (Set.p …
    -/
      rw [PartialEquiv.EqOnSource.source_inter_preimage_eq (Z.localTrivAsPartialEquiv_trans i j)]
    /-
      case h.e'_3
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      e : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv i
      e' : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv j
      f : PartialEquiv (Prod B F) (Prod B F) := e.symm.trans e'
      this : IsOpen (Inter.inter f.source (Set.preimage (↑f) s))
      ⊢ Eq (Inter.inter (Z.localTrivAsPartialEquiv i).target (Inter.inter (Set.preim …
    -/
      exact (continuousOn_open_iff (Z.trivChange i j).open_source).1
    /-
      case h.e'_3
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i✝ : ι
      b : B
      a : F
      i : ι
      t : Set Z.TotalSpace
      j : ι
      s : Set (Prod B F)
      s_open : IsOpen s
      ts : Eq t (Inter.inter (Z.localTrivAsPartialEquiv j).source (Set.preimage (↑(Z …
      e : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv i
      e' : PartialEquiv Z.TotalSpace (Prod B F) := Z.localTrivAsPartialEquiv j
      f : PartialEquiv (Prod B F) (Prod B F) := e.symm.trans e'
      this : IsOpen (Inter.inter f.source (Set.preimage (↑f) s))
      ⊢ Eq (Inter.inter (Z.localTrivAsPartialEquiv i).target (Inter.inter (Set.preim …
    -/
        (Z.trivChange i j).continuousOn _ s_open
    /-
      🎉 no goals
    -/
    convert this using 1
    dsimp [f, PartialEquiv.trans_source]
    rw [← preimage_comp, inter_assoc]
  toPartialEquiv := Z.localTrivAsPartialEquiv i


/-- Preferred local trivialization of a fiber bundle constructed from core, at a given point, as
a bundle trivialization -/
def localTrivAt (b : B) : Trivialization F (π F Z.Fiber) :=
  Z.localTriv (Z.indexAt b)


@[simp, mfld_simps]
theorem localTrivAt_def (b : B) : Z.localTriv (Z.indexAt b) = Z.localTrivAt b :=
  rfl


theorem localTrivAt_snd (b : B) (p) :
    (Z.localTrivAt b p).2 = Z.coordChange (Z.indexAt p.1) (Z.indexAt b) p.1 p.2 :=
  rfl


/-- If an element of `F` is invariant under all coordinate changes, then one can define a
corresponding section of the fiber bundle, which is continuous. This applies in particular to the
zero section of a vector bundle. Another example (not yet defined) would be the identity
section of the endomorphism bundle of a vector bundle. -/
theorem continuous_const_section (v : F)
    (h : ∀ i j, ∀ x ∈ Z.baseSet i ∩ Z.baseSet j, Z.coordChange i j x v = v) :
    Continuous (show B → Z.TotalSpace from fun x => ⟨x, v⟩) := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    v : F
    h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
    ⊢ Continuous (letFun (fun x => { proj := x, snd := v }) fun this => this)
  -/
  refine continuous_iff_continuousAt.2 fun x => ?_
  have A : Z.baseSet (Z.indexAt x) ∈ 𝓝 x :=
    IsOpen.mem_nhds (Z.isOpen_baseSet (Z.indexAt x)) (Z.mem_baseSet_at x)
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    v : F
    h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
    x : B
    A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
    ⊢ ContinuousAt (letFun (fun x => { proj := x, snd := v }) fun this => this) x
  -/
  refine ((Z.localTrivAt x).toPartialHomeomorph.continuousAt_iff_continuousAt_comp_left ?_).2 ?_
    /-
      case refine_1
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      v : F
      h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
      x : B
      A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
      ⊢ Membership.mem (nhds x) (Set.preimage (letFun (fun x => { proj := x, snd :=  …
    -/
  · exact A
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      v : F
      h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
      x : B
      A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
      ⊢ ContinuousAt (Function.comp (↑(Z.localTrivAt x).toPartialHomeomorph) (letFun …
    -/
  · apply continuousAt_id.prod
    /-
      case refine_2
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      v : F
      h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
      x : B
      A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
      ⊢ ContinuousAt (fun x_1 => Z.coordChange (Z.indexAt (letFun (fun x => { proj : …
    -/
    simp only [(· ∘ ·), mfld_simps, localTrivAt_snd]
    /-
      case refine_2
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      v : F
      h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
      x : B
      A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
      ⊢ ContinuousAt (fun x_1 => Z.coordChange (Z.indexAt x_1) (Z.indexAt x) x_1 v) x
    -/
    have : ContinuousOn (fun _ : B => v) (Z.baseSet (Z.indexAt x)) := continuousOn_const
    /-
      case refine_2
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      v : F
      h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
      x : B
      A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
      this : ContinuousOn (fun x => v) (Z.baseSet (Z.indexAt x))
      ⊢ ContinuousAt (fun x_1 => Z.coordChange (Z.indexAt x_1) (Z.indexAt x) x_1 v) x
    -/
    refine (this.congr fun y hy ↦ ?_).continuousAt A
    /-
      case refine_2
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      v : F
      h : ∀ (i j : ι) (x : B), Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet  …
      x : B
      A : Membership.mem (nhds x) (Z.baseSet (Z.indexAt x))
      this : ContinuousOn (fun x => v) (Z.baseSet (Z.indexAt x))
      y : B
      hy : Membership.mem (Z.baseSet (Z.indexAt x)) y
      ⊢ Eq (Z.coordChange (Z.indexAt y) (Z.indexAt x) y v) v
    -/
    exact h _ _ _ ⟨mem_baseSet_at _ _, hy⟩
    /-
      🎉 no goals
    -/


@[simp, mfld_simps]
theorem localTrivAsPartialEquiv_coe : ⇑(Z.localTrivAsPartialEquiv i) = Z.localTriv i :=
  rfl


@[simp, mfld_simps]
theorem localTrivAsPartialEquiv_source :
    (Z.localTrivAsPartialEquiv i).source = (Z.localTriv i).source :=
  rfl


@[simp, mfld_simps]
theorem localTrivAsPartialEquiv_target :
    (Z.localTrivAsPartialEquiv i).target = (Z.localTriv i).target :=
  rfl


@[simp, mfld_simps]
theorem localTrivAsPartialEquiv_symm :
    (Z.localTrivAsPartialEquiv i).symm = (Z.localTriv i).toPartialEquiv.symm :=
  rfl


@[simp, mfld_simps]
theorem baseSet_at : Z.baseSet i = (Z.localTriv i).baseSet :=
  rfl


@[simp, mfld_simps]
theorem localTriv_apply (p : Z.TotalSpace) :
    (Z.localTriv i) p = ⟨p.1, Z.coordChange (Z.indexAt p.1) i p.1 p.2⟩ :=
  rfl


@[simp, mfld_simps]
theorem localTrivAt_apply (p : Z.TotalSpace) : (Z.localTrivAt p.1) p = ⟨p.1, p.2⟩ := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    p : Z.TotalSpace
    ⊢ Eq (↑(Z.localTrivAt p.proj) p) { fst := p.proj, snd := p.snd }
  -/
  rw [localTrivAt, localTriv_apply, coordChange_self]
  /-
    case a
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    p : Z.TotalSpace
    ⊢ Membership.mem (Z.baseSet (Z.indexAt p.proj)) p.proj
  -/
  exact Z.mem_baseSet_at p.1
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem localTrivAt_apply_mk (b : B) (a : F) : (Z.localTrivAt b) ⟨b, a⟩ = ⟨b, a⟩ :=
  Z.localTrivAt_apply _


@[simp, mfld_simps]
theorem mem_localTriv_source (p : Z.TotalSpace) :
    p ∈ (Z.localTriv i).source ↔ p.1 ∈ (Z.localTriv i).baseSet :=
  Iff.rfl


@[simp, mfld_simps]
theorem mem_localTrivAt_source (p : Z.TotalSpace) (b : B) :
    p ∈ (Z.localTrivAt b).source ↔ p.1 ∈ (Z.localTrivAt b).baseSet :=
  Iff.rfl


@[simp, mfld_simps]
theorem mem_localTriv_target (p : B × F) :
    p ∈ (Z.localTriv i).target ↔ p.1 ∈ (Z.localTriv i).baseSet :=
  Trivialization.mem_target _


@[simp, mfld_simps]
theorem mem_localTrivAt_target (p : B × F) (b : B) :
    p ∈ (Z.localTrivAt b).target ↔ p.1 ∈ (Z.localTrivAt b).baseSet :=
  Trivialization.mem_target _


@[simp, mfld_simps]
theorem localTriv_symm_apply (p : B × F) :
    (Z.localTriv i).toPartialHomeomorph.symm p = ⟨p.1, Z.coordChange i (Z.indexAt p.1) p.1 p.2⟩ :=
  rfl


@[simp, mfld_simps]
theorem mem_localTrivAt_baseSet (b : B) : b ∈ (Z.localTrivAt b).baseSet := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    b : B
    ⊢ Membership.mem (Z.localTrivAt b).baseSet b
  -/
  rw [localTrivAt, ← baseSet_at]
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    b : B
    ⊢ Membership.mem (Z.baseSet (Z.indexAt b)) b
  -/
  exact Z.mem_baseSet_at b
  /-
    🎉 no goals
  -/


theorem mk_mem_localTrivAt_source : (⟨b, a⟩ : Z.TotalSpace) ∈ (Z.localTrivAt b).source := by
  /-
    ι : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    Z : FiberBundleCore ι B F
    b : B
    a : F
    ⊢ Membership.mem (Z.localTrivAt b).source { proj := b, snd := a }
  -/
  simp only [mfld_simps]
  /-
    🎉 no goals
  -/


/-- A fiber bundle constructed from core is indeed a fiber bundle. -/
instance fiberBundle : FiberBundle F Z.Fiber where
  totalSpaceMk_isInducing' b := isInducing_iff_nhds.2 fun x ↦ by
    rw [(Z.localTrivAt b).nhds_eq_comap_inf_principal (mk_mem_localTrivAt_source _ _ _), comap_inf,
      comap_principal, comap_comap]
    simp only [Function.comp_def, localTrivAt_apply_mk, Trivialization.coe_coe,
      ← (isEmbedding_prodMk b).nhds_eq_comap]
    /-
      ι : Type u_1
      B : Type u_2
      F : Type u_3
      X : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      Z : FiberBundleCore ι B F
      i : ι
      b✝ : B
      a : F
      b : B
      x : Z.Fiber b
      ⊢ Eq (nhds x) (Min.min (nhds x) (Filter.principal (Set.preimage (Bundle.TotalS …
    -/
    convert_to 𝓝 x = 𝓝 x ⊓ 𝓟 univ
      /-
        case h.e'_3
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace B
        inst✝ : TopologicalSpace F
        Z : FiberBundleCore ι B F
        i : ι
        b✝ : B
        a : F
        b : B
        x : Z.Fiber b
        ⊢ Eq (Min.min (nhds x) (Filter.principal (Set.preimage (Bundle.TotalSpace.mk b …
      -/
    · congr
      /-
        case h.e'_3.e_a.e_s
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace B
        inst✝ : TopologicalSpace F
        Z : FiberBundleCore ι B F
        i : ι
        b✝ : B
        a : F
        b : B
        x : Z.Fiber b
        ⊢ Eq (Set.preimage (Bundle.TotalSpace.mk b) (Z.localTrivAt b).source) Set.univ
      -/
      exact eq_univ_of_forall (mk_mem_localTrivAt_source Z _)
      /-
        🎉 no goals
      -/
      /-
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace B
        inst✝ : TopologicalSpace F
        Z : FiberBundleCore ι B F
        i : ι
        b✝ : B
        a : F
        b : B
        x : Z.Fiber b
        ⊢ Eq (nhds x) (Min.min (nhds x) (Filter.principal Set.univ))
      -/
    · rw [principal_univ, inf_top_eq]
      /-
        🎉 no goals
      -/
  trivializationAtlas' := Set.range Z.localTriv
  trivializationAt' := Z.localTrivAt
  mem_baseSet_trivializationAt' := Z.mem_baseSet_at
  trivialization_mem_atlas' b := ⟨Z.indexAt b, rfl⟩


/-- The inclusion of a fiber into the total space is a continuous map. -/
@[continuity]
theorem continuous_totalSpaceMk (b : B) :
    Continuous (TotalSpace.mk b : Z.Fiber b → Bundle.TotalSpace F Z.Fiber) :=
  FiberBundle.continuous_totalSpaceMk F Z.Fiber b


/-- The projection on the base of a fiber bundle created from core is continuous -/
nonrec theorem continuous_proj : Continuous Z.proj :=
  FiberBundle.continuous_proj F Z.Fiber


/-- The projection on the base of a fiber bundle created from core is an open map -/
nonrec theorem isOpenMap_proj : IsOpenMap Z.proj :=
  FiberBundle.isOpenMap_proj F Z.Fiber


/-- This structure permits to define a fiber bundle when trivializations are given as local
equivalences but there is not yet a topology on the total space. The total space is hence given a
topology in such a way that there is a fiber bundle structure for which the partial equivalences
are also partial homeomorphisms and hence local trivializations. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure FiberPrebundle where
  pretrivializationAtlas : Set (Pretrivialization F (π F E))
  pretrivializationAt : B → Pretrivialization F (π F E)
  mem_base_pretrivializationAt : ∀ x : B, x ∈ (pretrivializationAt x).baseSet
  pretrivialization_mem_atlas : ∀ x : B, pretrivializationAt x ∈ pretrivializationAtlas
  continuous_trivChange : ∀ e, e ∈ pretrivializationAtlas → ∀ e', e' ∈ pretrivializationAtlas →
    ContinuousOn (e ∘ e'.toPartialEquiv.symm) (e'.target ∩ e'.toPartialEquiv.symm ⁻¹' e.source)
  totalSpaceMk_isInducing : ∀ b : B, IsInducing (pretrivializationAt b ∘ TotalSpace.mk b)


/-- Topology on the total space that will make the prebundle into a bundle. -/
def totalSpaceTopology (a : FiberPrebundle F E) : TopologicalSpace (TotalSpace F E) :=
  ⨆ (e : Pretrivialization F (π F E)) (_ : e ∈ a.pretrivializationAtlas),
    coinduced e.setSymm instTopologicalSpaceSubtype


theorem continuous_symm_of_mem_pretrivializationAtlas (he : e ∈ a.pretrivializationAtlas) :
    @ContinuousOn _ _ _ a.totalSpaceTopology e.toPartialEquiv.symm e.target := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e : Pretrivialization F Bundle.TotalSpace.proj
    he : Membership.mem a.pretrivializationAtlas e
    ⊢ ContinuousOn (↑e.symm) e.target
  -/
  refine fun z H U h => preimage_nhdsWithin_coinduced' H (le_def.1 (nhds_mono ?_) U h)
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e : Pretrivialization F Bundle.TotalSpace.proj
    he : Membership.mem a.pretrivializationAtlas e
    z : Prod B F
    H : Membership.mem e.target z
    U : Set (Bundle.TotalSpace F E)
    h : Membership.mem (nhds (↑e.symm z)) U
    ⊢ LE.le (TopologicalSpace.coinduced (fun x => ↑e.symm ↑x) inferInstance) a.tot …
  -/
  exact le_iSup₂ (α := TopologicalSpace (TotalSpace F E)) e he
  /-
    🎉 no goals
  -/


theorem isOpen_source (e : Pretrivialization F (π F E)) :
    IsOpen[a.totalSpaceTopology] e.source := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e : Pretrivialization F Bundle.TotalSpace.proj
    ⊢ IsOpen e.source
  -/
  refine isOpen_iSup_iff.mpr fun e' => isOpen_iSup_iff.mpr fun _ => ?_
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    x✝ : Membership.mem a.pretrivializationAtlas e'
    ⊢ IsOpen e.source
  -/
  refine isOpen_coinduced.mpr (isOpen_induced_iff.mpr ⟨e.target, e.open_target, ?_⟩)
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    x✝ : Membership.mem a.pretrivializationAtlas e'
    ⊢ Eq (Set.preimage Subtype.val e.target) (Set.preimage e'.setSymm e.source)
  -/
  ext ⟨x, hx⟩
  simp only [mem_preimage, Pretrivialization.setSymm, restrict, e.mem_target, e.mem_source,
    e'.proj_symm_apply hx]


theorem isOpen_target_of_mem_pretrivializationAtlas_inter (e e' : Pretrivialization F (π F E))
    (he' : e' ∈ a.pretrivializationAtlas) :
    IsOpen (e'.toPartialEquiv.target ∩ e'.toPartialEquiv.symm ⁻¹' e.source) := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    he' : Membership.mem a.pretrivializationAtlas e'
    ⊢ IsOpen (Inter.inter e'.target (Set.preimage (↑e'.symm) e.source))
  -/
  letI := a.totalSpaceTopology
  obtain ⟨u, hu1, hu2⟩ := continuousOn_iff'.mp (a.continuous_symm_of_mem_pretrivializationAtlas he')
    e.source (a.isOpen_source e)
  /-
    case intro.intro
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    he' : Membership.mem a.pretrivializationAtlas e'
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    u : Set (Prod B F)
    hu1 : IsOpen u
    hu2 : Eq (Inter.inter (Set.preimage (↑e'.symm) e.source) e'.target) (Inter.int …
    ⊢ IsOpen (Inter.inter e'.target (Set.preimage (↑e'.symm) e.source))
  -/
  rw [inter_comm, hu2]
  /-
    case intro.intro
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    he' : Membership.mem a.pretrivializationAtlas e'
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    u : Set (Prod B F)
    hu1 : IsOpen u
    hu2 : Eq (Inter.inter (Set.preimage (↑e'.symm) e.source) e'.target) (Inter.int …
    ⊢ IsOpen (Inter.inter u e'.target)
  -/
  exact hu1.inter e'.open_target
  /-
    🎉 no goals
  -/


/-- Promotion from a `Pretrivialization` to a `Trivialization`. -/
def trivializationOfMemPretrivializationAtlas (he : e ∈ a.pretrivializationAtlas) :
    @Trivialization B F _ _ _ a.totalSpaceTopology (π F E) :=
  let _ := a.totalSpaceTopology
  { e with
    open_source := a.isOpen_source e,
    continuousOn_toFun := by
      refine continuousOn_iff'.mpr fun s hs => ⟨e ⁻¹' s ∩ e.source,
        isOpen_iSup_iff.mpr fun e' => ?_, by rw [inter_assoc, inter_self]; rfl⟩
      /-
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        ⊢ IsOpen (Inter.inter (Set.preimage (↑e) s) e.source)
      -/
      refine isOpen_iSup_iff.mpr fun he' => ?_
      /-
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        ⊢ IsOpen (Inter.inter (Set.preimage (↑e) s) e.source)
      -/
      rw [isOpen_coinduced, isOpen_induced_iff]
      /-
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage Subtype.val t) (Set.preimag …
      -/
      obtain ⟨u, hu1, hu2⟩ := continuousOn_iff'.mp (a.continuous_trivChange _ he _ he') s hs
      /-
        case intro.intro
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        u : Set (Prod B F)
        hu1 : IsOpen u
        hu2 : Eq (Inter.inter (Set.preimage (Function.comp ↑e ↑e'.symm) s) (Inter.inte …
        ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage Subtype.val t) (Set.preimag …
      -/
      have hu3 := congr_arg (fun s => (fun x : e'.target => (x : B × F)) ⁻¹' s) hu2
      /-
        case intro.intro
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        u : Set (Prod B F)
        hu1 : IsOpen u
        hu2 : Eq (Inter.inter (Set.preimage (Function.comp ↑e ↑e'.symm) s) (Inter.inte …
        hu3 : Eq ((fun s => Set.preimage (fun x => ↑x) s) (Inter.inter (Set.preimage ( …
        ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage Subtype.val t) (Set.preimag …
      -/
      simp only [Subtype.coe_preimage_self, preimage_inter, univ_inter] at hu3
      refine ⟨u ∩ e'.toPartialEquiv.target ∩ e'.toPartialEquiv.symm ⁻¹' e.source, ?_, by
        simp only [preimage_inter, inter_univ, Subtype.coe_preimage_self, hu3.symm]; rfl⟩
      /-
        case intro.intro
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        u : Set (Prod B F)
        hu1 : IsOpen u
        hu2 : Eq (Inter.inter (Set.preimage (Function.comp ↑e ↑e'.symm) s) (Inter.inte …
        hu3 : Eq (Inter.inter (Set.preimage (fun x => ↑x) (Set.preimage (Function.comp …
        ⊢ IsOpen (Inter.inter (Inter.inter u e'.target) (Set.preimage (↑e'.symm) e.sou …
      -/
      rw [inter_assoc]
      /-
        case intro.intro
        ι : Type u_1
        B : Type u_2
        F : Type u_3
        X : Type u_4
        inst✝³ : TopologicalSpace X
        E : B → Type u_5
        inst✝² : TopologicalSpace B
        inst✝¹ : TopologicalSpace F
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : FiberPrebundle F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        x✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        s : Set (Prod B F)
        hs : IsOpen s
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        u : Set (Prod B F)
        hu1 : IsOpen u
        hu2 : Eq (Inter.inter (Set.preimage (Function.comp ↑e ↑e'.symm) s) (Inter.inte …
        hu3 : Eq (Inter.inter (Set.preimage (fun x => ↑x) (Set.preimage (Function.comp …
        ⊢ IsOpen (Inter.inter u (Inter.inter e'.target (Set.preimage (↑e'.symm) e.sour …
      -/
      exact hu1.inter (a.isOpen_target_of_mem_pretrivializationAtlas_inter e e' he')
      /-
        🎉 no goals
      -/
    continuousOn_invFun := a.continuous_symm_of_mem_pretrivializationAtlas he }


theorem mem_pretrivializationAt_source (b : B) (x : E b) :
    ⟨b, x⟩ ∈ (a.pretrivializationAt b).source := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    x : E b
    ⊢ Membership.mem (a.pretrivializationAt b).source { proj := b, snd := x }
  -/
  simp only [(a.pretrivializationAt b).source_eq, mem_preimage, TotalSpace.proj]
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    x : E b
    ⊢ Membership.mem (a.pretrivializationAt b).baseSet b
  -/
  exact a.mem_base_pretrivializationAt b
  /-
    🎉 no goals
  -/


@[simp]
theorem totalSpaceMk_preimage_source (b : B) :
    TotalSpace.mk b ⁻¹' (a.pretrivializationAt b).source = univ :=
  eq_univ_of_forall (a.mem_pretrivializationAt_source b)


@[continuity]
theorem continuous_totalSpaceMk (b : B) :
    Continuous[_, a.totalSpaceTopology] (TotalSpace.mk b) := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    ⊢ Continuous (Bundle.TotalSpace.mk b)
  -/
  letI := a.totalSpaceTopology
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    ⊢ Continuous (Bundle.TotalSpace.mk b)
  -/
  let e := a.trivializationOfMemPretrivializationAtlas (a.pretrivialization_mem_atlas b)
  rw [e.toPartialHomeomorph.continuous_iff_continuous_comp_left
      (a.totalSpaceMk_preimage_source b)]
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
    ⊢ Continuous (Function.comp (↑e.toPartialHomeomorph) (Bundle.TotalSpace.mk b))
  -/
  exact continuous_iff_le_induced.2 (a.totalSpaceMk_isInducing b).eq_induced.le
  /-
    🎉 no goals
  -/


theorem inducing_totalSpaceMk_of_inducing_comp (b : B)
    (h : IsInducing (a.pretrivializationAt b ∘ TotalSpace.mk b)) :
    @IsInducing _ _ _ a.totalSpaceTopology (TotalSpace.mk b) := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    h : Topology.IsInducing (Function.comp (↑(a.pretrivializationAt b)) (Bundle.To …
    ⊢ Topology.IsInducing (Bundle.TotalSpace.mk b)
  -/
  letI := a.totalSpaceTopology
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    h : Topology.IsInducing (Function.comp (↑(a.pretrivializationAt b)) (Bundle.To …
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    ⊢ Topology.IsInducing (Bundle.TotalSpace.mk b)
  -/
  rw [← restrict_comp_codRestrict (a.mem_pretrivializationAt_source b)] at h
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    h : Topology.IsInducing (Function.comp ((a.pretrivializationAt b).source.restr …
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    ⊢ Topology.IsInducing (Bundle.TotalSpace.mk b)
  -/
  apply IsInducing.of_codRestrict (a.mem_pretrivializationAt_source b)
  refine h.of_comp ?_ (continuousOn_iff_continuous_restrict.mp
    (a.trivializationOfMemPretrivializationAtlas (a.pretrivialization_mem_atlas b)).continuousOn)
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    b : B
    h : Topology.IsInducing (Function.comp ((a.pretrivializationAt b).source.restr …
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    ⊢ Continuous (Set.codRestrict (Bundle.TotalSpace.mk b) (a.pretrivializationAt  …
  -/
  exact (a.continuous_totalSpaceMk b).codRestrict (a.mem_pretrivializationAt_source b)
  /-
    🎉 no goals
  -/


/-- Make a `FiberBundle` from a `FiberPrebundle`.  Concretely this means
that, given a `FiberPrebundle` structure for a sigma-type `E` -- which consists of a
number of "pretrivializations" identifying parts of `E` with product spaces `U × F` -- one
establishes that for the topology constructed on the sigma-type using
`FiberPrebundle.totalSpaceTopology`, these "pretrivializations" are actually
"trivializations" (i.e., homeomorphisms with respect to the constructed topology). -/
def toFiberBundle : @FiberBundle B F _ _ E a.totalSpaceTopology _ :=
  let _ := a.totalSpaceTopology
  { totalSpaceMk_isInducing' := fun b ↦ a.inducing_totalSpaceMk_of_inducing_comp b
      (a.totalSpaceMk_isInducing b)
    trivializationAtlas' :=
      { e | ∃ (e₀ : _) (he₀ : e₀ ∈ a.pretrivializationAtlas),
        e = a.trivializationOfMemPretrivializationAtlas he₀ },
    trivializationAt' := fun x ↦
      a.trivializationOfMemPretrivializationAtlas (a.pretrivialization_mem_atlas x),
    mem_baseSet_trivializationAt' := a.mem_base_pretrivializationAt
    trivialization_mem_atlas' := fun x ↦ ⟨_, a.pretrivialization_mem_atlas x, rfl⟩ }


theorem continuous_proj : @Continuous _ _ a.totalSpaceTopology _ (π F E) := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    ⊢ Continuous Bundle.TotalSpace.proj
  -/
  letI := a.totalSpaceTopology
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    ⊢ Continuous Bundle.TotalSpace.proj
  -/
  letI := a.toFiberBundle
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    this : FiberBundle F E := a.toFiberBundle
    ⊢ Continuous Bundle.TotalSpace.proj
  -/
  exact FiberBundle.continuous_proj F E
  /-
    🎉 no goals
  -/


instance {e₀} (he₀ : e₀ ∈ a.pretrivializationAtlas) :
    (letI := a.totalSpaceTopology; letI := a.toFiberBundle
      MemTrivializationAtlas (a.trivializationOfMemPretrivializationAtlas he₀)) :=
  letI := a.totalSpaceTopology; letI := a.toFiberBundle; ⟨e₀, he₀, rfl⟩


/-- For a fiber bundle `E` over `B` constructed using the `FiberPrebundle` mechanism,
continuity of a function `TotalSpace F E → X` on an open set `s` can be checked by precomposing at
each point with the pretrivialization used for the construction at that point. -/
theorem continuousOn_of_comp_right {X : Type*} [TopologicalSpace X] {f : TotalSpace F E → X}
    {s : Set B} (hs : IsOpen s) (hf : ∀ b ∈ s,
      ContinuousOn (f ∘ (a.pretrivializationAt b).toPartialEquiv.symm)
        ((s ∩ (a.pretrivializationAt b).baseSet) ×ˢ (Set.univ : Set F))) :
    @ContinuousOn _ _ a.totalSpaceTopology _ f (π F E ⁻¹' s) := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    X : Type u_6
    inst✝ : TopologicalSpace X
    f : Bundle.TotalSpace F E → X
    s : Set B
    hs : IsOpen s
    hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
    ⊢ ContinuousOn f (Set.preimage Bundle.TotalSpace.proj s)
  -/
  letI := a.totalSpaceTopology
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    X : Type u_6
    inst✝ : TopologicalSpace X
    f : Bundle.TotalSpace F E → X
    s : Set B
    hs : IsOpen s
    hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    ⊢ ContinuousOn f (Set.preimage Bundle.TotalSpace.proj s)
  -/
  intro z hz
  let e : Trivialization F (π F E) :=
    a.trivializationOfMemPretrivializationAtlas (a.pretrivialization_mem_atlas z.proj)
  refine (e.continuousAt_of_comp_right ?_
    ((hf z.proj hz).continuousAt (IsOpen.mem_nhds ?_ ?_))).continuousWithinAt
    /-
      case refine_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_5
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      a : FiberPrebundle F E
      X : Type u_6
      inst✝ : TopologicalSpace X
      f : Bundle.TotalSpace F E → X
      s : Set B
      hs : IsOpen s
      hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
      this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
      z : Bundle.TotalSpace F E
      hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
      e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
      ⊢ Membership.mem e.baseSet z.proj
    -/
  · exact a.mem_base_pretrivializationAt z.proj
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      B : Type u_2
      F : Type u_3
      E : B → Type u_5
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      a : FiberPrebundle F E
      X : Type u_6
      inst✝ : TopologicalSpace X
      f : Bundle.TotalSpace F E → X
      s : Set B
      hs : IsOpen s
      hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
      this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
      z : Bundle.TotalSpace F E
      hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
      e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
      ⊢ IsOpen (SProd.sprod (Inter.inter s (a.pretrivializationAt z.proj).baseSet) S …
    -/
  · exact (hs.inter (a.pretrivializationAt z.proj).open_baseSet).prod isOpen_univ
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    X : Type u_6
    inst✝ : TopologicalSpace X
    f : Bundle.TotalSpace F E → X
    s : Set B
    hs : IsOpen s
    hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    z : Bundle.TotalSpace F E
    hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
    e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
    ⊢ Membership.mem (SProd.sprod (Inter.inter s (a.pretrivializationAt z.proj).ba …
  -/
  refine ⟨?_, mem_univ _⟩
  /-
    case refine_3
    B : Type u_2
    F : Type u_3
    E : B → Type u_5
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : (x : B) → TopologicalSpace (E x)
    a : FiberPrebundle F E
    X : Type u_6
    inst✝ : TopologicalSpace X
    f : Bundle.TotalSpace F E → X
    s : Set B
    hs : IsOpen s
    hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
    this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
    z : Bundle.TotalSpace F E
    hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
    e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
    ⊢ Membership.mem (Inter.inter s (a.pretrivializationAt z.proj).baseSet) (↑e z).1
  -/
  rw [e.coe_fst]
    /-
      case refine_3
      B : Type u_2
      F : Type u_3
      E : B → Type u_5
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      a : FiberPrebundle F E
      X : Type u_6
      inst✝ : TopologicalSpace X
      f : Bundle.TotalSpace F E → X
      s : Set B
      hs : IsOpen s
      hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
      this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
      z : Bundle.TotalSpace F E
      hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
      e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
      ⊢ Membership.mem (Inter.inter s (a.pretrivializationAt z.proj).baseSet) z.proj
    -/
  · exact ⟨hz, a.mem_base_pretrivializationAt z.proj⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      B : Type u_2
      F : Type u_3
      E : B → Type u_5
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      a : FiberPrebundle F E
      X : Type u_6
      inst✝ : TopologicalSpace X
      f : Bundle.TotalSpace F E → X
      s : Set B
      hs : IsOpen s
      hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
      this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
      z : Bundle.TotalSpace F E
      hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
      e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
      ⊢ Membership.mem e.source z
    -/
  · rw [e.mem_source]
    /-
      case refine_3
      B : Type u_2
      F : Type u_3
      E : B → Type u_5
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      a : FiberPrebundle F E
      X : Type u_6
      inst✝ : TopologicalSpace X
      f : Bundle.TotalSpace F E → X
      s : Set B
      hs : IsOpen s
      hf : ∀ (b : B), Membership.mem s b → ContinuousOn (Function.comp f ↑(a.pretriv …
      this : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
      z : Bundle.TotalSpace F E
      hz : Membership.mem (Set.preimage Bundle.TotalSpace.proj s) z
      e : Trivialization F Bundle.TotalSpace.proj := a.trivializationOfMemPretrivial …
      ⊢ Membership.mem e.baseSet z.proj
    -/
    exact a.mem_base_pretrivializationAt z.proj
    /-
      🎉 no goals
    -/


