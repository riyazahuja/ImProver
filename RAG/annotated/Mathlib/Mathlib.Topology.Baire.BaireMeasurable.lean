/-- Notation for `=ᶠ[residual _]`. That is, eventual equality with respect to
the filter of residual sets.-/
scoped[Topology] notation:50 f " =ᵇ " g:50 => Filter.EventuallyEq (residual _) f g


/-- Notation to say that a property of points in a topological space holds
almost everywhere in the sense of Baire category. That is, on a residual set. -/
scoped[Topology] notation3 "∀ᵇ "(...)", "r:(scoped p => Filter.Eventually p <| residual _) => r


/-- Notation to say that a property of points in a topological space holds on a non meager set. -/
scoped[Topology] notation3 "∃ᵇ "(...)", "r:(scoped p => Filter.Frequently p <| residual _) => r


theorem coborder_mem_residual {s : Set α} (hs : IsLocallyClosed s) : coborder s ∈ residual α :=
  residual_of_dense_open hs.isOpen_coborder dense_coborder


theorem closure_residualEq {s : Set α} (hs : IsLocallyClosed s) : closure s =ᵇ s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsLocallyClosed s
    ⊢ (residual α).EventuallyEq (closure s) s
  -/
  rw [Filter.eventuallyEq_set]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsLocallyClosed s
    ⊢ Filter.Eventually (fun x => Iff (Membership.mem (closure s) x) (Membership.m …
  -/
  filter_upwards [coborder_mem_residual hs] with x hx
  /-
    case h
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsLocallyClosed s
    x : α
    hx : Membership.mem (coborder s) x
    ⊢ Iff (Membership.mem (closure s) x) (Membership.mem s x)
  -/
  nth_rewrite 2 [← closure_inter_coborder (s := s)]
  /-
    case h
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsLocallyClosed s
    x : α
    hx : Membership.mem (coborder s) x
    ⊢ Iff (Membership.mem (closure s) x) (Membership.mem (Inter.inter (closure s)  …
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


/-- We say a set is a `BaireMeasurableSet` if it differs from some Borel set by
a meager set. This forms a σ-algebra.

It is equivalent, and a more standard definition, to say that the set differs from
some *open* set by a meager set. See `BaireMeasurableSet.iff_residualEq_isOpen` -/
def BaireMeasurableSet (s : Set α) : Prop :=
  @MeasurableSet _ (EventuallyMeasurableSpace (borel _) (residual _)) s


theorem of_mem_residual (h : s ∈ residual _) : BaireMeasurableSet s :=
  eventuallyMeasurableSet_of_mem_filter (α := α) h


theorem _root_.MeasurableSet.baireMeasurableSet [MeasurableSpace α] [BorelSpace α]
    (h : MeasurableSet s) : BaireMeasurableSet s := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    s : Set α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    h : MeasurableSet s
    ⊢ BaireMeasurableSet s
  -/
  borelize α
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    s : Set α
    inst✝ : BorelSpace α
    h : MeasurableSet s
    this✝ : MeasurableSpace α := borel α
    ⊢ BaireMeasurableSet s
  -/
  exact h.eventuallyMeasurableSet
  /-
    🎉 no goals
  -/


theorem _root_.IsOpen.baireMeasurableSet (h : IsOpen s) : BaireMeasurableSet s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    h : IsOpen s
    ⊢ BaireMeasurableSet s
  -/
  borelize α
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    h : IsOpen s
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    ⊢ BaireMeasurableSet s
  -/
  exact h.measurableSet.baireMeasurableSet
  /-
    🎉 no goals
  -/


theorem compl (h : BaireMeasurableSet s) : BaireMeasurableSet sᶜ := MeasurableSet.compl h


theorem of_compl (h : BaireMeasurableSet sᶜ) : BaireMeasurableSet s := MeasurableSet.of_compl h


theorem _root_.IsMeagre.baireMeasurableSet (h : IsMeagre s) : BaireMeasurableSet s :=
  (of_mem_residual h).of_compl


theorem iUnion {ι : Sort*} [Countable ι] {s : ι → Set α}
    (h : ∀ i, BaireMeasurableSet (s i)) : BaireMeasurableSet (⋃ i, s i) :=
  MeasurableSet.iUnion h


theorem biUnion {ι : Type*}  {s : ι → Set α} {t : Set ι} (ht : t.Countable)
    (h : ∀ i ∈ t, BaireMeasurableSet (s i)) : BaireMeasurableSet (⋃ i ∈ t, s i) :=
  MeasurableSet.biUnion ht h


theorem sUnion {s : Set (Set α)} (hs : s.Countable)
    (h : ∀ t ∈ s, BaireMeasurableSet t) : BaireMeasurableSet (⋃₀ s) :=
  MeasurableSet.sUnion hs h


theorem iInter {ι : Sort*} [Countable ι] {s : ι → Set α}
    (h : ∀ i, BaireMeasurableSet (s i)) : BaireMeasurableSet (⋂ i, s i) :=
  MeasurableSet.iInter h


theorem biInter {ι : Type*}  {s : ι → Set α} {t : Set ι} (ht : t.Countable)
    (h : ∀ i ∈ t, BaireMeasurableSet (s i)) : BaireMeasurableSet (⋂ i ∈ t, s i) :=
  MeasurableSet.biInter ht h


theorem sInter {s : Set (Set α)} (hs : s.Countable)
    (h : ∀ t ∈ s, BaireMeasurableSet t) : BaireMeasurableSet (⋂₀ s) :=
  MeasurableSet.sInter hs h


theorem union (hs : BaireMeasurableSet s) (ht : BaireMeasurableSet t) :
    BaireMeasurableSet (s ∪ t) :=
  MeasurableSet.union hs ht


theorem inter (hs : BaireMeasurableSet s) (ht : BaireMeasurableSet t) :
    BaireMeasurableSet (s ∩ t) :=
  MeasurableSet.inter hs ht


theorem diff (hs : BaireMeasurableSet s) (ht : BaireMeasurableSet t) :
    BaireMeasurableSet (s \ t) :=
  MeasurableSet.diff hs ht


theorem congr (hs : BaireMeasurableSet s) (h : s =ᵇ t) : BaireMeasurableSet t :=
  EventuallyMeasurableSet.congr (α := α) hs h.symm


/--Any Borel set differs from some open set by a meager set. -/
theorem MeasurableSet.residualEq_isOpen [MeasurableSpace α] [BorelSpace α] (h : MeasurableSet s) :
    ∃ u : Set α, IsOpen u ∧ s =ᵇ u := by
  induction s, h using MeasurableSet.induction_on_open with
  | isOpen U hU => exact ⟨U, hU, .rfl⟩
  | compl s _ ihs =>
    obtain ⟨U, Uo, hsU⟩ := ihs
    use (closure U)ᶜ, isClosed_closure.isOpen_compl
    exact .compl <| hsU.trans <| .symm <| closure_residualEq Uo.isLocallyClosed
  | iUnion f _ _ ihf =>
    choose u uo su using ihf
    exact ⟨⋃ i, u i, isOpen_iUnion uo, EventuallyEq.countable_iUnion su⟩


/--Any `BaireMeasurableSet` differs from some open set by a meager set. -/
theorem BaireMeasurableSet.residualEq_isOpen (h : BaireMeasurableSet s) :
    ∃ u : Set α, (IsOpen u) ∧ s =ᵇ u := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    h : BaireMeasurableSet s
    ⊢ Exists fun u => And (IsOpen u) ((residual α).EventuallyEq s u)
  -/
  borelize α
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    h : BaireMeasurableSet s
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    ⊢ Exists fun u => And (IsOpen u) ((residual α).EventuallyEq s u)
  -/
  rcases h with ⟨t, ht, hst⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    t : Set α
    ht : MeasurableSet t
    hst : (residual α).EventuallyEq s t
    ⊢ Exists fun u => And (IsOpen u) ((residual α).EventuallyEq s u)
  -/
  rcases ht.residualEq_isOpen with ⟨u, hu, htu⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    t : Set α
    ht : MeasurableSet t
    hst : (residual α).EventuallyEq s t
    u : Set α
    hu : IsOpen u
    htu : (residual α).EventuallyEq t u
    ⊢ Exists fun u => And (IsOpen u) ((residual α).EventuallyEq s u)
  -/
  exact ⟨u, hu, hst.trans htu⟩
  /-
    🎉 no goals
  -/


/--A set is Baire measurable if and only if it differs from some open set by a meager set. -/
theorem BaireMeasurableSet.iff_residualEq_isOpen :
    BaireMeasurableSet s ↔ ∃ u : Set α, (IsOpen u) ∧ s =ᵇ u :=
  ⟨fun h => h.residualEq_isOpen , fun ⟨_, uo, ueq⟩ => uo.baireMeasurableSet.congr ueq.symm⟩


theorem tendsto_residual_of_isOpenMap (hc : Continuous f) (ho : IsOpenMap f) :
    Tendsto f (residual α) (residual β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    ⊢ Filter.Tendsto f (residual α) (residual β)
  -/
  apply le_countableGenerate_iff_of_countableInterFilter.mpr
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    ⊢ HasSubset.Subset (setOf fun t => And (IsOpen t) (Dense t)) (Filter.map f (re …
  -/
  rintro t ⟨ht, htd⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    t : Set β
    ht : IsOpen t
    htd : Dense t
    ⊢ Membership.mem (Filter.map f (residual α)).sets t
  -/
  exact residual_of_dense_open (ht.preimage hc) (htd.preimage ho)
  /-
    🎉 no goals
  -/


/-- The preimage of a meager set under a continuous open map is meager. -/
theorem IsMeagre.preimage_of_isOpenMap (hc : Continuous f) (ho : IsOpenMap f)
    {s : Set β} (h : IsMeagre s) : IsMeagre (f ⁻¹' s) :=
  tendsto_residual_of_isOpenMap hc ho h


/-- The preimage of a `BaireMeasurableSet` under a continuous open map is Baire measurable. -/
theorem BaireMeasurableSet.preimage (hc : Continuous f) (ho : IsOpenMap f)
    {s : Set β} (h : BaireMeasurableSet s) : BaireMeasurableSet (f⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    s : Set β
    h : BaireMeasurableSet s
    ⊢ BaireMeasurableSet (Set.preimage f s)
  -/
  rcases h with ⟨u, hu, hsu⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    s u : Set β
    hu : MeasurableSet u
    hsu : (residual β).EventuallyEq s u
    ⊢ BaireMeasurableSet (Set.preimage f s)
  -/
  refine ⟨f ⁻¹' u, ?_, hsu.filter_mono <| tendsto_residual_of_isOpenMap hc ho⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    s u : Set β
    hu : MeasurableSet u
    hsu : (residual β).EventuallyEq s u
    ⊢ MeasurableSet (Set.preimage f u)
  -/
  borelize α β
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hc : Continuous f
    ho : IsOpenMap f
    s u : Set β
    hu : MeasurableSet u
    hsu : (residual β).EventuallyEq s u
    this✝³ : MeasurableSpace α := borel α
    this✝² : BorelSpace α
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasurableSet (Set.preimage f u)
  -/
  exact hc.measurable hu
  /-
    🎉 no goals
  -/


theorem Homeomorph.residual_map_eq (h : α ≃ₜ β) : (residual α).map h = residual β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    h : Homeomorph α β
    ⊢ Eq (Filter.map (⇑h) (residual α)) (residual β)
  -/
  refine le_antisymm (tendsto_residual_of_isOpenMap h.continuous h.isOpenMap) (le_map ?_)
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    h : Homeomorph α β
    ⊢ ∀ (s : Set α), Membership.mem (residual α) s → Membership.mem (residual β) ( …
  -/
  simp_rw [← preimage_symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    h : Homeomorph α β
    ⊢ ∀ (s : Set α), Membership.mem (residual α) s → Membership.mem (residual β) ( …
  -/
  exact tendsto_residual_of_isOpenMap h.symm.continuous h.symm.isOpenMap
  /-
    🎉 no goals
  -/


