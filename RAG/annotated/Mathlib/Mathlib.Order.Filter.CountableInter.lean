/-- A filter `l` has the countable intersection property if for any countable collection
of sets `s ∈ l` their intersection belongs to `l` as well. -/
class CountableInterFilter (l : Filter α) : Prop where
  /-- For a countable collection of sets `s ∈ l`, their intersection belongs to `l` as well. -/
  countable_sInter_mem : ∀ S : Set (Set α), S.Countable → (∀ s ∈ S, s ∈ l) → ⋂₀ S ∈ l


theorem countable_sInter_mem {S : Set (Set α)} (hSc : S.Countable) : ⋂₀ S ∈ l ↔ ∀ s ∈ S, s ∈ l :=
  ⟨fun hS _s hs => mem_of_superset hS (sInter_subset_of_mem hs),
    CountableInterFilter.countable_sInter_mem _ hSc⟩


theorem countable_iInter_mem [Countable ι] {s : ι → Set α} : (⋂ i, s i) ∈ l ↔ ∀ i, s i ∈ l :=
  sInter_range s ▸ (countable_sInter_mem (countable_range _)).trans forall_mem_range


theorem countable_bInter_mem {ι : Type*} {S : Set ι} (hS : S.Countable) {s : ∀ i ∈ S, Set α} :
    (⋂ i, ⋂ hi : i ∈ S, s i ‹_›) ∈ l ↔ ∀ i, ∀ hi : i ∈ S, s i ‹_› ∈ l := by
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s : (i : ι) → Membership.mem S i → Set α
    ⊢ Iff (Membership.mem l (Set.iInter fun i => Set.iInter fun hi => s i hi)) (∀  …
  -/
  rw [biInter_eq_iInter]
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s : (i : ι) → Membership.mem S i → Set α
    ⊢ Iff (Membership.mem l (Set.iInter fun x => s ↑x ⋯)) (∀ (i : ι) (hi : Members …
  -/
  haveI := hS.toEncodable
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s : (i : ι) → Membership.mem S i → Set α
    this : Encodable ↑S
    ⊢ Iff (Membership.mem l (Set.iInter fun x => s ↑x ⋯)) (∀ (i : ι) (hi : Members …
  -/
  exact countable_iInter_mem.trans Subtype.forall
  /-
    🎉 no goals
  -/


theorem eventually_countable_forall [Countable ι] {p : α → ι → Prop} :
    (∀ᶠ x in l, ∀ i, p x i) ↔ ∀ i, ∀ᶠ x in l, p x i := by
  simpa only [Filter.Eventually, setOf_forall] using
    @countable_iInter_mem _ _ l _ _ fun i => { x | p x i }


theorem eventually_countable_ball {ι : Type*} {S : Set ι} (hS : S.Countable)
    {p : α → ∀ i ∈ S, Prop} :
    (∀ᶠ x in l, ∀ i hi, p x i hi) ↔ ∀ i hi, ∀ᶠ x in l, p x i hi := by
  simpa only [Filter.Eventually, setOf_forall] using
    @countable_bInter_mem _ l _ _ _ hS fun i hi => { x | p x i hi }


theorem EventuallyLE.countable_iUnion [Countable ι] {s t : ι → Set α} (h : ∀ i, s i ≤ᶠ[l] t i) :
    ⋃ i, s i ≤ᶠ[l] ⋃ i, t i :=
  (eventually_countable_forall.2 h).mono fun _ hst hs => mem_iUnion.2 <| (mem_iUnion.1 hs).imp hst


theorem EventuallyEq.countable_iUnion [Countable ι] {s t : ι → Set α} (h : ∀ i, s i =ᶠ[l] t i) :
    ⋃ i, s i =ᶠ[l] ⋃ i, t i :=
  (EventuallyLE.countable_iUnion fun i => (h i).le).antisymm
    (EventuallyLE.countable_iUnion fun i => (h i).symm.le)


theorem EventuallyLE.countable_bUnion {ι : Type*} {S : Set ι} (hS : S.Countable)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi ≤ᶠ[l] t i hi) :
    ⋃ i ∈ S, s i ‹_› ≤ᶠ[l] ⋃ i ∈ S, t i ‹_› := by
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iUnion fun i => Set.iUnion fun h => s i h) (Set.iUnion f …
  -/
  simp only [biUnion_eq_iUnion]
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iUnion fun x => s ↑x ⋯) (Set.iUnion fun x => t ↑x ⋯)
  -/
  haveI := hS.toEncodable
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    this : Encodable ↑S
    ⊢ l.EventuallyLE (Set.iUnion fun x => s ↑x ⋯) (Set.iUnion fun x => t ↑x ⋯)
  -/
  exact EventuallyLE.countable_iUnion fun i => h i i.2
  /-
    🎉 no goals
  -/


theorem EventuallyEq.countable_bUnion {ι : Type*} {S : Set ι} (hS : S.Countable)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi =ᶠ[l] t i hi) :
    ⋃ i ∈ S, s i ‹_› =ᶠ[l] ⋃ i ∈ S, t i ‹_› :=
  (EventuallyLE.countable_bUnion hS fun i hi => (h i hi).le).antisymm
    (EventuallyLE.countable_bUnion hS fun i hi => (h i hi).symm.le)


theorem EventuallyLE.countable_iInter [Countable ι] {s t : ι → Set α} (h : ∀ i, s i ≤ᶠ[l] t i) :
    ⋂ i, s i ≤ᶠ[l] ⋂ i, t i :=
  (eventually_countable_forall.2 h).mono fun _ hst hs =>
    mem_iInter.2 fun i => hst _ (mem_iInter.1 hs i)


theorem EventuallyEq.countable_iInter [Countable ι] {s t : ι → Set α} (h : ∀ i, s i =ᶠ[l] t i) :
    ⋂ i, s i =ᶠ[l] ⋂ i, t i :=
  (EventuallyLE.countable_iInter fun i => (h i).le).antisymm
    (EventuallyLE.countable_iInter fun i => (h i).symm.le)


theorem EventuallyLE.countable_bInter {ι : Type*} {S : Set ι} (hS : S.Countable)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi ≤ᶠ[l] t i hi) :
    ⋂ i ∈ S, s i ‹_› ≤ᶠ[l] ⋂ i ∈ S, t i ‹_› := by
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iInter fun i => Set.iInter fun h => s i h) (Set.iInter f …
  -/
  simp only [biInter_eq_iInter]
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iInter fun x => s ↑x ⋯) (Set.iInter fun x => t ↑x ⋯)
  -/
  haveI := hS.toEncodable
  /-
    α : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    ι : Type u_4
    S : Set ι
    hS : S.Countable
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    this : Encodable ↑S
    ⊢ l.EventuallyLE (Set.iInter fun x => s ↑x ⋯) (Set.iInter fun x => t ↑x ⋯)
  -/
  exact EventuallyLE.countable_iInter fun i => h i i.2
  /-
    🎉 no goals
  -/


theorem EventuallyEq.countable_bInter {ι : Type*} {S : Set ι} (hS : S.Countable)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi =ᶠ[l] t i hi) :
    ⋂ i ∈ S, s i ‹_› =ᶠ[l] ⋂ i ∈ S, t i ‹_› :=
  (EventuallyLE.countable_bInter hS fun i hi => (h i hi).le).antisymm
    (EventuallyLE.countable_bInter hS fun i hi => (h i hi).symm.le)


/-- Construct a filter with countable intersection property. This constructor deduces
`Filter.univ_sets` and `Filter.inter_sets` from the countable intersection property. -/
def Filter.ofCountableInter (l : Set (Set α))
    (hl : ∀ S : Set (Set α), S.Countable → S ⊆ l → ⋂₀ S ∈ l)
    (h_mono : ∀ s t, s ∈ l → s ⊆ t → t ∈ l) : Filter α where
  sets := l
  univ_sets := @sInter_empty α ▸ hl _ countable_empty (empty_subset _)
  sets_of_superset := h_mono _ _
  inter_sets {s t} hs ht := sInter_pair s t ▸
    hl _ ((countable_singleton _).insert _) (insert_subset_iff.2 ⟨hs, singleton_subset_iff.2 ht⟩)


instance Filter.countableInter_ofCountableInter (l : Set (Set α))
    (hl : ∀ S : Set (Set α), S.Countable → S ⊆ l → ⋂₀ S ∈ l)
    (h_mono : ∀ s t, s ∈ l → s ⊆ t → t ∈ l) :
    CountableInterFilter (Filter.ofCountableInter l hl h_mono) :=
  ⟨hl⟩


@[simp]
theorem Filter.mem_ofCountableInter {l : Set (Set α)}
    (hl : ∀ S : Set (Set α), S.Countable → S ⊆ l → ⋂₀ S ∈ l) (h_mono : ∀ s t, s ∈ l → s ⊆ t → t ∈ l)
    {s : Set α} : s ∈ Filter.ofCountableInter l hl h_mono ↔ s ∈ l :=
  Iff.rfl


/-- Construct a filter with countable intersection property.
Similarly to `Filter.comk`, a set belongs to this filter if its complement satisfies the property.
Similarly to `Filter.ofCountableInter`,
this constructor deduces some properties from the countable intersection property
which becomes the countable union property because we take complements of all sets. -/
def Filter.ofCountableUnion (l : Set (Set α))
    (hUnion : ∀ S : Set (Set α), S.Countable → (∀ s ∈ S, s ∈ l) → ⋃₀ S ∈ l)
    (hmono : ∀ t ∈ l, ∀ s ⊆ t, s ∈ l) : Filter α := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝ : CountableInterFilter l✝
    l : Set (Set α)
    hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
    hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
    ⊢ Filter α
  -/
  refine .ofCountableInter {s | sᶜ ∈ l} (fun S hSc hSp ↦ ?_) fun s t ht hsub ↦ ?_
    /-
      case refine_1
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : S.Countable
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      ⊢ Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) S.sInter
    -/
  · rw [mem_setOf_eq, compl_sInter]
    /-
      case refine_1
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : S.Countable
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      ⊢ Membership.mem l (Set.image HasCompl.compl S).sUnion
    -/
    apply hUnion (compl '' S) (hSc.image _)
    /-
      case refine_1
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : S.Countable
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      ⊢ ∀ (s : Set α), Membership.mem (Set.image HasCompl.compl S) s → Membership.me …
    -/
    intro s hs
    /-
      case refine_1
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : S.Countable
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      s : Set α
      hs : Membership.mem (Set.image HasCompl.compl S) s
      ⊢ Membership.mem l s
    -/
    rw [mem_image] at hs
    /-
      case refine_1
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : S.Countable
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      s : Set α
      hs : Exists fun x => And (Membership.mem S x) (Eq (HasCompl.compl x) s)
      ⊢ Membership.mem l s
    -/
    rcases hs with ⟨t, ht, rfl⟩
    /-
      case refine_1.intro.intro
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : S.Countable
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      t : Set α
      ht : Membership.mem S t
      ⊢ Membership.mem l (HasCompl.compl t)
    -/
    apply hSp ht
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      s t : Set α
      ht : Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) s
      hsub : HasSubset.Subset s t
      ⊢ Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) t
    -/
  · rw [mem_setOf_eq]
    /-
      case refine_2
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      s t : Set α
      ht : Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) s
      hsub : HasSubset.Subset s t
      ⊢ Membership.mem l (HasCompl.compl t)
    -/
    rw [← compl_subset_compl] at hsub
    /-
      case refine_2
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      l✝ : Filter α
      inst✝ : CountableInterFilter l✝
      l : Set (Set α)
      hUnion : ∀ (S : Set (Set α)), S.Countable → (∀ (s : Set α), Membership.mem S s …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      s t : Set α
      ht : Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) s
      hsub : HasSubset.Subset (HasCompl.compl t) (HasCompl.compl s)
      ⊢ Membership.mem l (HasCompl.compl t)
    -/
    exact hmono sᶜ ht tᶜ hsub
    /-
      🎉 no goals
    -/


instance Filter.countableInter_ofCountableUnion (l : Set (Set α)) (h₁ h₂) :
    CountableInterFilter (Filter.ofCountableUnion l h₁ h₂) :=
  countableInter_ofCountableInter ..


@[simp]
theorem Filter.mem_ofCountableUnion {l : Set (Set α)} {hunion hmono s} :
    s ∈ ofCountableUnion l hunion hmono ↔ l sᶜ :=
  Iff.rfl


instance countableInterFilter_principal (s : Set α) : CountableInterFilter (𝓟 s) :=
  ⟨fun _ _ hS => subset_sInter hS⟩


instance countableInterFilter_bot : CountableInterFilter (⊥ : Filter α) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝ : CountableInterFilter l
    ⊢ CountableInterFilter Bot.bot
  -/
  rw [← principal_empty]
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝ : CountableInterFilter l
    ⊢ CountableInterFilter (Filter.principal EmptyCollection.emptyCollection)
  -/
  apply countableInterFilter_principal
  /-
    🎉 no goals
  -/


instance countableInterFilter_top : CountableInterFilter (⊤ : Filter α) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝ : CountableInterFilter l
    ⊢ CountableInterFilter Top.top
  -/
  rw [← principal_univ]
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝ : CountableInterFilter l
    ⊢ CountableInterFilter (Filter.principal Set.univ)
  -/
  apply countableInterFilter_principal
  /-
    🎉 no goals
  -/


instance (l : Filter β) [CountableInterFilter l] (f : α → β) :
    CountableInterFilter (comap f l) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter β
    inst✝ : CountableInterFilter l
    f : α → β
    ⊢ CountableInterFilter (Filter.comap f l)
  -/
  refine ⟨fun S hSc hS => ?_⟩
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter β
    inst✝ : CountableInterFilter l
    f : α → β
    S : Set (Set α)
    hSc : S.Countable
    hS : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.comap f l) s
    ⊢ Membership.mem (Filter.comap f l) S.sInter
  -/
  choose! t htl ht using hS
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter β
    inst✝ : CountableInterFilter l
    f : α → β
    S : Set (Set α)
    hSc : S.Countable
    t : Set α → Set β
    htl : ∀ (s : Set α), Membership.mem S s → Membership.mem l (t s)
    ht : ∀ (s : Set α), Membership.mem S s → HasSubset.Subset (Set.preimage f (t s …
    ⊢ Membership.mem (Filter.comap f l) S.sInter
  -/
  have : (⋂ s ∈ S, t s) ∈ l := (countable_bInter_mem hSc).2 htl
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter β
    inst✝ : CountableInterFilter l
    f : α → β
    S : Set (Set α)
    hSc : S.Countable
    t : Set α → Set β
    htl : ∀ (s : Set α), Membership.mem S s → Membership.mem l (t s)
    ht : ∀ (s : Set α), Membership.mem S s → HasSubset.Subset (Set.preimage f (t s …
    this : Membership.mem l (Set.iInter fun s => Set.iInter fun h => t s)
    ⊢ Membership.mem (Filter.comap f l) S.sInter
  -/
  refine ⟨_, this, ?_⟩
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter β
    inst✝ : CountableInterFilter l
    f : α → β
    S : Set (Set α)
    hSc : S.Countable
    t : Set α → Set β
    htl : ∀ (s : Set α), Membership.mem S s → Membership.mem l (t s)
    ht : ∀ (s : Set α), Membership.mem S s → HasSubset.Subset (Set.preimage f (t s …
    this : Membership.mem l (Set.iInter fun s => Set.iInter fun h => t s)
    ⊢ HasSubset.Subset (Set.preimage f (Set.iInter fun s => Set.iInter fun h => t  …
  -/
  simpa [preimage_iInter] using iInter₂_mono ht
  /-
    🎉 no goals
  -/


instance (l : Filter α) [CountableInterFilter l] (f : α → β) : CountableInterFilter (map f l) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter α
    inst✝ : CountableInterFilter l
    f : α → β
    ⊢ CountableInterFilter (Filter.map f l)
  -/
  refine ⟨fun S hSc hS => ?_⟩
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter α
    inst✝ : CountableInterFilter l
    f : α → β
    S : Set (Set β)
    hSc : S.Countable
    hS : ∀ (s : Set β), Membership.mem S s → Membership.mem (Filter.map f l) s
    ⊢ Membership.mem (Filter.map f l) S.sInter
  -/
  simp only [mem_map, sInter_eq_biInter, preimage_iInter₂] at hS ⊢
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l✝ : Filter α
    inst✝¹ : CountableInterFilter l✝
    l : Filter α
    inst✝ : CountableInterFilter l
    f : α → β
    S : Set (Set β)
    hSc : S.Countable
    hS : ∀ (s : Set β), Membership.mem S s → Membership.mem l (Set.preimage f s)
    ⊢ Membership.mem l (Set.iInter fun i => Set.iInter fun j => Set.preimage f i)
  -/
  exact (countable_bInter_mem hSc).2 hS
  /-
    🎉 no goals
  -/


/-- Infimum of two `CountableInterFilter`s is a `CountableInterFilter`. This is useful, e.g.,
to automatically get an instance for `residual α ⊓ 𝓟 s`. -/
instance countableInterFilter_inf (l₁ l₂ : Filter α) [CountableInterFilter l₁]
    [CountableInterFilter l₂] : CountableInterFilter (l₁ ⊓ l₂) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    ⊢ CountableInterFilter (Min.min l₁ l₂)
  -/
  refine ⟨fun S hSc hS => ?_⟩
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    hS : ∀ (s : Set α), Membership.mem S s → Membership.mem (Min.min l₁ l₂) s
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  choose s hs t ht hst using hS
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    s : (s : Set α) → Membership.mem S s → Set α
    hs : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Membership.mem l₁ (s s_1 a)
    t : (s : Set α) → Membership.mem S s → Set α
    ht : ∀ (s : Set α) (a : Membership.mem S s), Membership.mem l₂ (t s a)
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  replace hs : (⋂ i ∈ S, s i ‹_›) ∈ l₁ := (countable_bInter_mem hSc).2 hs
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    s t : (s : Set α) → Membership.mem S s → Set α
    ht : ∀ (s : Set α) (a : Membership.mem S s), Membership.mem l₂ (t s a)
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  replace ht : (⋂ i ∈ S, t i ‹_›) ∈ l₂ := (countable_bInter_mem hSc).2 ht
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    s t : (s : Set α) → Membership.mem S s → Set α
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ht : Membership.mem l₂ (Set.iInter fun i => Set.iInter fun h => t i h)
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  refine mem_of_superset (inter_mem_inf hs ht) (subset_sInter fun i hi => ?_)
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    s t : (s : Set α) → Membership.mem S s → Set α
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ht : Membership.mem l₂ (Set.iInter fun i => Set.iInter fun h => t i h)
    i : Set α
    hi : Membership.mem S i
    ⊢ HasSubset.Subset (Inter.inter (Set.iInter fun i => Set.iInter fun h => s i h …
  -/
  rw [hst i hi]
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    s t : (s : Set α) → Membership.mem S s → Set α
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ht : Membership.mem l₂ (Set.iInter fun i => Set.iInter fun h => t i h)
    i : Set α
    hi : Membership.mem S i
    ⊢ HasSubset.Subset (Inter.inter (Set.iInter fun i => Set.iInter fun h => s i h …
  -/
                               /-
                                 🎉 no goals
                               -/
  apply inter_subset_inter <;> exact iInter_subset_of_subset i (iInter_subset _ _)
                               /-
                                 🎉 no goals
                               -/


/-- Supremum of two `CountableInterFilter`s is a `CountableInterFilter`. -/
instance countableInterFilter_sup (l₁ l₂ : Filter α) [CountableInterFilter l₁]
    [CountableInterFilter l₂] : CountableInterFilter (l₁ ⊔ l₂) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    ⊢ CountableInterFilter (Max.max l₁ l₂)
  -/
  refine ⟨fun S hSc hS => ⟨?_, ?_⟩⟩ <;> refine (countable_sInter_mem hSc).2 fun s hs => ?_
  /-
    case refine_1
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝² : CountableInterFilter l
    l₁ l₂ : Filter α
    inst✝¹ : CountableInterFilter l₁
    inst✝ : CountableInterFilter l₂
    S : Set (Set α)
    hSc : S.Countable
    hS : ∀ (s : Set α), Membership.mem S s → Membership.mem (Max.max l₁ l₂) s
    s : Set α
    hs : Membership.mem S s
    ⊢ Membership.mem l₁ s
  -/
  exacts [(hS s hs).1, (hS s hs).2]
  /-
    🎉 no goals
  -/


instance CountableInterFilter.curry {α β : Type*} {l : Filter α} {m : Filter β}
    [CountableInterFilter l] [CountableInterFilter m] : CountableInterFilter (l.curry m) := ⟨by
  /-
    ι : Sort u_1
    α✝ : Type u_2
    β✝ : Type u_3
    l✝ : Filter α✝
    inst✝² : CountableInterFilter l✝
    α : Type u_4
    β : Type u_5
    l : Filter α
    m : Filter β
    inst✝¹ : CountableInterFilter l
    inst✝ : CountableInterFilter m
    ⊢ ∀ (S : Set (Set (Prod α β))), S.Countable → (∀ (s : Set (Prod α β)), Members …
  -/
  intro S Sct hS
  simp_rw [mem_curry_iff, mem_sInter, eventually_countable_ball (p := fun _ _ _ => (_ ,_) ∈ _) Sct,
    eventually_countable_ball (p := fun _ _ _ => ∀ᶠ (_ : β) in m, _)  Sct, ← mem_curry_iff]
  /-
    ι : Sort u_1
    α✝ : Type u_2
    β✝ : Type u_3
    l✝ : Filter α✝
    inst✝² : CountableInterFilter l✝
    α : Type u_4
    β : Type u_5
    l : Filter α
    m : Filter β
    inst✝¹ : CountableInterFilter l
    inst✝ : CountableInterFilter m
    S : Set (Set (Prod α β))
    Sct : S.Countable
    hS : ∀ (s : Set (Prod α β)), Membership.mem S s → Membership.mem (l.curry m) s
    ⊢ ∀ (i : Set (Prod α β)), Membership.mem S i → Membership.mem (l.curry m) i
  -/
  exact hS⟩
  /-
    🎉 no goals
  -/


/-- `Filter.CountableGenerateSets g` is the (sets of the)
greatest `countableInterFilter` containing `g`. -/
inductive CountableGenerateSets : Set α → Prop
  | basic {s : Set α} : s ∈ g → CountableGenerateSets s
  | univ : CountableGenerateSets univ
  | superset {s t : Set α} : CountableGenerateSets s → s ⊆ t → CountableGenerateSets t
  | sInter {S : Set (Set α)} :
    S.Countable → (∀ s ∈ S, CountableGenerateSets s) → CountableGenerateSets (⋂₀ S)


/-- `Filter.countableGenerate g` is the greatest `countableInterFilter` containing `g`. -/
def countableGenerate : Filter α :=
  ofCountableInter (CountableGenerateSets g) (fun _ => CountableGenerateSets.sInter) fun _ _ =>
    CountableGenerateSets.superset
  --deriving CountableInterFilter

-- Porting note: could not de derived

instance : CountableInterFilter (countableGenerate g) := by
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    l : Filter α
    inst✝ : CountableInterFilter l
    g : Set (Set α)
    ⊢ CountableInterFilter (Filter.countableGenerate g)
  -/
  delta countableGenerate; infer_instance
                           /-
                             🎉 no goals
                           -/


/-- A set is in the `countableInterFilter` generated by `g` if and only if
it contains a countable intersection of elements of `g`. -/
theorem mem_countableGenerate_iff {s : Set α} :
    s ∈ countableGenerate g ↔ ∃ S : Set (Set α), S ⊆ g ∧ S.Countable ∧ ⋂₀ S ⊆ s := by
  /-
    α : Type u_2
    g : Set (Set α)
    s : Set α
    ⊢ Iff (Membership.mem (Filter.countableGenerate g) s) (Exists fun S => And (Ha …
  -/
  constructor <;> intro h
  · induction h with
    | @basic s hs => exact ⟨{s}, by simp [hs, subset_refl]⟩
    | univ => exact ⟨∅, by simp⟩
    | superset _ _ ih => refine Exists.imp (fun S => ?_) ih; tauto
    | @sInter S Sct _ ih =>
      choose T Tg Tct hT using ih
      refine ⟨⋃ (s) (H : s ∈ S), T s H, by simpa, Sct.biUnion Tct, ?_⟩
      apply subset_sInter
      intro s H
      exact subset_trans (sInter_subset_sInter (subset_iUnion₂ s H)) (hT s H)
  /-
    case mpr
    α : Type u_2
    g : Set (Set α)
    s : Set α
    h : Exists fun S => And (HasSubset.Subset S g) (And S.Countable (HasSubset.Sub …
    ⊢ Membership.mem (Filter.countableGenerate g) s
  -/
  rcases h with ⟨S, Sg, Sct, hS⟩
  /-
    case mpr.intro.intro.intro
    α : Type u_2
    g : Set (Set α)
    s : Set α
    S : Set (Set α)
    Sg : HasSubset.Subset S g
    Sct : S.Countable
    hS : HasSubset.Subset S.sInter s
    ⊢ Membership.mem (Filter.countableGenerate g) s
  -/
  refine mem_of_superset ((countable_sInter_mem Sct).mpr ?_) hS
  /-
    case mpr.intro.intro.intro
    α : Type u_2
    g : Set (Set α)
    s : Set α
    S : Set (Set α)
    Sg : HasSubset.Subset S g
    Sct : S.Countable
    hS : HasSubset.Subset S.sInter s
    ⊢ ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.countableGenerate …
  -/
  intro s H
  /-
    case mpr.intro.intro.intro
    α : Type u_2
    g : Set (Set α)
    s✝ : Set α
    S : Set (Set α)
    Sg : HasSubset.Subset S g
    Sct : S.Countable
    hS : HasSubset.Subset S.sInter s✝
    s : Set α
    H : Membership.mem S s
    ⊢ Membership.mem (Filter.countableGenerate g) s
  -/
  exact CountableGenerateSets.basic (Sg H)
  /-
    🎉 no goals
  -/


theorem le_countableGenerate_iff_of_countableInterFilter {f : Filter α} [CountableInterFilter f] :
    f ≤ countableGenerate g ↔ g ⊆ f.sets := by
  /-
    α : Type u_2
    g : Set (Set α)
    f : Filter α
    inst✝ : CountableInterFilter f
    ⊢ Iff (LE.le f (Filter.countableGenerate g)) (HasSubset.Subset g f.sets)
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_2
      g : Set (Set α)
      f : Filter α
      inst✝ : CountableInterFilter f
      h : LE.le f (Filter.countableGenerate g)
      ⊢ HasSubset.Subset g f.sets
    -/
  · exact subset_trans (fun s => CountableGenerateSets.basic) h
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_2
    g : Set (Set α)
    f : Filter α
    inst✝ : CountableInterFilter f
    h : HasSubset.Subset g f.sets
    ⊢ LE.le f (Filter.countableGenerate g)
  -/
  intro s hs
  induction hs with
  | basic hs => exact h hs
  | univ => exact univ_mem
  | superset _ st ih => exact mem_of_superset ih st
  | sInter Sct _ ih => exact (countable_sInter_mem Sct).mpr ih


/-- `countableGenerate g` is the greatest `countableInterFilter` containing `g`. -/
theorem countableGenerate_isGreatest :
    IsGreatest { f : Filter α | CountableInterFilter f ∧ g ⊆ f.sets } (countableGenerate g) := by
  /-
    α : Type u_2
    g : Set (Set α)
    ⊢ IsGreatest (setOf fun f => And (CountableInterFilter f) (HasSubset.Subset g  …
  -/
  refine ⟨⟨inferInstance, fun s => CountableGenerateSets.basic⟩, ?_⟩
  /-
    α : Type u_2
    g : Set (Set α)
    ⊢ Membership.mem (upperBounds (setOf fun f => And (CountableInterFilter f) (Ha …
  -/
  rintro f ⟨fct, hf⟩
  /-
    case intro
    α : Type u_2
    g : Set (Set α)
    f : Filter α
    fct : CountableInterFilter f
    hf : HasSubset.Subset g f.sets
    ⊢ LE.le f (Filter.countableGenerate g)
  -/
  rwa [@le_countableGenerate_iff_of_countableInterFilter _ _ _ fct]
  /-
    🎉 no goals
  -/


