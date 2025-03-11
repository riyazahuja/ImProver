/-- A set `s` is Lindelöf if every nontrivial filter `f` with the countable intersection
  property that contains `s`, has a clusterpoint in `s`. The filter-free definition is given by
  `isLindelof_iff_countable_subcover`. -/
def IsLindelof (s : Set X) :=
  ∀ ⦃f⦄ [NeBot f] [CountableInterFilter f], f ≤ 𝓟 s → ∃ x ∈ s, ClusterPt x f


/-- The complement to a Lindelöf set belongs to a filter `f` with the countable intersection
  property if it belongs to each filter `𝓝 x ⊓ f`, `x ∈ s`. -/
theorem IsLindelof.compl_mem_sets (hs : IsLindelof s) {f : Filter X} [CountableInterFilter f]
    (hf : ∀ x ∈ s, sᶜ ∈ 𝓝 x ⊓ f) : sᶜ ∈ f := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    f : Filter X
    inst✝ : CountableInterFilter f
    hf : ∀ (x : X), Membership.mem s x → Membership.mem (Min.min (nhds x) f) (HasC …
    ⊢ Membership.mem f (HasCompl.compl s)
  -/
  contrapose! hf
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    f : Filter X
    inst✝ : CountableInterFilter f
    hf : Not (Membership.mem f (HasCompl.compl s))
    ⊢ Exists fun x => And (Membership.mem s x) (Not (Membership.mem (Min.min (nhds …
  -/
  simp only [not_mem_iff_inf_principal_compl, compl_compl, inf_assoc] at hf ⊢
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    f : Filter X
    inst✝ : CountableInterFilter f
    hf : (Min.min f (Filter.principal s)).NeBot
    ⊢ Exists fun x => And (Membership.mem s x) (Min.min (nhds x) (Min.min f (Filte …
  -/
  exact hs inf_le_right
  /-
    🎉 no goals
  -/


/-- The complement to a Lindelöf set belongs to a filter `f` with the countable intersection
  property if each `x ∈ s` has a neighborhood `t` within `s` such that `tᶜ` belongs to `f`. -/
theorem IsLindelof.compl_mem_sets_of_nhdsWithin (hs : IsLindelof s) {f : Filter X}
    [CountableInterFilter f] (hf : ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, tᶜ ∈ f) : sᶜ ∈ f := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    f : Filter X
    inst✝ : CountableInterFilter f
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    ⊢ Membership.mem f (HasCompl.compl s)
  -/
  refine hs.compl_mem_sets fun x hx ↦ ?_
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    f : Filter X
    inst✝ : CountableInterFilter f
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x : X
    hx : Membership.mem s x
    ⊢ Membership.mem (Min.min (nhds x) f) (HasCompl.compl s)
  -/
  rw [← disjoint_principal_right, disjoint_right_comm, (basis_sets _).disjoint_iff_left]
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    f : Filter X
    inst✝ : CountableInterFilter f
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun i => And (Membership.mem (Min.min (nhds x) (Filter.principal s))  …
  -/
  exact hf x hx
  /-
    🎉 no goals
  -/


/-- If `p : Set X → Prop` is stable under restriction and union, and each point `x`
  of a Lindelöf set `s` has a neighborhood `t` within `s` such that `p t`, then `p s` holds. -/
@[elab_as_elim]
theorem IsLindelof.induction_on (hs : IsLindelof s) {p : Set X → Prop}
    (hmono : ∀ ⦃s t⦄, s ⊆ t → p t → p s)
    (hcountable_union : ∀ (S : Set (Set X)), S.Countable → (∀ s ∈ S, p s) → p (⋃₀ S))
    (hnhds : ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, p t) : p s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    p : Set X → Prop
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → p t → p s
    hcountable_union : ∀ (S : Set (Set X)), S.Countable → (∀ (s : Set X), Membersh …
    hnhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (n …
    ⊢ p s
  -/
  let f : Filter X := ofCountableUnion p hcountable_union (fun t ht _ hsub ↦ hmono hsub ht)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    p : Set X → Prop
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → p t → p s
    hcountable_union : ∀ (S : Set (Set X)), S.Countable → (∀ (s : Set X), Membersh …
    hnhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (n …
    f : Filter X := Filter.ofCountableUnion p hcountable_union ⋯
    ⊢ p s
  -/
  have : sᶜ ∈ f := hs.compl_mem_sets_of_nhdsWithin (by simpa [f] using hnhds)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    p : Set X → Prop
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → p t → p s
    hcountable_union : ∀ (S : Set (Set X)), S.Countable → (∀ (s : Set X), Membersh …
    hnhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (n …
    f : Filter X := Filter.ofCountableUnion p hcountable_union ⋯
    this : Membership.mem f (HasCompl.compl s)
    ⊢ p s
  -/
  rwa [← compl_compl s]
  /-
    🎉 no goals
  -/


/-- The intersection of a Lindelöf set and a closed set is a Lindelöf set. -/
theorem IsLindelof.inter_right (hs : IsLindelof s) (ht : IsClosed t) : IsLindelof (s ∩ t) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsLindelof s
    ht : IsClosed t
    ⊢ IsLindelof (Inter.inter s t)
  -/
  intro f hnf _ hstf
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s t : Set X
    hs : IsLindelof s
    ht : IsClosed t
    f : Filter X
    hnf : f.NeBot
    inst✝ : CountableInterFilter f
    hstf : LE.le f (Filter.principal (Inter.inter s t))
    ⊢ Exists fun x => And (Membership.mem (Inter.inter s t) x) (ClusterPt x f)
  -/
  rw [← inf_principal, le_inf_iff] at hstf
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s t : Set X
    hs : IsLindelof s
    ht : IsClosed t
    f : Filter X
    hnf : f.NeBot
    inst✝ : CountableInterFilter f
    hstf : And (LE.le f (Filter.principal s)) (LE.le f (Filter.principal t))
    ⊢ Exists fun x => And (Membership.mem (Inter.inter s t) x) (ClusterPt x f)
  -/
  obtain ⟨x, hsx, hx⟩ : ∃ x ∈ s, ClusterPt x f := hs hstf.1
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s t : Set X
    hs : IsLindelof s
    ht : IsClosed t
    f : Filter X
    hnf : f.NeBot
    inst✝ : CountableInterFilter f
    hstf : And (LE.le f (Filter.principal s)) (LE.le f (Filter.principal t))
    x : X
    hsx : Membership.mem s x
    hx : ClusterPt x f
    ⊢ Exists fun x => And (Membership.mem (Inter.inter s t) x) (ClusterPt x f)
  -/
  have hxt : x ∈ t := ht.mem_of_nhdsWithin_neBot <| hx.mono hstf.2
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s t : Set X
    hs : IsLindelof s
    ht : IsClosed t
    f : Filter X
    hnf : f.NeBot
    inst✝ : CountableInterFilter f
    hstf : And (LE.le f (Filter.principal s)) (LE.le f (Filter.principal t))
    x : X
    hsx : Membership.mem s x
    hx : ClusterPt x f
    hxt : Membership.mem t x
    ⊢ Exists fun x => And (Membership.mem (Inter.inter s t) x) (ClusterPt x f)
  -/
  exact ⟨x, ⟨hsx, hxt⟩, hx⟩
  /-
    🎉 no goals
  -/


/-- The intersection of a closed set and a Lindelöf set is a Lindelöf set. -/
theorem IsLindelof.inter_left (ht : IsLindelof t) (hs : IsClosed s) : IsLindelof (s ∩ t) :=
  inter_comm t s ▸ ht.inter_right hs


/-- The set difference of a Lindelöf set and an open set is a Lindelöf set. -/
theorem IsLindelof.diff (hs : IsLindelof s) (ht : IsOpen t) : IsLindelof (s \ t) :=
  hs.inter_right (isClosed_compl_iff.mpr ht)


/-- A closed subset of a Lindelöf set is a Lindelöf set. -/
theorem IsLindelof.of_isClosed_subset (hs : IsLindelof s) (ht : IsClosed t) (h : t ⊆ s) :
    IsLindelof t := inter_eq_self_of_subset_right h ▸ hs.inter_right ht


/-- A continuous image of a Lindelöf set is a Lindelöf set. -/
theorem IsLindelof.image_of_continuousOn {f : X → Y} (hs : IsLindelof s) (hf : ContinuousOn f s) :
    IsLindelof (f '' s) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsLindelof s
    hf : ContinuousOn f s
    ⊢ IsLindelof (Set.image f s)
  -/
  intro l lne _ ls
  have : NeBot (l.comap f ⊓ 𝓟 s) :=
    comap_inf_principal_neBot_of_image_mem lne (le_principal_iff.1 ls)
  /-
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsLindelof s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
    inst✝ : CountableInterFilter l
    ls : LE.le l (Filter.principal (Set.image f s))
    this : (Min.min (Filter.comap f l) (Filter.principal s)).NeBot
    ⊢ Exists fun x => And (Membership.mem (Set.image f s) x) (ClusterPt x l)
  -/
  obtain ⟨x, hxs, hx⟩ : ∃ x ∈ s, ClusterPt x (l.comap f ⊓ 𝓟 s) := @hs _ this _ inf_le_right
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsLindelof s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
    inst✝ : CountableInterFilter l
    ls : LE.le l (Filter.principal (Set.image f s))
    this : (Min.min (Filter.comap f l) (Filter.principal s)).NeBot
    x : X
    hxs : Membership.mem s x
    hx : ClusterPt x (Min.min (Filter.comap f l) (Filter.principal s))
    ⊢ Exists fun x => And (Membership.mem (Set.image f s) x) (ClusterPt x l)
  -/
  haveI := hx.neBot
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsLindelof s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
    inst✝ : CountableInterFilter l
    ls : LE.le l (Filter.principal (Set.image f s))
    this✝ : (Min.min (Filter.comap f l) (Filter.principal s)).NeBot
    x : X
    hxs : Membership.mem s x
    hx : ClusterPt x (Min.min (Filter.comap f l) (Filter.principal s))
    this : (Min.min (nhds x) (Min.min (Filter.comap f l) (Filter.principal s))).Ne …
    ⊢ Exists fun x => And (Membership.mem (Set.image f s) x) (ClusterPt x l)
  -/
  use f x, mem_image_of_mem f hxs
  have : Tendsto f (𝓝 x ⊓ (comap f l ⊓ 𝓟 s)) (𝓝 (f x) ⊓ l) := by
    convert (hf x hxs).inf (@tendsto_comap _ _ f l) using 1
    rw [nhdsWithin]
    ac_rfl
  /-
    case right
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsLindelof s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
    inst✝ : CountableInterFilter l
    ls : LE.le l (Filter.principal (Set.image f s))
    this✝¹ : (Min.min (Filter.comap f l) (Filter.principal s)).NeBot
    x : X
    hxs : Membership.mem s x
    hx : ClusterPt x (Min.min (Filter.comap f l) (Filter.principal s))
    this✝ : (Min.min (nhds x) (Min.min (Filter.comap f l) (Filter.principal s))).N …
    this : Filter.Tendsto f (Min.min (nhds x) (Min.min (Filter.comap f l) (Filter. …
    ⊢ ClusterPt (f x) l
  -/
  exact this.neBot
  /-
    🎉 no goals
  -/


/-- A continuous image of a Lindelöf set is a Lindelöf set within the codomain. -/
theorem IsLindelof.image {f : X → Y} (hs : IsLindelof s) (hf : Continuous f) :
    IsLindelof (f '' s) := hs.image_of_continuousOn hf.continuousOn


/-- A filter with the countable intersection property that is finer than the principal filter on
a Lindelöf set `s` contains any open set that contains all clusterpoints of `s`. -/
theorem IsLindelof.adherence_nhdset {f : Filter X} [CountableInterFilter f] (hs : IsLindelof s)
    (hf₂ : f ≤ 𝓟 s) (ht₁ : IsOpen t) (ht₂ : ∀ x ∈ s, ClusterPt x f → x ∈ t) : t ∈ f :=
  (eq_or_neBot _).casesOn mem_of_eq_bot fun _ ↦
    let ⟨x, hx, hfx⟩ := @hs (f ⊓ 𝓟 tᶜ) _ _ <| inf_le_of_left_le hf₂
    have : x ∈ t := ht₂ x hx hfx.of_inf_left
    have : tᶜ ∩ t ∈ 𝓝[tᶜ] x := inter_mem_nhdsWithin _ (ht₁.mem_nhds this)
    have A : 𝓝[tᶜ] x = ⊥ := empty_mem_iff_bot.1 <| compl_inter_self t ▸ this
    have : 𝓝[tᶜ] x ≠ ⊥ := hfx.of_inf_right.ne
    absurd A this


/-- For every open cover of a Lindelöf set, there exists a countable subcover. -/
theorem IsLindelof.elim_countable_subcover {ι : Type v} (hs : IsLindelof s) (U : ι → Set X)
    (hUo : ∀ i, IsOpen (U i)) (hsU : s ⊆ ⋃ i, U i) :
    ∃ r : Set ι, r.Countable ∧ (s ⊆ ⋃ i ∈ r, U i) := by
  have hmono : ∀ ⦃s t : Set X⦄, s ⊆ t → (∃ r : Set ι, r.Countable ∧ t ⊆ ⋃ i ∈ r, U i)
      → (∃ r : Set ι, r.Countable ∧ s ⊆ ⋃ i ∈ r, U i) := by
    intro _ _ hst ⟨r, ⟨hrcountable, hsub⟩⟩
    exact ⟨r, hrcountable, Subset.trans hst hsub⟩
  have hcountable_union : ∀ (S : Set (Set X)), S.Countable
      → (∀ s ∈ S, ∃ r : Set ι, r.Countable ∧ (s ⊆ ⋃ i ∈ r, U i))
      → ∃ r : Set ι, r.Countable ∧ (⋃₀ S ⊆ ⋃ i ∈ r, U i) := by
    intro S hS hsr
    choose! r hr using hsr
    refine ⟨⋃ s ∈ S, r s, hS.biUnion_iff.mpr (fun s hs ↦ (hr s hs).1), ?_⟩
    refine sUnion_subset ?h.right.h
    simp only [mem_iUnion, exists_prop, iUnion_exists, biUnion_and']
    exact fun i is x hx ↦ mem_biUnion is ((hr i is).2 hx)
  have h_nhds : ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, ∃ r : Set ι, r.Countable ∧ (t ⊆ ⋃ i ∈ r, U i) := by
    intro x hx
    let ⟨i, hi⟩ := mem_iUnion.1 (hsU hx)
    refine ⟨U i, mem_nhdsWithin_of_mem_nhds ((hUo i).mem_nhds hi), {i}, by simp, ?_⟩
    simp only [mem_singleton_iff, iUnion_iUnion_eq_left]
    exact Subset.refl _
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → (Exists fun r => And r.Countab …
    hcountable_union : ∀ (S : Set (Set X)), S.Countable → (∀ (s : Set X), Membersh …
    h_nhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem ( …
    ⊢ Exists fun r => And r.Countable (HasSubset.Subset s (Set.iUnion fun i => Set …
  -/
  exact hs.induction_on hmono hcountable_union h_nhds
  /-
    🎉 no goals
  -/


theorem IsLindelof.elim_nhds_subcover' (hs : IsLindelof s) (U : ∀ x ∈ s, Set X)
    (hU : ∀ x (hx : x ∈ s), U x ‹x ∈ s› ∈ 𝓝 x) :
    ∃ t : Set s, t.Countable ∧ s ⊆ ⋃ x ∈ t, U (x : s) x.2 := by
  have := hs.elim_countable_subcover (fun x : s ↦ interior (U x x.2)) (fun _ ↦ isOpen_interior)
    fun x hx ↦
      mem_iUnion.2 ⟨⟨x, hx⟩, mem_interior_iff_mem_nhds.2 <| hU _ _⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    this : Exists fun r => And r.Countable (HasSubset.Subset s (Set.iUnion fun i = …
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun x => Set …
  -/
  rcases this with ⟨r, ⟨hr, hs⟩⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    r : Set ↑s
    hr : r.Countable
    hs : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => (fun x => int …
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun x => Set …
  -/
  use r, hr
  /-
    case right
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    r : Set ↑s
    hr : r.Countable
    hs : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => (fun x => int …
    ⊢ HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
  -/
  apply Subset.trans hs
  /-
    case right
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    r : Set ↑s
    hr : r.Countable
    hs : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => (fun x => int …
    ⊢ HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => (fun x => interior …
  -/
  apply iUnion₂_subset
  /-
    case right.h
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    r : Set ↑s
    hr : r.Countable
    hs : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => (fun x => int …
    ⊢ ∀ (i : ↑s), Membership.mem r i → HasSubset.Subset ((fun x => interior (U ↑x  …
  -/
  intro i hi
  /-
    case right.h
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    r : Set ↑s
    hr : r.Countable
    hs : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => (fun x => int …
    i : ↑s
    hi : Membership.mem r i
    ⊢ HasSubset.Subset ((fun x => interior (U ↑x ⋯)) i) (Set.iUnion fun x => Set.i …
  -/
  apply Subset.trans interior_subset
  /-
    case right.h
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsLindelof s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    r : Set ↑s
    hr : r.Countable
    hs : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => (fun x => int …
    i : ↑s
    hi : Membership.mem r i
    ⊢ HasSubset.Subset (U ↑i ⋯) (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
  -/
  exact subset_iUnion_of_subset i (subset_iUnion_of_subset hi (Subset.refl _))
  /-
    🎉 no goals
  -/


theorem IsLindelof.elim_nhds_subcover (hs : IsLindelof s) (U : X → Set X)
    (hU : ∀ x ∈ s, U x ∈ 𝓝 x) :
    ∃ t : Set X, t.Countable ∧ (∀ x ∈ t, x ∈ s) ∧ s ⊆ ⋃ x ∈ t, U x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    U : X → Set X
    hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
    ⊢ Exists fun t => And t.Countable (And (∀ (x : X), Membership.mem t x → Member …
  -/
  let ⟨t, ⟨htc, htsub⟩⟩ := hs.elim_nhds_subcover' (fun x _ ↦ U x) hU
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    U : X → Set X
    hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
    t : Set ↑s
    htc : t.Countable
    htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
    ⊢ Exists fun t => And t.Countable (And (∀ (x : X), Membership.mem t x → Member …
  -/
  refine ⟨↑t, Countable.image htc Subtype.val, ?_⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsLindelof s
    U : X → Set X
    hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
    t : Set ↑s
    htc : t.Countable
    htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
    ⊢ And (∀ (x : X), Membership.mem (Set.image Subtype.val t) x → Membership.mem  …
  -/
  constructor
    /-
      case left
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      hs : IsLindelof s
      U : X → Set X
      hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
      t : Set ↑s
      htc : t.Countable
      htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
      ⊢ ∀ (x : X), Membership.mem (Set.image Subtype.val t) x → Membership.mem s x
    -/
  · intro _
    /-
      case left
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      hs : IsLindelof s
      U : X → Set X
      hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
      t : Set ↑s
      htc : t.Countable
      htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
      x✝ : X
      ⊢ Membership.mem (Set.image Subtype.val t) x✝ → Membership.mem s x✝
    -/
    simp only [mem_image, Subtype.exists, exists_and_right, exists_eq_right, forall_exists_index]
    /-
      case left
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      hs : IsLindelof s
      U : X → Set X
      hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
      t : Set ↑s
      htc : t.Countable
      htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
      x✝ : X
      ⊢ ∀ (x : Membership.mem s x✝), Membership.mem t ⟨x✝, ⋯⟩ → Membership.mem s x✝
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case right
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      hs : IsLindelof s
      U : X → Set X
      hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
      t : Set ↑s
      htc : t.Countable
      htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
      ⊢ HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    -/
  · have : ⋃ x ∈ t, U ↑x = ⋃ x ∈ Subtype.val '' t, U x := biUnion_image.symm
    /-
      case right
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      hs : IsLindelof s
      U : X → Set X
      hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
      t : Set ↑s
      htc : t.Countable
      htsub : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
      this : Eq (Set.iUnion fun x => Set.iUnion fun h => U ↑x) (Set.iUnion fun x =>  …
      ⊢ HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    -/
    rwa [← this]
    /-
      🎉 no goals
    -/


/-- For every nonempty open cover of a Lindelöf set, there exists a subcover indexed by ℕ. -/
theorem IsLindelof.indexed_countable_subcover {ι : Type v} [Nonempty ι]
    (hs : IsLindelof s) (U : ι → Set X) (hUo : ∀ i, IsOpen (U i)) (hsU : s ⊆ ⋃ i, U i) :
    ∃ f : ℕ → ι, s ⊆ ⋃ n, U (f n) := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    ⊢ Exists fun f => HasSubset.Subset s (Set.iUnion fun n => U (f n))
  -/
  obtain ⟨c, ⟨c_count, c_cov⟩⟩ := hs.elim_countable_subcover U hUo hsU
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    c : Set ι
    c_count : c.Countable
    c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ Exists fun f => HasSubset.Subset s (Set.iUnion fun n => U (f n))
  -/
  rcases c.eq_empty_or_nonempty with rfl | c_nonempty
    /-
      case intro.intro.inl
      X : Type u
      inst✝¹ : TopologicalSpace X
      s : Set X
      ι : Type v
      inst✝ : Nonempty ι
      hs : IsLindelof s
      U : ι → Set X
      hUo : ∀ (i : ι), IsOpen (U i)
      hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
      c_count : EmptyCollection.emptyCollection.Countable
      c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
      ⊢ Exists fun f => HasSubset.Subset s (Set.iUnion fun n => U (f n))
    -/
  · simp only [mem_empty_iff_false, iUnion_of_empty, iUnion_empty] at c_cov
    /-
      case intro.intro.inl
      X : Type u
      inst✝¹ : TopologicalSpace X
      s : Set X
      ι : Type v
      inst✝ : Nonempty ι
      hs : IsLindelof s
      U : ι → Set X
      hUo : ∀ (i : ι), IsOpen (U i)
      hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
      c_count : EmptyCollection.emptyCollection.Countable
      c_cov : HasSubset.Subset s EmptyCollection.emptyCollection
      ⊢ Exists fun f => HasSubset.Subset s (Set.iUnion fun n => U (f n))
    -/
    simp only [subset_eq_empty c_cov rfl, empty_subset, exists_const]
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    c : Set ι
    c_count : c.Countable
    c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    c_nonempty : c.Nonempty
    ⊢ Exists fun f => HasSubset.Subset s (Set.iUnion fun n => U (f n))
  -/
  obtain ⟨f, f_surj⟩ := (Set.countable_iff_exists_surjective c_nonempty).mp c_count
  /-
    case intro.intro.inr.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    c : Set ι
    c_count : c.Countable
    c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    c_nonempty : c.Nonempty
    f : Nat → ↑c
    f_surj : Function.Surjective f
    ⊢ Exists fun f => HasSubset.Subset s (Set.iUnion fun n => U (f n))
  -/
  refine ⟨fun x ↦ f x, c_cov.trans <| iUnion₂_subset_iff.mpr (?_ : ∀ i ∈ c, U i ⊆ ⋃ n, U (f n))⟩
  /-
    case intro.intro.inr.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    c : Set ι
    c_count : c.Countable
    c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    c_nonempty : c.Nonempty
    f : Nat → ↑c
    f_surj : Function.Surjective f
    ⊢ ∀ (i : ι), Membership.mem c i → HasSubset.Subset (U i) (Set.iUnion fun n =>  …
  -/
  intro x hx
  /-
    case intro.intro.inr.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    c : Set ι
    c_count : c.Countable
    c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    c_nonempty : c.Nonempty
    f : Nat → ↑c
    f_surj : Function.Surjective f
    x : ι
    hx : Membership.mem c x
    ⊢ HasSubset.Subset (U x) (Set.iUnion fun n => U ↑(f n))
  -/
  obtain ⟨n, hn⟩ := f_surj ⟨x, hx⟩
  /-
    case intro.intro.inr.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    ι : Type v
    inst✝ : Nonempty ι
    hs : IsLindelof s
    U : ι → Set X
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    c : Set ι
    c_count : c.Countable
    c_cov : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    c_nonempty : c.Nonempty
    f : Nat → ↑c
    f_surj : Function.Surjective f
    x : ι
    hx : Membership.mem c x
    n : Nat
    hn : Eq (f n) ⟨x, hx⟩
    ⊢ HasSubset.Subset (U x) (Set.iUnion fun n => U ↑(f n))
  -/
  exact subset_iUnion_of_subset n <| subset_of_eq (by rw [hn])
  /-
    🎉 no goals
  -/


/-- The neighborhood filter of a Lindelöf set is disjoint with a filter `l` with the countable
intersection property if and only if the neighborhood filter of each point of this set
is disjoint with `l`. -/
theorem IsLindelof.disjoint_nhdsSet_left {l : Filter X} [CountableInterFilter l]
    (hs : IsLindelof s) :
    Disjoint (𝓝ˢ s) l ↔ ∀ x ∈ s, Disjoint (𝓝 x) l := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    ⊢ Iff (Disjoint (nhdsSet s) l) (∀ (x : X), Membership.mem s x → Disjoint (nhds …
  -/
  refine ⟨fun h x hx ↦ h.mono_left <| nhds_le_nhdsSet hx, fun H ↦ ?_⟩
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    ⊢ Disjoint (nhdsSet s) l
  -/
  choose! U hxU hUl using fun x hx ↦ (nhds_basis_opens x).disjoint_iff_left.1 (H x hx)
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem s x → And (Membership.mem (U x) x) (IsOpen (U  …
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    ⊢ Disjoint (nhdsSet s) l
  -/
  choose hxU hUo using hxU
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    hxU : ∀ (x : X), Membership.mem s x → Membership.mem (U x) x
    hUo : ∀ (x : X), Membership.mem s x → IsOpen (U x)
    ⊢ Disjoint (nhdsSet s) l
  -/
  rcases hs.elim_nhds_subcover U fun x hx ↦ (hUo x hx).mem_nhds (hxU x hx) with ⟨t, htc, hts, hst⟩
  refine (hasBasis_nhdsSet _).disjoint_iff_left.2
    ⟨⋃ x ∈ t, U x, ⟨isOpen_biUnion fun x hx ↦ hUo x (hts x hx), hst⟩, ?_⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    hxU : ∀ (x : X), Membership.mem s x → Membership.mem (U x) x
    hUo : ∀ (x : X), Membership.mem s x → IsOpen (U x)
    t : Set X
    htc : t.Countable
    hts : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hst : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Membership.mem l (HasCompl.compl (Set.iUnion fun x => Set.iUnion fun h => U  …
  -/
  rw [compl_iUnion₂]
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    hxU : ∀ (x : X), Membership.mem s x → Membership.mem (U x) x
    hUo : ∀ (x : X), Membership.mem s x → IsOpen (U x)
    t : Set X
    htc : t.Countable
    hts : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hst : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Membership.mem l (Set.iInter fun i => Set.iInter fun j => HasCompl.compl (U  …
  -/
  exact (countable_bInter_mem htc).mpr (fun i hi ↦ hUl _ (hts _ hi))
  /-
    🎉 no goals
  -/


/-- A filter `l` with the countable intersection property is disjoint with the neighborhood
filter of a Lindelöf set if and only if it is disjoint with the neighborhood filter of each point
of this set. -/
theorem IsLindelof.disjoint_nhdsSet_right {l : Filter X} [CountableInterFilter l]
    (hs : IsLindelof s) : Disjoint l (𝓝ˢ s) ↔ ∀ x ∈ s, Disjoint l (𝓝 x) := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    l : Filter X
    inst✝ : CountableInterFilter l
    hs : IsLindelof s
    ⊢ Iff (Disjoint l (nhdsSet s)) (∀ (x : X), Membership.mem s x → Disjoint l (nh …
  -/
  simpa only [disjoint_comm] using hs.disjoint_nhdsSet_left
  /-
    🎉 no goals
  -/


/-- For every family of closed sets whose intersection avoids a Lindelö set,
there exists a countable subfamily whose intersection avoids this Lindelöf set. -/
theorem IsLindelof.elim_countable_subfamily_closed {ι : Type v} (hs : IsLindelof s)
    (t : ι → Set X) (htc : ∀ i, IsClosed (t i)) (hst : (s ∩ ⋂ i, t i) = ∅) :
    ∃ u : Set ι, u.Countable ∧ (s ∩ ⋂ i ∈ u, t i) = ∅ := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    ⊢ Exists fun u => And u.Countable (Eq (Inter.inter s (Set.iInter fun i => Set. …
  -/
  let U := tᶜ
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    U : ι → Set X := HasCompl.compl t
    ⊢ Exists fun u => And u.Countable (Eq (Inter.inter s (Set.iInter fun i => Set. …
  -/
  have hUo : ∀ i, IsOpen (U i) := by simp only [U, Pi.compl_apply, isOpen_compl_iff]; exact htc
  have hsU : s ⊆ ⋃ i, U i := by
    simp only [U, Pi.compl_apply]
    rw [← compl_iInter]
    apply disjoint_compl_left_iff_subset.mp
    simp only [compl_iInter, compl_iUnion, compl_compl]
    apply Disjoint.symm
    exact disjoint_iff_inter_eq_empty.mpr hst
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    U : ι → Set X := HasCompl.compl t
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    ⊢ Exists fun u => And u.Countable (Eq (Inter.inter s (Set.iInter fun i => Set. …
  -/
  rcases hs.elim_countable_subcover U hUo hsU with ⟨u, ⟨hucount, husub⟩⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    U : ι → Set X := HasCompl.compl t
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    u : Set ι
    hucount : u.Countable
    husub : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ Exists fun u => And u.Countable (Eq (Inter.inter s (Set.iInter fun i => Set. …
  -/
  use u, hucount
  /-
    case right
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    U : ι → Set X := HasCompl.compl t
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    u : Set ι
    hucount : u.Countable
    husub : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => t i)) EmptyCollec …
  -/
  rw [← disjoint_compl_left_iff_subset] at husub
  /-
    case right
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    U : ι → Set X := HasCompl.compl t
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    u : Set ι
    hucount : u.Countable
    husub : Disjoint (HasCompl.compl (Set.iUnion fun i => Set.iUnion fun h => U i) …
    ⊢ Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => t i)) EmptyCollec …
  -/
  simp only [U, Pi.compl_apply, compl_iUnion, compl_compl] at husub
  /-
    case right
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    U : ι → Set X := HasCompl.compl t
    hUo : ∀ (i : ι), IsOpen (U i)
    hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
    u : Set ι
    hucount : u.Countable
    husub : Disjoint (Set.iInter fun i => Set.iInter fun x => t i) s
    ⊢ Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => t i)) EmptyCollec …
  -/
  exact disjoint_iff_inter_eq_empty.mp (Disjoint.symm husub)
  /-
    🎉 no goals
  -/


/-- To show that a Lindelöf set intersects the intersection of a family of closed sets,
  it is sufficient to show that it intersects every countable subfamily. -/
theorem IsLindelof.inter_iInter_nonempty {ι : Type v} (hs : IsLindelof s) (t : ι → Set X)
    (htc : ∀ i, IsClosed (t i)) (hst : ∀ u : Set ι, u.Countable ∧ (s ∩ ⋂ i ∈ u, t i).Nonempty) :
    (s ∩ ⋂ i, t i).Nonempty := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : ∀ (u : Set ι), And u.Countable (Inter.inter s (Set.iInter fun i => Set.i …
    ⊢ (Inter.inter s (Set.iInter fun i => t i)).Nonempty
  -/
  contrapose! hst
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    ⊢ Exists fun u => u.Countable → Eq (Inter.inter s (Set.iInter fun i => Set.iIn …
  -/
  rcases hs.elim_countable_subfamily_closed t htc hst with ⟨u, ⟨_, husub⟩⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsLindelof s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    u : Set ι
    left✝ : u.Countable
    husub : Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => t i)) Empty …
    ⊢ Exists fun u => u.Countable → Eq (Inter.inter s (Set.iInter fun i => Set.iIn …
  -/
  exact ⟨u, fun _ ↦ husub⟩
  /-
    🎉 no goals
  -/


/-- For every open cover of a Lindelöf set, there exists a countable subcover. -/
theorem IsLindelof.elim_countable_subcover_image {b : Set ι} {c : ι → Set X} (hs : IsLindelof s)
    (hc₁ : ∀ i ∈ b, IsOpen (c i)) (hc₂ : s ⊆ ⋃ i ∈ b, c i) :
    ∃ b', b' ⊆ b ∧ Set.Countable b' ∧ s ⊆ ⋃ i ∈ b', c i := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsLindelof s
    hc₁ : ∀ (i : ι), Membership.mem b i → IsOpen (c i)
    hc₂ : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c i)
    ⊢ Exists fun b' => And (HasSubset.Subset b' b) (And b'.Countable (HasSubset.Su …
  -/
  simp only [Subtype.forall', biUnion_eq_iUnion] at hc₁ hc₂
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsLindelof s
    hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
    hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
    ⊢ Exists fun b' => And (HasSubset.Subset b' b) (And b'.Countable (HasSubset.Su …
  -/
  rcases hs.elim_countable_subcover (fun i ↦ c i : b → Set X) hc₁ hc₂ with ⟨d, hd⟩
  /-
    case intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsLindelof s
    hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
    hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
    d : Set ↑b
    hd : And d.Countable (HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h …
    ⊢ Exists fun b' => And (HasSubset.Subset b' b) (And b'.Countable (HasSubset.Su …
  -/
  refine ⟨Subtype.val '' d, by simp, Countable.image hd.1 Subtype.val, ?_⟩
  /-
    case intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsLindelof s
    hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
    hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
    d : Set ↑b
    hd : And d.Countable (HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h …
    ⊢ HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c i)
  -/
  rw [biUnion_image]
  /-
    case intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsLindelof s
    hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
    hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
    d : Set ↑b
    hd : And d.Countable (HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h …
    ⊢ HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => c ↑y)
  -/
  exact hd.2
  /-
    🎉 no goals
  -/



/-- A set `s` is Lindelöf if for every open cover of `s`, there exists a countable subcover. -/
theorem isLindelof_of_countable_subcover
    (h : ∀ {ι : Type u} (U : ι → Set X), (∀ i, IsOpen (U i)) → (s ⊆ ⋃ i, U i) →
    ∃ t : Set ι, t.Countable ∧ s ⊆ ⋃ i ∈ t, U i) :
    IsLindelof s := fun f hf hfs ↦ by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    h : ∀ {ι : Type u} (U : ι → Set X), (∀ (i : ι), IsOpen (U i)) → HasSubset.Subs …
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    ⊢ LE.le f (Filter.principal s) → Exists fun x => And (Membership.mem s x) (Clu …
  -/
  contrapose! h
  simp only [ClusterPt, not_neBot, ← disjoint_iff, SetCoe.forall',
    (nhds_basis_opens _).disjoint_iff_left] at h
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    h : And (LE.le f (Filter.principal s)) (∀ (x : ↑s), Exists fun i => And (And ( …
    ⊢ Exists fun {ι} => Exists fun U => And (∀ (i : ι), IsOpen (U i)) (And (HasSub …
  -/
  choose fsub U hU hUf using h
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    ⊢ Exists fun {ι} => Exists fun U => And (∀ (i : ι), IsOpen (U i)) (And (HasSub …
  -/
  refine ⟨s, U, fun x ↦ (hU x).2, fun x hx ↦ mem_iUnion.2 ⟨⟨x, hx⟩, (hU _).1 ⟩, ?_⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    ⊢ ∀ (t : Set ↑s), t.Countable → Not (HasSubset.Subset s (Set.iUnion fun i => S …
  -/
  intro t ht h
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Set ↑s
    ht : t.Countable
    h : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ False
  -/
  have uinf := f.sets_of_superset (le_principal_iff.1 fsub) h
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Set ↑s
    ht : t.Countable
    h : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    uinf : Membership.mem f.sets (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ False
  -/
  have uninf : ⋂ i ∈ t, (U i)ᶜ ∈ f := (countable_bInter_mem ht).mpr (fun _ _ ↦ hUf _)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Set ↑s
    ht : t.Countable
    h : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    uinf : Membership.mem f.sets (Set.iUnion fun i => Set.iUnion fun h => U i)
    uninf : Membership.mem f (Set.iInter fun i => Set.iInter fun h => HasCompl.com …
    ⊢ False
  -/
  rw [← compl_iUnion₂] at uninf
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Set ↑s
    ht : t.Countable
    h : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    uinf : Membership.mem f.sets (Set.iUnion fun i => Set.iUnion fun h => U i)
    uninf : Membership.mem f (HasCompl.compl (Set.iUnion fun i => Set.iUnion fun j …
    ⊢ False
  -/
  have uninf := compl_not_mem uninf
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Set ↑s
    ht : t.Countable
    h : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    uinf : Membership.mem f.sets (Set.iUnion fun i => Set.iUnion fun h => U i)
    uninf✝ : Membership.mem f (HasCompl.compl (Set.iUnion fun i => Set.iUnion fun  …
    uninf : Not (Membership.mem f (HasCompl.compl (HasCompl.compl (Set.iUnion fun  …
    ⊢ False
  -/
  simp only [compl_compl] at uninf
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : CountableInterFilter f
    fsub : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Set ↑s
    ht : t.Countable
    h : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    uinf : Membership.mem f.sets (Set.iUnion fun i => Set.iUnion fun h => U i)
    uninf✝ : Membership.mem f (HasCompl.compl (Set.iUnion fun i => Set.iUnion fun  …
    uninf : Not (Membership.mem f (Set.iUnion fun i => Set.iUnion fun j => U i))
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


/-- A set `s` is Lindelöf if for every family of closed sets whose intersection avoids `s`,
there exists a countable subfamily whose intersection avoids `s`. -/
theorem isLindelof_of_countable_subfamily_closed
    (h :
      ∀ {ι : Type u} (t : ι → Set X), (∀ i, IsClosed (t i)) → (s ∩ ⋂ i, t i) = ∅ →
        ∃ u : Set ι, u.Countable ∧ (s ∩ ⋂ i ∈ u, t i) = ∅) :
    IsLindelof s :=
  isLindelof_of_countable_subcover fun U hUo hsU ↦ by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Inter.in …
      ι✝ : Type u
      U : ι✝ → Set X
      hUo : ∀ (i : ι✝), IsOpen (U i)
      hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
      ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun i => Set …
    -/
    rw [← disjoint_compl_right_iff_subset, compl_iUnion, disjoint_iff] at hsU
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Inter.in …
      ι✝ : Type u
      U : ι✝ → Set X
      hUo : ∀ (i : ι✝), IsOpen (U i)
      hsU : Eq (Min.min s (Set.iInter fun i => HasCompl.compl (U i))) Bot.bot
      ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun i => Set …
    -/
    rcases h (fun i ↦ (U i)ᶜ) (fun i ↦ (hUo _).isClosed_compl) hsU with ⟨t, ht⟩
    /-
      case intro
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Inter.in …
      ι✝ : Type u
      U : ι✝ → Set X
      hUo : ∀ (i : ι✝), IsOpen (U i)
      hsU : Eq (Min.min s (Set.iInter fun i => HasCompl.compl (U i))) Bot.bot
      t : Set ι✝
      ht : And t.Countable (Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h  …
      ⊢ Exists fun t => And t.Countable (HasSubset.Subset s (Set.iUnion fun i => Set …
    -/
    refine ⟨t, ?_⟩
    /-
      case intro
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Inter.in …
      ι✝ : Type u
      U : ι✝ → Set X
      hUo : ∀ (i : ι✝), IsOpen (U i)
      hsU : Eq (Min.min s (Set.iInter fun i => HasCompl.compl (U i))) Bot.bot
      t : Set ι✝
      ht : And t.Countable (Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h  …
      ⊢ And t.Countable (HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => …
    -/
    rwa [← disjoint_compl_right_iff_subset, compl_iUnion₂, disjoint_iff]
    /-
      🎉 no goals
    -/


/-- A set `s` is Lindelöf if and only if
for every open cover of `s`, there exists a countable subcover. -/
theorem isLindelof_iff_countable_subcover :
    IsLindelof s ↔ ∀ {ι : Type u} (U : ι → Set X),
      (∀ i, IsOpen (U i)) → (s ⊆ ⋃ i, U i) → ∃ t : Set ι, t.Countable ∧ s ⊆ ⋃ i ∈ t, U i :=
  ⟨fun hs ↦ hs.elim_countable_subcover, isLindelof_of_countable_subcover⟩


/-- A set `s` is Lindelöf if and only if
for every family of closed sets whose intersection avoids `s`,
there exists a countable subfamily whose intersection avoids `s`. -/
theorem isLindelof_iff_countable_subfamily_closed :
    IsLindelof s ↔ ∀ {ι : Type u} (t : ι → Set X),
    (∀ i, IsClosed (t i)) → (s ∩ ⋂ i, t i) = ∅
    → ∃ u : Set ι, u.Countable ∧ (s ∩ ⋂ i ∈ u, t i) = ∅ :=
  ⟨fun hs ↦ hs.elim_countable_subfamily_closed, isLindelof_of_countable_subfamily_closed⟩


/-- The empty set is a Lindelof set. -/
@[simp]
theorem isLindelof_empty : IsLindelof (∅ : Set X) := fun _f hnf _ hsf ↦
  Not.elim hnf.ne <| empty_mem_iff_bot.1 <| le_principal_iff.1 hsf


/-- A singleton set is a Lindelof set. -/
@[simp]
theorem isLindelof_singleton {x : X} : IsLindelof ({x} : Set X) := fun _ hf _ hfa ↦
  ⟨x, rfl, ClusterPt.of_le_nhds'
                     /-
                       X : Type u
                       inst✝ : TopologicalSpace X
                       x : X
                       x✝¹ : Filter X
                       hf : x✝¹.NeBot
                       x✝ : CountableInterFilter x✝¹
                       hfa : LE.le x✝¹ (Filter.principal (Singleton.singleton x))
                       ⊢ LE.le (Filter.principal (Singleton.singleton x)) (nhds x)
                     -/
    (hfa.trans <| by simpa only [principal_singleton] using pure_le_nhds x) hf⟩
                     /-
                       🎉 no goals
                     -/


theorem Set.Subsingleton.isLindelof (hs : s.Subsingleton) : IsLindelof s :=
  Subsingleton.induction_on hs isLindelof_empty fun _ ↦ isLindelof_singleton


theorem Set.Countable.isLindelof_biUnion {s : Set ι} {f : ι → Set X} (hs : s.Countable)
    (hf : ∀ i ∈ s, IsLindelof (f i)) : IsLindelof (⋃ i ∈ s, f i) := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set ι
    f : ι → Set X
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
    ⊢ IsLindelof (Set.iUnion fun i => Set.iUnion fun h => f i)
  -/
  apply isLindelof_of_countable_subcover
  /-
    case h
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set ι
    f : ι → Set X
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
    ⊢ ∀ {ι_1 : Type u} (U : ι_1 → Set X), (∀ (i : ι_1), IsOpen (U i)) → HasSubset. …
  -/
  intro i U hU hUcover
  have hiU : ∀ i ∈ s, f i ⊆ ⋃ i, U i :=
    fun _ is ↦ _root_.subset_trans (subset_biUnion_of_mem is) hUcover
  /-
    case h
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set ι
    f : ι → Set X
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
    i : Type u
    U : i → Set X
    hU : ∀ (i : i), IsOpen (U i)
    hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
    hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset (Set.iUnion fun i => Set.i …
  -/
  have iSets := fun i is ↦ (hf i is).elim_countable_subcover U hU (hiU i is)
  /-
    case h
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set ι
    f : ι → Set X
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
    i : Type u
    U : i → Set X
    hU : ∀ (i : i), IsOpen (U i)
    hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
    hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
    iSets : ∀ (i_1 : ι), Membership.mem s i_1 → Exists fun r => And r.Countable (H …
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset (Set.iUnion fun i => Set.i …
  -/
  choose! r hr using iSets
  /-
    case h
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set ι
    f : ι → Set X
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
    i : Type u
    U : i → Set X
    hU : ∀ (i : i), IsOpen (U i)
    hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
    hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
    r : ι → Set i
    hr : ∀ (i_1 : ι), Membership.mem s i_1 → And (r i_1).Countable (HasSubset.Subs …
    ⊢ Exists fun t => And t.Countable (HasSubset.Subset (Set.iUnion fun i => Set.i …
  -/
  use ⋃ i ∈ s, r i
  /-
    case h
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set ι
    f : ι → Set X
    hs : s.Countable
    hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
    i : Type u
    U : i → Set X
    hU : ∀ (i : i), IsOpen (U i)
    hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
    hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
    r : ι → Set i
    hr : ∀ (i_1 : ι), Membership.mem s i_1 → And (r i_1).Countable (HasSubset.Subs …
    ⊢ And (Set.iUnion fun i_1 => Set.iUnion fun h => r i_1).Countable (HasSubset.S …
  -/
  constructor
    /-
      case h.left
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i : Type u
      U : i → Set X
      hU : ∀ (i : i), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
      r : ι → Set i
      hr : ∀ (i_1 : ι), Membership.mem s i_1 → And (r i_1).Countable (HasSubset.Subs …
      ⊢ (Set.iUnion fun i_1 => Set.iUnion fun h => r i_1).Countable
    -/
  · refine (Countable.biUnion_iff hs).mpr ?h.left.a
    /-
      case h.left.a
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i : Type u
      U : i → Set X
      hU : ∀ (i : i), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
      r : ι → Set i
      hr : ∀ (i_1 : ι), Membership.mem s i_1 → And (r i_1).Countable (HasSubset.Subs …
      ⊢ ∀ (a : ι), Membership.mem s a → (r a).Countable
    -/
    exact fun s hs ↦ (hr s hs).1
    /-
      🎉 no goals
    -/
    /-
      case h.right
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i : Type u
      U : i → Set X
      hU : ∀ (i : i), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
      r : ι → Set i
      hr : ∀ (i_1 : ι), Membership.mem s i_1 → And (r i_1).Countable (HasSubset.Subs …
      ⊢ HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set.iUnion f …
    -/
  · refine iUnion₂_subset ?h.right.h
    /-
      case h.right.h
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i : Type u
      U : i → Set X
      hU : ∀ (i : i), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion …
      r : ι → Set i
      hr : ∀ (i_1 : ι), Membership.mem s i_1 → And (r i_1).Countable (HasSubset.Subs …
      ⊢ ∀ (i_1 : ι), Membership.mem s i_1 → HasSubset.Subset (f i_1) (Set.iUnion fun …
    -/
    intro i is
    /-
      case h.right.h
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i✝ : Type u
      U : i✝ → Set X
      hU : ∀ (i : i✝), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i : ι), Membership.mem s i → HasSubset.Subset (f i) (Set.iUnion fun i …
      r : ι → Set i✝
      hr : ∀ (i : ι), Membership.mem s i → And (r i).Countable (HasSubset.Subset (f  …
      i : ι
      is : Membership.mem s i
      ⊢ HasSubset.Subset (f i) (Set.iUnion fun i => Set.iUnion fun h => U i)
    -/
    simp only [mem_iUnion, exists_prop, iUnion_exists, biUnion_and']
    /-
      case h.right.h
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i✝ : Type u
      U : i✝ → Set X
      hU : ∀ (i : i✝), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i : ι), Membership.mem s i → HasSubset.Subset (f i) (Set.iUnion fun i …
      r : ι → Set i✝
      hr : ∀ (i : ι), Membership.mem s i → And (r i).Countable (HasSubset.Subset (f  …
      i : ι
      is : Membership.mem s i
      ⊢ HasSubset.Subset (f i) (Set.iUnion fun y => Set.iUnion fun hy => Set.iUnion  …
    -/
    intro x hx
    /-
      case h.right.h
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Countable
      hf : ∀ (i : ι), Membership.mem s i → IsLindelof (f i)
      i✝ : Type u
      U : i✝ → Set X
      hU : ∀ (i : i✝), IsOpen (U i)
      hUcover : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set. …
      hiU : ∀ (i : ι), Membership.mem s i → HasSubset.Subset (f i) (Set.iUnion fun i …
      r : ι → Set i✝
      hr : ∀ (i : ι), Membership.mem s i → And (r i).Countable (HasSubset.Subset (f  …
      i : ι
      is : Membership.mem s i
      x : X
      hx : Membership.mem (f i) x
      ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun hy => Set.iUnion fun x => …
    -/
    exact mem_biUnion is ((hr i is).2 hx)
    /-
      🎉 no goals
    -/



theorem Set.Finite.isLindelof_biUnion {s : Set ι} {f : ι → Set X} (hs : s.Finite)
    (hf : ∀ i ∈ s, IsLindelof (f i)) : IsLindelof (⋃ i ∈ s, f i) :=
  Set.Countable.isLindelof_biUnion (countable hs) hf


theorem Finset.isLindelof_biUnion (s : Finset ι) {f : ι → Set X} (hf : ∀ i ∈ s, IsLindelof (f i)) :
    IsLindelof (⋃ i ∈ s, f i) :=
  s.finite_toSet.isLindelof_biUnion hf


theorem isLindelof_accumulate {K : ℕ → Set X} (hK : ∀ n, IsLindelof (K n)) (n : ℕ) :
    IsLindelof (Accumulate K n) :=
  (finite_le_nat n).isLindelof_biUnion fun k _ => hK k


theorem Set.Countable.isLindelof_sUnion {S : Set (Set X)} (hf : S.Countable)
    (hc : ∀ s ∈ S, IsLindelof s) : IsLindelof (⋃₀ S) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    hf : S.Countable
    hc : ∀ (s : Set X), Membership.mem S s → IsLindelof s
    ⊢ IsLindelof S.sUnion
  -/
  rw [sUnion_eq_biUnion]; exact hf.isLindelof_biUnion hc
                          /-
                            🎉 no goals
                          -/


theorem Set.Finite.isLindelof_sUnion {S : Set (Set X)} (hf : S.Finite)
    (hc : ∀ s ∈ S, IsLindelof s) : IsLindelof (⋃₀ S) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    hf : S.Finite
    hc : ∀ (s : Set X), Membership.mem S s → IsLindelof s
    ⊢ IsLindelof S.sUnion
  -/
  rw [sUnion_eq_biUnion]; exact hf.isLindelof_biUnion hc
                          /-
                            🎉 no goals
                          -/


theorem isLindelof_iUnion {ι : Sort*} {f : ι → Set X} [Countable ι] (h : ∀ i, IsLindelof (f i)) :
    IsLindelof (⋃ i, f i) := (countable_range f).isLindelof_sUnion  <| forall_mem_range.2 h


theorem Set.Countable.isLindelof (hs : s.Countable) : IsLindelof s :=
  biUnion_of_singleton s ▸ hs.isLindelof_biUnion fun _ _ => isLindelof_singleton


theorem Set.Finite.isLindelof (hs : s.Finite) : IsLindelof s :=
  biUnion_of_singleton s ▸ hs.isLindelof_biUnion fun _ _ => isLindelof_singleton


theorem IsLindelof.countable_of_discrete [DiscreteTopology X] (hs : IsLindelof s) :
    s.Countable := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsLindelof s
    ⊢ s.Countable
  -/
  have : ∀ x : X, ({x} : Set X) ∈ 𝓝 x := by simp [nhds_discrete]
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsLindelof s
    this : ∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)
    ⊢ s.Countable
  -/
  rcases hs.elim_nhds_subcover (fun x => {x}) fun x _ => this x with ⟨t, ht, _, hssubt⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsLindelof s
    this : ∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)
    t : Set X
    ht : t.Countable
    left✝ : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hssubt : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => Singleton …
    ⊢ s.Countable
  -/
  rw [biUnion_of_singleton] at hssubt
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsLindelof s
    this : ∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)
    t : Set X
    ht : t.Countable
    left✝ : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hssubt : HasSubset.Subset s t
    ⊢ s.Countable
  -/
  exact ht.mono hssubt
  /-
    🎉 no goals
  -/


theorem isLindelof_iff_countable [DiscreteTopology X] : IsLindelof s ↔ s.Countable :=
  ⟨fun h => h.countable_of_discrete, fun h => h.isLindelof⟩


theorem IsLindelof.union (hs : IsLindelof s) (ht : IsLindelof t) : IsLindelof (s ∪ t) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsLindelof s
    ht : IsLindelof t
    ⊢ IsLindelof (Union.union s t)
  -/
  rw [union_eq_iUnion]; exact isLindelof_iUnion fun b => by cases b <;> assumption
                        /-
                          🎉 no goals
                        -/


protected theorem IsLindelof.insert (hs : IsLindelof s) (a) : IsLindelof (insert a s) :=
  isLindelof_singleton.union hs


/-- If `X` has a basis consisting of compact opens, then an open set in `X` is compact open iff
it is a finite union of some elements in the basis -/
theorem isLindelof_open_iff_eq_countable_iUnion_of_isTopologicalBasis (b : ι → Set X)
    (hb : IsTopologicalBasis (Set.range b)) (hb' : ∀ i, IsLindelof (b i)) (U : Set X) :
    IsLindelof U ∧ IsOpen U ↔ ∃ s : Set ι, s.Countable ∧ U = ⋃ i ∈ s, b i := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    b : ι → Set X
    hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
    hb' : ∀ (i : ι), IsLindelof (b i)
    U : Set X
    ⊢ Iff (And (IsLindelof U) (IsOpen U)) (Exists fun s => And s.Countable (Eq U ( …
  -/
  constructor
    /-
      case mp
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      U : Set X
      ⊢ And (IsLindelof U) (IsOpen U) → Exists fun s => And s.Countable (Eq U (Set.i …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mp.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      U : Set X
      h₁ : IsLindelof U
      h₂ : IsOpen U
      ⊢ Exists fun s => And s.Countable (Eq U (Set.iUnion fun i => Set.iUnion fun h  …
    -/
    obtain ⟨Y, f, rfl, hf⟩ := hb.open_eq_iUnion h₂
    /-
      case mp.intro.intro.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      Y : Type u
      f : Y → Set X
      hf : ∀ (i : Y), Membership.mem (Set.range b) (f i)
      h₁ : IsLindelof (Set.iUnion fun i => f i)
      h₂ : IsOpen (Set.iUnion fun i => f i)
      ⊢ Exists fun s => And s.Countable (Eq (Set.iUnion fun i => f i) (Set.iUnion fu …
    -/
    choose f' hf' using hf
    /-
      case mp.intro.intro.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      Y : Type u
      f : Y → Set X
      h₁ : IsLindelof (Set.iUnion fun i => f i)
      h₂ : IsOpen (Set.iUnion fun i => f i)
      f' : Y → ι
      hf' : ∀ (i : Y), Eq (b (f' i)) (f i)
      ⊢ Exists fun s => And s.Countable (Eq (Set.iUnion fun i => f i) (Set.iUnion fu …
    -/
    have : b ∘ f' = f := funext hf'
    /-
      case mp.intro.intro.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      Y : Type u
      f : Y → Set X
      h₁ : IsLindelof (Set.iUnion fun i => f i)
      h₂ : IsOpen (Set.iUnion fun i => f i)
      f' : Y → ι
      hf' : ∀ (i : Y), Eq (b (f' i)) (f i)
      this : Eq (Function.comp b f') f
      ⊢ Exists fun s => And s.Countable (Eq (Set.iUnion fun i => f i) (Set.iUnion fu …
    -/
    subst this
    obtain ⟨t, ht⟩ :=
      h₁.elim_countable_subcover (b ∘ f') (fun i => hb.isOpen (Set.mem_range_self _)) Subset.rfl
    /-
      case mp.intro.intro.intro.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      Y : Type u
      f' : Y → ι
      h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
      h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
      hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
      t : Set Y
      ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
      ⊢ Exists fun s => And s.Countable (Eq (Set.iUnion fun i => Function.comp b f'  …
    -/
    refine ⟨t.image f', Countable.image (ht.1) f', le_antisymm ?_ ?_⟩
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        ⊢ LE.le (Set.iUnion fun i => Function.comp b f' i) (Set.iUnion fun i => Set.iU …
      -/
    · refine Set.Subset.trans ht.2 ?_
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        ⊢ HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => Function.comp b f' …
      -/
      simp only [Set.iUnion_subset_iff]
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        ⊢ ∀ (i : Y), Membership.mem t i → HasSubset.Subset (Function.comp b f' i) (Set …
      -/
      intro i hi
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        i : Y
        hi : Membership.mem t i
        ⊢ HasSubset.Subset (Function.comp b f' i) (Set.iUnion fun i => Set.iUnion fun  …
      -/
      rw [← Set.iUnion_subtype (fun x : ι => x ∈ t.image f') fun i => b i.1]
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        i : Y
        hi : Membership.mem t i
        ⊢ HasSubset.Subset (Function.comp b f' i) (Set.iUnion fun x => b ↑x)
      -/
      exact Set.subset_iUnion (fun i : t.image f' => b i) ⟨_, mem_image_of_mem _ hi⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.refine_2
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        ⊢ LE.le (Set.iUnion fun i => Set.iUnion fun h => b i) (Set.iUnion fun i => Fun …
      -/
    · apply Set.iUnion₂_subset
      /-
        case mp.intro.intro.intro.intro.intro.refine_2.h
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        ⊢ ∀ (i : ι), Membership.mem (Set.image f' t) i → HasSubset.Subset (b i) (Set.i …
      -/
      rintro i hi
      /-
        case mp.intro.intro.intro.intro.intro.refine_2.h
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        i : ι
        hi : Membership.mem (Set.image f' t) i
        ⊢ HasSubset.Subset (b i) (Set.iUnion fun i => Function.comp b f' i)
      -/
      obtain ⟨j, -, rfl⟩ := (mem_image ..).mp hi
      /-
        case mp.intro.intro.intro.intro.intro.refine_2.h.intro.intro
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        Y : Type u
        f' : Y → ι
        h₁ : IsLindelof (Set.iUnion fun i => Function.comp b f' i)
        h₂ : IsOpen (Set.iUnion fun i => Function.comp b f' i)
        hf' : ∀ (i : Y), Eq (b (f' i)) (Function.comp b f' i)
        t : Set Y
        ht : And t.Countable (HasSubset.Subset (Set.iUnion fun i => Function.comp b f' …
        j : Y
        hi : Membership.mem (Set.image f' t) (f' j)
        ⊢ HasSubset.Subset (b (f' j)) (Set.iUnion fun i => Function.comp b f' i)
      -/
      exact Set.subset_iUnion (b ∘ f') j
      /-
        🎉 no goals
      -/
    /-
      case mpr
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      U : Set X
      ⊢ (Exists fun s => And s.Countable (Eq U (Set.iUnion fun i => Set.iUnion fun h …
    -/
  · rintro ⟨s, hs, rfl⟩
    /-
      case mpr.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsLindelof (b i)
      s : Set ι
      hs : s.Countable
      ⊢ And (IsLindelof (Set.iUnion fun i => Set.iUnion fun h => b i)) (IsOpen (Set. …
    -/
    constructor
      /-
        case mpr.intro.intro.left
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        s : Set ι
        hs : s.Countable
        ⊢ IsLindelof (Set.iUnion fun i => Set.iUnion fun h => b i)
      -/
    · exact hs.isLindelof_biUnion fun i _ => hb' i
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.right
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsLindelof (b i)
        s : Set ι
        hs : s.Countable
        ⊢ IsOpen (Set.iUnion fun i => Set.iUnion fun h => b i)
      -/
    · exact isOpen_biUnion fun i _ => hb.isOpen (Set.mem_range_self _)
      /-
        🎉 no goals
      -/


/-- `Filter.coLindelof` is the filter generated by complements to Lindelöf sets. -/
def Filter.coLindelof (X : Type*) [TopologicalSpace X] : Filter X :=
  --`Filter.coLindelof` is the filter generated by complements to Lindelöf sets.
  ⨅ (s : Set X) (_ : IsLindelof s), 𝓟 sᶜ


theorem hasBasis_coLindelof : (coLindelof X).HasBasis IsLindelof compl :=
  hasBasis_biInf_principal'
    (fun s hs t ht =>
      ⟨s ∪ t, hs.union ht, compl_subset_compl.2 subset_union_left,
        compl_subset_compl.2 subset_union_right⟩)
    ⟨∅, isLindelof_empty⟩


theorem mem_coLindelof : s ∈ coLindelof X ↔ ∃ t, IsLindelof t ∧ tᶜ ⊆ s :=
  hasBasis_coLindelof.mem_iff


theorem mem_coLindelof' : s ∈ coLindelof X ↔ ∃ t, IsLindelof t ∧ sᶜ ⊆ t :=
  mem_coLindelof.trans <| exists_congr fun _ => and_congr_right fun _ => compl_subset_comm


theorem _root_.IsLindelof.compl_mem_coLindelof (hs : IsLindelof s) : sᶜ ∈ coLindelof X :=
  hasBasis_coLindelof.mem_of_mem hs


theorem coLindelof_le_cofinite : coLindelof X ≤ cofinite := fun s hs =>
  compl_compl s ▸ hs.isLindelof.compl_mem_coLindelof


theorem Tendsto.isLindelof_insert_range_of_coLindelof {f : X → Y} {y}
    (hf : Tendsto f (coLindelof X) (𝓝 y)) (hfc : Continuous f) :
    IsLindelof (insert y (range f)) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
    hfc : Continuous f
    ⊢ IsLindelof (Insert.insert y (Set.range f))
  -/
  intro l hne _ hle
  /-
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    inst✝ : CountableInterFilter l
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  by_cases hy : ClusterPt y l
    /-
      case pos
      X : Type u
      Y : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      f : X → Y
      y : Y
      hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
      hfc : Continuous f
      l : Filter Y
      hne : l.NeBot
      inst✝ : CountableInterFilter l
      hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
      hy : ClusterPt y l
      ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
    -/
  · exact ⟨y, Or.inl rfl, hy⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    inst✝ : CountableInterFilter l
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    hy : Not (ClusterPt y l)
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  simp only [clusterPt_iff, not_forall, ← not_disjoint_iff_nonempty_inter, not_not] at hy
  /-
    case neg
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    inst✝ : CountableInterFilter l
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    hy : Exists fun x => Exists fun h => Exists fun x_1 => Exists fun h => Disjoin …
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  rcases hy with ⟨s, hsy, t, htl, hd⟩
  /-
    case neg.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    inst✝ : CountableInterFilter l
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    s : Set Y
    hsy : Membership.mem (nhds y) s
    t : Set Y
    htl : Membership.mem l t
    hd : Disjoint s t
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  rcases mem_coLindelof.1 (hf hsy) with ⟨K, hKc, hKs⟩
  have : f '' K ∈ l := by
    filter_upwards [htl, le_principal_iff.1 hle] with y hyt hyf
    rcases hyf with (rfl | ⟨x, rfl⟩)
    exacts [(hd.le_bot ⟨mem_of_mem_nhds hsy, hyt⟩).elim,
      mem_image_of_mem _ (not_not.1 fun hxK => hd.le_bot ⟨hKs hxK, hyt⟩)]
  /-
    case neg.intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    inst✝ : CountableInterFilter l
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    s : Set Y
    hsy : Membership.mem (nhds y) s
    t : Set Y
    htl : Membership.mem l t
    hd : Disjoint s t
    K : Set X
    hKc : IsLindelof K
    hKs : HasSubset.Subset (HasCompl.compl K) (Set.preimage f s)
    this : Membership.mem l (Set.image f K)
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  rcases hKc.image hfc (le_principal_iff.2 this) with ⟨y, hy, hyl⟩
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    y✝ : Y
    hf : Filter.Tendsto f (Filter.coLindelof X) (nhds y✝)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    inst✝ : CountableInterFilter l
    hle : LE.le l (Filter.principal (Insert.insert y✝ (Set.range f)))
    s : Set Y
    hsy : Membership.mem (nhds y✝) s
    t : Set Y
    htl : Membership.mem l t
    hd : Disjoint s t
    K : Set X
    hKc : IsLindelof K
    hKs : HasSubset.Subset (HasCompl.compl K) (Set.preimage f s)
    this : Membership.mem l (Set.image f K)
    y : Y
    hy : Membership.mem (Set.image f K) y
    hyl : ClusterPt y l
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y✝ (Set.range f)) x) (Clu …
  -/
  exact ⟨y, Or.inr <| image_subset_range _ _ hy, hyl⟩
  /-
    🎉 no goals
  -/


/-- `Filter.coclosedLindelof` is the filter generated by complements to closed Lindelof sets. -/
def Filter.coclosedLindelof (X : Type*) [TopologicalSpace X] : Filter X :=
  -- `Filter.coclosedLindelof` is the filter generated by complements to closed Lindelof sets.
  ⨅ (s : Set X) (_ : IsClosed s) (_ : IsLindelof s), 𝓟 sᶜ


theorem hasBasis_coclosedLindelof :
    (Filter.coclosedLindelof X).HasBasis (fun s => IsClosed s ∧ IsLindelof s) compl := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ (Filter.coclosedLindelof X).HasBasis (fun s => And (IsClosed s) (IsLindelof  …
  -/
  simp only [Filter.coclosedLindelof, iInf_and']
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ (iInf fun s => iInf fun h => Filter.principal (HasCompl.compl s)).HasBasis ( …
  -/
  refine hasBasis_biInf_principal' ?_ ⟨∅, isClosed_empty, isLindelof_empty⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ ∀ (i : Set X), And (IsClosed i) (IsLindelof i) → ∀ (j : Set X), And (IsClose …
  -/
  rintro s ⟨hs₁, hs₂⟩ t ⟨ht₁, ht₂⟩
  exact ⟨s ∪ t, ⟨⟨hs₁.union ht₁, hs₂.union ht₂⟩, compl_subset_compl.2 subset_union_left,
    compl_subset_compl.2 subset_union_right⟩⟩


theorem mem_coclosedLindelof : s ∈ coclosedLindelof X ↔
    ∃ t, IsClosed t ∧ IsLindelof t ∧ tᶜ ⊆ s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.coclosedLindelof X) s) (Exists fun t => And (IsC …
  -/
  simp only [hasBasis_coclosedLindelof.mem_iff, and_assoc]
  /-
    🎉 no goals
  -/


theorem mem_coclosed_Lindelof' : s ∈ coclosedLindelof X ↔
    ∃ t, IsClosed t ∧ IsLindelof t ∧ sᶜ ⊆ t := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.coclosedLindelof X) s) (Exists fun t => And (IsC …
  -/
  simp only [mem_coclosedLindelof, compl_subset_comm]
  /-
    🎉 no goals
  -/


theorem coLindelof_le_coclosedLindelof : coLindelof X ≤ coclosedLindelof X :=
  iInf_mono fun _ => le_iInf fun _ => le_rfl


theorem IsLindeof.compl_mem_coclosedLindelof_of_isClosed (hs : IsLindelof s) (hs' : IsClosed s) :
    sᶜ ∈ Filter.coclosedLindelof X :=
  hasBasis_coclosedLindelof.mem_of_mem ⟨hs', hs⟩


/-- X is a Lindelöf space iff every open cover has a countable subcover. -/
class LindelofSpace (X : Type*) [TopologicalSpace X] : Prop where
  /-- In a Lindelöf space, `Set.univ` is a Lindelöf set. -/
  isLindelof_univ : IsLindelof (univ : Set X)


instance (priority := 10) Subsingleton.lindelofSpace [Subsingleton X] : LindelofSpace X :=
  ⟨subsingleton_univ.isLindelof⟩


theorem isLindelof_univ_iff : IsLindelof (univ : Set X) ↔ LindelofSpace X :=
  ⟨fun h => ⟨h⟩, fun h => h.1⟩


theorem isLindelof_univ [h : LindelofSpace X] : IsLindelof (univ : Set X) :=
  h.isLindelof_univ


theorem cluster_point_of_Lindelof [LindelofSpace X] (f : Filter X) [NeBot f]
    [CountableInterFilter f] : ∃ x, ClusterPt x f := by
  /-
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : LindelofSpace X
    f : Filter X
    inst✝¹ : f.NeBot
    inst✝ : CountableInterFilter f
    ⊢ Exists fun x => ClusterPt x f
  -/
  simpa using isLindelof_univ (show f ≤ 𝓟 univ by simp)
  /-
    🎉 no goals
  -/


theorem LindelofSpace.elim_nhds_subcover [LindelofSpace X] (U : X → Set X) (hU : ∀ x, U x ∈ 𝓝 x) :
    ∃ t : Set X, t.Countable ∧ ⋃ x ∈ t, U x = univ := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : LindelofSpace X
    U : X → Set X
    hU : ∀ (x : X), Membership.mem (nhds x) (U x)
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  obtain ⟨t, tc, -, s⟩ := IsLindelof.elim_nhds_subcover isLindelof_univ U fun x _ => hU x
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : LindelofSpace X
    U : X → Set X
    hU : ∀ (x : X), Membership.mem (nhds x) (U x)
    t : Set X
    tc : t.Countable
    s : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun x => Set.iUnion fun h => …
  -/
  use t, tc
  /-
    case right
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : LindelofSpace X
    U : X → Set X
    hU : ∀ (x : X), Membership.mem (nhds x) (U x)
    t : Set X
    tc : t.Countable
    s : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => U x) Set.univ
  -/
  apply top_unique s
  /-
    🎉 no goals
  -/


theorem lindelofSpace_of_countable_subfamily_closed
    (h : ∀ {ι : Type u} (t : ι → Set X), (∀ i, IsClosed (t i)) → ⋂ i, t i = ∅ →
      ∃ u : Set ι, u.Countable ∧ ⋂ i ∈ u, t i = ∅) :
    LindelofSpace X where
                                                                          /-
                                                                            X : Type u
                                                                            inst✝ : TopologicalSpace X
                                                                            h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Set.iInt …
                                                                            ι✝ : Type u
                                                                            t : ι✝ → Set X
                                                                            ⊢ (∀ (i : ι✝), IsClosed (t i)) → Eq (Inter.inter Set.univ (Set.iInter fun i => …
                                                                          -/
  isLindelof_univ := isLindelof_of_countable_subfamily_closed fun t => by simpa using h t
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem IsClosed.isLindelof [LindelofSpace X] (h : IsClosed s) : IsLindelof s :=
  isLindelof_univ.of_isClosed_subset h (subset_univ _)


/-- A compact set `s` is Lindelöf. -/
theorem IsCompact.isLindelof (hs : IsCompact s) :
                       /-
                         X : Type u
                         inst✝ : TopologicalSpace X
                         s : Set X
                         hs : IsCompact s
                         ⊢ IsLindelof s
                       -/
    IsLindelof s := by tauto
                       /-
                         🎉 no goals
                       -/


/-- A σ-compact set `s` is Lindelöf -/
theorem IsSigmaCompact.isLindelof (hs : IsSigmaCompact s) :
    IsLindelof s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsSigmaCompact s
    ⊢ IsLindelof s
  -/
  rw [IsSigmaCompact] at hs
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : Exists fun K => And (∀ (n : Nat), IsCompact (K n)) (Eq (Set.iUnion fun n  …
    ⊢ IsLindelof s
  -/
  rcases hs with ⟨K, ⟨hc, huniv⟩⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    K : Nat → Set X
    hc : ∀ (n : Nat), IsCompact (K n)
    huniv : Eq (Set.iUnion fun n => K n) s
    ⊢ IsLindelof s
  -/
  rw [← huniv]
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    K : Nat → Set X
    hc : ∀ (n : Nat), IsCompact (K n)
    huniv : Eq (Set.iUnion fun n => K n) s
    ⊢ IsLindelof (Set.iUnion fun n => K n)
  -/
  have hl : ∀ n, IsLindelof (K n) := fun n ↦ IsCompact.isLindelof (hc n)
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    K : Nat → Set X
    hc : ∀ (n : Nat), IsCompact (K n)
    huniv : Eq (Set.iUnion fun n => K n) s
    hl : ∀ (n : Nat), IsLindelof (K n)
    ⊢ IsLindelof (Set.iUnion fun n => K n)
  -/
  exact isLindelof_iUnion hl
  /-
    🎉 no goals
  -/


/-- A compact space `X` is Lindelöf. -/
instance (priority := 100) [CompactSpace X] : LindelofSpace X :=
  { isLindelof_univ := isCompact_univ.isLindelof}


/-- A sigma-compact space `X` is Lindelöf. -/
instance (priority := 100) [SigmaCompactSpace X] : LindelofSpace X :=
  { isLindelof_univ := isSigmaCompact_univ.isLindelof}


/-- `X` is a non-Lindelöf topological space if it is not a Lindelöf space. -/
class NonLindelofSpace (X : Type*) [TopologicalSpace X] : Prop where
  /-- In a non-Lindelöf space, `Set.univ` is not a Lindelöf set. -/
  nonLindelof_univ : ¬IsLindelof (univ : Set X)


lemma nonLindelof_univ (X : Type*) [TopologicalSpace X] [NonLindelofSpace X] :
    ¬IsLindelof (univ : Set X) :=
  NonLindelofSpace.nonLindelof_univ


theorem IsLindelof.ne_univ [NonLindelofSpace X] (hs : IsLindelof s) : s ≠ univ := fun h ↦
  nonLindelof_univ X (h ▸ hs)


instance [NonLindelofSpace X] : NeBot (Filter.coLindelof X) := by
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s t : Set X
    inst✝ : NonLindelofSpace X
    ⊢ (Filter.coLindelof X).NeBot
  -/
  refine hasBasis_coLindelof.neBot_iff.2 fun {s} hs => ?_
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s✝ t : Set X
    inst✝ : NonLindelofSpace X
    s : Set X
    hs : IsLindelof s
    ⊢ (HasCompl.compl s).Nonempty
  -/
  contrapose hs
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s✝ t : Set X
    inst✝ : NonLindelofSpace X
    s : Set X
    hs : Not (HasCompl.compl s).Nonempty
    ⊢ Not (IsLindelof s)
  -/
  rw [not_nonempty_iff_eq_empty, compl_empty_iff] at hs
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s✝ t : Set X
    inst✝ : NonLindelofSpace X
    s : Set X
    hs : Eq s Set.univ
    ⊢ Not (IsLindelof s)
  -/
  rw [hs]
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s✝ t : Set X
    inst✝ : NonLindelofSpace X
    s : Set X
    hs : Eq s Set.univ
    ⊢ Not (IsLindelof Set.univ)
  -/
  exact nonLindelof_univ X
  /-
    🎉 no goals
  -/


@[simp]
theorem Filter.coLindelof_eq_bot [LindelofSpace X] : Filter.coLindelof X = ⊥ :=
  hasBasis_coLindelof.eq_bot_iff.mpr ⟨Set.univ, isLindelof_univ, Set.compl_univ⟩


instance [NonLindelofSpace X] : NeBot (Filter.coclosedLindelof X) :=
  neBot_of_le coLindelof_le_coclosedLindelof


theorem nonLindelofSpace_of_neBot (_ : NeBot (Filter.coLindelof X)) : NonLindelofSpace X :=
  ⟨fun h' => (Filter.nonempty_of_mem h'.compl_mem_coLindelof).ne_empty compl_univ⟩


theorem Filter.coLindelof_neBot_iff : NeBot (Filter.coLindelof X) ↔ NonLindelofSpace X :=
  ⟨nonLindelofSpace_of_neBot, fun _ => inferInstance⟩



theorem not_LindelofSpace_iff : ¬LindelofSpace X ↔ NonLindelofSpace X :=
  ⟨fun h₁ => ⟨fun h₂ => h₁ ⟨h₂⟩⟩, fun ⟨h₁⟩ ⟨h₂⟩ => h₁ h₂⟩


theorem countable_of_Lindelof_of_discrete [LindelofSpace X] [DiscreteTopology X] : Countable X :=
  countable_univ_iff.mp isLindelof_univ.countable_of_discrete


theorem countable_cover_nhds_interior [LindelofSpace X] {U : X → Set X} (hU : ∀ x, U x ∈ 𝓝 x) :
    ∃ t : Set X, t.Countable ∧ ⋃ x ∈ t, interior (U x) = univ :=
  let ⟨t, ht⟩ := isLindelof_univ.elim_countable_subcover (fun x => interior (U x))
    (fun _ => isOpen_interior) fun x _ => mem_iUnion.2 ⟨x, mem_interior_iff_mem_nhds.2 (hU x)⟩
  ⟨t, ⟨ht.1, univ_subset_iff.1 ht.2⟩⟩


theorem countable_cover_nhds [LindelofSpace X] {U : X → Set X} (hU : ∀ x, U x ∈ 𝓝 x) :
    ∃ t : Set X, t.Countable ∧ ⋃ x ∈ t, U x = univ :=
  let ⟨t, ht⟩ := countable_cover_nhds_interior hU
  ⟨t, ⟨ht.1, univ_subset_iff.1 <| ht.2.symm.subset.trans <|
    iUnion₂_mono fun _ _ => interior_subset⟩⟩


/-- The comap of the coLindelöf filter on `Y` by a continuous function `f : X → Y` is less than or
equal to the coLindelöf filter on `X`.
This is a reformulation of the fact that images of Lindelöf sets are Lindelöf. -/
theorem Filter.comap_coLindelof_le {f : X → Y} (hf : Continuous f) :
    (Filter.coLindelof Y).comap f ≤ Filter.coLindelof X := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    ⊢ LE.le (Filter.comap f (Filter.coLindelof Y)) (Filter.coLindelof X)
  -/
  rw [(hasBasis_coLindelof.comap f).le_basis_iff hasBasis_coLindelof]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    ⊢ ∀ (i' : Set X), IsLindelof i' → Exists fun i => And (IsLindelof i) (HasSubse …
  -/
  intro t ht
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    t : Set X
    ht : IsLindelof t
    ⊢ Exists fun i => And (IsLindelof i) (HasSubset.Subset (Set.preimage f (HasCom …
  -/
  refine ⟨f '' t, ht.image hf, ?_⟩
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    t : Set X
    ht : IsLindelof t
    ⊢ HasSubset.Subset (Set.preimage f (HasCompl.compl (Set.image f t))) (HasCompl …
  -/
  simpa using t.subset_preimage_image f
  /-
    🎉 no goals
  -/


theorem isLindelof_range [LindelofSpace X] {f : X → Y} (hf : Continuous f) :
                               /-
                                 X : Type u
                                 Y : Type v
                                 inst✝² : TopologicalSpace X
                                 inst✝¹ : TopologicalSpace Y
                                 inst✝ : LindelofSpace X
                                 f : X → Y
                                 hf : Continuous f
                                 ⊢ IsLindelof (Set.range f)
                               -/
    IsLindelof (range f) := by rw [← image_univ]; exact isLindelof_univ.image hf
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem isLindelof_diagonal [LindelofSpace X] : IsLindelof (diagonal X) :=
  @range_diag X ▸ isLindelof_range (continuous_id.prod_mk continuous_id)


/-- If `f : X → Y` is an inducing map, the image `f '' s` of a set `s` is Lindelöf
  if and only if `s` is compact. -/
theorem Topology.IsInducing.isLindelof_iff {f : X → Y} (hf : IsInducing f) :
    IsLindelof s ↔ IsLindelof (f '' s) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ Iff (IsLindelof s) (IsLindelof (Set.image f s))
  -/
  refine ⟨fun hs => hs.image hf.continuous, fun hs F F_ne_bot _ F_le => ?_⟩
  obtain ⟨_, ⟨x, x_in : x ∈ s, rfl⟩, hx : ClusterPt (f x) (map f F)⟩ :=
    hs ((map_mono F_le).trans_eq map_principal)
  /-
    case intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Topology.IsInducing f
    hs : IsLindelof (Set.image f s)
    F : Filter X
    F_ne_bot : F.NeBot
    x✝ : CountableInterFilter F
    F_le : LE.le F (Filter.principal s)
    x : X
    x_in : Membership.mem s x
    hx : ClusterPt (f x) (Filter.map f F)
    ⊢ Exists fun x => And (Membership.mem s x) (ClusterPt x F)
  -/
  exact ⟨x, x_in, hf.mapClusterPt_iff.1 hx⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias Inducing.isLindelof_iff := IsInducing.isLindelof_iff


/-- If `f : X → Y` is an `Embedding`, the image `f '' s` of a set `s` is Lindelöf
  if and only if `s` is Lindelöf. -/
theorem Topology.IsEmbedding.isLindelof_iff {f : X → Y} (hf : IsEmbedding f) :
    IsLindelof s ↔ IsLindelof (f '' s) := hf.isInducing.isLindelof_iff


@[deprecated (since := "2024-10-26")]
alias Embedding.isLindelof_iff := IsEmbedding.isLindelof_iff


/-- The preimage of a Lindelöf set under an inducing map is a Lindelöf set. -/
theorem Topology.IsInducing.isLindelof_preimage {f : X → Y} (hf : IsInducing f)
    (hf' : IsClosed (range f)) {K : Set Y} (hK : IsLindelof K) : IsLindelof (f ⁻¹' K) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    hf' : IsClosed (Set.range f)
    K : Set Y
    hK : IsLindelof K
    ⊢ IsLindelof (Set.preimage f K)
  -/
  replace hK := hK.inter_right hf'
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    hf' : IsClosed (Set.range f)
    K : Set Y
    hK : IsLindelof (Inter.inter K (Set.range f))
    ⊢ IsLindelof (Set.preimage f K)
  -/
  rwa [hf.isLindelof_iff, image_preimage_eq_inter_range]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.isLindelof_preimage := IsInducing.isLindelof_preimage


/-- The preimage of a Lindelöf set under a closed embedding is a Lindelöf set. -/
theorem Topology.IsClosedEmbedding.isLindelof_preimage {f : X → Y} (hf : IsClosedEmbedding f)
    {K : Set Y} (hK : IsLindelof K) : IsLindelof (f ⁻¹' K) :=
  hf.isInducing.isLindelof_preimage (hf.isClosed_range) hK


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.isLindelof_preimage := IsClosedEmbedding.isLindelof_preimage


/-- A closed embedding is proper, ie, inverse images of Lindelöf sets are contained in Lindelöf.
Moreover, the preimage of a Lindelöf set is Lindelöf, see
`Topology.IsClosedEmbedding.isLindelof_preimage`. -/
theorem Topology.IsClosedEmbedding.tendsto_coLindelof {f : X → Y} (hf : IsClosedEmbedding f) :
    Tendsto f (Filter.coLindelof X) (Filter.coLindelof Y) :=
  hasBasis_coLindelof.tendsto_right_iff.mpr fun _K hK =>
    (hf.isLindelof_preimage hK).compl_mem_coLindelof


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.tendsto_coLindelof := IsClosedEmbedding.tendsto_coLindelof


/-- Sets of subtype are Lindelöf iff the image under a coercion is. -/
theorem Subtype.isLindelof_iff {p : X → Prop} {s : Set { x // p x }} :
    IsLindelof s ↔ IsLindelof ((↑) '' s : Set X) :=
  IsEmbedding.subtypeVal.isLindelof_iff


theorem isLindelof_iff_isLindelof_univ : IsLindelof s ↔ IsLindelof (univ : Set s) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsLindelof s) (IsLindelof Set.univ)
  -/
  rw [Subtype.isLindelof_iff, image_univ, Subtype.range_coe]
  /-
    🎉 no goals
  -/


theorem isLindelof_iff_LindelofSpace : IsLindelof s ↔ LindelofSpace s :=
  isLindelof_iff_isLindelof_univ.trans isLindelof_univ_iff


lemma IsLindelof.of_coe [LindelofSpace s] : IsLindelof s := isLindelof_iff_LindelofSpace.mpr ‹_›


theorem IsLindelof.countable (hs : IsLindelof s) (hs' : DiscreteTopology s) : s.Countable :=
  countable_coe_iff.mp
  (@countable_of_Lindelof_of_discrete _ _ (isLindelof_iff_LindelofSpace.mp hs) hs')


protected theorem Topology.IsClosedEmbedding.nonLindelofSpace [NonLindelofSpace X] {f : X → Y}
    (hf : IsClosedEmbedding f) : NonLindelofSpace Y :=
  nonLindelofSpace_of_neBot hf.tendsto_coLindelof.neBot


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.nonLindelofSpace := IsClosedEmbedding.nonLindelofSpace


protected theorem Topology.IsClosedEmbedding.LindelofSpace [h : LindelofSpace Y] {f : X → Y}
    (hf : IsClosedEmbedding f) : LindelofSpace X :=
      /-
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        h : LindelofSpace Y
        f : X → Y
        hf : Topology.IsClosedEmbedding f
        ⊢ IsLindelof Set.univ
      -/
  ⟨by rw [hf.isInducing.isLindelof_iff, image_univ]; exact hf.isClosed_range.isLindelof⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.LindelofSpace := IsClosedEmbedding.LindelofSpace


/-- Countable topological spaces are Lindelof. -/
instance (priority := 100) Countable.LindelofSpace [Countable X] : LindelofSpace X where
  isLindelof_univ := countable_univ.isLindelof


/-- The disjoint union of two Lindelöf spaces is Lindelöf. -/
instance [LindelofSpace X] [LindelofSpace Y] : LindelofSpace (X ⊕ Y) where
  isLindelof_univ := by
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      s t : Set X
      inst✝¹ : LindelofSpace X
      inst✝ : LindelofSpace Y
      ⊢ IsLindelof Set.univ
    -/
    rw [← range_inl_union_range_inr]
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      s t : Set X
      inst✝¹ : LindelofSpace X
      inst✝ : LindelofSpace Y
      ⊢ IsLindelof (Union.union (Set.range Sum.inl) (Set.range Sum.inr))
    -/
    exact (isLindelof_range continuous_inl).union (isLindelof_range continuous_inr)
    /-
      🎉 no goals
    -/


instance {X : ι → Type*} [Countable ι] [∀ i, TopologicalSpace (X i)] [∀ i, LindelofSpace (X i)] :
    LindelofSpace (Σi, X i) where
  isLindelof_univ := by
    /-
      X✝ : Type u
      Y : Type v
      ι : Type u_1
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t : Set X✝
      X : ι → Type u_2
      inst✝² : Countable ι
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), LindelofSpace (X i)
      ⊢ IsLindelof Set.univ
    -/
    rw [Sigma.univ]
    /-
      X✝ : Type u
      Y : Type v
      ι : Type u_1
      inst✝⁴ : TopologicalSpace X✝
      inst✝³ : TopologicalSpace Y
      s t : Set X✝
      X : ι → Type u_2
      inst✝² : Countable ι
      inst✝¹ : (i : ι) → TopologicalSpace (X i)
      inst✝ : ∀ (i : ι), LindelofSpace (X i)
      ⊢ IsLindelof (Set.iUnion fun a => Set.range (Sigma.mk a))
    -/
    exact isLindelof_iUnion fun i => isLindelof_range continuous_sigmaMk
    /-
      🎉 no goals
    -/


instance Quot.LindelofSpace {r : X → X → Prop} [LindelofSpace X] : LindelofSpace (Quot r) where
  isLindelof_univ := by
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t : Set X
      r : X → X → Prop
      inst✝ : _root_.LindelofSpace X
      ⊢ IsLindelof Set.univ
    -/
    rw [← range_quot_mk]
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t : Set X
      r : X → X → Prop
      inst✝ : _root_.LindelofSpace X
      ⊢ IsLindelof (Set.range (Quot.mk r))
    -/
    exact isLindelof_range continuous_quot_mk
    /-
      🎉 no goals
    -/


instance Quotient.LindelofSpace {s : Setoid X} [LindelofSpace X] : LindelofSpace (Quotient s) :=
  Quot.LindelofSpace


/-- A continuous image of a Lindelöf set is a Lindelöf set within the codomain. -/
theorem LindelofSpace.of_continuous_surjective {f : X → Y} [LindelofSpace X] (hf : Continuous f)
    (hsur : Function.Surjective f) : LindelofSpace Y where
  isLindelof_univ := by
    /-
      X : Type u
      Y : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      f : X → Y
      inst✝ : LindelofSpace X
      hf : Continuous f
      hsur : Function.Surjective f
      ⊢ IsLindelof Set.univ
    -/
    rw [← Set.image_univ_of_surjective hsur]
    /-
      X : Type u
      Y : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      f : X → Y
      inst✝ : LindelofSpace X
      hf : Continuous f
      hsur : Function.Surjective f
      ⊢ IsLindelof (Set.image f Set.univ)
    -/
    exact IsLindelof.image (isLindelof_univ_iff.mpr ‹_›) hf
    /-
      🎉 no goals
    -/


/-- A set `s` is Hereditarily Lindelöf if every subset is a Lindelof set. We require this only
for open sets in the definition, and then conclude that this holds for all sets by ADD. -/
def IsHereditarilyLindelof (s : Set X) :=
  ∀ t ⊆ s, IsLindelof t


/-- Type class for Hereditarily Lindelöf spaces. -/
class HereditarilyLindelofSpace (X : Type*) [TopologicalSpace X] : Prop where
  /-- In a Hereditarily Lindelöf space, `Set.univ` is a Hereditarily Lindelöf set. -/
  isHereditarilyLindelof_univ : IsHereditarilyLindelof (univ : Set X)


lemma IsHereditarilyLindelof.isLindelof_subset (hs : IsHereditarilyLindelof s) (ht : t ⊆ s) :
    IsLindelof t := hs t ht


lemma IsHereditarilyLindelof.isLindelof (hs : IsHereditarilyLindelof s) :
    IsLindelof s := hs.isLindelof_subset Subset.rfl


instance (priority := 100) HereditarilyLindelof.to_Lindelof [HereditarilyLindelofSpace X] :
    LindelofSpace X where
  isLindelof_univ := HereditarilyLindelofSpace.isHereditarilyLindelof_univ.isLindelof


theorem HereditarilyLindelof_LindelofSets [HereditarilyLindelofSpace X] (s : Set X) :
    IsLindelof s := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : HereditarilyLindelofSpace X
    s : Set X
    ⊢ IsLindelof s
  -/
  apply HereditarilyLindelofSpace.isHereditarilyLindelof_univ
  /-
    case a
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : HereditarilyLindelofSpace X
    s : Set X
    ⊢ HasSubset.Subset s Set.univ
  -/
  exact subset_univ s
  /-
    🎉 no goals
  -/


instance (priority := 100) SecondCountableTopology.toHereditarilyLindelof
    [SecondCountableTopology X] : HereditarilyLindelofSpace X where
  isHereditarilyLindelof_univ t _ _ := by
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t✝ : Set X
      inst✝ : SecondCountableTopology X
      t : Set X
      x✝¹ : HasSubset.Subset t Set.univ
      x✝ : Filter X
      ⊢ ∀ [inst : x✝.NeBot] [inst : CountableInterFilter x✝], LE.le x✝ (Filter.princ …
    -/
    apply isLindelof_iff_countable_subcover.mpr
    /-
      case a
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t✝ : Set X
      inst✝ : SecondCountableTopology X
      t : Set X
      x✝¹ : HasSubset.Subset t Set.univ
      x✝ : Filter X
      ⊢ ∀ {ι : Type u} (U : ι → Set X), (∀ (i : ι), IsOpen (U i)) → HasSubset.Subset …
    -/
    intro ι U hι hcover
    /-
      case a
      X : Type u
      Y : Type v
      ι✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t✝ : Set X
      inst✝ : SecondCountableTopology X
      t : Set X
      x✝¹ : HasSubset.Subset t Set.univ
      x✝ : Filter X
      ι : Type u
      U : ι → Set X
      hι : ∀ (i : ι), IsOpen (U i)
      hcover : HasSubset.Subset t (Set.iUnion fun i => U i)
      ⊢ Exists fun t_1 => And t_1.Countable (HasSubset.Subset t (Set.iUnion fun i => …
    -/
    have := @isOpen_iUnion_countable X _ _ ι U hι
    /-
      case a
      X : Type u
      Y : Type v
      ι✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t✝ : Set X
      inst✝ : SecondCountableTopology X
      t : Set X
      x✝¹ : HasSubset.Subset t Set.univ
      x✝ : Filter X
      ι : Type u
      U : ι → Set X
      hι : ∀ (i : ι), IsOpen (U i)
      hcover : HasSubset.Subset t (Set.iUnion fun i => U i)
      this : Exists fun T => And T.Countable (Eq (Set.iUnion fun i => Set.iUnion fun …
      ⊢ Exists fun t_1 => And t_1.Countable (HasSubset.Subset t (Set.iUnion fun i => …
    -/
    rcases this with ⟨t, ⟨htc, htu⟩⟩
    /-
      case a.intro.intro
      X : Type u
      Y : Type v
      ι✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t✝¹ : Set X
      inst✝ : SecondCountableTopology X
      t✝ : Set X
      x✝¹ : HasSubset.Subset t✝ Set.univ
      x✝ : Filter X
      ι : Type u
      U : ι → Set X
      hι : ∀ (i : ι), IsOpen (U i)
      hcover : HasSubset.Subset t✝ (Set.iUnion fun i => U i)
      t : Set ι
      htc : t.Countable
      htu : Eq (Set.iUnion fun i => Set.iUnion fun h => U i) (Set.iUnion fun i => U i)
      ⊢ Exists fun t => And t.Countable (HasSubset.Subset t✝ (Set.iUnion fun i => Se …
    -/
    use t, htc
    /-
      case right
      X : Type u
      Y : Type v
      ι✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t✝¹ : Set X
      inst✝ : SecondCountableTopology X
      t✝ : Set X
      x✝¹ : HasSubset.Subset t✝ Set.univ
      x✝ : Filter X
      ι : Type u
      U : ι → Set X
      hι : ∀ (i : ι), IsOpen (U i)
      hcover : HasSubset.Subset t✝ (Set.iUnion fun i => U i)
      t : Set ι
      htc : t.Countable
      htu : Eq (Set.iUnion fun i => Set.iUnion fun h => U i) (Set.iUnion fun i => U i)
      ⊢ HasSubset.Subset t✝ (Set.iUnion fun i => Set.iUnion fun h => U i)
    -/
    exact subset_of_subset_of_eq hcover (id htu.symm)
    /-
      🎉 no goals
    -/


lemma eq_open_union_countable [HereditarilyLindelofSpace X] {ι : Type u} (U : ι → Set X)
    (h : ∀ i, IsOpen (U i)) : ∃ t : Set ι, t.Countable ∧ ⋃ i∈t, U i = ⋃ i, U i := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : HereditarilyLindelofSpace X
    ι : Type u
    U : ι → Set X
    h : ∀ (i : ι), IsOpen (U i)
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun i => Set.iUnion fun h => …
  -/
  have : IsLindelof (⋃ i, U i) := HereditarilyLindelof_LindelofSets (⋃ i, U i)
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : HereditarilyLindelofSpace X
    ι : Type u
    U : ι → Set X
    h : ∀ (i : ι), IsOpen (U i)
    this : IsLindelof (Set.iUnion fun i => U i)
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun i => Set.iUnion fun h => …
  -/
  rcases isLindelof_iff_countable_subcover.mp this U h (Eq.subset rfl) with ⟨t, ⟨htc, htu⟩⟩
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : HereditarilyLindelofSpace X
    ι : Type u
    U : ι → Set X
    h : ∀ (i : ι), IsOpen (U i)
    this : IsLindelof (Set.iUnion fun i => U i)
    t : Set ι
    htc : t.Countable
    htu : HasSubset.Subset (Set.iUnion fun i => U i) (Set.iUnion fun i => Set.iUni …
    ⊢ Exists fun t => And t.Countable (Eq (Set.iUnion fun i => Set.iUnion fun h => …
  -/
  use t, htc
  /-
    case right
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : HereditarilyLindelofSpace X
    ι : Type u
    U : ι → Set X
    h : ∀ (i : ι), IsOpen (U i)
    this : IsLindelof (Set.iUnion fun i => U i)
    t : Set ι
    htc : t.Countable
    htu : HasSubset.Subset (Set.iUnion fun i => U i) (Set.iUnion fun i => Set.iUni …
    ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => U i) (Set.iUnion fun i => U i)
  -/
  apply eq_of_subset_of_subset (iUnion₂_subset_iUnion (fun i ↦ i ∈ t) fun i ↦ U i) htu
  /-
    🎉 no goals
  -/


instance HereditarilyLindelof.lindelofSpace_subtype [HereditarilyLindelofSpace X] (p : X → Prop) :
    LindelofSpace {x // p x} := by
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s t : Set X
    inst✝ : HereditarilyLindelofSpace X
    p : X → Prop
    ⊢ LindelofSpace (Subtype fun x => p x)
  -/
  apply isLindelof_iff_LindelofSpace.mp
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s t : Set X
    inst✝ : HereditarilyLindelofSpace X
    p : X → Prop
    ⊢ IsLindelof p
  -/
  exact HereditarilyLindelof_LindelofSets fun x ↦ p x
  /-
    🎉 no goals
  -/


