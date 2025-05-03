lemma IsCompact.exists_clusterPt (hs : IsCompact s) {f : Filter X} [NeBot f] (hf : f ≤ 𝓟 s) :
    ∃ x ∈ s, ClusterPt x f := hs hf


lemma IsCompact.exists_mapClusterPt {ι : Type*} (hs : IsCompact s) {f : Filter ι} [NeBot f]
    {u : ι → X} (hf : Filter.map u f ≤ 𝓟 s) :
    ∃ x ∈ s, MapClusterPt x f u := hs hf


/-- The complement to a compact set belongs to a filter `f` if it belongs to each filter
`𝓝 x ⊓ f`, `x ∈ s`. -/
theorem IsCompact.compl_mem_sets (hs : IsCompact s) {f : Filter X} (hf : ∀ x ∈ s, sᶜ ∈ 𝓝 x ⊓ f) :
    sᶜ ∈ f := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Membership.mem (Min.min (nhds x) f) (HasC …
    ⊢ Membership.mem f (HasCompl.compl s)
  -/
  contrapose! hf
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : Not (Membership.mem f (HasCompl.compl s))
    ⊢ Exists fun x => And (Membership.mem s x) (Not (Membership.mem (Min.min (nhds …
  -/
  simp only [not_mem_iff_inf_principal_compl, compl_compl, inf_assoc] at hf ⊢
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : (Min.min f (Filter.principal s)).NeBot
    ⊢ Exists fun x => And (Membership.mem s x) (Min.min (nhds x) (Min.min f (Filte …
  -/
  exact @hs _ hf inf_le_right
  /-
    🎉 no goals
  -/


/-- The complement to a compact set belongs to a filter `f` if each `x ∈ s` has a neighborhood `t`
within `s` such that `tᶜ` belongs to `f`. -/
theorem IsCompact.compl_mem_sets_of_nhdsWithin (hs : IsCompact s) {f : Filter X}
    (hf : ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, tᶜ ∈ f) : sᶜ ∈ f := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    ⊢ Membership.mem f (HasCompl.compl s)
  -/
  refine hs.compl_mem_sets fun x hx => ?_
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x : X
    hx : Membership.mem s x
    ⊢ Membership.mem (Min.min (nhds x) f) (HasCompl.compl s)
  -/
  rcases hf x hx with ⟨t, ht, hst⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x : X
    hx : Membership.mem s x
    t : Set X
    ht : Membership.mem (nhdsWithin x s) t
    hst : Membership.mem f (HasCompl.compl t)
    ⊢ Membership.mem (Min.min (nhds x) f) (HasCompl.compl s)
  -/
  replace ht := mem_inf_principal.1 ht
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x : X
    hx : Membership.mem s x
    t : Set X
    hst : Membership.mem f (HasCompl.compl t)
    ht : Membership.mem (nhds x) (setOf fun x => Membership.mem s x → Membership.m …
    ⊢ Membership.mem (Min.min (nhds x) f) (HasCompl.compl s)
  -/
  apply mem_inf_of_inter ht hst
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x : X
    hx : Membership.mem s x
    t : Set X
    hst : Membership.mem f (HasCompl.compl t)
    ht : Membership.mem (nhds x) (setOf fun x => Membership.mem s x → Membership.m …
    ⊢ HasSubset.Subset (Inter.inter (setOf fun x => Membership.mem s x → Membershi …
  -/
  rintro x ⟨h₁, h₂⟩ hs
  /-
    case intro.intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs✝ : IsCompact s
    f : Filter X
    hf : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (nhds …
    x✝ : X
    hx : Membership.mem s x✝
    t : Set X
    hst : Membership.mem f (HasCompl.compl t)
    ht : Membership.mem (nhds x✝) (setOf fun x => Membership.mem s x → Membership. …
    x : X
    h₁ : Membership.mem (setOf fun x => Membership.mem s x → Membership.mem t x) x
    h₂ : Membership.mem (HasCompl.compl t) x
    hs : Membership.mem s x
    ⊢ False
  -/
  exact h₂ (h₁ hs)
  /-
    🎉 no goals
  -/


/-- If `p : Set X → Prop` is stable under restriction and union, and each point `x`
  of a compact set `s` has a neighborhood `t` within `s` such that `p t`, then `p s` holds. -/
@[elab_as_elim]
theorem IsCompact.induction_on (hs : IsCompact s) {p : Set X → Prop} (he : p ∅)
    (hmono : ∀ ⦃s t⦄, s ⊆ t → p t → p s) (hunion : ∀ ⦃s t⦄, p s → p t → p (s ∪ t))
    (hnhds : ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, p t) : p s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    p : Set X → Prop
    he : p EmptyCollection.emptyCollection
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → p t → p s
    hunion : ∀ ⦃s t : Set X⦄, p s → p t → p (Union.union s t)
    hnhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (n …
    ⊢ p s
  -/
  let f : Filter X := comk p he (fun _t ht _s hsub ↦ hmono hsub ht) (fun _s hs _t ht ↦ hunion hs ht)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    p : Set X → Prop
    he : p EmptyCollection.emptyCollection
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → p t → p s
    hunion : ∀ ⦃s t : Set X⦄, p s → p t → p (Union.union s t)
    hnhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (n …
    f : Filter X := Filter.comk p he ⋯ ⋯
    ⊢ p s
  -/
  have : sᶜ ∈ f := hs.compl_mem_sets_of_nhdsWithin (by simpa [f] using hnhds)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    p : Set X → Prop
    he : p EmptyCollection.emptyCollection
    hmono : ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → p t → p s
    hunion : ∀ ⦃s t : Set X⦄, p s → p t → p (Union.union s t)
    hnhds : ∀ (x : X), Membership.mem s x → Exists fun t => And (Membership.mem (n …
    f : Filter X := Filter.comk p he ⋯ ⋯
    this : Membership.mem f (HasCompl.compl s)
    ⊢ p s
  -/
  rwa [← compl_compl s]
  /-
    🎉 no goals
  -/


/-- The intersection of a compact set and a closed set is a compact set. -/
theorem IsCompact.inter_right (hs : IsCompact s) (ht : IsClosed t) : IsCompact (s ∩ t) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsClosed t
    ⊢ IsCompact (Inter.inter s t)
  -/
  intro f hnf hstf
  obtain ⟨x, hsx, hx⟩ : ∃ x ∈ s, ClusterPt x f :=
    hs (le_trans hstf (le_principal_iff.2 inter_subset_left))
  have : x ∈ t := ht.mem_of_nhdsWithin_neBot <|
    hx.mono <| le_trans hstf (le_principal_iff.2 inter_subset_right)
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsClosed t
    f : Filter X
    hnf : f.NeBot
    hstf : LE.le f (Filter.principal (Inter.inter s t))
    x : X
    hsx : Membership.mem s x
    hx : ClusterPt x f
    this : Membership.mem t x
    ⊢ Exists fun x => And (Membership.mem (Inter.inter s t) x) (ClusterPt x f)
  -/
  exact ⟨x, ⟨hsx, this⟩, hx⟩
  /-
    🎉 no goals
  -/


/-- The intersection of a closed set and a compact set is a compact set. -/
theorem IsCompact.inter_left (ht : IsCompact t) (hs : IsClosed s) : IsCompact (s ∩ t) :=
  inter_comm t s ▸ ht.inter_right hs


/-- The set difference of a compact set and an open set is a compact set. -/
theorem IsCompact.diff (hs : IsCompact s) (ht : IsOpen t) : IsCompact (s \ t) :=
  hs.inter_right (isClosed_compl_iff.mpr ht)


/-- A closed subset of a compact set is a compact set. -/
theorem IsCompact.of_isClosed_subset (hs : IsCompact s) (ht : IsClosed t) (h : t ⊆ s) :
    IsCompact t :=
  inter_eq_self_of_subset_right h ▸ hs.inter_right ht


theorem IsCompact.image_of_continuousOn {f : X → Y} (hs : IsCompact s) (hf : ContinuousOn f s) :
    IsCompact (f '' s) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsCompact s
    hf : ContinuousOn f s
    ⊢ IsCompact (Set.image f s)
  -/
  intro l lne ls
  have : NeBot (l.comap f ⊓ 𝓟 s) :=
    comap_inf_principal_neBot_of_image_mem lne (le_principal_iff.1 ls)
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsCompact s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
    ls : LE.le l (Filter.principal (Set.image f s))
    this : (Min.min (Filter.comap f l) (Filter.principal s)).NeBot
    ⊢ Exists fun x => And (Membership.mem (Set.image f s) x) (ClusterPt x l)
  -/
  obtain ⟨x, hxs, hx⟩ : ∃ x ∈ s, ClusterPt x (l.comap f ⊓ 𝓟 s) := @hs _ this inf_le_right
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsCompact s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
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
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsCompact s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
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
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hs : IsCompact s
    hf : ContinuousOn f s
    l : Filter Y
    lne : l.NeBot
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


theorem IsCompact.image {f : X → Y} (hs : IsCompact s) (hf : Continuous f) : IsCompact (f '' s) :=
  hs.image_of_continuousOn hf.continuousOn


theorem IsCompact.adherence_nhdset {f : Filter X} (hs : IsCompact s) (hf₂ : f ≤ 𝓟 s)
    (ht₁ : IsOpen t) (ht₂ : ∀ x ∈ s, ClusterPt x f → x ∈ t) : t ∈ f :=
  Classical.by_cases mem_of_eq_bot fun (this : f ⊓ 𝓟 tᶜ ≠ ⊥) =>
    let ⟨x, hx, (hfx : ClusterPt x <| f ⊓ 𝓟 tᶜ)⟩ := @hs _ ⟨this⟩ <| inf_le_of_left_le hf₂
    have : x ∈ t := ht₂ x hx hfx.of_inf_left
    have : tᶜ ∩ t ∈ 𝓝[tᶜ] x := inter_mem_nhdsWithin _ (IsOpen.mem_nhds ht₁ this)
    have A : 𝓝[tᶜ] x = ⊥ := empty_mem_iff_bot.1 <| compl_inter_self t ▸ this
    have : 𝓝[tᶜ] x ≠ ⊥ := hfx.of_inf_right.ne
    absurd A this


theorem isCompact_iff_ultrafilter_le_nhds :
    IsCompact s ↔ ∀ f : Ultrafilter X, ↑f ≤ 𝓟 s → ∃ x ∈ s, ↑f ≤ 𝓝 x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsCompact s) (∀ (f : Ultrafilter X), LE.le (↑f) (Filter.principal s) →  …
  -/
  refine (forall_neBot_le_iff ?_).trans ?_
    /-
      case refine_1
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ Monotone fun f => Exists fun x => And (Membership.mem s x) (ClusterPt x f)
    -/
  · rintro f g hle ⟨x, hxs, hxf⟩
    /-
      case refine_1.intro.intro
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      f g : Filter X
      hle : LE.le f g
      x : X
      hxs : Membership.mem s x
      hxf : ClusterPt x f
      ⊢ Exists fun x => And (Membership.mem s x) (ClusterPt x g)
    -/
    exact ⟨x, hxs, hxf.mono hle⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ Iff (∀ (f : Ultrafilter X), LE.le (↑f) (Filter.principal s) → Exists fun x = …
    -/
  · simp only [Ultrafilter.clusterPt_iff]
    /-
      🎉 no goals
    -/


alias ⟨IsCompact.ultrafilter_le_nhds, _⟩ := isCompact_iff_ultrafilter_le_nhds


theorem isCompact_iff_ultrafilter_le_nhds' :
    IsCompact s ↔ ∀ f : Ultrafilter X, s ∈ f → ∃ x ∈ s, ↑f ≤ 𝓝 x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsCompact s) (∀ (f : Ultrafilter X), Membership.mem f s → Exists fun x  …
  -/
  simp only [isCompact_iff_ultrafilter_le_nhds, le_principal_iff, Ultrafilter.mem_coe]
  /-
    🎉 no goals
  -/


alias ⟨IsCompact.ultrafilter_le_nhds', _⟩ := isCompact_iff_ultrafilter_le_nhds'


/-- If a compact set belongs to a filter and this filter has a unique cluster point `y` in this set,
then the filter is less than or equal to `𝓝 y`. -/
lemma IsCompact.le_nhds_of_unique_clusterPt (hs : IsCompact s) {l : Filter X} {y : X}
    (hmem : s ∈ l) (h : ∀ x ∈ s, ClusterPt x l → x = y) : l ≤ 𝓝 y := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    l : Filter X
    y : X
    hmem : Membership.mem l s
    h : ∀ (x : X), Membership.mem s x → ClusterPt x l → Eq x y
    ⊢ LE.le l (nhds y)
  -/
  refine le_iff_ultrafilter.2 fun f hf ↦ ?_
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    l : Filter X
    y : X
    hmem : Membership.mem l s
    h : ∀ (x : X), Membership.mem s x → ClusterPt x l → Eq x y
    f : Ultrafilter X
    hf : LE.le (↑f) l
    ⊢ LE.le (↑f) (nhds y)
  -/
  rcases hs.ultrafilter_le_nhds' f (hf hmem) with ⟨x, hxs, hx⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    l : Filter X
    y : X
    hmem : Membership.mem l s
    h : ∀ (x : X), Membership.mem s x → ClusterPt x l → Eq x y
    f : Ultrafilter X
    hf : LE.le (↑f) l
    x : X
    hxs : Membership.mem s x
    hx : LE.le (↑f) (nhds x)
    ⊢ LE.le (↑f) (nhds y)
  -/
  convert ← hx
  /-
    case h.e'_4.h.e'_3
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    l : Filter X
    y : X
    hmem : Membership.mem l s
    h : ∀ (x : X), Membership.mem s x → ClusterPt x l → Eq x y
    f : Ultrafilter X
    hf : LE.le (↑f) l
    x : X
    hxs : Membership.mem s x
    hx : LE.le (↑f) (nhds x)
    ⊢ Eq x y
  -/
  exact h x hxs (.mono (.of_le_nhds hx) hf)
  /-
    🎉 no goals
  -/


/-- If values of `f : Y → X` belong to a compact set `s` eventually along a filter `l`
and `y` is a unique `MapClusterPt` for `f` along `l` in `s`,
then `f` tends to `𝓝 y` along `l`. -/
lemma IsCompact.tendsto_nhds_of_unique_mapClusterPt {Y} {l : Filter Y} {y : X} {f : Y → X}
    (hs : IsCompact s) (hmem : ∀ᶠ x in l, f x ∈ s) (h : ∀ x ∈ s, MapClusterPt x l f → x = y) :
    Tendsto f l (𝓝 y) :=
  hs.le_nhds_of_unique_clusterPt (mem_map.2 hmem) h


/-- For every open directed cover of a compact set, there exists a single element of the
cover which itself includes the set. -/
theorem IsCompact.elim_directed_cover {ι : Type v} [hι : Nonempty ι] (hs : IsCompact s)
    (U : ι → Set X) (hUo : ∀ i, IsOpen (U i)) (hsU : s ⊆ ⋃ i, U i) (hdU : Directed (· ⊆ ·) U) :
    ∃ i, s ⊆ U i :=
  hι.elim fun i₀ =>
    IsCompact.induction_on hs ⟨i₀, empty_subset _⟩ (fun _ _ hs ⟨i, hi⟩ => ⟨i, hs.trans hi⟩)
      (fun _ _ ⟨i, hi⟩ ⟨j, hj⟩ =>
        let ⟨k, hki, hkj⟩ := hdU i j
        ⟨k, union_subset (Subset.trans hi hki) (Subset.trans hj hkj)⟩)
      fun _x hx =>
      let ⟨i, hi⟩ := mem_iUnion.1 (hsU hx)
      ⟨U i, mem_nhdsWithin_of_mem_nhds (IsOpen.mem_nhds (hUo i) hi), i, Subset.refl _⟩


/-- For every open cover of a compact set, there exists a finite subcover. -/
theorem IsCompact.elim_finite_subcover {ι : Type v} (hs : IsCompact s) (U : ι → Set X)
    (hUo : ∀ i, IsOpen (U i)) (hsU : s ⊆ ⋃ i, U i) : ∃ t : Finset ι, s ⊆ ⋃ i ∈ t, U i :=
  hs.elim_directed_cover _ (fun _ => isOpen_biUnion fun i _ => hUo i)
    (iUnion_eq_iUnion_finset U ▸ hsU)
    (directed_of_isDirected_le fun _ _ h => biUnion_subset_biUnion_left h)


lemma IsCompact.elim_nhds_subcover_nhdsSet' (hs : IsCompact s) (U : ∀ x ∈ s, Set X)
    (hU : ∀ x hx, U x hx ∈ 𝓝 x) : ∃ t : Finset s, (⋃ x ∈ t, U x.1 x.2) ∈ 𝓝ˢ s := by
  rcases hs.elim_finite_subcover (fun x : s ↦ interior (U x x.2)) (fun _ ↦ isOpen_interior)
    fun x hx ↦ mem_iUnion.2 ⟨⟨x, hx⟩, mem_interior_iff_mem_nhds.2 <| hU _ _⟩ with ⟨t, hst⟩
  /-
    case intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    t : Finset ↑s
    hst : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => interior (U  …
    ⊢ Exists fun t => Membership.mem (nhdsSet s) (Set.iUnion fun x => Set.iUnion f …
  -/
  refine ⟨t, mem_nhdsSet_iff_forall.2 fun x hx ↦ ?_⟩
  /-
    case intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    t : Finset ↑s
    hst : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => interior (U  …
    x : X
    hx : Membership.mem s x
    ⊢ Membership.mem (nhds x) (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
  -/
  rcases mem_iUnion₂.1 (hst hx) with ⟨y, hyt, hy⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    t : Finset ↑s
    hst : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => interior (U  …
    x : X
    hx : Membership.mem s x
    y : ↑s
    hyt : Membership.mem t y
    hy : Membership.mem (interior (U ↑y ⋯)) x
    ⊢ Membership.mem (nhds x) (Set.iUnion fun x => Set.iUnion fun h => U ↑x ⋯)
  -/
  refine mem_of_superset ?_ (subset_biUnion_of_mem hyt)
  /-
    case intro.intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    U : (x : X) → Membership.mem s x → Set X
    hU : ∀ (x : X) (hx : Membership.mem s x), Membership.mem (nhds x) (U x hx)
    t : Finset ↑s
    hst : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => interior (U  …
    x : X
    hx : Membership.mem s x
    y : ↑s
    hyt : Membership.mem t y
    hy : Membership.mem (interior (U ↑y ⋯)) x
    ⊢ Membership.mem (nhds x) (U ↑y ⋯)
  -/
  exact mem_interior_iff_mem_nhds.1 hy
  /-
    🎉 no goals
  -/


lemma IsCompact.elim_nhds_subcover_nhdsSet (hs : IsCompact s) {U : X → Set X}
    (hU : ∀ x ∈ s, U x ∈ 𝓝 x) : ∃ t : Finset X, (∀ x ∈ t, x ∈ s) ∧ (⋃ x ∈ t, U x) ∈ 𝓝ˢ s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsCompact s
    U : X → Set X
    hU : ∀ (x : X), Membership.mem s x → Membership.mem (nhds x) (U x)
    ⊢ Exists fun t => And (∀ (x : X), Membership.mem t x → Membership.mem s x) (Me …
  -/
  let ⟨t, ht⟩ := hs.elim_nhds_subcover_nhdsSet' (fun x _ => U x) hU
  classical
  exact ⟨t.image (↑), fun x hx =>
    let ⟨y, _, hyx⟩ := Finset.mem_image.1 hx
    hyx ▸ y.2,
    by rwa [Finset.set_biUnion_finset_image]⟩


theorem IsCompact.elim_nhds_subcover' (hs : IsCompact s) (U : ∀ x ∈ s, Set X)
    (hU : ∀ x (hx : x ∈ s), U x ‹x ∈ s› ∈ 𝓝 x) : ∃ t : Finset s, s ⊆ ⋃ x ∈ t, U (x : s) x.2 :=
  (hs.elim_nhds_subcover_nhdsSet' U hU).imp fun _ ↦ subset_of_mem_nhdsSet


theorem IsCompact.elim_nhds_subcover (hs : IsCompact s) (U : X → Set X) (hU : ∀ x ∈ s, U x ∈ 𝓝 x) :
    ∃ t : Finset X, (∀ x ∈ t, x ∈ s) ∧ s ⊆ ⋃ x ∈ t, U x :=
  (hs.elim_nhds_subcover_nhdsSet hU).imp fun _ h ↦ h.imp_right subset_of_mem_nhdsSet


/-- The neighborhood filter of a compact set is disjoint with a filter `l` if and only if the
neighborhood filter of each point of this set is disjoint with `l`. -/
theorem IsCompact.disjoint_nhdsSet_left {l : Filter X} (hs : IsCompact s) :
    Disjoint (𝓝ˢ s) l ↔ ∀ x ∈ s, Disjoint (𝓝 x) l := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    ⊢ Iff (Disjoint (nhdsSet s) l) (∀ (x : X), Membership.mem s x → Disjoint (nhds …
  -/
  refine ⟨fun h x hx => h.mono_left <| nhds_le_nhdsSet hx, fun H => ?_⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    ⊢ Disjoint (nhdsSet s) l
  -/
  choose! U hxU hUl using fun x hx => (nhds_basis_opens x).disjoint_iff_left.1 (H x hx)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem s x → And (Membership.mem (U x) x) (IsOpen (U  …
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    ⊢ Disjoint (nhdsSet s) l
  -/
  choose hxU hUo using hxU
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    hxU : ∀ (x : X), Membership.mem s x → Membership.mem (U x) x
    hUo : ∀ (x : X), Membership.mem s x → IsOpen (U x)
    ⊢ Disjoint (nhdsSet s) l
  -/
  rcases hs.elim_nhds_subcover U fun x hx => (hUo x hx).mem_nhds (hxU x hx) with ⟨t, hts, hst⟩
  refine (hasBasis_nhdsSet _).disjoint_iff_left.2
    ⟨⋃ x ∈ t, U x, ⟨isOpen_biUnion fun x hx => hUo x (hts x hx), hst⟩, ?_⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    hxU : ∀ (x : X), Membership.mem s x → Membership.mem (U x) x
    hUo : ∀ (x : X), Membership.mem s x → IsOpen (U x)
    t : Finset X
    hts : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hst : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Membership.mem l (HasCompl.compl (Set.iUnion fun x => Set.iUnion fun h => U  …
  -/
  rw [compl_iUnion₂, biInter_finset_mem]
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    H : ∀ (x : X), Membership.mem s x → Disjoint (nhds x) l
    U : X → Set X
    hUl : ∀ (x : X), Membership.mem s x → Membership.mem l (HasCompl.compl (U x))
    hxU : ∀ (x : X), Membership.mem s x → Membership.mem (U x) x
    hUo : ∀ (x : X), Membership.mem s x → IsOpen (U x)
    t : Finset X
    hts : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hst : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ ∀ (i : X), Membership.mem t i → Membership.mem l (HasCompl.compl (U i))
  -/
  exact fun x hx => hUl x (hts x hx)
  /-
    🎉 no goals
  -/


/-- A filter `l` is disjoint with the neighborhood filter of a compact set if and only if it is
disjoint with the neighborhood filter of each point of this set. -/
theorem IsCompact.disjoint_nhdsSet_right {l : Filter X} (hs : IsCompact s) :
    Disjoint l (𝓝ˢ s) ↔ ∀ x ∈ s, Disjoint l (𝓝 x) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    l : Filter X
    hs : IsCompact s
    ⊢ Iff (Disjoint l (nhdsSet s)) (∀ (x : X), Membership.mem s x → Disjoint l (nh …
  -/
  simpa only [disjoint_comm] using hs.disjoint_nhdsSet_left
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: reformulate using `Disjoint`

/-- For every directed family of closed sets whose intersection avoids a compact set,
there exists a single element of the family which itself avoids this compact set. -/
theorem IsCompact.elim_directed_family_closed {ι : Type v} [Nonempty ι] (hs : IsCompact s)
    (t : ι → Set X) (htc : ∀ i, IsClosed (t i)) (hst : (s ∩ ⋂ i, t i) = ∅)
    (hdt : Directed (· ⊇ ·) t) : ∃ i : ι, s ∩ t i = ∅ :=
  let ⟨t, ht⟩ :=
    hs.elim_directed_cover (compl ∘ t) (fun i => (htc i).isOpen_compl)
      (by
        simpa only [subset_def, not_forall, eq_empty_iff_forall_not_mem, mem_iUnion, exists_prop,
          mem_inter_iff, not_and, mem_iInter, mem_compl_iff] using hst)
      (hdt.mono_comp _ fun _ _ => compl_subset_compl.mpr)
  ⟨t, by
    simpa only [subset_def, not_forall, eq_empty_iff_forall_not_mem, mem_iUnion, exists_prop,
      mem_inter_iff, not_and, mem_iInter, mem_compl_iff] using ht⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: reformulate using `Disjoint`

/-- For every family of closed sets whose intersection avoids a compact set,
there exists a finite subfamily whose intersection avoids this compact set. -/
theorem IsCompact.elim_finite_subfamily_closed {ι : Type v} (hs : IsCompact s)
    (t : ι → Set X) (htc : ∀ i, IsClosed (t i)) (hst : (s ∩ ⋂ i, t i) = ∅) :
    ∃ u : Finset ι, (s ∩ ⋂ i ∈ u, t i) = ∅ :=
  hs.elim_directed_family_closed _ (fun _ ↦ isClosed_biInter fun _ _ ↦ htc _)
        /-
          X : Type u
          inst✝ : TopologicalSpace X
          s : Set X
          ι : Type v
          hs : IsCompact s
          t : ι → Set X
          htc : ∀ (i : ι), IsClosed (t i)
          hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
          ⊢ Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun i_1 => Set.iInter fun  …
        -/
    (by rwa [← iInter_eq_iInter_finset])
        /-
          🎉 no goals
        -/
    (directed_of_isDirected_le fun _ _ h ↦ biInter_subset_biInter_left h)


/-- If `s` is a compact set in a topological space `X` and `f : ι → Set X` is a locally finite
family of sets, then `f i ∩ s` is nonempty only for a finitely many `i`. -/
theorem LocallyFinite.finite_nonempty_inter_compact {f : ι → Set X}
    (hf : LocallyFinite f) (hs : IsCompact s) : { i | (f i ∩ s).Nonempty }.Finite := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    f : ι → Set X
    hf : LocallyFinite f
    hs : IsCompact s
    ⊢ (setOf fun i => (Inter.inter (f i) s).Nonempty).Finite
  -/
  choose U hxU hUf using hf
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    f : ι → Set X
    hs : IsCompact s
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem (nhds x) (U x)
    hUf : ∀ (x : X), (setOf fun i => (Inter.inter (f i) (U x)).Nonempty).Finite
    ⊢ (setOf fun i => (Inter.inter (f i) s).Nonempty).Finite
  -/
  rcases hs.elim_nhds_subcover U fun x _ => hxU x with ⟨t, -, hsU⟩
  /-
    case intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    f : ι → Set X
    hs : IsCompact s
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem (nhds x) (U x)
    hUf : ∀ (x : X), (setOf fun i => (Inter.inter (f i) (U x)).Nonempty).Finite
    t : Finset X
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ (setOf fun i => (Inter.inter (f i) s).Nonempty).Finite
  -/
  refine (t.finite_toSet.biUnion fun x _ => hUf x).subset ?_
  /-
    case intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    f : ι → Set X
    hs : IsCompact s
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem (nhds x) (U x)
    hUf : ∀ (x : X), (setOf fun i => (Inter.inter (f i) (U x)).Nonempty).Finite
    t : Finset X
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ HasSubset.Subset (setOf fun i => (Inter.inter (f i) s).Nonempty) (Set.iUnion …
  -/
  rintro i ⟨x, hx⟩
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    f : ι → Set X
    hs : IsCompact s
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem (nhds x) (U x)
    hUf : ∀ (x : X), (setOf fun i => (Inter.inter (f i) (U x)).Nonempty).Finite
    t : Finset X
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    i : ι
    x : X
    hx : Membership.mem (Inter.inter (f i) s) x
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => setOf fun i_1 => (In …
  -/
  rcases mem_iUnion₂.1 (hsU hx.2) with ⟨c, hct, hcx⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    f : ι → Set X
    hs : IsCompact s
    U : X → Set X
    hxU : ∀ (x : X), Membership.mem (nhds x) (U x)
    hUf : ∀ (x : X), (setOf fun i => (Inter.inter (f i) (U x)).Nonempty).Finite
    t : Finset X
    hsU : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => U x)
    i : ι
    x : X
    hx : Membership.mem (Inter.inter (f i) s) x
    c : X
    hct : Membership.mem t c
    hcx : Membership.mem (U c) x
    ⊢ Membership.mem (Set.iUnion fun i => Set.iUnion fun h => setOf fun i_1 => (In …
  -/
  exact mem_biUnion hct ⟨x, hx.1, hcx⟩
  /-
    🎉 no goals
  -/


/-- To show that a compact set intersects the intersection of a family of closed sets,
  it is sufficient to show that it intersects every finite subfamily. -/
theorem IsCompact.inter_iInter_nonempty {ι : Type v} (hs : IsCompact s) (t : ι → Set X)
    (htc : ∀ i, IsClosed (t i)) (hst : ∀ u : Finset ι, (s ∩ ⋂ i ∈ u, t i).Nonempty) :
    (s ∩ ⋂ i, t i).Nonempty := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsCompact s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : ∀ (u : Finset ι), (Inter.inter s (Set.iInter fun i => Set.iInter fun h = …
    ⊢ (Inter.inter s (Set.iInter fun i => t i)).Nonempty
  -/
  contrapose! hst
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ι : Type v
    hs : IsCompact s
    t : ι → Set X
    htc : ∀ (i : ι), IsClosed (t i)
    hst : Eq (Inter.inter s (Set.iInter fun i => t i)) EmptyCollection.emptyCollec …
    ⊢ Exists fun u => Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => t …
  -/
  exact hs.elim_finite_subfamily_closed t htc hst
  /-
    🎉 no goals
  -/


/-- Cantor's intersection theorem for `iInter`:
the intersection of a directed family of nonempty compact closed sets is nonempty. -/
theorem IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed
    {ι : Type v} [hι : Nonempty ι] (t : ι → Set X) (htd : Directed (· ⊇ ·) t)
    (htn : ∀ i, (t i).Nonempty) (htc : ∀ i, IsCompact (t i)) (htcl : ∀ i, IsClosed (t i)) :
    (⋂ i, t i).Nonempty := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htn : ∀ (i : ι), (t i).Nonempty
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    ⊢ (Set.iInter fun i => t i).Nonempty
  -/
  let i₀ := hι.some
  suffices (t i₀ ∩ ⋂ i, t i).Nonempty by
    rwa [inter_eq_right.mpr (iInter_subset _ i₀)] at this
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htn : ∀ (i : ι), (t i).Nonempty
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    ⊢ (Inter.inter (t i₀) (Set.iInter fun i => t i)).Nonempty
  -/
  simp only [nonempty_iff_ne_empty] at htn ⊢
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    htn : ∀ (i : ι), Ne (t i) EmptyCollection.emptyCollection
    ⊢ Ne (Inter.inter (t i₀) (Set.iInter fun i => t i)) EmptyCollection.emptyColle …
  -/
  apply mt ((htc i₀).elim_directed_family_closed t htcl)
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    htn : ∀ (i : ι), Ne (t i) EmptyCollection.emptyCollection
    ⊢ Not (Directed (fun x1 x2 => Superset x1 x2) t → Exists fun i => Eq (Inter.in …
  -/
  push_neg
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    htn : ∀ (i : ι), Ne (t i) EmptyCollection.emptyCollection
    ⊢ And (Directed (fun x1 x2 => Superset x1 x2) t) (∀ (i : ι), (Inter.inter (t i …
  -/
  simp only [← nonempty_iff_ne_empty] at htn ⊢
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    htn : ∀ (i : ι), (t i).Nonempty
    ⊢ And (Directed (fun x1 x2 => Superset x1 x2) t) (∀ (i : ι), (Inter.inter (t i …
  -/
  refine ⟨htd, fun i => ?_⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    htn : ∀ (i : ι), (t i).Nonempty
    i : ι
    ⊢ (Inter.inter (t i₀) (t i)).Nonempty
  -/
  rcases htd i₀ i with ⟨j, hji₀, hji⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    ι : Type v
    hι : Nonempty ι
    t : ι → Set X
    htd : Directed (fun x1 x2 => Superset x1 x2) t
    htc : ∀ (i : ι), IsCompact (t i)
    htcl : ∀ (i : ι), IsClosed (t i)
    i₀ : ι := hι.some
    htn : ∀ (i : ι), (t i).Nonempty
    i j : ι
    hji₀ : Superset (t i₀) (t j)
    hji : Superset (t i) (t j)
    ⊢ (Inter.inter (t i₀) (t i)).Nonempty
  -/
  exact (htn j).mono (subset_inter hji₀ hji)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-28")]
alias IsCompact.nonempty_iInter_of_directed_nonempty_compact_closed :=
  IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed


/-- Cantor's intersection theorem for `sInter`:
the intersection of a directed family of nonempty compact closed sets is nonempty. -/
theorem IsCompact.nonempty_sInter_of_directed_nonempty_isCompact_isClosed
    {S : Set (Set X)} [hS : Nonempty S] (hSd : DirectedOn (· ⊇ ·) S) (hSn : ∀ U ∈ S, U.Nonempty)
    (hSc : ∀ U ∈ S, IsCompact U) (hScl : ∀ U ∈ S, IsClosed U) : (⋂₀ S).Nonempty := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    hS : Nonempty ↑S
    hSd : DirectedOn (fun x1 x2 => Superset x1 x2) S
    hSn : ∀ (U : Set X), Membership.mem S U → U.Nonempty
    hSc : ∀ (U : Set X), Membership.mem S U → IsCompact U
    hScl : ∀ (U : Set X), Membership.mem S U → IsClosed U
    ⊢ S.sInter.Nonempty
  -/
  rw [sInter_eq_iInter]
  exact IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed _
    (DirectedOn.directed_val hSd) (fun i ↦ hSn i i.2) (fun i ↦ hSc i i.2) (fun i ↦ hScl i i.2)


/-- Cantor's intersection theorem for sequences indexed by `ℕ`:
the intersection of a decreasing sequence of nonempty compact closed sets is nonempty. -/
theorem IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed (t : ℕ → Set X)
    (htd : ∀ i, t (i + 1) ⊆ t i) (htn : ∀ i, (t i).Nonempty) (ht0 : IsCompact (t 0))
    (htcl : ∀ i, IsClosed (t i)) : (⋂ i, t i).Nonempty :=
  have tmono : Antitone t := antitone_nat_of_succ_le htd
  have htd : Directed (· ⊇ ·) t := tmono.directed_ge
  have : ∀ i, t i ⊆ t 0 := fun i => tmono <| Nat.zero_le i
  have htc : ∀ i, IsCompact (t i) := fun i => ht0.of_isClosed_subset (htcl i) (this i)
  IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed t htd htn htc htcl


@[deprecated (since := "2024-02-28")]
alias IsCompact.nonempty_iInter_of_sequence_nonempty_compact_closed :=
  IsCompact.nonempty_iInter_of_sequence_nonempty_isCompact_isClosed


/-- For every open cover of a compact set, there exists a finite subcover. -/
theorem IsCompact.elim_finite_subcover_image {b : Set ι} {c : ι → Set X} (hs : IsCompact s)
    (hc₁ : ∀ i ∈ b, IsOpen (c i)) (hc₂ : s ⊆ ⋃ i ∈ b, c i) :
    ∃ b', b' ⊆ b ∧ Set.Finite b' ∧ s ⊆ ⋃ i ∈ b', c i := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsCompact s
    hc₁ : ∀ (i : ι), Membership.mem b i → IsOpen (c i)
    hc₂ : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c i)
    ⊢ Exists fun b' => And (HasSubset.Subset b' b) (And b'.Finite (HasSubset.Subse …
  -/
  simp only [Subtype.forall', biUnion_eq_iUnion] at hc₁ hc₂
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsCompact s
    hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
    hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
    ⊢ Exists fun b' => And (HasSubset.Subset b' b) (And b'.Finite (HasSubset.Subse …
  -/
  rcases hs.elim_finite_subcover (fun i => c i : b → Set X) hc₁ hc₂ with ⟨d, hd⟩
  /-
    case intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    b : Set ι
    c : ι → Set X
    hs : IsCompact s
    hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
    hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
    d : Finset ↑b
    hd : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c ↑i)
    ⊢ Exists fun b' => And (HasSubset.Subset b' b) (And b'.Finite (HasSubset.Subse …
  -/
  refine ⟨Subtype.val '' d.toSet, ?_, d.finite_toSet.image _, ?_⟩
    /-
      case intro.refine_1
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      b : Set ι
      c : ι → Set X
      hs : IsCompact s
      hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
      hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
      d : Finset ↑b
      hd : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c ↑i)
      ⊢ HasSubset.Subset (Set.image Subtype.val ↑d) b
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      b : Set ι
      c : ι → Set X
      hs : IsCompact s
      hc₁ : ∀ (x : Subtype fun a => Membership.mem b a), IsOpen (c ↑x)
      hc₂ : HasSubset.Subset s (Set.iUnion fun x => c ↑x)
      d : Finset ↑b
      hd : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c ↑i)
      ⊢ HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => c i)
    -/
  · rwa [biUnion_image]
    /-
      🎉 no goals
    -/


/-- A set `s` is compact if for every open cover of `s`, there exists a finite subcover. -/
theorem isCompact_of_finite_subcover
    (h : ∀ {ι : Type u} (U : ι → Set X), (∀ i, IsOpen (U i)) → (s ⊆ ⋃ i, U i) →
      ∃ t : Finset ι, s ⊆ ⋃ i ∈ t, U i) :
    IsCompact s := fun f hf hfs => by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    h : ∀ {ι : Type u} (U : ι → Set X), (∀ (i : ι), IsOpen (U i)) → HasSubset.Subs …
    f : Filter X
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    ⊢ Exists fun x => And (Membership.mem s x) (ClusterPt x f)
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
    hfs : LE.le f (Filter.principal s)
    h : ∀ (x : ↑s), Exists fun i => And (And (Membership.mem i ↑x) (IsOpen i)) (Me …
    ⊢ Exists fun {ι} => Exists fun U => And (∀ (i : ι), IsOpen (U i)) (And (HasSub …
  -/
  choose U hU hUf using h
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    ⊢ Exists fun {ι} => Exists fun U => And (∀ (i : ι), IsOpen (U i)) (And (HasSub …
  -/
  refine ⟨s, U, fun x => (hU x).2, fun x hx => mem_iUnion.2 ⟨⟨x, hx⟩, (hU _).1⟩, fun t ht => ?_⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Finset ↑s
    ht : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ False
  -/
  refine compl_not_mem (le_principal_iff.1 hfs) ?_
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Finset ↑s
    ht : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ Membership.mem f (HasCompl.compl s)
  -/
  refine mem_of_superset ((biInter_finset_mem t).2 fun x _ => hUf x) ?_
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Finset ↑s
    ht : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ HasSubset.Subset (Set.iInter fun i => Set.iInter fun h => HasCompl.compl (U  …
  -/
  rw [subset_compl_comm, compl_iInter₂]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    f : Filter X
    hf : f.NeBot
    hfs : LE.le f (Filter.principal s)
    U : ↑s → Set X
    hU : ∀ (x : ↑s), And (Membership.mem (U x) ↑x) (IsOpen (U x))
    hUf : ∀ (x : ↑s), Membership.mem f (HasCompl.compl (U x))
    t : Finset ↑s
    ht : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    ⊢ HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun j => HasCompl.compl ( …
  -/
  simpa only [compl_compl]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: reformulate using `Disjoint`

/-- A set `s` is compact if for every family of closed sets whose intersection avoids `s`,
there exists a finite subfamily whose intersection avoids `s`. -/
theorem isCompact_of_finite_subfamily_closed
    (h : ∀ {ι : Type u} (t : ι → Set X), (∀ i, IsClosed (t i)) → (s ∩ ⋂ i, t i) = ∅ →
      ∃ u : Finset ι, (s ∩ ⋂ i ∈ u, t i) = ∅) :
    IsCompact s :=
  isCompact_of_finite_subcover fun U hUo hsU => by
    /-
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Inter.in …
      ι✝ : Type u
      U : ι✝ → Set X
      hUo : ∀ (i : ι✝), IsOpen (U i)
      hsU : HasSubset.Subset s (Set.iUnion fun i => U i)
      ⊢ Exists fun t => HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h =>  …
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
      ⊢ Exists fun t => HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h =>  …
    -/
    rcases h (fun i => (U i)ᶜ) (fun i => (hUo _).isClosed_compl) hsU with ⟨t, ht⟩
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
      t : Finset ι✝
      ht : Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => HasCompl.compl …
      ⊢ Exists fun t => HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h =>  …
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
      t : Finset ι✝
      ht : Eq (Inter.inter s (Set.iInter fun i => Set.iInter fun h => HasCompl.compl …
      ⊢ HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => U i)
    -/
    rwa [← disjoint_compl_right_iff_subset, compl_iUnion₂, disjoint_iff]
    /-
      🎉 no goals
    -/


/-- A set `s` is compact if and only if
for every open cover of `s`, there exists a finite subcover. -/
theorem isCompact_iff_finite_subcover :
    IsCompact s ↔ ∀ {ι : Type u} (U : ι → Set X),
      (∀ i, IsOpen (U i)) → (s ⊆ ⋃ i, U i) → ∃ t : Finset ι, s ⊆ ⋃ i ∈ t, U i :=
  ⟨fun hs => hs.elim_finite_subcover, isCompact_of_finite_subcover⟩


/-- A set `s` is compact if and only if
for every family of closed sets whose intersection avoids `s`,
there exists a finite subfamily whose intersection avoids `s`. -/
theorem isCompact_iff_finite_subfamily_closed :
    IsCompact s ↔ ∀ {ι : Type u} (t : ι → Set X),
      (∀ i, IsClosed (t i)) → (s ∩ ⋂ i, t i) = ∅ → ∃ u : Finset ι, (s ∩ ⋂ i ∈ u, t i) = ∅ :=
  ⟨fun hs => hs.elim_finite_subfamily_closed, isCompact_of_finite_subfamily_closed⟩


/-- If `s : Set (X × Y)` belongs to `𝓝 x ×ˢ l` for all `x` from a compact set `K`,
then it belongs to `(𝓝ˢ K) ×ˢ l`,
i.e., there exist an open `U ⊇ K` and `t ∈ l` such that `U ×ˢ t ⊆ s`. -/
theorem IsCompact.mem_nhdsSet_prod_of_forall {K : Set X} {Y} {l : Filter Y} {s : Set (X × Y)}
    (hK : IsCompact K) (hs : ∀ x ∈ K, s ∈ 𝓝 x ×ˢ l) : s ∈ (𝓝ˢ K) ×ˢ l := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    K : Set X
    Y : Type u_2
    l : Filter Y
    s : Set (Prod X Y)
    hK : IsCompact K
    hs : ∀ (x : X), Membership.mem K x → Membership.mem (SProd.sprod (nhds x) l) s
    ⊢ Membership.mem (SProd.sprod (nhdsSet K) l) s
  -/
  refine hK.induction_on (by simp) (fun t t' ht hs ↦ ?_) (fun t t' ht ht' ↦ ?_) fun x hx ↦ ?_
    /-
      case refine_1
      X : Type u
      inst✝ : TopologicalSpace X
      K : Set X
      Y : Type u_2
      l : Filter Y
      s : Set (Prod X Y)
      hK : IsCompact K
      hs✝ : ∀ (x : X), Membership.mem K x → Membership.mem (SProd.sprod (nhds x) l) s
      t t' : Set X
      ht : HasSubset.Subset t t'
      hs : Membership.mem (SProd.sprod (nhdsSet t') l) s
      ⊢ Membership.mem (SProd.sprod (nhdsSet t) l) s
    -/
  · exact prod_mono (nhdsSet_mono ht) le_rfl hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      inst✝ : TopologicalSpace X
      K : Set X
      Y : Type u_2
      l : Filter Y
      s : Set (Prod X Y)
      hK : IsCompact K
      hs : ∀ (x : X), Membership.mem K x → Membership.mem (SProd.sprod (nhds x) l) s
      t t' : Set X
      ht : Membership.mem (SProd.sprod (nhdsSet t) l) s
      ht' : Membership.mem (SProd.sprod (nhdsSet t') l) s
      ⊢ Membership.mem (SProd.sprod (nhdsSet (Union.union t t')) l) s
    -/
  · simp [sup_prod, *]
    /-
      🎉 no goals
    -/
  · rcases ((nhds_basis_opens _).prod l.basis_sets).mem_iff.1 (hs x hx)
      with ⟨⟨u, v⟩, ⟨⟨hx, huo⟩, hv⟩, hs⟩
    /-
      case refine_3.intro.mk.intro.intro.intro
      X : Type u
      inst✝ : TopologicalSpace X
      K : Set X
      Y : Type u_2
      l : Filter Y
      s : Set (Prod X Y)
      hK : IsCompact K
      hs✝ : ∀ (x : X), Membership.mem K x → Membership.mem (SProd.sprod (nhds x) l) s
      x : X
      hx✝ : Membership.mem K x
      u : Set X
      v : Set Y
      hs : HasSubset.Subset (SProd.sprod { fst := u, snd := v }.1 (id { fst := u, sn …
      hv : Membership.mem l { fst := u, snd := v }.2
      hx : Membership.mem { fst := u, snd := v }.1 x
      huo : IsOpen { fst := u, snd := v }.1
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x K) t) (Membership.mem (SPr …
    -/
    refine ⟨u, nhdsWithin_le_nhds (huo.mem_nhds hx), mem_of_superset ?_ hs⟩
    /-
      case refine_3.intro.mk.intro.intro.intro
      X : Type u
      inst✝ : TopologicalSpace X
      K : Set X
      Y : Type u_2
      l : Filter Y
      s : Set (Prod X Y)
      hK : IsCompact K
      hs✝ : ∀ (x : X), Membership.mem K x → Membership.mem (SProd.sprod (nhds x) l) s
      x : X
      hx✝ : Membership.mem K x
      u : Set X
      v : Set Y
      hs : HasSubset.Subset (SProd.sprod { fst := u, snd := v }.1 (id { fst := u, sn …
      hv : Membership.mem l { fst := u, snd := v }.2
      hx : Membership.mem { fst := u, snd := v }.1 x
      huo : IsOpen { fst := u, snd := v }.1
      ⊢ Membership.mem (SProd.sprod (nhdsSet u) l) (SProd.sprod { fst := u, snd := v …
    -/
    exact prod_mem_prod (huo.mem_nhdsSet.2 Subset.rfl) hv
    /-
      🎉 no goals
    -/


theorem IsCompact.nhdsSet_prod_eq_biSup {K : Set X} (hK : IsCompact K) {Y} (l : Filter Y) :
    (𝓝ˢ K) ×ˢ l = ⨆ x ∈ K, 𝓝 x ×ˢ l :=
                                                              /-
                                                                X : Type u
                                                                inst✝ : TopologicalSpace X
                                                                K : Set X
                                                                hK : IsCompact K
                                                                Y : Type u_2
                                                                l : Filter Y
                                                                s : Set (Prod X Y)
                                                                hs : Membership.mem (iSup fun x => iSup fun h => SProd.sprod (nhds x) l) s
                                                                ⊢ ∀ (x : X), Membership.mem K x → Membership.mem (SProd.sprod (nhds x) l) s
                                                              -/
  le_antisymm (fun s hs ↦ hK.mem_nhdsSet_prod_of_forall <| by simpa using hs)
                                                              /-
                                                                🎉 no goals
                                                              -/
    (iSup₂_le fun _ hx ↦ prod_mono (nhds_le_nhdsSet hx) le_rfl)


theorem IsCompact.prod_nhdsSet_eq_biSup {K : Set Y} (hK : IsCompact K) {X} (l : Filter X) :
    l ×ˢ (𝓝ˢ K) = ⨆ y ∈ K, l ×ˢ 𝓝 y := by
  /-
    Y : Type v
    inst✝ : TopologicalSpace Y
    K : Set Y
    hK : IsCompact K
    X : Type u_2
    l : Filter X
    ⊢ Eq (SProd.sprod l (nhdsSet K)) (iSup fun y => iSup fun h => SProd.sprod l (n …
  -/
  simp only [prod_comm (f := l), hK.nhdsSet_prod_eq_biSup, map_iSup]
  /-
    🎉 no goals
  -/


/-- If `s : Set (X × Y)` belongs to `l ×ˢ 𝓝 y` for all `y` from a compact set `K`,
then it belongs to `l ×ˢ (𝓝ˢ K)`,
i.e., there exist `t ∈ l` and an open `U ⊇ K` such that `t ×ˢ U ⊆ s`. -/
theorem IsCompact.mem_prod_nhdsSet_of_forall {K : Set Y} {X} {l : Filter X} {s : Set (X × Y)}
    (hK : IsCompact K) (hs : ∀ y ∈ K, s ∈ l ×ˢ 𝓝 y) : s ∈ l ×ˢ 𝓝ˢ K :=
                                         /-
                                           Y : Type v
                                           inst✝ : TopologicalSpace Y
                                           K : Set Y
                                           X : Type u_2
                                           l : Filter X
                                           s : Set (Prod X Y)
                                           hK : IsCompact K
                                           hs : ∀ (y : Y), Membership.mem K y → Membership.mem (SProd.sprod l (nhds y)) s
                                           ⊢ Membership.mem (iSup fun y => iSup fun h => SProd.sprod l (nhds y)) s
                                         -/
  (hK.prod_nhdsSet_eq_biSup l).symm ▸ by simpa using hs
                                         /-
                                           🎉 no goals
                                         -/

-- TODO: Is there a way to prove directly the `inf` version and then deduce the `Prod` one ?
-- That would seem a bit more natural.

theorem IsCompact.nhdsSet_inf_eq_biSup {K : Set X} (hK : IsCompact K) (l : Filter X) :
    (𝓝ˢ K) ⊓ l = ⨆ x ∈ K, 𝓝 x ⊓ l := by
  have : ∀ f : Filter X, f ⊓ l = comap (fun x ↦ (x, x)) (f ×ˢ l) := fun f ↦ by
    simpa only [comap_prod] using congrArg₂ (· ⊓ ·) comap_id.symm comap_id.symm
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    K : Set X
    hK : IsCompact K
    l : Filter X
    this : ∀ (f : Filter X), Eq (Min.min f l) (Filter.comap (fun x => { fst := x,  …
    ⊢ Eq (Min.min (nhdsSet K) l) (iSup fun x => iSup fun h => Min.min (nhds x) l)
  -/
  simp_rw [this, ← comap_iSup, hK.nhdsSet_prod_eq_biSup]
  /-
    🎉 no goals
  -/


theorem IsCompact.inf_nhdsSet_eq_biSup {K : Set X} (hK : IsCompact K) (l : Filter X) :
    l ⊓ (𝓝ˢ K) = ⨆ x ∈ K, l ⊓ 𝓝 x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    K : Set X
    hK : IsCompact K
    l : Filter X
    ⊢ Eq (Min.min l (nhdsSet K)) (iSup fun x => iSup fun h => Min.min l (nhds x))
  -/
  simp only [inf_comm l, hK.nhdsSet_inf_eq_biSup]
  /-
    🎉 no goals
  -/


/-- If `s : Set X` belongs to `𝓝 x ⊓ l` for all `x` from a compact set `K`,
then it belongs to `(𝓝ˢ K) ⊓ l`,
i.e., there exist an open `U ⊇ K` and `T ∈ l` such that `U ∩ T ⊆ s`. -/
theorem IsCompact.mem_nhdsSet_inf_of_forall {K : Set X} {l : Filter X} {s : Set X}
    (hK : IsCompact K) (hs : ∀ x ∈ K, s ∈ 𝓝 x ⊓ l) : s ∈ (𝓝ˢ K) ⊓ l :=
                                        /-
                                          X : Type u
                                          inst✝ : TopologicalSpace X
                                          K : Set X
                                          l : Filter X
                                          s : Set X
                                          hK : IsCompact K
                                          hs : ∀ (x : X), Membership.mem K x → Membership.mem (Min.min (nhds x) l) s
                                          ⊢ Membership.mem (iSup fun x => iSup fun h => Min.min (nhds x) l) s
                                        -/
  (hK.nhdsSet_inf_eq_biSup l).symm ▸ by simpa using hs
                                        /-
                                          🎉 no goals
                                        -/


/-- If `s : Set S` belongs to `l ⊓ 𝓝 x` for all `x` from a compact set `K`,
then it belongs to `l ⊓ (𝓝ˢ K)`,
i.e., there exist `T ∈ l` and an open `U ⊇ K` such that `T ∩ U ⊆ s`. -/
theorem IsCompact.mem_inf_nhdsSet_of_forall {K : Set X} {l : Filter X} {s : Set X}
    (hK : IsCompact K) (hs : ∀ y ∈ K, s ∈ l ⊓ 𝓝 y) : s ∈ l ⊓ 𝓝ˢ K :=
                                        /-
                                          X : Type u
                                          inst✝ : TopologicalSpace X
                                          K : Set X
                                          l : Filter X
                                          s : Set X
                                          hK : IsCompact K
                                          hs : ∀ (y : X), Membership.mem K y → Membership.mem (Min.min l (nhds y)) s
                                          ⊢ Membership.mem (iSup fun x => iSup fun h => Min.min l (nhds x)) s
                                        -/
  (hK.inf_nhdsSet_eq_biSup l).symm ▸ by simpa using hs
                                        /-
                                          🎉 no goals
                                        -/


/-- To show that `∀ y ∈ K, P x y` holds for `x` close enough to `x₀` when `K` is compact,
it is sufficient to show that for all `y₀ ∈ K` there `P x y` holds for `(x, y)` close enough
to `(x₀, y₀)`.

Provided for backwards compatibility,
see `IsCompact.mem_prod_nhdsSet_of_forall` for a stronger statement.
-/
theorem IsCompact.eventually_forall_of_forall_eventually {x₀ : X} {K : Set Y} (hK : IsCompact K)
    {P : X → Y → Prop} (hP : ∀ y ∈ K, ∀ᶠ z : X × Y in 𝓝 (x₀, y), P z.1 z.2) :
    ∀ᶠ x in 𝓝 x₀, ∀ y ∈ K, P x y := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x₀ : X
    K : Set Y
    hK : IsCompact K
    P : X → Y → Prop
    hP : ∀ (y : Y), Membership.mem K y → Filter.Eventually (fun z => P z.1 z.2) (n …
    ⊢ Filter.Eventually (fun x => ∀ (y : Y), Membership.mem K y → P x y) (nhds x₀)
  -/
  simp only [nhds_prod_eq, ← eventually_iSup, ← hK.prod_nhdsSet_eq_biSup] at hP
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x₀ : X
    K : Set Y
    hK : IsCompact K
    P : X → Y → Prop
    hP : Filter.Eventually (fun z => P z.1 z.2) (SProd.sprod (nhds x₀) (nhdsSet K))
    ⊢ Filter.Eventually (fun x => ∀ (y : Y), Membership.mem K y → P x y) (nhds x₀)
  -/
  exact hP.curry.mono fun _ h ↦ h.self_of_nhdsSet
  /-
    🎉 no goals
  -/


@[simp]
theorem isCompact_empty : IsCompact (∅ : Set X) := fun _f hnf hsf =>
  Not.elim hnf.ne <| empty_mem_iff_bot.1 <| le_principal_iff.1 hsf


@[simp]
theorem isCompact_singleton {x : X} : IsCompact ({x} : Set X) := fun _ hf hfa =>
  ⟨x, rfl, ClusterPt.of_le_nhds'
                     /-
                       X : Type u
                       inst✝ : TopologicalSpace X
                       x : X
                       x✝ : Filter X
                       hf : x✝.NeBot
                       hfa : LE.le x✝ (Filter.principal (Singleton.singleton x))
                       ⊢ LE.le (Filter.principal (Singleton.singleton x)) (nhds x)
                     -/
    (hfa.trans <| by simpa only [principal_singleton] using pure_le_nhds x) hf⟩
                     /-
                       🎉 no goals
                     -/


theorem Set.Subsingleton.isCompact (hs : s.Subsingleton) : IsCompact s :=
  Subsingleton.induction_on hs isCompact_empty fun _ => isCompact_singleton

-- Porting note: golfed a proof instead of fixing it

theorem Set.Finite.isCompact_biUnion {s : Set ι} {f : ι → Set X} (hs : s.Finite)
    (hf : ∀ i ∈ s, IsCompact (f i)) : IsCompact (⋃ i ∈ s, f i) :=
  isCompact_iff_ultrafilter_le_nhds'.2 fun l hl => by
    /-
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Finite
      hf : ∀ (i : ι), Membership.mem s i → IsCompact (f i)
      l : Ultrafilter X
      hl : Membership.mem l (Set.iUnion fun i => Set.iUnion fun h => f i)
      ⊢ Exists fun x => And (Membership.mem (Set.iUnion fun i => Set.iUnion fun h => …
    -/
    rw [Ultrafilter.finite_biUnion_mem_iff hs] at hl
    /-
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Finite
      hf : ∀ (i : ι), Membership.mem s i → IsCompact (f i)
      l : Ultrafilter X
      hl : Exists fun i => And (Membership.mem s i) (Membership.mem l (f i))
      ⊢ Exists fun x => And (Membership.mem (Set.iUnion fun i => Set.iUnion fun h => …
    -/
    rcases hl with ⟨i, his, hi⟩
    /-
      case intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Finite
      hf : ∀ (i : ι), Membership.mem s i → IsCompact (f i)
      l : Ultrafilter X
      i : ι
      his : Membership.mem s i
      hi : Membership.mem l (f i)
      ⊢ Exists fun x => And (Membership.mem (Set.iUnion fun i => Set.iUnion fun h => …
    -/
    rcases (hf i his).ultrafilter_le_nhds _ (le_principal_iff.2 hi) with ⟨x, hxi, hlx⟩
    /-
      case intro.intro.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      s : Set ι
      f : ι → Set X
      hs : s.Finite
      hf : ∀ (i : ι), Membership.mem s i → IsCompact (f i)
      l : Ultrafilter X
      i : ι
      his : Membership.mem s i
      hi : Membership.mem l (f i)
      x : X
      hxi : Membership.mem (f i) x
      hlx : LE.le (↑l) (nhds x)
      ⊢ Exists fun x => And (Membership.mem (Set.iUnion fun i => Set.iUnion fun h => …
    -/
    exact ⟨x, mem_iUnion₂.2 ⟨i, his, hxi⟩, hlx⟩
    /-
      🎉 no goals
    -/


theorem Finset.isCompact_biUnion (s : Finset ι) {f : ι → Set X} (hf : ∀ i ∈ s, IsCompact (f i)) :
    IsCompact (⋃ i ∈ s, f i) :=
  s.finite_toSet.isCompact_biUnion hf


theorem isCompact_accumulate {K : ℕ → Set X} (hK : ∀ n, IsCompact (K n)) (n : ℕ) :
    IsCompact (Accumulate K n) :=
  (finite_le_nat n).isCompact_biUnion fun k _ => hK k


theorem Set.Finite.isCompact_sUnion {S : Set (Set X)} (hf : S.Finite) (hc : ∀ s ∈ S, IsCompact s) :
    IsCompact (⋃₀ S) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    hf : S.Finite
    hc : ∀ (s : Set X), Membership.mem S s → IsCompact s
    ⊢ IsCompact S.sUnion
  -/
  rw [sUnion_eq_biUnion]; exact hf.isCompact_biUnion hc
                          /-
                            🎉 no goals
                          -/

-- Porting note: generalized to `ι : Sort*`

theorem isCompact_iUnion {ι : Sort*} {f : ι → Set X} [Finite ι] (h : ∀ i, IsCompact (f i)) :
    IsCompact (⋃ i, f i) :=
  (finite_range f).isCompact_sUnion <| forall_mem_range.2 h


theorem Set.Finite.isCompact (hs : s.Finite) : IsCompact s :=
  biUnion_of_singleton s ▸ hs.isCompact_biUnion fun _ _ => isCompact_singleton


theorem IsCompact.finite_of_discrete [DiscreteTopology X] (hs : IsCompact s) : s.Finite := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsCompact s
    ⊢ s.Finite
  -/
  have : ∀ x : X, ({x} : Set X) ∈ 𝓝 x := by simp [nhds_discrete]
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsCompact s
    this : ∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)
    ⊢ s.Finite
  -/
  rcases hs.elim_nhds_subcover (fun x => {x}) fun x _ => this x with ⟨t, _, hst⟩
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsCompact s
    this : ∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)
    t : Finset X
    left✝ : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hst : HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => Singleton.si …
    ⊢ s.Finite
  -/
  simp only [← t.set_biUnion_coe, biUnion_of_singleton] at hst
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : DiscreteTopology X
    hs : IsCompact s
    this : ∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)
    t : Finset X
    left✝ : ∀ (x : X), Membership.mem t x → Membership.mem s x
    hst : HasSubset.Subset s ↑t
    ⊢ s.Finite
  -/
  exact t.finite_toSet.subset hst
  /-
    🎉 no goals
  -/


theorem isCompact_iff_finite [DiscreteTopology X] : IsCompact s ↔ s.Finite :=
  ⟨fun h => h.finite_of_discrete, fun h => h.isCompact⟩


theorem IsCompact.union (hs : IsCompact s) (ht : IsCompact t) : IsCompact (s ∪ t) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsCompact t
    ⊢ IsCompact (Union.union s t)
  -/
  rw [union_eq_iUnion]; exact isCompact_iUnion fun b => by cases b <;> assumption
                        /-
                          🎉 no goals
                        -/


protected theorem IsCompact.insert (hs : IsCompact s) (a) : IsCompact (insert a s) :=
  isCompact_singleton.union hs

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: reformulate using `𝓝ˢ`

/-- If `V : ι → Set X` is a decreasing family of closed compact sets then any neighborhood of
`⋂ i, V i` contains some `V i`. We assume each `V i` is compact *and* closed because `X` is
not assumed to be Hausdorff. See `exists_subset_nhd_of_compact` for version assuming this. -/
theorem exists_subset_nhds_of_isCompact' [Nonempty ι] {V : ι → Set X}
    (hV : Directed (· ⊇ ·) V) (hV_cpct : ∀ i, IsCompact (V i)) (hV_closed : ∀ i, IsClosed (V i))
    {U : Set X} (hU : ∀ x ∈ ⋂ i, V i, U ∈ 𝓝 x) : ∃ i, V i ⊆ U := by
  /-
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Nonempty ι
    V : ι → Set X
    hV : Directed (fun x1 x2 => Superset x1 x2) V
    hV_cpct : ∀ (i : ι), IsCompact (V i)
    hV_closed : ∀ (i : ι), IsClosed (V i)
    U : Set X
    hU : ∀ (x : X), Membership.mem (Set.iInter fun i => V i) x → Membership.mem (n …
    ⊢ Exists fun i => HasSubset.Subset (V i) U
  -/
  obtain ⟨W, hsubW, W_op, hWU⟩ := exists_open_set_nhds hU
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Nonempty ι
    V : ι → Set X
    hV : Directed (fun x1 x2 => Superset x1 x2) V
    hV_cpct : ∀ (i : ι), IsCompact (V i)
    hV_closed : ∀ (i : ι), IsClosed (V i)
    U : Set X
    hU : ∀ (x : X), Membership.mem (Set.iInter fun i => V i) x → Membership.mem (n …
    W : Set X
    hsubW : HasSubset.Subset (Set.iInter fun i => V i) W
    W_op : IsOpen W
    hWU : HasSubset.Subset W U
    ⊢ Exists fun i => HasSubset.Subset (V i) U
  -/
  suffices ∃ i, V i ⊆ W from this.imp fun i hi => hi.trans hWU
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Nonempty ι
    V : ι → Set X
    hV : Directed (fun x1 x2 => Superset x1 x2) V
    hV_cpct : ∀ (i : ι), IsCompact (V i)
    hV_closed : ∀ (i : ι), IsClosed (V i)
    U : Set X
    hU : ∀ (x : X), Membership.mem (Set.iInter fun i => V i) x → Membership.mem (n …
    W : Set X
    hsubW : HasSubset.Subset (Set.iInter fun i => V i) W
    W_op : IsOpen W
    hWU : HasSubset.Subset W U
    ⊢ Exists fun i => HasSubset.Subset (V i) W
  -/
  by_contra! H
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Nonempty ι
    V : ι → Set X
    hV : Directed (fun x1 x2 => Superset x1 x2) V
    hV_cpct : ∀ (i : ι), IsCompact (V i)
    hV_closed : ∀ (i : ι), IsClosed (V i)
    U : Set X
    hU : ∀ (x : X), Membership.mem (Set.iInter fun i => V i) x → Membership.mem (n …
    W : Set X
    hsubW : HasSubset.Subset (Set.iInter fun i => V i) W
    W_op : IsOpen W
    hWU : HasSubset.Subset W U
    H : ∀ (i : ι), Not (HasSubset.Subset (V i) W)
    ⊢ False
  -/
  replace H : ∀ i, (V i ∩ Wᶜ).Nonempty := fun i => Set.inter_compl_nonempty_iff.mpr (H i)
  have : (⋂ i, V i ∩ Wᶜ).Nonempty := by
    refine
      IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed _ (fun i j => ?_) H
        (fun i => (hV_cpct i).inter_right W_op.isClosed_compl) fun i =>
        (hV_closed i).inter W_op.isClosed_compl
    rcases hV i j with ⟨k, hki, hkj⟩
    refine ⟨k, ⟨fun x => ?_, fun x => ?_⟩⟩ <;> simp only [and_imp, mem_inter_iff, mem_compl_iff] <;>
      tauto
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Nonempty ι
    V : ι → Set X
    hV : Directed (fun x1 x2 => Superset x1 x2) V
    hV_cpct : ∀ (i : ι), IsCompact (V i)
    hV_closed : ∀ (i : ι), IsClosed (V i)
    U : Set X
    hU : ∀ (x : X), Membership.mem (Set.iInter fun i => V i) x → Membership.mem (n …
    W : Set X
    hsubW : HasSubset.Subset (Set.iInter fun i => V i) W
    W_op : IsOpen W
    hWU : HasSubset.Subset W U
    H : ∀ (i : ι), (Inter.inter (V i) (HasCompl.compl W)).Nonempty
    this : (Set.iInter fun i => Inter.inter (V i) (HasCompl.compl W)).Nonempty
    ⊢ False
  -/
  have : ¬⋂ i : ι, V i ⊆ W := by simpa [← iInter_inter, inter_compl_nonempty_iff]
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Nonempty ι
    V : ι → Set X
    hV : Directed (fun x1 x2 => Superset x1 x2) V
    hV_cpct : ∀ (i : ι), IsCompact (V i)
    hV_closed : ∀ (i : ι), IsClosed (V i)
    U : Set X
    hU : ∀ (x : X), Membership.mem (Set.iInter fun i => V i) x → Membership.mem (n …
    W : Set X
    hsubW : HasSubset.Subset (Set.iInter fun i => V i) W
    W_op : IsOpen W
    hWU : HasSubset.Subset W U
    H : ∀ (i : ι), (Inter.inter (V i) (HasCompl.compl W)).Nonempty
    this✝ : (Set.iInter fun i => Inter.inter (V i) (HasCompl.compl W)).Nonempty
    this : Not (HasSubset.Subset (Set.iInter fun i => V i) W)
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


lemma eq_finite_iUnion_of_isTopologicalBasis_of_isCompact_open (b : ι → Set X)
    (hb : IsTopologicalBasis (Set.range b)) (U : Set X) (hUc : IsCompact U) (hUo : IsOpen U) :
    ∃ s : Set ι, s.Finite ∧ U = ⋃ i ∈ s, b i := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    b : ι → Set X
    hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    ⊢ Exists fun s => And s.Finite (Eq U (Set.iUnion fun i => Set.iUnion fun h =>  …
  -/
  obtain ⟨Y, f, e, hf⟩ := hb.open_eq_iUnion hUo
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    b : ι → Set X
    hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    Y : Type u
    f : Y → Set X
    e : Eq U (Set.iUnion fun i => f i)
    hf : ∀ (i : Y), Membership.mem (Set.range b) (f i)
    ⊢ Exists fun s => And s.Finite (Eq U (Set.iUnion fun i => Set.iUnion fun h =>  …
  -/
  choose f' hf' using hf
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    b : ι → Set X
    hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    Y : Type u
    f : Y → Set X
    e : Eq U (Set.iUnion fun i => f i)
    f' : Y → ι
    hf' : ∀ (i : Y), Eq (b (f' i)) (f i)
    ⊢ Exists fun s => And s.Finite (Eq U (Set.iUnion fun i => Set.iUnion fun h =>  …
  -/
  have : b ∘ f' = f := funext hf'
  /-
    case intro.intro.intro
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    b : ι → Set X
    hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    Y : Type u
    f : Y → Set X
    e : Eq U (Set.iUnion fun i => f i)
    f' : Y → ι
    hf' : ∀ (i : Y), Eq (b (f' i)) (f i)
    this : Eq (Function.comp b f') f
    ⊢ Exists fun s => And s.Finite (Eq U (Set.iUnion fun i => Set.iUnion fun h =>  …
  -/
  subst this
  obtain ⟨t, ht⟩ :=
    hUc.elim_finite_subcover (b ∘ f') (fun i => hb.isOpen (Set.mem_range_self _)) (by rw [e])
  classical
  refine ⟨t.image f', Set.toFinite _, le_antisymm ?_ ?_⟩
  · refine Set.Subset.trans ht ?_
    simp only [Set.iUnion_subset_iff]
    intro i hi
    erw [← Set.iUnion_subtype (fun x : ι => x ∈ t.image f') fun i => b i.1]
    exact Set.subset_iUnion (fun i : t.image f' => b i) ⟨_, Finset.mem_image_of_mem _ hi⟩
  · apply Set.iUnion₂_subset
    rintro i hi
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp hi
    rw [e]
    exact Set.subset_iUnion (b ∘ f') j


lemma eq_sUnion_finset_of_isTopologicalBasis_of_isCompact_open (b : Set (Set X))
    (hb : IsTopologicalBasis b) (U : Set X) (hUc : IsCompact U) (hUo : IsOpen U) :
    ∃ s : Finset b, U = s.toSet.sUnion := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis b
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    ⊢ Exists fun s => Eq U (Set.image Subtype.val ↑s).sUnion
  -/
  have hb' : b = range (fun i ↦ i : b → Set X) := by simp
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis b
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    hb' : Eq b (Set.range fun i => ↑i)
    ⊢ Exists fun s => Eq U (Set.image Subtype.val ↑s).sUnion
  -/
  rw [hb'] at hb
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis (Set.range fun i => ↑i)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    hb' : Eq b (Set.range fun i => ↑i)
    ⊢ Exists fun s => Eq U (Set.image Subtype.val ↑s).sUnion
  -/
  choose s hs hU using eq_finite_iUnion_of_isTopologicalBasis_of_isCompact_open _ hb U hUc hUo
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis (Set.range fun i => ↑i)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    hb' : Eq b (Set.range fun i => ↑i)
    s : Set ↑b
    hs : s.Finite
    hU : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑i)
    ⊢ Exists fun s => Eq U (Set.image Subtype.val ↑s).sUnion
  -/
  have : Finite s := hs
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis (Set.range fun i => ↑i)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    hb' : Eq b (Set.range fun i => ↑i)
    s : Set ↑b
    hs : s.Finite
    hU : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑i)
    this : Finite ↑s
    ⊢ Exists fun s => Eq U (Set.image Subtype.val ↑s).sUnion
  -/
  let _ : Fintype s := Fintype.ofFinite _
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis (Set.range fun i => ↑i)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    hb' : Eq b (Set.range fun i => ↑i)
    s : Set ↑b
    hs : s.Finite
    hU : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑i)
    this : Finite ↑s
    x✝ : Fintype ↑s := Fintype.ofFinite ↑s
    ⊢ Exists fun s => Eq U (Set.image Subtype.val ↑s).sUnion
  -/
  use s.toFinset
  /-
    case h
    X : Type u
    inst✝ : TopologicalSpace X
    b : Set (Set X)
    hb : TopologicalSpace.IsTopologicalBasis (Set.range fun i => ↑i)
    U : Set X
    hUc : IsCompact U
    hUo : IsOpen U
    hb' : Eq b (Set.range fun i => ↑i)
    s : Set ↑b
    hs : s.Finite
    hU : Eq U (Set.iUnion fun i => Set.iUnion fun h => ↑i)
    this : Finite ↑s
    x✝ : Fintype ↑s := Fintype.ofFinite ↑s
    ⊢ Eq U (Set.image Subtype.val ↑s.toFinset).sUnion
  -/
  simp [hU]
  /-
    🎉 no goals
  -/


/-- If `X` has a basis consisting of compact opens, then an open set in `X` is compact open iff
  it is a finite union of some elements in the basis -/
theorem isCompact_open_iff_eq_finite_iUnion_of_isTopologicalBasis (b : ι → Set X)
    (hb : IsTopologicalBasis (Set.range b)) (hb' : ∀ i, IsCompact (b i)) (U : Set X) :
    IsCompact U ∧ IsOpen U ↔ ∃ s : Set ι, s.Finite ∧ U = ⋃ i ∈ s, b i := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    b : ι → Set X
    hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
    hb' : ∀ (i : ι), IsCompact (b i)
    U : Set X
    ⊢ Iff (And (IsCompact U) (IsOpen U)) (Exists fun s => And s.Finite (Eq U (Set. …
  -/
  constructor
    /-
      case mp
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsCompact (b i)
      U : Set X
      ⊢ And (IsCompact U) (IsOpen U) → Exists fun s => And s.Finite (Eq U (Set.iUnio …
    -/
  · exact fun ⟨h₁, h₂⟩ ↦ eq_finite_iUnion_of_isTopologicalBasis_of_isCompact_open _ hb U h₁ h₂
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
      hb' : ∀ (i : ι), IsCompact (b i)
      U : Set X
      ⊢ (Exists fun s => And s.Finite (Eq U (Set.iUnion fun i => Set.iUnion fun h => …
    -/
  · rintro ⟨s, hs, rfl⟩
    /-
      case mpr.intro.intro
      X : Type u
      ι : Type u_1
      inst✝ : TopologicalSpace X
      b : ι → Set X
      hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
      hb' : ∀ (i : ι), IsCompact (b i)
      s : Set ι
      hs : s.Finite
      ⊢ And (IsCompact (Set.iUnion fun i => Set.iUnion fun h => b i)) (IsOpen (Set.i …
    -/
    constructor
      /-
        case mpr.intro.intro.left
        X : Type u
        ι : Type u_1
        inst✝ : TopologicalSpace X
        b : ι → Set X
        hb : TopologicalSpace.IsTopologicalBasis (Set.range b)
        hb' : ∀ (i : ι), IsCompact (b i)
        s : Set ι
        hs : s.Finite
        ⊢ IsCompact (Set.iUnion fun i => Set.iUnion fun h => b i)
      -/
    · exact hs.isCompact_biUnion fun i _ => hb' i
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
        hb' : ∀ (i : ι), IsCompact (b i)
        s : Set ι
        hs : s.Finite
        ⊢ IsOpen (Set.iUnion fun i => Set.iUnion fun h => b i)
      -/
    · exact isOpen_biUnion fun i _ => hb.isOpen (Set.mem_range_self _)
      /-
        🎉 no goals
      -/


theorem hasBasis_cocompact : (cocompact X).HasBasis IsCompact compl :=
  hasBasis_biInf_principal'
    (fun s hs t ht =>
      ⟨s ∪ t, hs.union ht, compl_subset_compl.2 subset_union_left,
        compl_subset_compl.2 subset_union_right⟩)
    ⟨∅, isCompact_empty⟩


theorem mem_cocompact : s ∈ cocompact X ↔ ∃ t, IsCompact t ∧ tᶜ ⊆ s :=
  hasBasis_cocompact.mem_iff


theorem mem_cocompact' : s ∈ cocompact X ↔ ∃ t, IsCompact t ∧ sᶜ ⊆ t :=
  mem_cocompact.trans <| exists_congr fun _ => and_congr_right fun _ => compl_subset_comm


theorem _root_.IsCompact.compl_mem_cocompact (hs : IsCompact s) : sᶜ ∈ Filter.cocompact X :=
  hasBasis_cocompact.mem_of_mem hs


theorem cocompact_le_cofinite : cocompact X ≤ cofinite := fun s hs =>
  compl_compl s ▸ hs.isCompact.compl_mem_cocompact


theorem cocompact_eq_cofinite (X : Type*) [TopologicalSpace X] [DiscreteTopology X] :
    cocompact X = cofinite := by
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    ⊢ Eq (Filter.cocompact X) Filter.cofinite
  -/
  simp only [cocompact, hasBasis_cofinite.eq_biInf, isCompact_iff_finite]
  /-
    🎉 no goals
  -/


/-- A filter is disjoint from the cocompact filter if and only if it contains a compact set. -/
theorem disjoint_cocompact_left (f : Filter X) :
    Disjoint (Filter.cocompact X) f ↔ ∃ K ∈ f, IsCompact K := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ Iff (Disjoint (Filter.cocompact X) f) (Exists fun K => And (Membership.mem f …
  -/
  simp_rw [hasBasis_cocompact.disjoint_iff_left, compl_compl]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ Iff (Exists fun i => And (IsCompact i) (Membership.mem f i)) (Exists fun K = …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- A filter is disjoint from the cocompact filter if and only if it contains a compact set. -/
theorem disjoint_cocompact_right (f : Filter X) :
    Disjoint f (Filter.cocompact X) ↔ ∃ K ∈ f, IsCompact K := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ Iff (Disjoint f (Filter.cocompact X)) (Exists fun K => And (Membership.mem f …
  -/
  simp_rw [hasBasis_cocompact.disjoint_iff_right, compl_compl]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    f : Filter X
    ⊢ Iff (Exists fun i => And (IsCompact i) (Membership.mem f i)) (Exists fun K = …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[deprecated "see `cocompact_eq_atTop` with `import Mathlib.Topology.Instances.Nat`"
  (since := "2024-02-07")]
theorem _root_.Nat.cocompact_eq : cocompact ℕ = atTop :=
  (cocompact_eq_cofinite ℕ).trans Nat.cofinite_eq_atTop


theorem Tendsto.isCompact_insert_range_of_cocompact {f : X → Y} {y}
    (hf : Tendsto f (cocompact X) (𝓝 y)) (hfc : Continuous f) : IsCompact (insert y (range f)) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
    hfc : Continuous f
    ⊢ IsCompact (Insert.insert y (Set.range f))
  -/
  intro l hne hle
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  by_cases hy : ClusterPt y l
    /-
      case pos
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      y : Y
      hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
      hfc : Continuous f
      l : Filter Y
      hne : l.NeBot
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
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    hy : Not (ClusterPt y l)
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  simp only [clusterPt_iff, not_forall, ← not_disjoint_iff_nonempty_inter, not_not] at hy
  /-
    case neg
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    hy : Exists fun x => Exists fun h => Exists fun x_1 => Exists fun h => Disjoin …
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  rcases hy with ⟨s, hsy, t, htl, hd⟩
  /-
    case neg.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    s : Set Y
    hsy : Membership.mem (nhds y) s
    t : Set Y
    htl : Membership.mem l t
    hd : Disjoint s t
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  rcases mem_cocompact.1 (hf hsy) with ⟨K, hKc, hKs⟩
  have : f '' K ∈ l := by
    filter_upwards [htl, le_principal_iff.1 hle] with y hyt hyf
    rcases hyf with (rfl | ⟨x, rfl⟩)
    exacts [(hd.le_bot ⟨mem_of_mem_nhds hsy, hyt⟩).elim,
      mem_image_of_mem _ (not_not.1 fun hxK => hd.le_bot ⟨hKs hxK, hyt⟩)]
  /-
    case neg.intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    hle : LE.le l (Filter.principal (Insert.insert y (Set.range f)))
    s : Set Y
    hsy : Membership.mem (nhds y) s
    t : Set Y
    htl : Membership.mem l t
    hd : Disjoint s t
    K : Set X
    hKc : IsCompact K
    hKs : HasSubset.Subset (HasCompl.compl K) (Set.preimage f s)
    this : Membership.mem l (Set.image f K)
    ⊢ Exists fun x => And (Membership.mem (Insert.insert y (Set.range f)) x) (Clus …
  -/
  rcases hKc.image hfc (le_principal_iff.2 this) with ⟨y, hy, hyl⟩
  /-
    case neg.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    y✝ : Y
    hf : Filter.Tendsto f (Filter.cocompact X) (nhds y✝)
    hfc : Continuous f
    l : Filter Y
    hne : l.NeBot
    hle : LE.le l (Filter.principal (Insert.insert y✝ (Set.range f)))
    s : Set Y
    hsy : Membership.mem (nhds y✝) s
    t : Set Y
    htl : Membership.mem l t
    hd : Disjoint s t
    K : Set X
    hKc : IsCompact K
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


theorem Tendsto.isCompact_insert_range_of_cofinite {f : ι → X} {x} (hf : Tendsto f cofinite (𝓝 x)) :
    IsCompact (insert x (range f)) := by
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    f : ι → X
    x : X
    hf : Filter.Tendsto f Filter.cofinite (nhds x)
    ⊢ IsCompact (Insert.insert x (Set.range f))
  -/
  letI : TopologicalSpace ι := ⊥; haveI h : DiscreteTopology ι := ⟨rfl⟩
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    f : ι → X
    x : X
    hf : Filter.Tendsto f Filter.cofinite (nhds x)
    this : TopologicalSpace ι := Bot.bot
    h : DiscreteTopology ι
    ⊢ IsCompact (Insert.insert x (Set.range f))
  -/
  rw [← cocompact_eq_cofinite ι] at hf
  /-
    X : Type u
    ι : Type u_1
    inst✝ : TopologicalSpace X
    f : ι → X
    x : X
    this : TopologicalSpace ι := Bot.bot
    hf : Filter.Tendsto f (Filter.cocompact ι) (nhds x)
    h : DiscreteTopology ι
    ⊢ IsCompact (Insert.insert x (Set.range f))
  -/
  exact hf.isCompact_insert_range_of_cocompact continuous_of_discreteTopology
  /-
    🎉 no goals
  -/


theorem Tendsto.isCompact_insert_range {f : ℕ → X} {x} (hf : Tendsto f atTop (𝓝 x)) :
    IsCompact (insert x (range f)) :=
  Filter.Tendsto.isCompact_insert_range_of_cofinite <| Nat.cofinite_eq_atTop.symm ▸ hf


theorem hasBasis_coclosedCompact :
    (Filter.coclosedCompact X).HasBasis (fun s => IsClosed s ∧ IsCompact s) compl := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ (Filter.coclosedCompact X).HasBasis (fun s => And (IsClosed s) (IsCompact s) …
  -/
  simp only [Filter.coclosedCompact, iInf_and']
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ (iInf fun s => iInf fun h => Filter.principal (HasCompl.compl s)).HasBasis ( …
  -/
  refine hasBasis_biInf_principal' ?_ ⟨∅, isClosed_empty, isCompact_empty⟩
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ ∀ (i : Set X), And (IsClosed i) (IsCompact i) → ∀ (j : Set X), And (IsClosed …
  -/
  rintro s ⟨hs₁, hs₂⟩ t ⟨ht₁, ht₂⟩
  exact ⟨s ∪ t, ⟨⟨hs₁.union ht₁, hs₂.union ht₂⟩, compl_subset_compl.2 subset_union_left,
    compl_subset_compl.2 subset_union_right⟩⟩


/-- A set belongs to `coclosedCompact` if and only if the closure of its complement is compact. -/
theorem mem_coclosedCompact_iff :
    s ∈ coclosedCompact X ↔ IsCompact (closure sᶜ) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.coclosedCompact X) s) (IsCompact (closure (HasCo …
  -/
  refine hasBasis_coclosedCompact.mem_iff.trans ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ (Exists fun i => And (And (IsClosed i) (IsCompact i)) (HasSubset.Subset (Has …
    -/
  · rintro ⟨t, ⟨htcl, htco⟩, hst⟩
    exact htco.of_isClosed_subset isClosed_closure <|
      closure_minimal (compl_subset_comm.2 hst) htcl
    /-
      case refine_2
      X : Type u
      inst✝ : TopologicalSpace X
      s : Set X
      h : IsCompact (closure (HasCompl.compl s))
      ⊢ Exists fun i => And (And (IsClosed i) (IsCompact i)) (HasSubset.Subset (HasC …
    -/
  · exact ⟨closure sᶜ, ⟨isClosed_closure, h⟩, compl_subset_comm.2 subset_closure⟩
    /-
      🎉 no goals
    -/


@[deprecated mem_coclosedCompact_iff (since := "2024-02-16")]
theorem mem_coclosedCompact : s ∈ coclosedCompact X ↔ ∃ t, IsClosed t ∧ IsCompact t ∧ tᶜ ⊆ s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.coclosedCompact X) s) (Exists fun t => And (IsCl …
  -/
  simp only [hasBasis_coclosedCompact.mem_iff, and_assoc]
  /-
    🎉 no goals
  -/


@[deprecated mem_coclosedCompact_iff (since := "2024-02-16")]
theorem mem_coclosed_compact' : s ∈ coclosedCompact X ↔ ∃ t, IsClosed t ∧ IsCompact t ∧ sᶜ ⊆ t := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.coclosedCompact X) s) (Exists fun t => And (IsCl …
  -/
  simp only [hasBasis_coclosedCompact.mem_iff, compl_subset_comm, and_assoc]
  /-
    🎉 no goals
  -/


/-- Complement of a set belongs to `coclosedCompact` if and only if its closure is compact. -/
theorem compl_mem_coclosedCompact : sᶜ ∈ coclosedCompact X ↔ IsCompact (closure s) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.coclosedCompact X) (HasCompl.compl s)) (IsCompac …
  -/
  rw [mem_coclosedCompact_iff, compl_compl]
  /-
    🎉 no goals
  -/


theorem cocompact_le_coclosedCompact : cocompact X ≤ coclosedCompact X :=
  iInf_mono fun _ => le_iInf fun _ => le_rfl


theorem IsCompact.compl_mem_coclosedCompact_of_isClosed (hs : IsCompact s) (hs' : IsClosed s) :
    sᶜ ∈ Filter.coclosedCompact X :=
  hasBasis_coclosedCompact.mem_of_mem ⟨hs', hs⟩


variable (X) in
/-- Sets that are contained in a compact set form a bornology. Its `cobounded` filter is
`Filter.cocompact`. See also `Bornology.relativelyCompact` the bornology of sets with compact
closure. -/
def inCompact : Bornology X where
  cobounded' := Filter.cocompact X
  le_cofinite' := Filter.cocompact_le_cofinite


theorem inCompact.isBounded_iff : @IsBounded _ (inCompact X) s ↔ ∃ t, IsCompact t ∧ s ⊆ t := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Bornology.IsBounded s) (Exists fun t => And (IsCompact t) (HasSubset.Su …
  -/
  change sᶜ ∈ Filter.cocompact X ↔ _
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Membership.mem (Filter.cocompact X) (HasCompl.compl s)) (Exists fun t = …
  -/
  rw [Filter.mem_cocompact]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Exists fun t => And (IsCompact t) (HasSubset.Subset (HasCompl.compl t)  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `s` and `t` are compact sets, then the set neighborhoods filter of `s ×ˢ t`
is the product of set neighborhoods filters for `s` and `t`.

For general sets, only the `≤` inequality holds, see `nhdsSet_prod_le`. -/
theorem IsCompact.nhdsSet_prod_eq {t : Set Y} (hs : IsCompact s) (ht : IsCompact t) :
    𝓝ˢ (s ×ˢ t) = 𝓝ˢ s ×ˢ 𝓝ˢ t := by
  simp_rw [hs.nhdsSet_prod_eq_biSup, ht.prod_nhdsSet_eq_biSup, nhdsSet, sSup_image, biSup_prod,
    nhds_prod_eq]


theorem nhdsSet_prod_le_of_disjoint_cocompact {f : Filter Y} (hs : IsCompact s)
    (hf : Disjoint f (Filter.cocompact Y)) :
    𝓝ˢ s ×ˢ f ≤ 𝓝ˢ (s ×ˢ Set.univ) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : Filter Y
    hs : IsCompact s
    hf : Disjoint f (Filter.cocompact Y)
    ⊢ LE.le (SProd.sprod (nhdsSet s) f) (nhdsSet (SProd.sprod s Set.univ))
  -/
  obtain ⟨K, hKf, hK⟩ := (disjoint_cocompact_right f).mp hf
  calc
    𝓝ˢ s ×ˢ f
    _ ≤ 𝓝ˢ s ×ˢ 𝓟 K        := Filter.prod_mono_right _ (Filter.le_principal_iff.mpr hKf)
    _ ≤ 𝓝ˢ s ×ˢ 𝓝ˢ K       := Filter.prod_mono_right _ principal_le_nhdsSet
    _ = 𝓝ˢ (s ×ˢ K)         := (hs.nhdsSet_prod_eq hK).symm
    _ ≤ 𝓝ˢ (s ×ˢ Set.univ)  := nhdsSet_mono (prod_mono_right le_top)


theorem prod_nhdsSet_le_of_disjoint_cocompact {t : Set Y} {f : Filter X} (ht : IsCompact t)
    (hf : Disjoint f (Filter.cocompact X)) :
    f ×ˢ 𝓝ˢ t ≤ 𝓝ˢ (Set.univ ×ˢ t) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    t : Set Y
    f : Filter X
    ht : IsCompact t
    hf : Disjoint f (Filter.cocompact X)
    ⊢ LE.le (SProd.sprod f (nhdsSet t)) (nhdsSet (SProd.sprod Set.univ t))
  -/
  obtain ⟨K, hKf, hK⟩ := (disjoint_cocompact_right f).mp hf
  calc
    f ×ˢ 𝓝ˢ t
    _ ≤ (𝓟 K) ×ˢ 𝓝ˢ t      := Filter.prod_mono_left _ (Filter.le_principal_iff.mpr hKf)
    _ ≤ 𝓝ˢ K ×ˢ 𝓝ˢ t       := Filter.prod_mono_left _ principal_le_nhdsSet
    _ = 𝓝ˢ (K ×ˢ t)         := (hK.nhdsSet_prod_eq ht).symm
    _ ≤ 𝓝ˢ (Set.univ ×ˢ t)  := nhdsSet_mono (prod_mono_left le_top)


theorem nhds_prod_le_of_disjoint_cocompact {f : Filter Y} (x : X)
    (hf : Disjoint f (Filter.cocompact Y)) :
    𝓝 x ×ˢ f ≤ 𝓝ˢ ({x} ×ˢ Set.univ) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : Filter Y
    x : X
    hf : Disjoint f (Filter.cocompact Y)
    ⊢ LE.le (SProd.sprod (nhds x) f) (nhdsSet (SProd.sprod (Singleton.singleton x) …
  -/
  simpa using nhdsSet_prod_le_of_disjoint_cocompact isCompact_singleton hf
  /-
    🎉 no goals
  -/


theorem prod_nhds_le_of_disjoint_cocompact {f : Filter X} (y : Y)
    (hf : Disjoint f (Filter.cocompact X)) :
    f ×ˢ 𝓝 y ≤ 𝓝ˢ (Set.univ ×ˢ {y}) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : Filter X
    y : Y
    hf : Disjoint f (Filter.cocompact X)
    ⊢ LE.le (SProd.sprod f (nhds y)) (nhdsSet (SProd.sprod Set.univ (Singleton.sin …
  -/
  simpa using prod_nhdsSet_le_of_disjoint_cocompact isCompact_singleton hf
  /-
    🎉 no goals
  -/


/-- If `s` and `t` are compact sets and `n` is an open neighborhood of `s × t`, then there exist
open neighborhoods `u ⊇ s` and `v ⊇ t` such that `u × v ⊆ n`.

See also `IsCompact.nhdsSet_prod_eq`. -/
theorem generalized_tube_lemma (hs : IsCompact s) {t : Set Y} (ht : IsCompact t)
    {n : Set (X × Y)} (hn : IsOpen n) (hp : s ×ˢ t ⊆ n) :
    ∃ (u : Set X) (v : Set Y), IsOpen u ∧ IsOpen v ∧ s ⊆ u ∧ t ⊆ v ∧ u ×ˢ v ⊆ n := by
  rw [← hn.mem_nhdsSet, hs.nhdsSet_prod_eq ht,
    ((hasBasis_nhdsSet _).prod (hasBasis_nhdsSet _)).mem_iff] at hp
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    hs : IsCompact s
    t : Set Y
    ht : IsCompact t
    n : Set (Prod X Y)
    hn : IsOpen n
    hp : Exists fun i => And (And (And (IsOpen i.1) (HasSubset.Subset s i.1)) (And …
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (HasSubs …
  -/
  rcases hp with ⟨⟨u, v⟩, ⟨⟨huo, hsu⟩, hvo, htv⟩, hn⟩
  /-
    case intro.mk.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    hs : IsCompact s
    t : Set Y
    ht : IsCompact t
    n : Set (Prod X Y)
    hn✝ : IsOpen n
    u : Set X
    v : Set Y
    hn : HasSubset.Subset (SProd.sprod { fst := u, snd := v }.1 { fst := u, snd := …
    huo : IsOpen { fst := u, snd := v }.1
    hsu : HasSubset.Subset s { fst := u, snd := v }.1
    hvo : IsOpen { fst := u, snd := v }.2
    htv : HasSubset.Subset t { fst := u, snd := v }.2
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (HasSubs …
  -/
  exact ⟨u, v, huo, hvo, hsu, htv, hn⟩
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 10) Subsingleton.compactSpace [Subsingleton X] : CompactSpace X :=
  ⟨subsingleton_univ.isCompact⟩


theorem isCompact_univ_iff : IsCompact (univ : Set X) ↔ CompactSpace X :=
  ⟨fun h => ⟨h⟩, fun h => h.1⟩


theorem isCompact_univ [h : CompactSpace X] : IsCompact (univ : Set X) :=
  h.isCompact_univ


theorem exists_clusterPt_of_compactSpace [CompactSpace X] (f : Filter X) [NeBot f] :
    ∃ x, ClusterPt x f := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    f : Filter X
    inst✝ : f.NeBot
    ⊢ Exists fun x => ClusterPt x f
  -/
  simpa using isCompact_univ (show f ≤ 𝓟 univ by simp)
  /-
    🎉 no goals
  -/


nonrec theorem Ultrafilter.le_nhds_lim [CompactSpace X] (F : Ultrafilter X) : ↑F ≤ 𝓝 F.lim := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    F : Ultrafilter X
    ⊢ LE.le (↑F) (nhds F.lim)
  -/
  rcases isCompact_univ.ultrafilter_le_nhds F (by simp) with ⟨x, -, h⟩
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    F : Ultrafilter X
    x : X
    h : LE.le (↑F) (nhds x)
    ⊢ LE.le (↑F) (nhds F.lim)
  -/
  exact le_nhds_lim ⟨x, h⟩
  /-
    🎉 no goals
  -/


theorem CompactSpace.elim_nhds_subcover [CompactSpace X] (U : X → Set X) (hU : ∀ x, U x ∈ 𝓝 x) :
    ∃ t : Finset X, ⋃ x ∈ t, U x = ⊤ := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    U : X → Set X
    hU : ∀ (x : X), Membership.mem (nhds x) (U x)
    ⊢ Exists fun t => Eq (Set.iUnion fun x => Set.iUnion fun h => U x) Top.top
  -/
  obtain ⟨t, -, s⟩ := IsCompact.elim_nhds_subcover isCompact_univ U fun x _ => hU x
  /-
    case intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    U : X → Set X
    hU : ∀ (x : X), Membership.mem (nhds x) (U x)
    t : Finset X
    s : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Exists fun t => Eq (Set.iUnion fun x => Set.iUnion fun h => U x) Top.top
  -/
  exact ⟨t, top_unique s⟩
  /-
    🎉 no goals
  -/


theorem compactSpace_of_finite_subfamily_closed
    (h : ∀ {ι : Type u} (t : ι → Set X), (∀ i, IsClosed (t i)) → ⋂ i, t i = ∅ →
      ∃ u : Finset ι, ⋂ i ∈ u, t i = ∅) :
    CompactSpace X where
                                                                     /-
                                                                       X : Type u
                                                                       inst✝ : TopologicalSpace X
                                                                       h : ∀ {ι : Type u} (t : ι → Set X), (∀ (i : ι), IsClosed (t i)) → Eq (Set.iInt …
                                                                       ι✝ : Type u
                                                                       t : ι✝ → Set X
                                                                       ⊢ (∀ (i : ι✝), IsClosed (t i)) → Eq (Inter.inter Set.univ (Set.iInter fun i => …
                                                                     -/
  isCompact_univ := isCompact_of_finite_subfamily_closed fun t => by simpa using h t
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem IsClosed.isCompact [CompactSpace X] (h : IsClosed s) : IsCompact s :=
  isCompact_univ.of_isClosed_subset h (subset_univ _)


/-- If a filter has a unique cluster point `y` in a compact topological space,
then the filter is less than or equal to `𝓝 y`. -/
lemma le_nhds_of_unique_clusterPt [CompactSpace X] {l : Filter X} {y : X}
    (h : ∀ x, ClusterPt x l → x = y) : l ≤ 𝓝 y :=
  isCompact_univ.le_nhds_of_unique_clusterPt univ_mem fun x _ ↦ h x


/-- If `y` is a unique `MapClusterPt` for `f` along `l`
and the codomain of `f` is a compact space,
then `f` tends to `𝓝 y` along `l`. -/
lemma tendsto_nhds_of_unique_mapClusterPt [CompactSpace X] {Y} {l : Filter Y} {y : X} {f : Y → X}
    (h : ∀ x, MapClusterPt x l f → x = y) :
    Tendsto f l (𝓝 y) :=
  le_nhds_of_unique_clusterPt h

-- Porting note: a lemma instead of `export` to make `X` explicit

lemma noncompact_univ (X : Type*) [TopologicalSpace X] [NoncompactSpace X] :
    ¬IsCompact (univ : Set X) :=
  NoncompactSpace.noncompact_univ


theorem IsCompact.ne_univ [NoncompactSpace X] (hs : IsCompact s) : s ≠ univ := fun h =>
  noncompact_univ X (h ▸ hs)


instance [NoncompactSpace X] : NeBot (Filter.cocompact X) := by
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s t : Set X
    f : X → Y
    inst✝ : NoncompactSpace X
    ⊢ (Filter.cocompact X).NeBot
  -/
  refine Filter.hasBasis_cocompact.neBot_iff.2 fun hs => ?_
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s t : Set X
    f : X → Y
    inst✝ : NoncompactSpace X
    i✝ : Set X
    hs : IsCompact i✝
    ⊢ (HasCompl.compl i✝).Nonempty
  -/
  contrapose hs; rw [not_nonempty_iff_eq_empty, compl_empty_iff] at hs
  /-
    X : Type u
    Y : Type v
    ι : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    s t : Set X
    f : X → Y
    inst✝ : NoncompactSpace X
    i✝ : Set X
    hs : Eq i✝ Set.univ
    ⊢ Not (IsCompact i✝)
  -/
  rw [hs]; exact noncompact_univ X
           /-
             🎉 no goals
           -/


@[simp]
theorem Filter.cocompact_eq_bot [CompactSpace X] : Filter.cocompact X = ⊥ :=
  Filter.hasBasis_cocompact.eq_bot_iff.mpr ⟨Set.univ, isCompact_univ, Set.compl_univ⟩


instance [NoncompactSpace X] : NeBot (Filter.coclosedCompact X) :=
  neBot_of_le Filter.cocompact_le_coclosedCompact


theorem noncompactSpace_of_neBot (_ : NeBot (Filter.cocompact X)) : NoncompactSpace X :=
  ⟨fun h' => (Filter.nonempty_of_mem h'.compl_mem_cocompact).ne_empty compl_univ⟩


theorem Filter.cocompact_neBot_iff : NeBot (Filter.cocompact X) ↔ NoncompactSpace X :=
  ⟨noncompactSpace_of_neBot, fun _ => inferInstance⟩


theorem not_compactSpace_iff : ¬CompactSpace X ↔ NoncompactSpace X :=
  ⟨fun h₁ => ⟨fun h₂ => h₁ ⟨h₂⟩⟩, fun ⟨h₁⟩ ⟨h₂⟩ => h₁ h₂⟩


instance : NoncompactSpace ℤ :=
                                 /-
                                   X : Type u
                                   Y : Type v
                                   ι : Type u_1
                                   inst✝¹ : TopologicalSpace X
                                   inst✝ : TopologicalSpace Y
                                   s t : Set X
                                   f : X → Y
                                   ⊢ (Filter.cocompact Int).NeBot
                                 -/
  noncompactSpace_of_neBot <| by simp only [Filter.cocompact_eq_cofinite, Filter.cofinite_neBot]
                                 /-
                                   🎉 no goals
                                 -/

-- Note: We can't make this into an instance because it loops with `Finite.compactSpace`.

/-- A compact discrete space is finite. -/
theorem finite_of_compact_of_discrete [CompactSpace X] [DiscreteTopology X] : Finite X :=
  Finite.of_finite_univ <| isCompact_univ.finite_of_discrete


lemma Set.Infinite.exists_accPt_cofinite_inf_principal_of_subset_isCompact
    {K : Set X} (hs : s.Infinite) (hK : IsCompact K) (hsub : s ⊆ K) :
    ∃ x ∈ K, AccPt x (cofinite ⊓ 𝓟 s) :=
  (@hK _ hs.cofinite_inf_principal_neBot (inf_le_right.trans <| principal_mono.2 hsub)).imp
    fun x hx ↦ by rwa [acc_iff_cluster, inf_comm, inf_right_comm,
      (finite_singleton _).cofinite_inf_principal_compl]


lemma Set.Infinite.exists_accPt_of_subset_isCompact {K : Set X} (hs : s.Infinite)
    (hK : IsCompact K) (hsub : s ⊆ K) : ∃ x ∈ K, AccPt x (𝓟 s) :=
  let ⟨x, hxK, hx⟩ := hs.exists_accPt_cofinite_inf_principal_of_subset_isCompact hK hsub
  ⟨x, hxK, hx.mono inf_le_right⟩


lemma Set.Infinite.exists_accPt_cofinite_inf_principal [CompactSpace X] (hs : s.Infinite) :
    ∃ x, AccPt x (cofinite ⊓ 𝓟 s) := by
  simpa only [mem_univ, true_and]
    using hs.exists_accPt_cofinite_inf_principal_of_subset_isCompact isCompact_univ s.subset_univ


lemma Set.Infinite.exists_accPt_principal [CompactSpace X] (hs : s.Infinite) : ∃ x, AccPt x (𝓟 s) :=
  hs.exists_accPt_cofinite_inf_principal.imp fun _x hx ↦ hx.mono inf_le_right


theorem exists_nhds_ne_neBot (X : Type*) [TopologicalSpace X] [CompactSpace X] [Infinite X] :
    ∃ z : X, (𝓝[≠] z).NeBot := by
  /-
    X : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : Infinite X
    ⊢ Exists fun z => (nhdsWithin z (HasCompl.compl (Singleton.singleton z))).NeBot
  -/
  simpa [AccPt] using (@infinite_univ X _).exists_accPt_principal
  /-
    🎉 no goals
  -/


theorem finite_cover_nhds_interior [CompactSpace X] {U : X → Set X} (hU : ∀ x, U x ∈ 𝓝 x) :
    ∃ t : Finset X, ⋃ x ∈ t, interior (U x) = univ :=
  let ⟨t, ht⟩ := isCompact_univ.elim_finite_subcover (fun x => interior (U x))
    (fun _ => isOpen_interior) fun x _ => mem_iUnion.2 ⟨x, mem_interior_iff_mem_nhds.2 (hU x)⟩
  ⟨t, univ_subset_iff.1 ht⟩


theorem finite_cover_nhds [CompactSpace X] {U : X → Set X} (hU : ∀ x, U x ∈ 𝓝 x) :
    ∃ t : Finset X, ⋃ x ∈ t, U x = univ :=
  let ⟨t, ht⟩ := finite_cover_nhds_interior hU
  ⟨t, univ_subset_iff.1 <| ht.symm.subset.trans <| iUnion₂_mono fun _ _ => interior_subset⟩


/-- If `X` is a compact space, then a locally finite family of sets of `X` can have only finitely
many nonempty elements. -/
theorem LocallyFinite.finite_nonempty_of_compact [CompactSpace X] {f : ι → Set X}
    (hf : LocallyFinite f) : { i | (f i).Nonempty }.Finite := by
  /-
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    f : ι → Set X
    hf : LocallyFinite f
    ⊢ (setOf fun i => (f i).Nonempty).Finite
  -/
  simpa only [inter_univ] using hf.finite_nonempty_inter_compact isCompact_univ
  /-
    🎉 no goals
  -/


/-- If `X` is a compact space, then a locally finite family of nonempty sets of `X` can have only
finitely many elements, `Set.Finite` version. -/
theorem LocallyFinite.finite_of_compact [CompactSpace X] {f : ι → Set X}
    (hf : LocallyFinite f) (hne : ∀ i, (f i).Nonempty) : (univ : Set ι).Finite := by
  /-
    X : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    f : ι → Set X
    hf : LocallyFinite f
    hne : ∀ (i : ι), (f i).Nonempty
    ⊢ Set.univ.Finite
  -/
  simpa only [hne] using hf.finite_nonempty_of_compact
  /-
    🎉 no goals
  -/


/-- If `X` is a compact space, then a locally finite family of nonempty sets of `X` can have only
finitely many elements, `Fintype` version. -/
noncomputable def LocallyFinite.fintypeOfCompact [CompactSpace X] {f : ι → Set X}
    (hf : LocallyFinite f) (hne : ∀ i, (f i).Nonempty) : Fintype ι :=
  fintypeOfFiniteUniv (hf.finite_of_compact hne)


/-- The comap of the cocompact filter on `Y` by a continuous function `f : X → Y` is less than or
equal to the cocompact filter on `X`.
This is a reformulation of the fact that images of compact sets are compact. -/
theorem Filter.comap_cocompact_le {f : X → Y} (hf : Continuous f) :
    (Filter.cocompact Y).comap f ≤ Filter.cocompact X := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    ⊢ LE.le (Filter.comap f (Filter.cocompact Y)) (Filter.cocompact X)
  -/
  rw [(Filter.hasBasis_cocompact.comap f).le_basis_iff Filter.hasBasis_cocompact]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    ⊢ ∀ (i' : Set X), IsCompact i' → Exists fun i => And (IsCompact i) (HasSubset. …
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
    ht : IsCompact t
    ⊢ Exists fun i => And (IsCompact i) (HasSubset.Subset (Set.preimage f (HasComp …
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
    ht : IsCompact t
    ⊢ HasSubset.Subset (Set.preimage f (HasCompl.compl (Set.image f t))) (HasCompl …
  -/
  simpa using t.subset_preimage_image f
  /-
    🎉 no goals
  -/


/-- If a filter is disjoint from the cocompact filter, so is its image under any continuous
function. -/
theorem disjoint_map_cocompact {g : X → Y} {f : Filter X} (hg : Continuous g)
    (hf : Disjoint f (Filter.cocompact X)) : Disjoint (map g f) (Filter.cocompact Y) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    g : X → Y
    f : Filter X
    hg : Continuous g
    hf : Disjoint f (Filter.cocompact X)
    ⊢ Disjoint (Filter.map g f) (Filter.cocompact Y)
  -/
  rw [← Filter.disjoint_comap_iff_map, disjoint_iff_inf_le]
  calc
    f ⊓ (comap g (cocompact Y))
    _ ≤ f ⊓ Filter.cocompact X := inf_le_inf_left f (Filter.comap_cocompact_le hg)
    _ = ⊥ := disjoint_iff.mp hf


theorem isCompact_range [CompactSpace X] {f : X → Y} (hf : Continuous f) : IsCompact (range f) := by
  /-
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace X
    f : X → Y
    hf : Continuous f
    ⊢ IsCompact (Set.range f)
  -/
  rw [← image_univ]; exact isCompact_univ.image hf
                     /-
                       🎉 no goals
                     -/


theorem isCompact_diagonal [CompactSpace X] : IsCompact (diagonal X) :=
  @range_diag X ▸ isCompact_range (continuous_id.prod_mk continuous_id)

-- Porting note: renamed, golfed

/-- If `X` is a compact topological space, then `Prod.snd : X × Y → Y` is a closed map. -/
theorem isClosedMap_snd_of_compactSpace [CompactSpace X] :
    IsClosedMap (Prod.snd : X × Y → Y) := fun s hs => by
  /-
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace X
    s : Set (Prod X Y)
    hs : IsClosed s
    ⊢ IsClosed (Set.image Prod.snd s)
  -/
  rw [← isOpen_compl_iff, isOpen_iff_mem_nhds]
  /-
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace X
    s : Set (Prod X Y)
    hs : IsClosed s
    ⊢ ∀ (x : Y), Membership.mem (HasCompl.compl (Set.image Prod.snd s)) x → Member …
  -/
  intro y hy
  have : univ ×ˢ {y} ⊆ sᶜ := by
    exact fun (x, y') ⟨_, rfl⟩ hs => hy ⟨(x, y'), hs, rfl⟩
  rcases generalized_tube_lemma isCompact_univ isCompact_singleton hs.isOpen_compl this
    with ⟨U, V, -, hVo, hU, hV, hs⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace X
    s : Set (Prod X Y)
    hs✝ : IsClosed s
    y : Y
    hy : Membership.mem (HasCompl.compl (Set.image Prod.snd s)) y
    this : HasSubset.Subset (SProd.sprod Set.univ (Singleton.singleton y)) (HasCom …
    U : Set X
    V : Set Y
    hVo : IsOpen V
    hU : HasSubset.Subset Set.univ U
    hV : HasSubset.Subset (Singleton.singleton y) V
    hs : HasSubset.Subset (SProd.sprod U V) (HasCompl.compl s)
    ⊢ Membership.mem (nhds y) (HasCompl.compl (Set.image Prod.snd s))
  -/
  refine mem_nhds_iff.2 ⟨V, ?_, hVo, hV rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace X
    s : Set (Prod X Y)
    hs✝ : IsClosed s
    y : Y
    hy : Membership.mem (HasCompl.compl (Set.image Prod.snd s)) y
    this : HasSubset.Subset (SProd.sprod Set.univ (Singleton.singleton y)) (HasCom …
    U : Set X
    V : Set Y
    hVo : IsOpen V
    hU : HasSubset.Subset Set.univ U
    hV : HasSubset.Subset (Singleton.singleton y) V
    hs : HasSubset.Subset (SProd.sprod U V) (HasCompl.compl s)
    ⊢ HasSubset.Subset V (HasCompl.compl (Set.image Prod.snd s))
  -/
  rintro _ hzV ⟨z, hzs, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace X
    s : Set (Prod X Y)
    hs✝ : IsClosed s
    y : Y
    hy : Membership.mem (HasCompl.compl (Set.image Prod.snd s)) y
    this : HasSubset.Subset (SProd.sprod Set.univ (Singleton.singleton y)) (HasCom …
    U : Set X
    V : Set Y
    hVo : IsOpen V
    hU : HasSubset.Subset Set.univ U
    hV : HasSubset.Subset (Singleton.singleton y) V
    hs : HasSubset.Subset (SProd.sprod U V) (HasCompl.compl s)
    z : Prod X Y
    hzs : Membership.mem s z
    hzV : Membership.mem V z.2
    ⊢ False
  -/
  exact hs ⟨hU trivial, hzV⟩ hzs
  /-
    🎉 no goals
  -/


/-- If `Y` is a compact topological space, then `Prod.fst : X × Y → X` is a closed map. -/
theorem isClosedMap_fst_of_compactSpace [CompactSpace Y] : IsClosedMap (Prod.fst : X × Y → X) :=
  isClosedMap_snd_of_compactSpace.comp isClosedMap_swap


theorem exists_subset_nhds_of_compactSpace [CompactSpace X] [Nonempty ι]
    {V : ι → Set X} (hV : Directed (· ⊇ ·) V) (hV_closed : ∀ i, IsClosed (V i)) {U : Set X}
    (hU : ∀ x ∈ ⋂ i, V i, U ∈ 𝓝 x) : ∃ i, V i ⊆ U :=
  exists_subset_nhds_of_isCompact' hV (fun i => (hV_closed i).isCompact) hV_closed hU


/-- If `f : X → Y` is an inducing map, the image `f '' s` of a set `s` is compact
  if and only if `s` is compact. -/
theorem Topology.IsInducing.isCompact_iff {f : X → Y} (hf : IsInducing f) :
    IsCompact s ↔ IsCompact (f '' s) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ Iff (IsCompact s) (IsCompact (Set.image f s))
  -/
  refine ⟨fun hs => hs.image hf.continuous, fun hs F F_ne_bot F_le => ?_⟩
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
    hs : IsCompact (Set.image f s)
    F : Filter X
    F_ne_bot : F.NeBot
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


@[deprecated (since := "2024-10-28")] alias Inducing.isCompact_iff := IsInducing.isCompact_iff


/-- If `f : X → Y` is an `Embedding`, the image `f '' s` of a set `s` is compact
  if and only if `s` is compact. -/
theorem Topology.IsEmbedding.isCompact_iff {f : X → Y} (hf : IsEmbedding f) :
    IsCompact s ↔ IsCompact (f '' s) := hf.isInducing.isCompact_iff


@[deprecated (since := "2024-10-26")]
alias Embedding.isCompact_iff := IsEmbedding.isCompact_iff


/-- The preimage of a compact set under an inducing map is a compact set. -/
theorem Topology.IsInducing.isCompact_preimage (hf : IsInducing f) (hf' : IsClosed (range f))
    {K : Set Y} (hK : IsCompact K) : IsCompact (f ⁻¹' K) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    hf' : IsClosed (Set.range f)
    K : Set Y
    hK : IsCompact K
    ⊢ IsCompact (Set.preimage f K)
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
    hK : IsCompact (Inter.inter K (Set.range f))
    ⊢ IsCompact (Set.preimage f K)
  -/
  rwa [hf.isCompact_iff, image_preimage_eq_inter_range]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.isCompact_preimage := IsInducing.isCompact_preimage


lemma Topology.IsInducing.isCompact_preimage_iff {f : X → Y} (hf : IsInducing f) {K : Set Y}
    (Kf : K ⊆ range f) : IsCompact (f ⁻¹' K) ↔ IsCompact K := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    K : Set Y
    Kf : HasSubset.Subset K (Set.range f)
    ⊢ Iff (IsCompact (Set.preimage f K)) (IsCompact K)
  -/
  rw [hf.isCompact_iff, image_preimage_eq_of_subset Kf]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Inducing.isCompact_preimage_iff := IsInducing.isCompact_preimage_iff


/-- The preimage of a compact set in the image of an inducing map is compact. -/
lemma Topology.IsInducing.isCompact_preimage' (hf : IsInducing f) {K : Set Y}
    (hK : IsCompact K) (Kf : K ⊆ range f) : IsCompact (f ⁻¹' K) :=
  (hf.isCompact_preimage_iff Kf).2 hK


@[deprecated (since := "2024-10-28")]
alias Inducing.isCompact_preimage' := IsInducing.isCompact_preimage'


/-- The preimage of a compact set under a closed embedding is a compact set. -/
theorem Topology.IsClosedEmbedding.isCompact_preimage (hf : IsClosedEmbedding f)
    {K : Set Y} (hK : IsCompact K) : IsCompact (f ⁻¹' K) :=
  hf.isInducing.isCompact_preimage (hf.isClosed_range) hK


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.isCompact_preimage := IsClosedEmbedding.isCompact_preimage


/-- A closed embedding is proper, ie, inverse images of compact sets are contained in compacts.
Moreover, the preimage of a compact set is compact, see `IsClosedEmbedding.isCompact_preimage`. -/
theorem Topology.IsClosedEmbedding.tendsto_cocompact (hf : IsClosedEmbedding f) :
    Tendsto f (Filter.cocompact X) (Filter.cocompact Y) :=
  Filter.hasBasis_cocompact.tendsto_right_iff.mpr fun _K hK =>
    (hf.isCompact_preimage hK).compl_mem_cocompact


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.tendsto_cocompact := IsClosedEmbedding.tendsto_cocompact


/-- Sets of subtype are compact iff the image under a coercion is. -/
theorem Subtype.isCompact_iff {p : X → Prop} {s : Set { x // p x }} :
    IsCompact s ↔ IsCompact ((↑) '' s : Set X) :=
  IsEmbedding.subtypeVal.isCompact_iff


theorem isCompact_iff_isCompact_univ : IsCompact s ↔ IsCompact (univ : Set s) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsCompact s) (IsCompact Set.univ)
  -/
  rw [Subtype.isCompact_iff, image_univ, Subtype.range_coe]
  /-
    🎉 no goals
  -/


theorem isCompact_iff_compactSpace : IsCompact s ↔ CompactSpace s :=
  isCompact_iff_isCompact_univ.trans isCompact_univ_iff


theorem IsCompact.finite (hs : IsCompact s) (hs' : DiscreteTopology s) : s.Finite :=
  finite_coe_iff.mp (@finite_of_compact_of_discrete _ _ (isCompact_iff_compactSpace.mp hs) hs')


theorem exists_nhds_ne_inf_principal_neBot (hs : IsCompact s) (hs' : s.Infinite) :
    ∃ z ∈ s, (𝓝[≠] z ⊓ 𝓟 s).NeBot :=
  hs'.exists_accPt_of_subset_isCompact hs Subset.rfl


protected theorem Topology.IsClosedEmbedding.noncompactSpace [NoncompactSpace X] {f : X → Y}
    (hf : IsClosedEmbedding f) : NoncompactSpace Y :=
  noncompactSpace_of_neBot hf.tendsto_cocompact.neBot


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.noncompactSpace := IsClosedEmbedding.noncompactSpace


protected theorem Topology.IsClosedEmbedding.compactSpace [h : CompactSpace Y] {f : X → Y}
    (hf : IsClosedEmbedding f) : CompactSpace X :=
      /-
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        h : CompactSpace Y
        f : X → Y
        hf : Topology.IsClosedEmbedding f
        ⊢ IsCompact Set.univ
      -/
  ⟨by rw [hf.isInducing.isCompact_iff, image_univ]; exact hf.isClosed_range.isCompact⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.compactSpace := IsClosedEmbedding.compactSpace


theorem IsCompact.prod {t : Set Y} (hs : IsCompact s) (ht : IsCompact t) :
    IsCompact (s ×ˢ t) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : IsCompact s
    ht : IsCompact t
    ⊢ IsCompact (SProd.sprod s t)
  -/
  rw [isCompact_iff_ultrafilter_le_nhds'] at hs ht ⊢
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : ∀ (f : Ultrafilter X), Membership.mem f s → Exists fun x => And (Membersh …
    ht : ∀ (f : Ultrafilter Y), Membership.mem f t → Exists fun x => And (Membersh …
    ⊢ ∀ (f : Ultrafilter (Prod X Y)), Membership.mem f (SProd.sprod s t) → Exists  …
  -/
  intro f hfs
  obtain ⟨x : X, sx : x ∈ s, hx : map Prod.fst f.1 ≤ 𝓝 x⟩ :=
    hs (f.map Prod.fst) (mem_map.2 <| mem_of_superset hfs fun x => And.left)
  obtain ⟨y : Y, ty : y ∈ t, hy : map Prod.snd f.1 ≤ 𝓝 y⟩ :=
    ht (f.map Prod.snd) (mem_map.2 <| mem_of_superset hfs fun x => And.right)
  /-
    case intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : ∀ (f : Ultrafilter X), Membership.mem f s → Exists fun x => And (Membersh …
    ht : ∀ (f : Ultrafilter Y), Membership.mem f t → Exists fun x => And (Membersh …
    f : Ultrafilter (Prod X Y)
    hfs : Membership.mem f (SProd.sprod s t)
    x : X
    sx : Membership.mem s x
    hx : LE.le (Filter.map Prod.fst ↑f) (nhds x)
    y : Y
    ty : Membership.mem t y
    hy : LE.le (Filter.map Prod.snd ↑f) (nhds y)
    ⊢ Exists fun x => And (Membership.mem (SProd.sprod s t) x) (LE.le (↑f) (nhds x))
  -/
  rw [map_le_iff_le_comap] at hx hy
  /-
    case intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : ∀ (f : Ultrafilter X), Membership.mem f s → Exists fun x => And (Membersh …
    ht : ∀ (f : Ultrafilter Y), Membership.mem f t → Exists fun x => And (Membersh …
    f : Ultrafilter (Prod X Y)
    hfs : Membership.mem f (SProd.sprod s t)
    x : X
    sx : Membership.mem s x
    hx : LE.le (↑f) (Filter.comap Prod.fst (nhds x))
    y : Y
    ty : Membership.mem t y
    hy : LE.le (↑f) (Filter.comap Prod.snd (nhds y))
    ⊢ Exists fun x => And (Membership.mem (SProd.sprod s t) x) (LE.le (↑f) (nhds x))
  -/
  refine ⟨⟨x, y⟩, ⟨sx, ty⟩, ?_⟩
  /-
    case intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : ∀ (f : Ultrafilter X), Membership.mem f s → Exists fun x => And (Membersh …
    ht : ∀ (f : Ultrafilter Y), Membership.mem f t → Exists fun x => And (Membersh …
    f : Ultrafilter (Prod X Y)
    hfs : Membership.mem f (SProd.sprod s t)
    x : X
    sx : Membership.mem s x
    hx : LE.le (↑f) (Filter.comap Prod.fst (nhds x))
    y : Y
    ty : Membership.mem t y
    hy : LE.le (↑f) (Filter.comap Prod.snd (nhds y))
    ⊢ LE.le (↑f) (nhds { fst := x, snd := y })
  -/
  rw [nhds_prod_eq]; exact le_inf hx hy
                     /-
                       🎉 no goals
                     -/


/-- Finite topological spaces are compact. -/
instance (priority := 100) Finite.compactSpace [Finite X] : CompactSpace X where
  isCompact_univ := finite_univ.isCompact


instance ULift.compactSpace [CompactSpace X] : CompactSpace (ULift.{v} X) :=
  IsClosedEmbedding.uliftDown.compactSpace


/-- The product of two compact spaces is compact. -/
instance [CompactSpace X] [CompactSpace Y] : CompactSpace (X × Y) :=
      /-
        X : Type u
        Y : Type v
        ι : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        s t : Set X
        f : X → Y
        inst✝¹ : CompactSpace X
        inst✝ : CompactSpace Y
        ⊢ IsCompact Set.univ
      -/
  ⟨by rw [← univ_prod_univ]; exact isCompact_univ.prod isCompact_univ⟩
                             /-
                               🎉 no goals
                             -/


/-- The disjoint union of two compact spaces is compact. -/
instance [CompactSpace X] [CompactSpace Y] : CompactSpace (X ⊕ Y) :=
  ⟨by
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      s t : Set X
      f : X → Y
      inst✝¹ : CompactSpace X
      inst✝ : CompactSpace Y
      ⊢ IsCompact Set.univ
    -/
    rw [← range_inl_union_range_inr]
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      s t : Set X
      f : X → Y
      inst✝¹ : CompactSpace X
      inst✝ : CompactSpace Y
      ⊢ IsCompact (Union.union (Set.range Sum.inl) (Set.range Sum.inr))
    -/
    exact (isCompact_range continuous_inl).union (isCompact_range continuous_inr)⟩
    /-
      🎉 no goals
    -/


instance {X : ι → Type*} [Finite ι] [∀ i, TopologicalSpace (X i)] [∀ i, CompactSpace (X i)] :
    CompactSpace (Σi, X i) := by
  /-
    X✝ : Type u
    Y : Type v
    ι : Type u_1
    inst✝⁴ : TopologicalSpace X✝
    inst✝³ : TopologicalSpace Y
    s t : Set X✝
    f : X✝ → Y
    X : ι → Type u_2
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), CompactSpace (X i)
    ⊢ CompactSpace (Sigma fun i => X i)
  -/
  refine ⟨?_⟩
  /-
    X✝ : Type u
    Y : Type v
    ι : Type u_1
    inst✝⁴ : TopologicalSpace X✝
    inst✝³ : TopologicalSpace Y
    s t : Set X✝
    f : X✝ → Y
    X : ι → Type u_2
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), CompactSpace (X i)
    ⊢ IsCompact Set.univ
  -/
  rw [Sigma.univ]
  /-
    X✝ : Type u
    Y : Type v
    ι : Type u_1
    inst✝⁴ : TopologicalSpace X✝
    inst✝³ : TopologicalSpace Y
    s t : Set X✝
    f : X✝ → Y
    X : ι → Type u_2
    inst✝² : Finite ι
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : ∀ (i : ι), CompactSpace (X i)
    ⊢ IsCompact (Set.iUnion fun a => Set.range (Sigma.mk a))
  -/
  exact isCompact_iUnion fun i => isCompact_range continuous_sigmaMk
  /-
    🎉 no goals
  -/


/-- The coproduct of the cocompact filters on two topological spaces is the cocompact filter on
their product. -/
theorem Filter.coprod_cocompact :
    (Filter.cocompact X).coprod (Filter.cocompact Y) = Filter.cocompact (X × Y) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ Eq ((Filter.cocompact X).coprod (Filter.cocompact Y)) (Filter.cocompact (Pro …
  -/
  apply le_antisymm
    /-
      case a
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      ⊢ LE.le ((Filter.cocompact X).coprod (Filter.cocompact Y)) (Filter.cocompact ( …
    -/
  · exact sup_le (comap_cocompact_le continuous_fst) (comap_cocompact_le continuous_snd)
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      ⊢ LE.le (Filter.cocompact (Prod X Y)) ((Filter.cocompact X).coprod (Filter.coc …
    -/
  · refine (hasBasis_cocompact.coprod hasBasis_cocompact).ge_iff.2 fun K hK ↦ ?_
    /-
      case a
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      K : Prod (Set X) (Set Y)
      hK : And (IsCompact K.1) (IsCompact K.2)
      ⊢ Membership.mem (Filter.cocompact (Prod X Y)) (Union.union (Set.preimage Prod …
    -/
    rw [← univ_prod, ← prod_univ, ← compl_prod_eq_union]
    /-
      case a
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      K : Prod (Set X) (Set Y)
      hK : And (IsCompact K.1) (IsCompact K.2)
      ⊢ Membership.mem (Filter.cocompact (Prod X Y)) (HasCompl.compl (SProd.sprod K. …
    -/
    exact (hK.1.prod hK.2).compl_mem_cocompact
    /-
      🎉 no goals
    -/


theorem Prod.noncompactSpace_iff :
    NoncompactSpace (X × Y) ↔ NoncompactSpace X ∧ Nonempty Y ∨ Nonempty X ∧ NoncompactSpace Y := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ Iff (NoncompactSpace (Prod X Y)) (Or (And (NoncompactSpace X) (Nonempty Y))  …
  -/
  simp [← Filter.cocompact_neBot_iff, ← Filter.coprod_cocompact, Filter.coprod_neBot_iff]
  /-
    🎉 no goals
  -/

-- See Note [lower instance priority]

instance (priority := 100) Prod.noncompactSpace_left [NoncompactSpace X] [Nonempty Y] :
    NoncompactSpace (X × Y) :=
  Prod.noncompactSpace_iff.2 (Or.inl ⟨‹_›, ‹_›⟩)

-- See Note [lower instance priority]

instance (priority := 100) Prod.noncompactSpace_right [Nonempty X] [NoncompactSpace Y] :
    NoncompactSpace (X × Y) :=
  Prod.noncompactSpace_iff.2 (Or.inr ⟨‹_›, ‹_›⟩)


/-- **Tychonoff's theorem**: product of compact sets is compact. -/
theorem isCompact_pi_infinite {s : ∀ i, Set (X i)} :
    (∀ i, IsCompact (s i)) → IsCompact { x : ∀ i, X i | ∀ i, x i ∈ s i } := by
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : (i : ι) → Set (X i)
    ⊢ (∀ (i : ι), IsCompact (s i)) → IsCompact (setOf fun x => ∀ (i : ι), Membersh …
  -/
  simp only [isCompact_iff_ultrafilter_le_nhds, nhds_pi, le_pi, le_principal_iff]
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : (i : ι) → Set (X i)
    ⊢ (∀ (i : ι) (f : Ultrafilter (X i)), Membership.mem (↑f) (s i) → Exists fun x …
  -/
  intro h f hfs
  have : ∀ i : ι, ∃ x, x ∈ s i ∧ Tendsto (Function.eval i) f (𝓝 x) := by
    refine fun i => h i (f.map _) (mem_map.2 ?_)
    exact mem_of_superset hfs fun x hx => hx i
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : (i : ι) → Set (X i)
    h : ∀ (i : ι) (f : Ultrafilter (X i)), Membership.mem (↑f) (s i) → Exists fun  …
    f : Ultrafilter ((i : ι) → X i)
    hfs : Membership.mem (↑f) (setOf fun x => ∀ (i : ι), Membership.mem (s i) (x i))
    this : ∀ (i : ι), Exists fun x => And (Membership.mem (s i) x) (Filter.Tendsto …
    ⊢ Exists fun x => And (Membership.mem (setOf fun x => ∀ (i : ι), Membership.me …
  -/
  choose x hx using this
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : (i : ι) → Set (X i)
    h : ∀ (i : ι) (f : Ultrafilter (X i)), Membership.mem (↑f) (s i) → Exists fun  …
    f : Ultrafilter ((i : ι) → X i)
    hfs : Membership.mem (↑f) (setOf fun x => ∀ (i : ι), Membership.mem (s i) (x i))
    x : (i : ι) → X i
    hx : ∀ (i : ι), And (Membership.mem (s i) (x i)) (Filter.Tendsto (Function.eva …
    ⊢ Exists fun x => And (Membership.mem (setOf fun x => ∀ (i : ι), Membership.me …
  -/
  exact ⟨x, fun i => (hx i).left, fun i => (hx i).right⟩
  /-
    🎉 no goals
  -/


/-- **Tychonoff's theorem** formulated using `Set.pi`: product of compact sets is compact. -/
theorem isCompact_univ_pi {s : ∀ i, Set (X i)} (h : ∀ i, IsCompact (s i)) :
    IsCompact (pi univ s) := by
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : (i : ι) → Set (X i)
    h : ∀ (i : ι), IsCompact (s i)
    ⊢ IsCompact (Set.univ.pi s)
  -/
  convert isCompact_pi_infinite h
  /-
    case h.e'_3
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : (i : ι) → Set (X i)
    h : ∀ (i : ι), IsCompact (s i)
    ⊢ Eq (Set.univ.pi s) (setOf fun x => ∀ (i : ι), Membership.mem (s i) (x i))
  -/
  simp only [← mem_univ_pi, setOf_mem_eq]
  /-
    🎉 no goals
  -/


instance Pi.compactSpace [∀ i, CompactSpace (X i)] : CompactSpace (∀ i, X i) :=
      /-
        X✝ : Type u
        Y : Type v
        ι : Type u_1
        inst✝³ : TopologicalSpace X✝
        inst✝² : TopologicalSpace Y
        s t : Set X✝
        f : X✝ → Y
        X : ι → Type u_2
        inst✝¹ : (i : ι) → TopologicalSpace (X i)
        inst✝ : ∀ (i : ι), CompactSpace (X i)
        ⊢ IsCompact Set.univ
      -/
  ⟨by rw [← pi_univ univ]; exact isCompact_univ_pi fun i => isCompact_univ⟩
                           /-
                             🎉 no goals
                           -/


instance Function.compactSpace [CompactSpace Y] : CompactSpace (ι → Y) :=
  Pi.compactSpace


lemma Pi.isCompact_iff_of_isClosed {s : Set (Π i, X i)} (hs : IsClosed s) :
    IsCompact s ↔ ∀ i, IsCompact (eval i '' s) := by
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : Set ((i : ι) → X i)
    hs : IsClosed s
    ⊢ Iff (IsCompact s) (∀ (i : ι), IsCompact (Set.image (Function.eval i) s))
  -/
  constructor <;> intro H
    /-
      case mp
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s : Set ((i : ι) → X i)
      hs : IsClosed s
      H : IsCompact s
      ⊢ ∀ (i : ι), IsCompact (Set.image (Function.eval i) s)
    -/
  · exact fun i ↦ H.image <| continuous_apply i
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s : Set ((i : ι) → X i)
      hs : IsClosed s
      H : ∀ (i : ι), IsCompact (Set.image (Function.eval i) s)
      ⊢ IsCompact s
    -/
  · exact IsCompact.of_isClosed_subset (isCompact_univ_pi H) hs (subset_pi_eval_image univ s)
    /-
      🎉 no goals
    -/


protected lemma Pi.exists_compact_superset_iff {s : Set (Π i, X i)} :
    (∃ K, IsCompact K ∧ s ⊆ K) ↔ ∀ i, ∃ Ki, IsCompact Ki ∧ s ⊆ eval i ⁻¹' Ki := by
  /-
    ι : Type u_1
    X : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (X i)
    s : Set ((i : ι) → X i)
    ⊢ Iff (Exists fun K => And (IsCompact K) (HasSubset.Subset s K)) (∀ (i : ι), E …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s : Set ((i : ι) → X i)
      ⊢ (Exists fun K => And (IsCompact K) (HasSubset.Subset s K)) → ∀ (i : ι), Exis …
    -/
  · intro ⟨K, hK, hsK⟩ i
    /-
      case mp
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s K : Set ((i : ι) → X i)
      hK : IsCompact K
      hsK : HasSubset.Subset s K
      i : ι
      ⊢ Exists fun Ki => And (IsCompact Ki) (HasSubset.Subset s (Set.preimage (Funct …
    -/
    exact ⟨eval i '' K, hK.image <| continuous_apply i, hsK.trans <| K.subset_preimage_image _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s : Set ((i : ι) → X i)
      ⊢ (∀ (i : ι), Exists fun Ki => And (IsCompact Ki) (HasSubset.Subset s (Set.pre …
    -/
  · intro H
    /-
      case mpr
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s : Set ((i : ι) → X i)
      H : ∀ (i : ι), Exists fun Ki => And (IsCompact Ki) (HasSubset.Subset s (Set.pr …
      ⊢ Exists fun K => And (IsCompact K) (HasSubset.Subset s K)
    -/
    choose K hK hsK using H
    /-
      case mpr
      ι : Type u_1
      X : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (X i)
      s : Set ((i : ι) → X i)
      K : (i : ι) → Set (X i)
      hK : ∀ (i : ι), IsCompact (K i)
      hsK : ∀ (i : ι), HasSubset.Subset s (Set.preimage (Function.eval i) (K i))
      ⊢ Exists fun K => And (IsCompact K) (HasSubset.Subset s K)
    -/
    exact ⟨pi univ K, isCompact_univ_pi hK, fun _ hx i _ ↦ hsK i hx⟩
    /-
      🎉 no goals
    -/


/-- **Tychonoff's theorem** formulated in terms of filters: `Filter.cocompact` on an indexed product
type `Π d, X d` the `Filter.coprodᵢ` of filters `Filter.cocompact` on `X d`. -/
theorem Filter.coprodᵢ_cocompact {X : ι → Type*} [∀ d, TopologicalSpace (X d)] :
    (Filter.coprodᵢ fun d => Filter.cocompact (X d)) = Filter.cocompact (∀ d, X d) := by
  /-
    ι : Type u_1
    X : ι → Type u_3
    inst✝ : (d : ι) → TopologicalSpace (X d)
    ⊢ Eq (Filter.coprodᵢ fun d => Filter.cocompact (X d)) (Filter.cocompact ((d :  …
  -/
  refine le_antisymm (iSup_le fun i => Filter.comap_cocompact_le (continuous_apply i)) ?_
  /-
    ι : Type u_1
    X : ι → Type u_3
    inst✝ : (d : ι) → TopologicalSpace (X d)
    ⊢ LE.le (Filter.cocompact ((d : ι) → X d)) (Filter.coprodᵢ fun d => Filter.coc …
  -/
  refine compl_surjective.forall.2 fun s H => ?_
  /-
    ι : Type u_1
    X : ι → Type u_3
    inst✝ : (d : ι) → TopologicalSpace (X d)
    s : Set ((i : ι) → X i)
    H : Membership.mem (Filter.coprodᵢ fun d => Filter.cocompact (X d)) (HasCompl. …
    ⊢ Membership.mem (Filter.cocompact ((d : ι) → X d)) (HasCompl.compl s)
  -/
  simp only [compl_mem_coprodᵢ, Filter.mem_cocompact, compl_subset_compl, image_subset_iff] at H ⊢
  /-
    ι : Type u_1
    X : ι → Type u_3
    inst✝ : (d : ι) → TopologicalSpace (X d)
    s : Set ((i : ι) → X i)
    H : ∀ (i : ι), Exists fun t => And (IsCompact t) (HasSubset.Subset s (Set.prei …
    ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset s t)
  -/
  choose K hKc htK using H
  /-
    ι : Type u_1
    X : ι → Type u_3
    inst✝ : (d : ι) → TopologicalSpace (X d)
    s : Set ((i : ι) → X i)
    K : (i : ι) → Set (X i)
    hKc : ∀ (i : ι), IsCompact (K i)
    htK : ∀ (i : ι), HasSubset.Subset s (Set.preimage (Function.eval i) (K i))
    ⊢ Exists fun t => And (IsCompact t) (HasSubset.Subset s t)
  -/
  exact ⟨Set.pi univ K, isCompact_univ_pi hKc, fun f hf i _ => htK i hf⟩
  /-
    🎉 no goals
  -/


instance Quot.compactSpace {r : X → X → Prop} [CompactSpace X] : CompactSpace (Quot r) :=
  ⟨by
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t : Set X
      f : X → Y
      r : X → X → Prop
      inst✝ : CompactSpace X
      ⊢ IsCompact Set.univ
    -/
    rw [← range_quot_mk]
    /-
      X : Type u
      Y : Type v
      ι : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      s t : Set X
      f : X → Y
      r : X → X → Prop
      inst✝ : CompactSpace X
      ⊢ IsCompact (Set.range (Quot.mk r))
    -/
    exact isCompact_range continuous_quot_mk⟩
    /-
      🎉 no goals
    -/


instance Quotient.compactSpace {s : Setoid X} [CompactSpace X] : CompactSpace (Quotient s) :=
  Quot.compactSpace


theorem IsClosed.exists_minimal_nonempty_closed_subset [CompactSpace X] {S : Set X}
    (hS : IsClosed S) (hne : S.Nonempty) :
    ∃ V : Set X, V ⊆ S ∧ V.Nonempty ∧ IsClosed V ∧
      ∀ V' : Set X, V' ⊆ V → V'.Nonempty → IsClosed V' → V' = V := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    ⊢ Exists fun V => And (HasSubset.Subset V S) (And V.Nonempty (And (IsClosed V) …
  -/
  let opens := { U : Set X | Sᶜ ⊆ U ∧ IsOpen U ∧ Uᶜ.Nonempty }
  obtain ⟨U, h⟩ :=
    zorn_subset opens fun c hc hz => by
      by_cases hcne : c.Nonempty
      · obtain ⟨U₀, hU₀⟩ := hcne
        haveI : Nonempty { U // U ∈ c } := ⟨⟨U₀, hU₀⟩⟩
        obtain ⟨U₀compl, -, -⟩ := hc hU₀
        use ⋃₀ c
        refine ⟨⟨?_, ?_, ?_⟩, fun U hU _ hx => ⟨U, hU, hx⟩⟩
        · exact fun _ hx => ⟨U₀, hU₀, U₀compl hx⟩
        · exact isOpen_sUnion fun _ h => (hc h).2.1
        · convert_to (⋂ U : { U // U ∈ c }, U.1ᶜ).Nonempty
          · ext
            simp only [not_exists, exists_prop, not_and, Set.mem_iInter, Subtype.forall,
              mem_setOf_eq, mem_compl_iff, mem_sUnion]
          apply IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed
          · rintro ⟨U, hU⟩ ⟨U', hU'⟩
            obtain ⟨V, hVc, hVU, hVU'⟩ := hz.directedOn U hU U' hU'
            exact ⟨⟨V, hVc⟩, Set.compl_subset_compl.mpr hVU, Set.compl_subset_compl.mpr hVU'⟩
          · exact fun U => (hc U.2).2.2
          · exact fun U => (hc U.2).2.1.isClosed_compl.isCompact
          · exact fun U => (hc U.2).2.1.isClosed_compl
      · use Sᶜ
        refine ⟨⟨Set.Subset.refl _, isOpen_compl_iff.mpr hS, ?_⟩, fun U Uc => (hcne ⟨U, Uc⟩).elim⟩
        rw [compl_compl]
        exact hne
  /-
    case intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    opens : Set (Set X) := setOf fun U => And (HasSubset.Subset (HasCompl.compl S) …
    U : Set X
    h : Maximal (fun x => Membership.mem opens x) U
    ⊢ Exists fun V => And (HasSubset.Subset V S) (And V.Nonempty (And (IsClosed V) …
  -/
  obtain ⟨Uc, Uo, Ucne⟩ := h.prop
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    opens : Set (Set X) := setOf fun U => And (HasSubset.Subset (HasCompl.compl S) …
    U : Set X
    h : Maximal (fun x => Membership.mem opens x) U
    Uc : HasSubset.Subset (HasCompl.compl S) U
    Uo : IsOpen U
    Ucne : (HasCompl.compl U).Nonempty
    ⊢ Exists fun V => And (HasSubset.Subset V S) (And V.Nonempty (And (IsClosed V) …
  -/
  refine ⟨Uᶜ, Set.compl_subset_comm.mp Uc, Ucne, Uo.isClosed_compl, ?_⟩
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    opens : Set (Set X) := setOf fun U => And (HasSubset.Subset (HasCompl.compl S) …
    U : Set X
    h : Maximal (fun x => Membership.mem opens x) U
    Uc : HasSubset.Subset (HasCompl.compl S) U
    Uo : IsOpen U
    Ucne : (HasCompl.compl U).Nonempty
    ⊢ ∀ (V' : Set X), HasSubset.Subset V' (HasCompl.compl U) → V'.Nonempty → IsClo …
  -/
  intro V' V'sub V'ne V'cls
  have : V'ᶜ = U := by
    refine h.eq_of_ge ⟨?_, isOpen_compl_iff.mpr V'cls, ?_⟩ (subset_compl_comm.2 V'sub)
    · exact Set.Subset.trans Uc (Set.subset_compl_comm.mp V'sub)
    · simp only [compl_compl, V'ne]
  /-
    case intro.intro.intro
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    S : Set X
    hS : IsClosed S
    hne : S.Nonempty
    opens : Set (Set X) := setOf fun U => And (HasSubset.Subset (HasCompl.compl S) …
    U : Set X
    h : Maximal (fun x => Membership.mem opens x) U
    Uc : HasSubset.Subset (HasCompl.compl S) U
    Uo : IsOpen U
    Ucne : (HasCompl.compl U).Nonempty
    V' : Set X
    V'sub : HasSubset.Subset V' (HasCompl.compl U)
    V'ne : V'.Nonempty
    V'cls : IsClosed V'
    this : Eq (HasCompl.compl V') U
    ⊢ Eq V' (HasCompl.compl U)
  -/
  rw [← this, compl_compl]
  /-
    🎉 no goals
  -/


