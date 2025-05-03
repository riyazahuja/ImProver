lemma subset_coborder :
    s ⊆ coborder s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ HasSubset.Subset s (coborder s)
  -/
  rw [coborder, subset_compl_iff_disjoint_right]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Disjoint s (SDiff.sdiff (closure s) s)
  -/
  exact disjoint_sdiff_self_right
  /-
    🎉 no goals
  -/


lemma coborder_inter_closure :
    coborder s ∩ closure s = s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (Inter.inter (coborder s) (closure s)) s
  -/
  rw [coborder, ← diff_eq_compl_inter, diff_diff_right_self, inter_eq_right]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ HasSubset.Subset s (closure s)
  -/
  exact subset_closure
  /-
    🎉 no goals
  -/


lemma closure_inter_coborder :
    closure s ∩ coborder s = s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (Inter.inter (closure s) (coborder s)) s
  -/
  rw [inter_comm, coborder_inter_closure]
  /-
    🎉 no goals
  -/


lemma coborder_eq_union_frontier_compl :
    coborder s = s ∪ (frontier s)ᶜ := by
  rw [coborder, compl_eq_comm, compl_union, compl_compl, ← diff_eq_compl_inter,
    ← union_diff_right, union_comm, ← closure_eq_self_union_frontier]


lemma coborder_eq_univ_iff :
    coborder s = univ ↔ IsClosed s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Eq (coborder s) Set.univ) (IsClosed s)
  -/
  simp [coborder, diff_eq_empty, closure_subset_iff_isClosed]
  /-
    🎉 no goals
  -/


alias ⟨_, IsClosed.coborder_eq⟩ := coborder_eq_univ_iff


lemma coborder_eq_compl_frontier_iff :
    coborder s = (frontier s)ᶜ ↔ IsOpen s := by
  simp_rw [coborder_eq_union_frontier_compl, union_eq_right, subset_compl_iff_disjoint_left,
    disjoint_frontier_iff_isOpen]


theorem coborder_eq_union_closure_compl {s : Set X} : coborder s = s ∪ (closure s)ᶜ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (coborder s) (Union.union s (HasCompl.compl (closure s)))
  -/
  rw [coborder, compl_eq_comm, compl_union, compl_compl, inter_comm]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq (Inter.inter (closure s) (HasCompl.compl s)) (SDiff.sdiff (closure s) s)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The coborder of any set is dense -/
theorem dense_coborder {s : Set X} :
    Dense (coborder s) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Dense (coborder s)
  -/
  rw [dense_iff_closure_eq, coborder_eq_union_closure_compl, closure_union, ← univ_subset_iff]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ HasSubset.Subset Set.univ (Union.union (closure s) (closure (HasCompl.compl  …
  -/
  refine _root_.subset_trans ?_ (union_subset_union_right _ (subset_closure))
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ HasSubset.Subset Set.univ (Union.union (closure s) (HasCompl.compl (closure  …
  -/
  simp
  /-
    🎉 no goals
  -/


alias ⟨_, IsOpen.coborder_eq⟩ := coborder_eq_compl_frontier_iff


lemma IsOpenMap.coborder_preimage_subset (hf : IsOpenMap f) (s : Set Y) :
    coborder (f ⁻¹' s) ⊆ f ⁻¹' (coborder s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : IsOpenMap f
    s : Set Y
    ⊢ HasSubset.Subset (coborder (Set.preimage f s)) (Set.preimage f (coborder s))
  -/
  rw [coborder, coborder, preimage_compl, preimage_diff, compl_subset_compl]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : IsOpenMap f
    s : Set Y
    ⊢ HasSubset.Subset (SDiff.sdiff (Set.preimage f (closure s)) (Set.preimage f s …
  -/
  apply diff_subset_diff_left
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : IsOpenMap f
    s : Set Y
    ⊢ HasSubset.Subset (Set.preimage f (closure s)) (closure (Set.preimage f s))
  -/
  exact hf.preimage_closure_subset_closure_preimage
  /-
    🎉 no goals
  -/


lemma Continuous.preimage_coborder_subset (hf : Continuous f) (s : Set Y) :
    f ⁻¹' (coborder s) ⊆ coborder (f ⁻¹' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    s : Set Y
    ⊢ HasSubset.Subset (Set.preimage f (coborder s)) (coborder (Set.preimage f s))
  -/
  rw [coborder, coborder, preimage_compl, preimage_diff, compl_subset_compl]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    s : Set Y
    ⊢ HasSubset.Subset (SDiff.sdiff (closure (Set.preimage f s)) (Set.preimage f s …
  -/
  apply diff_subset_diff_left
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    s : Set Y
    ⊢ HasSubset.Subset (closure (Set.preimage f s)) (Set.preimage f (closure s))
  -/
  exact hf.closure_preimage_subset s
  /-
    🎉 no goals
  -/


lemma coborder_preimage (hf : IsOpenMap f) (hf' : Continuous f) (s : Set Y) :
    coborder (f ⁻¹' s) = f ⁻¹' (coborder s) :=
  (hf.coborder_preimage_subset s).antisymm (hf'.preimage_coborder_subset s)


protected
lemma Topology.IsOpenEmbedding.coborder_preimage (hf : IsOpenEmbedding f) (s : Set Y) :
    coborder (f ⁻¹' s) = f ⁻¹' coborder s :=
  coborder_preimage hf.isOpenMap hf.continuous s


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.coborder_preimage := IsOpenEmbedding.coborder_preimage


lemma isClosed_preimage_val_coborder :
    IsClosed (coborder s ↓∩ s) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ IsClosed (Set.preimage Subtype.val s)
  -/
  rw [isClosed_preimage_val, inter_eq_right.mpr subset_coborder, coborder_inter_closure]
  /-
    🎉 no goals
  -/


lemma IsLocallyClosed.inter (hs : IsLocallyClosed s) (ht : IsLocallyClosed t) :
    IsLocallyClosed (s ∩ t) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsLocallyClosed s
    ht : IsLocallyClosed t
    ⊢ IsLocallyClosed (Inter.inter s t)
  -/
  obtain ⟨U₁, Z₁, hU₁, hZ₁, rfl⟩ := hs
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    t : Set X
    ht : IsLocallyClosed t
    U₁ Z₁ : Set X
    hU₁ : IsOpen U₁
    hZ₁ : IsClosed Z₁
    ⊢ IsLocallyClosed (Inter.inter (Inter.inter U₁ Z₁) t)
  -/
  obtain ⟨U₂, Z₂, hU₂, hZ₂, rfl⟩ := ht
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    U₁ Z₁ : Set X
    hU₁ : IsOpen U₁
    hZ₁ : IsClosed Z₁
    U₂ Z₂ : Set X
    hU₂ : IsOpen U₂
    hZ₂ : IsClosed Z₂
    ⊢ IsLocallyClosed (Inter.inter (Inter.inter U₁ Z₁) (Inter.inter U₂ Z₂))
  -/
  refine ⟨_, _, hU₁.inter hU₂, hZ₁.inter hZ₂, inter_inter_inter_comm U₁ Z₁ U₂ Z₂⟩
  /-
    🎉 no goals
  -/


lemma IsLocallyClosed.preimage {s : Set Y} (hs : IsLocallyClosed s)
    {f : X → Y} (hf : Continuous f) :
    IsLocallyClosed (f ⁻¹' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    hs : IsLocallyClosed s
    f : X → Y
    hf : Continuous f
    ⊢ IsLocallyClosed (Set.preimage f s)
  -/
  obtain ⟨U, Z, hU, hZ, rfl⟩ := hs
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Continuous f
    U Z : Set Y
    hU : IsOpen U
    hZ : IsClosed Z
    ⊢ IsLocallyClosed (Set.preimage f (Inter.inter U Z))
  -/
  exact ⟨_, _, hU.preimage hf, hZ.preimage hf, preimage_inter⟩
  /-
    🎉 no goals
  -/


nonrec
lemma Topology.IsInducing.isLocallyClosed_iff {s : Set X}
    {f : X → Y} (hf : IsInducing f) :
    IsLocallyClosed s ↔ ∃ s' : Set Y, IsLocallyClosed s' ∧ f ⁻¹' s' = s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ Iff (IsLocallyClosed s) (Exists fun s' => And (IsLocallyClosed s') (Eq (Set. …
  -/
  simp_rw [IsLocallyClosed, hf.isOpen_iff, hf.isClosed_iff]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    f : X → Y
    hf : Topology.IsInducing f
    ⊢ Iff (Exists fun U => Exists fun Z => And (Exists fun t => And (IsOpen t) (Eq …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      f : X → Y
      hf : Topology.IsInducing f
      ⊢ (Exists fun U => Exists fun Z => And (Exists fun t => And (IsOpen t) (Eq (Se …
    -/
  · rintro ⟨_, _, ⟨U, hU, rfl⟩, ⟨Z, hZ, rfl⟩, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : Topology.IsInducing f
      U : Set Y
      hU : IsOpen U
      Z : Set Y
      hZ : IsClosed Z
      ⊢ Exists fun s' => And (Exists fun U => Exists fun Z => And (IsOpen U) (And (I …
    -/
    exact ⟨_, ⟨U, Z, hU, hZ, rfl⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      f : X → Y
      hf : Topology.IsInducing f
      ⊢ (Exists fun s' => And (Exists fun U => Exists fun Z => And (IsOpen U) (And ( …
    -/
  · rintro ⟨_, ⟨U, Z, hU, hZ, rfl⟩, rfl⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : Topology.IsInducing f
      U Z : Set Y
      hU : IsOpen U
      hZ : IsClosed Z
      ⊢ Exists fun U_1 => Exists fun Z_1 => And (Exists fun t => And (IsOpen t) (Eq  …
    -/
    exact ⟨_, _, ⟨U, hU, rfl⟩, ⟨Z, hZ, rfl⟩, rfl⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-28")]
alias Inducing.isLocallyClosed_iff := IsInducing.isLocallyClosed_iff


lemma Topology.IsEmbedding.isLocallyClosed_iff {s : Set X}
    {f : X → Y} (hf : IsEmbedding f) :
    IsLocallyClosed s ↔ ∃ s' : Set Y, IsLocallyClosed s' ∧ s' ∩ range f = f '' s := by
  simp_rw [hf.isInducing.isLocallyClosed_iff,
    ← (image_injective.mpr hf.injective).eq_iff, image_preimage_eq_inter_range]


@[deprecated (since := "2024-10-26")]
alias Embedding.isLocallyClosed_iff := IsEmbedding.isLocallyClosed_iff


lemma IsLocallyClosed.image {s : Set X} (hs : IsLocallyClosed s)
    {f : X → Y} (hf : IsInducing f) (hf' : IsLocallyClosed (range f)) :
    IsLocallyClosed (f '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    hs : IsLocallyClosed s
    f : X → Y
    hf : Topology.IsInducing f
    hf' : IsLocallyClosed (Set.range f)
    ⊢ IsLocallyClosed (Set.image f s)
  -/
  obtain ⟨t, ht, rfl⟩ := hf.isLocallyClosed_iff.mp hs
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    hf' : IsLocallyClosed (Set.range f)
    t : Set Y
    ht : IsLocallyClosed t
    hs : IsLocallyClosed (Set.preimage f t)
    ⊢ IsLocallyClosed (Set.image f (Set.preimage f t))
  -/
  rw [image_preimage_eq_inter_range]
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsInducing f
    hf' : IsLocallyClosed (Set.range f)
    t : Set Y
    ht : IsLocallyClosed t
    hs : IsLocallyClosed (Set.preimage f t)
    ⊢ IsLocallyClosed (Inter.inter t (Set.range f))
  -/
  exact ht.inter hf'
  /-
    🎉 no goals
  -/


/--
A set `s` is locally closed if one of the equivalent conditions below hold
1. It is the intersection of some open set and some closed set.
2. The coborder `(closure s \ s)ᶜ` is open.
3. `s` is closed in some neighborhood of `x` for all `x ∈ s`.
4. Every `x ∈ s` has some open neighborhood `U` such that `U ∩ closure s ⊆ s`.
5. `s` is open in the closure of `s`.
-/
lemma isLocallyClosed_tfae (s : Set X) :
    List.TFAE
    [ IsLocallyClosed s,
      IsOpen (coborder s),
      ∀ x ∈ s, ∃ U ∈ 𝓝 x, IsClosed (U ↓∩ s),
      ∀ x ∈ s, ∃ U, x ∈ U ∧ IsOpen U ∧ U ∩ closure s ⊆ s,
      IsOpen (closure s ↓∩ s)] := by
  tfae_have 1 → 2 := by
    rintro ⟨U, Z, hU, hZ, rfl⟩
    have : Z ∪ (frontier (U ∩ Z))ᶜ = univ := by
      nth_rw 1 [← hZ.closure_eq]
      rw [← compl_subset_iff_union, compl_subset_compl]
      refine frontier_subset_closure.trans (closure_mono inter_subset_right)
    rw [coborder_eq_union_frontier_compl, inter_union_distrib_right, this,
      inter_univ]
    exact hU.union isClosed_frontier.isOpen_compl
  tfae_have 2 → 3
  | h, x => (⟨coborder s, h.mem_nhds <| subset_coborder ·, isClosed_preimage_val_coborder⟩)
  tfae_have 3 → 4
  | h, x, hx => by
    obtain ⟨t, ht, ht'⟩ := h x hx
    obtain ⟨U, hUt, hU, hxU⟩ := mem_nhds_iff.mp ht
    rw [isClosed_preimage_val] at ht'
    exact ⟨U, hxU, hU, (subset_inter (inter_subset_left.trans hUt) (hU.inter_closure.trans
      (closure_mono <| inter_subset_inter hUt subset_rfl))).trans ht'⟩
  tfae_have 4 → 5
  | H => by
    choose U hxU hU e using H
    refine ⟨⋃ x ∈ s, U x ‹_›, isOpen_iUnion (isOpen_iUnion <| hU ·), ext fun x ↦ ⟨?_, ?_⟩⟩
    · rintro ⟨_, ⟨⟨y, rfl⟩, ⟨_, ⟨hy, rfl⟩, hxU⟩⟩⟩
      exact e y hy ⟨hxU, x.2⟩
    · exact (subset_iUnion₂ _ _ <| hxU x ·)
  tfae_have 5 → 1
  | H => by
    convert H.isLocallyClosed.image IsInducing.subtypeVal
      (by simpa using isClosed_closure.isLocallyClosed)
    simpa using subset_closure
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    tfae_1_to_2 : IsLocallyClosed s → IsOpen (coborder s)
    tfae_2_to_3 : IsOpen (coborder s) → ∀ (x : X), Membership.mem s x → Exists fun …
    tfae_3_to_4 : (∀ (x : X), Membership.mem s x → Exists fun U => And (Membership …
    tfae_4_to_5 : (∀ (x : X), Membership.mem s x → Exists fun U => And (Membership …
    tfae_5_to_1 : IsOpen (Set.preimage Subtype.val s) → IsLocallyClosed s
    ⊢ (List.cons (IsLocallyClosed s) (List.cons (IsOpen (coborder s)) (List.cons ( …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma isLocallyClosed_iff_isOpen_coborder : IsLocallyClosed s ↔ IsOpen (coborder s) :=
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Eq ((List.cons (IsLocallyClosed s) (List.cons (IsOpen (coborder s)) (List.co …
  -/
  /-
    🎉 no goals
  -/
  (isLocallyClosed_tfae s).out 0 1
  /-
    🎉 no goals
  -/


alias ⟨IsLocallyClosed.isOpen_coborder, _⟩ := isLocallyClosed_iff_isOpen_coborder


lemma IsLocallyClosed.isOpen_preimage_val_closure (hs : IsLocallyClosed s) :
    IsOpen (closure s ↓∩ s) :=
   /-
     X : Type u_1
     inst✝ : TopologicalSpace X
     s : Set X
     hs : IsLocallyClosed s
     ⊢ Eq ((List.cons (IsLocallyClosed s) (List.cons (IsOpen (coborder s)) (List.co …
   -/
   /-
     🎉 no goals
   -/
  ((isLocallyClosed_tfae s).out 0 4).mp hs
   /-
     🎉 no goals
   -/

