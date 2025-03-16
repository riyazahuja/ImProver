/-- Preconnected sets are either contained in or disjoint to any given clopen set. -/
theorem IsPreconnected.subset_isClopen {s t : Set α} (hs : IsPreconnected s) (ht : IsClopen t)
    (hne : (s ∩ t).Nonempty) : s ⊆ t :=
                                                                                    /-
                                                                                      α : Type u
                                                                                      inst✝ : TopologicalSpace α
                                                                                      s t : Set α
                                                                                      hs : IsPreconnected s
                                                                                      ht : IsClopen t
                                                                                      hne : (Inter.inter s t).Nonempty
                                                                                      ⊢ HasSubset.Subset s (Union.union t (HasCompl.compl t))
                                                                                    -/
  hs.subset_left_of_subset_union ht.isOpen ht.compl.isOpen disjoint_compl_right (by simp) hne
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem Sigma.isConnected_iff [∀ i, TopologicalSpace (π i)] {s : Set (Σi, π i)} :
    IsConnected s ↔ ∃ i t, IsConnected t ∧ s = Sigma.mk i '' t := by
  /-
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : Set (Sigma fun i => π i)
    ⊢ Iff (IsConnected s) (Exists fun i => Exists fun t => And (IsConnected t) (Eq …
  -/
  refine ⟨fun hs => ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      s : Set (Sigma fun i => π i)
      hs : IsConnected s
      ⊢ Exists fun i => Exists fun t => And (IsConnected t) (Eq s (Set.image (Sigma. …
    -/
  · obtain ⟨⟨i, x⟩, hx⟩ := hs.nonempty
    have : s ⊆ range (Sigma.mk i) :=
      hs.isPreconnected.subset_isClopen isClopen_range_sigmaMk ⟨⟨i, x⟩, hx, x, rfl⟩
    exact ⟨i, Sigma.mk i ⁻¹' s, hs.preimage_of_isOpenMap sigma_mk_injective isOpenMap_sigmaMk this,
      (Set.image_preimage_eq_of_subset this).symm⟩
    /-
      case refine_2
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      s : Set (Sigma fun i => π i)
      ⊢ (Exists fun i => Exists fun t => And (IsConnected t) (Eq s (Set.image (Sigma …
    -/
  · rintro ⟨i, t, ht, rfl⟩
    /-
      case refine_2.intro.intro.intro
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      i : ι
      t : Set (π i)
      ht : IsConnected t
      ⊢ IsConnected (Set.image (Sigma.mk i) t)
    -/
    exact ht.image _ continuous_sigmaMk.continuousOn
    /-
      🎉 no goals
    -/


theorem Sigma.isPreconnected_iff [hι : Nonempty ι] [∀ i, TopologicalSpace (π i)]
    {s : Set (Σi, π i)} : IsPreconnected s ↔ ∃ i t, IsPreconnected t ∧ s = Sigma.mk i '' t := by
  /-
    ι : Type u_1
    π : ι → Type u_2
    hι : Nonempty ι
    inst✝ : (i : ι) → TopologicalSpace (π i)
    s : Set (Sigma fun i => π i)
    ⊢ Iff (IsPreconnected s) (Exists fun i => Exists fun t => And (IsPreconnected  …
  -/
  refine ⟨fun hs => ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      π : ι → Type u_2
      hι : Nonempty ι
      inst✝ : (i : ι) → TopologicalSpace (π i)
      s : Set (Sigma fun i => π i)
      hs : IsPreconnected s
      ⊢ Exists fun i => Exists fun t => And (IsPreconnected t) (Eq s (Set.image (Sig …
    -/
  · obtain rfl | h := s.eq_empty_or_nonempty
      /-
        case refine_1.inl
        ι : Type u_1
        π : ι → Type u_2
        hι : Nonempty ι
        inst✝ : (i : ι) → TopologicalSpace (π i)
        hs : IsPreconnected EmptyCollection.emptyCollection
        ⊢ Exists fun i => Exists fun t => And (IsPreconnected t) (Eq EmptyCollection.e …
      -/
    · exact ⟨Classical.choice hι, ∅, isPreconnected_empty, (Set.image_empty _).symm⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        ι : Type u_1
        π : ι → Type u_2
        hι : Nonempty ι
        inst✝ : (i : ι) → TopologicalSpace (π i)
        s : Set (Sigma fun i => π i)
        hs : IsPreconnected s
        h : s.Nonempty
        ⊢ Exists fun i => Exists fun t => And (IsPreconnected t) (Eq s (Set.image (Sig …
      -/
    · obtain ⟨a, t, ht, rfl⟩ := Sigma.isConnected_iff.1 ⟨h, hs⟩
      /-
        case refine_1.inr.intro.intro.intro
        ι : Type u_1
        π : ι → Type u_2
        hι : Nonempty ι
        inst✝ : (i : ι) → TopologicalSpace (π i)
        a : ι
        t : Set (π a)
        ht : IsConnected t
        hs : IsPreconnected (Set.image (Sigma.mk a) t)
        h : (Set.image (Sigma.mk a) t).Nonempty
        ⊢ Exists fun i => Exists fun t_1 => And (IsPreconnected t_1) (Eq (Set.image (S …
      -/
      exact ⟨a, t, ht.isPreconnected, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ι : Type u_1
      π : ι → Type u_2
      hι : Nonempty ι
      inst✝ : (i : ι) → TopologicalSpace (π i)
      s : Set (Sigma fun i => π i)
      ⊢ (Exists fun i => Exists fun t => And (IsPreconnected t) (Eq s (Set.image (Si …
    -/
  · rintro ⟨a, t, ht, rfl⟩
    /-
      case refine_2.intro.intro.intro
      ι : Type u_1
      π : ι → Type u_2
      hι : Nonempty ι
      inst✝ : (i : ι) → TopologicalSpace (π i)
      a : ι
      t : Set (π a)
      ht : IsPreconnected t
      ⊢ IsPreconnected (Set.image (Sigma.mk a) t)
    -/
    exact ht.image _ continuous_sigmaMk.continuousOn
    /-
      🎉 no goals
    -/


theorem Sum.isConnected_iff [TopologicalSpace β] {s : Set (α ⊕ β)} :
    IsConnected s ↔
      (∃ t, IsConnected t ∧ s = Sum.inl '' t) ∨ ∃ t, IsConnected t ∧ s = Sum.inr '' t := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set (Sum α β)
    ⊢ Iff (IsConnected s) (Or (Exists fun t => And (IsConnected t) (Eq s (Set.imag …
  -/
  refine ⟨fun hs => ?_, ?_⟩
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set (Sum α β)
      hs : IsConnected s
      ⊢ Or (Exists fun t => And (IsConnected t) (Eq s (Set.image Sum.inl t))) (Exist …
    -/
  · obtain ⟨x | x, hx⟩ := hs.nonempty
    · have h : s ⊆ range Sum.inl :=
        hs.isPreconnected.subset_isClopen isClopen_range_inl ⟨.inl x, hx, x, rfl⟩
      /-
        case refine_1.intro.inl
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        s : Set (Sum α β)
        hs : IsConnected s
        x : α
        hx : Membership.mem s (Sum.inl x)
        h : HasSubset.Subset s (Set.range Sum.inl)
        ⊢ Or (Exists fun t => And (IsConnected t) (Eq s (Set.image Sum.inl t))) (Exist …
      -/
      refine Or.inl ⟨Sum.inl ⁻¹' s, ?_, ?_⟩
        /-
          case refine_1.intro.inl.refine_1
          α : Type u
          β : Type v
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalSpace β
          s : Set (Sum α β)
          hs : IsConnected s
          x : α
          hx : Membership.mem s (Sum.inl x)
          h : HasSubset.Subset s (Set.range Sum.inl)
          ⊢ IsConnected (Set.preimage Sum.inl s)
        -/
      · exact hs.preimage_of_isOpenMap Sum.inl_injective isOpenMap_inl h
        /-
          🎉 no goals
        -/
        /-
          case refine_1.intro.inl.refine_2
          α : Type u
          β : Type v
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalSpace β
          s : Set (Sum α β)
          hs : IsConnected s
          x : α
          hx : Membership.mem s (Sum.inl x)
          h : HasSubset.Subset s (Set.range Sum.inl)
          ⊢ Eq s (Set.image Sum.inl (Set.preimage Sum.inl s))
        -/
      · exact (image_preimage_eq_of_subset h).symm
        /-
          🎉 no goals
        -/
    · have h : s ⊆ range Sum.inr :=
        hs.isPreconnected.subset_isClopen isClopen_range_inr ⟨.inr x, hx, x, rfl⟩
      /-
        case refine_1.intro.inr
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        s : Set (Sum α β)
        hs : IsConnected s
        x : β
        hx : Membership.mem s (Sum.inr x)
        h : HasSubset.Subset s (Set.range Sum.inr)
        ⊢ Or (Exists fun t => And (IsConnected t) (Eq s (Set.image Sum.inl t))) (Exist …
      -/
      refine Or.inr ⟨Sum.inr ⁻¹' s, ?_, ?_⟩
        /-
          case refine_1.intro.inr.refine_1
          α : Type u
          β : Type v
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalSpace β
          s : Set (Sum α β)
          hs : IsConnected s
          x : β
          hx : Membership.mem s (Sum.inr x)
          h : HasSubset.Subset s (Set.range Sum.inr)
          ⊢ IsConnected (Set.preimage Sum.inr s)
        -/
      · exact hs.preimage_of_isOpenMap Sum.inr_injective isOpenMap_inr h
        /-
          🎉 no goals
        -/
        /-
          case refine_1.intro.inr.refine_2
          α : Type u
          β : Type v
          inst✝¹ : TopologicalSpace α
          inst✝ : TopologicalSpace β
          s : Set (Sum α β)
          hs : IsConnected s
          x : β
          hx : Membership.mem s (Sum.inr x)
          h : HasSubset.Subset s (Set.range Sum.inr)
          ⊢ Eq s (Set.image Sum.inr (Set.preimage Sum.inr s))
        -/
      · exact (image_preimage_eq_of_subset h).symm
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set (Sum α β)
      ⊢ Or (Exists fun t => And (IsConnected t) (Eq s (Set.image Sum.inl t))) (Exist …
    -/
  · rintro (⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩)
      /-
        case refine_2.inl.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        t : Set α
        ht : IsConnected t
        ⊢ IsConnected (Set.image Sum.inl t)
      -/
    · exact ht.image _ continuous_inl.continuousOn
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        t : Set β
        ht : IsConnected t
        ⊢ IsConnected (Set.image Sum.inr t)
      -/
    · exact ht.image _ continuous_inr.continuousOn
      /-
        🎉 no goals
      -/


theorem Sum.isPreconnected_iff [TopologicalSpace β] {s : Set (α ⊕ β)} :
    IsPreconnected s ↔
      (∃ t, IsPreconnected t ∧ s = Sum.inl '' t) ∨ ∃ t, IsPreconnected t ∧ s = Sum.inr '' t := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    s : Set (Sum α β)
    ⊢ Iff (IsPreconnected s) (Or (Exists fun t => And (IsPreconnected t) (Eq s (Se …
  -/
  refine ⟨fun hs => ?_, ?_⟩
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set (Sum α β)
      hs : IsPreconnected s
      ⊢ Or (Exists fun t => And (IsPreconnected t) (Eq s (Set.image Sum.inl t))) (Ex …
    -/
  · obtain rfl | h := s.eq_empty_or_nonempty
      /-
        case refine_1.inl
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        hs : IsPreconnected EmptyCollection.emptyCollection
        ⊢ Or (Exists fun t => And (IsPreconnected t) (Eq EmptyCollection.emptyCollecti …
      -/
    · exact Or.inl ⟨∅, isPreconnected_empty, (Set.image_empty _).symm⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set (Sum α β)
      hs : IsPreconnected s
      h : s.Nonempty
      ⊢ Or (Exists fun t => And (IsPreconnected t) (Eq s (Set.image Sum.inl t))) (Ex …
    -/
    obtain ⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩ := Sum.isConnected_iff.1 ⟨h, hs⟩
      /-
        case refine_1.inr.inl.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        t : Set α
        ht : IsConnected t
        hs : IsPreconnected (Set.image Sum.inl t)
        h : (Set.image Sum.inl t).Nonempty
        ⊢ Or (Exists fun t_1 => And (IsPreconnected t_1) (Eq (Set.image Sum.inl t) (Se …
      -/
    · exact Or.inl ⟨t, ht.isPreconnected, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.inr.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        t : Set β
        ht : IsConnected t
        hs : IsPreconnected (Set.image Sum.inr t)
        h : (Set.image Sum.inr t).Nonempty
        ⊢ Or (Exists fun t_1 => And (IsPreconnected t_1) (Eq (Set.image Sum.inr t) (Se …
      -/
    · exact Or.inr ⟨t, ht.isPreconnected, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      s : Set (Sum α β)
      ⊢ Or (Exists fun t => And (IsPreconnected t) (Eq s (Set.image Sum.inl t))) (Ex …
    -/
  · rintro (⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩)
      /-
        case refine_2.inl.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        t : Set α
        ht : IsPreconnected t
        ⊢ IsPreconnected (Set.image Sum.inl t)
      -/
    · exact ht.image _ continuous_inl.continuousOn
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.intro
        α : Type u
        β : Type v
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        t : Set β
        ht : IsPreconnected t
        ⊢ IsPreconnected (Set.image Sum.inr t)
      -/
    · exact ht.image _ continuous_inr.continuousOn
      /-
        🎉 no goals
      -/


/-- A continuous map from a connected space to a disjoint union `Σ i, π i` can be lifted to one of
the components `π i`. See also `ContinuousMap.exists_lift_sigma` for a version with bundled
`ContinuousMap`s. -/
theorem Continuous.exists_lift_sigma [ConnectedSpace α] [∀ i, TopologicalSpace (π i)]
    {f : α → Σ i, π i} (hf : Continuous f) :
    ∃ (i : ι) (g : α → π i), Continuous g ∧ f = Sigma.mk i ∘ g := by
  obtain ⟨i, hi⟩ : ∃ i, range f ⊆ range (.mk i) := by
    rcases Sigma.isConnected_iff.1 (isConnected_range hf) with ⟨i, s, -, hs⟩
    exact ⟨i, hs.trans_subset (image_subset_range _ _)⟩
  /-
    case intro
    α : Type u
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : ConnectedSpace α
    inst✝ : (i : ι) → TopologicalSpace (π i)
    f : α → Sigma fun i => π i
    hf : Continuous f
    i : ι
    hi : HasSubset.Subset (Set.range f) (Set.range (Sigma.mk i))
    ⊢ Exists fun i => Exists fun g => And (Continuous g) (Eq f (Function.comp (Sig …
  -/
  rcases range_subset_range_iff_exists_comp.1 hi with ⟨g, rfl⟩
  /-
    case intro.intro
    α : Type u
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : ConnectedSpace α
    inst✝ : (i : ι) → TopologicalSpace (π i)
    i : ι
    g : α → π i
    hf : Continuous (Function.comp (Sigma.mk i) g)
    hi : HasSubset.Subset (Set.range (Function.comp (Sigma.mk i) g)) (Set.range (S …
    ⊢ Exists fun i_1 => Exists fun g_1 => And (Continuous g_1) (Eq (Function.comp  …
  -/
  refine ⟨i, g, ?_, rfl⟩
  /-
    case intro.intro
    α : Type u
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : ConnectedSpace α
    inst✝ : (i : ι) → TopologicalSpace (π i)
    i : ι
    g : α → π i
    hf : Continuous (Function.comp (Sigma.mk i) g)
    hi : HasSubset.Subset (Set.range (Function.comp (Sigma.mk i) g)) (Set.range (S …
    ⊢ Continuous g
  -/
  rwa [← IsEmbedding.sigmaMk.continuous_iff] at hf
  /-
    🎉 no goals
  -/


theorem nonempty_inter [PreconnectedSpace α] {s t : Set α} :
    IsOpen s → IsOpen t → s ∪ t = univ → s.Nonempty → t.Nonempty → (s ∩ t).Nonempty := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s t : Set α
    ⊢ IsOpen s → IsOpen t → Eq (Union.union s t) Set.univ → s.Nonempty → t.Nonempt …
  -/
  simpa only [univ_inter, univ_subset_iff] using @PreconnectedSpace.isPreconnected_univ α _ _ s t
  /-
    🎉 no goals
  -/


theorem isClopen_iff [PreconnectedSpace α] {s : Set α} : IsClopen s ↔ s = ∅ ∨ s = univ :=
  ⟨fun hs =>
    by_contradiction fun h =>
      have h1 : s ≠ ∅ ∧ sᶜ ≠ ∅ :=
        ⟨mt Or.inl h,
                                      /-
                                        α : Type u
                                        inst✝¹ : TopologicalSpace α
                                        inst✝ : PreconnectedSpace α
                                        s : Set α
                                        hs : IsClopen s
                                        h : Not (Or (Eq s EmptyCollection.emptyCollection) (Eq s Set.univ))
                                        h2 : Eq (HasCompl.compl s) EmptyCollection.emptyCollection
                                        ⊢ Eq s Set.univ
                                      -/
          mt (fun h2 => Or.inr <| (by rw [← compl_compl s, h2, compl_empty] : s = univ)) h⟩
                                      /-
                                        🎉 no goals
                                      -/
      let ⟨_, h2, h3⟩ :=
        nonempty_inter hs.2 hs.1.isOpen_compl (union_compl_self s) (nonempty_iff_ne_empty.2 h1.1)
          (nonempty_iff_ne_empty.2 h1.2)
      h3 h2,
       /-
         α : Type u
         inst✝¹ : TopologicalSpace α
         inst✝ : PreconnectedSpace α
         s : Set α
         ⊢ Or (Eq s EmptyCollection.emptyCollection) (Eq s Set.univ) → IsClopen s
       -/
    by rintro (rfl | rfl) <;> [exact isClopen_empty; exact isClopen_univ]⟩
       /-
         🎉 no goals
       -/


theorem IsClopen.eq_univ [PreconnectedSpace α] {s : Set α} (h' : IsClopen s) (h : s.Nonempty) :
    s = univ :=
  (isClopen_iff.mp h').resolve_left h.ne_empty


open Set.Notation in
lemma isClopen_preimage_val {X : Type*} [TopologicalSpace X] {u v : Set X}
    (hu : IsOpen u) (huv : Disjoint (frontier u) v) : IsClopen (v ↓∩ u) := by
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsOpen u
    huv : Disjoint (frontier u) v
    ⊢ IsClopen (Set.preimage Subtype.val u)
  -/
  refine ⟨?_, isOpen_induced hu (f := Subtype.val)⟩
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsOpen u
    huv : Disjoint (frontier u) v
    ⊢ IsClosed (Set.preimage Subtype.val u)
  -/
  refine isClosed_induced_iff.mpr ⟨closure u, isClosed_closure, ?_⟩
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsOpen u
    huv : Disjoint (frontier u) v
    ⊢ Eq (Set.preimage Subtype.val (closure u)) (Set.preimage Subtype.val u)
  -/
  apply image_val_injective
  /-
    case a
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsOpen u
    huv : Disjoint (frontier u) v
    ⊢ Eq (Set.image Subtype.val (Set.preimage Subtype.val (closure u))) (Set.image …
  -/
  simp only [Subtype.image_preimage_coe]
  rw [closure_eq_self_union_frontier, inter_union_distrib_left, inter_comm _ (frontier u),
    huv.inter_eq, union_empty]


/-- In a preconnected space, any disjoint family of non-empty clopen subsets has at most one
element. -/
lemma subsingleton_of_disjoint_isClopen
    (h_clopen : ∀ i, IsClopen (s i)) :
    Subsingleton ι := by
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_clopen : ∀ (i : ι), IsClopen (s i)
    ⊢ Subsingleton ι
  -/
  replace h_nonempty : ∀ i, s i ≠ ∅ := by intro i; rw [← nonempty_iff_ne_empty]; exact h_nonempty i
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_clopen : ∀ (i : ι), IsClopen (s i)
    h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
    ⊢ Subsingleton ι
  -/
  rw [← not_nontrivial_iff_subsingleton]
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_clopen : ∀ (i : ι), IsClopen (s i)
    h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
    ⊢ Not (Nontrivial ι)
  -/
  by_contra contra
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_clopen : ∀ (i : ι), IsClopen (s i)
    h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
    contra : Nontrivial ι
    ⊢ False
  -/
  obtain ⟨i, j, h_ne⟩ := contra
  replace h_ne : s i ∩ s j = ∅ := by
    simpa only [← bot_eq_empty, eq_bot_iff, ← inf_eq_inter, ← disjoint_iff_inf_le] using h_disj h_ne
  /-
    case mk.intro.intro
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_clopen : ∀ (i : ι), IsClopen (s i)
    h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
    i j : ι
    h_ne : Eq (Inter.inter (s i) (s j)) EmptyCollection.emptyCollection
    ⊢ False
  -/
  cases' isClopen_iff.mp (h_clopen i) with hi hi
    /-
      case mk.intro.intro.inl
      α : Type u
      ι : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : PreconnectedSpace α
      s : ι → Set α
      h_disj : Pairwise (Function.onFun Disjoint s)
      h_clopen : ∀ (i : ι), IsClopen (s i)
      h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
      i j : ι
      h_ne : Eq (Inter.inter (s i) (s j)) EmptyCollection.emptyCollection
      hi : Eq (s i) EmptyCollection.emptyCollection
      ⊢ False
    -/
  · exact h_nonempty i hi
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.intro.inr
      α : Type u
      ι : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : PreconnectedSpace α
      s : ι → Set α
      h_disj : Pairwise (Function.onFun Disjoint s)
      h_clopen : ∀ (i : ι), IsClopen (s i)
      h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
      i j : ι
      h_ne : Eq (Inter.inter (s i) (s j)) EmptyCollection.emptyCollection
      hi : Eq (s i) Set.univ
      ⊢ False
    -/
  · rw [hi, univ_inter] at h_ne
    /-
      case mk.intro.intro.inr
      α : Type u
      ι : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : PreconnectedSpace α
      s : ι → Set α
      h_disj : Pairwise (Function.onFun Disjoint s)
      h_clopen : ∀ (i : ι), IsClopen (s i)
      h_nonempty : ∀ (i : ι), Ne (s i) EmptyCollection.emptyCollection
      i j : ι
      h_ne : Eq (s j) EmptyCollection.emptyCollection
      hi : Eq (s i) Set.univ
      ⊢ False
    -/
    exact h_nonempty j h_ne
    /-
      🎉 no goals
    -/


/-- In a preconnected space, any disjoint cover by non-empty open subsets has at most one
element. -/
lemma subsingleton_of_disjoint_isOpen_iUnion_eq_univ
    (h_open : ∀ i, IsOpen (s i)) (h_Union : ⋃ i, s i = univ) :
    Subsingleton ι := by
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_open : ∀ (i : ι), IsOpen (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    ⊢ Subsingleton ι
  -/
  refine subsingleton_of_disjoint_isClopen h_nonempty h_disj (fun i ↦ ⟨?_, h_open i⟩)
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_open : ∀ (i : ι), IsOpen (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    i : ι
    ⊢ IsClosed (s i)
  -/
  rw [← isOpen_compl_iff, compl_eq_univ_diff, ← h_Union, iUnion_diff]
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_open : ∀ (i : ι), IsOpen (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    i : ι
    ⊢ IsOpen (Set.iUnion fun i_1 => SDiff.sdiff (s i_1) (s i))
  -/
  refine isOpen_iUnion (fun j ↦ ?_)
  /-
    α : Type u
    ι : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    h_open : ∀ (i : ι), IsOpen (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    i j : ι
    ⊢ IsOpen (SDiff.sdiff (s j) (s i))
  -/
  rcases eq_or_ne i j with rfl | h_ne
    /-
      case inl
      α : Type u
      ι : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : PreconnectedSpace α
      s : ι → Set α
      h_nonempty : ∀ (i : ι), (s i).Nonempty
      h_disj : Pairwise (Function.onFun Disjoint s)
      h_open : ∀ (i : ι), IsOpen (s i)
      h_Union : Eq (Set.iUnion fun i => s i) Set.univ
      i : ι
      ⊢ IsOpen (SDiff.sdiff (s i) (s i))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      ι : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : PreconnectedSpace α
      s : ι → Set α
      h_nonempty : ∀ (i : ι), (s i).Nonempty
      h_disj : Pairwise (Function.onFun Disjoint s)
      h_open : ∀ (i : ι), IsOpen (s i)
      h_Union : Eq (Set.iUnion fun i => s i) Set.univ
      i j : ι
      h_ne : Ne i j
      ⊢ IsOpen (SDiff.sdiff (s j) (s i))
    -/
  · simpa only [(h_disj h_ne.symm).sdiff_eq_left] using h_open j
    /-
      🎉 no goals
    -/


/-- In a preconnected space, any finite disjoint cover by non-empty closed subsets has at most one
element. -/
lemma subsingleton_of_disjoint_isClosed_iUnion_eq_univ [Finite ι]
    (h_closed : ∀ i, IsClosed (s i)) (h_Union : ⋃ i, s i = univ) :
    Subsingleton ι := by
  /-
    α : Type u
    ι : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    inst✝ : Finite ι
    h_closed : ∀ (i : ι), IsClosed (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    ⊢ Subsingleton ι
  -/
  refine subsingleton_of_disjoint_isClopen h_nonempty h_disj (fun i ↦ ⟨h_closed i, ?_⟩)
  /-
    α : Type u
    ι : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    inst✝ : Finite ι
    h_closed : ∀ (i : ι), IsClosed (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    i : ι
    ⊢ IsOpen (s i)
  -/
  rw [← isClosed_compl_iff, compl_eq_univ_diff, ← h_Union, iUnion_diff]
  /-
    α : Type u
    ι : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    inst✝ : Finite ι
    h_closed : ∀ (i : ι), IsClosed (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    i : ι
    ⊢ IsClosed (Set.iUnion fun i_1 => SDiff.sdiff (s i_1) (s i))
  -/
  refine isClosed_iUnion_of_finite (fun j ↦ ?_)
  /-
    α : Type u
    ι : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : PreconnectedSpace α
    s : ι → Set α
    h_nonempty : ∀ (i : ι), (s i).Nonempty
    h_disj : Pairwise (Function.onFun Disjoint s)
    inst✝ : Finite ι
    h_closed : ∀ (i : ι), IsClosed (s i)
    h_Union : Eq (Set.iUnion fun i => s i) Set.univ
    i j : ι
    ⊢ IsClosed (SDiff.sdiff (s j) (s i))
  -/
  rcases eq_or_ne i j with rfl | h_ne
    /-
      case inl
      α : Type u
      ι : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : PreconnectedSpace α
      s : ι → Set α
      h_nonempty : ∀ (i : ι), (s i).Nonempty
      h_disj : Pairwise (Function.onFun Disjoint s)
      inst✝ : Finite ι
      h_closed : ∀ (i : ι), IsClosed (s i)
      h_Union : Eq (Set.iUnion fun i => s i) Set.univ
      i : ι
      ⊢ IsClosed (SDiff.sdiff (s i) (s i))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      ι : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : PreconnectedSpace α
      s : ι → Set α
      h_nonempty : ∀ (i : ι), (s i).Nonempty
      h_disj : Pairwise (Function.onFun Disjoint s)
      inst✝ : Finite ι
      h_closed : ∀ (i : ι), IsClosed (s i)
      h_Union : Eq (Set.iUnion fun i => s i) Set.univ
      i j : ι
      h_ne : Ne i j
      ⊢ IsClosed (SDiff.sdiff (s j) (s i))
    -/
  · simpa only [(h_disj h_ne.symm).sdiff_eq_left] using h_closed j
    /-
      🎉 no goals
    -/


theorem frontier_eq_empty_iff [PreconnectedSpace α] {s : Set α} :
    frontier s = ∅ ↔ s = ∅ ∨ s = univ :=
  isClopen_iff_frontier_eq_empty.symm.trans isClopen_iff


theorem nonempty_frontier_iff [PreconnectedSpace α] {s : Set α} :
    (frontier s).Nonempty ↔ s.Nonempty ∧ s ≠ univ := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    s : Set α
    ⊢ Iff (frontier s).Nonempty (And s.Nonempty (Ne s Set.univ))
  -/
  simp only [nonempty_iff_ne_empty, Ne, frontier_eq_empty_iff, not_or]
  /-
    🎉 no goals
  -/


/-- In a preconnected space, given a transitive relation `P`, if `P x y` and `P y x` are true
for `y` close enough to `x`, then `P x y` holds for all `x, y`. This is a version of the fact
that, if an equivalence relation has open classes, then it has a single equivalence class. -/
lemma PreconnectedSpace.induction₂' [PreconnectedSpace α] (P : α → α → Prop)
    (h : ∀ x, ∀ᶠ y in 𝓝 x, P x y ∧ P y x) (h' : Transitive P) (x y : α) :
    P x y := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => And (P x y) (P y x)) (nhds x)
    h' : Transitive P
    x y : α
    ⊢ P x y
  -/
  let u := {z | P x z}
  have A : IsClosed u := by
    apply isClosed_iff_nhds.2 (fun z hz ↦ ?_)
    rcases hz _ (h z) with ⟨t, ht, h't⟩
    exact h' h't ht.2
  have B : IsOpen u := by
    apply isOpen_iff_mem_nhds.2 (fun z hz ↦ ?_)
    filter_upwards [h z] with t ht
    exact h' hz ht.1
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => And (P x y) (P y x)) (nhds x)
    h' : Transitive P
    x y : α
    u : Set α := setOf fun z => P x z
    A : IsClosed u
    B : IsOpen u
    ⊢ P x y
  -/
  have C : u.Nonempty := ⟨x, (mem_of_mem_nhds (h x)).1⟩
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => And (P x y) (P y x)) (nhds x)
    h' : Transitive P
    x y : α
    u : Set α := setOf fun z => P x z
    A : IsClosed u
    B : IsOpen u
    C : u.Nonempty
    ⊢ P x y
  -/
  have D : u = Set.univ := IsClopen.eq_univ ⟨A, B⟩ C
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => And (P x y) (P y x)) (nhds x)
    h' : Transitive P
    x y : α
    u : Set α := setOf fun z => P x z
    A : IsClosed u
    B : IsOpen u
    C : u.Nonempty
    D : Eq u Set.univ
    ⊢ P x y
  -/
  show y ∈ u
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => And (P x y) (P y x)) (nhds x)
    h' : Transitive P
    x y : α
    u : Set α := setOf fun z => P x z
    A : IsClosed u
    B : IsOpen u
    C : u.Nonempty
    D : Eq u Set.univ
    ⊢ Membership.mem u y
  -/
  simp [D]
  /-
    🎉 no goals
  -/


/-- In a preconnected space, if a symmetric transitive relation `P x y` is true for `y` close
enough to `x`, then it holds for all `x, y`. This is a version of the fact that, if an equivalence
relation has open classes, then it has a single equivalence class. -/
lemma PreconnectedSpace.induction₂ [PreconnectedSpace α] (P : α → α → Prop)
    (h : ∀ x, ∀ᶠ y in 𝓝 x, P x y) (h' : Transitive P) (h'' : Symmetric P) (x y : α) :
    P x y := by
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => P x y) (nhds x)
    h' : Transitive P
    h'' : Symmetric P
    x y : α
    ⊢ P x y
  -/
  refine PreconnectedSpace.induction₂' P (fun z ↦ ?_) h' x y
  /-
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => P x y) (nhds x)
    h' : Transitive P
    h'' : Symmetric P
    x y z : α
    ⊢ Filter.Eventually (fun y => And (P z y) (P y z)) (nhds z)
  -/
  filter_upwards [h z] with a ha
  /-
    case h
    α : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : PreconnectedSpace α
    P : α → α → Prop
    h : ∀ (x : α), Filter.Eventually (fun y => P x y) (nhds x)
    h' : Transitive P
    h'' : Symmetric P
    x y z a : α
    ha : P z a
    ⊢ And (P z a) (P a z)
  -/
  exact ⟨ha, h'' ha⟩
  /-
    🎉 no goals
  -/


/-- In a preconnected set, given a transitive relation `P`, if `P x y` and `P y x` are true
for `y` close enough to `x`, then `P x y` holds for all `x, y`. This is a version of the fact
that, if an equivalence relation has open classes, then it has a single equivalence class. -/
lemma IsPreconnected.induction₂' {s : Set α} (hs : IsPreconnected s) (P : α → α → Prop)
    (h : ∀ x ∈ s, ∀ᶠ y in 𝓝[s] x, P x y ∧ P y x)
    (h' : ∀ x y z, x ∈ s → y ∈ s → z ∈ s → P x y → P y z → P x z)
    {x y : α} (hx : x ∈ s) (hy : y ∈ s) : P x y := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ P x y
  -/
  let Q : s → s → Prop := fun a b ↦ P a b
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
    ⊢ P x y
  -/
  show Q ⟨x, hx⟩ ⟨y, hy⟩
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
    ⊢ Q ⟨x, hx⟩ ⟨y, hy⟩
  -/
  have : PreconnectedSpace s := Subtype.preconnectedSpace hs
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
    this : PreconnectedSpace ↑s
    ⊢ Q ⟨x, hx⟩ ⟨y, hy⟩
  -/
  apply PreconnectedSpace.induction₂'
    /-
      case h
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsPreconnected s
      P : α → α → Prop
      h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
      h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
      x y : α
      hx : Membership.mem s x
      hy : Membership.mem s y
      Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
      this : PreconnectedSpace ↑s
      ⊢ ∀ (x : Subtype fun x => Membership.mem s x), Filter.Eventually (fun y => And …
    -/
  · rintro ⟨x, hx⟩
    /-
      case h.mk
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsPreconnected s
      P : α → α → Prop
      h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
      h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
      x✝ y : α
      hx✝ : Membership.mem s x✝
      hy : Membership.mem s y
      Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
      this : PreconnectedSpace ↑s
      x : α
      hx : Membership.mem s x
      ⊢ Filter.Eventually (fun y => And (Q ⟨x, hx⟩ y) (Q y ⟨x, hx⟩)) (nhds ⟨x, hx⟩)
    -/
    have Z := h x hx
    /-
      case h.mk
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsPreconnected s
      P : α → α → Prop
      h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
      h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
      x✝ y : α
      hx✝ : Membership.mem s x✝
      hy : Membership.mem s y
      Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
      this : PreconnectedSpace ↑s
      x : α
      hx : Membership.mem s x
      Z : Filter.Eventually (fun y => And (P x y) (P y x)) (nhdsWithin x s)
      ⊢ Filter.Eventually (fun y => And (Q ⟨x, hx⟩ y) (Q y ⟨x, hx⟩)) (nhds ⟨x, hx⟩)
    -/
    rwa [nhdsWithin_eq_map_subtype_coe] at Z
    /-
      🎉 no goals
    -/
    /-
      case h'
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsPreconnected s
      P : α → α → Prop
      h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
      h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
      x y : α
      hx : Membership.mem s x
      hy : Membership.mem s y
      Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
      this : PreconnectedSpace ↑s
      ⊢ Transitive Q
    -/
  · rintro ⟨a, ha⟩ ⟨b, hb⟩ ⟨c, hc⟩ hab hbc
    /-
      case h'.mk.mk.mk
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsPreconnected s
      P : α → α → Prop
      h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => And (P x y) (P …
      h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
      x y : α
      hx : Membership.mem s x
      hy : Membership.mem s y
      Q : ↑s → ↑s → Prop := fun a b => P ↑a ↑b
      this : PreconnectedSpace ↑s
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem s c
      hab : Q ⟨a, ha⟩ ⟨b, hb⟩
      hbc : Q ⟨b, hb⟩ ⟨c, hc⟩
      ⊢ Q ⟨a, ha⟩ ⟨c, hc⟩
    -/
    exact h' a b c ha hb hc hab hbc
    /-
      🎉 no goals
    -/


/-- In a preconnected set, if a symmetric transitive relation `P x y` is true for `y` close
enough to `x`, then it holds for all `x, y`. This is a version of the fact that, if an equivalence
relation has open classes, then it has a single equivalence class. -/
lemma IsPreconnected.induction₂ {s : Set α} (hs : IsPreconnected s) (P : α → α → Prop)
    (h : ∀ x ∈ s, ∀ᶠ y in 𝓝[s] x, P x y)
    (h' : ∀ x y z, x ∈ s → y ∈ s → z ∈ s → P x y → P y z → P x z)
    (h'' : ∀ x y, x ∈ s → y ∈ s → P x y → P y x)
    {x y : α} (hx : x ∈ s) (hy : y ∈ s) : P x y := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => P x y) (nhdsWi …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    h'' : ∀ (x y : α), Membership.mem s x → Membership.mem s y → P x y → P y x
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ P x y
  -/
  apply hs.induction₂' P (fun z hz ↦ ?_) h' hx hy
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => P x y) (nhdsWi …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    h'' : ∀ (x y : α), Membership.mem s x → Membership.mem s y → P x y → P y x
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    z : α
    hz : Membership.mem s z
    ⊢ Filter.Eventually (fun y => And (P z y) (P y z)) (nhdsWithin z s)
  -/
  filter_upwards [h z hz, self_mem_nhdsWithin] with a ha h'a
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsPreconnected s
    P : α → α → Prop
    h : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun y => P x y) (nhdsWi …
    h' : ∀ (x y z : α), Membership.mem s x → Membership.mem s y → Membership.mem s …
    h'' : ∀ (x y : α), Membership.mem s x → Membership.mem s y → P x y → P y x
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    z : α
    hz : Membership.mem s z
    a : α
    ha : P z a
    h'a : Membership.mem s a
    ⊢ And (P z a) (P a z)
  -/
  exact ⟨ha, h'' z a hz h'a ha⟩
  /-
    🎉 no goals
  -/


/-- A set `s` is preconnected if and only if for every cover by two open sets that are disjoint on
`s`, it is contained in one of the two covering sets. -/
theorem isPreconnected_iff_subset_of_disjoint {s : Set α} :
    IsPreconnected s ↔
      ∀ u v, IsOpen u → IsOpen v → s ⊆ u ∪ v → s ∩ (u ∩ v) = ∅ → s ⊆ u ∨ s ⊆ v := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    ⊢ Iff (IsPreconnected s) (∀ (u v : Set α), IsOpen u → IsOpen v → HasSubset.Sub …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : IsPreconnected s
      ⊢ ∀ (u v : Set α), IsOpen u → IsOpen v → HasSubset.Subset s (Union.union u v)  …
    -/
  · intro u v hu hv hs huv
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : IsPreconnected s
      u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
    specialize h u v hu hv hs
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
    contrapose! huv
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      huv : And (Not (HasSubset.Subset s u)) (Not (HasSubset.Subset s v))
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    simp only [not_subset] at huv
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      huv : And (Exists fun a => And (Membership.mem s a) (Not (Membership.mem u a)) …
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    rcases huv with ⟨⟨x, hxs, hxu⟩, ⟨y, hys, hyv⟩⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      x : α
      hxs : Membership.mem s x
      hxu : Not (Membership.mem u x)
      y : α
      hys : Membership.mem s y
      hyv : Not (Membership.mem v y)
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    have hxv : x ∈ v := or_iff_not_imp_left.mp (hs hxs) hxu
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      x : α
      hxs : Membership.mem s x
      hxu : Not (Membership.mem u x)
      y : α
      hys : Membership.mem s y
      hyv : Not (Membership.mem v y)
      hxv : Membership.mem v x
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    have hyu : y ∈ u := or_iff_not_imp_right.mp (hs hys) hyv
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      x : α
      hxs : Membership.mem s x
      hxu : Not (Membership.mem u x)
      y : α
      hys : Membership.mem s y
      hyv : Not (Membership.mem v y)
      hxv : Membership.mem v x
      hyu : Membership.mem u y
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    exact h ⟨y, hys, hyu⟩ ⟨x, hxs, hxv⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsOpen u → IsOpen v → HasSubset.Subset s (Union.union u v …
      ⊢ IsPreconnected s
    -/
  · intro u v hu hv hs hsu hsv
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsOpen u → IsOpen v → HasSubset.Subset s (Union.union u v …
      u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    by_contra H
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsOpen u → IsOpen v → HasSubset.Subset s (Union.union u v …
      u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      H : Not (Inter.inter s (Inter.inter u v)).Nonempty
      ⊢ False
    -/
    specialize h u v hu hv hs (Set.not_nonempty_iff_eq_empty.mp H)
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      H : Not (Inter.inter s (Inter.inter u v)).Nonempty
      h : Or (HasSubset.Subset s u) (HasSubset.Subset s v)
      ⊢ False
    -/
    apply H
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsOpen u
      hv : IsOpen v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      H : Not (Inter.inter s (Inter.inter u v)).Nonempty
      h : Or (HasSubset.Subset s u) (HasSubset.Subset s v)
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    cases' h with h h
      /-
        case mpr.inl
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        hs : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hsv : (Inter.inter s v).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s u
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
    · rcases hsv with ⟨x, hxs, hxv⟩
      /-
        case mpr.inl.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        hs : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s u
        x : α
        hxs : Membership.mem s x
        hxv : Membership.mem v x
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
      exact ⟨x, hxs, ⟨h hxs, hxv⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        hs : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hsv : (Inter.inter s v).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s v
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
    · rcases hsu with ⟨x, hxs, hxu⟩
      /-
        case mpr.inr.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsOpen u
        hv : IsOpen v
        hs : HasSubset.Subset s (Union.union u v)
        hsv : (Inter.inter s v).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s v
        x : α
        hxs : Membership.mem s x
        hxu : Membership.mem u x
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
      exact ⟨x, hxs, ⟨hxu, h hxs⟩⟩
      /-
        🎉 no goals
      -/


/-- A set `s` is connected if and only if
for every cover by a finite collection of open sets that are pairwise disjoint on `s`,
it is contained in one of the members of the collection. -/
theorem isConnected_iff_sUnion_disjoint_open {s : Set α} :
    IsConnected s ↔
      ∀ U : Finset (Set α), (∀ u v : Set α, u ∈ U → v ∈ U → (s ∩ (u ∩ v)).Nonempty → u = v) →
        (∀ u ∈ U, IsOpen u) → (s ⊆ ⋃₀ ↑U) → ∃ u ∈ U, s ⊆ u := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    ⊢ Iff (IsConnected s) (∀ (U : Finset (Set α)), (∀ (u v : Set α), Membership.me …
  -/
  rw [IsConnected, isPreconnected_iff_subset_of_disjoint]
  classical
  refine ⟨fun ⟨hne, h⟩ U hU hUo hsU => ?_, fun h => ⟨?_, fun u v hu hv hs hsuv => ?_⟩⟩
  · induction U using Finset.induction_on with
    | empty => exact absurd (by simpa using hsU) hne.not_subset_empty
    | @insert u U uU IH =>
      simp only [← forall_cond_comm, Finset.forall_mem_insert, Finset.exists_mem_insert,
        Finset.coe_insert, sUnion_insert, implies_true, true_and] at *
      refine (h _ hUo.1 (⋃₀ ↑U) (isOpen_sUnion hUo.2) hsU ?_).imp_right ?_
      · refine subset_empty_iff.1 fun x ⟨hxs, hxu, v, hvU, hxv⟩ => ?_
        exact ne_of_mem_of_not_mem hvU uU (hU.1 v hvU ⟨x, hxs, hxu, hxv⟩).symm
      · exact IH (fun u hu => (hU.2 u hu).2) hUo.2
  · simpa [subset_empty_iff, nonempty_iff_ne_empty] using h ∅
  · rw [← not_nonempty_iff_eq_empty] at hsuv
    have := hsuv; rw [inter_comm u] at this
    simpa [*, or_imp, forall_and] using h {u, v}

-- Porting note: `IsPreconnected.subset_isClopen` moved up from here


/-- Preconnected sets are either contained in or disjoint to any given clopen set. -/
theorem disjoint_or_subset_of_isClopen {s t : Set α} (hs : IsPreconnected s) (ht : IsClopen t) :
    Disjoint s t ∨ s ⊆ t :=
  (disjoint_or_nonempty_inter s t).imp_right <| hs.subset_isClopen ht


/-- A set `s` is preconnected if and only if
for every cover by two closed sets that are disjoint on `s`,
it is contained in one of the two covering sets. -/
theorem isPreconnected_iff_subset_of_disjoint_closed :
    IsPreconnected s ↔
      ∀ u v, IsClosed u → IsClosed v → s ⊆ u ∪ v → s ∩ (u ∩ v) = ∅ → s ⊆ u ∨ s ⊆ v := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    ⊢ Iff (IsPreconnected s) (∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : IsPreconnected s
      ⊢ ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union u …
    -/
  · intro u v hu hv hs huv
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : IsPreconnected s
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
    rw [isPreconnected_closed_iff] at h
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.uni …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
    specialize h u v hu hv hs
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
    contrapose! huv
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      huv : And (Not (HasSubset.Subset s u)) (Not (HasSubset.Subset s v))
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    simp only [not_subset] at huv
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      huv : And (Exists fun a => And (Membership.mem s a) (Not (Membership.mem u a)) …
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    rcases huv with ⟨⟨x, hxs, hxu⟩, ⟨y, hys, hyv⟩⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      x : α
      hxs : Membership.mem s x
      hxu : Not (Membership.mem u x)
      y : α
      hys : Membership.mem s y
      hyv : Not (Membership.mem v y)
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    have hxv : x ∈ v := or_iff_not_imp_left.mp (hs hxs) hxu
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      x : α
      hxs : Membership.mem s x
      hxu : Not (Membership.mem u x)
      y : α
      hys : Membership.mem s y
      hyv : Not (Membership.mem v y)
      hxv : Membership.mem v x
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    have hyu : y ∈ u := or_iff_not_imp_right.mp (hs hys) hyv
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      h : (Inter.inter s u).Nonempty → (Inter.inter s v).Nonempty → (Inter.inter s ( …
      x : α
      hxs : Membership.mem s x
      hxu : Not (Membership.mem u x)
      y : α
      hys : Membership.mem s y
      hyv : Not (Membership.mem v y)
      hxv : Membership.mem v x
      hyu : Membership.mem u y
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    exact h ⟨y, hys, hyu⟩ ⟨x, hxs, hxv⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      ⊢ IsPreconnected s
    -/
  · rw [isPreconnected_closed_iff]
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      ⊢ ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset s (Union.union …
    -/
    intro u v hu hv hs hsu hsv
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    by_contra H
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      h : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      H : Not (Inter.inter s (Inter.inter u v)).Nonempty
      ⊢ False
    -/
    specialize h u v hu hv hs (Set.not_nonempty_iff_eq_empty.mp H)
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      H : Not (Inter.inter s (Inter.inter u v)).Nonempty
      h : Or (HasSubset.Subset s u) (HasSubset.Subset s v)
      ⊢ False
    -/
    apply H
    /-
      case mpr
      α : Type u
      inst✝ : TopologicalSpace α
      s u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hs : HasSubset.Subset s (Union.union u v)
      hsu : (Inter.inter s u).Nonempty
      hsv : (Inter.inter s v).Nonempty
      H : Not (Inter.inter s (Inter.inter u v)).Nonempty
      h : Or (HasSubset.Subset s u) (HasSubset.Subset s v)
      ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
    -/
    cases' h with h h
      /-
        case mpr.inl
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsClosed u
        hv : IsClosed v
        hs : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hsv : (Inter.inter s v).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s u
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
    · rcases hsv with ⟨x, hxs, hxv⟩
      /-
        case mpr.inl.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsClosed u
        hv : IsClosed v
        hs : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s u
        x : α
        hxs : Membership.mem s x
        hxv : Membership.mem v x
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
      exact ⟨x, hxs, ⟨h hxs, hxv⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsClosed u
        hv : IsClosed v
        hs : HasSubset.Subset s (Union.union u v)
        hsu : (Inter.inter s u).Nonempty
        hsv : (Inter.inter s v).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s v
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
    · rcases hsu with ⟨x, hxs, hxu⟩
      /-
        case mpr.inr.intro.intro
        α : Type u
        inst✝ : TopologicalSpace α
        s u v : Set α
        hu : IsClosed u
        hv : IsClosed v
        hs : HasSubset.Subset s (Union.union u v)
        hsv : (Inter.inter s v).Nonempty
        H : Not (Inter.inter s (Inter.inter u v)).Nonempty
        h : HasSubset.Subset s v
        x : α
        hxs : Membership.mem s x
        hxu : Membership.mem u x
        ⊢ (Inter.inter s (Inter.inter u v)).Nonempty
      -/
      exact ⟨x, hxs, ⟨hxu, h hxs⟩⟩
      /-
        🎉 no goals
      -/


/-- A closed set `s` is preconnected if and only if for every cover by two closed sets that are
disjoint, it is contained in one of the two covering sets. -/
theorem isPreconnected_iff_subset_of_fully_disjoint_closed {s : Set α} (hs : IsClosed s) :
    IsPreconnected s ↔
      ∀ u v, IsClosed u → IsClosed v → s ⊆ u ∪ v → Disjoint u v → s ⊆ u ∨ s ⊆ v := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsClosed s
    ⊢ Iff (IsPreconnected s) (∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset …
  -/
  refine isPreconnected_iff_subset_of_disjoint_closed.trans ⟨?_, ?_⟩ <;> intro H u v hu hv hss huv
    /-
      case refine_1
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsClosed s
      H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hss : HasSubset.Subset s (Union.union u v)
      huv : Disjoint u v
      ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
    -/
  · apply H u v hu hv hss
    /-
      case refine_1
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsClosed s
      H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hss : HasSubset.Subset s (Union.union u v)
      huv : Disjoint u v
      ⊢ Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    -/
    rw [huv.inter_eq, inter_empty]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsClosed s
    H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    hss : HasSubset.Subset s (Union.union u v)
    huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
  -/
  have H1 := H (u ∩ s) (v ∩ s)
  /-
    case refine_2
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsClosed s
    H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    hss : HasSubset.Subset s (Union.union u v)
    huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    H1 : IsClosed (Inter.inter u s) → IsClosed (Inter.inter v s) → HasSubset.Subse …
    ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
  -/
  rw [subset_inter_iff, subset_inter_iff] at H1
  /-
    case refine_2
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsClosed s
    H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    hss : HasSubset.Subset s (Union.union u v)
    huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    H1 : IsClosed (Inter.inter u s) → IsClosed (Inter.inter v s) → HasSubset.Subse …
    ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
  -/
  simp only [Subset.refl, and_true] at H1
  /-
    case refine_2
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsClosed s
    H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    hss : HasSubset.Subset s (Union.union u v)
    huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    H1 : IsClosed (Inter.inter u s) → IsClosed (Inter.inter v s) → HasSubset.Subse …
    ⊢ Or (HasSubset.Subset s u) (HasSubset.Subset s v)
  -/
  apply H1 (hu.inter hs) (hv.inter hs)
    /-
      case refine_2.a
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsClosed s
      H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hss : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      H1 : IsClosed (Inter.inter u s) → IsClosed (Inter.inter v s) → HasSubset.Subse …
      ⊢ HasSubset.Subset s (Union.union (Inter.inter u s) (Inter.inter v s))
    -/
  · rw [← union_inter_distrib_right]
    /-
      case refine_2.a
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsClosed s
      H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hss : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      H1 : IsClosed (Inter.inter u s) → IsClosed (Inter.inter v s) → HasSubset.Subse …
      ⊢ HasSubset.Subset s (Inter.inter (Union.union u v) s)
    -/
    exact subset_inter hss Subset.rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2.a
      α : Type u
      inst✝ : TopologicalSpace α
      s : Set α
      hs : IsClosed s
      H : ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset s (Union.union …
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      hss : HasSubset.Subset s (Union.union u v)
      huv : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      H1 : IsClosed (Inter.inter u s) → IsClosed (Inter.inter v s) → HasSubset.Subse …
      ⊢ Disjoint (Inter.inter u s) (Inter.inter v s)
    -/
  · rwa [disjoint_iff_inter_eq_empty, ← inter_inter_distrib_right, inter_comm]
    /-
      🎉 no goals
    -/


theorem IsClopen.connectedComponent_subset {x} (hs : IsClopen s) (hx : x ∈ s) :
    connectedComponent x ⊆ s :=
  isPreconnected_connectedComponent.subset_isClopen hs ⟨x, mem_connectedComponent, hx⟩


/-- The connected component of a point is always a subset of the intersection of all its clopen
neighbourhoods. -/
theorem connectedComponent_subset_iInter_isClopen {x : α} :
    connectedComponent x ⊆ ⋂ Z : { Z : Set α // IsClopen Z ∧ x ∈ Z }, Z :=
  subset_iInter fun Z => Z.2.1.connectedComponent_subset Z.2.2


/-- A clopen set is the union of its connected components. -/
theorem IsClopen.biUnion_connectedComponent_eq {Z : Set α} (h : IsClopen Z) :
    ⋃ x ∈ Z, connectedComponent x = Z :=
  Subset.antisymm (iUnion₂_subset fun _ => h.connectedComponent_subset) fun _ h =>
    mem_iUnion₂_of_mem h mem_connectedComponent


open Set.Notation in
/-- If `u v : Set X` and `u ⊆ v` is clopen in `v`, then `u` is the union of the connected
components of `v` in `X` which intersect `u`. -/
lemma IsClopen.biUnion_connectedComponentIn {X : Type*} [TopologicalSpace X] {u v : Set X}
    (hu : IsClopen (v ↓∩ u)) (huv₁ : u ⊆ v) :
    u = ⋃ x ∈ u, connectedComponentIn v x := by
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsClopen (Set.preimage Subtype.val u)
    huv₁ : HasSubset.Subset u v
    ⊢ Eq u (Set.iUnion fun x => Set.iUnion fun h => connectedComponentIn v x)
  -/
  have := congr(((↑) : Set v → Set X) $(hu.biUnion_connectedComponent_eq.symm))
  simp only [Subtype.image_preimage_coe, mem_preimage, iUnion_coe_set, image_val_iUnion,
    inter_eq_right.mpr huv₁] at this
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsClopen (Set.preimage Subtype.val u)
    huv₁ : HasSubset.Subset u v
    this : Eq u (Set.iUnion fun i => Set.iUnion fun x => Set.iUnion fun i_1 => Set …
    ⊢ Eq u (Set.iUnion fun x => Set.iUnion fun h => connectedComponentIn v x)
  -/
  nth_rw 1 [this]
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsClopen (Set.preimage Subtype.val u)
    huv₁ : HasSubset.Subset u v
    this : Eq u (Set.iUnion fun i => Set.iUnion fun x => Set.iUnion fun i_1 => Set …
    ⊢ Eq (Set.iUnion fun i => Set.iUnion fun x => Set.iUnion fun i_1 => Set.image  …
  -/
  congr! 2 with x hx
  /-
    case h.e'_3.h
    X : Type u_3
    inst✝ : TopologicalSpace X
    u v : Set X
    hu : IsClopen (Set.preimage Subtype.val u)
    huv₁ : HasSubset.Subset u v
    this : Eq u (Set.iUnion fun i => Set.iUnion fun x => Set.iUnion fun i_1 => Set …
    x : X
    ⊢ Eq (Set.iUnion fun x_1 => Set.iUnion fun i => Set.image Subtype.val (connect …
  -/
  simp only [← connectedComponentIn_eq_image]
  exact le_antisymm (iUnion_subset fun _ ↦ le_rfl) <|
    iUnion_subset fun hx ↦ subset_iUnion₂_of_subset (huv₁ hx) hx le_rfl


/-- The preimage of a connected component is preconnected if the function has connected fibers
and a subset is closed iff the preimage is. -/
theorem preimage_connectedComponent_connected
    (connected_fibers : ∀ t : β, IsConnected (f ⁻¹' {t}))
    (hcl : ∀ T : Set β, IsClosed T ↔ IsClosed (f ⁻¹' T)) (t : β) :
    IsConnected (f ⁻¹' connectedComponent t) := by
  -- The following proof is essentially https://stacks.math.columbia.edu/tag/0377
  -- although the statement is slightly different
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    ⊢ IsConnected (Set.preimage f (connectedComponent t))
  -/
  have hf : Surjective f := Surjective.of_comp fun t : β => (connected_fibers t).1
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    ⊢ IsConnected (Set.preimage f (connectedComponent t))
  -/
  refine ⟨Nonempty.preimage connectedComponent_nonempty hf, ?_⟩
  have hT : IsClosed (f ⁻¹' connectedComponent t) :=
    (hcl (connectedComponent t)).1 isClosed_connectedComponent
  -- To show it's preconnected we decompose (f ⁻¹' connectedComponent t) as a subset of two
  -- closed disjoint sets in α. We want to show that it's a subset of either.
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    hT : IsClosed (Set.preimage f (connectedComponent t))
    ⊢ IsPreconnected (Set.preimage f (connectedComponent t))
  -/
  rw [isPreconnected_iff_subset_of_fully_disjoint_closed hT]
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    hT : IsClosed (Set.preimage f (connectedComponent t))
    ⊢ ∀ (u v : Set α), IsClosed u → IsClosed v → HasSubset.Subset (Set.preimage f  …
  -/
  intro u v hu hv huv uv_disj
  -- To do this we decompose connectedComponent t into T₁ and T₂
  -- we will show that connectedComponent t is a subset of either and hence
  -- (f ⁻¹' connectedComponent t) is a subset of u or v
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    hT : IsClosed (Set.preimage f (connectedComponent t))
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
    uv_disj : Disjoint u v
    ⊢ Or (HasSubset.Subset (Set.preimage f (connectedComponent t)) u) (HasSubset.S …
  -/
  let T₁ := { t' ∈ connectedComponent t | f ⁻¹' {t'} ⊆ u }
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    hT : IsClosed (Set.preimage f (connectedComponent t))
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
    uv_disj : Disjoint u v
    T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
    ⊢ Or (HasSubset.Subset (Set.preimage f (connectedComponent t)) u) (HasSubset.S …
  -/
  let T₂ := { t' ∈ connectedComponent t | f ⁻¹' {t'} ⊆ v }
  have fiber_decomp : ∀ t' ∈ connectedComponent t, f ⁻¹' {t'} ⊆ u ∨ f ⁻¹' {t'} ⊆ v := by
    intro t' ht'
    apply isPreconnected_iff_subset_of_disjoint_closed.1 (connected_fibers t').2 u v hu hv
    · exact Subset.trans (preimage_mono (singleton_subset_iff.2 ht')) huv
    rw [uv_disj.inter_eq, inter_empty]
  have T₁_u : f ⁻¹' T₁ = f ⁻¹' connectedComponent t ∩ u := by
    apply eq_of_subset_of_subset
    · rw [← biUnion_preimage_singleton]
      refine iUnion₂_subset fun t' ht' => subset_inter ?_ ht'.2
      rw [hf.preimage_subset_preimage_iff, singleton_subset_iff]
      exact ht'.1
    rintro a ⟨hat, hau⟩
    constructor
    · exact mem_preimage.1 hat
    refine (fiber_decomp (f a) (mem_preimage.1 hat)).resolve_right fun h => ?_
    exact uv_disj.subset_compl_right hau (h rfl)
  -- This proof is exactly the same as the above (modulo some symmetry)
  have T₂_v : f ⁻¹' T₂ = f ⁻¹' connectedComponent t ∩ v := by
    apply eq_of_subset_of_subset
    · rw [← biUnion_preimage_singleton]
      refine iUnion₂_subset fun t' ht' => subset_inter ?_ ht'.2
      rw [hf.preimage_subset_preimage_iff, singleton_subset_iff]
      exact ht'.1
    rintro a ⟨hat, hav⟩
    constructor
    · exact mem_preimage.1 hat
    · refine (fiber_decomp (f a) (mem_preimage.1 hat)).resolve_left fun h => ?_
      exact uv_disj.subset_compl_left hav (h rfl)
  -- Now we show T₁, T₂ are closed, cover connectedComponent t and are disjoint.
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    hT : IsClosed (Set.preimage f (connectedComponent t))
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
    uv_disj : Disjoint u v
    T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
    T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
    fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
    T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
    T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
    ⊢ Or (HasSubset.Subset (Set.preimage f (connectedComponent t)) u) (HasSubset.S …
  -/
  have hT₁ : IsClosed T₁ := (hcl T₁).2 (T₁_u.symm ▸ IsClosed.inter hT hu)
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
    hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
    t : β
    hf : Function.Surjective f
    hT : IsClosed (Set.preimage f (connectedComponent t))
    u v : Set α
    hu : IsClosed u
    hv : IsClosed v
    huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
    uv_disj : Disjoint u v
    T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
    T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
    fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
    T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
    T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
    hT₁ : IsClosed T₁
    ⊢ Or (HasSubset.Subset (Set.preimage f (connectedComponent t)) u) (HasSubset.S …
  -/
  have hT₂ : IsClosed T₂ := (hcl T₂).2 (T₂_v.symm ▸ IsClosed.inter hT hv)
  have T_decomp : connectedComponent t ⊆ T₁ ∪ T₂ := fun t' ht' => by
    rw [mem_union t' T₁ T₂]
    cases' fiber_decomp t' ht' with htu htv
    · left
      exact ⟨ht', htu⟩
    right
    exact ⟨ht', htv⟩
  have T_disjoint : Disjoint T₁ T₂ := by
    refine Disjoint.of_preimage hf ?_
    rw [T₁_u, T₂_v, disjoint_iff_inter_eq_empty, ← inter_inter_distrib_left, uv_disj.inter_eq,
      inter_empty]
  -- Now we do cases on whether (connectedComponent t) is a subset of T₁ or T₂ to show
  -- that the preimage is a subset of u or v.
  cases' (isPreconnected_iff_subset_of_fully_disjoint_closed isClosed_connectedComponent).1
    isPreconnected_connectedComponent T₁ T₂ hT₁ hT₂ T_decomp T_disjoint with h h
    /-
      case inl
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
      hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
      t : β
      hf : Function.Surjective f
      hT : IsClosed (Set.preimage f (connectedComponent t))
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
      uv_disj : Disjoint u v
      T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
      T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
      T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
      hT₁ : IsClosed T₁
      hT₂ : IsClosed T₂
      T_decomp : HasSubset.Subset (connectedComponent t) (Union.union T₁ T₂)
      T_disjoint : Disjoint T₁ T₂
      h : HasSubset.Subset (connectedComponent t) T₁
      ⊢ Or (HasSubset.Subset (Set.preimage f (connectedComponent t)) u) (HasSubset.S …
    -/
  · left
    /-
      case inl.h
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
      hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
      t : β
      hf : Function.Surjective f
      hT : IsClosed (Set.preimage f (connectedComponent t))
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
      uv_disj : Disjoint u v
      T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
      T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
      T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
      hT₁ : IsClosed T₁
      hT₂ : IsClosed T₂
      T_decomp : HasSubset.Subset (connectedComponent t) (Union.union T₁ T₂)
      T_disjoint : Disjoint T₁ T₂
      h : HasSubset.Subset (connectedComponent t) T₁
      ⊢ HasSubset.Subset (Set.preimage f (connectedComponent t)) u
    -/
    rw [Subset.antisymm_iff] at T₁_u
    suffices f ⁻¹' connectedComponent t ⊆ f ⁻¹' T₁
      from (this.trans T₁_u.1).trans inter_subset_right
    /-
      case inl.h
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
      hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
      t : β
      hf : Function.Surjective f
      hT : IsClosed (Set.preimage f (connectedComponent t))
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
      uv_disj : Disjoint u v
      T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
      T₁_u : And (HasSubset.Subset (Set.preimage f T₁) (Inter.inter (Set.preimage f  …
      T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
      hT₁ : IsClosed T₁
      hT₂ : IsClosed T₂
      T_decomp : HasSubset.Subset (connectedComponent t) (Union.union T₁ T₂)
      T_disjoint : Disjoint T₁ T₂
      h : HasSubset.Subset (connectedComponent t) T₁
      ⊢ HasSubset.Subset (Set.preimage f (connectedComponent t)) (Set.preimage f T₁)
    -/
    exact preimage_mono h
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
      hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
      t : β
      hf : Function.Surjective f
      hT : IsClosed (Set.preimage f (connectedComponent t))
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
      uv_disj : Disjoint u v
      T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
      T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
      T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
      hT₁ : IsClosed T₁
      hT₂ : IsClosed T₂
      T_decomp : HasSubset.Subset (connectedComponent t) (Union.union T₁ T₂)
      T_disjoint : Disjoint T₁ T₂
      h : HasSubset.Subset (connectedComponent t) T₂
      ⊢ Or (HasSubset.Subset (Set.preimage f (connectedComponent t)) u) (HasSubset.S …
    -/
  · right
    /-
      case inr.h
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
      hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
      t : β
      hf : Function.Surjective f
      hT : IsClosed (Set.preimage f (connectedComponent t))
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
      uv_disj : Disjoint u v
      T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
      T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
      T₂_v : Eq (Set.preimage f T₂) (Inter.inter (Set.preimage f (connectedComponent …
      hT₁ : IsClosed T₁
      hT₂ : IsClosed T₂
      T_decomp : HasSubset.Subset (connectedComponent t) (Union.union T₁ T₂)
      T_disjoint : Disjoint T₁ T₂
      h : HasSubset.Subset (connectedComponent t) T₂
      ⊢ HasSubset.Subset (Set.preimage f (connectedComponent t)) v
    -/
    rw [Subset.antisymm_iff] at T₂_v
    suffices f ⁻¹' connectedComponent t ⊆ f ⁻¹' T₂
      from (this.trans T₂_v.1).trans inter_subset_right
    /-
      case inr.h
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      connected_fibers : ∀ (t : β), IsConnected (Set.preimage f (Singleton.singleton …
      hcl : ∀ (T : Set β), Iff (IsClosed T) (IsClosed (Set.preimage f T))
      t : β
      hf : Function.Surjective f
      hT : IsClosed (Set.preimage f (connectedComponent t))
      u v : Set α
      hu : IsClosed u
      hv : IsClosed v
      huv : HasSubset.Subset (Set.preimage f (connectedComponent t)) (Union.union u v)
      uv_disj : Disjoint u v
      T₁ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      T₂ : Set β := setOf fun t' => And (Membership.mem (connectedComponent t) t') ( …
      fiber_decomp : ∀ (t' : β), Membership.mem (connectedComponent t) t' → Or (HasS …
      T₁_u : Eq (Set.preimage f T₁) (Inter.inter (Set.preimage f (connectedComponent …
      T₂_v : And (HasSubset.Subset (Set.preimage f T₂) (Inter.inter (Set.preimage f  …
      hT₁ : IsClosed T₁
      hT₂ : IsClosed T₂
      T_decomp : HasSubset.Subset (connectedComponent t) (Union.union T₁ T₂)
      T_disjoint : Disjoint T₁ T₂
      h : HasSubset.Subset (connectedComponent t) T₂
      ⊢ HasSubset.Subset (Set.preimage f (connectedComponent t)) (Set.preimage f T₂)
    -/
    exact preimage_mono h
    /-
      🎉 no goals
    -/


theorem Topology.IsQuotientMap.preimage_connectedComponent (hf : IsQuotientMap f)
    (h_fibers : ∀ y : β, IsConnected (f ⁻¹' {y})) (a : α) :
    f ⁻¹' connectedComponent (f a) = connectedComponent a :=
  ((preimage_connectedComponent_connected h_fibers (fun _ => hf.isClosed_preimage.symm)
      _).subset_connectedComponent mem_connectedComponent).antisymm
    (hf.continuous.mapsTo_connectedComponent a)


@[deprecated (since := "2024-10-22")]
alias QuotientMap.preimage_connectedComponent := IsQuotientMap.preimage_connectedComponent


lemma Topology.IsQuotientMap.image_connectedComponent {f : α → β} (hf : IsQuotientMap f)
    (h_fibers : ∀ y : β, IsConnected (f ⁻¹' {y})) (a : α) :
    f '' connectedComponent a = connectedComponent (f a) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Topology.IsQuotientMap f
    h_fibers : ∀ (y : β), IsConnected (Set.preimage f (Singleton.singleton y))
    a : α
    ⊢ Eq (Set.image f (connectedComponent a)) (connectedComponent (f a))
  -/
  rw [← hf.preimage_connectedComponent h_fibers, image_preimage_eq _ hf.surjective]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-22")]
alias QuotientMap.image_connectedComponent := IsQuotientMap.image_connectedComponent


/-- The setoid of connected components of a topological space -/
def connectedComponentSetoid (α : Type*) [TopologicalSpace α] : Setoid α :=
  ⟨fun x y => connectedComponent x = connectedComponent y,
                 /-
                   α✝ : Type u
                   β : Type v
                   ι : Type u_1
                   π : ι → Type u_2
                   inst✝¹ : TopologicalSpace α✝
                   s t u v : Set α✝
                   α : Type u_3
                   inst✝ : TopologicalSpace α
                   x : α
                   ⊢ Eq (connectedComponent x) (connectedComponent x)
                 -/
    ⟨fun x => by trivial, fun h1 => h1.symm, fun h1 h2 => h1.trans h2⟩⟩
                 /-
                   🎉 no goals
                 -/


/-- The quotient of a space by its connected components -/
def ConnectedComponents (α : Type u) [TopologicalSpace α] :=
  Quotient (connectedComponentSetoid α)


/-- Coercion from a topological space to the set of connected components of this space. -/
def mk : α → ConnectedComponents α := Quotient.mk''


instance : CoeTC α (ConnectedComponents α) := ⟨mk⟩


@[simp]
theorem coe_eq_coe {x y : α} :
    (x : ConnectedComponents α) = y ↔ connectedComponent x = connectedComponent y :=
  Quotient.eq''


theorem coe_ne_coe {x y : α} :
    (x : ConnectedComponents α) ≠ y ↔ connectedComponent x ≠ connectedComponent y :=
  coe_eq_coe.not


theorem coe_eq_coe' {x y : α} : (x : ConnectedComponents α) = y ↔ x ∈ connectedComponent y :=
  coe_eq_coe.trans connectedComponent_eq_iff_mem


instance [Inhabited α] : Inhabited (ConnectedComponents α) :=
  ⟨mk default⟩


instance : TopologicalSpace (ConnectedComponents α) :=
  inferInstanceAs (TopologicalSpace (Quotient _))


theorem surjective_coe : Surjective (mk : α → ConnectedComponents α) :=
  Quot.mk_surjective


theorem isQuotientMap_coe : IsQuotientMap (mk : α → ConnectedComponents α) :=
  isQuotientMap_quot_mk


@[deprecated (since := "2024-10-22")]
alias quotientMap_coe := isQuotientMap_coe


@[continuity]
theorem continuous_coe : Continuous (mk : α → ConnectedComponents α) :=
  isQuotientMap_coe.continuous


@[simp]
theorem range_coe : range (mk : α → ConnectedComponents α) = univ :=
  surjective_coe.range_eq


/-- The preimage of a singleton in `connectedComponents` is the connected component
of an element in the equivalence class. -/
theorem connectedComponents_preimage_singleton {x : α} :
    (↑) ⁻¹' ({↑x} : Set (ConnectedComponents α)) = connectedComponent x := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    ⊢ Eq (Set.preimage ConnectedComponents.mk (Singleton.singleton (ConnectedCompo …
  -/
  ext y
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    x y : α
    ⊢ Iff (Membership.mem (Set.preimage ConnectedComponents.mk (Singleton.singleto …
  -/
  rw [mem_preimage, mem_singleton_iff, ConnectedComponents.coe_eq_coe']
  /-
    🎉 no goals
  -/


/-- The preimage of the image of a set under the quotient map to `connectedComponents α`
is the union of the connected components of the elements in it. -/
theorem connectedComponents_preimage_image (U : Set α) :
    (↑) ⁻¹' ((↑) '' U : Set (ConnectedComponents α)) = ⋃ x ∈ U, connectedComponent x := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    U : Set α
    ⊢ Eq (Set.preimage ConnectedComponents.mk (Set.image ConnectedComponents.mk U) …
  -/
  simp only [connectedComponents_preimage_singleton, preimage_iUnion₂, image_eq_iUnion]
  /-
    🎉 no goals
  -/




/-- If every map to `Bool` (a discrete two-element space), that is
continuous on a set `s`, is constant on s, then s is preconnected -/
theorem isPreconnected_of_forall_constant {s : Set α}
    (hs : ∀ f : α → Bool, ContinuousOn f s → ∀ x ∈ s, ∀ y ∈ s, f x = f y) : IsPreconnected s := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : ∀ (f : α → Bool), ContinuousOn f s → ∀ (x : α), Membership.mem s x → ∀ (y …
    ⊢ IsPreconnected s
  -/
  unfold IsPreconnected
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : ∀ (f : α → Bool), ContinuousOn f s → ∀ (x : α), Membership.mem s x → ∀ (y …
    ⊢ ∀ (u v : Set α), IsOpen u → IsOpen v → HasSubset.Subset s (Union.union u v)  …
  -/
  by_contra!
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : ∀ (f : α → Bool), ContinuousOn f s → ∀ (x : α), Membership.mem s x → ∀ (y …
    this : Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Ha …
    ⊢ False
  -/
  rcases this with ⟨u, v, u_op, v_op, hsuv, ⟨x, x_in_s, x_in_u⟩, ⟨y, y_in_s, y_in_v⟩, H⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    hs : ∀ (f : α → Bool), ContinuousOn f s → ∀ (x : α), Membership.mem s x → ∀ (y …
    u v : Set α
    u_op : IsOpen u
    v_op : IsOpen v
    hsuv : HasSubset.Subset s (Union.union u v)
    x : α
    x_in_s : Membership.mem s x
    x_in_u : Membership.mem u x
    H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    y : α
    y_in_s : Membership.mem s y
    y_in_v : Membership.mem v y
    ⊢ False
  -/
  have hy : y ∉ u := fun y_in_u => eq_empty_iff_forall_not_mem.mp H y ⟨y_in_s, ⟨y_in_u, y_in_v⟩⟩
  have : ContinuousOn u.boolIndicator s := by
    apply (continuousOn_boolIndicator_iff_isClopen _ _).mpr ⟨_, _⟩
    · rw [preimage_subtype_coe_eq_compl hsuv H]
      exact (v_op.preimage continuous_subtype_val).isClosed_compl
    · exact u_op.preimage continuous_subtype_val
  simpa [(u.mem_iff_boolIndicator _).mp x_in_u, (u.not_mem_iff_boolIndicator _).mp hy] using
    hs _ this x x_in_s y y_in_s


/-- A `PreconnectedSpace` version of `isPreconnected_of_forall_constant` -/
theorem preconnectedSpace_of_forall_constant
    (hs : ∀ f : α → Bool, Continuous f → ∀ x y, f x = f y) : PreconnectedSpace α :=
  ⟨isPreconnected_of_forall_constant fun f hf x _ y _ =>
      hs f (continuous_iff_continuousOn_univ.mpr hf) x y⟩

