theorem Set.restrictPreimage_isInducing (s : Set β) (h : IsInducing f) :
    IsInducing (s.restrictPreimage f) := by
  simp_rw [← IsInducing.subtypeVal.of_comp_iff, isInducing_iff_nhds, restrictPreimage,
    MapsTo.coe_restrict, restrict_eq, ← @Filter.comap_comap _ _ _ _ _ f, Function.comp_apply] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set β
    h : ∀ (x : α), Eq (nhds x) (Filter.comap f (nhds (f x)))
    ⊢ ∀ (x : ↑(Set.preimage f s)), Eq (nhds x) (Filter.comap Subtype.val (Filter.c …
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set β
    h : ∀ (x : α), Eq (nhds x) (Filter.comap f (nhds (f x)))
    a : ↑(Set.preimage f s)
    ⊢ Eq (nhds a) (Filter.comap Subtype.val (Filter.comap f (nhds (f ↑a))))
  -/
  rw [← h, ← IsInducing.subtypeVal.nhds_eq_comap]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias Set.restrictPreimage_inducing := Set.restrictPreimage_isInducing


alias Topology.IsInducing.restrictPreimage := Set.restrictPreimage_isInducing


@[deprecated (since := "2024-10-28")] alias Inducing.restrictPreimage := IsInducing.restrictPreimage


theorem Set.restrictPreimage_isEmbedding (s : Set β) (h : IsEmbedding f) :
    IsEmbedding (s.restrictPreimage f) :=
  ⟨h.1.restrictPreimage s, h.2.restrictPreimage s⟩


@[deprecated (since := "2024-10-26")]
alias Set.restrictPreimage_embedding := Set.restrictPreimage_isEmbedding


alias Topology.IsEmbedding.restrictPreimage := Set.restrictPreimage_isEmbedding


@[deprecated (since := "2024-10-26")]
alias Embedding.restrictPreimage := IsEmbedding.restrictPreimage


theorem Set.restrictPreimage_isOpenEmbedding (s : Set β) (h : IsOpenEmbedding f) :
    IsOpenEmbedding (s.restrictPreimage f) :=
  ⟨h.1.restrictPreimage s,
    (s.range_restrictPreimage f).symm ▸ continuous_subtype_val.isOpen_preimage _ h.isOpen_range⟩


@[deprecated (since := "2024-10-18")]
alias Set.restrictPreimage_openEmbedding := Set.restrictPreimage_isOpenEmbedding


alias Topology.IsOpenEmbedding.restrictPreimage := Set.restrictPreimage_isOpenEmbedding


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.restrictPreimage := IsOpenEmbedding.restrictPreimage


theorem Set.restrictPreimage_isClosedEmbedding (s : Set β) (h : IsClosedEmbedding f) :
    IsClosedEmbedding (s.restrictPreimage f) :=
  ⟨h.1.restrictPreimage s,
    (s.range_restrictPreimage f).symm ▸ IsInducing.subtypeVal.isClosed_preimage _ h.isClosed_range⟩


@[deprecated (since := "2024-10-20")]
alias Set.restrictPreimage_closedEmbedding := Set.restrictPreimage_isClosedEmbedding


alias Topology.IsClosedEmbedding.restrictPreimage := Set.restrictPreimage_isClosedEmbedding


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding.restrictPreimage := IsClosedEmbedding.restrictPreimage


theorem IsClosedMap.restrictPreimage (H : IsClosedMap f) (s : Set β) :
    IsClosedMap (s.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    H : IsClosedMap f
    s : Set β
    ⊢ IsClosedMap (s.restrictPreimage f)
  -/
  intro t
  suffices ∀ u, IsClosed u → Subtype.val ⁻¹' u = t →
    ∃ v, IsClosed v ∧ Subtype.val ⁻¹' v = s.restrictPreimage f '' t by
      simpa [isClosed_induced_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    H : IsClosedMap f
    s : Set β
    t : Set ↑(Set.preimage f s)
    ⊢ ∀ (u : Set α), IsClosed u → Eq (Set.preimage Subtype.val u) t → Exists fun v …
  -/
  exact fun u hu e => ⟨f '' u, H u hu, by simp [← e, image_restrictPreimage]⟩
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided." (since := "2024-04-02")]
theorem Set.restrictPreimage_isClosedMap (s : Set β) (H : IsClosedMap f) :
    IsClosedMap (s.restrictPreimage f) := H.restrictPreimage s


theorem IsOpenMap.restrictPreimage (H : IsOpenMap f) (s : Set β) :
    IsOpenMap (s.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    H : IsOpenMap f
    s : Set β
    ⊢ IsOpenMap (s.restrictPreimage f)
  -/
  intro t
  suffices ∀ u, IsOpen u → Subtype.val ⁻¹' u = t →
    ∃ v, IsOpen v ∧ Subtype.val ⁻¹' v = s.restrictPreimage f '' t by
      simpa [isOpen_induced_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    H : IsOpenMap f
    s : Set β
    t : Set ↑(Set.preimage f s)
    ⊢ ∀ (u : Set α), IsOpen u → Eq (Set.preimage Subtype.val u) t → Exists fun v = …
  -/
  exact fun u hu e => ⟨f '' u, H u hu, by simp [← e, image_restrictPreimage]⟩
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided." (since := "2024-04-02")]
theorem Set.restrictPreimage_isOpenMap (s : Set β) (H : IsOpenMap f) :
    IsOpenMap (s.restrictPreimage f) := H.restrictPreimage s


theorem isOpen_iff_inter_of_iSup_eq_top (s : Set β) : IsOpen s ↔ ∀ i, IsOpen (s ∩ U i) := by
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    ⊢ Iff (IsOpen s) (∀ (i : ι), IsOpen (Inter.inter s ↑(U i)))
  -/
  constructor
    /-
      case mp
      β : Type u_2
      inst✝ : TopologicalSpace β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      s : Set β
      ⊢ IsOpen s → ∀ (i : ι), IsOpen (Inter.inter s ↑(U i))
    -/
  · exact fun H i => H.inter (U i).2
    /-
      🎉 no goals
    -/
    /-
      case mpr
      β : Type u_2
      inst✝ : TopologicalSpace β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      s : Set β
      ⊢ (∀ (i : ι), IsOpen (Inter.inter s ↑(U i))) → IsOpen s
    -/
  · intro H
    have : ⋃ i, (U i : Set β) = Set.univ := by
      convert congr_arg (SetLike.coe) hU
      simp
    /-
      case mpr
      β : Type u_2
      inst✝ : TopologicalSpace β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      s : Set β
      H : ∀ (i : ι), IsOpen (Inter.inter s ↑(U i))
      this : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      ⊢ IsOpen s
    -/
    rw [← s.inter_univ, ← this, Set.inter_iUnion]
    /-
      case mpr
      β : Type u_2
      inst✝ : TopologicalSpace β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      s : Set β
      H : ∀ (i : ι), IsOpen (Inter.inter s ↑(U i))
      this : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      ⊢ IsOpen (Set.iUnion fun i => Inter.inter s ↑(U i))
    -/
    exact isOpen_iUnion H
    /-
      🎉 no goals
    -/


theorem isOpen_iff_coe_preimage_of_iSup_eq_top (s : Set β) :
    IsOpen s ↔ ∀ i, IsOpen ((↑) ⁻¹' s : Set (U i)) := by
  -- Porting note: rewrote to avoid ´simp´ issues
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    ⊢ Iff (IsOpen s) (∀ (i : ι), IsOpen (Set.preimage Subtype.val s))
  -/
  rw [isOpen_iff_inter_of_iSup_eq_top hU s]
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    ⊢ Iff (∀ (i : ι), IsOpen (Inter.inter s ↑(U i))) (∀ (i : ι), IsOpen (Set.preim …
  -/
  refine forall_congr' fun i => ?_
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    i : ι
    ⊢ Iff (IsOpen (Inter.inter s ↑(U i))) (IsOpen (Set.preimage Subtype.val s))
  -/
  rw [(U _).2.isOpenEmbedding_subtypeVal.isOpen_iff_image_isOpen]
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    i : ι
    ⊢ Iff (IsOpen (Inter.inter s ↑(U i))) (IsOpen (Set.image Subtype.val (Set.prei …
  -/
  erw [Set.image_preimage_eq_inter_range]
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    i : ι
    ⊢ Iff (IsOpen (Inter.inter s ↑(U i))) (IsOpen (Inter.inter s (Set.range Subtyp …
  -/
  rw [Subtype.range_coe, Opens.carrier_eq_coe]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_coe_preimage_of_iSup_eq_top (s : Set β) :
    IsClosed s ↔ ∀ i, IsClosed ((↑) ⁻¹' s : Set (U i)) := by
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    ⊢ Iff (IsClosed s) (∀ (i : ι), IsClosed (Set.preimage Subtype.val s))
  -/
  simpa using isOpen_iff_coe_preimage_of_iSup_eq_top hU sᶜ
  /-
    🎉 no goals
  -/


theorem isLocallyClosed_iff_coe_preimage_of_iSup_eq_top (s : Set β) :
    IsLocallyClosed s ↔ ∀ i, IsLocallyClosed ((↑) ⁻¹' s : Set (U i)) := by
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    ⊢ Iff (IsLocallyClosed s) (∀ (i : ι), IsLocallyClosed (Set.preimage Subtype.va …
  -/
  simp_rw [isLocallyClosed_iff_isOpen_coborder]
  /-
    β : Type u_2
    inst✝ : TopologicalSpace β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    s : Set β
    ⊢ Iff (IsOpen (coborder s)) (∀ (i : ι), IsOpen (coborder (Set.preimage Subtype …
  -/
  rw [isOpen_iff_coe_preimage_of_iSup_eq_top hU]
  exact forall_congr' fun i ↦ by
    have : coborder ((↑) ⁻¹' s : Set (U i)) = Subtype.val ⁻¹' coborder s := by
      exact (U i).isOpen.isOpenEmbedding_subtypeVal.coborder_preimage _
    rw [this]


theorem isOpenMap_iff_isOpenMap_of_iSup_eq_top :
    IsOpenMap f ↔ ∀ i, IsOpenMap ((U i).1.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ Iff (IsOpenMap f) (∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f))
  -/
  refine ⟨fun h i => h.restrictPreimage _, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ (∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f)) → IsOpenMap f
  -/
  rintro H s hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsOpen s
    ⊢ IsOpen (Set.image f s)
  -/
  rw [isOpen_iff_coe_preimage_of_iSup_eq_top hU]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsOpen s
    ⊢ ∀ (i : ι), IsOpen (Set.preimage Subtype.val (Set.image f s))
  -/
  intro i
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsOpen s
    i : ι
    ⊢ IsOpen (Set.preimage Subtype.val (Set.image f s))
  -/
  convert H i _ (hs.preimage continuous_subtype_val)
  /-
    case h.e'_3.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsOpen s
    i : ι
    e_1✝ : Eq (Subtype fun x => Membership.mem (U i) x) ↑(U i).carrier
    ⊢ Eq (Set.preimage Subtype.val (Set.image f s)) (Set.image ((U i).carrier.rest …
  -/
  ext ⟨x, hx⟩
  suffices (∃ y, y ∈ s ∧ f y = x) ↔ ∃ y, y ∈ s ∧ f y ∈ U i ∧ f y = x by
    simpa [Set.restrictPreimage, ← Subtype.coe_inj]
  /-
    case h.e'_3.h.h.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsOpenMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsOpen s
    i : ι
    e_1✝ : Eq (Subtype fun x => Membership.mem (U i) x) ↑(U i).carrier
    x : β
    hx : Membership.mem (U i) x
    ⊢ Iff (Exists fun y => And (Membership.mem s y) (Eq (f y) x)) (Exists fun y => …
  -/
  exact ⟨fun ⟨a, b, c⟩ => ⟨a, b, c.symm ▸ hx, c⟩, fun ⟨a, b, _, c⟩ => ⟨a, b, c⟩⟩
  /-
    🎉 no goals
  -/


theorem isClosedMap_iff_isClosedMap_of_iSup_eq_top :
    IsClosedMap f ↔ ∀ i, IsClosedMap ((U i).1.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ Iff (IsClosedMap f) (∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage  …
  -/
  refine ⟨fun h i => h.restrictPreimage _, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ (∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage f)) → IsClosedMap f
  -/
  rintro H s hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsClosed s
    ⊢ IsClosed (Set.image f s)
  -/
  rw [isClosed_iff_coe_preimage_of_iSup_eq_top hU]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsClosed s
    ⊢ ∀ (i : ι), IsClosed (Set.preimage Subtype.val (Set.image f s))
  -/
  intro i
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsClosed s
    i : ι
    ⊢ IsClosed (Set.preimage Subtype.val (Set.image f s))
  -/
  convert H i _ ⟨⟨_, hs.1, eq_compl_comm.mpr rfl⟩⟩
  /-
    case h.e'_3.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsClosed s
    i : ι
    e_1✝ : Eq (Subtype fun x => Membership.mem (U i) x) ↑(U i).carrier
    ⊢ Eq (Set.preimage Subtype.val (Set.image f s)) (Set.image ((U i).carrier.rest …
  -/
  ext ⟨x, hx⟩
  suffices (∃ y, y ∈ s ∧ f y = x) ↔ ∃ y, y ∈ s ∧ f y ∈ U i ∧ f y = x by
    simpa [Set.restrictPreimage, ← Subtype.coe_inj]
  /-
    case h.e'_3.h.h.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), IsClosedMap ((U i).carrier.restrictPreimage f)
    s : Set α
    hs : IsClosed s
    i : ι
    e_1✝ : Eq (Subtype fun x => Membership.mem (U i) x) ↑(U i).carrier
    x : β
    hx : Membership.mem (U i) x
    ⊢ Iff (Exists fun y => And (Membership.mem s y) (Eq (f y) x)) (Exists fun y => …
  -/
  exact ⟨fun ⟨a, b, c⟩ => ⟨a, b, c.symm ▸ hx, c⟩, fun ⟨a, b, _, c⟩ => ⟨a, b, c⟩⟩
  /-
    🎉 no goals
  -/


theorem inducing_iff_inducing_of_iSup_eq_top (h : Continuous f) :
    IsInducing f ↔ ∀ i, IsInducing ((U i).1.restrictPreimage f) := by
  simp_rw [← IsInducing.subtypeVal.of_comp_iff, isInducing_iff_nhds, restrictPreimage,
    MapsTo.coe_restrict, restrict_eq, ← @Filter.comap_comap _ _ _ _ _ f]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (∀ (x : α), Eq (nhds x) (Filter.comap f (nhds (f x)))) (∀ (i : ι) (x : ↑ …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ (∀ (x : α), Eq (nhds x) (Filter.comap f (nhds (f x)))) → ∀ (i : ι) (x : ↑(Se …
    -/
  · intro H i x
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      H : ∀ (x : α), Eq (nhds x) (Filter.comap f (nhds (f x)))
      i : ι
      x : ↑(Set.preimage f (U i).carrier)
      ⊢ Eq (nhds x) (Filter.comap Subtype.val (Filter.comap f (nhds (Function.comp f …
    -/
    rw [Function.comp_apply, ← H, ← IsInducing.subtypeVal.nhds_eq_comap]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ (∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), Eq (nhds x) (Filter.comap  …
    -/
  · intro H x
    obtain ⟨i, hi⟩ :=
      Opens.mem_iSup.mp
        (show f x ∈ iSup U by
          rw [hU]
          trivial)
    rw [← IsOpenEmbedding.map_nhds_eq (h.1 _ (U i).2).isOpenEmbedding_subtypeVal ⟨x, hi⟩,
      (H i) ⟨x, hi⟩, Filter.subtype_coe_map_comap, Function.comp_apply, Subtype.coe_mk,
      inf_eq_left, Filter.le_principal_iff]
    /-
      case mpr.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      H : ∀ (i : ι) (x : ↑(Set.preimage f (U i).carrier)), Eq (nhds x) (Filter.comap …
      x : α
      i : ι
      hi : Membership.mem (U i) (f x)
      ⊢ Membership.mem (Filter.comap f (nhds (f x))) (Set.preimage f (U i).carrier)
    -/
    exact Filter.preimage_mem_comap ((U i).2.mem_nhds hi)
    /-
      🎉 no goals
    -/


theorem isEmbedding_iff_of_iSup_eq_top (h : Continuous f) :
    IsEmbedding f ↔ ∀ i, IsEmbedding ((U i).1.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (Topology.IsEmbedding f) (∀ (i : ι), Topology.IsEmbedding ((U i).carrier …
  -/
  simp_rw [isEmbedding_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (And (Topology.IsInducing f) (Function.Injective f)) (∀ (i : ι), And (To …
  -/
  rw [forall_and]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (And (Topology.IsInducing f) (Function.Injective f)) (And (∀ (x : ι), To …
  -/
  apply and_congr
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (Topology.IsInducing f) (∀ (x : ι), Topology.IsInducing ((U x).carrier.r …
    -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  · apply inducing_iff_inducing_of_iSup_eq_top <;> assumption
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (Function.Injective f) (∀ (x : ι), Function.Injective ((U x).carrier.res …
    -/
  · apply Set.injective_iff_injective_of_iUnion_eq_univ
    /-
      case h₂.hU
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Eq (Set.iUnion fun i => (U i).carrier) Set.univ
    -/
    convert congr_arg SetLike.coe hU
    /-
      case h.e'_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Eq (Set.iUnion fun i => (U i).carrier) ↑(iSup U)
    -/
    simp
    /-
      🎉 no goals
    -/


omit hU in
/--
Given a continuous map `f : X → Y` between topological spaces.
Suppose we have an open cover `V i` of the range of `f`, and an open cover `U i` of `X` that is
coarser than the pullback of `V` under `f`.
To check that `f` is an embedding it suffices to check that `U i → Y` is an embedding for all `i`.
-/
theorem isEmbedding_of_iSup_eq_top_of_preimage_subset_range
    {X Y} [TopologicalSpace X] [TopologicalSpace Y]
    (f : X → Y) (h : Continuous f) {ι : Type*}
    (U : ι → Opens Y) (hU : Set.range f ⊆ (iSup U : _))
    (V : ι → Type*) [∀ i, TopologicalSpace (V i)]
    (iV : ∀ i, V i → X) (hiV : ∀ i, Continuous (iV i)) (hV : ∀ i, f ⁻¹' U i ⊆ Set.range (iV i))
    (hV' : ∀ i, IsEmbedding (f ∘ iV i)) : IsEmbedding f := by
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    ⊢ Topology.IsEmbedding f
  -/
  wlog hU' : iSup U = ⊤
    /-
      case inr
      X : Type u_6
      Y : Type u_7
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      f : X → Y
      h : Continuous f
      ι : Type u_4
      U : ι → TopologicalSpace.Opens Y
      hU : HasSubset.Subset (Set.range f) ↑(iSup U)
      V : ι → Type u_5
      inst✝ : (i : ι) → TopologicalSpace (V i)
      iV : (i : ι) → V i → X
      hiV : ∀ (i : ι), Continuous (iV i)
      hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
      hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
      this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
      hU' : Not (Eq (iSup U) Top.top)
      ⊢ Topology.IsEmbedding f
    -/
  · let f₀ : X → Set.range f := fun x ↦ ⟨f x, ⟨x, rfl⟩⟩
    /-
      case inr
      X : Type u_6
      Y : Type u_7
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      f : X → Y
      h : Continuous f
      ι : Type u_4
      U : ι → TopologicalSpace.Opens Y
      hU : HasSubset.Subset (Set.range f) ↑(iSup U)
      V : ι → Type u_5
      inst✝ : (i : ι) → TopologicalSpace (V i)
      iV : (i : ι) → V i → X
      hiV : ∀ (i : ι), Continuous (iV i)
      hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
      hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
      this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
      hU' : Not (Eq (iSup U) Top.top)
      f₀ : X → ↑(Set.range f) := fun x => ⟨f x, ⋯⟩
      ⊢ Topology.IsEmbedding f
    -/
    suffices IsEmbedding f₀ from IsEmbedding.subtypeVal.comp this
    have hU'' : (⨆ i, (U i).comap ⟨Subtype.val, continuous_subtype_val⟩ :
        Opens (Set.range f)) = ⊤ := by
      rw [← top_le_iff]
      simpa [Set.range_subset_iff, SetLike.le_def] using hU
    /-
      case inr
      X : Type u_6
      Y : Type u_7
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      f : X → Y
      h : Continuous f
      ι : Type u_4
      U : ι → TopologicalSpace.Opens Y
      hU : HasSubset.Subset (Set.range f) ↑(iSup U)
      V : ι → Type u_5
      inst✝ : (i : ι) → TopologicalSpace (V i)
      iV : (i : ι) → V i → X
      hiV : ∀ (i : ι), Continuous (iV i)
      hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
      hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
      this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
      hU' : Not (Eq (iSup U) Top.top)
      f₀ : X → ↑(Set.range f) := fun x => ⟨f x, ⋯⟩
      hU'' : Eq (iSup fun i => (TopologicalSpace.Opens.comap { toFun := Subtype.val, …
      ⊢ Topology.IsEmbedding f₀
    -/
    refine this _ ?_ _ ?_ V iV hiV ?_ ?_ hU''
      /-
        case inr.refine_1
        X : Type u_6
        Y : Type u_7
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        f : X → Y
        h : Continuous f
        ι : Type u_4
        U : ι → TopologicalSpace.Opens Y
        hU : HasSubset.Subset (Set.range f) ↑(iSup U)
        V : ι → Type u_5
        inst✝ : (i : ι) → TopologicalSpace (V i)
        iV : (i : ι) → V i → X
        hiV : ∀ (i : ι), Continuous (iV i)
        hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
        hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
        this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
        hU' : Not (Eq (iSup U) Top.top)
        f₀ : X → ↑(Set.range f) := fun x => ⟨f x, ⋯⟩
        hU'' : Eq (iSup fun i => (TopologicalSpace.Opens.comap { toFun := Subtype.val, …
        ⊢ Continuous f₀
      -/
    · fun_prop
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2
        X : Type u_6
        Y : Type u_7
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        f : X → Y
        h : Continuous f
        ι : Type u_4
        U : ι → TopologicalSpace.Opens Y
        hU : HasSubset.Subset (Set.range f) ↑(iSup U)
        V : ι → Type u_5
        inst✝ : (i : ι) → TopologicalSpace (V i)
        iV : (i : ι) → V i → X
        hiV : ∀ (i : ι), Continuous (iV i)
        hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
        hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
        this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
        hU' : Not (Eq (iSup U) Top.top)
        f₀ : X → ↑(Set.range f) := fun x => ⟨f x, ⋯⟩
        hU'' : Eq (iSup fun i => (TopologicalSpace.Opens.comap { toFun := Subtype.val, …
        ⊢ HasSubset.Subset (Set.range f₀) ↑(iSup fun i => (TopologicalSpace.Opens.coma …
      -/
    · rw [hU'']; simp
                 /-
                   🎉 no goals
                 -/
      /-
        case inr.refine_3
        X : Type u_6
        Y : Type u_7
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        f : X → Y
        h : Continuous f
        ι : Type u_4
        U : ι → TopologicalSpace.Opens Y
        hU : HasSubset.Subset (Set.range f) ↑(iSup U)
        V : ι → Type u_5
        inst✝ : (i : ι) → TopologicalSpace (V i)
        iV : (i : ι) → V i → X
        hiV : ∀ (i : ι), Continuous (iV i)
        hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
        hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
        this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
        hU' : Not (Eq (iSup U) Top.top)
        f₀ : X → ↑(Set.range f) := fun x => ⟨f x, ⋯⟩
        hU'' : Eq (iSup fun i => (TopologicalSpace.Opens.comap { toFun := Subtype.val, …
        ⊢ ∀ (i : ι), HasSubset.Subset (Set.preimage f₀ ↑((TopologicalSpace.Opens.comap …
      -/
    · exact hV
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_4
        X : Type u_6
        Y : Type u_7
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        f : X → Y
        h : Continuous f
        ι : Type u_4
        U : ι → TopologicalSpace.Opens Y
        hU : HasSubset.Subset (Set.range f) ↑(iSup U)
        V : ι → Type u_5
        inst✝ : (i : ι) → TopologicalSpace (V i)
        iV : (i : ι) → V i → X
        hiV : ∀ (i : ι), Continuous (iV i)
        hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
        hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
        this : ∀ {X : Type u_6} {Y : Type u_7} [inst : TopologicalSpace X] [inst_1 : T …
        hU' : Not (Eq (iSup U) Top.top)
        f₀ : X → ↑(Set.range f) := fun x => ⟨f x, ⋯⟩
        hU'' : Eq (iSup fun i => (TopologicalSpace.Opens.comap { toFun := Subtype.val, …
        ⊢ ∀ (i : ι), Topology.IsEmbedding (Function.comp f₀ (iV i))
      -/
    · exact fun i ↦ IsEmbedding.of_comp (by fun_prop) continuous_subtype_val (hV' i)
      /-
        🎉 no goals
      -/
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    ⊢ Topology.IsEmbedding f
  -/
  rw [isEmbedding_iff_of_iSup_eq_top hU' h]
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    ⊢ ∀ (i : ι), Topology.IsEmbedding ((U i).carrier.restrictPreimage f)
  -/
  intro i
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    ⊢ Topology.IsEmbedding ((U i).carrier.restrictPreimage f)
  -/
  let f' := (Subtype.val ∘ (f ⁻¹' U i).restrictPreimage (iV i))
  have : IsEmbedding f' :=
    IsEmbedding.subtypeVal.comp ((IsEmbedding.of_comp (hiV i) h (hV' _)).restrictPreimage _)
  have hf' : Set.range f' = f ⁻¹' U i := by
    simpa [f', Set.range_comp, Set.range_restrictPreimage] using hV i
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    ⊢ Topology.IsEmbedding ((U i).carrier.restrictPreimage f)
  -/
  let e := (Homeomorph.ofIsEmbedding _ this).trans (Homeomorph.setCongr hf')
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    e : Homeomorph ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) ↑(Set.preimage f …
    ⊢ Topology.IsEmbedding ((U i).carrier.restrictPreimage f)
  -/
  refine IsEmbedding.of_comp (by fun_prop) continuous_subtype_val ?_
  /-
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    e : Homeomorph ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) ↑(Set.preimage f …
    ⊢ Topology.IsEmbedding (Function.comp Subtype.val ((U i).carrier.restrictPreim …
  -/
  convert ((hV' i).comp IsEmbedding.subtypeVal).comp e.symm.isEmbedding
  /-
    case h.e'_5.h
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    e : Homeomorph ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) ↑(Set.preimage f …
    e_1✝ : Eq ↑(Set.preimage f (U i).carrier) ↑(Set.preimage f ↑(U i))
    ⊢ Eq (Function.comp Subtype.val ((U i).carrier.restrictPreimage f)) (Function. …
  -/
  ext x
  /-
    case h.e'_5.h.h
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    e : Homeomorph ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) ↑(Set.preimage f …
    e_1✝ : Eq ↑(Set.preimage f (U i).carrier) ↑(Set.preimage f ↑(U i))
    x : ↑(Set.preimage f (U i).carrier)
    ⊢ Eq (Function.comp Subtype.val ((U i).carrier.restrictPreimage f) x) (Functio …
  -/
  obtain ⟨x, rfl⟩ := e.surjective x
  /-
    case h.e'_5.h.h.intro
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    e : Homeomorph ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) ↑(Set.preimage f …
    e_1✝ : Eq ↑(Set.preimage f (U i).carrier) ↑(Set.preimage f ↑(U i))
    x : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i)))
    ⊢ Eq (Function.comp Subtype.val ((U i).carrier.restrictPreimage f) (e x)) (Fun …
  -/
  simp
  /-
    case h.e'_5.h.h.intro
    X : Type u_6
    Y : Type u_7
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    h : Continuous f
    ι : Type u_4
    U : ι → TopologicalSpace.Opens Y
    hU : HasSubset.Subset (Set.range f) ↑(iSup U)
    V : ι → Type u_5
    inst✝ : (i : ι) → TopologicalSpace (V i)
    iV : (i : ι) → V i → X
    hiV : ∀ (i : ι), Continuous (iV i)
    hV : ∀ (i : ι), HasSubset.Subset (Set.preimage f ↑(U i)) (Set.range (iV i))
    hV' : ∀ (i : ι), Topology.IsEmbedding (Function.comp f (iV i))
    hU' : Eq (iSup U) Top.top
    i : ι
    f' : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) → X := Function.comp Subty …
    this : Topology.IsEmbedding f'
    hf' : Eq (Set.range f') (Set.preimage f ↑(U i))
    e : Homeomorph ↑(Set.preimage (iV i) (Set.preimage f ↑(U i))) ↑(Set.preimage f …
    e_1✝ : Eq ↑(Set.preimage f (U i).carrier) ↑(Set.preimage f ↑(U i))
    x : ↑(Set.preimage (iV i) (Set.preimage f ↑(U i)))
    ⊢ Eq (f ↑(e x)) (f (iV i ↑x))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias embedding_iff_embedding_of_iSup_eq_top := isEmbedding_iff_of_iSup_eq_top


theorem isOpenEmbedding_iff_isOpenEmbedding_of_iSup_eq_top (h : Continuous f) :
    IsOpenEmbedding f ↔ ∀ i, IsOpenEmbedding ((U i).1.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (Topology.IsOpenEmbedding f) (∀ (i : ι), Topology.IsOpenEmbedding ((U i) …
  -/
  simp_rw [isOpenEmbedding_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (And (Topology.IsEmbedding f) (IsOpen (Set.range f))) (∀ (i : ι), And (T …
  -/
  rw [forall_and]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (And (Topology.IsEmbedding f) (IsOpen (Set.range f))) (And (∀ (x : ι), T …
  -/
  apply and_congr
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (Topology.IsEmbedding f) (∀ (x : ι), Topology.IsEmbedding ((U x).carrier …
    -/
                                             /-
                                               🎉 no goals
                                             -/
  · apply isEmbedding_iff_of_iSup_eq_top <;> assumption
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (IsOpen (Set.range f)) (∀ (x : ι), IsOpen (Set.range ((U x).carrier.rest …
    -/
  · simp_rw [Set.range_restrictPreimage]
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (IsOpen (Set.range f)) (∀ (x : ι), IsOpen (Set.preimage Subtype.val (Set …
    -/
    apply isOpen_iff_coe_preimage_of_iSup_eq_top hU
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_openEmbedding_of_iSup_eq_top :=
  isOpenEmbedding_iff_isOpenEmbedding_of_iSup_eq_top


theorem isClosedEmbedding_iff_isClosedEmbedding_of_iSup_eq_top (h : Continuous f) :
    IsClosedEmbedding f ↔ ∀ i, IsClosedEmbedding ((U i).1.restrictPreimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (Topology.IsClosedEmbedding f) (∀ (i : ι), Topology.IsClosedEmbedding (( …
  -/
  simp_rw [isClosedEmbedding_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (And (Topology.IsEmbedding f) (IsClosed (Set.range f))) (∀ (i : ι), And  …
  -/
  rw [forall_and]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    h : Continuous f
    ⊢ Iff (And (Topology.IsEmbedding f) (IsClosed (Set.range f))) (And (∀ (x : ι), …
  -/
  apply and_congr
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (Topology.IsEmbedding f) (∀ (x : ι), Topology.IsEmbedding ((U x).carrier …
    -/
                                             /-
                                               🎉 no goals
                                             -/
  · apply isEmbedding_iff_of_iSup_eq_top <;> assumption
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (IsClosed (Set.range f)) (∀ (x : ι), IsClosed (Set.range ((U x).carrier. …
    -/
  · simp_rw [Set.range_restrictPreimage]
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      ι : Type u_3
      U : ι → TopologicalSpace.Opens β
      hU : Eq (iSup U) Top.top
      h : Continuous f
      ⊢ Iff (IsClosed (Set.range f)) (∀ (x : ι), IsClosed (Set.preimage Subtype.val  …
    -/
    apply isClosed_iff_coe_preimage_of_iSup_eq_top hU
    /-
      🎉 no goals
    -/


omit [TopologicalSpace α] in
theorem denseRange_iff_denseRange_of_iSup_eq_top :
    DenseRange f ↔ ∀ i, DenseRange ((U i).1.restrictPreimage f) := by
  simp_rw [denseRange_iff_closure_range, Set.range_restrictPreimage,
    ← (U _).2.isOpenEmbedding_subtypeVal.isOpenMap.preimage_closure_eq_closure_preimage
      continuous_subtype_val]
  simp only [Opens.carrier_eq_coe, SetLike.coe_sort_coe, preimage_eq_univ_iff,
    Subtype.range_coe_subtype, SetLike.mem_coe]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ Iff (Eq (closure (Set.range f)) Set.univ) (∀ (i : ι), HasSubset.Subset (setO …
  -/
  rw [← iUnion_subset_iff, ← Set.univ_subset_iff, iff_iff_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ Eq (HasSubset.Subset Set.univ (closure (Set.range f))) (HasSubset.Subset (Se …
  -/
  congr 1
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    f : α → β
    ι : Type u_3
    U : ι → TopologicalSpace.Opens β
    hU : Eq (iSup U) Top.top
    ⊢ Eq Set.univ (Set.iUnion fun i => setOf fun x => Membership.mem (U i) x)
  -/
  simpa using congr(($hU).1).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_iff_closedEmbedding_of_iSup_eq_top :=
 isClosedEmbedding_iff_isClosedEmbedding_of_iSup_eq_top

