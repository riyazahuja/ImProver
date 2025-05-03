/-- A T1 space with a clopen basis is totally separated. -/
theorem totallySeparatedSpace_of_t1_of_basis_clopen [T1Space X]
    (h : IsTopologicalBasis { s : Set X | IsClopen s }) : TotallySeparatedSpace X := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    h : TopologicalSpace.IsTopologicalBasis (setOf fun s => IsClopen s)
    ⊢ TotallySeparatedSpace X
  -/
  constructor
  /-
    case isTotallySeparated_univ
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    h : TopologicalSpace.IsTopologicalBasis (setOf fun s => IsClopen s)
    ⊢ IsTotallySeparated Set.univ
  -/
  rintro x - y - hxy
  /-
    case isTotallySeparated_univ
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T1Space X
    h : TopologicalSpace.IsTopologicalBasis (setOf fun s => IsClopen s)
    x y : X
    hxy : Ne x y
    ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
  -/
  rcases h.mem_nhds_iff.mp (isOpen_ne.mem_nhds hxy) with ⟨U, hU, hxU, hyU⟩
  exact ⟨U, Uᶜ, hU.isOpen, hU.compl.isOpen, hxU, fun h => hyU h rfl, (union_compl_self U).superset,
    disjoint_compl_right⟩


theorem nhds_basis_clopen (x : X) : (𝓝 x).HasBasis (fun s : Set X => x ∈ s ∧ IsClopen s) id :=
  ⟨fun U => by
    /-
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : T2Space X
      inst✝¹ : CompactSpace X
      inst✝ : TotallyDisconnectedSpace X
      x : X
      U : Set X
      ⊢ Iff (Membership.mem (nhds x) U) (Exists fun i => And (And (Membership.mem i  …
    -/
    constructor
    · have hx : connectedComponent x = {x} :=
        totallyDisconnectedSpace_iff_connectedComponent_singleton.mp ‹_› x
      /-
        case mp
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (connectedComponent x) (Singleton.singleton x)
        ⊢ Membership.mem (nhds x) U → Exists fun i => And (And (Membership.mem i x) (I …
      -/
      rw [connectedComponent_eq_iInter_isClopen] at hx
      /-
        case mp
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
        ⊢ Membership.mem (nhds x) U → Exists fun i => And (And (Membership.mem i x) (I …
      -/
      intro hU
      /-
        case mp
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
        hU : Membership.mem (nhds x) U
        ⊢ Exists fun i => And (And (Membership.mem i x) (IsClopen i)) (HasSubset.Subse …
      -/
      let N := { s // IsClopen s ∧ x ∈ s }
      /-
        case mp
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
        hU : Membership.mem (nhds x) U
        N : Type (max 0 u_1) := Subtype fun s => And (IsClopen s) (Membership.mem s x)
        ⊢ Exists fun i => And (And (Membership.mem i x) (IsClopen i)) (HasSubset.Subse …
      -/
      rsuffices ⟨⟨s, hs, hs'⟩, hs''⟩ : ∃ s : N, s.val ⊆ U
        /-
          case mp.intro.mk.intro
          X : Type u_1
          inst✝³ : TopologicalSpace X
          inst✝² : T2Space X
          inst✝¹ : CompactSpace X
          inst✝ : TotallyDisconnectedSpace X
          x : X
          U : Set X
          hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
          hU : Membership.mem (nhds x) U
          N : Type (max 0 u_1) := Subtype fun s => And (IsClopen s) (Membership.mem s x)
          s : Set X
          hs : IsClopen s
          hs' : Membership.mem s x
          hs'' : HasSubset.Subset (↑⟨s, ⋯⟩) U
          ⊢ Exists fun i => And (And (Membership.mem i x) (IsClopen i)) (HasSubset.Subse …
        -/
      · exact ⟨s, ⟨hs', hs⟩, hs''⟩
        /-
          🎉 no goals
        -/
      /-
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
        hU : Membership.mem (nhds x) U
        N : Type (max 0 u_1) := Subtype fun s => And (IsClopen s) (Membership.mem s x)
        ⊢ Exists fun s => HasSubset.Subset (↑s) U
      -/
      haveI : Nonempty N := ⟨⟨univ, isClopen_univ, mem_univ x⟩⟩
      /-
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
        hU : Membership.mem (nhds x) U
        N : Type (max 0 u_1) := Subtype fun s => And (IsClopen s) (Membership.mem s x)
        this : Nonempty N
        ⊢ Exists fun s => HasSubset.Subset (↑s) U
      -/
      have hNcl : ∀ s : N, IsClosed s.val := fun s => s.property.1.1
      have hdir : Directed Superset fun s : N => s.val := by
        rintro ⟨s, hs, hxs⟩ ⟨t, ht, hxt⟩
        exact ⟨⟨s ∩ t, hs.inter ht, ⟨hxs, hxt⟩⟩, inter_subset_left, inter_subset_right⟩
      have h_nhd : ∀ y ∈ ⋂ s : N, s.val, U ∈ 𝓝 y := fun y y_in => by
        rw [hx, mem_singleton_iff] at y_in
        rwa [y_in]
      /-
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        hx : Eq (Set.iInter fun s => ↑s) (Singleton.singleton x)
        hU : Membership.mem (nhds x) U
        N : Type (max 0 u_1) := Subtype fun s => And (IsClopen s) (Membership.mem s x)
        this : Nonempty N
        hNcl : ∀ (s : N), IsClosed ↑s
        hdir : Directed Superset fun s => ↑s
        h_nhd : ∀ (y : X), Membership.mem (Set.iInter fun s => ↑s) y → Membership.mem  …
        ⊢ Exists fun s => HasSubset.Subset (↑s) U
      -/
      exact exists_subset_nhds_of_compactSpace hdir hNcl h_nhd
      /-
        🎉 no goals
      -/
      /-
        case mpr
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U : Set X
        ⊢ (Exists fun i => And (And (Membership.mem i x) (IsClopen i)) (HasSubset.Subs …
      -/
    · rintro ⟨V, ⟨hxV, -, V_op⟩, hUV : V ⊆ U⟩
      /-
        case mpr.intro.intro.intro.intro
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U V : Set X
        hUV : HasSubset.Subset V U
        hxV : Membership.mem V x
        V_op : IsOpen V
        ⊢ Membership.mem (nhds x) U
      -/
      rw [mem_nhds_iff]
      /-
        case mpr.intro.intro.intro.intro
        X : Type u_1
        inst✝³ : TopologicalSpace X
        inst✝² : T2Space X
        inst✝¹ : CompactSpace X
        inst✝ : TotallyDisconnectedSpace X
        x : X
        U V : Set X
        hUV : HasSubset.Subset V U
        hxV : Membership.mem V x
        V_op : IsOpen V
        ⊢ Exists fun t => And (HasSubset.Subset t U) (And (IsOpen t) (Membership.mem t …
      -/
      exact ⟨V, hUV, V_op, hxV⟩⟩
      /-
        🎉 no goals
      -/


theorem isTopologicalBasis_isClopen : IsTopologicalBasis { s : Set X | IsClopen s } := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : T2Space X
    inst✝¹ : CompactSpace X
    inst✝ : TotallyDisconnectedSpace X
    ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun s => IsClopen s)
  -/
  apply isTopologicalBasis_of_isOpen_of_nhds fun U (hU : IsClopen U) => hU.2
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : T2Space X
    inst✝¹ : CompactSpace X
    inst✝ : TotallyDisconnectedSpace X
    ⊢ ∀ (a : X) (u : Set X), Membership.mem u a → IsOpen u → Exists fun v => And ( …
  -/
  intro x U hxU U_op
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : T2Space X
    inst✝¹ : CompactSpace X
    inst✝ : TotallyDisconnectedSpace X
    x : X
    U : Set X
    hxU : Membership.mem U x
    U_op : IsOpen U
    ⊢ Exists fun v => And (Membership.mem (fun U => And (IsClosed U) (IsOpen U)) v …
  -/
  have : U ∈ 𝓝 x := IsOpen.mem_nhds U_op hxU
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : T2Space X
    inst✝¹ : CompactSpace X
    inst✝ : TotallyDisconnectedSpace X
    x : X
    U : Set X
    hxU : Membership.mem U x
    U_op : IsOpen U
    this : Membership.mem (nhds x) U
    ⊢ Exists fun v => And (Membership.mem (fun U => And (IsClosed U) (IsOpen U)) v …
  -/
  rcases (nhds_basis_clopen x).mem_iff.mp this with ⟨V, ⟨hxV, hV⟩, hVU : V ⊆ U⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : T2Space X
    inst✝¹ : CompactSpace X
    inst✝ : TotallyDisconnectedSpace X
    x : X
    U : Set X
    hxU : Membership.mem U x
    U_op : IsOpen U
    this : Membership.mem (nhds x) U
    V : Set X
    hVU : HasSubset.Subset V U
    hxV : Membership.mem V x
    hV : IsClopen V
    ⊢ Exists fun v => And (Membership.mem (fun U => And (IsClosed U) (IsOpen U)) v …
  -/
  use V
  /-
    case h
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : T2Space X
    inst✝¹ : CompactSpace X
    inst✝ : TotallyDisconnectedSpace X
    x : X
    U : Set X
    hxU : Membership.mem U x
    U_op : IsOpen U
    this : Membership.mem (nhds x) U
    V : Set X
    hVU : HasSubset.Subset V U
    hxV : Membership.mem V x
    hV : IsClopen V
    ⊢ And (Membership.mem (fun U => And (IsClosed U) (IsOpen U)) V) (And (Membersh …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- Every member of an open set in a compact Hausdorff totally disconnected space
  is contained in a clopen set contained in the open set. -/
theorem compact_exists_isClopen_in_isOpen {x : X} {U : Set X} (is_open : IsOpen U) (memU : x ∈ U) :
    ∃ V : Set X, IsClopen V ∧ x ∈ V ∧ V ⊆ U :=
  isTopologicalBasis_isClopen.mem_nhds_iff.1 (is_open.mem_nhds memU)


/-- A locally compact Hausdorff totally disconnected space has a basis with clopen elements. -/
theorem loc_compact_Haus_tot_disc_of_zero_dim [TotallyDisconnectedSpace H] :
    IsTopologicalBasis { s : Set H | IsClopen s } := by
  /-
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun s => IsClopen s)
  -/
  refine isTopologicalBasis_of_isOpen_of_nhds (fun u hu => hu.2) fun x U memU hU => ?_
  /-
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    x : H
    U : Set H
    memU : Membership.mem U x
    hU : IsOpen U
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  obtain ⟨s, comp, xs, sU⟩ := exists_compact_subset hU memU
  /-
    case intro.intro.intro
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    x : H
    U : Set H
    memU : Membership.mem U x
    hU : IsOpen U
    s : Set H
    comp : IsCompact s
    xs : Membership.mem (interior s) x
    sU : HasSubset.Subset s U
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  let u : Set s := ((↑) : s → H) ⁻¹' interior s
  /-
    case intro.intro.intro
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    x : H
    U : Set H
    memU : Membership.mem U x
    hU : IsOpen U
    s : Set H
    comp : IsCompact s
    xs : Membership.mem (interior s) x
    sU : HasSubset.Subset s U
    u : Set ↑s := Set.preimage Subtype.val (interior s)
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  have u_open_in_s : IsOpen u := isOpen_interior.preimage continuous_subtype_val
  /-
    case intro.intro.intro
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    x : H
    U : Set H
    memU : Membership.mem U x
    hU : IsOpen U
    s : Set H
    comp : IsCompact s
    xs : Membership.mem (interior s) x
    sU : HasSubset.Subset s U
    u : Set ↑s := Set.preimage Subtype.val (interior s)
    u_open_in_s : IsOpen u
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  lift x to s using interior_subset xs
  /-
    case intro.intro.intro.intro
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    U : Set H
    hU : IsOpen U
    s : Set H
    comp : IsCompact s
    sU : HasSubset.Subset s U
    u : Set ↑s := Set.preimage Subtype.val (interior s)
    u_open_in_s : IsOpen u
    x : Subtype fun x => Membership.mem s x
    memU : Membership.mem U ↑x
    xs : Membership.mem (interior s) ↑x
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  haveI : CompactSpace s := isCompact_iff_compactSpace.1 comp
  /-
    case intro.intro.intro.intro
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    U : Set H
    hU : IsOpen U
    s : Set H
    comp : IsCompact s
    sU : HasSubset.Subset s U
    u : Set ↑s := Set.preimage Subtype.val (interior s)
    u_open_in_s : IsOpen u
    x : Subtype fun x => Membership.mem s x
    memU : Membership.mem U ↑x
    xs : Membership.mem (interior s) ↑x
    this : CompactSpace ↑s
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  obtain ⟨V : Set s, VisClopen, Vx, V_sub⟩ := compact_exists_isClopen_in_isOpen u_open_in_s xs
  have VisClopen' : IsClopen (((↑) : s → H) '' V) := by
    refine ⟨comp.isClosed.isClosedEmbedding_subtypeVal.isClosed_iff_image_isClosed.1 VisClopen.1,
      ?_⟩
    let v : Set u := ((↑) : u → s) ⁻¹' V
    have : ((↑) : u → H) = ((↑) : s → H) ∘ ((↑) : u → s) := rfl
    have f0 : IsEmbedding ((↑) : u → H) := IsEmbedding.subtypeVal.comp IsEmbedding.subtypeVal
    have f1 : IsOpenEmbedding ((↑) : u → H) := by
      refine ⟨f0, ?_⟩
      · have : Set.range ((↑) : u → H) = interior s := by
          rw [this, Set.range_comp, Subtype.range_coe, Subtype.image_preimage_coe]
          apply Set.inter_eq_self_of_subset_right interior_subset
        rw [this]
        apply isOpen_interior
    have f2 : IsOpen v := VisClopen.2.preimage continuous_subtype_val
    have f3 : ((↑) : s → H) '' V = ((↑) : u → H) '' v := by
      rw [this, image_comp, Subtype.image_preimage_coe, inter_eq_self_of_subset_right V_sub]
    rw [f3]
    apply f1.isOpenMap v f2
  /-
    case intro.intro.intro.intro.intro.intro.intro
    H : Type u_3
    inst✝³ : TopologicalSpace H
    inst✝² : LocallyCompactSpace H
    inst✝¹ : T2Space H
    inst✝ : TotallyDisconnectedSpace H
    U : Set H
    hU : IsOpen U
    s : Set H
    comp : IsCompact s
    sU : HasSubset.Subset s U
    u : Set ↑s := Set.preimage Subtype.val (interior s)
    u_open_in_s : IsOpen u
    x : Subtype fun x => Membership.mem s x
    memU : Membership.mem U ↑x
    xs : Membership.mem (interior s) ↑x
    this : CompactSpace ↑s
    V : Set ↑s
    VisClopen : IsClopen V
    Vx : Membership.mem V x
    V_sub : HasSubset.Subset V u
    VisClopen' : IsClopen (Set.image Subtype.val V)
    ⊢ Exists fun v => And (Membership.mem (setOf fun s => IsClopen s) v) (And (Mem …
  -/
  use (↑) '' V, VisClopen', by simp [Vx], Subset.trans (by simp) sU
  /-
    🎉 no goals
  -/


/-- A locally compact Hausdorff space is totally disconnected
  if and only if it is totally separated. -/
theorem loc_compact_t2_tot_disc_iff_tot_sep :
    TotallyDisconnectedSpace H ↔ TotallySeparatedSpace H := by
  /-
    H : Type u_3
    inst✝² : TopologicalSpace H
    inst✝¹ : LocallyCompactSpace H
    inst✝ : T2Space H
    ⊢ Iff (TotallyDisconnectedSpace H) (TotallySeparatedSpace H)
  -/
  constructor
    /-
      case mp
      H : Type u_3
      inst✝² : TopologicalSpace H
      inst✝¹ : LocallyCompactSpace H
      inst✝ : T2Space H
      ⊢ TotallyDisconnectedSpace H → TotallySeparatedSpace H
    -/
  · intro h
    /-
      case mp
      H : Type u_3
      inst✝² : TopologicalSpace H
      inst✝¹ : LocallyCompactSpace H
      inst✝ : T2Space H
      h : TotallyDisconnectedSpace H
      ⊢ TotallySeparatedSpace H
    -/
    exact totallySeparatedSpace_of_t1_of_basis_clopen loc_compact_Haus_tot_disc_of_zero_dim
    /-
      🎉 no goals
    -/
  /-
    case mpr
    H : Type u_3
    inst✝² : TopologicalSpace H
    inst✝¹ : LocallyCompactSpace H
    inst✝ : T2Space H
    ⊢ TotallySeparatedSpace H → TotallyDisconnectedSpace H
  -/
  apply TotallySeparatedSpace.totallyDisconnectedSpace
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-18")] alias compact_t2_tot_disc_iff_tot_sep :=
  loc_compact_t2_tot_disc_iff_tot_sep


/-- A totally disconnected compact Hausdorff space is totally separated. -/
instance (priority := 100) [TotallyDisconnectedSpace H] : TotallySeparatedSpace H :=
  loc_compact_t2_tot_disc_iff_tot_sep.mp inferInstance


