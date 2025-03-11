theorem exist_openNormalSubgroup_sub_clopen_nhd_of_one {G : Type*} [Group G] [TopologicalSpace G]
    [TopologicalGroup G] [CompactSpace G] {W : Set G} (WClopen : IsClopen W) (einW : 1 ∈ W) :
    ∃ H : OpenNormalSubgroup G, (H : Set G) ⊆ W := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    einW : Membership.mem W 1
    ⊢ Exists fun H => HasSubset.Subset (↑H) W
  -/
  rcases exist_openSubgroup_sub_clopen_nhd_of_one WClopen einW with ⟨H, hH⟩
  /-
    case intro
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    einW : Membership.mem W 1
    H : OpenSubgroup G
    hH : HasSubset.Subset (↑H) W
    ⊢ Exists fun H => HasSubset.Subset (↑H) W
  -/
  have : Subgroup.FiniteIndex H.toSubgroup := H.finiteIndex_of_finite_quotient
  use { toSubgroup := Subgroup.normalCore H
        isOpen' := Subgroup.isOpen_of_isClosed_of_finiteIndex _ (H.normalCore_isClosed H.isClosed) }
  /-
    case h
    G : Type u_1
    inst✝³ : Group G
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalGroup G
    inst✝ : CompactSpace G
    W : Set G
    WClopen : IsClopen W
    einW : Membership.mem W 1
    H : OpenSubgroup G
    hH : HasSubset.Subset (↑H) W
    this : (↑H).FiniteIndex
    ⊢ HasSubset.Subset (↑{ toSubgroup := (↑H).normalCore, isOpen' := ⋯, isNormal'  …
  -/
  exact fun _ b ↦ hH (H.normalCore_le b)
  /-
    🎉 no goals
  -/


