/-- The completion of a nonarchimedean additive group is a nonarchimedean additive group. -/
instance {G : Type*} [AddGroup G] [UniformSpace G] [UniformAddGroup G] [NonarchimedeanAddGroup G] :
    NonarchimedeanAddGroup (Completion G) where
  is_nonarchimedean := by
    /- Let `U` be a neighborhood of `0` in `Completion G`. We wish to show that `U` contains an open
    additive subgroup of `Completion G`. -/
    /-
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      ⊢ ∀ (U : Set (UniformSpace.Completion G)), Membership.mem (nhds 0) U → Exists  …
    -/
    intro U hU
    /- Since `Completion G` is regular, there is a closed neighborhood `C` of `0` which is
    contained in `U`. -/
    /-
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      U : Set (UniformSpace.Completion G)
      hU : Membership.mem (nhds 0) U
      ⊢ Exists fun V => HasSubset.Subset (↑V) U
    -/
    obtain ⟨C, ⟨hC, C_closed⟩, C_subset_U⟩ := (closed_nhds_basis 0).mem_iff.mp hU
    /- By continuity, the preimage of `C` in `G`, written `toCompl ⁻¹' U'`,
    is a neighborhood of `0`. -/
    have : toCompl ⁻¹' C ∈ 𝓝 0 :=
      continuous_toCompl.continuousAt.preimage_mem_nhds (by rwa [map_zero])
    /- Therefore, since `G` is nonarchimedean, there exists an open subgroup `W` of `G` that is
    contained within `toCompl ⁻¹' C`. -/
    /-
      case intro.intro.intro
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      U : Set (UniformSpace.Completion G)
      hU : Membership.mem (nhds 0) U
      C : Set (UniformSpace.Completion G)
      C_subset_U : HasSubset.Subset (id C) U
      hC : Membership.mem (nhds 0) C
      C_closed : IsClosed C
      this : Membership.mem (nhds 0) (Set.preimage (⇑UniformSpace.Completion.toCompl …
      ⊢ Exists fun V => HasSubset.Subset (↑V) U
    -/
    obtain ⟨W, hCW⟩ := NonarchimedeanAddGroup.is_nonarchimedean (toCompl ⁻¹' C) this
    /- Now, let `V = (W.map toCompl).topologicalClosure` be the result of mapping `W` back to
    `Completion G` and taking the topological closure. -/
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      U : Set (UniformSpace.Completion G)
      hU : Membership.mem (nhds 0) U
      C : Set (UniformSpace.Completion G)
      C_subset_U : HasSubset.Subset (id C) U
      hC : Membership.mem (nhds 0) C
      C_closed : IsClosed C
      this : Membership.mem (nhds 0) (Set.preimage (⇑UniformSpace.Completion.toCompl …
      W : OpenAddSubgroup G
      hCW : HasSubset.Subset (↑W) (Set.preimage (⇑UniformSpace.Completion.toCompl) C)
      ⊢ Exists fun V => HasSubset.Subset (↑V) U
    -/
    let V : Set (Completion G) := (W.map toCompl).topologicalClosure
    /- We claim that this set `V` satisfies the
    desired properties. There are three conditions to check:

    1. `V` is a subgroup of `Completion G`.
    2. `V` is open.
    3. `V ⊆ U`.

    The first condition follows directly from the fact that the topological closure of a subgroup
    is a subgroup. Now, let us check that `V` is open. -/
    have : IsOpen V := by
      /- Since `V` is a subgroup of `Completion G`, it suffices to show that it is a neighborhood of
      `0` in `Completion G`. This follows from the fact that `toCompl : G → Completion G` is dense
      inducing and `W` is a neighborhood of `0` in `G`. -/
      apply isOpen_of_mem_nhds (g := 0)
      apply (isDenseInducing_toCompl _).closure_image_mem_nhds
      exact mem_nhds_zero W
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      U : Set (UniformSpace.Completion G)
      hU : Membership.mem (nhds 0) U
      C : Set (UniformSpace.Completion G)
      C_subset_U : HasSubset.Subset (id C) U
      hC : Membership.mem (nhds 0) C
      C_closed : IsClosed C
      this✝ : Membership.mem (nhds 0) (Set.preimage (⇑UniformSpace.Completion.toComp …
      W : OpenAddSubgroup G
      hCW : HasSubset.Subset (↑W) (Set.preimage (⇑UniformSpace.Completion.toCompl) C)
      V : Set (UniformSpace.Completion G) := ↑(AddSubgroup.map UniformSpace.Completi …
      this : IsOpen V
      ⊢ Exists fun V => HasSubset.Subset (↑V) U
    -/
    use ⟨_, this⟩
    /- Finally, it remains to show that `V ⊆ U`. It suffices to show that `V ⊆ C`, which
    follows from the fact that `W ⊆ toCompl ⁻¹' C` and `C` is closed. -/
    /-
      case h
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      U : Set (UniformSpace.Completion G)
      hU : Membership.mem (nhds 0) U
      C : Set (UniformSpace.Completion G)
      C_subset_U : HasSubset.Subset (id C) U
      hC : Membership.mem (nhds 0) C
      C_closed : IsClosed C
      this✝ : Membership.mem (nhds 0) (Set.preimage (⇑UniformSpace.Completion.toComp …
      W : OpenAddSubgroup G
      hCW : HasSubset.Subset (↑W) (Set.preimage (⇑UniformSpace.Completion.toCompl) C)
      V : Set (UniformSpace.Completion G) := ↑(AddSubgroup.map UniformSpace.Completi …
      this : IsOpen V
      ⊢ HasSubset.Subset (↑{ toAddSubgroup := (AddSubgroup.map UniformSpace.Completi …
    -/
    suffices V ⊆ C from this.trans C_subset_U
    /-
      case h
      G : Type u_1
      inst✝³ : AddGroup G
      inst✝² : UniformSpace G
      inst✝¹ : UniformAddGroup G
      inst✝ : NonarchimedeanAddGroup G
      U : Set (UniformSpace.Completion G)
      hU : Membership.mem (nhds 0) U
      C : Set (UniformSpace.Completion G)
      C_subset_U : HasSubset.Subset (id C) U
      hC : Membership.mem (nhds 0) C
      C_closed : IsClosed C
      this✝ : Membership.mem (nhds 0) (Set.preimage (⇑UniformSpace.Completion.toComp …
      W : OpenAddSubgroup G
      hCW : HasSubset.Subset (↑W) (Set.preimage (⇑UniformSpace.Completion.toCompl) C)
      V : Set (UniformSpace.Completion G) := ↑(AddSubgroup.map UniformSpace.Completi …
      this : IsOpen V
      ⊢ HasSubset.Subset V C
    -/
    exact closure_minimal (Set.image_subset_iff.mpr hCW) C_closed
    /-
      🎉 no goals
    -/


/-- The completion of a nonarchimedean ring is a nonarchimedean ring. -/
instance {R : Type*} [Ring R] [UniformSpace R] [TopologicalRing R] [UniformAddGroup R]
    [NonarchimedeanRing R] :
    NonarchimedeanRing (Completion R) where
  is_nonarchimedean := NonarchimedeanAddGroup.is_nonarchimedean

