@[to_additive]
theorem orbit.eq_or_disjoint (a b : X) :
    orbit G a = orbit G b ∨ Disjoint (orbit G a) (orbit G b) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    a b : X
    ⊢ Or (Eq (MulAction.orbit G a) (MulAction.orbit G b)) (Disjoint (MulAction.orb …
  -/
  apply (em (Disjoint (orbit G a) (orbit G b))).symm.imp _ id
  simp +contextual
    only [Set.not_disjoint_iff, ← orbit_eq_iff, forall_exists_index, and_imp, eq_comm, implies_true]


@[to_additive]
theorem orbit.pairwiseDisjoint :
    (Set.range fun x : X => orbit G x).PairwiseDisjoint id := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    ⊢ (Set.range fun x => MulAction.orbit G x).PairwiseDisjoint id
  -/
  rintro s ⟨x, rfl⟩ t ⟨y, rfl⟩ h
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    x y : X
    h : Ne ((fun x => MulAction.orbit G x) x) ((fun x => MulAction.orbit G x) y)
    ⊢ Function.onFun Disjoint id ((fun x => MulAction.orbit G x) x) ((fun x => Mul …
  -/
  contrapose! h
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    x y : X
    h : Not (Function.onFun Disjoint id (MulAction.orbit G x) (MulAction.orbit G y))
    ⊢ Eq (MulAction.orbit G x) (MulAction.orbit G y)
  -/
  exact (orbit.eq_or_disjoint x y).resolve_right h
  /-
    🎉 no goals
  -/


/-- Orbits of an element form a partition -/
@[to_additive]
theorem IsPartition.of_orbits :
    Setoid.IsPartition (Set.range fun a : X => orbit G a) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    ⊢ Setoid.IsPartition (Set.range fun a => MulAction.orbit G a)
  -/
  apply orbit.pairwiseDisjoint.isPartition_of_exists_of_ne_empty
    /-
      case h₂
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      ⊢ ∀ (a : X), Exists fun x => And (Membership.mem (Set.range fun x => MulAction …
    -/
  · intro x
    /-
      case h₂
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      x : X
      ⊢ Exists fun x_1 => And (Membership.mem (Set.range fun x => MulAction.orbit G  …
    -/
    exact ⟨_, ⟨x, rfl⟩, mem_orbit_self x⟩
    /-
      🎉 no goals
    -/
    /-
      case h₃
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      ⊢ Not (Membership.mem (Set.range fun x => MulAction.orbit G x) EmptyCollection …
    -/
  · rintro ⟨a, ha : orbit G a = ∅⟩
    /-
      case h₃.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      ha : Eq (MulAction.orbit G a) EmptyCollection.emptyCollection
      ⊢ False
    -/
    exact (MulAction.orbit_nonempty a).ne_empty ha
    /-
      🎉 no goals
    -/


/-- A set `B` is a `G`-fixed block if `g • B = B` for all `g : G`. -/
@[to_additive "A set `B` is a `G`-fixed block if `g +ᵥ B = B` for all `g : G`."]
def IsFixedBlock (B : Set X) := ∀ g : G, g • B = B


/-- A set `B` is a `G`-invariant block if `g • B ⊆ B` for all `g : G`.

Note: It is not necessarily a block when the action is not by a group. -/
@[to_additive
"A set `B` is a `G`-invariant block if `g +ᵥ B ⊆ B` for all `g : G`.

Note: It is not necessarily a block when the action is not by a group. "]
def IsInvariantBlock (B : Set X) := ∀ g : G, g • B ⊆ B


/-- A trivial block is a `Set X` which is either a subsingleton or `univ`.

Note: It is not necessarily a block when the action is not by a group. -/
@[to_additive
"A trivial block is a `Set X` which is either a subsingleton or `univ`.

Note: It is not necessarily a block when the action is not by a group."]
def IsTrivialBlock (B : Set X) := B.Subsingleton ∨ B = univ


/-- A set `B` is a `G`-block iff the sets of the form `g • B` are pairwise equal or disjoint. -/
@[to_additive
"A set `B` is a `G`-block iff the sets of the form `g +ᵥ B` are pairwise equal or disjoint. "]
def IsBlock (B : Set X) := ∀ ⦃g₁ g₂ : G⦄, g₁ • B ≠ g₂ • B → Disjoint (g₁ • B) (g₂ • B)


@[to_additive]
lemma isBlock_iff_smul_eq_smul_of_nonempty :
    IsBlock G B ↔ ∀ ⦃g₁ g₂ : G⦄, (g₁ • B ∩ g₂ • B).Nonempty → g₁ • B = g₂ • B := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝ : SMul G X
    B : Set X
    ⊢ Iff (MulAction.IsBlock G B) (∀ ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ B)  …
  -/
  simp_rw [IsBlock, ← not_disjoint_iff_nonempty_inter, not_imp_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isBlock_iff_pairwiseDisjoint_range_smul :
    IsBlock G B ↔ (range fun g : G ↦ g • B).PairwiseDisjoint id := pairwiseDisjoint_range_iff.symm


@[to_additive]
lemma isBlock_iff_smul_eq_smul_or_disjoint :
    IsBlock G B ↔ ∀ g₁ g₂ : G, g₁ • B = g₂ • B ∨ Disjoint (g₁ • B) (g₂ • B) :=
  forall₂_congr fun _ _ ↦ or_iff_not_imp_left.symm


@[to_additive]
lemma IsBlock.smul_eq_smul_of_subset (hB : IsBlock G B) (hg : g₁ • B ⊆ g₂ • B) :
    g₁ • B = g₂ • B := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝ : SMul G X
    B : Set X
    g₁ g₂ : G
    hB : MulAction.IsBlock G B
    hg : HasSubset.Subset (HSMul.hSMul g₁ B) (HSMul.hSMul g₂ B)
    ⊢ Eq (HSMul.hSMul g₁ B) (HSMul.hSMul g₂ B)
  -/
  by_contra! hg'
  /-
    G : Type u_1
    X : Type u_2
    inst✝ : SMul G X
    B : Set X
    g₁ g₂ : G
    hB : MulAction.IsBlock G B
    hg : HasSubset.Subset (HSMul.hSMul g₁ B) (HSMul.hSMul g₂ B)
    hg' : Ne (HSMul.hSMul g₁ B) (HSMul.hSMul g₂ B)
    ⊢ False
  -/
  obtain rfl : B = ∅ := by simpa using (hB hg').eq_bot_of_le hg
  /-
    G : Type u_1
    X : Type u_2
    inst✝ : SMul G X
    g₁ g₂ : G
    hB : MulAction.IsBlock G EmptyCollection.emptyCollection
    hg : HasSubset.Subset (HSMul.hSMul g₁ EmptyCollection.emptyCollection) (HSMul. …
    hg' : Ne (HSMul.hSMul g₁ EmptyCollection.emptyCollection) (HSMul.hSMul g₂ Empt …
    ⊢ False
  -/
  simp at hg'
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsBlock.not_smul_set_ssubset_smul_set (hB : IsBlock G B) : ¬ g₁ • B ⊂ g₂ • B :=
  fun hab ↦ hab.ne <| hB.smul_eq_smul_of_subset hab.subset


@[to_additive]
lemma IsBlock.disjoint_smul_set_smul (hB : IsBlock G B) (hgs : ¬ g • B ⊆ s • B) :
    Disjoint (g • B) (s • B) := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝ : SMul G X
    B : Set X
    s : Set G
    g : G
    hB : MulAction.IsBlock G B
    hgs : Not (HasSubset.Subset (HSMul.hSMul g B) (HSMul.hSMul s B))
    ⊢ Disjoint (HSMul.hSMul g B) (HSMul.hSMul s B)
  -/
  rw [← iUnion_smul_set, disjoint_iUnion₂_right]
  /-
    G : Type u_1
    X : Type u_2
    inst✝ : SMul G X
    B : Set X
    s : Set G
    g : G
    hB : MulAction.IsBlock G B
    hgs : Not (HasSubset.Subset (HSMul.hSMul g B) (HSMul.hSMul s B))
    ⊢ ∀ (i : G), Membership.mem s i → Disjoint (HSMul.hSMul g B) (HSMul.hSMul i B)
  -/
  exact fun b hb ↦ hB fun h ↦ hgs <| h.trans_subset <| smul_set_subset_smul hb
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsBlock.disjoint_smul_smul_set (hB : IsBlock G B) (hgs : ¬ g • B ⊆ s • B) :
    Disjoint (s • B) (g • B) := (hB.disjoint_smul_set_smul hgs).symm


alias ⟨IsBlock.smul_eq_smul_of_nonempty, _⟩ := isBlock_iff_smul_eq_smul_of_nonempty

alias ⟨IsBlock.pairwiseDisjoint_range_smul, _⟩ := isBlock_iff_pairwiseDisjoint_range_smul

alias ⟨IsBlock.smul_eq_smul_or_disjoint, _⟩ := isBlock_iff_smul_eq_smul_or_disjoint


/-- A fixed block is a block. -/
@[to_additive "A fixed block is a block."]
                                                                        /-
                                                                          G : Type u_1
                                                                          X : Type u_2
                                                                          inst✝ : SMul G X
                                                                          B : Set X
                                                                          hfB : MulAction.IsFixedBlock G B
                                                                          ⊢ MulAction.IsBlock G B
                                                                        -/
lemma IsFixedBlock.isBlock (hfB : IsFixedBlock G B) : IsBlock G B := by simp [IsBlock, hfB _]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The empty set is a block. -/
@[to_additive (attr := simp) "The empty set is a block."]
                                                  /-
                                                    G : Type u_1
                                                    X : Type u_2
                                                    inst✝ : SMul G X
                                                    ⊢ MulAction.IsBlock G EmptyCollection.emptyCollection
                                                  -/
lemma IsBlock.empty : IsBlock G (∅ : Set X) := by simp [IsBlock]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- A singleton is a block. -/
@[to_additive "A singleton is a block."]
                                                        /-
                                                          G : Type u_1
                                                          X : Type u_2
                                                          inst✝ : SMul G X
                                                          a : X
                                                          ⊢ MulAction.IsBlock G (Singleton.singleton a)
                                                        -/
lemma IsBlock.singleton : IsBlock G ({a} : Set X) := by simp [IsBlock]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Subsingletons are (trivial) blocks. -/
@[to_additive "Subsingletons are (trivial) blocks."]
lemma IsBlock.of_subsingleton (hB : B.Subsingleton) : IsBlock G B :=
  hB.induction_on .empty fun _ ↦ .singleton


/-- A fixed block is an invariant block. -/
@[to_additive "A fixed block is an invariant block."]
lemma IsFixedBlock.isInvariantBlock (hB : IsFixedBlock G B) : IsInvariantBlock G B :=
  fun _ ↦ (hB _).le


@[to_additive]
lemma IsBlock.disjoint_smul_right (hB : IsBlock M B) (hs : ¬ B ⊆ s • B) : Disjoint B (s • B) := by
  /-
    M : Type u_1
    X : Type u_2
    inst✝¹ : Monoid M
    inst✝ : MulAction M X
    B : Set X
    s : Set M
    hB : MulAction.IsBlock M B
    hs : Not (HasSubset.Subset B (HSMul.hSMul s B))
    ⊢ Disjoint B (HSMul.hSMul s B)
  -/
  simpa using hB.disjoint_smul_set_smul (g := 1) (by simpa using hs)
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsBlock.disjoint_smul_left (hB : IsBlock M B) (hs : ¬ B ⊆ s • B) : Disjoint (s • B) B :=
  (hB.disjoint_smul_right hs).symm


@[to_additive]
lemma isBlock_iff_disjoint_smul_of_ne :
    IsBlock G B ↔ ∀ ⦃g : G⦄, g • B ≠ B → Disjoint (g • B) B := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    ⊢ Iff (MulAction.IsBlock G B) (∀ ⦃g : G⦄, Ne (HSMul.hSMul g B) B → Disjoint (H …
  -/
  refine ⟨fun hB g ↦ by simpa using hB (g₂ := 1), fun hB g₁ g₂ h ↦ ?_⟩
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    hB : ∀ ⦃g : G⦄, Ne (HSMul.hSMul g B) B → Disjoint (HSMul.hSMul g B) B
    g₁ g₂ : G
    h : Ne (HSMul.hSMul g₁ B) (HSMul.hSMul g₂ B)
    ⊢ Disjoint (HSMul.hSMul g₁ B) (HSMul.hSMul g₂ B)
  -/
  simp only [disjoint_smul_set_right, ne_eq, ← inv_smul_eq_iff, smul_smul] at h ⊢
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    hB : ∀ ⦃g : G⦄, Ne (HSMul.hSMul g B) B → Disjoint (HSMul.hSMul g B) B
    g₁ g₂ : G
    h : Not (Eq (HSMul.hSMul (HMul.hMul (Inv.inv g₂) g₁) B) B)
    ⊢ Disjoint (HSMul.hSMul (HMul.hMul (Inv.inv g₂) g₁) B) B
  -/
  exact hB h
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isBlock_iff_smul_eq_of_nonempty :
    IsBlock G B ↔ ∀ ⦃g : G⦄, (g • B ∩ B).Nonempty → g • B = B := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    ⊢ Iff (MulAction.IsBlock G B) (∀ ⦃g : G⦄, (Inter.inter (HSMul.hSMul g B) B).No …
  -/
  simp_rw [isBlock_iff_disjoint_smul_of_ne, ← not_disjoint_iff_nonempty_inter, not_imp_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isBlock_iff_smul_eq_or_disjoint :
    IsBlock G B ↔ ∀ g : G, g • B = B ∨ Disjoint (g • B) B :=
  isBlock_iff_disjoint_smul_of_ne.trans <| forall_congr' fun _ ↦ or_iff_not_imp_left.symm


@[to_additive]
lemma isBlock_iff_smul_eq_of_mem :
    IsBlock G B ↔ ∀ ⦃g : G⦄ ⦃a : X⦄, a ∈ B → g • a ∈ B → g • B = B := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    ⊢ Iff (MulAction.IsBlock G B) (∀ ⦃g : G⦄ ⦃a : X⦄, Membership.mem B a → Members …
  -/
  simp [isBlock_iff_smul_eq_of_nonempty, Set.Nonempty, mem_smul_set]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨IsBlock.disjoint_smul_of_ne, _⟩ := isBlock_iff_disjoint_smul_of_ne

@[to_additive] alias ⟨IsBlock.smul_eq_of_nonempty, _⟩ := isBlock_iff_smul_eq_of_nonempty

@[to_additive] alias ⟨IsBlock.smul_eq_or_disjoint, _⟩ := isBlock_iff_smul_eq_or_disjoint

@[to_additive] alias ⟨IsBlock.smul_eq_of_mem, _⟩ := isBlock_iff_smul_eq_of_mem

-- TODO: Generalise to `SubgroupClass`

/-- If `B` is a `G`-block, then it is also a `H`-block for any subgroup `H` of `G`. -/
@[to_additive
"If `B` is a `G`-block, then it is also a `H`-block for any subgroup `H` of `G`."]
lemma IsBlock.subgroup {H : Subgroup G} (hB : IsBlock G B) : IsBlock H B := fun _ _ h ↦ hB h


/-- A block of a group action is invariant iff it is fixed. -/
@[to_additive "A block of a group action is invariant iff it is fixed."]
lemma isInvariantBlock_iff_isFixedBlock : IsInvariantBlock G B ↔ IsFixedBlock G B :=
  ⟨fun hB g ↦ (hB g).antisymm <| subset_set_smul_iff.2 <| hB _, IsFixedBlock.isInvariantBlock⟩


/-- An invariant block of a group action is a fixed block. -/
@[to_additive "An invariant block of a group action is a fixed block."]
alias ⟨IsInvariantBlock.isFixedBlock, _⟩ := isInvariantBlock_iff_isFixedBlock


/-- An invariant block  of a group action is a block. -/
@[to_additive "An invariant block of a group action is a block."]
lemma IsInvariantBlock.isBlock (hB : IsInvariantBlock G B) : IsBlock G B := hB.isFixedBlock.isBlock


/-- The full set is a fixed block. -/
@[to_additive "The full set is a fixed block."]
                                                                      /-
                                                                        G : Type u_1
                                                                        inst✝¹ : Group G
                                                                        X : Type u_2
                                                                        inst✝ : MulAction G X
                                                                        x✝ : G
                                                                        ⊢ Eq (HSMul.hSMul x✝ Set.univ) Set.univ
                                                                      -/
lemma IsFixedBlock.univ : IsFixedBlock G (univ : Set X) := fun _ ↦ by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The full set is a block. -/
@[to_additive (attr := simp) "The full set is a block."]
lemma IsBlock.univ : IsBlock G (univ : Set X) := IsFixedBlock.univ.isBlock


/-- The intersection of two blocks is a block. -/
@[to_additive "The intersection of two blocks is a block."]
lemma IsBlock.inter {B₁ B₂ : Set X} (h₁ : IsBlock G B₁) (h₂ : IsBlock G B₂) :
    IsBlock G (B₁ ∩ B₂) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B₁ B₂ : Set X
    h₁ : MulAction.IsBlock G B₁
    h₂ : MulAction.IsBlock G B₂
    ⊢ MulAction.IsBlock G (Inter.inter B₁ B₂)
  -/
  simp only [isBlock_iff_smul_eq_smul_of_nonempty, smul_set_inter] at h₁ h₂ ⊢
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B₁ B₂ : Set X
    h₁ : ∀ ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ B₁) (HSMul.hSMul g₂ B₁)).None …
    h₂ : ∀ ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ B₂) (HSMul.hSMul g₂ B₂)).None …
    ⊢ ∀ ⦃g₁ g₂ : G⦄, (Inter.inter (Inter.inter (HSMul.hSMul g₁ B₁) (HSMul.hSMul g₁ …
  -/
  rintro g₁ g₂ ⟨a, ha₁, ha₂⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B₁ B₂ : Set X
    h₁ : ∀ ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ B₁) (HSMul.hSMul g₂ B₁)).None …
    h₂ : ∀ ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ B₂) (HSMul.hSMul g₂ B₂)).None …
    g₁ g₂ : G
    a : X
    ha₁ : Membership.mem (Inter.inter (HSMul.hSMul g₁ B₁) (HSMul.hSMul g₁ B₂)) a
    ha₂ : Membership.mem (Inter.inter (HSMul.hSMul g₂ B₁) (HSMul.hSMul g₂ B₂)) a
    ⊢ Eq (Inter.inter (HSMul.hSMul g₁ B₁) (HSMul.hSMul g₁ B₂)) (Inter.inter (HSMul …
  -/
  rw [h₁ ⟨a, ha₁.1, ha₂.1⟩, h₂ ⟨a, ha₁.2, ha₂.2⟩]
  /-
    🎉 no goals
  -/


/-- An intersection of blocks is a block. -/
@[to_additive "An intersection of blocks is a block."]
lemma IsBlock.iInter {ι : Sort*} {B : ι → Set X} (hB : ∀ i, IsBlock G (B i)) :
    IsBlock G (⋂ i, B i) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    ι : Sort u_3
    B : ι → Set X
    hB : ∀ (i : ι), MulAction.IsBlock G (B i)
    ⊢ MulAction.IsBlock G (Set.iInter fun i => B i)
  -/
  simp only [isBlock_iff_smul_eq_smul_of_nonempty, smul_set_iInter] at hB ⊢
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    ι : Sort u_3
    B : ι → Set X
    hB : ∀ (i : ι) ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ (B i)) (HSMul.hSMul g …
    ⊢ ∀ ⦃g₁ g₂ : G⦄, (Inter.inter (Set.iInter fun i => HSMul.hSMul g₁ (B i)) (Set. …
  -/
  rintro g₁ g₂ ⟨a, ha₁, ha₂⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    ι : Sort u_3
    B : ι → Set X
    hB : ∀ (i : ι) ⦃g₁ g₂ : G⦄, (Inter.inter (HSMul.hSMul g₁ (B i)) (HSMul.hSMul g …
    g₁ g₂ : G
    a : X
    ha₁ : Membership.mem (Set.iInter fun i => HSMul.hSMul g₁ (B i)) a
    ha₂ : Membership.mem (Set.iInter fun i => HSMul.hSMul g₂ (B i)) a
    ⊢ Eq (Set.iInter fun i => HSMul.hSMul g₁ (B i)) (Set.iInter fun i => HSMul.hSM …
  -/
  simp_rw [fun i ↦ hB i ⟨a, iInter_subset _ i ha₁, iInter_subset _ i ha₂⟩]
  /-
    🎉 no goals
  -/


/-- A trivial block is a block. -/
@[to_additive "A trivial block is a block."]
lemma IsTrivialBlock.isBlock (hB : IsTrivialBlock B) : IsBlock G B := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    hB : MulAction.IsTrivialBlock B
    ⊢ MulAction.IsBlock G B
  -/
  obtain hB | rfl := hB
    /-
      case inl
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hB : B.Subsingleton
      ⊢ MulAction.IsBlock G B
    -/
  · exact .of_subsingleton hB
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      ⊢ MulAction.IsBlock G Set.univ
    -/
  · exact .univ
    /-
      🎉 no goals
    -/


/-- An orbit is a fixed block. -/
@[to_additive "An orbit is a fixed block."]
protected lemma IsFixedBlock.orbit (a : X) : IsFixedBlock G (orbit G a) := (smul_orbit · a)


/-- An orbit is a block. -/
@[to_additive "An orbit is a block."]
protected lemma IsBlock.orbit (a : X) : IsBlock G (orbit G a) := (IsFixedBlock.orbit a).isBlock


@[to_additive]
lemma isBlock_top : IsBlock (⊤ : Subgroup G) B ↔ IsBlock G B :=
  Subgroup.topEquiv.toEquiv.forall_congr fun _ ↦ Subgroup.topEquiv.toEquiv.forall_congr_left


lemma IsBlock.preimage {H Y : Type*} [Group H] [MulAction H Y]
    {φ : H → G} (j : Y →ₑ[φ] X) (hB : IsBlock G B) :
    IsBlock H (j ⁻¹' B) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    X : Type u_2
    inst✝² : MulAction G X
    B : Set X
    H : Type u_3
    Y : Type u_4
    inst✝¹ : Group H
    inst✝ : MulAction H Y
    φ : H → G
    j : MulActionHom φ Y X
    hB : MulAction.IsBlock G B
    ⊢ MulAction.IsBlock H (Set.preimage (⇑j) B)
  -/
  rintro g₁ g₂ hg
  /-
    G : Type u_1
    inst✝³ : Group G
    X : Type u_2
    inst✝² : MulAction G X
    B : Set X
    H : Type u_3
    Y : Type u_4
    inst✝¹ : Group H
    inst✝ : MulAction H Y
    φ : H → G
    j : MulActionHom φ Y X
    hB : MulAction.IsBlock G B
    g₁ g₂ : H
    hg : Ne (HSMul.hSMul g₁ (Set.preimage (⇑j) B)) (HSMul.hSMul g₂ (Set.preimage ( …
    ⊢ Disjoint (HSMul.hSMul g₁ (Set.preimage (⇑j) B)) (HSMul.hSMul g₂ (Set.preimag …
  -/
  rw [← Group.preimage_smul_setₛₗ, ← Group.preimage_smul_setₛₗ] at hg ⊢
  /-
    G : Type u_1
    inst✝³ : Group G
    X : Type u_2
    inst✝² : MulAction G X
    B : Set X
    H : Type u_3
    Y : Type u_4
    inst✝¹ : Group H
    inst✝ : MulAction H Y
    φ : H → G
    j : MulActionHom φ Y X
    hB : MulAction.IsBlock G B
    g₁ g₂ : H
    hg : Ne (Set.preimage (⇑j) (HSMul.hSMul (φ g₁) B)) (Set.preimage (⇑j) (HSMul.h …
    ⊢ Disjoint (Set.preimage (⇑j) (HSMul.hSMul (φ g₁) B)) (Set.preimage (⇑j) (HSMu …
  -/
  exact (hB <| ne_of_apply_ne _ hg).preimage _
  /-
    🎉 no goals
  -/


theorem IsBlock.image {H Y : Type*} [Group H] [MulAction H Y]
    {φ : G →* H} (j : X →ₑ[φ] Y)
    (hφ : Function.Surjective φ) (hj : Function.Injective j)
    (hB : IsBlock G B) :
    IsBlock H (j '' B) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    X : Type u_2
    inst✝² : MulAction G X
    B : Set X
    H : Type u_3
    Y : Type u_4
    inst✝¹ : Group H
    inst✝ : MulAction H Y
    φ : MonoidHom G H
    j : MulActionHom (⇑φ) X Y
    hφ : Function.Surjective ⇑φ
    hj : Function.Injective ⇑j
    hB : MulAction.IsBlock G B
    ⊢ MulAction.IsBlock H (Set.image (⇑j) B)
  -/
  simp only [IsBlock, hφ.forall, ← image_smul_setₛₗ]
  /-
    G : Type u_1
    inst✝³ : Group G
    X : Type u_2
    inst✝² : MulAction G X
    B : Set X
    H : Type u_3
    Y : Type u_4
    inst✝¹ : Group H
    inst✝ : MulAction H Y
    φ : MonoidHom G H
    j : MulActionHom (⇑φ) X Y
    hφ : Function.Surjective ⇑φ
    hj : Function.Injective ⇑j
    hB : MulAction.IsBlock G B
    ⊢ ∀ (x x_1 : G), Ne (Set.image (⇑j) (HSMul.hSMul x B)) (Set.image (⇑j) (HSMul. …
  -/
  exact fun g₁ g₂ hg ↦ disjoint_image_of_injective hj <| hB <| ne_of_apply_ne _ hg
  /-
    🎉 no goals
  -/


theorem IsBlock.subtype_val_preimage {C : SubMulAction G X} (hB : IsBlock G B) :
    IsBlock G (Subtype.val ⁻¹' B : Set C) :=
  hB.preimage C.inclusion


theorem isBlock_subtypeVal {C : SubMulAction G X} {B : Set C} :
    IsBlock G (Subtype.val '' B : Set X) ↔ IsBlock G B := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    C : SubMulAction G X
    B : Set (Subtype fun x => Membership.mem C x)
    ⊢ Iff (MulAction.IsBlock G (Set.image Subtype.val B)) (MulAction.IsBlock G B)
  -/
  refine forall₂_congr fun g₁ g₂ ↦ ?_
  rw [← SubMulAction.inclusion.coe_eq, ← image_smul_set, ← image_smul_set, ne_eq,
    Set.image_eq_image C.inclusion_injective, disjoint_image_iff C.inclusion_injective]


theorem IsBlock.of_subgroup_of_conjugate {H : Subgroup G} (hB : IsBlock H B) (g : G) :
    IsBlock (Subgroup.map (MulEquiv.toMonoidHom (MulAut.conj g)) H) (g • B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    H : Subgroup G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem H x) B
    g : G
    ⊢ MulAction.IsBlock (Subtype fun x => Membership.mem (Subgroup.map (MulEquiv.t …
  -/
  rw [isBlock_iff_smul_eq_or_disjoint]
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    H : Subgroup G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem H x) B
    g : G
    ⊢ ∀ (g_1 : Subtype fun x => Membership.mem (Subgroup.map (MulEquiv.toMonoidHom …
  -/
  intro h'
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    H : Subgroup G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem H x) B
    g : G
    h' : Subtype fun x => Membership.mem (Subgroup.map (MulEquiv.toMonoidHom (MulA …
    ⊢ Or (Eq (HSMul.hSMul h' (HSMul.hSMul g B)) (HSMul.hSMul g B)) (Disjoint (HSMu …
  -/
  obtain ⟨h, hH, hh⟩ := Subgroup.mem_map.mp (SetLike.coe_mem h')
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    H : Subgroup G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem H x) B
    g : G
    h' : Subtype fun x => Membership.mem (Subgroup.map (MulEquiv.toMonoidHom (MulA …
    h : G
    hH : Membership.mem H h
    hh : Eq ((MulEquiv.toMonoidHom (MulAut.conj g)) h) ↑h'
    ⊢ Or (Eq (HSMul.hSMul h' (HSMul.hSMul g B)) (HSMul.hSMul g B)) (Disjoint (HSMu …
  -/
  simp only [MulEquiv.coe_toMonoidHom, MulAut.conj_apply] at hh
  suffices h' • g • B = g • h • B by
    simp only [this]
    apply (hB.smul_eq_or_disjoint ⟨h, hH⟩).imp
    · intro; congr
    · exact Set.disjoint_image_of_injective (MulAction.injective g)
  suffices (h' : G) • g • B = g • h • B by
    rw [← this]; rfl
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    H : Subgroup G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem H x) B
    g : G
    h' : Subtype fun x => Membership.mem (Subgroup.map (MulEquiv.toMonoidHom (MulA …
    h : G
    hH : Membership.mem H h
    hh : Eq (HMul.hMul (HMul.hMul g h) (Inv.inv g)) ↑h'
    ⊢ Eq (HSMul.hSMul (↑h') (HSMul.hSMul g B)) (HSMul.hSMul g (HSMul.hSMul h B))
  -/
  rw [← hh, smul_smul (g * h * g⁻¹) g B, smul_smul g h B, inv_mul_cancel_right]
  /-
    🎉 no goals
  -/


/-- A translate of a block is a block -/
theorem IsBlock.translate (g : G) (hB : IsBlock G B) :
    IsBlock G (g • B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    g : G
    hB : MulAction.IsBlock G B
    ⊢ MulAction.IsBlock G (HSMul.hSMul g B)
  -/
  rw [← isBlock_top] at hB ⊢
  rw [← Subgroup.map_comap_eq_self_of_surjective
          (G := G) (f := MulAut.conj g) (MulAut.conj g).surjective ⊤]
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    g : G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem Top.top x) B
    ⊢ MulAction.IsBlock (Subtype fun x => Membership.mem (Subgroup.map (↑(MulAut.c …
  -/
  apply IsBlock.of_subgroup_of_conjugate
  /-
    case hB
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    g : G
    hB : MulAction.IsBlock (Subtype fun x => Membership.mem Top.top x) B
    ⊢ MulAction.IsBlock (Subtype fun x => Membership.mem (Subgroup.comap (↑(MulAut …
  -/
  rwa [Subgroup.comap_top]
  /-
    🎉 no goals
  -/


variable (G) in
/-- For `SMul G X`, a block system of `X` is a partition of `X` into blocks
  for the action of `G` -/
def IsBlockSystem (ℬ : Set (Set X)) := Setoid.IsPartition ℬ ∧ ∀ ⦃B⦄, B ∈ ℬ → IsBlock G B


/-- Translates of a block form a block system. -/
theorem IsBlock.isBlockSystem [hGX : MulAction.IsPretransitive G X]
    (hB : IsBlock G B) (hBe : B.Nonempty) :
    IsBlockSystem G (Set.range fun g : G => g • B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    hGX : MulAction.IsPretransitive G X
    hB : MulAction.IsBlock G B
    hBe : B.Nonempty
    ⊢ MulAction.IsBlockSystem G (Set.range fun g => HSMul.hSMul g B)
  -/
  refine ⟨⟨?nonempty, ?cover⟩, ?mem_blocks⟩
  /-
    case nonempty
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    B : Set X
    hGX : MulAction.IsPretransitive G X
    hB : MulAction.IsBlock G B
    hBe : B.Nonempty
    ⊢ Not (Membership.mem (Set.range fun g => HSMul.hSMul g B) EmptyCollection.emp …
  -/
  case mem_blocks => rintro B' ⟨g, rfl⟩; exact hB.translate g
    /-
      case nonempty
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      hBe : B.Nonempty
      ⊢ Not (Membership.mem (Set.range fun g => HSMul.hSMul g B) EmptyCollection.emp …
    -/
  · simp only [Set.mem_range, not_exists]
    /-
      case nonempty
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      hBe : B.Nonempty
      ⊢ ∀ (x : G), Not (Eq (HSMul.hSMul x B) EmptyCollection.emptyCollection)
    -/
    intro g hg
    /-
      case nonempty
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      hBe : B.Nonempty
      g : G
      hg : Eq (HSMul.hSMul g B) EmptyCollection.emptyCollection
      ⊢ False
    -/
    apply hBe.ne_empty
    /-
      case nonempty
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      hBe : B.Nonempty
      g : G
      hg : Eq (HSMul.hSMul g B) EmptyCollection.emptyCollection
      ⊢ Eq B EmptyCollection.emptyCollection
    -/
    simpa only [Set.smul_set_eq_empty] using hg
    /-
      🎉 no goals
    -/
    /-
      case cover
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      hBe : B.Nonempty
      ⊢ ∀ (a : X), ExistsUnique fun b => And (Membership.mem (Set.range fun g => HSM …
    -/
  · intro a
    /-
      case cover
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      hBe : B.Nonempty
      a : X
      ⊢ ExistsUnique fun b => And (Membership.mem (Set.range fun g => HSMul.hSMul g  …
    -/
    obtain ⟨b : X, hb : b ∈ B⟩ := hBe
    /-
      case cover.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a b : X
      hb : Membership.mem B b
      ⊢ ExistsUnique fun b => And (Membership.mem (Set.range fun g => HSMul.hSMul g  …
    -/
    obtain ⟨g, rfl⟩ := exists_smul_eq G b a
    /-
      case cover.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      b : X
      hb : Membership.mem B b
      g : G
      ⊢ ExistsUnique fun b_1 => And (Membership.mem (Set.range fun g => HSMul.hSMul  …
    -/
    use g • B
    simp only [Set.smul_mem_smul_set_iff, hb, existsUnique_iff_exists, Set.mem_range,
      exists_apply_eq_apply, exists_const, exists_prop, and_imp, forall_exists_index,
      forall_apply_eq_imp_iff, true_and]
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      hGX : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      b : X
      hb : Membership.mem B b
      g : G
      ⊢ ∀ (a : G), Membership.mem (HSMul.hSMul a B) (HSMul.hSMul g b) → Eq (HSMul.hS …
    -/
    exact fun g' ha ↦ hB.smul_eq_smul_of_nonempty ⟨g • b, ha, ⟨b, hb, rfl⟩⟩
    /-
      🎉 no goals
    -/


lemma smul_orbit_eq_orbit_smul (N : Subgroup G) [nN : N.Normal] (a : X) (g : G) :
    g • orbit N a = orbit N (g • a) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    N : Subgroup G
    nN : N.Normal
    a : X
    g : G
    ⊢ Eq (HSMul.hSMul g (MulAction.orbit (Subtype fun x => Membership.mem N x) a)) …
  -/
  simp only [orbit, Set.image_smul, Set.smul_set_range]
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    N : Subgroup G
    nN : N.Normal
    a : X
    g : G
    ⊢ Eq (Set.range fun i => HSMul.hSMul g (HSMul.hSMul i a)) (Set.range fun m =>  …
  -/
  ext
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    N : Subgroup G
    nN : N.Normal
    a : X
    g : G
    x✝ : X
    ⊢ Iff (Membership.mem (Set.range fun i => HSMul.hSMul g (HSMul.hSMul i a)) x✝) …
  -/
  simp only [Set.mem_range]
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    N : Subgroup G
    nN : N.Normal
    a : X
    g : G
    x✝ : X
    ⊢ Iff (Exists fun y => Eq (HSMul.hSMul g (HSMul.hSMul y a)) x✝) (Exists fun y  …
  -/
  constructor
    /-
      case h.mp
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g : G
      x✝ : X
      ⊢ (Exists fun y => Eq (HSMul.hSMul g (HSMul.hSMul y a)) x✝) → Exists fun y =>  …
    -/
  · rintro ⟨⟨k, hk⟩, rfl⟩
    /-
      case h.mp.intro.mk
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g k : G
      hk : Membership.mem N k
      ⊢ Exists fun y => Eq (HSMul.hSMul y (HSMul.hSMul g a)) (HSMul.hSMul g (HSMul.h …
    -/
    use ⟨g * k * g⁻¹, nN.conj_mem k hk g⟩
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g k : G
      hk : Membership.mem N k
      ⊢ Eq (HSMul.hSMul ⟨HMul.hMul (HMul.hMul g k) (Inv.inv g), ⋯⟩ (HSMul.hSMul g a) …
    -/
    simp only [Subgroup.mk_smul]
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g k : G
      hk : Membership.mem N k
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul g k) (Inv.inv g)) (HSMul.hSMul g a)) ( …
    -/
    rw [smul_smul, inv_mul_cancel_right, ← smul_smul]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g : G
      x✝ : X
      ⊢ (Exists fun y => Eq (HSMul.hSMul y (HSMul.hSMul g a)) x✝) → Exists fun y =>  …
    -/
  · rintro ⟨⟨k, hk⟩, rfl⟩
    /-
      case h.mpr.intro.mk
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g k : G
      hk : Membership.mem N k
      ⊢ Exists fun y => Eq (HSMul.hSMul g (HSMul.hSMul y a)) (HSMul.hSMul ⟨k, hk⟩ (H …
    -/
    use ⟨g⁻¹ * k * g, nN.conj_mem' k hk g⟩
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g k : G
      hk : Membership.mem N k
      ⊢ Eq (HSMul.hSMul g (HSMul.hSMul ⟨HMul.hMul (HMul.hMul (Inv.inv g) k) g, ⋯⟩ a) …
    -/
    simp only [Subgroup.mk_smul]
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      N : Subgroup G
      nN : N.Normal
      a : X
      g k : G
      hk : Membership.mem N k
      ⊢ Eq (HSMul.hSMul g (HSMul.hSMul (HMul.hMul (HMul.hMul (Inv.inv g) k) g) a)) ( …
    -/
    simp only [← mul_assoc, ← smul_smul, smul_inv_smul, inv_inv]
    /-
      🎉 no goals
    -/


/-- An orbit of a normal subgroup is a block -/
theorem IsBlock.orbit_of_normal {N : Subgroup G} [N.Normal] (a : X) :
    IsBlock G (orbit N a) := by
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    N : Subgroup G
    inst✝ : N.Normal
    a : X
    ⊢ MulAction.IsBlock G (MulAction.orbit (Subtype fun x => Membership.mem N x) a)
  -/
  rw [isBlock_iff_smul_eq_or_disjoint]
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    N : Subgroup G
    inst✝ : N.Normal
    a : X
    ⊢ ∀ (g : G), Or (Eq (HSMul.hSMul g (MulAction.orbit (Subtype fun x => Membersh …
  -/
  intro g
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    N : Subgroup G
    inst✝ : N.Normal
    a : X
    g : G
    ⊢ Or (Eq (HSMul.hSMul g (MulAction.orbit (Subtype fun x => Membership.mem N x) …
  -/
  rw [smul_orbit_eq_orbit_smul]
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    N : Subgroup G
    inst✝ : N.Normal
    a : X
    g : G
    ⊢ Or (Eq (MulAction.orbit (Subtype fun x => Membership.mem N x) (HSMul.hSMul g …
  -/
  apply orbit.eq_or_disjoint
  /-
    🎉 no goals
  -/


/-- The orbits of a normal subgroup form a block system -/
theorem IsBlockSystem.of_normal {N : Subgroup G} [N.Normal] :
    IsBlockSystem G (Set.range fun a : X => orbit N a) := by
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    N : Subgroup G
    inst✝ : N.Normal
    ⊢ MulAction.IsBlockSystem G (Set.range fun a => MulAction.orbit (Subtype fun x …
  -/
  constructor
    /-
      case left
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      N : Subgroup G
      inst✝ : N.Normal
      ⊢ Setoid.IsPartition (Set.range fun a => MulAction.orbit (Subtype fun x => Mem …
    -/
  · apply IsPartition.of_orbits
    /-
      🎉 no goals
    -/
    /-
      case right
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      N : Subgroup G
      inst✝ : N.Normal
      ⊢ ∀ ⦃B : Set X⦄, Membership.mem (Set.range fun a => MulAction.orbit (Subtype f …
    -/
  · intro b; rintro ⟨a, rfl⟩
    /-
      case right.intro
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      N : Subgroup G
      inst✝ : N.Normal
      a : X
      ⊢ MulAction.IsBlock G ((fun a => MulAction.orbit (Subtype fun x => Membership. …
    -/
    exact .orbit_of_normal a
    /-
      🎉 no goals
    -/


/-- See `MulAction.isBlock_subgroup'` for a version that works for the right action of a group on
itself. -/
@[to_additive "See `AddAction.isBlock_subgroup'` for a version that works for the right action
of a group on itself."]
lemma isBlock_subgroup : IsBlock G (s : Set H) := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G H H
    ⊢ MulAction.IsBlock G ↑s
  -/
  simp only [IsBlock, disjoint_left]
  /-
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G H H
    ⊢ ∀ ⦃g₁ g₂ : G⦄, Ne (HSMul.hSMul g₁ ↑s) (HSMul.hSMul g₂ ↑s) → ∀ ⦃a : H⦄, Membe …
  -/
  rintro a b hab _ ⟨c, hc, rfl⟩ ⟨d, hd, (hcd : b • d = a • c)⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G H H
    a b : G
    hab : Ne (HSMul.hSMul a ↑s) (HSMul.hSMul b ↑s)
    c : H
    hc : Membership.mem (↑s) c
    d : H
    hd : Membership.mem (↑s) d
    hcd : Eq (HSMul.hSMul b d) (HSMul.hSMul a c)
    ⊢ False
  -/
  refine hab ?_
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G H H
    a b : G
    hab : Ne (HSMul.hSMul a ↑s) (HSMul.hSMul b ↑s)
    c : H
    hc : Membership.mem (↑s) c
    d : H
    hd : Membership.mem (↑s) d
    hcd : Eq (HSMul.hSMul b d) (HSMul.hSMul a c)
    ⊢ Eq (HSMul.hSMul a ↑s) (HSMul.hSMul b ↑s)
  -/
  rw [← smul_coe_set hc, ← smul_assoc, ← hcd, smul_assoc, smul_coe_set hc, smul_coe_set hd]
  /-
    🎉 no goals
  -/


/-- See `MulAction.isBlock_subgroup` for a version that works for the left action of a group on
itself. -/
@[to_additive "See `AddAction.isBlock_subgroup` for a version that works for the left action
of a group on itself."]
lemma isBlock_subgroup' : IsBlock G (s : Set H) := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G (MulOpposite H) H
    ⊢ MulAction.IsBlock G ↑s
  -/
  simp only [IsBlock, disjoint_left]
  /-
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G (MulOpposite H) H
    ⊢ ∀ ⦃g₁ g₂ : G⦄, Ne (HSMul.hSMul g₁ ↑s) (HSMul.hSMul g₂ ↑s) → ∀ ⦃a : H⦄, Membe …
  -/
  rintro a b hab _ ⟨c, hc, rfl⟩ ⟨d, hd, (hcd : b • d = a • c)⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    S : Type u_3
    H : Type u_4
    inst✝⁴ : Group H
    inst✝³ : SetLike S H
    inst✝² : SubgroupClass S H
    s : S
    inst✝¹ : MulAction G H
    inst✝ : IsScalarTower G (MulOpposite H) H
    a b : G
    hab : Ne (HSMul.hSMul a ↑s) (HSMul.hSMul b ↑s)
    c : H
    hc : Membership.mem (↑s) c
    d : H
    hd : Membership.mem (↑s) d
    hcd : Eq (HSMul.hSMul b d) (HSMul.hSMul a c)
    ⊢ False
  -/
  refine hab ?_
  rw [← op_smul_coe_set hc, ← smul_assoc, ← op_smul, ← hcd, op_smul, smul_assoc, op_smul_coe_set hc,
    op_smul_coe_set hd]


/-- The orbit of `a` under a subgroup containing the stabilizer of `a` is a block -/
theorem IsBlock.of_orbit {H : Subgroup G} {a : X} (hH : stabilizer G a ≤ H) :
    IsBlock G (MulAction.orbit H a) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    H : Subgroup G
    a : X
    hH : LE.le (MulAction.stabilizer G a) H
    ⊢ MulAction.IsBlock G (MulAction.orbit (Subtype fun x => Membership.mem H x) a)
  -/
  rw [isBlock_iff_smul_eq_of_nonempty]
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    H : Subgroup G
    a : X
    hH : LE.le (MulAction.stabilizer G a) H
    ⊢ ∀ ⦃g : G⦄, (Inter.inter (HSMul.hSMul g (MulAction.orbit (Subtype fun x => Me …
  -/
  rintro g ⟨-, ⟨-, ⟨h₁, rfl⟩, h⟩, h₂, rfl⟩
  suffices g ∈ H by
    rw [← Subgroup.coe_mk H g this, ← H.toSubmonoid.smul_def, smul_orbit (⟨g, this⟩ : H) a]
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    H : Subgroup G
    a : X
    hH : LE.le (MulAction.stabilizer G a) H
    g : G
    h₁ h₂ : Subtype fun x => Membership.mem H x
    h : Eq ((fun x => HSMul.hSMul g x) ((fun m => HSMul.hSMul m a) h₁)) ((fun m => …
    ⊢ Membership.mem H g
  -/
  rw [← mul_mem_cancel_left h₂⁻¹.2, ← mul_mem_cancel_right h₁.2]
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    H : Subgroup G
    a : X
    hH : LE.le (MulAction.stabilizer G a) H
    g : G
    h₁ h₂ : Subtype fun x => Membership.mem H x
    h : Eq ((fun x => HSMul.hSMul g x) ((fun m => HSMul.hSMul m a) h₁)) ((fun m => …
    ⊢ Membership.mem H (HMul.hMul (HMul.hMul (↑(Inv.inv h₂)) g) ↑h₁)
  -/
  apply hH
  /-
    case intro.intro.intro.intro.intro.intro.a
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    H : Subgroup G
    a : X
    hH : LE.le (MulAction.stabilizer G a) H
    g : G
    h₁ h₂ : Subtype fun x => Membership.mem H x
    h : Eq ((fun x => HSMul.hSMul g x) ((fun m => HSMul.hSMul m a) h₁)) ((fun m => …
    ⊢ Membership.mem (MulAction.stabilizer G a) (HMul.hMul (HMul.hMul (↑(Inv.inv h …
  -/
  simpa only [mem_stabilizer_iff, InvMemClass.coe_inv, mul_smul, inv_smul_eq_iff]
  /-
    🎉 no goals
  -/


/-- If `B` is a block containing `a`, then the stabilizer of `B` contains the stabilizer of `a` -/
theorem IsBlock.stabilizer_le (hB : IsBlock G B) {a : X} (ha : a ∈ B) :
    stabilizer G a ≤ stabilizer G B :=
                                           /-
                                             G : Type u_1
                                             inst✝¹ : Group G
                                             X : Type u_2
                                             inst✝ : MulAction G X
                                             B : Set X
                                             hB : MulAction.IsBlock G B
                                             a : X
                                             ha : Membership.mem B a
                                             g : G
                                             hg : Membership.mem (MulAction.stabilizer G a) g
                                             ⊢ Membership.mem (HSMul.hSMul g B) a
                                           -/
  fun g hg ↦ hB.smul_eq_of_nonempty ⟨a, by rwa [← hg, smul_mem_smul_set_iff], ha⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- A block containing `a` is the orbit of `a` under its stabilizer -/
theorem IsBlock.orbit_stabilizer_eq [IsPretransitive G X] (hB : IsBlock G B) {a : X} (ha : a ∈ B) :
    MulAction.orbit (stabilizer G B) a = B := by
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    B : Set X
    inst✝ : MulAction.IsPretransitive G X
    hB : MulAction.IsBlock G B
    a : X
    ha : Membership.mem B a
    ⊢ Eq (MulAction.orbit (Subtype fun x => Membership.mem (MulAction.stabilizer G …
  -/
  ext x
  /-
    case h
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    B : Set X
    inst✝ : MulAction.IsPretransitive G X
    hB : MulAction.IsBlock G B
    a : X
    ha : Membership.mem B a
    x : X
    ⊢ Iff (Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (MulAc …
  -/
  constructor
    /-
      case h.mp
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      x : X
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (MulAction. …
    -/
  · rintro ⟨⟨k, k_mem⟩, rfl⟩
    /-
      case h.mp.intro.mk
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      k : G
      k_mem : Membership.mem (MulAction.stabilizer G B) k
      ⊢ Membership.mem B ((fun m => HSMul.hSMul m a) ⟨k, k_mem⟩)
    -/
    simp only [Subgroup.mk_smul]
    /-
      case h.mp.intro.mk
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      k : G
      k_mem : Membership.mem (MulAction.stabilizer G B) k
      ⊢ Membership.mem B (HSMul.hSMul k a)
    -/
    rw [← k_mem, Set.smul_mem_smul_set_iff]
    /-
      case h.mp.intro.mk
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      k : G
      k_mem : Membership.mem (MulAction.stabilizer G B) k
      ⊢ Membership.mem B a
    -/
    exact ha
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      x : X
      ⊢ Membership.mem B x → Membership.mem (MulAction.orbit (Subtype fun x => Membe …
    -/
  · intro hx
    /-
      case h.mpr
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      x : X
      hx : Membership.mem B x
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (MulAction. …
    -/
    obtain ⟨k, rfl⟩ := exists_smul_eq G a x
    /-
      case h.mpr.intro
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      B : Set X
      inst✝ : MulAction.IsPretransitive G X
      hB : MulAction.IsBlock G B
      a : X
      ha : Membership.mem B a
      k : G
      hx : Membership.mem B (HSMul.hSMul k a)
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem (MulAction. …
    -/
    exact ⟨⟨k, hB.smul_eq_of_mem ha hx⟩, rfl⟩
    /-
      🎉 no goals
    -/


/-- A subgroup containing the stabilizer of `a`
  is the stabilizer of the orbit of `a` under that subgroup -/
theorem stabilizer_orbit_eq {a : X} {H : Subgroup G} (hH : stabilizer G a ≤ H) :
    stabilizer G (orbit H a) = H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    a : X
    H : Subgroup G
    hH : LE.le (MulAction.stabilizer G a) H
    ⊢ Eq (MulAction.stabilizer G (MulAction.orbit (Subtype fun x => Membership.mem …
  -/
  ext g
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    X : Type u_2
    inst✝ : MulAction G X
    a : X
    H : Subgroup G
    hH : LE.le (MulAction.stabilizer G a) H
    g : G
    ⊢ Iff (Membership.mem (MulAction.stabilizer G (MulAction.orbit (Subtype fun x  …
  -/
  constructor
    /-
      case h.mp
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      ⊢ Membership.mem (MulAction.stabilizer G (MulAction.orbit (Subtype fun x => Me …
    -/
  · intro hg
    /-
      case h.mp
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      hg : Membership.mem (MulAction.stabilizer G (MulAction.orbit (Subtype fun x => …
      ⊢ Membership.mem H g
    -/
    obtain ⟨-, ⟨b, rfl⟩, h⟩ := hg.symm ▸ mem_orbit_self a
    /-
      case h.mp.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      hg : Membership.mem (MulAction.stabilizer G (MulAction.orbit (Subtype fun x => …
      b : Subtype fun x => Membership.mem H x
      h : Eq ((fun x => HSMul.hSMul g x) ((fun m => HSMul.hSMul m a) b)) a
      ⊢ Membership.mem H g
    -/
    simp_rw [H.toSubmonoid.smul_def, ← mul_smul, ← mem_stabilizer_iff] at h
    /-
      case h.mp.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      hg : Membership.mem (MulAction.stabilizer G (MulAction.orbit (Subtype fun x => …
      b : Subtype fun x => Membership.mem H x
      h : Membership.mem (MulAction.stabilizer G a) (HMul.hMul g ↑b)
      ⊢ Membership.mem H g
    -/
    exact (mul_mem_cancel_right b.2).mp (hH h)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      ⊢ Membership.mem H g → Membership.mem (MulAction.stabilizer G (MulAction.orbit …
    -/
  · intro hg
    /-
      case h.mpr
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      hg : Membership.mem H g
      ⊢ Membership.mem (MulAction.stabilizer G (MulAction.orbit (Subtype fun x => Me …
    -/
    rw [mem_stabilizer_iff, ← Subgroup.coe_mk H g hg, ← Submonoid.smul_def (S := H.toSubmonoid)]
    /-
      case h.mpr
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      a : X
      H : Subgroup G
      hH : LE.le (MulAction.stabilizer G a) H
      g : G
      hg : Membership.mem H g
      ⊢ Eq (HSMul.hSMul ⟨g, hg⟩ (MulAction.orbit (Subtype fun x => Membership.mem H  …
    -/
    apply smul_orbit (G := H)
    /-
      🎉 no goals
    -/


/-- Order equivalence between blocks in `X` containing a point `a`
 and subgroups of `G` containing the stabilizer of `a` (Wielandt, th. 7.5)-/
def block_stabilizerOrderIso [htGX : IsPretransitive G X] (a : X) :
    { B : Set X // a ∈ B ∧ IsBlock G B } ≃o Set.Ici (stabilizer G a) where
  toFun := fun ⟨B, ha, hB⟩ => ⟨stabilizer G B, hB.stabilizer_le ha⟩
  invFun := fun ⟨H, hH⟩ =>
    ⟨MulAction.orbit H a, MulAction.mem_orbit_self a, IsBlock.of_orbit hH⟩
  left_inv := fun ⟨_, ha, hB⟩ =>
    (id (propext Subtype.mk_eq_mk)).mpr (hB.orbit_stabilizer_eq ha)
  right_inv := fun ⟨_, hH⟩ =>
    (id (propext Subtype.mk_eq_mk)).mpr (stabilizer_orbit_eq hH)
  map_rel_iff' := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B : Set X
      htGX : MulAction.IsPretransitive G X
      a : X
      ⊢ ∀ {a_1 b : Subtype fun B => And (Membership.mem B a) (MulAction.IsBlock G B) …
    -/
    rintro ⟨B, ha, hB⟩; rintro ⟨B', ha', hB'⟩
    /-
      case mk.intro.mk.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B✝ : Set X
      htGX : MulAction.IsPretransitive G X
      a : X
      B : Set X
      ha : Membership.mem B a
      hB : MulAction.IsBlock G B
      B' : Set X
      ha' : Membership.mem B' a
      hB' : MulAction.IsBlock G B'
      ⊢ Iff (LE.le ({ toFun := fun x => MulAction.block_stabilizerOrderIso.match_1 G …
    -/
    simp only [Equiv.coe_fn_mk, Subtype.mk_le_mk, Set.le_eq_subset]
    /-
      case mk.intro.mk.intro
      G : Type u_1
      inst✝¹ : Group G
      X : Type u_2
      inst✝ : MulAction G X
      B✝ : Set X
      htGX : MulAction.IsPretransitive G X
      a : X
      B : Set X
      ha : Membership.mem B a
      hB : MulAction.IsBlock G B
      B' : Set X
      ha' : Membership.mem B' a
      hB' : MulAction.IsBlock G B'
      ⊢ Iff (LE.le (MulAction.stabilizer G B) (MulAction.stabilizer G B')) (HasSubse …
    -/
    constructor
      /-
        case mk.intro.mk.intro.mp
        G : Type u_1
        inst✝¹ : Group G
        X : Type u_2
        inst✝ : MulAction G X
        B✝ : Set X
        htGX : MulAction.IsPretransitive G X
        a : X
        B : Set X
        ha : Membership.mem B a
        hB : MulAction.IsBlock G B
        B' : Set X
        ha' : Membership.mem B' a
        hB' : MulAction.IsBlock G B'
        ⊢ LE.le (MulAction.stabilizer G B) (MulAction.stabilizer G B') → HasSubset.Sub …
      -/
    · rintro hBB' b hb
      /-
        case mk.intro.mk.intro.mp
        G : Type u_1
        inst✝¹ : Group G
        X : Type u_2
        inst✝ : MulAction G X
        B✝ : Set X
        htGX : MulAction.IsPretransitive G X
        a : X
        B : Set X
        ha : Membership.mem B a
        hB : MulAction.IsBlock G B
        B' : Set X
        ha' : Membership.mem B' a
        hB' : MulAction.IsBlock G B'
        hBB' : LE.le (MulAction.stabilizer G B) (MulAction.stabilizer G B')
        b : X
        hb : Membership.mem B b
        ⊢ Membership.mem B' b
      -/
      obtain ⟨k, rfl⟩ := htGX.exists_smul_eq a b
      suffices k ∈ stabilizer G B' by
        exact this.symm ▸ (Set.smul_mem_smul_set ha')
      /-
        case mk.intro.mk.intro.mp.intro
        G : Type u_1
        inst✝¹ : Group G
        X : Type u_2
        inst✝ : MulAction G X
        B✝ : Set X
        htGX : MulAction.IsPretransitive G X
        a : X
        B : Set X
        ha : Membership.mem B a
        hB : MulAction.IsBlock G B
        B' : Set X
        ha' : Membership.mem B' a
        hB' : MulAction.IsBlock G B'
        hBB' : LE.le (MulAction.stabilizer G B) (MulAction.stabilizer G B')
        k : G
        hb : Membership.mem B (HSMul.hSMul k a)
        ⊢ Membership.mem (MulAction.stabilizer G B') k
      -/
      exact hBB' (hB.smul_eq_of_mem ha hb)
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.mk.intro.mpr
        G : Type u_1
        inst✝¹ : Group G
        X : Type u_2
        inst✝ : MulAction G X
        B✝ : Set X
        htGX : MulAction.IsPretransitive G X
        a : X
        B : Set X
        ha : Membership.mem B a
        hB : MulAction.IsBlock G B
        B' : Set X
        ha' : Membership.mem B' a
        hB' : MulAction.IsBlock G B'
        ⊢ HasSubset.Subset B B' → LE.le (MulAction.stabilizer G B) (MulAction.stabiliz …
      -/
    · intro hBB' g hgB
      /-
        case mk.intro.mk.intro.mpr
        G : Type u_1
        inst✝¹ : Group G
        X : Type u_2
        inst✝ : MulAction G X
        B✝ : Set X
        htGX : MulAction.IsPretransitive G X
        a : X
        B : Set X
        ha : Membership.mem B a
        hB : MulAction.IsBlock G B
        B' : Set X
        ha' : Membership.mem B' a
        hB' : MulAction.IsBlock G B'
        hBB' : HasSubset.Subset B B'
        g : G
        hgB : Membership.mem (MulAction.stabilizer G B) g
        ⊢ Membership.mem (MulAction.stabilizer G B') g
      -/
      apply hB'.smul_eq_of_mem ha'
      /-
        case mk.intro.mk.intro.mpr
        G : Type u_1
        inst✝¹ : Group G
        X : Type u_2
        inst✝ : MulAction G X
        B✝ : Set X
        htGX : MulAction.IsPretransitive G X
        a : X
        B : Set X
        ha : Membership.mem B a
        hB : MulAction.IsBlock G B
        B' : Set X
        ha' : Membership.mem B' a
        hB' : MulAction.IsBlock G B'
        hBB' : HasSubset.Subset B B'
        g : G
        hgB : Membership.mem (MulAction.stabilizer G B) g
        ⊢ Membership.mem B' (HSMul.hSMul g a)
      -/
      exact hBB' <| hgB.symm ▸ (Set.smul_mem_smul_set ha)
      /-
        🎉 no goals
      -/


theorem ncard_block_eq_relindex (hB : IsBlock G B) {x : X} (hx : x ∈ B) :
    B.ncard = (stabilizer G x).relindex (stabilizer G B) := by
  have key : (stabilizer G x).subgroupOf (stabilizer G B) = stabilizer (stabilizer G B) x := by
    ext; rfl
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    hB : MulAction.IsBlock G B
    x : X
    hx : Membership.mem B x
    key : Eq ((MulAction.stabilizer G x).subgroupOf (MulAction.stabilizer G B)) (M …
    ⊢ Eq B.ncard ((MulAction.stabilizer G x).relindex (MulAction.stabilizer G B))
  -/
  rw [Subgroup.relindex, key, index_stabilizer, hB.orbit_stabilizer_eq hx]
  /-
    🎉 no goals
  -/


/-- The cardinality of the ambient space is the product of the cardinality of a block
  by the cardinality of the set of translates of that block -/
theorem ncard_block_mul_ncard_orbit_eq (hB : IsBlock G B) (hB_ne : B.Nonempty) :
    Set.ncard B * Set.ncard (orbit G B) = Nat.card X := by
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    hB : MulAction.IsBlock G B
    hB_ne : B.Nonempty
    ⊢ Eq (HMul.hMul B.ncard (MulAction.orbit G B).ncard) (Nat.card X)
  -/
  obtain ⟨x, hx⟩ := hB_ne
  rw [ncard_block_eq_relindex hB hx, ← index_stabilizer,
      Subgroup.relindex_mul_index (hB.stabilizer_le hx), index_stabilizer_of_transitive]


/-- The cardinality of a block divides the cardinality of the ambient type -/
theorem ncard_dvd_card (hB : IsBlock G B) (hB_ne : B.Nonempty) :
    Set.ncard B ∣ Nat.card X :=
  Dvd.intro _ (hB.ncard_block_mul_ncard_orbit_eq hB_ne)


/-- A too large block is equal to `univ` -/
theorem eq_univ_of_card_lt [hX : Finite X] (hB : IsBlock G B) (hB' : Nat.card X < Set.ncard B * 2) :
    B = Set.univ := by
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    hX : Finite X
    hB : MulAction.IsBlock G B
    hB' : LT.lt (Nat.card X) (HMul.hMul B.ncard 2)
    ⊢ Eq B Set.univ
  -/
  rcases Set.eq_empty_or_nonempty B with rfl | hB_ne
    /-
      case inl
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      inst✝ : MulAction.IsPretransitive G X
      hX : Finite X
      hB : MulAction.IsBlock G EmptyCollection.emptyCollection
      hB' : LT.lt (Nat.card X) (HMul.hMul EmptyCollection.emptyCollection.ncard 2)
      ⊢ Eq EmptyCollection.emptyCollection Set.univ
    -/
  · simp only [Set.ncard_empty, zero_mul, not_lt_zero'] at hB'
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    hX : Finite X
    hB : MulAction.IsBlock G B
    hB' : LT.lt (Nat.card X) (HMul.hMul B.ncard 2)
    hB_ne : B.Nonempty
    ⊢ Eq B Set.univ
  -/
  have key := hB.ncard_block_mul_ncard_orbit_eq hB_ne
  /-
    case inr
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    hX : Finite X
    hB : MulAction.IsBlock G B
    hB' : LT.lt (Nat.card X) (HMul.hMul B.ncard 2)
    hB_ne : B.Nonempty
    key : Eq (HMul.hMul B.ncard (MulAction.orbit G B).ncard) (Nat.card X)
    ⊢ Eq B Set.univ
  -/
  rw [← key, mul_lt_mul_iff_of_pos_left (by rwa [Set.ncard_pos])] at hB'
  /-
    case inr
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    hX : Finite X
    hB : MulAction.IsBlock G B
    hB' : LT.lt (MulAction.orbit G B).ncard 2
    hB_ne : B.Nonempty
    key : Eq (HMul.hMul B.ncard (MulAction.orbit G B).ncard) (Nat.card X)
    ⊢ Eq B Set.univ
  -/
  interval_cases (orbit G B).ncard
    /-
      case inr.«0»
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      inst✝ : MulAction.IsPretransitive G X
      B : Set X
      hX : Finite X
      hB : MulAction.IsBlock G B
      hB_ne : B.Nonempty
      hB' : LT.lt 0 2
      key : Eq (HMul.hMul B.ncard 0) (Nat.card X)
      ⊢ Eq B Set.univ
    -/
  · rw [mul_zero, eq_comm, Nat.card_eq_zero, or_iff_left hX.not_infinite] at key
    /-
      case inr.«0»
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      inst✝ : MulAction.IsPretransitive G X
      B : Set X
      hX : Finite X
      hB : MulAction.IsBlock G B
      hB_ne : B.Nonempty
      hB' : LT.lt 0 2
      key : IsEmpty X
      ⊢ Eq B Set.univ
    -/
    exact (IsEmpty.exists_iff.mp hB_ne).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.«1»
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      inst✝ : MulAction.IsPretransitive G X
      B : Set X
      hX : Finite X
      hB : MulAction.IsBlock G B
      hB_ne : B.Nonempty
      hB' : LT.lt 1 2
      key : Eq (HMul.hMul B.ncard 1) (Nat.card X)
      ⊢ Eq B Set.univ
    -/
  · rw [mul_one, ← Set.ncard_univ] at key
    /-
      case inr.«1»
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      inst✝ : MulAction.IsPretransitive G X
      B : Set X
      hX : Finite X
      hB : MulAction.IsBlock G B
      hB_ne : B.Nonempty
      hB' : LT.lt 1 2
      key : Eq B.ncard Set.univ.ncard
      ⊢ Eq B Set.univ
    -/
    rw [Set.eq_of_subset_of_ncard_le (Set.subset_univ B) key.ge]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-29")] alias eq_univ_card_lt := eq_univ_of_card_lt


/-- If a block has too many translates, then it is a (sub)singleton  -/
theorem subsingleton_of_card_lt [Finite X] (hB : IsBlock G B)
    (hB' : Nat.card X < 2 * Set.ncard (orbit G B)) :
    B.Subsingleton := by
  suffices Set.ncard B < 2 by
    rw [Nat.lt_succ_iff, Set.ncard_le_one_iff_eq] at this
    cases this with
    | inl h => rw [h]; exact Set.subsingleton_empty
    | inr h =>
      obtain ⟨a, ha⟩ := h; rw [ha]; exact Set.subsingleton_singleton
  cases Set.eq_empty_or_nonempty B with
  | inl h => rw [h, Set.ncard_empty]; norm_num
  | inr h =>
    rw [← hB.ncard_block_mul_ncard_orbit_eq h, lt_iff_not_ge] at hB'
    rw [← not_le]
    exact fun hb ↦ hB' (Nat.mul_le_mul_right _ hb)

/- The assumption `B.Finite` is necessary :
   For G = ℤ acting on itself, a = 0 and B = ℕ, the translates `k • B` of the statement
   are just `k + ℕ`, for `k ≤ 0`, and the corresponding intersection is `ℕ`, which is not a block.
   (Remark by Thomas Browning) -/

/-- The intersection of the translates of a *finite* subset which contain a given point
is a block (Wielandt, th. 7.3 )-/
theorem of_subset (a : X) (hfB : B.Finite) :
    IsBlock G (⋂ (k : G) (_ : a ∈ k • B), k • B) := by
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    ⊢ MulAction.IsBlock G (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)
  -/
  let B' := ⋂ (k : G) (_ : a ∈ k • B), k • B
  /-
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    ⊢ MulAction.IsBlock G (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)
  -/
  cases' Set.eq_empty_or_nonempty B with hfB_e hfB_ne
    /-
      case inl
      G : Type u_1
      inst✝² : Group G
      X : Type u_2
      inst✝¹ : MulAction G X
      inst✝ : MulAction.IsPretransitive G X
      B : Set X
      a : X
      hfB : B.Finite
      B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
      hfB_e : Eq B EmptyCollection.emptyCollection
      ⊢ MulAction.IsBlock G (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)
    -/
  · simp [hfB_e]
    /-
      🎉 no goals
    -/
  have hB'₀ : ∀ (k : G) (_ : a ∈ k • B), B' ≤ k • B := by
    intro k hk
    exact Set.biInter_subset_of_mem hk
  have hfB' : B'.Finite := by
    obtain ⟨b, hb : b ∈ B⟩ := hfB_ne
    obtain ⟨k, hk : k • b = a⟩ := exists_smul_eq G b a
    apply Set.Finite.subset (Set.Finite.map _ hfB) (hB'₀ k ⟨b, hb, hk⟩)
  have hag : ∀ g : G, a ∈ g • B' → B' ≤ g • B' :=  by
    intro g hg x hx
    -- a = g • b; b ∈ B'; a ∈ k • B → b ∈ k • B
    simp only [B', Set.mem_iInter, Set.mem_smul_set_iff_inv_smul_mem,
      smul_smul, ← mul_inv_rev] at hg hx ⊢
    exact fun _ ↦ hx _ ∘ hg _
  have hag' (g : G) (hg : a ∈ g • B') : B' = g • B' := by
    rw [eq_comm, ← mem_stabilizer_iff, mem_stabilizer_set_iff_subset_smul_set hfB']
    exact hag g hg
  /-
    case inr
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    ⊢ MulAction.IsBlock G (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)
  -/
  rw [isBlock_iff_smul_eq_of_nonempty]
  /-
    case inr
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    ⊢ ∀ ⦃g : G⦄, (Inter.inter (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x …
  -/
  rintro g ⟨b : X, hb' : b ∈ g • B', hb : b ∈ B'⟩
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    g : G
    b : X
    hb' : Membership.mem (HSMul.hSMul g B') b
    hb : Membership.mem B' b
    ⊢ Eq (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)) …
  -/
  obtain ⟨k : G, hk : k • a = b⟩ := exists_smul_eq G a b
  have hak : a ∈ k⁻¹ • B' := by
    refine ⟨b, hb, ?_⟩
    simp only [← hk, inv_smul_smul]
  have hagk : a ∈ (k⁻¹ * g) • B' := by
    rw [mul_smul, Set.mem_smul_set_iff_inv_smul_mem, inv_inv, hk]
    exact hb'
  /-
    case inr.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    g : G
    b : X
    hb' : Membership.mem (HSMul.hSMul g B') b
    hb : Membership.mem B' b
    k : G
    hk : Eq (HSMul.hSMul k a) b
    hak : Membership.mem (HSMul.hSMul (Inv.inv k) B') a
    hagk : Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv k) g) B') a
    ⊢ Eq (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)) …
  -/
  have hkB' : B' = k⁻¹ • B' := hag' k⁻¹ hak
  /-
    case inr.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    g : G
    b : X
    hb' : Membership.mem (HSMul.hSMul g B') b
    hb : Membership.mem B' b
    k : G
    hk : Eq (HSMul.hSMul k a) b
    hak : Membership.mem (HSMul.hSMul (Inv.inv k) B') a
    hagk : Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv k) g) B') a
    hkB' : Eq B' (HSMul.hSMul (Inv.inv k) B')
    ⊢ Eq (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)) …
  -/
  have hgkB' : B' = (k⁻¹ * g) • B' := hag' (k⁻¹ * g) hagk
  /-
    case inr.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    g : G
    b : X
    hb' : Membership.mem (HSMul.hSMul g B') b
    hb : Membership.mem B' b
    k : G
    hk : Eq (HSMul.hSMul k a) b
    hak : Membership.mem (HSMul.hSMul (Inv.inv k) B') a
    hagk : Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv k) g) B') a
    hkB' : Eq B' (HSMul.hSMul (Inv.inv k) B')
    hgkB' : Eq B' (HSMul.hSMul (HMul.hMul (Inv.inv k) g) B')
    ⊢ Eq (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)) …
  -/
  rw [mul_smul] at hgkB'
  /-
    case inr.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    g : G
    b : X
    hb' : Membership.mem (HSMul.hSMul g B') b
    hb : Membership.mem B' b
    k : G
    hk : Eq (HSMul.hSMul k a) b
    hak : Membership.mem (HSMul.hSMul (Inv.inv k) B') a
    hagk : Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv k) g) B') a
    hkB' : Eq B' (HSMul.hSMul (Inv.inv k) B')
    hgkB' : Eq B' (HSMul.hSMul (Inv.inv k) (HSMul.hSMul g B'))
    ⊢ Eq (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)) …
  -/
  rw [← smul_eq_iff_eq_inv_smul] at hkB' hgkB'
  /-
    case inr.intro.intro.intro
    G : Type u_1
    inst✝² : Group G
    X : Type u_2
    inst✝¹ : MulAction G X
    inst✝ : MulAction.IsPretransitive G X
    B : Set X
    a : X
    hfB : B.Finite
    B' : Set X := Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B
    hfB_ne : B.Nonempty
    hB'₀ : ∀ (k : G), Membership.mem (HSMul.hSMul k B) a → LE.le B' (HSMul.hSMul k …
    hfB' : B'.Finite
    hag : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → LE.le B' (HSMul.hSMul g …
    hag' : ∀ (g : G), Membership.mem (HSMul.hSMul g B') a → Eq B' (HSMul.hSMul g B')
    g : G
    b : X
    hb' : Membership.mem (HSMul.hSMul g B') b
    hb : Membership.mem B' b
    k : G
    hk : Eq (HSMul.hSMul k a) b
    hak : Membership.mem (HSMul.hSMul (Inv.inv k) B') a
    hagk : Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv k) g) B') a
    hkB' : Eq (HSMul.hSMul k B') B'
    hgkB' : Eq (HSMul.hSMul k B') (HSMul.hSMul g B')
    ⊢ Eq (HSMul.hSMul g (Set.iInter fun k => Set.iInter fun x => HSMul.hSMul k B)) …
  -/
  rw [← hgkB', hkB']
  /-
    🎉 no goals
  -/


