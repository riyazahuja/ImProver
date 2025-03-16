/-- A `GroupFilterBasis` on a group is a `FilterBasis` satisfying some additional axioms.
  Example : if `G` is a topological group then the neighbourhoods of the identity are a
  `GroupFilterBasis`. Conversely given a `GroupFilterBasis` one can define a topology
  compatible with the group structure on `G`. -/
class GroupFilterBasis (G : Type u) [Group G] extends FilterBasis G where
  one' : ∀ {U}, U ∈ sets → (1 : G) ∈ U
  mul' : ∀ {U}, U ∈ sets → ∃ V ∈ sets, V * V ⊆ U
  inv' : ∀ {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ x⁻¹) ⁻¹' U
  conj' : ∀ x₀, ∀ {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ x₀ * x * x₀⁻¹) ⁻¹' U


/-- An `AddGroupFilterBasis` on an additive group is a `FilterBasis` satisfying some additional
  axioms. Example : if `G` is a topological group then the neighbourhoods of the identity are an
  `AddGroupFilterBasis`. Conversely given an `AddGroupFilterBasis` one can define a topology
  compatible with the group structure on `G`. -/
class AddGroupFilterBasis (A : Type u) [AddGroup A] extends FilterBasis A where
  zero' : ∀ {U}, U ∈ sets → (0 : A) ∈ U
  add' : ∀ {U}, U ∈ sets → ∃ V ∈ sets, V + V ⊆ U
  neg' : ∀ {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ -x) ⁻¹' U
  conj' : ∀ x₀, ∀ {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ x₀ + x + -x₀) ⁻¹' U


attribute [to_additive existing] GroupFilterBasis GroupFilterBasis.conj'
  GroupFilterBasis.toFilterBasis


/-- `GroupFilterBasis` constructor in the commutative group case. -/
@[to_additive "`AddGroupFilterBasis` constructor in the additive commutative group case."]
def groupFilterBasisOfComm {G : Type*} [CommGroup G] (sets : Set (Set G))
    (nonempty : sets.Nonempty) (inter_sets : ∀ x y, x ∈ sets → y ∈ sets → ∃ z ∈ sets, z ⊆ x ∩ y)
    (one : ∀ U ∈ sets, (1 : G) ∈ U) (mul : ∀ U ∈ sets, ∃ V ∈ sets, V * V ⊆ U)
    (inv : ∀ U ∈ sets, ∃ V ∈ sets, V ⊆ (fun x ↦ x⁻¹) ⁻¹' U) : GroupFilterBasis G :=
  { sets := sets
    nonempty := nonempty
    inter_sets := inter_sets _ _
    one' := one _
    mul' := mul _
    inv' := inv _
                                         /-
                                           G : Type u_1
                                           inst✝ : CommGroup G
                                           sets : Set (Set G)
                                           nonempty : sets.Nonempty
                                           inter_sets : ∀ (x y : Set G), Membership.mem sets x → Membership.mem sets y →  …
                                           one : ∀ (U : Set G), Membership.mem sets U → Membership.mem U 1
                                           mul : ∀ (U : Set G), Membership.mem sets U → Exists fun V => And (Membership.m …
                                           inv : ∀ (U : Set G), Membership.mem sets U → Exists fun V => And (Membership.m …
                                           x : G
                                           U : Set G
                                           U_in : Membership.mem { sets := sets, nonempty := nonempty, inter_sets := ⋯ }. …
                                           ⊢ HasSubset.Subset U (Set.preimage (fun x_1 => HMul.hMul (HMul.hMul x x_1) (In …
                                         -/
    conj' := fun x U U_in ↦ ⟨U, U_in, by simp only [mul_inv_cancel_comm, preimage_id']; rfl⟩ }
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[to_additive]
instance : Membership (Set G) (GroupFilterBasis G) :=
  ⟨fun f s ↦ s ∈ f.sets⟩


@[to_additive]
theorem one {U : Set G} : U ∈ B → (1 : G) ∈ U :=
  GroupFilterBasis.one'


@[to_additive]
theorem mul {U : Set G} : U ∈ B → ∃ V ∈ B, V * V ⊆ U :=
  GroupFilterBasis.mul'


@[to_additive]
theorem inv {U : Set G} : U ∈ B → ∃ V ∈ B, V ⊆ (fun x ↦ x⁻¹) ⁻¹' U :=
  GroupFilterBasis.inv'


@[to_additive]
theorem conj : ∀ x₀, ∀ {U}, U ∈ B → ∃ V ∈ B, V ⊆ (fun x ↦ x₀ * x * x₀⁻¹) ⁻¹' U :=
  GroupFilterBasis.conj'


/-- The trivial group filter basis consists of `{1}` only. The associated topology
is discrete. -/
@[to_additive "The trivial additive group filter basis consists of `{0}` only. The associated
topology is discrete."]
instance : Inhabited (GroupFilterBasis G) where
  default := {
    sets := {{1}}
    nonempty := singleton_nonempty _
                     /-
                       G : Type u
                       inst✝ : Group G
                       B : GroupFilterBasis G
                       ⊢ ∀ {x y : Set G}, Membership.mem (Singleton.singleton (Singleton.singleton 1) …
                     -/
    inter_sets := by simp
                     /-
                       🎉 no goals
                     -/
               /-
                 G : Type u
                 inst✝ : Group G
                 B : GroupFilterBasis G
                 ⊢ ∀ {U : Set G}, Membership.mem { sets := Singleton.singleton (Singleton.singl …
               -/
    one' := by simp
               /-
                 🎉 no goals
               -/
               /-
                 G : Type u
                 inst✝ : Group G
                 B : GroupFilterBasis G
                 ⊢ ∀ {U : Set G}, Membership.mem { sets := Singleton.singleton (Singleton.singl …
               -/
    mul' := by simp
               /-
                 🎉 no goals
               -/
               /-
                 G : Type u
                 inst✝ : Group G
                 B : GroupFilterBasis G
                 ⊢ ∀ {U : Set G}, Membership.mem { sets := Singleton.singleton (Singleton.singl …
               -/
    inv' := by simp
               /-
                 🎉 no goals
               -/
                /-
                  G : Type u
                  inst✝ : Group G
                  B : GroupFilterBasis G
                  ⊢ ∀ (x₀ : G) {U : Set G}, Membership.mem { sets := Singleton.singleton (Single …
                -/
    conj' := by simp }
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem subset_mul_self (B : GroupFilterBasis G) {U : Set G} (h : U ∈ B) : U ⊆ U * U :=
  fun x x_in ↦ ⟨1, one h, x, x_in, one_mul x⟩


/-- The neighborhood function of a `GroupFilterBasis`. -/
@[to_additive "The neighborhood function of an `AddGroupFilterBasis`."]
def N (B : GroupFilterBasis G) : G → Filter G :=
  fun x ↦ map (fun y ↦ x * y) B.toFilterBasis.filter


@[to_additive (attr := simp)]
theorem N_one (B : GroupFilterBasis G) : B.N 1 = B.toFilterBasis.filter := by
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    ⊢ Eq (B.N 1) GroupFilterBasis.toFilterBasis.filter
  -/
  simp only [N, one_mul, map_id']
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem hasBasis (B : GroupFilterBasis G) (x : G) :
    HasBasis (B.N x) (fun V : Set G ↦ V ∈ B) fun V ↦ (fun y ↦ x * y) '' V :=
  HasBasis.map (fun y ↦ x * y) toFilterBasis.hasBasis


/-- The topological space structure coming from a group filter basis. -/
@[to_additive "The topological space structure coming from an additive group filter basis."]
def topology (B : GroupFilterBasis G) : TopologicalSpace G :=
  TopologicalSpace.mkOfNhds B.N


@[to_additive]
theorem nhds_eq (B : GroupFilterBasis G) {x₀ : G} : @nhds G B.topology x₀ = B.N x₀ := by
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    x₀ : G
    ⊢ Eq (nhds x₀) (B.N x₀)
  -/
  apply TopologicalSpace.nhds_mkOfNhds_of_hasBasis (fun x ↦ (FilterBasis.hasBasis _).map _)
    /-
      case hpure
      G : Type u
      inst✝ : Group G
      B : GroupFilterBasis G
      x₀ : G
      ⊢ ∀ (a : G) (i : Set G), Membership.mem GroupFilterBasis.toFilterBasis i → Mem …
    -/
  · intro a U U_in
    /-
      case hpure
      G : Type u
      inst✝ : Group G
      B : GroupFilterBasis G
      x₀ a : G
      U : Set G
      U_in : Membership.mem GroupFilterBasis.toFilterBasis U
      ⊢ Membership.mem (Set.image (fun y => HMul.hMul a y) (id U)) a
    -/
    exact ⟨1, B.one U_in, mul_one a⟩
    /-
      🎉 no goals
    -/
    /-
      case hopen
      G : Type u
      inst✝ : Group G
      B : GroupFilterBasis G
      x₀ : G
      ⊢ ∀ (a : G) (i : Set G), Membership.mem GroupFilterBasis.toFilterBasis i → Fil …
    -/
  · intro a U U_in
    /-
      case hopen
      G : Type u
      inst✝ : Group G
      B : GroupFilterBasis G
      x₀ a : G
      U : Set G
      U_in : Membership.mem GroupFilterBasis.toFilterBasis U
      ⊢ Filter.Eventually (fun x => Membership.mem (Filter.map (fun y => HMul.hMul x …
    -/
    rcases GroupFilterBasis.mul U_in with ⟨V, V_in, hVU⟩
    /-
      case hopen.intro.intro
      G : Type u
      inst✝ : Group G
      B : GroupFilterBasis G
      x₀ a : G
      U : Set G
      U_in : Membership.mem GroupFilterBasis.toFilterBasis U
      V : Set G
      V_in : Membership.mem B V
      hVU : HasSubset.Subset (HMul.hMul V V) U
      ⊢ Filter.Eventually (fun x => Membership.mem (Filter.map (fun y => HMul.hMul x …
    -/
    filter_upwards [image_mem_map (B.mem_filter_of_mem V_in)]
    /-
      case h
      G : Type u
      inst✝ : Group G
      B : GroupFilterBasis G
      x₀ a : G
      U : Set G
      U_in : Membership.mem GroupFilterBasis.toFilterBasis U
      V : Set G
      V_in : Membership.mem B V
      hVU : HasSubset.Subset (HMul.hMul V V) U
      ⊢ ∀ (a_1 : G), Membership.mem (Set.image (fun y => HMul.hMul a y) V) a_1 → Mem …
    -/
    rintro _ ⟨x, hx, rfl⟩
    calc
      (a * x) • V ∈ (a * x) • B.filter := smul_set_mem_smul_filter <| B.mem_filter_of_mem V_in
      _ = a • x • V := smul_smul .. |>.symm
      _ ⊆ a • (V * V) := smul_set_mono <| smul_set_subset_smul hx
      _ ⊆ a • U := smul_set_mono hVU


@[to_additive]
theorem nhds_one_eq (B : GroupFilterBasis G) :
    @nhds G B.topology (1 : G) = B.toFilterBasis.filter := by
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    ⊢ Eq (nhds 1) GroupFilterBasis.toFilterBasis.filter
  -/
  rw [B.nhds_eq]
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    ⊢ Eq (B.N 1) GroupFilterBasis.toFilterBasis.filter
  -/
  simp only [N, one_mul]
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    ⊢ Eq (Filter.map (fun y => y) GroupFilterBasis.toFilterBasis.filter) GroupFilt …
  -/
  exact map_id
  /-
    🎉 no goals
  -/


@[to_additive]
theorem nhds_hasBasis (B : GroupFilterBasis G) (x₀ : G) :
    HasBasis (@nhds G B.topology x₀) (fun V : Set G ↦ V ∈ B) fun V ↦ (fun y ↦ x₀ * y) '' V := by
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    x₀ : G
    ⊢ (nhds x₀).HasBasis (fun V => Membership.mem B V) fun V => Set.image (fun y = …
  -/
  rw [B.nhds_eq]
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    x₀ : G
    ⊢ (B.N x₀).HasBasis (fun V => Membership.mem B V) fun V => Set.image (fun y => …
  -/
  apply B.hasBasis
  /-
    🎉 no goals
  -/


@[to_additive]
theorem nhds_one_hasBasis (B : GroupFilterBasis G) :
    HasBasis (@nhds G B.topology 1) (fun V : Set G ↦ V ∈ B) id := by
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    ⊢ (nhds 1).HasBasis (fun V => Membership.mem B V) id
  -/
  rw [B.nhds_one_eq]
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    ⊢ GroupFilterBasis.toFilterBasis.filter.HasBasis (fun V => Membership.mem B V) …
  -/
  exact B.toFilterBasis.hasBasis
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_nhds_one (B : GroupFilterBasis G) {U : Set G} (hU : U ∈ B) :
    U ∈ @nhds G B.topology 1 := by
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    U : Set G
    hU : Membership.mem B U
    ⊢ Membership.mem (nhds 1) U
  -/
  rw [B.nhds_one_hasBasis.mem_iff]
  /-
    G : Type u
    inst✝ : Group G
    B : GroupFilterBasis G
    U : Set G
    hU : Membership.mem B U
    ⊢ Exists fun i => And (Membership.mem B i) (HasSubset.Subset (id i) U)
  -/
  exact ⟨U, hU, rfl.subset⟩
  /-
    🎉 no goals
  -/

-- See note [lower instance priority]

/-- If a group is endowed with a topological structure coming from a group filter basis then it's a
topological group. -/
@[to_additive "If a group is endowed with a topological structure coming from a group filter basis
then it's a topological group."]
instance (priority := 100) isTopologicalGroup (B : GroupFilterBasis G) :
    @TopologicalGroup G B.topology _ := by
  /-
    G : Type u
    inst✝ : Group G
    B✝ B : GroupFilterBasis G
    ⊢ TopologicalGroup G
  -/
  letI := B.topology
  /-
    G : Type u
    inst✝ : Group G
    B✝ B : GroupFilterBasis G
    this : TopologicalSpace G := B.topology
    ⊢ TopologicalGroup G
  -/
  have basis := B.nhds_one_hasBasis
  /-
    G : Type u
    inst✝ : Group G
    B✝ B : GroupFilterBasis G
    this : TopologicalSpace G := B.topology
    basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
    ⊢ TopologicalGroup G
  -/
  have basis' := basis.prod basis
  /-
    G : Type u
    inst✝ : Group G
    B✝ B : GroupFilterBasis G
    this : TopologicalSpace G := B.topology
    basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
    basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
    ⊢ TopologicalGroup G
  -/
  refine TopologicalGroup.of_nhds_one ?_ ?_ ?_ ?_
    /-
      case refine_1
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.sprod  …
    -/
  · rw [basis'.tendsto_iff basis]
    /-
      case refine_1
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ ∀ (ib : Set G), Membership.mem B ib → Exists fun ia => And (And (Membership. …
    -/
    suffices ∀ U ∈ B, ∃ V W, (V ∈ B ∧ W ∈ B) ∧ ∀ a b, a ∈ V → b ∈ W → a * b ∈ U by simpa
    /-
      case refine_1
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ ∀ (U : Set G), Membership.mem B U → Exists fun V => Exists fun W => And (And …
    -/
    intro U U_in
    /-
      case refine_1
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      U : Set G
      U_in : Membership.mem B U
      ⊢ Exists fun V => Exists fun W => And (And (Membership.mem B V) (Membership.me …
    -/
    rcases mul U_in with ⟨V, V_in, hV⟩
    /-
      case refine_1.intro.intro
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      U : Set G
      U_in : Membership.mem B U
      V : Set G
      V_in : Membership.mem B V
      hV : HasSubset.Subset (HMul.hMul V V) U
      ⊢ Exists fun V => Exists fun W => And (And (Membership.mem B V) (Membership.me …
    -/
    refine ⟨V, V, ⟨V_in, V_in⟩, ?_⟩
    /-
      case refine_1.intro.intro
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      U : Set G
      U_in : Membership.mem B U
      V : Set G
      V_in : Membership.mem B V
      hV : HasSubset.Subset (HMul.hMul V V) U
      ⊢ ∀ (a b : G), Membership.mem V a → Membership.mem V b → Membership.mem U (HMu …
    -/
    intro a b a_in b_in
    /-
      case refine_1.intro.intro
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      U : Set G
      U_in : Membership.mem B U
      V : Set G
      V_in : Membership.mem B V
      hV : HasSubset.Subset (HMul.hMul V V) U
      a b : G
      a_in : Membership.mem V a
      b_in : Membership.mem V b
      ⊢ Membership.mem U (HMul.hMul a b)
    -/
    exact hV <| mul_mem_mul a_in b_in
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ Filter.Tendsto (fun x => Inv.inv x) (nhds 1) (nhds 1)
    -/
  · rw [basis.tendsto_iff basis]
    /-
      case refine_2
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ ∀ (ib : Set G), Membership.mem B ib → Exists fun ia => And (Membership.mem B …
    -/
    intro U U_in
    /-
      case refine_2
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      U : Set G
      U_in : Membership.mem B U
      ⊢ Exists fun ia => And (Membership.mem B ia) (∀ (x : G), Membership.mem (id ia …
    -/
    simpa using inv U_in
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ ∀ (x₀ : G), Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
    -/
  · intro x₀
    /-
      case refine_3
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      x₀ : G
      ⊢ Eq (nhds x₀) (Filter.map (fun x => HMul.hMul x₀ x) (nhds 1))
    -/
    rw [nhds_eq, nhds_one_eq]
    /-
      case refine_3
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      x₀ : G
      ⊢ Eq (B.N x₀) (Filter.map (fun x => HMul.hMul x₀ x) GroupFilterBasis.toFilterB …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      ⊢ ∀ (x₀ : G), Filter.Tendsto (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.inv x₀) …
    -/
  · intro x₀
    /-
      case refine_4
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      x₀ : G
      ⊢ Filter.Tendsto (fun x => HMul.hMul (HMul.hMul x₀ x) (Inv.inv x₀)) (nhds 1) ( …
    -/
    rw [basis.tendsto_iff basis]
    /-
      case refine_4
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      x₀ : G
      ⊢ ∀ (ib : Set G), Membership.mem B ib → Exists fun ia => And (Membership.mem B …
    -/
    intro U U_in
    /-
      case refine_4
      G : Type u
      inst✝ : Group G
      B✝ B : GroupFilterBasis G
      this : TopologicalSpace G := B.topology
      basis : (nhds 1).HasBasis (fun V => Membership.mem B V) id
      basis' : (SProd.sprod (nhds 1) (nhds 1)).HasBasis (fun i => And (Membership.me …
      x₀ : G
      U : Set G
      U_in : Membership.mem B U
      ⊢ Exists fun ia => And (Membership.mem B ia) (∀ (x : G), Membership.mem (id ia …
    -/
    exact conj x₀ U_in
    /-
      🎉 no goals
    -/


/-- A `RingFilterBasis` on a ring is a `FilterBasis` satisfying some additional axioms.
  Example : if `R` is a topological ring then the neighbourhoods of the identity are a
  `RingFilterBasis`. Conversely given a `RingFilterBasis` on a ring `R`, one can define a
  topology on `R` which is compatible with the ring structure. -/
class RingFilterBasis (R : Type u) [Ring R] extends AddGroupFilterBasis R where
  mul' : ∀ {U}, U ∈ sets → ∃ V ∈ sets, V * V ⊆ U
  mul_left' : ∀ (x₀ : R) {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ x₀ * x) ⁻¹' U
  mul_right' : ∀ (x₀ : R) {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ x * x₀) ⁻¹' U


instance : Membership (Set R) (RingFilterBasis R) :=
  ⟨fun B s ↦ s ∈ B.sets⟩


theorem mul {U : Set R} (hU : U ∈ B) : ∃ V ∈ B, V * V ⊆ U :=
  mul' hU


theorem mul_left (x₀ : R) {U : Set R} (hU : U ∈ B) : ∃ V ∈ B, V ⊆ (fun x ↦ x₀ * x) ⁻¹' U :=
  mul_left' x₀ hU


theorem mul_right (x₀ : R) {U : Set R} (hU : U ∈ B) : ∃ V ∈ B, V ⊆ (fun x ↦ x * x₀) ⁻¹' U :=
  mul_right' x₀ hU


/-- The topology associated to a ring filter basis.
It has the given basis as a basis of neighborhoods of zero. -/
def topology : TopologicalSpace R :=
  B.toAddGroupFilterBasis.topology


/-- If a ring is endowed with a topological structure coming from
a ring filter basis then it's a topological ring. -/
instance (priority := 100) isTopologicalRing {R : Type u} [Ring R] (B : RingFilterBasis R) :
    @TopologicalRing R B.topology _ := by
  /-
    R✝ : Type u
    inst✝¹ : Ring R✝
    B✝ : RingFilterBasis R✝
    R : Type u
    inst✝ : Ring R
    B : RingFilterBasis R
    ⊢ TopologicalRing R
  -/
  let B' := B.toAddGroupFilterBasis
  /-
    R✝ : Type u
    inst✝¹ : Ring R✝
    B✝ : RingFilterBasis R✝
    R : Type u
    inst✝ : Ring R
    B : RingFilterBasis R
    B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
    ⊢ TopologicalRing R
  -/
  letI := B'.topology
  /-
    R✝ : Type u
    inst✝¹ : Ring R✝
    B✝ : RingFilterBasis R✝
    R : Type u
    inst✝ : Ring R
    B : RingFilterBasis R
    B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
    this : TopologicalSpace R := B'.topology
    ⊢ TopologicalRing R
  -/
  have basis := B'.nhds_zero_hasBasis
  /-
    R✝ : Type u
    inst✝¹ : Ring R✝
    B✝ : RingFilterBasis R✝
    R : Type u
    inst✝ : Ring R
    B : RingFilterBasis R
    B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
    this : TopologicalSpace R := B'.topology
    basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
    ⊢ TopologicalRing R
  -/
  have basis' := basis.prod basis
  /-
    R✝ : Type u
    inst✝¹ : Ring R✝
    B✝ : RingFilterBasis R✝
    R : Type u
    inst✝ : Ring R
    B : RingFilterBasis R
    B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
    this : TopologicalSpace R := B'.topology
    basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
    basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
    ⊢ TopologicalRing R
  -/
  haveI := B'.isTopologicalAddGroup
  /-
    R✝ : Type u
    inst✝¹ : Ring R✝
    B✝ : RingFilterBasis R✝
    R : Type u
    inst✝ : Ring R
    B : RingFilterBasis R
    B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
    this✝ : TopologicalSpace R := B'.topology
    basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
    basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
    this : TopologicalAddGroup R
    ⊢ TopologicalRing R
  -/
  apply TopologicalRing.of_addGroup_of_nhds_zero
    /-
      case hmul
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      ⊢ Filter.Tendsto (Function.uncurry fun x1 x2 => HMul.hMul x1 x2) (SProd.sprod  …
    -/
  · rw [basis'.tendsto_iff basis]
    /-
      case hmul
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      ⊢ ∀ (ib : Set R), Membership.mem B' ib → Exists fun ia => And (And (Membership …
    -/
    suffices ∀ U ∈ B', ∃ V W, (V ∈ B' ∧ W ∈ B') ∧ ∀ a b, a ∈ V → b ∈ W → a * b ∈ U by simpa
    /-
      case hmul
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      ⊢ ∀ (U : Set R), Membership.mem B' U → Exists fun V => Exists fun W => And (An …
    -/
    intro U U_in
    /-
      case hmul
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      U : Set R
      U_in : Membership.mem B' U
      ⊢ Exists fun V => Exists fun W => And (And (Membership.mem B' V) (Membership.m …
    -/
    rcases B.mul U_in with ⟨V, V_in, hV⟩
    /-
      case hmul.intro.intro
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      U : Set R
      U_in : Membership.mem B' U
      V : Set R
      V_in : Membership.mem B V
      hV : HasSubset.Subset (HMul.hMul V V) U
      ⊢ Exists fun V => Exists fun W => And (And (Membership.mem B' V) (Membership.m …
    -/
    refine ⟨V, V, ⟨V_in, V_in⟩, ?_⟩
    /-
      case hmul.intro.intro
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      U : Set R
      U_in : Membership.mem B' U
      V : Set R
      V_in : Membership.mem B V
      hV : HasSubset.Subset (HMul.hMul V V) U
      ⊢ ∀ (a b : R), Membership.mem V a → Membership.mem V b → Membership.mem U (HMu …
    -/
    intro a b a_in b_in
    /-
      case hmul.intro.intro
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      U : Set R
      U_in : Membership.mem B' U
      V : Set R
      V_in : Membership.mem B V
      hV : HasSubset.Subset (HMul.hMul V V) U
      a b : R
      a_in : Membership.mem V a
      b_in : Membership.mem V b
      ⊢ Membership.mem U (HMul.hMul a b)
    -/
    exact hV <| mul_mem_mul a_in b_in
    /-
      🎉 no goals
    -/
    /-
      case hmul_left
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      ⊢ ∀ (x₀ : R), Filter.Tendsto (fun x => HMul.hMul x₀ x) (nhds 0) (nhds 0)
    -/
  · intro x₀
    /-
      case hmul_left
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      x₀ : R
      ⊢ Filter.Tendsto (fun x => HMul.hMul x₀ x) (nhds 0) (nhds 0)
    -/
    rw [basis.tendsto_iff basis]
    /-
      case hmul_left
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      x₀ : R
      ⊢ ∀ (ib : Set R), Membership.mem B' ib → Exists fun ia => And (Membership.mem  …
    -/
    intro U
    /-
      case hmul_left
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      x₀ : R
      U : Set R
      ⊢ Membership.mem B' U → Exists fun ia => And (Membership.mem B' ia) (∀ (x : R) …
    -/
    simpa using B.mul_left x₀
    /-
      🎉 no goals
    -/
    /-
      case hmul_right
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      ⊢ ∀ (x₀ : R), Filter.Tendsto (fun x => HMul.hMul x x₀) (nhds 0) (nhds 0)
    -/
  · intro x₀
    /-
      case hmul_right
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      x₀ : R
      ⊢ Filter.Tendsto (fun x => HMul.hMul x x₀) (nhds 0) (nhds 0)
    -/
    rw [basis.tendsto_iff basis]
    /-
      case hmul_right
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      x₀ : R
      ⊢ ∀ (ib : Set R), Membership.mem B' ib → Exists fun ia => And (Membership.mem  …
    -/
    intro U
    /-
      case hmul_right
      R✝ : Type u
      inst✝¹ : Ring R✝
      B✝ : RingFilterBasis R✝
      R : Type u
      inst✝ : Ring R
      B : RingFilterBasis R
      B' : AddGroupFilterBasis R := RingFilterBasis.toAddGroupFilterBasis
      this✝ : TopologicalSpace R := B'.topology
      basis : (nhds 0).HasBasis (fun V => Membership.mem B' V) id
      basis' : (SProd.sprod (nhds 0) (nhds 0)).HasBasis (fun i => And (Membership.me …
      this : TopologicalAddGroup R
      x₀ : R
      U : Set R
      ⊢ Membership.mem B' U → Exists fun ia => And (Membership.mem B' ia) (∀ (x : R) …
    -/
    simpa using B.mul_right x₀
    /-
      🎉 no goals
    -/


/-- A `ModuleFilterBasis` on a module is a `FilterBasis` satisfying some additional axioms.
  Example : if `M` is a topological module then the neighbourhoods of zero are a
  `ModuleFilterBasis`. Conversely given a `ModuleFilterBasis` one can define a topology
  compatible with the module structure on `M`. -/
structure ModuleFilterBasis (R M : Type*) [CommRing R] [TopologicalSpace R] [AddCommGroup M]
  [Module R M] extends AddGroupFilterBasis M where
  smul' : ∀ {U}, U ∈ sets → ∃ V ∈ 𝓝 (0 : R), ∃ W ∈ sets, V • W ⊆ U
  smul_left' : ∀ (x₀ : R) {U}, U ∈ sets → ∃ V ∈ sets, V ⊆ (fun x ↦ x₀ • x) ⁻¹' U
  smul_right' : ∀ (m₀ : M) {U}, U ∈ sets → ∀ᶠ x in 𝓝 (0 : R), x • m₀ ∈ U


instance GroupFilterBasis.hasMem : Membership (Set M) (ModuleFilterBasis R M) :=
  ⟨fun B s ↦ s ∈ B.sets⟩


theorem smul {U : Set M} (hU : U ∈ B) : ∃ V ∈ 𝓝 (0 : R), ∃ W ∈ B, V • W ⊆ U :=
  B.smul' hU


theorem smul_left (x₀ : R) {U : Set M} (hU : U ∈ B) : ∃ V ∈ B, V ⊆ (fun x ↦ x₀ • x) ⁻¹' U :=
  B.smul_left' x₀ hU


theorem smul_right (m₀ : M) {U : Set M} (hU : U ∈ B) : ∀ᶠ x in 𝓝 (0 : R), x • m₀ ∈ U :=
  B.smul_right' m₀ hU


/-- If `R` is discrete then the trivial additive group filter basis on any `R`-module is a
module filter basis. -/
instance [DiscreteTopology R] : Inhabited (ModuleFilterBasis R M) :=
  ⟨{
      show AddGroupFilterBasis M from
        default with
      smul' := by
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          ⊢ ∀ {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.sets U → Exi …
        -/
        rintro U (rfl : U ∈ {{(0 : M)}})
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun W => And (Member …
        -/
        use univ, univ_mem, {0}, rfl
        /-
          case right
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          ⊢ HasSubset.Subset (HSMul.hSMul Set.univ (Singleton.singleton 0)) (Singleton.s …
        -/
        rintro a ⟨x, -, m, rfl, rfl⟩
        /-
          case right.intro.intro.intro.intro
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          x : R
          ⊢ Membership.mem (Singleton.singleton 0) ((fun x1 x2 => HSMul.hSMul x1 x2) x 0)
        -/
        simp only [smul_zero, mem_singleton_iff]
        /-
          🎉 no goals
        -/
      smul_left' := by
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          ⊢ ∀ (x₀ : R) {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
        -/
        rintro x₀ U (h : U ∈ {{(0 : M)}})
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          x₀ : R
          U : Set M
          h : Membership.mem (Singleton.singleton (Singleton.singleton 0)) U
          ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
        -/
        rw [mem_singleton_iff] at h
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          x₀ : R
          U : Set M
          h : Eq U (Singleton.singleton 0)
          ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
        -/
        use {0}, rfl
        /-
          case right
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          x₀ : R
          U : Set M
          h : Eq U (Singleton.singleton 0)
          ⊢ HasSubset.Subset (Singleton.singleton 0) (Set.preimage (fun x => HSMul.hSMul …
        -/
        simp [h]
        /-
          🎉 no goals
        -/
      smul_right' := by
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          ⊢ ∀ (m₀ : M) {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
        -/
        rintro m₀ U (h : U ∈ (0 : Set (Set M)))
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          m₀ : M
          U : Set M
          h : Membership.mem 0 U
          ⊢ Filter.Eventually (fun x => Membership.mem U (HSMul.hSMul x m₀)) (nhds 0)
        -/
        rw [Set.mem_zero] at h
        /-
          R : Type u_1
          M : Type u_2
          inst✝⁴ : CommRing R
          inst✝³ : TopologicalSpace R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          B : ModuleFilterBasis R M
          inst✝ : DiscreteTopology R
          m₀ : M
          U : Set M
          h : Eq U 0
          ⊢ Filter.Eventually (fun x => Membership.mem U (HSMul.hSMul x m₀)) (nhds 0)
        -/
        simp [h, nhds_discrete] }⟩
        /-
          🎉 no goals
        -/


/-- The topology associated to a module filter basis on a module over a topological ring.
It has the given basis as a basis of neighborhoods of zero. -/
def topology : TopologicalSpace M :=
  B.toAddGroupFilterBasis.topology


/-- The topology associated to a module filter basis on a module over a topological ring.
It has the given basis as a basis of neighborhoods of zero. This version gets the ring
topology by unification instead of type class inference. -/
def topology' {R M : Type*} [CommRing R] {_ : TopologicalSpace R} [AddCommGroup M] [Module R M]
    (B : ModuleFilterBasis R M) : TopologicalSpace M :=
  B.toAddGroupFilterBasis.topology


/-- A topological add group with a basis of `𝓝 0` satisfying the axioms of `ModuleFilterBasis`
is a topological module.

This lemma is mathematically useless because one could obtain such a result by applying
`ModuleFilterBasis.continuousSMul` and use the fact that group topologies are characterized
by their neighborhoods of 0 to obtain the `ContinuousSMul` on the pre-existing topology.

But it turns out it's just easier to get it as a byproduct of the proof, so this is just a free
quality-of-life improvement. -/
theorem _root_.ContinuousSMul.of_basis_zero {ι : Type*} [TopologicalRing R] [TopologicalSpace M]
    [TopologicalAddGroup M] {p : ι → Prop} {b : ι → Set M} (h : HasBasis (𝓝 0) p b)
    (hsmul : ∀ {i}, p i → ∃ V ∈ 𝓝 (0 : R), ∃ j, p j ∧ V • b j ⊆ b i)
    (hsmul_left : ∀ (x₀ : R) {i}, p i → ∃ j, p j ∧ MapsTo (x₀ • ·) (b j) (b i))
    (hsmul_right : ∀ (m₀ : M) {i}, p i → ∀ᶠ x in 𝓝 (0 : R), x • m₀ ∈ b i) : ContinuousSMul R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : TopologicalRing R
    inst✝¹ : TopologicalSpace M
    inst✝ : TopologicalAddGroup M
    p : ι → Prop
    b : ι → Set M
    h : (nhds 0).HasBasis p b
    hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
    hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
    hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
    ⊢ ContinuousSMul R M
  -/
  apply ContinuousSMul.of_nhds_zero
    /-
      case hmul
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      ⊢ Filter.Tendsto (fun p => HSMul.hSMul p.1 p.2) (SProd.sprod (nhds 0) (nhds 0) …
    -/
  · rw [h.tendsto_right_iff]
    /-
      case hmul
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      ⊢ ∀ (i : ι), p i → Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSM …
    -/
    intro i hi
    /-
      case hmul
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      i : ι
      hi : p i
      ⊢ Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSMul x.1 x.2)) (SPr …
    -/
    rcases hsmul hi with ⟨V, V_in, j, hj, hVj⟩
    /-
      case hmul.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      i : ι
      hi : p i
      V : Set R
      V_in : Membership.mem (nhds 0) V
      j : ι
      hj : p j
      hVj : HasSubset.Subset (HSMul.hSMul V (b j)) (b i)
      ⊢ Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSMul x.1 x.2)) (SPr …
    -/
    apply mem_of_superset (prod_mem_prod V_in <| h.mem_of_mem hj)
    /-
      case hmul.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      i : ι
      hi : p i
      V : Set R
      V_in : Membership.mem (nhds 0) V
      j : ι
      hj : p j
      hVj : HasSubset.Subset (HSMul.hSMul V (b j)) (b i)
      ⊢ HasSubset.Subset (SProd.sprod V (b j)) (setOf fun x => (fun x => Membership. …
    -/
    rintro ⟨v, w⟩ ⟨v_in : v ∈ V, w_in : w ∈ b j⟩
    /-
      case hmul.intro.intro.intro.intro.mk.intro
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      i : ι
      hi : p i
      V : Set R
      V_in : Membership.mem (nhds 0) V
      j : ι
      hj : p j
      hVj : HasSubset.Subset (HSMul.hSMul V (b j)) (b i)
      v : R
      w : M
      v_in : Membership.mem V v
      w_in : Membership.mem (b j) w
      ⊢ Membership.mem (setOf fun x => (fun x => Membership.mem (b i) (HSMul.hSMul x …
    -/
    exact hVj (Set.smul_mem_smul v_in w_in)
    /-
      🎉 no goals
    -/
    /-
      case hmulleft
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      ⊢ ∀ (m : M), Filter.Tendsto (fun a => HSMul.hSMul a m) (nhds 0) (nhds 0)
    -/
  · intro m₀
    /-
      case hmulleft
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      m₀ : M
      ⊢ Filter.Tendsto (fun a => HSMul.hSMul a m₀) (nhds 0) (nhds 0)
    -/
    rw [h.tendsto_right_iff]
    /-
      case hmulleft
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      m₀ : M
      ⊢ ∀ (i : ι), p i → Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSM …
    -/
    intro i hi
    /-
      case hmulleft
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      m₀ : M
      i : ι
      hi : p i
      ⊢ Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSMul x m₀)) (nhds 0)
    -/
    exact hsmul_right m₀ hi
    /-
      🎉 no goals
    -/
    /-
      case hmulright
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      ⊢ ∀ (a : R), Filter.Tendsto (fun m => HSMul.hSMul a m) (nhds 0) (nhds 0)
    -/
  · intro x₀
    /-
      case hmulright
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      x₀ : R
      ⊢ Filter.Tendsto (fun m => HSMul.hSMul x₀ m) (nhds 0) (nhds 0)
    -/
    rw [h.tendsto_right_iff]
    /-
      case hmulright
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      x₀ : R
      ⊢ ∀ (i : ι), p i → Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSM …
    -/
    intro i hi
    /-
      case hmulright
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      x₀ : R
      i : ι
      hi : p i
      ⊢ Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSMul x₀ x)) (nhds 0)
    -/
    rcases hsmul_left x₀ hi with ⟨j, hj, hji⟩
    /-
      case hmulright.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : TopologicalRing R
      inst✝¹ : TopologicalSpace M
      inst✝ : TopologicalAddGroup M
      p : ι → Prop
      b : ι → Set M
      h : (nhds 0).HasBasis p b
      hsmul : ∀ {i : ι}, p i → Exists fun V => And (Membership.mem (nhds 0) V) (Exis …
      hsmul_left : ∀ (x₀ : R) {i : ι}, p i → Exists fun j => And (p j) (Set.MapsTo ( …
      hsmul_right : ∀ (m₀ : M) {i : ι}, p i → Filter.Eventually (fun x => Membership …
      x₀ : R
      i : ι
      hi : p i
      j : ι
      hj : p j
      hji : Set.MapsTo (fun x => HSMul.hSMul x₀ x) (b j) (b i)
      ⊢ Filter.Eventually (fun x => Membership.mem (b i) (HSMul.hSMul x₀ x)) (nhds 0)
    -/
    exact mem_of_superset (h.mem_of_mem hj) hji
    /-
      🎉 no goals
    -/


/-- If a module is endowed with a topological structure coming from
a module filter basis then it's a topological module. -/
instance (priority := 100) continuousSMul [TopologicalRing R] :
    @ContinuousSMul R M _ _ B.topology := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : TopologicalSpace R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    B : ModuleFilterBasis R M
    inst✝ : TopologicalRing R
    ⊢ ContinuousSMul R M
  -/
  let B' := B.toAddGroupFilterBasis
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : TopologicalSpace R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    B : ModuleFilterBasis R M
    inst✝ : TopologicalRing R
    B' : AddGroupFilterBasis M := B.toAddGroupFilterBasis
    ⊢ ContinuousSMul R M
  -/
  let _ := B'.topology
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : TopologicalSpace R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    B : ModuleFilterBasis R M
    inst✝ : TopologicalRing R
    B' : AddGroupFilterBasis M := B.toAddGroupFilterBasis
    x✝ : TopologicalSpace M := B'.topology
    ⊢ ContinuousSMul R M
  -/
  have _ := B'.isTopologicalAddGroup
  exact ContinuousSMul.of_basis_zero B'.nhds_zero_hasBasis
      (fun {_} => by simpa using B.smul)
      (by simpa using B.smul_left) B.smul_right


/-- Build a module filter basis from compatible ring and additive group filter bases. -/
def ofBases {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M] (BR : RingFilterBasis R)
    (BM : AddGroupFilterBasis M) (smul : ∀ {U}, U ∈ BM → ∃ V ∈ BR, ∃ W ∈ BM, V • W ⊆ U)
    (smul_left : ∀ (x₀ : R) {U}, U ∈ BM → ∃ V ∈ BM, V ⊆ (fun x ↦ x₀ • x) ⁻¹' U)
    (smul_right : ∀ (m₀ : M) {U}, U ∈ BM → ∃ V ∈ BR, V ⊆ (fun x ↦ x • m₀) ⁻¹' U) :
    @ModuleFilterBasis R M _ BR.topology _ _ :=
  let _ := BR.topology
  { BM with
    smul' := by
      /-
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝⁶ : CommRing R✝
        inst✝⁵ : TopologicalSpace R✝
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        B : ModuleFilterBasis R✝ M✝
        R : Type u_3
        M : Type u_4
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        BM : AddGroupFilterBasis M
        smul : ∀ {U : Set M}, Membership.mem BM U → Exists fun V => And (Membership.me …
        smul_left : ∀ (x₀ : R) {U : Set M}, Membership.mem BM U → Exists fun V => And  …
        smul_right : ∀ (m₀ : M) {U : Set M}, Membership.mem BM U → Exists fun V => And …
        x✝ : TopologicalSpace R := BR.topology
        ⊢ ∀ {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.sets U → Exi …
      -/
      intro U U_in
      /-
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝⁶ : CommRing R✝
        inst✝⁵ : TopologicalSpace R✝
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        B : ModuleFilterBasis R✝ M✝
        R : Type u_3
        M : Type u_4
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        BM : AddGroupFilterBasis M
        smul : ∀ {U : Set M}, Membership.mem BM U → Exists fun V => And (Membership.me …
        smul_left : ∀ (x₀ : R) {U : Set M}, Membership.mem BM U → Exists fun V => And  …
        smul_right : ∀ (m₀ : M) {U : Set M}, Membership.mem BM U → Exists fun V => And …
        x✝ : TopologicalSpace R := BR.topology
        U : Set M
        U_in : Membership.mem AddGroupFilterBasis.toFilterBasis.sets U
        ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun W => And (Member …
      -/
      rcases smul U_in with ⟨V, V_in, W, W_in, H⟩
      /-
        case intro.intro.intro.intro
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝⁶ : CommRing R✝
        inst✝⁵ : TopologicalSpace R✝
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        B : ModuleFilterBasis R✝ M✝
        R : Type u_3
        M : Type u_4
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        BM : AddGroupFilterBasis M
        smul : ∀ {U : Set M}, Membership.mem BM U → Exists fun V => And (Membership.me …
        smul_left : ∀ (x₀ : R) {U : Set M}, Membership.mem BM U → Exists fun V => And  …
        smul_right : ∀ (m₀ : M) {U : Set M}, Membership.mem BM U → Exists fun V => And …
        x✝ : TopologicalSpace R := BR.topology
        U : Set M
        U_in : Membership.mem AddGroupFilterBasis.toFilterBasis.sets U
        V : Set R
        V_in : Membership.mem BR V
        W : Set M
        W_in : Membership.mem BM W
        H : HasSubset.Subset (HSMul.hSMul V W) U
        ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun W => And (Member …
      -/
      exact ⟨V, BR.toAddGroupFilterBasis.mem_nhds_zero V_in, W, W_in, H⟩
      /-
        🎉 no goals
      -/
    smul_left' := smul_left
    smul_right' := by
      /-
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝⁶ : CommRing R✝
        inst✝⁵ : TopologicalSpace R✝
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        B : ModuleFilterBasis R✝ M✝
        R : Type u_3
        M : Type u_4
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        BM : AddGroupFilterBasis M
        smul : ∀ {U : Set M}, Membership.mem BM U → Exists fun V => And (Membership.me …
        smul_left : ∀ (x₀ : R) {U : Set M}, Membership.mem BM U → Exists fun V => And  …
        smul_right : ∀ (m₀ : M) {U : Set M}, Membership.mem BM U → Exists fun V => And …
        x✝ : TopologicalSpace R := BR.topology
        ⊢ ∀ (m₀ : M) {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
      -/
      intro m₀ U U_in
      /-
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝⁶ : CommRing R✝
        inst✝⁵ : TopologicalSpace R✝
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        B : ModuleFilterBasis R✝ M✝
        R : Type u_3
        M : Type u_4
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        BM : AddGroupFilterBasis M
        smul : ∀ {U : Set M}, Membership.mem BM U → Exists fun V => And (Membership.me …
        smul_left : ∀ (x₀ : R) {U : Set M}, Membership.mem BM U → Exists fun V => And  …
        smul_right : ∀ (m₀ : M) {U : Set M}, Membership.mem BM U → Exists fun V => And …
        x✝ : TopologicalSpace R := BR.topology
        m₀ : M
        U : Set M
        U_in : Membership.mem AddGroupFilterBasis.toFilterBasis.sets U
        ⊢ Filter.Eventually (fun x => Membership.mem U (HSMul.hSMul x m₀)) (nhds 0)
      -/
      rcases smul_right m₀ U_in with ⟨V, V_in, H⟩
      /-
        case intro.intro
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝⁶ : CommRing R✝
        inst✝⁵ : TopologicalSpace R✝
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        B : ModuleFilterBasis R✝ M✝
        R : Type u_3
        M : Type u_4
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        BM : AddGroupFilterBasis M
        smul : ∀ {U : Set M}, Membership.mem BM U → Exists fun V => And (Membership.me …
        smul_left : ∀ (x₀ : R) {U : Set M}, Membership.mem BM U → Exists fun V => And  …
        smul_right : ∀ (m₀ : M) {U : Set M}, Membership.mem BM U → Exists fun V => And …
        x✝ : TopologicalSpace R := BR.topology
        m₀ : M
        U : Set M
        U_in : Membership.mem AddGroupFilterBasis.toFilterBasis.sets U
        V : Set R
        V_in : Membership.mem BR V
        H : HasSubset.Subset V (Set.preimage (fun x => HSMul.hSMul x m₀) U)
        ⊢ Filter.Eventually (fun x => Membership.mem U (HSMul.hSMul x m₀)) (nhds 0)
      -/
      exact mem_of_superset (BR.toAddGroupFilterBasis.mem_nhds_zero V_in) H }
      /-
        🎉 no goals
      -/


