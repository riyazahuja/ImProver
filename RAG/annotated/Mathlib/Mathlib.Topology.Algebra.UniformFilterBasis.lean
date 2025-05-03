/-- The uniform space structure associated to an abelian group filter basis via the associated
topological abelian group structure. -/
protected def uniformSpace : UniformSpace G :=
  @TopologicalAddGroup.toUniformSpace G _ B.topology B.isTopologicalAddGroup


/-- The uniform space structure associated to an abelian group filter basis via the associated
topological abelian group structure is compatible with its group structure. -/
protected theorem uniformAddGroup : @UniformAddGroup G B.uniformSpace _ :=
  @comm_topologicalAddGroup_is_uniform G _ B.topology B.isTopologicalAddGroup


theorem cauchy_iff {F : Filter G} :
    @Cauchy G B.uniformSpace F ↔
      F.NeBot ∧ ∀ U ∈ B, ∃ M ∈ F, ∀ᵉ (x ∈ M) (y ∈ M), y - x ∈ U := by
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    B : AddGroupFilterBasis G
    F : Filter G
    ⊢ Iff (Cauchy F) (And F.NeBot (∀ (U : Set G), Membership.mem B U → Exists fun  …
  -/
  letI := B.uniformSpace
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    B : AddGroupFilterBasis G
    F : Filter G
    this : UniformSpace G := B.uniformSpace
    ⊢ Iff (Cauchy F) (And F.NeBot (∀ (U : Set G), Membership.mem B U → Exists fun  …
  -/
  haveI := B.uniformAddGroup
  suffices F ×ˢ F ≤ uniformity G ↔ ∀ U ∈ B, ∃ M ∈ F, ∀ᵉ (x ∈ M) (y ∈ M), y - x ∈ U by
    constructor <;> rintro ⟨h', h⟩ <;> refine ⟨h', ?_⟩ <;> [rwa [← this]; rwa [this]]
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    B : AddGroupFilterBasis G
    F : Filter G
    this✝ : UniformSpace G := B.uniformSpace
    this : UniformAddGroup G
    ⊢ Iff (LE.le (SProd.sprod F F) (uniformity G)) (∀ (U : Set G), Membership.mem  …
  -/
  rw [uniformity_eq_comap_nhds_zero G, ← map_le_iff_le_comap]
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    B : AddGroupFilterBasis G
    F : Filter G
    this✝ : UniformSpace G := B.uniformSpace
    this : UniformAddGroup G
    ⊢ Iff (LE.le (Filter.map (fun x => HSub.hSub x.2 x.1) (SProd.sprod F F)) (nhds …
  -/
  change Tendsto _ _ _ ↔ _
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    B : AddGroupFilterBasis G
    F : Filter G
    this✝ : UniformSpace G := B.uniformSpace
    this : UniformAddGroup G
    ⊢ Iff (Filter.Tendsto (fun x => HSub.hSub x.2 x.1) (SProd.sprod F F) (nhds 0)) …
  -/
  simp [(basis_sets F).prod_self.tendsto_iff B.nhds_zero_hasBasis, @forall_swap (_ ∈ _) G]
  /-
    🎉 no goals
  -/


