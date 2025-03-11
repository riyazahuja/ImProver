variable (I) in
/-- `p ∈ M` is an interior point of a manifold `M` iff its image in the extended chart
lies in the interior of the model space. -/
def IsInteriorPoint (x : M) := extChartAt I x x ∈ interior (range I)


variable (I) in
/-- `p ∈ M` is a boundary point of a manifold `M` iff its image in the extended chart
lies on the boundary of the model space. -/
def IsBoundaryPoint (x : M) := extChartAt I x x ∈ frontier (range I)


variable (M) in
/-- The **interior** of a manifold `M` is the set of its interior points. -/
protected def interior : Set M := { x : M | I.IsInteriorPoint x }


lemma isInteriorPoint_iff {x : M} :
    I.IsInteriorPoint x ↔ extChartAt I x x ∈ interior (extChartAt I x).target :=
  ⟨fun h ↦ (chartAt H x).mem_interior_extend_target (mem_chart_target H x) h,
    fun h ↦ PartialHomeomorph.interior_extend_target_subset_interior_range _ h⟩


variable (M) in
/-- The **boundary** of a manifold `M` is the set of its boundary points. -/
protected def boundary : Set M := { x : M | I.IsBoundaryPoint x }


lemma isBoundaryPoint_iff {x : M} : I.IsBoundaryPoint x ↔ extChartAt I x x ∈ frontier (range I) :=
  Iff.rfl


/-- Every point is either an interior or a boundary point. -/
lemma isInteriorPoint_or_isBoundaryPoint (x : M) : I.IsInteriorPoint x ∨ I.IsBoundaryPoint x := by
  rw [IsInteriorPoint, or_iff_not_imp_left, I.isBoundaryPoint_iff, ← closure_diff_interior,
    I.isClosed_range.closure_eq, mem_diff]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    x : M
    ⊢ Not (Membership.mem (interior (Set.range ↑I)) (↑(extChartAt I x) x)) → And ( …
  -/
  exact fun h => ⟨mem_range_self _, h⟩
  /-
    🎉 no goals
  -/


/-- A manifold decomposes into interior and boundary. -/
lemma interior_union_boundary_eq_univ : (I.interior M) ∪ (I.boundary M) = (univ : Set M) :=
  eq_univ_of_forall fun x => (mem_union _ _ _).mpr (I.isInteriorPoint_or_isBoundaryPoint x)


/-- The interior and boundary of a manifold `M` are disjoint. -/
lemma disjoint_interior_boundary : Disjoint (I.interior M) (I.boundary M) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ Disjoint (ModelWithCorners.interior M) (ModelWithCorners.boundary M)
  -/
  by_contra h
  -- Choose some x in the intersection of interior and boundary.
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Not (Disjoint (ModelWithCorners.interior M) (ModelWithCorners.boundary M))
    ⊢ False
  -/
  obtain ⟨x, h1, h2⟩ := not_disjoint_iff.mp h
  rw [← mem_empty_iff_false (extChartAt I x x),
    ← disjoint_iff_inter_eq_empty.mp (disjoint_interior_frontier (s := range I)), mem_inter_iff]
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Not (Disjoint (ModelWithCorners.interior M) (ModelWithCorners.boundary M))
    x : M
    h1 : Membership.mem (ModelWithCorners.interior M) x
    h2 : Membership.mem (ModelWithCorners.boundary M) x
    ⊢ And (Membership.mem (interior (Set.range ↑I)) (↑(extChartAt I x) x)) (Member …
  -/
  exact ⟨h1, h2⟩
  /-
    🎉 no goals
  -/


/-- The boundary is the complement of the interior. -/
lemma compl_interior : (I.interior M)ᶜ = I.boundary M:= by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ Eq (HasCompl.compl (ModelWithCorners.interior M)) (ModelWithCorners.boundary …
  -/
  apply compl_unique ?_ I.interior_union_boundary_eq_univ
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ Eq (Min.min (ModelWithCorners.interior M) (ModelWithCorners.boundary M)) Bot …
  -/
  exact disjoint_iff_inter_eq_empty.mp (I.disjoint_interior_boundary)
  /-
    🎉 no goals
  -/


/-- The interior is the complement of the boundary. -/
lemma compl_boundary : (I.boundary M)ᶜ = I.interior M:= by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ Eq (HasCompl.compl (ModelWithCorners.boundary M)) (ModelWithCorners.interior …
  -/
  rw [← compl_interior, compl_compl]
  /-
    🎉 no goals
  -/


lemma _root_.range_mem_nhds_isInteriorPoint {x : M} (h : I.IsInteriorPoint x) :
    range I ∈ 𝓝 (extChartAt I x x) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    x : M
    h : I.IsInteriorPoint x
    ⊢ Membership.mem (nhds (↑(extChartAt I x) x)) (Set.range ↑I)
  -/
  rw [mem_nhds_iff]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    x : M
    h : I.IsInteriorPoint x
    ⊢ Exists fun t => And (HasSubset.Subset t (Set.range ↑I)) (And (IsOpen t) (Mem …
  -/
  exact ⟨interior (range I), interior_subset, isOpen_interior, h⟩
  /-
    🎉 no goals
  -/


/-- Type class for manifold without boundary. This differs from `ModelWithCorners.Boundaryless`,
  which states that the `ModelWithCorners` maps to the whole model vector space. -/
class _root_.BoundarylessManifold {𝕜 : Type*} [NontriviallyNormedField 𝕜]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    {H : Type*} [TopologicalSpace H] (I : ModelWithCorners 𝕜 E H)
    (M : Type*) [TopologicalSpace M] [ChartedSpace H M] : Prop where
  isInteriorPoint' : ∀ x : M, IsInteriorPoint I x


/-- Boundaryless `ModelWithCorners` implies boundaryless manifold. -/
instance : BoundarylessManifold I M where
  isInteriorPoint' x := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : I.Boundaryless
      x : M
      ⊢ I.IsInteriorPoint x
    -/
    let r := ((chartAt H x).isOpen_extend_target (I := I)).interior_eq
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : I.Boundaryless
      x : M
      r : Eq (interior ((chartAt H x).extend I).target) ((chartAt H x).extend I).tar …
      ⊢ I.IsInteriorPoint x
    -/
    have : extChartAt I x = (chartAt H x).extend I := rfl
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : I.Boundaryless
      x : M
      r : Eq (interior ((chartAt H x).extend I).target) ((chartAt H x).extend I).tar …
      this : Eq (extChartAt I x) ((chartAt H x).extend I)
      ⊢ I.IsInteriorPoint x
    -/
    rw [← this] at r
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : I.Boundaryless
      x : M
      r : Eq (interior (extChartAt I x).target) (extChartAt I x).target
      this : Eq (extChartAt I x) ((chartAt H x).extend I)
      ⊢ I.IsInteriorPoint x
    -/
    rw [isInteriorPoint_iff, r]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : I.Boundaryless
      x : M
      r : Eq (interior (extChartAt I x).target) (extChartAt I x).target
      this : Eq (extChartAt I x) ((chartAt H x).extend I)
      ⊢ Membership.mem (extChartAt I x).target (↑(extChartAt I x) x)
    -/
    exact PartialEquiv.map_source _ (mem_extChartAt_source _)
    /-
      🎉 no goals
    -/


/-- The empty manifold is boundaryless. -/
instance BoundarylessManifold.of_empty [IsEmpty M] : BoundarylessManifold I M where
  isInteriorPoint' x := (IsEmpty.false x).elim


lemma _root_.BoundarylessManifold.isInteriorPoint {x : M} [BoundarylessManifold I M] :
    IsInteriorPoint I x := BoundarylessManifold.isInteriorPoint' x


/-- If `I` is boundaryless, `M` has full interior. -/
lemma interior_eq_univ [BoundarylessManifold I M] : I.interior M = univ :=
  eq_univ_of_forall fun _ => BoundarylessManifold.isInteriorPoint


/-- Boundaryless manifolds have empty boundary. -/
lemma Boundaryless.boundary_eq_empty [BoundarylessManifold I M] : I.boundary M = ∅ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : BoundarylessManifold I M
    ⊢ Eq (ModelWithCorners.boundary M) EmptyCollection.emptyCollection
  -/
  rw [← I.compl_interior, I.interior_eq_univ, compl_empty_iff]
  /-
    🎉 no goals
  -/


instance [BoundarylessManifold I M] : IsEmpty (I.boundary M) :=
  isEmpty_coe_sort.mpr Boundaryless.boundary_eq_empty


/-- `M` is boundaryless iff its boundary is empty. -/
lemma Boundaryless.iff_boundary_eq_empty : I.boundary M = ∅ ↔ BoundarylessManifold I M := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ Iff (Eq (ModelWithCorners.boundary M) EmptyCollection.emptyCollection) (Boun …
  -/
  refine ⟨fun h ↦ { isInteriorPoint' := ?_ }, fun a ↦ boundary_eq_empty⟩
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Eq (ModelWithCorners.boundary M) EmptyCollection.emptyCollection
    ⊢ ∀ (x : M), I.IsInteriorPoint x
  -/
  intro x
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Eq (ModelWithCorners.boundary M) EmptyCollection.emptyCollection
    x : M
    ⊢ I.IsInteriorPoint x
  -/
  show x ∈ I.interior M
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Eq (ModelWithCorners.boundary M) EmptyCollection.emptyCollection
    x : M
    ⊢ Membership.mem (ModelWithCorners.interior M) x
  -/
  rw [← compl_interior, compl_empty_iff] at h
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Eq (ModelWithCorners.interior M) Set.univ
    x : M
    ⊢ Membership.mem (ModelWithCorners.interior M) x
  -/
  rw [h]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    h : Eq (ModelWithCorners.interior M) Set.univ
    x : M
    ⊢ Membership.mem Set.univ x
  -/
  trivial
  /-
    🎉 no goals
  -/


/-- Manifolds with empty boundary are boundaryless. -/
lemma Boundaryless.of_boundary_eq_empty (h : I.boundary M = ∅) : BoundarylessManifold I M :=
  (Boundaryless.iff_boundary_eq_empty (I := I)).mp h


/-- The interior of `M × N` is the product of the interiors of `M` and `N`. -/
lemma interior_prod :
    (I.prod J).interior (M × N) = (I.interior M) ×ˢ (J.interior N) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    ⊢ Eq (ModelWithCorners.interior (Prod M N)) (SProd.sprod (ModelWithCorners.int …
  -/
  ext p
  have aux : (interior (range ↑I)) ×ˢ (interior (range J)) = interior (range (I.prod J)) := by
    rw [← interior_prod_eq, ← Set.range_prod_map, modelWithCorners_prod_coe]
  /-
    case h
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    p : Prod M N
    aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
    ⊢ Iff (Membership.mem (ModelWithCorners.interior (Prod M N)) p) (Membership.me …
  -/
  constructor <;> intro hp
    /-
      case h.mp
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (ModelWithCorners.interior (Prod M N)) p
      ⊢ Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorners. …
    -/
  · replace hp : (I.prod J).IsInteriorPoint p := hp
    /-
      case h.mp
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : (I.prod J).IsInteriorPoint p
      ⊢ Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorners. …
    -/
    rw [IsInteriorPoint, ← aux] at hp
    /-
      case h.mp
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (SProd.sprod (interior (Set.range ↑I)) (interior (Set.rang …
      ⊢ Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorners. …
    -/
    exact hp
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorne …
      ⊢ Membership.mem (ModelWithCorners.interior (Prod M N)) p
    -/
  · show (I.prod J).IsInteriorPoint p
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorne …
      ⊢ (I.prod J).IsInteriorPoint p
    -/
    rw [IsInteriorPoint, ← aux, mem_prod]
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorne …
      ⊢ And (Membership.mem (interior (Set.range ↑I)) (↑(extChartAt (I.prod J) p) p) …
    -/
    obtain h := Set.mem_prod.mp hp
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorne …
      h : And (Membership.mem (ModelWithCorners.interior M) p.1) (Membership.mem (Mo …
      ⊢ And (Membership.mem (interior (Set.range ↑I)) (↑(extChartAt (I.prod J) p) p) …
    -/
    rw [ModelWithCorners.interior] at h
    /-
      case h.mpr
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      N : Type u_7
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace H' N
      J : ModelWithCorners 𝕜 E' H'
      p : Prod M N
      aux : Eq (SProd.sprod (interior (Set.range ↑I)) (interior (Set.range ↑J))) (in …
      hp : Membership.mem (SProd.sprod (ModelWithCorners.interior M) (ModelWithCorne …
      h : And (Membership.mem (setOf fun x => I.IsInteriorPoint x) p.1) (Membership. …
      ⊢ And (Membership.mem (interior (Set.range ↑I)) (↑(extChartAt (I.prod J) p) p) …
    -/
    exact h
    /-
      🎉 no goals
    -/


/-- The boundary of `M × N` is `∂M × N ∪ (M × ∂N)`. -/
lemma boundary_prod :
    (I.prod J).boundary (M × N) = Set.prod univ (J.boundary N) ∪ Set.prod (I.boundary M) univ := by
  let h := calc (I.prod J).boundary (M × N)
    _ = ((I.prod J).interior (M × N))ᶜ := compl_interior.symm
    _ = ((I.interior M) ×ˢ (J.interior N))ᶜ := by rw [interior_prod]
    _ = (I.interior M)ᶜ ×ˢ univ ∪ univ ×ˢ (J.interior N)ᶜ := by rw [compl_prod_eq_union]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    h : Eq (ModelWithCorners.boundary (Prod M N)) (Union.union (SProd.sprod (HasCo …
    ⊢ Eq (ModelWithCorners.boundary (Prod M N)) (Union.union (Set.univ.prod (Model …
  -/
  rw [h, I.compl_interior, J.compl_interior, union_comm]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    N : Type u_7
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    h : Eq (ModelWithCorners.boundary (Prod M N)) (Union.union (SProd.sprod (HasCo …
    ⊢ Eq (Union.union (SProd.sprod Set.univ (ModelWithCorners.boundary N)) (SProd. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `M` is boundaryless, `∂(M×N) = M × ∂N`. -/
lemma boundary_of_boundaryless_left [BoundarylessManifold I M] :
    (I.prod J).boundary (M × N) = Set.prod (univ : Set M) (J.boundary N) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    N : Type u_7
    inst✝² : TopologicalSpace N
    inst✝¹ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    inst✝ : BoundarylessManifold I M
    ⊢ Eq (ModelWithCorners.boundary (Prod M N)) (Set.univ.prod (ModelWithCorners.b …
  -/
  rw [boundary_prod, Boundaryless.boundary_eq_empty (I := I)]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    N : Type u_7
    inst✝² : TopologicalSpace N
    inst✝¹ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    inst✝ : BoundarylessManifold I M
    ⊢ Eq (Union.union (Set.univ.prod (ModelWithCorners.boundary N)) (EmptyCollecti …
  -/
  have : Set.prod (∅ : Set M) (univ : Set N) = ∅ := Set.empty_prod
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    N : Type u_7
    inst✝² : TopologicalSpace N
    inst✝¹ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    inst✝ : BoundarylessManifold I M
    this : Eq (EmptyCollection.emptyCollection.prod Set.univ) EmptyCollection.empt …
    ⊢ Eq (Union.union (Set.univ.prod (ModelWithCorners.boundary N)) (EmptyCollecti …
  -/
  rw [this, union_empty]
  /-
    🎉 no goals
  -/


/-- If `N` is boundaryless, `∂(M×N) = ∂M × N`. -/
lemma boundary_of_boundaryless_right [BoundarylessManifold J N] :
    (I.prod J).boundary (M × N) = Set.prod (I.boundary M) (univ : Set N) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    N : Type u_7
    inst✝² : TopologicalSpace N
    inst✝¹ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    inst✝ : BoundarylessManifold J N
    ⊢ Eq (ModelWithCorners.boundary (Prod M N)) ((ModelWithCorners.boundary M).pro …
  -/
  rw [boundary_prod, Boundaryless.boundary_eq_empty (I := J)]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    N : Type u_7
    inst✝² : TopologicalSpace N
    inst✝¹ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    inst✝ : BoundarylessManifold J N
    ⊢ Eq (Union.union (Set.univ.prod EmptyCollection.emptyCollection) ((ModelWithC …
  -/
  have : Set.prod (univ : Set M) (∅ : Set N) = ∅ := Set.prod_empty
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    N : Type u_7
    inst✝² : TopologicalSpace N
    inst✝¹ : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    inst✝ : BoundarylessManifold J N
    this : Eq (Set.univ.prod EmptyCollection.emptyCollection) EmptyCollection.empt …
    ⊢ Eq (Union.union (Set.univ.prod EmptyCollection.emptyCollection) ((ModelWithC …
  -/
  rw [this, empty_union]
  /-
    🎉 no goals
  -/


/-- The product of two boundaryless manifolds is boundaryless. -/
instance BoundarylessManifold.prod [BoundarylessManifold I M] [BoundarylessManifold J N] :
    BoundarylessManifold (I.prod J) (M × N) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    N : Type u_7
    inst✝³ : TopologicalSpace N
    inst✝² : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    x : M
    y : N
    inst✝¹ : BoundarylessManifold I M
    inst✝ : BoundarylessManifold J N
    ⊢ BoundarylessManifold (I.prod J) (Prod M N)
  -/
  apply Boundaryless.of_boundary_eq_empty
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    N : Type u_7
    inst✝³ : TopologicalSpace N
    inst✝² : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    x : M
    y : N
    inst✝¹ : BoundarylessManifold I M
    inst✝ : BoundarylessManifold J N
    ⊢ Eq (ModelWithCorners.boundary (Prod M N)) EmptyCollection.emptyCollection
  -/
  simp only [boundary_prod, Boundaryless.boundary_eq_empty, union_empty_iff]
  -- These are simp lemmas, but `simp` does not apply them on its own:
  -- presumably because of the distinction between `Prod` and `ModelProd`
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    N : Type u_7
    inst✝³ : TopologicalSpace N
    inst✝² : ChartedSpace H' N
    J : ModelWithCorners 𝕜 E' H'
    x : M
    y : N
    inst✝¹ : BoundarylessManifold I M
    inst✝ : BoundarylessManifold J N
    ⊢ And (Eq (Set.univ.prod EmptyCollection.emptyCollection) EmptyCollection.empt …
  -/
  exact ⟨Set.prod_empty, Set.empty_prod⟩
  /-
    🎉 no goals
  -/


