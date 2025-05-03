/-- The set of closed points. -/
def closedPoints : Set X := setOf (IsClosed {·})


@[simp]
lemma mem_closedPoints_iff {x} : x ∈ closedPoints X ↔ IsClosed {x} := Iff.rfl


lemma preimage_closedPoints_subset (hf : Function.Injective f) (hf' : Continuous f) :
    f ⁻¹' closedPoints Y ⊆ closedPoints X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Function.Injective f
    hf' : Continuous f
    ⊢ HasSubset.Subset (Set.preimage f (closedPoints Y)) (closedPoints X)
  -/
  intros x hx
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Function.Injective f
    hf' : Continuous f
    x : X
    hx : Membership.mem (Set.preimage f (closedPoints Y)) x
    ⊢ Membership.mem (closedPoints X) x
  -/
  rw [mem_closedPoints_iff]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Function.Injective f
    hf' : Continuous f
    x : X
    hx : Membership.mem (Set.preimage f (closedPoints Y)) x
    ⊢ IsClosed (Singleton.singleton x)
  -/
  convert continuous_iff_isClosed.mp hf' _ hx
  /-
    case h.e'_3
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Function.Injective f
    hf' : Continuous f
    x : X
    hx : Membership.mem (Set.preimage f (closedPoints Y)) x
    ⊢ Eq (Singleton.singleton x) (Set.preimage f (Singleton.singleton (f x)))
  -/
  rw [← Set.image_singleton, Set.preimage_image_eq _ hf]
  /-
    🎉 no goals
  -/


lemma Topology.IsClosedEmbedding.preimage_closedPoints (hf : IsClosedEmbedding f) :
    f ⁻¹' closedPoints Y = closedPoints X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsClosedEmbedding f
    ⊢ Eq (Set.preimage f (closedPoints Y)) (closedPoints X)
  -/
  ext x
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsClosedEmbedding f
    x : X
    ⊢ Iff (Membership.mem (Set.preimage f (closedPoints Y)) x) (Membership.mem (cl …
  -/
  simp [mem_closedPoints_iff, ← Set.image_singleton, hf.isClosed_iff_image_isClosed]
  /-
    🎉 no goals
  -/


lemma closedPoints_eq_univ [T1Space X] :
    closedPoints X = Set.univ :=
  Set.eq_univ_iff_forall.mpr fun _ ↦ isClosed_singleton


/-- The class of jacobson spaces, i.e.
spaces such that the set of closed points are dense in every closed subspace. -/
@[mk_iff, stacks 005U]
class JacobsonSpace : Prop where
  closure_inter_closedPoints : ∀ {Z}, IsClosed Z → closure (Z ∩ closedPoints X) = Z


lemma closure_closedPoints [JacobsonSpace X] : closure (closedPoints X) = Set.univ := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    ⊢ Eq (closure (closedPoints X)) Set.univ
  -/
  simpa using closure_inter_closedPoints isClosed_univ
  /-
    🎉 no goals
  -/


lemma jacobsonSpace_iff_locallyClosed :
    JacobsonSpace X ↔ ∀ Z, Z.Nonempty → IsLocallyClosed Z → (Z ∩ closedPoints X).Nonempty := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (JacobsonSpace X) (∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inte …
  -/
  rw [jacobsonSpace_iff]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (∀ {Z : Set X}, IsClosed Z → Eq (closure (Inter.inter Z (closedPoints X) …
  -/
  constructor
  · simp_rw [isLocallyClosed_iff_isOpen_coborder, coborder, isOpen_compl_iff,
      Set.nonempty_iff_ne_empty]
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ (∀ {Z : Set X}, IsClosed Z → Eq (closure (Inter.inter Z (closedPoints X))) Z …
    -/
    intros H Z hZ hZ' e
    have : Z ⊆ closure Z \ Z := by
      refine subset_closure.trans ?_
      nth_rw 1 [← H isClosed_closure]
      rw [hZ'.closure_subset_iff, Set.subset_diff, Set.disjoint_iff, Set.inter_assoc,
        Set.inter_comm _ Z, e]
      exact ⟨Set.inter_subset_left, Set.inter_subset_right⟩
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      H : ∀ {Z : Set X}, IsClosed Z → Eq (closure (Inter.inter Z (closedPoints X))) Z
      Z : Set X
      hZ : Ne Z EmptyCollection.emptyCollection
      hZ' : IsClosed (SDiff.sdiff (closure Z) Z)
      e : Eq (Inter.inter Z (closedPoints X)) EmptyCollection.emptyCollection
      this : HasSubset.Subset Z (SDiff.sdiff (closure Z) Z)
      ⊢ False
    -/
    rw [Set.subset_diff, disjoint_self, Set.bot_eq_empty] at this
    /-
      case mp
      X : Type u_1
      inst✝ : TopologicalSpace X
      H : ∀ {Z : Set X}, IsClosed Z → Eq (closure (Inter.inter Z (closedPoints X))) Z
      Z : Set X
      hZ : Ne Z EmptyCollection.emptyCollection
      hZ' : IsClosed (SDiff.sdiff (closure Z) Z)
      e : Eq (Inter.inter Z (closedPoints X)) EmptyCollection.emptyCollection
      this : And (HasSubset.Subset Z (closure Z)) (Eq Z EmptyCollection.emptyCollect …
      ⊢ False
    -/
    exact hZ this.2
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ (∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (closedPoint …
    -/
  · intro H Z hZ
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      H : ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (closedPoin …
      Z : Set X
      hZ : IsClosed Z
      ⊢ Eq (closure (Inter.inter Z (closedPoints X))) Z
    -/
    refine subset_antisymm (hZ.closure_subset_iff.mpr Set.inter_subset_left) ?_
    rw [← Set.disjoint_compl_left_iff_subset, Set.disjoint_iff_inter_eq_empty,
      ← Set.not_nonempty_iff_eq_empty]
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      H : ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (closedPoin …
      Z : Set X
      hZ : IsClosed Z
      ⊢ Not (Inter.inter (HasCompl.compl (closure (Inter.inter Z (closedPoints X)))) …
    -/
    intro H'
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      H : ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (closedPoin …
      Z : Set X
      hZ : IsClosed Z
      H' : (Inter.inter (HasCompl.compl (closure (Inter.inter Z (closedPoints X))))  …
      ⊢ False
    -/
    have := H _ H' (isClosed_closure.isOpen_compl.isLocallyClosed.inter hZ.isLocallyClosed)
    rw [Set.nonempty_iff_ne_empty, Set.inter_assoc, ne_eq,
      ← Set.disjoint_iff_inter_eq_empty, Set.disjoint_compl_left_iff_subset] at this
    /-
      case mpr
      X : Type u_1
      inst✝ : TopologicalSpace X
      H : ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (closedPoin …
      Z : Set X
      hZ : IsClosed Z
      H' : (Inter.inter (HasCompl.compl (closure (Inter.inter Z (closedPoints X))))  …
      this : Not (HasSubset.Subset (Inter.inter Z (closedPoints X)) (closure (Inter. …
      ⊢ False
    -/
    exact this subset_closure
    /-
      🎉 no goals
    -/


lemma nonempty_inter_closedPoints [JacobsonSpace X] {Z : Set X}
    (hZ : Z.Nonempty) (hZ' : IsLocallyClosed Z) : (Z ∩ closedPoints X).Nonempty :=
  jacobsonSpace_iff_locallyClosed.mp inferInstance Z hZ hZ'


lemma isClosed_singleton_of_isLocallyClosed_singleton [JacobsonSpace X] {x : X}
    (hx : IsLocallyClosed {x}) : IsClosed {x} := by
  obtain ⟨_, ⟨y, rfl : y = x, rfl⟩, hy'⟩ :=
    nonempty_inter_closedPoints (Set.singleton_nonempty x) hx
  /-
    case intro.intro.refl
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    x : X
    hx : IsLocallyClosed (Singleton.singleton x)
    hy' : Membership.mem (closedPoints X) x
    ⊢ IsClosed (Singleton.singleton x)
  -/
  exact hy'
  /-
    🎉 no goals
  -/


lemma Topology.IsOpenEmbedding.preimage_closedPoints (hf : IsOpenEmbedding f) [JacobsonSpace Y] :
    f ⁻¹' closedPoints Y = closedPoints X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsOpenEmbedding f
    inst✝ : JacobsonSpace Y
    ⊢ Eq (Set.preimage f (closedPoints Y)) (closedPoints X)
  -/
  apply subset_antisymm (preimage_closedPoints_subset hf.injective hf.continuous)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsOpenEmbedding f
    inst✝ : JacobsonSpace Y
    ⊢ HasSubset.Subset (closedPoints X) (Set.preimage f (closedPoints Y))
  -/
  intros x hx
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsOpenEmbedding f
    inst✝ : JacobsonSpace Y
    x : X
    hx : Membership.mem (closedPoints X) x
    ⊢ Membership.mem (Set.preimage f (closedPoints Y)) x
  -/
  apply isClosed_singleton_of_isLocallyClosed_singleton
  /-
    case hx
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsOpenEmbedding f
    inst✝ : JacobsonSpace Y
    x : X
    hx : Membership.mem (closedPoints X) x
    ⊢ IsLocallyClosed (Singleton.singleton (f x))
  -/
  rw [← Set.image_singleton]
  /-
    case hx
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsOpenEmbedding f
    inst✝ : JacobsonSpace Y
    x : X
    hx : Membership.mem (closedPoints X) x
    ⊢ IsLocallyClosed (Set.image f (Singleton.singleton x))
  -/
  exact (hx.isLocallyClosed.image hf.isInducing hf.isOpen_range.isLocallyClosed)
  /-
    🎉 no goals
  -/


lemma JacobsonSpace.of_isOpenEmbedding [JacobsonSpace Y] (hf : IsOpenEmbedding f) :
    JacobsonSpace X := by
  /-
    X : Type u_2
    Y : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : JacobsonSpace Y
    hf : Topology.IsOpenEmbedding f
    ⊢ JacobsonSpace X
  -/
  rw [jacobsonSpace_iff_locallyClosed, ← hf.preimage_closedPoints]
  /-
    X : Type u_2
    Y : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : JacobsonSpace Y
    hf : Topology.IsOpenEmbedding f
    ⊢ ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (Set.preimage …
  -/
  intros Z hZ hZ'
  obtain ⟨_, ⟨x, hx, rfl⟩, hx'⟩ := nonempty_inter_closedPoints
    (hZ.image f) (hZ'.image hf.isInducing hf.isOpen_range.isLocallyClosed)
  /-
    case intro.intro.intro.intro
    X : Type u_2
    Y : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : JacobsonSpace Y
    hf : Topology.IsOpenEmbedding f
    Z : Set X
    hZ : Z.Nonempty
    hZ' : IsLocallyClosed Z
    x : X
    hx : Membership.mem Z x
    hx' : Membership.mem (closedPoints Y) (f x)
    ⊢ (Inter.inter Z (Set.preimage f (closedPoints Y))).Nonempty
  -/
  exact ⟨_, hx, hx'⟩
  /-
    🎉 no goals
  -/


lemma JacobsonSpace.of_isClosedEmbedding [JacobsonSpace Y] (hf : IsClosedEmbedding f) :
    JacobsonSpace X := by
  /-
    X : Type u_2
    Y : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : JacobsonSpace Y
    hf : Topology.IsClosedEmbedding f
    ⊢ JacobsonSpace X
  -/
  rw [jacobsonSpace_iff_locallyClosed, ← hf.preimage_closedPoints]
  /-
    X : Type u_2
    Y : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : JacobsonSpace Y
    hf : Topology.IsClosedEmbedding f
    ⊢ ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (Set.preimage …
  -/
  intros Z hZ hZ'
  obtain ⟨_, ⟨x, hx, rfl⟩, hx'⟩ := nonempty_inter_closedPoints
    (hZ.image f) (hZ'.image hf.isInducing hf.isClosed_range.isLocallyClosed)
  /-
    case intro.intro.intro.intro
    X : Type u_2
    Y : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    f : X → Y
    inst✝ : JacobsonSpace Y
    hf : Topology.IsClosedEmbedding f
    Z : Set X
    hZ : Z.Nonempty
    hZ' : IsLocallyClosed Z
    x : X
    hx : Membership.mem Z x
    hx' : Membership.mem (closedPoints Y) (f x)
    ⊢ (Inter.inter Z (Set.preimage f (closedPoints Y))).Nonempty
  -/
  exact ⟨_, hx, hx'⟩
  /-
    🎉 no goals
  -/


lemma JacobsonSpace.discreteTopology [JacobsonSpace X]
    (h : (closedPoints X).Finite) : DiscreteTopology X := by
  have : closedPoints X = Set.univ := by
    rw [← Set.univ_subset_iff, ← closure_closedPoints,
      closure_subset_iff_isClosed, ← (closedPoints X).biUnion_of_singleton]
    exact h.isClosed_biUnion fun _ ↦ id
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    ⊢ DiscreteTopology X
  -/
  have inst : Finite X := Set.finite_univ_iff.mp (this ▸ h)
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    inst : Finite X
    ⊢ DiscreteTopology X
  -/
  rw [← forall_open_iff_discrete]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    inst : Finite X
    ⊢ ∀ (s : Set X), IsOpen s
  -/
  intro s
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    inst : Finite X
    s : Set X
    ⊢ IsOpen s
  -/
  rw [← isClosed_compl_iff, ← sᶜ.biUnion_of_singleton]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    inst : Finite X
    s : Set X
    ⊢ IsClosed (Set.iUnion fun x => Set.iUnion fun h => Singleton.singleton x)
  -/
  refine sᶜ.toFinite.isClosed_biUnion fun x _ ↦ ?_
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    inst : Finite X
    s : Set X
    x : X
    x✝ : Membership.mem (HasCompl.compl s) x
    ⊢ IsClosed (Singleton.singleton x)
  -/
  rw [← mem_closedPoints_iff, this]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : JacobsonSpace X
    h : (closedPoints X).Finite
    this : Eq (closedPoints X) Set.univ
    inst : Finite X
    s : Set X
    x : X
    x✝ : Membership.mem (HasCompl.compl s) x
    ⊢ Membership.mem Set.univ x
  -/
  trivial
  /-
    🎉 no goals
  -/


instance (priority := 100) [Finite X] [JacobsonSpace X] : DiscreteTopology X :=
  JacobsonSpace.discreteTopology (Set.toFinite _)


instance (priority := 100) [T1Space X] : JacobsonSpace X :=
      /-
        X : Type u_1
        Y : Type ?u.8082
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        f : X → Y
        inst✝ : T1Space X
        ⊢ ∀ {Z : Set X}, IsClosed Z → Eq (closure (Inter.inter Z (closedPoints X))) Z
      -/
  ⟨by simp [closedPoints_eq_univ, closure_eq_iff_isClosed]⟩
      /-
        🎉 no goals
      -/


open TopologicalSpace in
lemma jacobsonSpace_iff_of_iSup_eq_top {ι : Type*} {U : ι → Opens X} (hU : iSup U = ⊤) :
    JacobsonSpace X ↔ ∀ i, JacobsonSpace (U i) := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    ⊢ Iff (JacobsonSpace X) (∀ (i : ι), JacobsonSpace (Subtype fun x => Membership …
  -/
  refine ⟨fun H i ↦ .of_isOpenEmbedding (U i).2.isOpenEmbedding_subtypeVal, fun H ↦ ?_⟩
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    ⊢ JacobsonSpace X
  -/
  rw [jacobsonSpace_iff_locallyClosed]
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    ⊢ ∀ (Z : Set X), Z.Nonempty → IsLocallyClosed Z → (Inter.inter Z (closedPoints …
  -/
  intros Z hZ hZ'
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    Z : Set X
    hZ : Z.Nonempty
    hZ' : IsLocallyClosed Z
    ⊢ (Inter.inter Z (closedPoints X)).Nonempty
  -/
  have : (⋃ i, (U i : Set X)) = Set.univ := by rw [← Opens.coe_iSup]; injection hU
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    Z : Set X
    hZ : Z.Nonempty
    hZ' : IsLocallyClosed Z
    this : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
    ⊢ (Inter.inter Z (closedPoints X)).Nonempty
  -/
  have : (⋃ i, Z ∩ U i) = Z := by rw [← Set.inter_iUnion, this, Set.inter_univ]
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    Z : Set X
    hZ : Z.Nonempty
    hZ' : IsLocallyClosed Z
    this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
    this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
    ⊢ (Inter.inter Z (closedPoints X)).Nonempty
  -/
  rw [← this, Set.nonempty_iUnion] at hZ
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    Z : Set X
    hZ : Exists fun i => (Inter.inter Z ↑(U i)).Nonempty
    hZ' : IsLocallyClosed Z
    this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
    this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
    ⊢ (Inter.inter Z (closedPoints X)).Nonempty
  -/
  obtain ⟨i, x, hx, hx'⟩ := hZ
  obtain ⟨y, hy, hy'⟩ := (jacobsonSpace_iff_locallyClosed.mp (H i)) (Subtype.val ⁻¹' Z)
    ⟨⟨x, hx'⟩, hx⟩ (hZ'.preimage continuous_subtype_val)
  /-
    case intro.intro.intro.intro.intro
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    Z : Set X
    hZ' : IsLocallyClosed Z
    this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
    this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
    i : ι
    x : X
    hx : Membership.mem Z x
    hx' : Membership.mem (↑(U i)) x
    y : Subtype fun x => Membership.mem (U i) x
    hy : Membership.mem (Set.preimage Subtype.val Z) y
    hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
    ⊢ (Inter.inter Z (closedPoints X)).Nonempty
  -/
  refine ⟨y, hy, (isClosed_iff_coe_preimage_of_iSup_eq_top hU _).mpr fun j ↦ ?_⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_2
    inst✝ : TopologicalSpace X
    ι : Type u_1
    U : ι → TopologicalSpace.Opens X
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
    Z : Set X
    hZ' : IsLocallyClosed Z
    this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
    this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
    i : ι
    x : X
    hx : Membership.mem Z x
    hx' : Membership.mem (↑(U i)) x
    y : Subtype fun x => Membership.mem (U i) x
    hy : Membership.mem (Set.preimage Subtype.val Z) y
    hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
    j : ι
    ⊢ IsClosed (Set.preimage Subtype.val (Singleton.singleton ↑y))
  -/
  by_cases h : (y : X) ∈ U j
    /-
      case pos
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Membership.mem (U j) ↑y
      ⊢ IsClosed (Set.preimage Subtype.val (Singleton.singleton ↑y))
    -/
  · convert_to IsClosed {(⟨y, h⟩ : U j)}
      /-
        case h.e'_3
        X : Type u_2
        inst✝ : TopologicalSpace X
        ι : Type u_1
        U : ι → TopologicalSpace.Opens X
        hU : Eq (iSup U) Top.top
        H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
        Z : Set X
        hZ' : IsLocallyClosed Z
        this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
        this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
        i : ι
        x : X
        hx : Membership.mem Z x
        hx' : Membership.mem (↑(U i)) x
        y : Subtype fun x => Membership.mem (U i) x
        hy : Membership.mem (Set.preimage Subtype.val Z) y
        hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
        j : ι
        h : Membership.mem (U j) ↑y
        ⊢ Eq (Set.preimage Subtype.val (Singleton.singleton ↑y)) (Singleton.singleton  …
      -/
    · ext z; exact @Subtype.coe_inj _ _ z ⟨y, h⟩
             /-
               🎉 no goals
             -/
    /-
      case pos
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Membership.mem (U j) ↑y
      ⊢ IsClosed (Singleton.singleton ⟨↑y, h⟩)
    -/
    apply isClosed_singleton_of_isLocallyClosed_singleton
    convert (hy'.isLocallyClosed.image IsEmbedding.subtypeVal.isInducing
      (U i).2.isOpenEmbedding_subtypeVal.isOpen_range.isLocallyClosed).preimage
      continuous_subtype_val
    /-
      case h.e'_3.h
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Membership.mem (U j) ↑y
      ⊢ Eq (Singleton.singleton ⟨↑y, h⟩) (Set.preimage Subtype.val (Set.image Subtyp …
    -/
    rw [Set.image_singleton]
    /-
      case h.e'_3.h
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Membership.mem (U j) ↑y
      ⊢ Eq (Singleton.singleton ⟨↑y, h⟩) (Set.preimage Subtype.val (Singleton.single …
    -/
    ext z
    /-
      case h.e'_3.h.h
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Membership.mem (U j) ↑y
      z : Subtype fun x => Membership.mem (U j) x
      ⊢ Iff (Membership.mem (Singleton.singleton ⟨↑y, h⟩) z) (Membership.mem (Set.pr …
    -/
    exact (@Subtype.coe_inj _ _ z ⟨y, h⟩).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Not (Membership.mem (U j) ↑y)
      ⊢ IsClosed (Set.preimage Subtype.val (Singleton.singleton ↑y))
    -/
  · convert isClosed_empty
    /-
      case h.e'_3
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Not (Membership.mem (U j) ↑y)
      ⊢ Eq (Set.preimage Subtype.val (Singleton.singleton ↑y)) EmptyCollection.empty …
    -/
    rw [Set.eq_empty_iff_forall_not_mem]
    /-
      case h.e'_3
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Not (Membership.mem (U j) ↑y)
      ⊢ ∀ (x : Subtype fun x => Membership.mem (U j) x), Not (Membership.mem (Set.pr …
    -/
    intro z (hz : z.1 = y.1)
    /-
      case h.e'_3
      X : Type u_2
      inst✝ : TopologicalSpace X
      ι : Type u_1
      U : ι → TopologicalSpace.Opens X
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), JacobsonSpace (Subtype fun x => Membership.mem (U i) x)
      Z : Set X
      hZ' : IsLocallyClosed Z
      this✝ : Eq (Set.iUnion fun i => ↑(U i)) Set.univ
      this : Eq (Set.iUnion fun i => Inter.inter Z ↑(U i)) Z
      i : ι
      x : X
      hx : Membership.mem Z x
      hx' : Membership.mem (↑(U i)) x
      y : Subtype fun x => Membership.mem (U i) x
      hy : Membership.mem (Set.preimage Subtype.val Z) y
      hy' : Membership.mem (closedPoints (Subtype fun x => Membership.mem (U i) x)) y
      j : ι
      h : Not (Membership.mem (U j) ↑y)
      z : Subtype fun x => Membership.mem (U j) x
      hz : Eq ↑z ↑y
      ⊢ False
    -/
    exact h (hz ▸ z.2)
    /-
      🎉 no goals
    -/

