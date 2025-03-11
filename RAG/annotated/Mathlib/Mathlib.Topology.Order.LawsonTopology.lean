/--
The Lawson topology is defined as the meet of `Topology.lower` and the `Topology.scott`.
-/
def lawson (α : Type*) [Preorder α] : TopologicalSpace α := lower α ⊓ scott α univ


/-- Predicate for an ordered topological space to be equipped with its Lawson topology.

The Lawson topology is defined as the meet of `Topology.lower` and the `Topology.scott`.
-/
class IsLawson : Prop where
  topology_eq_lawson : ‹TopologicalSpace α› = lawson α


/-- The complements of the upper closures of finite sets intersected with Scott open sets form
a basis for the lawson topology. -/
def lawsonBasis := { s : Set α | ∃ t : Set α, t.Finite ∧ ∃ u : Set α, IsOpen[scott α univ] u ∧
      u \ upperClosure t = s }


protected theorem isTopologicalBasis : TopologicalSpace.IsTopologicalBasis (lawsonBasis α) := by
  have lawsonBasis_image2 : lawsonBasis α =
      (image2 (fun x x_1 ↦ ⇑WithLower.toLower ⁻¹' x ∩ ⇑WithScott.toScott ⁻¹' x_1)
        (IsLower.lowerBasis (WithLower α)) {U | IsOpen[scott α univ] U}) := by
    rw [lawsonBasis, image2, IsLower.lowerBasis]
    simp_rw [diff_eq_compl_inter]
    aesop
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
    ⊢ TopologicalSpace.IsTopologicalBasis (Topology.IsLawson.lawsonBasis α)
  -/
  rw [lawsonBasis_image2]
  convert IsTopologicalBasis.inf_induced IsLower.isTopologicalBasis
    (isTopologicalBasis_opens (α := WithScott α))
    WithLower.toLower WithScott.toScott
  /-
    case h.e'_2
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
    ⊢ Eq inst✝¹ (Min.min (TopologicalSpace.induced (⇑Topology.WithLower.toLower) T …
  -/
  rw [@topology_eq_lawson α _ _ _, lawson]
  /-
    case h.e'_2
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
    ⊢ Eq (Min.min (Topology.lower α) (Topology.scott α Set.univ)) (Min.min (Topolo …
  -/
  apply (congrArg₂ min _) _
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsLawson α
      lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
      ⊢ Eq (Topology.lower α) (TopologicalSpace.induced (⇑Topology.WithLower.toLower …
    -/
  · letI _ := lower α
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsLawson α
      lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
      x✝ : TopologicalSpace α := Topology.lower α
      ⊢ Eq (Topology.lower α) (TopologicalSpace.induced (⇑Topology.WithLower.toLower …
    -/
    exact (@IsLower.withLowerHomeomorph α ‹_› (lower α) ⟨rfl⟩).isInducing.eq_induced
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsLawson α
      lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
      ⊢ Eq (Topology.scott α Set.univ) (TopologicalSpace.induced (⇑Topology.WithScot …
    -/
  · letI _ := scott α univ
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsLawson α
      lawsonBasis_image2 : Eq (Topology.IsLawson.lawsonBasis α) (Set.image2 (fun x x …
      x✝ : TopologicalSpace α := Topology.scott α Set.univ
      ⊢ Eq (Topology.scott α Set.univ) (TopologicalSpace.induced (⇑Topology.WithScot …
    -/
    exact (@IsScott.withScottHomeomorph α _ (scott α univ) ⟨rfl⟩).isInducing.eq_induced
    /-
      🎉 no goals
    -/


/--
Type synonym for a preorder equipped with the Lawson topology.
-/
def WithLawson (α : Type*) := α


/-- `toLawson` is the identity function to the `WithLawson` of a type. -/
@[match_pattern] def toLawson : α ≃ WithLawson α := Equiv.refl _


/-- `ofLawson` is the identity function from the `WithLawson` of a type. -/
@[match_pattern] def ofLawson : WithLawson α ≃ α := Equiv.refl _


@[simp] lemma to_Lawson_symm_eq : (@toLawson α).symm = ofLawson := rfl

@[simp] lemma of_Lawson_symm_eq : (@ofLawson α).symm = toLawson := rfl

@[simp] lemma toLawson_ofLawson (a : WithLawson α) : toLawson (ofLawson a) = a := rfl

@[simp] lemma ofLawson_toLawson (a : α) : ofLawson (toLawson a) = a := rfl


lemma toLawson_inj {a b : α} : toLawson a = toLawson b ↔ a = b := Iff.rfl


lemma ofLawson_inj {a b : WithLawson α} : ofLawson a = ofLawson b ↔ a = b := Iff.rfl


/-- A recursor for `WithLawson`. Use as `induction' x`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {β : WithLawson α → Sort*}
    (h : ∀ a, β (toLawson a)) : ∀ a, β a := fun a => h (ofLawson a)


instance [Nonempty α] : Nonempty (WithLawson α) := ‹Nonempty α›

instance [Inhabited α] : Inhabited (WithLawson α) := ‹Inhabited α›


instance instPreorder : Preorder (WithLawson α) := ‹Preorder α›

instance instTopologicalSpace : TopologicalSpace (WithLawson α) := lawson α

instance instIsLawson : IsLawson (WithLawson α) := ⟨rfl⟩


/-- If `α` is equipped with the Lawson topology, then it is homeomorphic to `WithLawson α`.
-/
def homeomorph [TopologicalSpace α] [IsLawson α] : WithLawson α ≃ₜ α :=
                                        /-
                                          α : Type u_1
                                          inst✝² : Preorder α
                                          inst✝¹ : TopologicalSpace α
                                          inst✝ : Topology.IsLawson α
                                          ⊢ Eq Topology.WithLawson.instTopologicalSpace (TopologicalSpace.induced (⇑Topo …
                                        -/
  ofLawson.toHomeomorphOfIsInducing ⟨by erw [IsLawson.topology_eq_lawson (α := α), induced_id]; rfl⟩
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem isOpen_preimage_ofLawson {S : Set α} :
    IsOpen (ofLawson ⁻¹' S) ↔ (lawson α).IsOpen S := Iff.rfl


theorem isClosed_preimage_ofLawson {S : Set α} :
    IsClosed (ofLawson ⁻¹' S) ↔ IsClosed[lawson α] S := Iff.rfl


theorem isOpen_def {T : Set (WithLawson α)} :
    IsOpen T ↔ (lawson α).IsOpen (toLawson ⁻¹' T) := Iff.rfl


lemma lawson_le_scott : lawson α ≤ scott α univ := inf_le_right


lemma lawson_le_lower : lawson α ≤ lower α := inf_le_left


lemma scottHausdorff_le_lawson : scottHausdorff α univ ≤ lawson α :=
  le_inf scottHausdorff_le_lower scottHausdorff_le_scott


lemma lawsonClosed_of_scottClosed (s : Set α) (h : IsClosed (WithScott.ofScott ⁻¹' s)) :
    IsClosed (WithLawson.ofLawson ⁻¹' s) := h.mono lawson_le_scott


lemma lawsonClosed_of_lowerClosed (s : Set α) (h : IsClosed (WithLower.ofLower ⁻¹' s)) :
    IsClosed (WithLawson.ofLawson ⁻¹' s) := h.mono lawson_le_lower


/-- An upper set is Lawson open if and only if it is Scott open -/
lemma lawsonOpen_iff_scottOpen_of_isUpperSet {s : Set α} (h : IsUpperSet s) :
    IsOpen (WithLawson.ofLawson ⁻¹' s) ↔ IsOpen (WithScott.ofScott ⁻¹' s) :=
  ⟨fun hs => IsScott.isOpen_iff_isUpperSet_and_scottHausdorff_open (D := univ).mpr
    ⟨h, (scottHausdorff_le_lawson s) hs⟩, lawson_le_scott _⟩


lemma isLawson_le_isScott : L ≤ S := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    L S : TopologicalSpace α
    inst✝¹ : Topology.IsLawson α
    inst✝ : Topology.IsScott α Set.univ
    ⊢ LE.le L S
  -/
  rw [@IsScott.topology_eq α univ _ S _, @IsLawson.topology_eq_lawson α _ L _]
  /-
    α : Type u_1
    inst✝² : Preorder α
    L S : TopologicalSpace α
    inst✝¹ : Topology.IsLawson α
    inst✝ : Topology.IsScott α Set.univ
    ⊢ LE.le (Topology.lawson α) (Topology.scott α Set.univ)
  -/
  exact inf_le_right
  /-
    🎉 no goals
  -/


lemma scottHausdorff_le_isLawson : scottHausdorff α univ ≤ L := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    L : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    ⊢ LE.le (Topology.scottHausdorff α Set.univ) L
  -/
  rw [@IsLawson.topology_eq_lawson α _ L _]
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    L : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    ⊢ LE.le (Topology.scottHausdorff α Set.univ) (Topology.lawson α)
  -/
  exact scottHausdorff_le_lawson
  /-
    🎉 no goals
  -/


/-- An upper set is Lawson open if and only if it is Scott open -/
lemma lawsonOpen_iff_scottOpen_of_isUpperSet' (s : Set α) (h : IsUpperSet s) :
    IsOpen[L] s ↔ IsOpen[S] s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    L S : TopologicalSpace α
    inst✝¹ : Topology.IsLawson α
    inst✝ : Topology.IsScott α Set.univ
    s : Set α
    h : IsUpperSet s
    ⊢ Iff (IsOpen s) (IsOpen s)
  -/
  rw [@IsLawson.topology_eq_lawson α _ L _, @IsScott.topology_eq α univ _ S _]
  /-
    α : Type u_1
    inst✝² : Preorder α
    L S : TopologicalSpace α
    inst✝¹ : Topology.IsLawson α
    inst✝ : Topology.IsScott α Set.univ
    s : Set α
    h : IsUpperSet s
    ⊢ Iff (IsOpen s) (IsOpen s)
  -/
  exact lawsonOpen_iff_scottOpen_of_isUpperSet h
  /-
    🎉 no goals
  -/


lemma lawsonClosed_iff_scottClosed_of_isLowerSet (s : Set α) (h : IsLowerSet s) :
    IsClosed[L] s ↔ IsClosed[S] s := by
  rw [← @isOpen_compl_iff, ← isOpen_compl_iff,
    (lawsonOpen_iff_scottOpen_of_isUpperSet' L S _ (isUpperSet_compl.mpr h))]


include S in
/-- A lower set is Lawson closed if and only if it is closed under sups of directed sets -/
lemma lawsonClosed_iff_dirSupClosed_of_isLowerSet (s : Set α) (h : IsLowerSet s) :
    IsClosed[L] s ↔ DirSupClosed s := by
  rw [lawsonClosed_iff_scottClosed_of_isLowerSet L S _ h,
    @IsScott.isClosed_iff_isLowerSet_and_dirSupClosed]
  /-
    α : Type u_1
    inst✝² : Preorder α
    L S : TopologicalSpace α
    inst✝¹ : Topology.IsLawson α
    inst✝ : Topology.IsScott α Set.univ
    s : Set α
    h : IsLowerSet s
    ⊢ Iff (And (IsLowerSet s) (DirSupClosed s)) (DirSupClosed s)
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma singleton_isClosed (a : α) : IsClosed ({a} : Set α) := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    a : α
    ⊢ IsClosed (Singleton.singleton a)
  -/
  simp only [IsLawson.topology_eq_lawson]
  rw [← (Set.OrdConnected.upperClosure_inter_lowerClosure ordConnected_singleton),
    ← WithLawson.isClosed_preimage_ofLawson]
  apply IsClosed.inter
    (lawsonClosed_of_lowerClosed _ (IsLower.isClosed_upperClosure (finite_singleton a)))
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    a : α
    ⊢ IsClosed ↑(lowerClosure (Singleton.singleton a))
  -/
  rw [lowerClosure_singleton, LowerSet.coe_Iic, ← WithLawson.isClosed_preimage_ofLawson]
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    a : α
    ⊢ IsClosed (Set.preimage (⇑Topology.WithLawson.ofLawson) (Set.Iic a))
  -/
  apply lawsonClosed_of_scottClosed
  /-
    case h
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLawson α
    a : α
    ⊢ IsClosed (Set.preimage (⇑Topology.WithScott.ofScott) (Set.Iic a))
  -/
  exact IsScott.isClosed_Iic
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

/-- The Lawson topology on a partial order is T₀. -/
instance (priority := 90) t0Space : T0Space α :=
  (t0Space_iff_inseparable α).2 fun a b h => by
    simpa only [inseparable_iff_closure_eq, closure_eq_iff_isClosed.mpr (singleton_isClosed a),
      closure_eq_iff_isClosed.mpr (singleton_isClosed b), singleton_eq_singleton_iff] using h


