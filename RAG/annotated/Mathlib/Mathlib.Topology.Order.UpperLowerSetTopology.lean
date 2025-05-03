/-- Topology whose open sets are upper sets.

Note: In general the upper set topology does not coincide with the upper topology. -/
def upperSet (α :  Type*) [Preorder α] : TopologicalSpace α where
  IsOpen := IsUpperSet
  isOpen_univ := isUpperSet_univ
  isOpen_inter _ _ := IsUpperSet.inter
  isOpen_sUnion _ := isUpperSet_sUnion


/-- Topology whose open sets are lower sets.

Note: In general the lower set topology does not coincide with the lower topology. -/
def lowerSet (α :  Type*) [Preorder α] : TopologicalSpace α where
  IsOpen := IsLowerSet
  isOpen_univ := isLowerSet_univ
  isOpen_inter _ _ := IsLowerSet.inter
  isOpen_sUnion _ := isLowerSet_sUnion


/-- Type synonym for a preorder equipped with the upper set topology. -/
def WithUpperSet (α : Type*) := α


/-- `toUpperSet` is the identity function to the `WithUpperSet` of a type. -/
@[match_pattern] def toUpperSet : α ≃ WithUpperSet α := Equiv.refl _


/-- `ofUpperSet` is the identity function from the `WithUpperSet` of a type. -/
@[match_pattern] def ofUpperSet : WithUpperSet α ≃ α := Equiv.refl _


@[simp] lemma toUpperSet_symm : (@toUpperSet α).symm = ofUpperSet := rfl

@[deprecated (since := "2024-10-10")] alias to_WithUpperSet_symm_eq := toUpperSet_symm

@[simp] lemma ofUpperSet_symm : (@ofUpperSet α).symm = toUpperSet := rfl

@[deprecated (since := "2024-10-10")] alias of_WithUpperSet_symm_eq := ofUpperSet_symm

@[simp] lemma toUpperSet_ofUpperSet (a : WithUpperSet α) : toUpperSet (ofUpperSet a) = a := rfl

@[simp] lemma ofUpperSet_toUpperSet (a : α) : ofUpperSet (toUpperSet a) = a := rfl

lemma toUpperSet_inj {a b : α} : toUpperSet a = toUpperSet b ↔ a = b := Iff.rfl

lemma ofUpperSet_inj {a b : WithUpperSet α} : ofUpperSet a = ofUpperSet b ↔ a = b := Iff.rfl


/-- A recursor for `WithUpperSet`. Use as `induction x`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {β : WithUpperSet α → Sort*} (h : ∀ a, β (toUpperSet a)) : ∀ a, β a :=
  fun a => h (ofUpperSet a)


instance [Nonempty α] : Nonempty (WithUpperSet α) := ‹Nonempty α›

instance [Inhabited α] : Inhabited (WithUpperSet α) := ‹Inhabited α›


instance : Preorder (WithUpperSet α) := ‹Preorder α›

instance : TopologicalSpace (WithUpperSet α) := upperSet α


lemma ofUpperSet_le_iff {a b : WithUpperSet α} : ofUpperSet a ≤ ofUpperSet b ↔ a ≤ b := Iff.rfl

lemma toUpperSet_le_iff {a b : α} : toUpperSet a ≤ toUpperSet b ↔ a ≤ b := Iff.rfl


/-- `ofUpperSet` as an `OrderIso` -/
def ofUpperSetOrderIso : WithUpperSet α ≃o α where
  toEquiv := ofUpperSet
  map_rel_iff' := ofUpperSet_le_iff


/-- `toUpperSet` as an `OrderIso` -/
def toUpperSetOrderIso : α ≃o WithUpperSet α where
  toEquiv := toUpperSet
  map_rel_iff' := toUpperSet_le_iff


/-- Type synonym for a preorder equipped with the lower set topology. -/
def WithLowerSet (α : Type*) := α


/-- `toLowerSet` is the identity function to the `WithLowerSet` of a type. -/
@[match_pattern] def toLowerSet : α ≃ WithLowerSet α := Equiv.refl _


/-- `ofLowerSet` is the identity function from the `WithLowerSet` of a type. -/
@[match_pattern] def ofLowerSet : WithLowerSet α ≃ α := Equiv.refl _


@[simp] lemma toLowerSet_symm : (@toLowerSet α).symm = ofLowerSet := rfl

@[deprecated (since := "2024-10-10")] alias to_WithLowerSet_symm_eq := toLowerSet_symm

@[simp] lemma ofLowerSet_symm : (@ofLowerSet α).symm = toLowerSet := rfl

@[deprecated (since := "2024-10-10")] alias of_WithLowerSet_symm_eq := ofLowerSet_symm

@[simp] lemma toLowerSet_ofLowerSet (a : WithLowerSet α) : toLowerSet (ofLowerSet a) = a := rfl

@[simp] lemma ofLowerSet_toLowerSet (a : α) : ofLowerSet (toLowerSet a) = a := rfl

lemma toLowerSet_inj {a b : α} : toLowerSet a = toLowerSet b ↔ a = b := Iff.rfl

lemma ofLowerSet_inj {a b : WithLowerSet α} : ofLowerSet a = ofLowerSet b ↔ a = b := Iff.rfl


/-- A recursor for `WithLowerSet`. Use as `induction x`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {β : WithLowerSet α → Sort*} (h : ∀ a, β (toLowerSet a)) : ∀ a, β a :=
  fun a => h (ofLowerSet a)


instance [Nonempty α] : Nonempty (WithLowerSet α) := ‹Nonempty α›

instance [Inhabited α] : Inhabited (WithLowerSet α) := ‹Inhabited α›


instance : Preorder (WithLowerSet α) := ‹Preorder α›

instance : TopologicalSpace (WithLowerSet α) := lowerSet α


lemma ofLowerSet_le_iff {a b : WithLowerSet α} : ofLowerSet a ≤ ofLowerSet b ↔ a ≤ b := Iff.rfl

lemma toLowerSet_le_iff {a b : α} : toLowerSet a ≤ toLowerSet b ↔ a ≤ b := Iff.rfl


/-- `ofLowerSet` as an `OrderIso` -/
def ofLowerSetOrderIso : WithLowerSet α ≃o α where
  toEquiv := ofLowerSet
  map_rel_iff' := ofLowerSet_le_iff


/-- `toLowerSet` as an `OrderIso` -/
def toLowerSetOrderIso : α ≃o WithLowerSet α where
  toEquiv := toLowerSet
  map_rel_iff' := toLowerSet_le_iff


/--
The Upper Set topology is homeomorphic to the Lower Set topology on the dual order
-/
def WithUpperSet.toDualHomeomorph [Preorder α] : WithUpperSet α ≃ₜ WithLowerSet αᵒᵈ where
  toFun := OrderDual.toDual
  invFun := OrderDual.ofDual
  left_inv := OrderDual.toDual_ofDual
  right_inv := OrderDual.ofDual_toDual
  continuous_toFun := continuous_coinduced_rng
  continuous_invFun := continuous_coinduced_rng


/-- Prop-valued mixin for an ordered topological space to be
The upper set topology is the topology where the open sets are the upper sets. In general the upper
set topology does not coincide with the upper topology.
-/
protected class IsUpperSet (α : Type*) [t : TopologicalSpace α] [Preorder α] : Prop where
  topology_eq_upperSetTopology : t = upperSet α


instance [Preorder α] : Topology.IsUpperSet (WithUpperSet α) := ⟨rfl⟩


instance [Preorder α] : @Topology.IsUpperSet α (upperSet α) _ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    ⊢ Topology.IsUpperSet α
  -/
  letI := upperSet α
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    this : TopologicalSpace α := Topology.upperSet α
    ⊢ Topology.IsUpperSet α
  -/
  exact ⟨rfl⟩
  /-
    🎉 no goals
  -/


/--
The lower set topology is the topology where the open sets are the lower sets. In general the lower
set topology does not coincide with the lower topology.
-/
protected class IsLowerSet (α : Type*) [t : TopologicalSpace α] [Preorder α] : Prop where
  topology_eq_lowerSetTopology : t = lowerSet α


instance [Preorder α] : Topology.IsLowerSet (WithLowerSet α) := ⟨rfl⟩


instance [Preorder α] : @Topology.IsLowerSet α (lowerSet α) _ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    ⊢ Topology.IsLowerSet α
  -/
  letI := lowerSet α
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Preorder α
    this : TopologicalSpace α := Topology.lowerSet α
    ⊢ Topology.IsLowerSet α
  -/
  exact ⟨rfl⟩
  /-
    🎉 no goals
  -/


lemma topology_eq : ‹_› = upperSet α := topology_eq_upperSetTopology


instance _root_.OrderDual.instIsLowerSet [Preorder α] [TopologicalSpace α] [Topology.IsUpperSet α] :
    Topology.IsLowerSet αᵒᵈ where
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       γ : Type u_3
                                       inst✝⁵ : Preorder α
                                       inst✝⁴ : TopologicalSpace α
                                       inst✝³ : Topology.IsUpperSet α
                                       s : Set α
                                       inst✝² : Preorder α
                                       inst✝¹ : TopologicalSpace α
                                       inst✝ : Topology.IsUpperSet α
                                       ⊢ Eq OrderDual.instTopologicalSpace (Topology.lowerSet (OrderDual α))
                                     -/
  topology_eq_lowerSetTopology := by ext; rw [IsUpperSet.topology_eq α]
                                          /-
                                            🎉 no goals
                                          -/


/-- If `α` is equipped with the upper set topology, then it is homeomorphic to
`WithUpperSet α`. -/
def WithUpperSetHomeomorph : WithUpperSet α ≃ₜ α :=
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         γ : Type u_3
                                                         inst✝² : Preorder α
                                                         inst✝¹ : TopologicalSpace α
                                                         inst✝ : Topology.IsUpperSet α
                                                         s : Set α
                                                         ⊢ Eq Topology.WithUpperSet.instTopologicalSpace (TopologicalSpace.induced (⇑To …
                                                       -/
  WithUpperSet.ofUpperSet.toHomeomorphOfIsInducing ⟨by erw [topology_eq α, induced_id]; rfl⟩
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


lemma isOpen_iff_isUpperSet : IsOpen s ↔ IsUpperSet s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsUpperSet α
    s : Set α
    ⊢ Iff (IsOpen s) (IsUpperSet s)
  -/
  rw [topology_eq α]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsUpperSet α
    s : Set α
    ⊢ Iff (IsOpen s) (IsUpperSet s)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance toAlexandrovDiscrete : AlexandrovDiscrete α where
                        /-
                          α : Type u_1
                          β : Type u_2
                          γ : Type u_3
                          inst✝² : Preorder α
                          inst✝¹ : TopologicalSpace α
                          inst✝ : Topology.IsUpperSet α
                          s : Set α
                          S : Set (Set α)
                          ⊢ (∀ (s : Set α), Membership.mem S s → IsOpen s) → IsOpen S.sInter
                        -/
  isOpen_sInter S := by simpa only [isOpen_iff_isUpperSet] using isUpperSet_sInter (α := α)
                        /-
                          🎉 no goals
                        -/

-- c.f. isClosed_iff_lower_and_subset_implies_LUB_mem

lemma isClosed_iff_isLower : IsClosed s ↔ IsLowerSet s := by
  rw [← isOpen_compl_iff, isOpen_iff_isUpperSet,
    isLowerSet_compl.symm, compl_compl]


lemma closure_eq_lowerClosure {s : Set α} : closure s = lowerClosure s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsUpperSet α
    s : Set α
    ⊢ Eq (closure s) ↑(lowerClosure s)
  -/
  rw [subset_antisymm_iff]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsUpperSet α
    s : Set α
    ⊢ And (HasSubset.Subset (closure s) ↑(lowerClosure s)) (HasSubset.Subset (↑(lo …
  -/
  refine ⟨?_, lowerClosure_min subset_closure (isClosed_iff_isLower.1 isClosed_closure)⟩
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsUpperSet α
      s : Set α
      ⊢ HasSubset.Subset (closure s) ↑(lowerClosure s)
    -/
  · apply closure_minimal subset_lowerClosure _
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsUpperSet α
      s : Set α
      ⊢ IsClosed ↑(lowerClosure s)
    -/
    rw [isClosed_iff_isLower]
    /-
      α : Type u_1
      inst✝² : Preorder α
      inst✝¹ : TopologicalSpace α
      inst✝ : Topology.IsUpperSet α
      s : Set α
      ⊢ IsLowerSet ↑(lowerClosure s)
    -/
    exact LowerSet.lower (lowerClosure s)
    /-
      🎉 no goals
    -/


/--
The closure of a singleton `{a}` in the upper set topology is the right-closed left-infinite
interval (-∞,a].
-/
@[simp] lemma closure_singleton {a : α} : closure {a} = Iic a := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsUpperSet α
    a : α
    ⊢ Eq (closure (Singleton.singleton a)) (Set.Iic a)
  -/
  rw [closure_eq_lowerClosure, lowerClosure_singleton]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsUpperSet α
    a : α
    ⊢ Eq (↑(LowerSet.Iic a)) (Set.Iic a)
  -/
  rfl
  /-
    🎉 no goals
  -/


protected lemma monotone_iff_continuous [TopologicalSpace α] [TopologicalSpace β]
    [Topology.IsUpperSet α] [Topology.IsUpperSet β] {f : α → β} : Monotone f ↔ Continuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsUpperSet α
    inst✝ : Topology.IsUpperSet β
    f : α → β
    ⊢ Iff (Monotone f) (Continuous f)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      ⊢ Monotone f → Continuous f
    -/
  · intro hf
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Monotone f
      ⊢ Continuous f
    -/
    simp_rw [continuous_def, isOpen_iff_isUpperSet]
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Monotone f
      ⊢ ∀ (s : Set β), IsUpperSet s → IsUpperSet (Set.preimage f s)
    -/
    exact fun _ hs ↦ IsUpperSet.preimage hs hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      ⊢ Continuous f → Monotone f
    -/
  · intro hf a b hab
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Continuous f
      a b : α
      hab : LE.le a b
      ⊢ LE.le (f a) (f b)
    -/
    rw [← mem_Iic, ← closure_singleton] at hab ⊢
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Continuous f
      a b : α
      hab : Membership.mem (closure (Singleton.singleton b)) a
      ⊢ Membership.mem (closure (Singleton.singleton (f b))) (f a)
    -/
    apply Continuous.closure_preimage_subset hf {f b}
    /-
      case mpr.a
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Continuous f
      a b : α
      hab : Membership.mem (closure (Singleton.singleton b)) a
      ⊢ Membership.mem (closure (Set.preimage f (Singleton.singleton (f b)))) a
    -/
    apply mem_of_mem_of_subset hab
    /-
      case mpr.a
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Continuous f
      a b : α
      hab : Membership.mem (closure (Singleton.singleton b)) a
      ⊢ HasSubset.Subset (closure (Singleton.singleton b)) (closure (Set.preimage f  …
    -/
    apply closure_mono
    /-
      case mpr.a.h
      α : Type u_1
      β : Type u_2
      inst✝⁵ : Preorder α
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : Topology.IsUpperSet α
      inst✝ : Topology.IsUpperSet β
      f : α → β
      hf : Continuous f
      a b : α
      hab : Membership.mem (closure (Singleton.singleton b)) a
      ⊢ HasSubset.Subset (Singleton.singleton b) (Set.preimage f (Singleton.singleto …
    -/
    rw [singleton_subset_iff, mem_preimage, mem_singleton_iff]
    /-
      🎉 no goals
    -/


lemma monotone_to_upperTopology_continuous [TopologicalSpace α] [TopologicalSpace β]
    [Topology.IsUpperSet α] [IsUpper β] {f : α → β} (hf : Monotone f) : Continuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsUpperSet α
    inst✝ : Topology.IsUpper β
    f : α → β
    hf : Monotone f
    ⊢ Continuous f
  -/
  simp_rw [continuous_def, isOpen_iff_isUpperSet]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsUpperSet α
    inst✝ : Topology.IsUpper β
    f : α → β
    hf : Monotone f
    ⊢ ∀ (s : Set β), IsOpen s → IsUpperSet (Set.preimage f s)
  -/
  intro s hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsUpperSet α
    inst✝ : Topology.IsUpper β
    f : α → β
    hf : Monotone f
    s : Set β
    hs : IsOpen s
    ⊢ IsUpperSet (Set.preimage f s)
  -/
  exact (IsUpper.isUpperSet_of_isOpen hs).preimage hf
  /-
    🎉 no goals
  -/


lemma upperSet_le_upper {t₁ t₂ : TopologicalSpace α} [@Topology.IsUpperSet α t₁ _]
    [@Topology.IsUpper α t₂ _] : t₁ ≤ t₂ := fun s hs => by
  /-
    α : Type u_1
    inst✝² : Preorder α
    t₁ t₂ : TopologicalSpace α
    inst✝¹ : Topology.IsUpperSet α
    inst✝ : Topology.IsUpper α
    s : Set α
    hs : IsOpen s
    ⊢ IsOpen s
  -/
  rw [@isOpen_iff_isUpperSet α _ t₁]
  /-
    α : Type u_1
    inst✝² : Preorder α
    t₁ t₂ : TopologicalSpace α
    inst✝¹ : Topology.IsUpperSet α
    inst✝ : Topology.IsUpper α
    s : Set α
    hs : IsOpen s
    ⊢ IsUpperSet s
  -/
  exact IsUpper.isUpperSet_of_isOpen hs
  /-
    🎉 no goals
  -/


lemma topology_eq : ‹_› = lowerSet α := topology_eq_lowerSetTopology


instance _root_.OrderDual.instIsUpperSet [Preorder α] [TopologicalSpace α] [Topology.IsLowerSet α] :
    Topology.IsUpperSet αᵒᵈ where
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       γ : Type u_3
                                       inst✝⁵ : Preorder α
                                       inst✝⁴ : TopologicalSpace α
                                       inst✝³ : Topology.IsLowerSet α
                                       s : Set α
                                       inst✝² : Preorder α
                                       inst✝¹ : TopologicalSpace α
                                       inst✝ : Topology.IsLowerSet α
                                       ⊢ Eq OrderDual.instTopologicalSpace (Topology.upperSet (OrderDual α))
                                     -/
  topology_eq_upperSetTopology := by ext; rw [IsLowerSet.topology_eq α]
                                          /-
                                            🎉 no goals
                                          -/


/-- If `α` is equipped with the lower set topology, then it is homeomorphic to `WithLowerSet α`. -/
def WithLowerSetHomeomorph : WithLowerSet α ≃ₜ α :=
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         γ : Type u_3
                                                         inst✝² : Preorder α
                                                         inst✝¹ : TopologicalSpace α
                                                         inst✝ : Topology.IsLowerSet α
                                                         s : Set α
                                                         ⊢ Eq Topology.WithLowerSet.instTopologicalSpace (TopologicalSpace.induced (⇑To …
                                                       -/
  WithLowerSet.ofLowerSet.toHomeomorphOfIsInducing ⟨by erw [topology_eq α, induced_id]; rfl⟩
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


                                                            /-
                                                              α : Type u_1
                                                              inst✝² : Preorder α
                                                              inst✝¹ : TopologicalSpace α
                                                              inst✝ : Topology.IsLowerSet α
                                                              s : Set α
                                                              ⊢ Iff (IsOpen s) (IsLowerSet s)
                                                            -/
lemma isOpen_iff_isLowerSet : IsOpen s ↔ IsLowerSet s := by rw [topology_eq α]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance toAlexandrovDiscrete : AlexandrovDiscrete α := IsUpperSet.toAlexandrovDiscrete (α := αᵒᵈ)


lemma isClosed_iff_isUpper : IsClosed s ↔ IsUpperSet s := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLowerSet α
    s : Set α
    ⊢ Iff (IsClosed s) (IsUpperSet s)
  -/
  rw [← isOpen_compl_iff, isOpen_iff_isLowerSet, isUpperSet_compl.symm, compl_compl]
  /-
    🎉 no goals
  -/


lemma closure_eq_upperClosure {s : Set α} : closure s = upperClosure s :=
  IsUpperSet.closure_eq_lowerClosure (α := αᵒᵈ)


/--
The closure of a singleton `{a}` in the lower set topology is the right-closed left-infinite
interval (-∞,a].
-/
@[simp] lemma closure_singleton {a : α} : closure {a} = Ici a := by
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLowerSet α
    a : α
    ⊢ Eq (closure (Singleton.singleton a)) (Set.Ici a)
  -/
  rw [closure_eq_upperClosure, upperClosure_singleton]
  /-
    α : Type u_1
    inst✝² : Preorder α
    inst✝¹ : TopologicalSpace α
    inst✝ : Topology.IsLowerSet α
    a : α
    ⊢ Eq (↑(UpperSet.Ici a)) (Set.Ici a)
  -/
  rfl
  /-
    🎉 no goals
  -/


protected lemma monotone_iff_continuous [TopologicalSpace α] [TopologicalSpace β]
    [Topology.IsLowerSet α] [Topology.IsLowerSet β] {f : α → β} : Monotone f ↔ Continuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : Topology.IsLowerSet α
    inst✝ : Topology.IsLowerSet β
    f : α → β
    ⊢ Iff (Monotone f) (Continuous f)
  -/
  rw [← monotone_dual_iff]
  exact IsUpperSet.monotone_iff_continuous (α := αᵒᵈ) (β := βᵒᵈ)
    (f := (toDual ∘ f ∘ ofDual : αᵒᵈ → βᵒᵈ))


lemma monotone_to_lowerTopology_continuous [TopologicalSpace α] [TopologicalSpace β]
    [Topology.IsLowerSet α] [IsLower β] {f : α → β} (hf : Monotone f) : Continuous f :=
  IsUpperSet.monotone_to_upperTopology_continuous (α := αᵒᵈ) (β := βᵒᵈ) hf.dual


lemma lowerSet_le_lower {t₁ t₂ : TopologicalSpace α} [@Topology.IsLowerSet α t₁ _]
    [@IsLower α t₂ _] : t₁ ≤ t₂ := fun s hs => by
  /-
    α : Type u_1
    inst✝² : Preorder α
    t₁ t₂ : TopologicalSpace α
    inst✝¹ : Topology.IsLowerSet α
    inst✝ : Topology.IsLower α
    s : Set α
    hs : IsOpen s
    ⊢ IsOpen s
  -/
  rw [@isOpen_iff_isLowerSet α _ t₁]
  /-
    α : Type u_1
    inst✝² : Preorder α
    t₁ t₂ : TopologicalSpace α
    inst✝¹ : Topology.IsLowerSet α
    inst✝ : Topology.IsLower α
    s : Set α
    hs : IsOpen s
    ⊢ IsLowerSet s
  -/
  exact IsLower.isLowerSet_of_isOpen hs
  /-
    🎉 no goals
  -/


lemma isUpperSet_orderDual [Preorder α] [TopologicalSpace α] :
    Topology.IsUpperSet αᵒᵈ ↔ Topology.IsLowerSet α := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    inst✝ : TopologicalSpace α
    ⊢ Iff (Topology.IsUpperSet (OrderDual α)) (Topology.IsLowerSet α)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : TopologicalSpace α
      ⊢ Topology.IsUpperSet (OrderDual α) → Topology.IsLowerSet α
    -/
  · apply OrderDual.instIsLowerSet
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : Preorder α
      inst✝ : TopologicalSpace α
      ⊢ Topology.IsLowerSet α → Topology.IsUpperSet (OrderDual α)
    -/
  · apply OrderDual.instIsUpperSet
    /-
      🎉 no goals
    -/


lemma isLowerSet_orderDual [Preorder α] [TopologicalSpace α] :
    Topology.IsLowerSet αᵒᵈ ↔ Topology.IsUpperSet α := isUpperSet_orderDual.symm


/-- A monotone map between preorders spaces induces a continuous map between themselves considered
with the upper set topology. -/
def map (f : α →o β) : C(WithUpperSet α, WithUpperSet β) where
  toFun := toUpperSet ∘ f ∘ ofUpperSet
  continuous_toFun := continuous_def.2 fun _s hs ↦ IsUpperSet.preimage hs f.monotone


@[simp] lemma map_id : map (OrderHom.id : α →o α) = ContinuousMap.id _ := rfl

@[simp] lemma map_comp (g : β →o γ) (f : α →o β) : map (g.comp f) = (map g).comp (map f) := rfl


@[simp] lemma toUpperSet_specializes_toUpperSet {a b : α} :
    toUpperSet a ⤳ toUpperSet b ↔ b ≤ a := by
  simp_rw [specializes_iff_closure_subset, IsUpperSet.closure_singleton, Iic_subset_Iic,
    toUpperSet_le_iff]


@[simp] lemma ofUpperSet_le_ofUpperSet {a b : WithUpperSet α} :
    ofUpperSet a ≤ ofUpperSet b ↔ b ⤳ a := toUpperSet_specializes_toUpperSet.symm


@[simp] lemma isUpperSet_toUpperSet_preimage {s : Set (WithUpperSet α)} :
    IsUpperSet (toUpperSet ⁻¹' s) ↔ IsOpen s := Iff.rfl


@[simp] lemma isOpen_ofUpperSet_preimage {s : Set α} :
    IsOpen (ofUpperSet ⁻¹' s) ↔ IsUpperSet s := isUpperSet_toUpperSet_preimage.symm


/-- A monotone map between preorders spaces induces a continuous map between themselves considered
with the lower set topology. -/
def map (f : α →o β) : C(WithLowerSet α, WithLowerSet β) where
  toFun := toLowerSet ∘ f ∘ ofLowerSet
  continuous_toFun := continuous_def.2 fun _s hs ↦ IsLowerSet.preimage hs f.monotone


@[simp] lemma toLowerSet_specializes_toLowerSet {a b : α} :
  toLowerSet a ⤳ toLowerSet b ↔ a ≤ b := by
  simp_rw [specializes_iff_closure_subset, IsLowerSet.closure_singleton, Ici_subset_Ici,
    toLowerSet_le_iff]


@[simp] lemma ofLowerSet_le_ofLowerSet {a b : WithLowerSet α} :
    ofLowerSet a ≤ ofLowerSet b ↔ a ⤳ b := toLowerSet_specializes_toLowerSet.symm


@[simp] lemma isLowerSet_toLowerSet_preimage {s : Set (WithLowerSet α)} :
    IsLowerSet (toLowerSet ⁻¹' s) ↔ IsOpen s := Iff.rfl


@[simp] lemma isOpen_ofLowerSet_preimage {s : Set α} :
    IsOpen (ofLowerSet ⁻¹' s) ↔ IsLowerSet s := isLowerSet_toLowerSet_preimage.symm


