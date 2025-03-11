open Topology.IsScott in
@[simp] lemma Topology.IsScott.ωscottContinuous_iff_continuous {α : Type*}
    [OmegaCompletePartialOrder α] [TopologicalSpace α]
    [Topology.IsScott α (Set.range fun c : Chain α => Set.range c)] {f : α → Prop} :
    ωScottContinuous f ↔ Continuous f := by
  rw [ωScottContinuous, scottContinuous_iff_continuous (fun a b hab => by
    use Chain.pair a b hab; exact OmegaCompletePartialOrder.Chain.range_pair a b hab)]

-- "Scott", "ωSup"

/-- `x` is an `ω`-Sup of a chain `c` if it is the least upper bound of the range of `c`. -/
def IsωSup {α : Type u} [Preorder α] (c : Chain α) (x : α) : Prop :=
  (∀ i, c i ≤ x) ∧ ∀ y, (∀ i, c i ≤ y) → x ≤ y


theorem isωSup_iff_isLUB {α : Type u} [Preorder α] {c : Chain α} {x : α} :
    IsωSup c x ↔ IsLUB (range c) x := by
  /-
    α : Type u
    inst✝ : Preorder α
    c : OmegaCompletePartialOrder.Chain α
    x : α
    ⊢ Iff (Scott.IsωSup c x) (IsLUB (Set.range ⇑c) x)
  -/
  simp [IsωSup, IsLUB, IsLeast, upperBounds, lowerBounds]
  /-
    🎉 no goals
  -/


/-- The characteristic function of open sets is monotone and preserves
the limits of chains. -/
def IsOpen (s : Set α) : Prop :=
  ωScottContinuous fun x ↦ x ∈ s


theorem isOpen_univ : IsOpen α univ := @CompleteLattice.ωScottContinuous.top α Prop _ _


theorem IsOpen.inter (s t : Set α) : IsOpen α s → IsOpen α t → IsOpen α (s ∩ t) :=
  CompleteLattice.ωScottContinuous.inf


theorem isOpen_sUnion (s : Set (Set α)) (hs : ∀ t ∈ s, IsOpen α t) : IsOpen α (⋃₀ s) := by
  /-
    α : Type u
    inst✝ : OmegaCompletePartialOrder α
    s : Set (Set α)
    hs : ∀ (t : Set α), Membership.mem s t → Scott.IsOpen α t
    ⊢ Scott.IsOpen α s.sUnion
  -/
  simp only [IsOpen] at hs ⊢
  /-
    α : Type u
    inst✝ : OmegaCompletePartialOrder α
    s : Set (Set α)
    hs : ∀ (t : Set α), Membership.mem s t → OmegaCompletePartialOrder.ωScottConti …
    ⊢ OmegaCompletePartialOrder.ωScottContinuous fun x => Membership.mem s.sUnion x
  -/
  convert CompleteLattice.ωScottContinuous.sSup hs
  /-
    case h.e'_5.h.h.e
    α : Type u
    inst✝ : OmegaCompletePartialOrder α
    s : Set (Set α)
    hs : ∀ (t : Set α), Membership.mem s t → OmegaCompletePartialOrder.ωScottConti …
    x✝ : α
    ⊢ Eq (Membership.mem s.sUnion) (SupSet.sSup s)
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem IsOpen.isUpperSet {s : Set α} (hs : IsOpen α s) : IsUpperSet s := hs.monotone


/-- A Scott topological space is defined on preorders
such that their open sets, seen as a function `α → Prop`,
preserves the joins of ω-chains  -/
abbrev Scott (α : Type u) := α


instance Scott.topologicalSpace (α : Type u) [OmegaCompletePartialOrder α] :
    TopologicalSpace (Scott α) where
  IsOpen := Scott.IsOpen α
  isOpen_univ := Scott.isOpen_univ α
  isOpen_inter := Scott.IsOpen.inter α
  isOpen_sUnion := Scott.isOpen_sUnion α


lemma isOpen_iff_ωScottContinuous_mem {α} [OmegaCompletePartialOrder α] {s : Set (Scott α)} :
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : OmegaCompletePartialOrder α
                                                      s : Set (Scott α)
                                                      ⊢ Iff (IsOpen s) (OmegaCompletePartialOrder.ωScottContinuous fun x => Membersh …
                                                    -/
    IsOpen s ↔ ωScottContinuous fun x ↦ x ∈ s := by rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma scott_eq_Scott {α} [OmegaCompletePartialOrder α] :
    Topology.scott α (Set.range fun c : Chain α => Set.range c) = Scott.topologicalSpace α := by
  /-
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    ⊢ Eq (Topology.scott α (Set.range fun c => Set.range ⇑c)) (Scott.topologicalSp …
  -/
  ext U
  /-
    case a.h.a
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    U : Set α
    ⊢ Iff (IsOpen U) (IsOpen U)
  -/
  letI := Topology.scott α (Set.range fun c : Chain α => Set.range c)
  rw [isOpen_iff_ωScottContinuous_mem, @isOpen_iff_continuous_mem,
    @Topology.IsScott.ωscottContinuous_iff_continuous _ _
      (Topology.scott α (Set.range fun c : Chain α => Set.range c)) ({ topology_eq_scott := rfl })]


/-- `notBelow` is an open set in `Scott α` used
to prove the monotonicity of continuous functions -/
def notBelow :=
  { x | ¬x ≤ y }


theorem notBelow_isOpen : IsOpen (notBelow y) := by
  /-
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    y : Scott α
    ⊢ IsOpen (notBelow y)
  -/
  have h : Monotone (notBelow y) := fun x z hle ↦ mt hle.trans
  /-
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    y : Scott α
    h : Monotone (notBelow y)
    ⊢ IsOpen (notBelow y)
  -/
  dsimp only [IsOpen, TopologicalSpace.IsOpen, Scott.IsOpen]
  /-
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    y : Scott α
    h : Monotone (notBelow y)
    ⊢ OmegaCompletePartialOrder.ωScottContinuous fun x => Membership.mem (notBelow …
  -/
  rw [ωScottContinuous_iff_monotone_map_ωSup]
  /-
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    y : Scott α
    h : Monotone (notBelow y)
    ⊢ Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (Membership.m …
  -/
  refine ⟨h, fun c ↦ eq_of_forall_ge_iff fun z ↦ ?_⟩
  simp only [ωSup_le_iff, notBelow, mem_setOf_eq, le_Prop_eq, OrderHom.coe_mk, Chain.map_coe,
    Function.comp_apply, exists_imp, not_forall]


theorem isωSup_ωSup {α} [OmegaCompletePartialOrder α] (c : Chain α) : IsωSup c (ωSup c) := by
  /-
    α : Type u_1
    inst✝ : OmegaCompletePartialOrder α
    c : OmegaCompletePartialOrder.Chain α
    ⊢ Scott.IsωSup c (OmegaCompletePartialOrder.ωSup c)
  -/
  constructor
    /-
      case left
      α : Type u_1
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      ⊢ ∀ (i : Nat), LE.le (c i) (OmegaCompletePartialOrder.ωSup c)
    -/
  · apply le_ωSup
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      ⊢ ∀ (y : α), (∀ (i : Nat), LE.le (c i) y) → LE.le (OmegaCompletePartialOrder.ω …
    -/
  · apply ωSup_le
    /-
      🎉 no goals
    -/


theorem scottContinuous_of_continuous {α β} [OmegaCompletePartialOrder α]
    [OmegaCompletePartialOrder β] (f : Scott α → Scott β) (hf : _root_.Continuous f) :
    OmegaCompletePartialOrder.ωScottContinuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    ⊢ OmegaCompletePartialOrder.ωScottContinuous f
  -/
  rw [ωScottContinuous_iff_monotone_map_ωSup]
  have h : Monotone f := fun x y h ↦ by
    have hf : IsUpperSet {x | ¬f x ≤ f y} := ((notBelow_isOpen (f y)).preimage hf).isUpperSet
    simpa only [mem_setOf_eq, le_refl, not_true, imp_false, not_not] using hf h
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    ⊢ Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain (Scott α)), Eq (f (O …
  -/
  refine ⟨h, fun c ↦ eq_of_forall_ge_iff fun z ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    c : OmegaCompletePartialOrder.Chain (Scott α)
    z : Scott β
    ⊢ Iff (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z) (LE.le (OmegaCompletePa …
  -/
  rcases (notBelow_isOpen z).preimage hf with hf''
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    c : OmegaCompletePartialOrder.Chain (Scott α)
    z : Scott β
    hf'' : IsOpen (Set.preimage f (notBelow z))
    ⊢ Iff (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z) (LE.le (OmegaCompletePa …
  -/
  let hf' := hf''.monotone_map_ωSup.2
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    c : OmegaCompletePartialOrder.Chain (Scott α)
    z : Scott β
    hf'' : IsOpen (Set.preimage f (notBelow z))
    hf' : ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (Membership.mem (Set.preim …
    ⊢ Iff (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z) (LE.le (OmegaCompletePa …
  -/
  specialize hf' c
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    c : OmegaCompletePartialOrder.Chain (Scott α)
    z : Scott β
    hf'' : IsOpen (Set.preimage f (notBelow z))
    hf' : Eq (Membership.mem (Set.preimage f (notBelow z)) (OmegaCompletePartialOr …
    ⊢ Iff (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z) (LE.le (OmegaCompletePa …
  -/
  simp only [OrderHom.coe_mk, mem_preimage, notBelow, mem_setOf_eq] at hf'
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    c : OmegaCompletePartialOrder.Chain (Scott α)
    z : Scott β
    hf'' : IsOpen (Set.preimage f (notBelow z))
    hf' : Eq (Not (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z)) (OmegaComplete …
    ⊢ Iff (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z) (LE.le (OmegaCompletePa …
  -/
  rw [← not_iff_not]
  simp only [ωSup_le_iff, hf', ωSup, iSup, sSup, mem_range, Chain.map_coe, Function.comp_apply,
    eq_iff_iff, not_forall, OrderHom.coe_mk]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : Continuous f
    h : Monotone f
    c : OmegaCompletePartialOrder.Chain (Scott α)
    z : Scott β
    hf'' : IsOpen (Set.preimage f (notBelow z))
    hf' : Eq (Not (LE.le (f (OmegaCompletePartialOrder.ωSup c)) z)) (OmegaComplete …
    ⊢ Iff (Exists fun a => And (Exists fun y => Iff (Not (LE.le (f (c y)) z)) a) a …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem continuous_of_scottContinuous {α β} [OmegaCompletePartialOrder α]
    [OmegaCompletePartialOrder β] (f : Scott α → Scott β)
    (hf : ωScottContinuous f) : Continuous f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    ⊢ Continuous f
  -/
  rw [continuous_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    ⊢ ∀ (s : Set (Scott β)), IsOpen s → IsOpen (Set.preimage f s)
  -/
  intro s hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    s : Set (Scott β)
    hs : IsOpen s
    ⊢ IsOpen (Set.preimage f s)
  -/
  dsimp only [IsOpen, TopologicalSpace.IsOpen, Scott.IsOpen]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    s : Set (Scott β)
    hs : IsOpen s
    ⊢ OmegaCompletePartialOrder.ωScottContinuous fun x => Membership.mem (Set.prei …
  -/
  simp_rw [mem_preimage, mem_def, ← Function.comp_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : Scott α → Scott β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    s : Set (Scott β)
    hs : IsOpen s
    ⊢ OmegaCompletePartialOrder.ωScottContinuous (Function.comp s f)
  -/
  apply ωScottContinuous.comp hs hf
  /-
    🎉 no goals
  -/

