/-- A permutation is a derangement if it has no fixed points. -/
def derangements (α : Type*) : Set (Perm α) :=
  { f : Perm α | ∀ x : α, f x ≠ x }


theorem mem_derangements_iff_fixedPoints_eq_empty {f : Perm α} :
    f ∈ derangements α ↔ fixedPoints f = ∅ :=
  Set.eq_empty_iff_forall_not_mem.symm


/-- If `α` is equivalent to `β`, then `derangements α` is equivalent to `derangements β`. -/
def Equiv.derangementsCongr (e : α ≃ β) : derangements α ≃ derangements β :=
  e.permCongr.subtypeEquiv fun {f} => e.forall_congr <| by
   /-
     α : Type u_1
     β : Type u_2
     e : Equiv α β
     f : Equiv.Perm α
     ⊢ ∀ (a : α), Iff (Ne (f a) a) (Ne ((e.permCongr f) (e a)) (e a))
   -/
   intro b; simp only [ne_eq, permCongr_apply, symm_apply_apply, EmbeddingLike.apply_eq_iff_eq]
            /-
              🎉 no goals
            -/


/-- Derangements on a subtype are equivalent to permutations on the original type where points are
fixed iff they are not in the subtype. -/
protected def subtypeEquiv (p : α → Prop) [DecidablePred p] :
    derangements (Subtype p) ≃ { f : Perm α // ∀ a, ¬p a ↔ a ∈ fixedPoints f } :=
  calc
    derangements (Subtype p) ≃ { f : { f : Perm α // ∀ a, ¬p a → a ∈ fixedPoints f } //
        ∀ a, a ∈ fixedPoints f → ¬p a } := by
      /-
        α : Type u_1
        β : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        ⊢ Equiv (↑(derangements (Subtype p))) (Subtype fun f => ∀ (a : α), Membership. …
      -/
      refine (Perm.subtypeEquivSubtypePerm p).subtypeEquiv fun f => ⟨fun hf a hfa ha => ?_, ?_⟩
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          p : α → Prop
          inst✝ : DecidablePred p
          f : Equiv.Perm (Subtype p)
          hf : Membership.mem (derangements (Subtype p)) f
          a : α
          hfa : Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEquivSubtypeP …
          ha : p a
          ⊢ False
        -/
      · refine hf ⟨a, ha⟩ (Subtype.ext ?_)
        simp_rw [mem_fixedPoints, IsFixedPt, Perm.subtypeEquivSubtypePerm,
        Equiv.coe_fn_mk, Perm.ofSubtype_apply_of_mem _ ha] at hfa
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          p : α → Prop
          inst✝ : DecidablePred p
          f : Equiv.Perm (Subtype p)
          hf : Membership.mem (derangements (Subtype p)) f
          a : α
          ha : p a
          hfa : Eq (↑(f ⟨a, ha⟩)) a
          ⊢ Eq ↑(f ⟨a, ha⟩) ↑⟨a, ha⟩
        -/
        assumption
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        α : Type u_1
        β : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        f : Equiv.Perm (Subtype p)
        ⊢ (∀ (a : α), Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEquiv …
      -/
      rintro hf ⟨a, ha⟩ hfa
      /-
        case refine_2.mk
        α : Type u_1
        β : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        f : Equiv.Perm (Subtype p)
        hf : ∀ (a : α), Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEqu …
        a : α
        ha : p a
        hfa : Eq (f ⟨a, ha⟩) ⟨a, ha⟩
        ⊢ False
      -/
      refine hf _ ?_ ha
      /-
        case refine_2.mk
        α : Type u_1
        β : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        f : Equiv.Perm (Subtype p)
        hf : ∀ (a : α), Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEqu …
        a : α
        ha : p a
        hfa : Eq (f ⟨a, ha⟩) ⟨a, ha⟩
        ⊢ Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEquivSubtypePerm  …
      -/
      simp only [Perm.subtypeEquivSubtypePerm_apply_coe, mem_fixedPoints]
      /-
        case refine_2.mk
        α : Type u_1
        β : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        f : Equiv.Perm (Subtype p)
        hf : ∀ (a : α), Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEqu …
        a : α
        ha : p a
        hfa : Eq (f ⟨a, ha⟩) ⟨a, ha⟩
        ⊢ Function.IsFixedPt (⇑(Equiv.Perm.ofSubtype f)) a
      -/
      dsimp [IsFixedPt]
      /-
        case refine_2.mk
        α : Type u_1
        β : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        f : Equiv.Perm (Subtype p)
        hf : ∀ (a : α), Membership.mem (Function.fixedPoints ⇑↑((Equiv.Perm.subtypeEqu …
        a : α
        ha : p a
        hfa : Eq (f ⟨a, ha⟩) ⟨a, ha⟩
        ⊢ Eq ((Equiv.Perm.ofSubtype f) a) a
      -/
      simp_rw [Perm.ofSubtype_apply_of_mem _ ha, hfa]
      /-
        🎉 no goals
      -/
    _ ≃ { f : Perm α // ∃ _h : ∀ a, ¬p a → a ∈ fixedPoints f, ∀ a, a ∈ fixedPoints f → ¬p a } :=
      subtypeSubtypeEquivSubtypeExists _ _
    _ ≃ { f : Perm α // ∀ a, ¬p a ↔ a ∈ fixedPoints f } :=
      subtypeEquivRight fun f => by
        /-
          α : Type u_1
          β : Type u_2
          p : α → Prop
          inst✝ : DecidablePred p
          f : Equiv.Perm α
          ⊢ Iff (Exists fun _h => ∀ (a : α), Membership.mem (Function.fixedPoints ⇑f) a  …
        -/
        simp_rw [exists_prop, ← forall_and, ← iff_iff_implies_and_implies]
        /-
          🎉 no goals
        -/


/-- The set of permutations that fix either `a` or nothing is equivalent to the sum of:
    - derangements on `α`
    - derangements on `α` minus `a`. -/
def atMostOneFixedPointEquivSum_derangements [DecidableEq α] (a : α) :
    { f : Perm α // fixedPoints f ⊆ {a} } ≃ (derangements ({a}ᶜ : Set α)) ⊕ (derangements α) :=
  calc
    { f : Perm α // fixedPoints f ⊆ {a} } ≃
        { f : { f : Perm α // fixedPoints f ⊆ {a} } // a ∈ fixedPoints f } ⊕
          { f : { f : Perm α // fixedPoints f ⊆ {a} } // a ∉ fixedPoints f } :=
      (Equiv.sumCompl _).symm
    _ ≃ { f : Perm α // fixedPoints f ⊆ {a} ∧ a ∈ fixedPoints f } ⊕
          { f : Perm α // fixedPoints f ⊆ {a} ∧ a ∉ fixedPoints f } := by
      -- Porting note: `subtypeSubtypeEquivSubtypeInter` no longer works with placeholder `_`s.
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        a : α
        ⊢ Equiv (Sum (Subtype fun f => Membership.mem (Function.fixedPoints ⇑↑f) a) (S …
      -/
      refine Equiv.sumCongr ?_ ?_
      · exact subtypeSubtypeEquivSubtypeInter
          (fun x : Perm α => fixedPoints x ⊆ {a})
          (a ∈ fixedPoints ·)
      · exact subtypeSubtypeEquivSubtypeInter
          (fun x : Perm α => fixedPoints x ⊆ {a})
          (¬a ∈ fixedPoints ·)
    _ ≃ { f : Perm α // fixedPoints f = {a} } ⊕ { f : Perm α // fixedPoints f = ∅ } := by
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        a : α
        ⊢ Equiv (Sum (Subtype fun f => And (HasSubset.Subset (Function.fixedPoints ⇑f) …
      -/
      refine Equiv.sumCongr (subtypeEquivRight fun f => ?_) (subtypeEquivRight fun f => ?_)
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          ⊢ Iff (And (HasSubset.Subset (Function.fixedPoints ⇑f) (Singleton.singleton a) …
        -/
      · rw [Set.eq_singleton_iff_unique_mem, and_comm]
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          ⊢ Iff (And (Membership.mem (Function.fixedPoints ⇑f) a) (HasSubset.Subset (Fun …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          β : Type u_2
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          ⊢ Iff (And (HasSubset.Subset (Function.fixedPoints ⇑f) (Singleton.singleton a) …
        -/
      · rw [Set.eq_empty_iff_forall_not_mem]
        /-
          case refine_2
          α : Type u_1
          β : Type u_2
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          ⊢ Iff (And (HasSubset.Subset (Function.fixedPoints ⇑f) (Singleton.singleton a) …
        -/
        exact ⟨fun h x hx => h.2 (h.1 hx ▸ hx), fun h => ⟨fun x hx => (h _ hx).elim, h _⟩⟩
        /-
          🎉 no goals
        -/
    _ ≃ derangements ({a}ᶜ : Set α) ⊕ derangements α := by
      -- Porting note: was `subtypeEquiv _` but now needs the placeholder to be provided explicitly
      refine
        Equiv.sumCongr ((derangements.subtypeEquiv (· ∈ ({a}ᶜ : Set α))).trans <|
            subtypeEquivRight fun x => ?_).symm
          (subtypeEquivRight fun f => mem_derangements_iff_fixedPoints_eq_empty.symm)
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        a : α
        x : Equiv.Perm α
        ⊢ Iff (∀ (a_1 : α), Iff (Not (Membership.mem (HasCompl.compl (Singleton.single …
      -/
      rw [eq_comm, Set.ext_iff]
      /-
        α : Type u_1
        β : Type u_2
        inst✝ : DecidableEq α
        a : α
        x : Equiv.Perm α
        ⊢ Iff (∀ (a_1 : α), Iff (Not (Membership.mem (HasCompl.compl (Singleton.single …
      -/
      simp_rw [Set.mem_compl_iff, Classical.not_not]
      /-
        🎉 no goals
      -/


/-- The set of permutations `f` such that the preimage of `(a, f)` under
    `Equiv.Perm.decomposeOption` is a derangement. -/
def RemoveNone.fiber (a : Option α) : Set (Perm α) :=
  { f : Perm α | (a, f) ∈ Equiv.Perm.decomposeOption '' derangements (Option α) }


theorem RemoveNone.mem_fiber (a : Option α) (f : Perm α) :
    f ∈ RemoveNone.fiber a ↔
      ∃ F : Perm (Option α), F ∈ derangements (Option α) ∧ F none = a ∧ removeNone F = f := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : Option α
    f : Equiv.Perm α
    ⊢ Iff (Membership.mem (derangements.Equiv.RemoveNone.fiber a) f) (Exists fun F …
  -/
  simp [RemoveNone.fiber, derangements]
  /-
    🎉 no goals
  -/


theorem RemoveNone.fiber_none : RemoveNone.fiber (@none α) = ∅ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ Eq (derangements.Equiv.RemoveNone.fiber Option.none) EmptyCollection.emptyCo …
  -/
  rw [Set.eq_empty_iff_forall_not_mem]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ ∀ (x : Equiv.Perm α), Not (Membership.mem (derangements.Equiv.RemoveNone.fib …
  -/
  intro f hyp
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hyp : Membership.mem (derangements.Equiv.RemoveNone.fiber Option.none) f
    ⊢ False
  -/
  rw [RemoveNone.mem_fiber] at hyp
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    hyp : Exists fun F => And (Membership.mem (derangements (Option α)) F) (And (E …
    ⊢ False
  -/
  rcases hyp with ⟨F, F_derangement, F_none, _⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    f : Equiv.Perm α
    F : Equiv.Perm (Option α)
    F_derangement : Membership.mem (derangements (Option α)) F
    F_none : Eq (F Option.none) Option.none
    right✝ : Eq (Equiv.removeNone F) f
    ⊢ False
  -/
  exact F_derangement none F_none
  /-
    🎉 no goals
  -/


/-- For any `a : α`, the fiber over `some a` is the set of permutations
    where `a` is the only possible fixed point. -/
theorem RemoveNone.fiber_some (a : α) :
    RemoveNone.fiber (some a) = { f : Perm α | fixedPoints f ⊆ {a} } := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (derangements.Equiv.RemoveNone.fiber (Option.some a)) (setOf fun f => Has …
  -/
  ext f
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    f : Equiv.Perm α
    ⊢ Iff (Membership.mem (derangements.Equiv.RemoveNone.fiber (Option.some a)) f) …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      f : Equiv.Perm α
      ⊢ Membership.mem (derangements.Equiv.RemoveNone.fiber (Option.some a)) f → Mem …
    -/
  · rw [RemoveNone.mem_fiber]
    /-
      case h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      f : Equiv.Perm α
      ⊢ (Exists fun F => And (Membership.mem (derangements (Option α)) F) (And (Eq ( …
    -/
    rintro ⟨F, F_derangement, F_none, rfl⟩ x x_fixed
    /-
      case h.mp.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      F : Equiv.Perm (Option α)
      F_derangement : Membership.mem (derangements (Option α)) F
      F_none : Eq (F Option.none) (Option.some a)
      x : α
      x_fixed : Membership.mem (Function.fixedPoints ⇑(Equiv.removeNone F)) x
      ⊢ Membership.mem (Singleton.singleton a) x
    -/
    rw [mem_fixedPoints_iff] at x_fixed
    /-
      case h.mp.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      F : Equiv.Perm (Option α)
      F_derangement : Membership.mem (derangements (Option α)) F
      F_none : Eq (F Option.none) (Option.some a)
      x : α
      x_fixed : Eq ((Equiv.removeNone F) x) x
      ⊢ Membership.mem (Singleton.singleton a) x
    -/
    apply_fun some at x_fixed
    /-
      case h.mp.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      F : Equiv.Perm (Option α)
      F_derangement : Membership.mem (derangements (Option α)) F
      F_none : Eq (F Option.none) (Option.some a)
      x : α
      x_fixed : Eq (Option.some ((Equiv.removeNone F) x)) (Option.some x)
      ⊢ Membership.mem (Singleton.singleton a) x
    -/
    cases' Fx : F (some x) with y
      /-
        case h.mp.intro.intro.intro.none
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        F : Equiv.Perm (Option α)
        F_derangement : Membership.mem (derangements (Option α)) F
        F_none : Eq (F Option.none) (Option.some a)
        x : α
        x_fixed : Eq (Option.some ((Equiv.removeNone F) x)) (Option.some x)
        Fx : Eq (F (Option.some x)) Option.none
        ⊢ Membership.mem (Singleton.singleton a) x
      -/
    · rwa [removeNone_none F Fx, F_none, Option.some_inj, eq_comm] at x_fixed
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.intro.intro.some
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        F : Equiv.Perm (Option α)
        F_derangement : Membership.mem (derangements (Option α)) F
        F_none : Eq (F Option.none) (Option.some a)
        x : α
        x_fixed : Eq (Option.some ((Equiv.removeNone F) x)) (Option.some x)
        y : α
        Fx : Eq (F (Option.some x)) (Option.some y)
        ⊢ Membership.mem (Singleton.singleton a) x
      -/
    · exfalso
      /-
        case h.mp.intro.intro.intro.some
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        F : Equiv.Perm (Option α)
        F_derangement : Membership.mem (derangements (Option α)) F
        F_none : Eq (F Option.none) (Option.some a)
        x : α
        x_fixed : Eq (Option.some ((Equiv.removeNone F) x)) (Option.some x)
        y : α
        Fx : Eq (F (Option.some x)) (Option.some y)
        ⊢ False
      -/
      rw [removeNone_some F ⟨y, Fx⟩] at x_fixed
      /-
        case h.mp.intro.intro.intro.some
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        F : Equiv.Perm (Option α)
        F_derangement : Membership.mem (derangements (Option α)) F
        F_none : Eq (F Option.none) (Option.some a)
        x : α
        x_fixed : Eq (F (Option.some x)) (Option.some x)
        y : α
        Fx : Eq (F (Option.some x)) (Option.some y)
        ⊢ False
      -/
      exact F_derangement _ x_fixed
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      f : Equiv.Perm α
      ⊢ Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints ⇑f) (S …
    -/
  · intro h_opfp
    /-
      case h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      f : Equiv.Perm α
      h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
      ⊢ Membership.mem (derangements.Equiv.RemoveNone.fiber (Option.some a)) f
    -/
    use Equiv.Perm.decomposeOption.symm (some a, f)
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      f : Equiv.Perm α
      h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
      ⊢ And (Membership.mem (derangements (Option α)) (Equiv.Perm.decomposeOption.sy …
    -/
    constructor
      /-
        case h.left
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        ⊢ Membership.mem (derangements (Option α)) (Equiv.Perm.decomposeOption.symm {  …
      -/
    · intro x
      /-
        case h.left
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : Option α
        ⊢ Ne ((Equiv.Perm.decomposeOption.symm { fst := Option.some a, snd := f }) x) x
      -/
      apply_fun fun x => Equiv.swap none (some a) x
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : Option α
        ⊢ Ne ((fun x => (Equiv.swap Option.none (Option.some a)) x) ((Equiv.Perm.decom …
      -/
      simp only [Perm.decomposeOption_symm_apply, swap_apply_self, Perm.coe_mul]
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : Option α
        ⊢ Ne ((Equiv.swap Option.none (Option.some a)) (Function.comp (⇑(Equiv.swap Op …
      -/
      cases' x with x
        /-
          case none
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
          ⊢ Ne ((Equiv.swap Option.none (Option.some a)) (Function.comp (⇑(Equiv.swap Op …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case some
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        ⊢ Ne ((Equiv.swap Option.none (Option.some a)) (Function.comp (⇑(Equiv.swap Op …
      -/
      simp only [comp, optionCongr_apply, Option.map_some', swap_apply_self]
      /-
        case some
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        ⊢ Ne (Option.some (f x)) ((Equiv.swap Option.none (Option.some a)) (Option.som …
      -/
      by_cases x_vs_a : x = a
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
          x : α
          x_vs_a : Eq x a
          ⊢ Ne (Option.some (f x)) ((Equiv.swap Option.none (Option.some a)) (Option.som …
        -/
      · rw [x_vs_a, swap_apply_right]
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          a : α
          f : Equiv.Perm α
          h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
          x : α
          x_vs_a : Eq x a
          ⊢ Ne (Option.some (f a)) Option.none
        -/
        apply Option.some_ne_none
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        x_vs_a : Not (Eq x a)
        ⊢ Ne (Option.some (f x)) ((Equiv.swap Option.none (Option.some a)) (Option.som …
      -/
      have ne_1 : some x ≠ none := Option.some_ne_none _
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        x_vs_a : Not (Eq x a)
        ne_1 : Ne (Option.some x) Option.none
        ⊢ Ne (Option.some (f x)) ((Equiv.swap Option.none (Option.some a)) (Option.som …
      -/
      have ne_2 : some x ≠ some a := (Option.some_injective α).ne_iff.mpr x_vs_a
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        x_vs_a : Not (Eq x a)
        ne_1 : Ne (Option.some x) Option.none
        ne_2 : Ne (Option.some x) (Option.some a)
        ⊢ Ne (Option.some (f x)) ((Equiv.swap Option.none (Option.some a)) (Option.som …
      -/
      rw [swap_apply_of_ne_of_ne ne_1 ne_2, (Option.some_injective α).ne_iff]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        x_vs_a : Not (Eq x a)
        ne_1 : Ne (Option.some x) Option.none
        ne_2 : Ne (Option.some x) (Option.some a)
        ⊢ Ne (f x) x
      -/
      intro contra
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        x : α
        x_vs_a : Not (Eq x a)
        ne_1 : Ne (Option.some x) Option.none
        ne_2 : Ne (Option.some x) (Option.some a)
        contra : Eq (f x) x
        ⊢ False
      -/
      exact x_vs_a (h_opfp contra)
      /-
        🎉 no goals
      -/
      /-
        case h.right
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        f : Equiv.Perm α
        h_opfp : Membership.mem (setOf fun f => HasSubset.Subset (Function.fixedPoints …
        ⊢ Eq (Equiv.Perm.decomposeOption (Equiv.Perm.decomposeOption.symm { fst := Opt …
      -/
    · rw [apply_symm_apply]
      /-
        🎉 no goals
      -/


/-- The set of derangements on `Option α` is equivalent to the union over `a : α`
    of "permutations with `a` the only possible fixed point". -/
def derangementsOptionEquivSigmaAtMostOneFixedPoint :
    derangements (Option α) ≃ Σa : α, { f : Perm α | fixedPoints f ⊆ {a} } := by
  have fiber_none_is_false : Equiv.RemoveNone.fiber (@none α) → False := by
    rw [Equiv.RemoveNone.fiber_none]
    exact IsEmpty.false
  calc
    derangements (Option α) ≃ Equiv.Perm.decomposeOption '' derangements (Option α) :=
      Equiv.image _ _
    _ ≃ Σa : Option α, ↥(Equiv.RemoveNone.fiber a) := setProdEquivSigma _
    _ ≃ Σa : α, ↥(Equiv.RemoveNone.fiber (some a)) :=
      sigmaOptionEquivOfSome _ fiber_none_is_false
    _ ≃ Σa : α, { f : Perm α | fixedPoints f ⊆ {a} } := by
      simp_rw [Equiv.RemoveNone.fiber_some]
      rfl


/-- The set of derangements on `Option α` is equivalent to the union over all `a : α` of
    "derangements on `α` ⊕ derangements on `{a}ᶜ`". -/
def derangementsRecursionEquiv :
    derangements (Option α) ≃
      Σa : α, derangements (({a}ᶜ : Set α) : Type _) ⊕ derangements α :=
  derangementsOptionEquivSigmaAtMostOneFixedPoint.trans
    (sigmaCongrRight atMostOneFixedPointEquivSum_derangements)


