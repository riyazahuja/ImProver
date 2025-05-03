theorem not_injective_limitation_set : ¬ InjOn g (Iio (ord <| succ #α)) := by
  /-
    α : Type u
    g : Ordinal.{u} → α
    ⊢ Not (Set.InjOn g (Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord))
  -/
  intro h_inj
  /-
    α : Type u
    g : Ordinal.{u} → α
    h_inj : Set.InjOn g (Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
    ⊢ False
  -/
  have h := lift_mk_le_lift_mk_of_injective <| injOn_iff_injective.1 h_inj
  have mk_initialSeg_subtype :
      #(Iio (ord <| succ #α)) = lift.{u + 1} (succ #α) := by
    simpa only [coe_setOf, card_typein, card_ord] using mk_Iio_ordinal (ord <| succ #α)
  /-
    α : Type u
    g : Ordinal.{u} → α
    h_inj : Set.InjOn g (Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
    h : LE.le (Cardinal.lift.{u, u + 1} (Cardinal.mk ↑(Set.Iio (SuccOrder.succ (Ca …
    mk_initialSeg_subtype : Eq (Cardinal.mk ↑(Set.Iio (SuccOrder.succ (Cardinal.mk …
    ⊢ False
  -/
  rw [mk_initialSeg_subtype, lift_lift, lift_le] at h
  /-
    α : Type u
    g : Ordinal.{u} → α
    h_inj : Set.InjOn g (Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
    h : LE.le (SuccOrder.succ (Cardinal.mk α)) (Cardinal.mk α)
    mk_initialSeg_subtype : Eq (Cardinal.mk ↑(Set.Iio (SuccOrder.succ (Cardinal.mk …
    ⊢ False
  -/
  exact not_le_of_lt (Order.lt_succ #α) h
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
/-- The ordinal-indexed sequence approximating the least fixed point greater than
an initial value `x`. It is defined in such a way that we have `lfpApprox 0 x = x` and
`lfpApprox a x = ⨆ b < a, f (lfpApprox b x)`. -/
def lfpApprox (a : Ordinal.{u}) : α :=
  sSup ({ f (lfpApprox b) | (b : Ordinal) (h : b < a) } ∪ {x})
termination_by a
/-
  a b : Ordinal.{u}
  h : LT.lt b a
  ⊢ LT.lt b a
-/
decreasing_by exact h
/-
  🎉 no goals
-/


theorem lfpApprox_monotone : Monotone (lfpApprox f x) := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    ⊢ Monotone (OrdinalApprox.lfpApprox f x)
  -/
  intros a b h
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h : LE.le a b
    ⊢ LE.le (OrdinalApprox.lfpApprox f x a) (OrdinalApprox.lfpApprox f x b)
  -/
  rw [lfpApprox, lfpApprox]
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h : LE.le a b
    ⊢ LE.le (SupSet.sSup (Union.union (setOf fun x_1 => Exists fun b => Exists fun …
  -/
  refine sSup_le_sSup ?h
  /-
    case h
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h : LE.le a b
    ⊢ HasSubset.Subset (Union.union (setOf fun x_1 => Exists fun b => Exists fun h …
  -/
  apply sup_le_sup_right
  simp only [exists_prop, Set.le_eq_subset, Set.setOf_subset_setOf, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂]
  /-
    case h.h₁
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h : LE.le a b
    ⊢ ∀ (a_1 : Ordinal.{u}), LT.lt a_1 a → Exists fun b_1 => And (LT.lt b_1 b) (Eq …
  -/
  intros a' h'
  /-
    case h.h₁
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h : LE.le a b
    a' : Ordinal.{u}
    h' : LT.lt a' a
    ⊢ Exists fun b_1 => And (LT.lt b_1 b) (Eq (f (OrdinalApprox.lfpApprox f x b_1) …
  -/
  use a'
  /-
    case h
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h : LE.le a b
    a' : Ordinal.{u}
    h' : LT.lt a' a
    ⊢ And (LT.lt a' b) (Eq (f (OrdinalApprox.lfpApprox f x a')) (f (OrdinalApprox. …
  -/
  exact ⟨lt_of_lt_of_le h' h, rfl⟩
  /-
    🎉 no goals
  -/


theorem le_lfpApprox {a : Ordinal} : x ≤ lfpApprox f x a := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a : Ordinal.{u}
    ⊢ LE.le x (OrdinalApprox.lfpApprox f x a)
  -/
  rw [lfpApprox]
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a : Ordinal.{u}
    ⊢ LE.le x (SupSet.sSup (Union.union (setOf fun x_1 => Exists fun b => Exists f …
  -/
  apply le_sSup
  /-
    case a
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a : Ordinal.{u}
    ⊢ Membership.mem (Union.union (setOf fun x_1 => Exists fun b => Exists fun h = …
  -/
  simp only [exists_prop, Set.union_singleton, Set.mem_insert_iff, Set.mem_setOf_eq, true_or]
  /-
    🎉 no goals
  -/


theorem lfpApprox_add_one (h : x ≤ f x) (a : Ordinal) :
    lfpApprox f x (a+1) = f (lfpApprox f x a) := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h : LE.le x (f x)
    a : Ordinal.{u}
    ⊢ Eq (OrdinalApprox.lfpApprox f x (HAdd.hAdd a 1)) (f (OrdinalApprox.lfpApprox …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ LE.le (OrdinalApprox.lfpApprox f x (HAdd.hAdd a 1)) (f (OrdinalApprox.lfpApp …
    -/
  · conv => left; rw [lfpApprox]
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ LE.le (SupSet.sSup (Union.union (setOf fun x_1 => Exists fun b => Exists fun …
    -/
    apply sSup_le
    simp only [Ordinal.add_one_eq_succ, lt_succ_iff, exists_prop, Set.union_singleton,
      Set.mem_insert_iff, Set.mem_setOf_eq, forall_eq_or_imp, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂]
    /-
      case a.a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ And (LE.le x (f (OrdinalApprox.lfpApprox f x a))) (∀ (a_1 : Ordinal.{u}), LE …
    -/
    apply And.intro
      /-
        case a.a.left
        α : Type u
        inst✝ : CompleteLattice α
        f : OrderHom α α
        x : α
        h : LE.le x (f x)
        a : Ordinal.{u}
        ⊢ LE.le x (f (OrdinalApprox.lfpApprox f x a))
      -/
    · apply le_trans h
      /-
        case a.a.left
        α : Type u
        inst✝ : CompleteLattice α
        f : OrderHom α α
        x : α
        h : LE.le x (f x)
        a : Ordinal.{u}
        ⊢ LE.le (f x) (f (OrdinalApprox.lfpApprox f x a))
      -/
      apply Monotone.imp f.monotone
      /-
        case a.a.left
        α : Type u
        inst✝ : CompleteLattice α
        f : OrderHom α α
        x : α
        h : LE.le x (f x)
        a : Ordinal.{u}
        ⊢ LE.le x (OrdinalApprox.lfpApprox f x a)
      -/
      exact le_lfpApprox f x
      /-
        🎉 no goals
      -/
      /-
        case a.a.right
        α : Type u
        inst✝ : CompleteLattice α
        f : OrderHom α α
        x : α
        h : LE.le x (f x)
        a : Ordinal.{u}
        ⊢ ∀ (a_1 : Ordinal.{u}), LE.le a_1 a → LE.le (f (OrdinalApprox.lfpApprox f x a …
      -/
    · intros a' h
      /-
        case a.a.right
        α : Type u
        inst✝ : CompleteLattice α
        f : OrderHom α α
        x : α
        h✝ : LE.le x (f x)
        a a' : Ordinal.{u}
        h : LE.le a' a
        ⊢ LE.le (f (OrdinalApprox.lfpApprox f x a')) (f (OrdinalApprox.lfpApprox f x a))
      -/
      apply f.2; apply lfpApprox_monotone; exact h
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ LE.le (f (OrdinalApprox.lfpApprox f x a)) (OrdinalApprox.lfpApprox f x (HAdd …
    -/
  · conv => right; rw [lfpApprox]
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ LE.le (f (OrdinalApprox.lfpApprox f x a)) (SupSet.sSup (Union.union (setOf f …
    -/
    apply le_sSup
    /-
      case a.a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ Membership.mem (Union.union (setOf fun x_1 => Exists fun b => Exists fun h = …
    -/
    simp only [Ordinal.add_one_eq_succ, lt_succ_iff, exists_prop]
    /-
      case a.a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ Membership.mem (Union.union (setOf fun x_1 => Exists fun b => And (LE.le b a …
    -/
    rw [Set.mem_union]
    /-
      case a.a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ Or (Membership.mem (setOf fun x_1 => Exists fun b => And (LE.le b a) (Eq (f  …
    -/
    apply Or.inl
    /-
      case a.a.h
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ Membership.mem (setOf fun x_1 => Exists fun b => And (LE.le b a) (Eq (f (Ord …
    -/
    simp only [Set.mem_setOf_eq]
    /-
      case a.a.h
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h : LE.le x (f x)
      a : Ordinal.{u}
      ⊢ Exists fun b => And (LE.le b a) (Eq (f (OrdinalApprox.lfpApprox f x b)) (f ( …
    -/
    use a
    /-
      🎉 no goals
    -/


theorem lfpApprox_mono_left : Monotone (lfpApprox : (α →o α) → _) := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    ⊢ Monotone OrdinalApprox.lfpApprox
  -/
  intro f g h x a
  induction a using Ordinal.induction with
  | h i ih =>
    rw [lfpApprox, lfpApprox]
    apply sSup_le
    simp only [exists_prop, Set.union_singleton, Set.mem_insert_iff, Set.mem_setOf_eq, sSup_insert,
      forall_eq_or_imp, le_sup_left, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂,
      true_and]
    intro i' h_lt
    apply le_sup_of_le_right
    apply le_sSup_of_le
    · use i'
    · apply le_trans (h _)
      simp only [OrderHom.toFun_eq_coe]
      exact g.monotone (ih i' h_lt)


theorem lfpApprox_mono_mid : Monotone (lfpApprox f) := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    ⊢ Monotone (OrdinalApprox.lfpApprox f)
  -/
  intro x₁ x₂ h a
  induction a using Ordinal.induction with
  | h i ih =>
    rw [lfpApprox, lfpApprox]
    apply sSup_le
    simp only [exists_prop, Set.union_singleton, Set.mem_insert_iff, Set.mem_setOf_eq, sSup_insert,
      forall_eq_or_imp, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    constructor
    · exact le_sup_of_le_left h
    · intro i' h_i'
      apply le_sup_of_le_right
      apply le_sSup_of_le
      · use i'
      · exact f.monotone (ih i' h_i')


/-- The approximations of the least fixed point stabilize at a fixed point of `f` -/
theorem lfpApprox_eq_of_mem_fixedPoints {a b : Ordinal} (h_init : x ≤ f x) (h_ab : a ≤ b)
    (h : lfpApprox f x a ∈ fixedPoints f) : lfpApprox f x b = lfpApprox f x a := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b : Ordinal.{u}
    h_init : LE.le x (f x)
    h_ab : LE.le a b
    h : Membership.mem (Function.fixedPoints ⇑f) (OrdinalApprox.lfpApprox f x a)
    ⊢ Eq (OrdinalApprox.lfpApprox f x b) (OrdinalApprox.lfpApprox f x a)
  -/
  rw [mem_fixedPoints_iff] at h
  induction b using Ordinal.induction with | h b IH =>
  apply le_antisymm
  · conv => left; rw [lfpApprox]
    apply sSup_le
    simp only [exists_prop, Set.union_singleton, Set.mem_insert_iff, Set.mem_setOf_eq,
      forall_eq_or_imp, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    apply And.intro (le_lfpApprox f x)
    intro a' ha'b
    by_cases haa : a' < a
    · rw [← lfpApprox_add_one f x h_init]
      apply lfpApprox_monotone
      simp only [Ordinal.add_one_eq_succ, succ_le_iff]
      exact haa
    · rw [IH a' ha'b (le_of_not_lt haa), h]
  · exact lfpApprox_monotone f x h_ab


/-- There are distinct indices smaller than the successor of the domain's cardinality
yielding the same value -/
theorem exists_lfpApprox_eq_lfpApprox : ∃ a < ord <| succ #α, ∃ b < ord <| succ #α,
    a ≠ b ∧ lfpApprox f x a = lfpApprox f x b := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    ⊢ Exists fun a => And (LT.lt a (Order.succ (Cardinal.mk α)).ord) (Exists fun b …
  -/
  have h_ninj := not_injective_limitation_set <| lfpApprox f x
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h_ninj : Not (Set.InjOn (OrdinalApprox.lfpApprox f x) (Set.Iio (SuccOrder.succ …
    ⊢ Exists fun a => And (LT.lt a (Order.succ (Cardinal.mk α)).ord) (Exists fun b …
  -/
  rw [Set.injOn_iff_injective, Function.not_injective_iff] at h_ninj
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h_ninj : Exists fun a => Exists fun b => And (Eq ((Set.Iio (SuccOrder.succ (Ca …
    ⊢ Exists fun a => And (LT.lt a (Order.succ (Cardinal.mk α)).ord) (Exists fun b …
  -/
  let ⟨a, b, h_fab, h_nab⟩ := h_ninj
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h_ninj : Exists fun a => Exists fun b => And (Eq ((Set.Iio (SuccOrder.succ (Ca …
    a b : ↑(Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
    h_fab : Eq ((Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord).restrict (OrdinalAp …
    h_nab : Ne a b
    ⊢ Exists fun a => And (LT.lt a (Order.succ (Cardinal.mk α)).ord) (Exists fun b …
  -/
  use a.val; apply And.intro a.prop
  /-
    case h
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h_ninj : Exists fun a => Exists fun b => And (Eq ((Set.Iio (SuccOrder.succ (Ca …
    a b : ↑(Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
    h_fab : Eq ((Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord).restrict (OrdinalAp …
    h_nab : Ne a b
    ⊢ Exists fun b => And (LT.lt b (Order.succ (Cardinal.mk α)).ord) (And (Ne (↑a) …
  -/
  use b.val; apply And.intro b.prop
  /-
    case h
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h_ninj : Exists fun a => Exists fun b => And (Eq ((Set.Iio (SuccOrder.succ (Ca …
    a b : ↑(Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
    h_fab : Eq ((Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord).restrict (OrdinalAp …
    h_nab : Ne a b
    ⊢ And (Ne ↑a ↑b) (Eq (OrdinalApprox.lfpApprox f x ↑a) (OrdinalApprox.lfpApprox …
  -/
  apply And.intro
    /-
      case h.left
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h_ninj : Exists fun a => Exists fun b => And (Eq ((Set.Iio (SuccOrder.succ (Ca …
      a b : ↑(Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
      h_fab : Eq ((Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord).restrict (OrdinalAp …
      h_nab : Ne a b
      ⊢ Ne ↑a ↑b
    -/
  · intro h_eq; rw [Subtype.coe_inj] at h_eq; exact h_nab h_eq
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case h.right
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      h_ninj : Exists fun a => Exists fun b => And (Eq ((Set.Iio (SuccOrder.succ (Ca …
      a b : ↑(Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord)
      h_fab : Eq ((Set.Iio (SuccOrder.succ (Cardinal.mk α)).ord).restrict (OrdinalAp …
      h_nab : Ne a b
      ⊢ Eq (OrdinalApprox.lfpApprox f x ↑a) (OrdinalApprox.lfpApprox f x ↑b)
    -/
  · exact h_fab
    /-
      🎉 no goals
    -/


/-- If the sequence of ordinal-indexed approximations takes a value twice,
then it actually stabilised at that value. -/
lemma lfpApprox_mem_fixedPoints_of_eq {a b c : Ordinal}
    (h_init : x ≤ f x) (h_ab : a < b) (h_ac : a ≤ c) (h_fab : lfpApprox f x a = lfpApprox f x b) :
    lfpApprox f x c ∈ fixedPoints f := by
  have lfpApprox_mem_fixedPoint :
      lfpApprox f x a ∈ fixedPoints f := by
    rw [mem_fixedPoints_iff, ← lfpApprox_add_one f x h_init]
    exact Monotone.eq_of_le_of_le (lfpApprox_monotone f x)
      h_fab (SuccOrder.le_succ a) (SuccOrder.succ_le_of_lt h_ab)
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    a b c : Ordinal.{u}
    h_init : LE.le x (f x)
    h_ab : LT.lt a b
    h_ac : LE.le a c
    h_fab : Eq (OrdinalApprox.lfpApprox f x a) (OrdinalApprox.lfpApprox f x b)
    lfpApprox_mem_fixedPoint : Membership.mem (Function.fixedPoints ⇑f) (OrdinalAp …
    ⊢ Membership.mem (Function.fixedPoints ⇑f) (OrdinalApprox.lfpApprox f x c)
  -/
  rw [lfpApprox_eq_of_mem_fixedPoints f x h_init]
    /-
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      a b c : Ordinal.{u}
      h_init : LE.le x (f x)
      h_ab : LT.lt a b
      h_ac : LE.le a c
      h_fab : Eq (OrdinalApprox.lfpApprox f x a) (OrdinalApprox.lfpApprox f x b)
      lfpApprox_mem_fixedPoint : Membership.mem (Function.fixedPoints ⇑f) (OrdinalAp …
      ⊢ Membership.mem (Function.fixedPoints ⇑f) (OrdinalApprox.lfpApprox f x ?m.142 …
    -/
  · exact lfpApprox_mem_fixedPoint
    /-
      🎉 no goals
    -/
    /-
      case h_ab
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      a b c : Ordinal.{u}
      h_init : LE.le x (f x)
      h_ab : LT.lt a b
      h_ac : LE.le a c
      h_fab : Eq (OrdinalApprox.lfpApprox f x a) (OrdinalApprox.lfpApprox f x b)
      lfpApprox_mem_fixedPoint : Membership.mem (Function.fixedPoints ⇑f) (OrdinalAp …
      ⊢ LE.le a c
    -/
  · exact h_ac
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      x : α
      a b c : Ordinal.{u}
      h_init : LE.le x (f x)
      h_ab : LT.lt a b
      h_ac : LE.le a c
      h_fab : Eq (OrdinalApprox.lfpApprox f x a) (OrdinalApprox.lfpApprox f x b)
      lfpApprox_mem_fixedPoint : Membership.mem (Function.fixedPoints ⇑f) (OrdinalAp …
      ⊢ Membership.mem (Function.fixedPoints ⇑f) (OrdinalApprox.lfpApprox f x a)
    -/
  · exact lfpApprox_mem_fixedPoint
    /-
      🎉 no goals
    -/


/-- The approximation at the index of the successor of the domain's cardinality is a fixed point -/
theorem lfpApprox_ord_mem_fixedPoint (h_init : x ≤ f x) :
    lfpApprox f x (ord <| succ #α) ∈ fixedPoints f := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    x : α
    h_init : LE.le x (f x)
    ⊢ Membership.mem (Function.fixedPoints ⇑f) (OrdinalApprox.lfpApprox f x (Order …
  -/
  let ⟨a, h_a, b, h_b, h_nab, h_fab⟩ := exists_lfpApprox_eq_lfpApprox f x
  cases le_total a b with
  | inl h_ab =>
    exact lfpApprox_mem_fixedPoints_of_eq f x h_init
      (h_nab.lt_of_le h_ab) (le_of_lt h_a) h_fab
  | inr h_ba =>
    exact lfpApprox_mem_fixedPoints_of_eq f x h_init
      (h_nab.symm.lt_of_le h_ba) (le_of_lt h_b) (h_fab.symm)


/-- Every value of the approximation is less or equal than every fixed point of `f`
greater or equal than the initial value -/
theorem lfpApprox_le_of_mem_fixedPoints {a : α}
    (h_a : a ∈ fixedPoints f) (h_le_init : x ≤ a) (i : Ordinal) : lfpApprox f x i ≤ a := by
  induction i using Ordinal.induction with
  | h i IH =>
    rw [lfpApprox]
    apply sSup_le
    simp only [exists_prop]
    intro y h_y
    simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_singleton_iff] at h_y
    cases h_y with
    | inl h_y =>
      let ⟨j, h_j_lt, h_j⟩ := h_y
      rw [← h_j, ← h_a]
      exact f.monotone' (IH j h_j_lt)
    | inr h_y =>
      rw [h_y]
      exact h_le_init


/-- The approximation sequence converges at the successor of the domain's cardinality
to the least fixed point if starting from `⊥` -/
theorem lfpApprox_ord_eq_lfp : lfpApprox f ⊥ (ord <| succ #α) = f.lfp := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    ⊢ Eq (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardinal.mk α)).ord) (Ord …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      ⊢ LE.le (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardinal.mk α)).ord) ( …
    -/
  · have h_lfp : ∃ y : fixedPoints f, f.lfp = y := by use ⊥; exact rfl
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      h_lfp : Exists fun y => Eq (OrderHom.lfp f) ↑y
      ⊢ LE.le (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardinal.mk α)).ord) ( …
    -/
    let ⟨y, h_y⟩ := h_lfp; rw [h_y]
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      h_lfp : Exists fun y => Eq (OrderHom.lfp f) ↑y
      y : ↑(Function.fixedPoints ⇑f)
      h_y : Eq (OrderHom.lfp f) ↑y
      ⊢ LE.le (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardinal.mk α)).ord) ↑y
    -/
    exact lfpApprox_le_of_mem_fixedPoints f ⊥ y.2 bot_le (ord <| succ #α)
    /-
      🎉 no goals
    -/
  · have h_fix : ∃ y : fixedPoints f, lfpApprox f ⊥ (ord <| succ #α) = y := by
      simpa only [Subtype.exists, mem_fixedPoints, exists_prop, exists_eq_right'] using
        lfpApprox_ord_mem_fixedPoint f ⊥ bot_le
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      h_fix : Exists fun y => Eq (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Car …
      ⊢ LE.le (OrderHom.lfp f) (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardi …
    -/
    let ⟨x, h_x⟩ := h_fix; rw [h_x]
    /-
      case a
      α : Type u
      inst✝ : CompleteLattice α
      f : OrderHom α α
      h_fix : Exists fun y => Eq (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Car …
      x : ↑(Function.fixedPoints ⇑f)
      h_x : Eq (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardinal.mk α)).ord) ↑x
      ⊢ LE.le (OrderHom.lfp f) ↑x
    -/
    exact lfp_le_fixed f x.prop
    /-
      🎉 no goals
    -/


/-- Some approximation of the least fixed point starting from `⊥` is the least fixed point. -/
theorem lfp_mem_range_lfpApprox : f.lfp ∈ Set.range (lfpApprox f ⊥) := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    ⊢ Membership.mem (Set.range (OrdinalApprox.lfpApprox f Bot.bot)) (OrderHom.lfp …
  -/
  use ord <| succ #α
  /-
    case h
    α : Type u
    inst✝ : CompleteLattice α
    f : OrderHom α α
    ⊢ Eq (OrdinalApprox.lfpApprox f Bot.bot (Order.succ (Cardinal.mk α)).ord) (Ord …
  -/
  exact lfpApprox_ord_eq_lfp f
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
/-- The ordinal-indexed sequence approximating the greatest fixed point greater than
an initial value `x`. It is defined in such a way that we have `gfpApprox 0 x = x` and
`gfpApprox a x = ⨅ b < a, f (lfpApprox b x)`. -/
def gfpApprox (a : Ordinal.{u}) : α :=
  sInf ({ f (gfpApprox b) | (b : Ordinal) (h : b < a) } ∪ {x})
termination_by a
/-
  a b : Ordinal.{u}
  h : LT.lt b a
  ⊢ LT.lt b a
-/
decreasing_by exact h
/-
  🎉 no goals
-/

-- By unsealing these recursive definitions we can relate them
-- by definitional equality

theorem gfpApprox_antitone : Antitone (gfpApprox f x) :=
  lfpApprox_monotone f.dual x


theorem gfpApprox_le {a : Ordinal} : gfpApprox f x a ≤ x :=
  le_lfpApprox f.dual x


theorem gfpApprox_add_one (h : f x ≤ x) (a : Ordinal) :
    gfpApprox f x (a+1) = f (gfpApprox f x a) :=
  lfpApprox_add_one f.dual x h a


theorem gfpApprox_mono_left : Monotone (gfpApprox : (α →o α) → _) := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    ⊢ Monotone OrdinalApprox.gfpApprox
  -/
  intro f g h
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f g : OrderHom α α
    h : LE.le f g
    ⊢ LE.le (OrdinalApprox.gfpApprox f) (OrdinalApprox.gfpApprox g)
  -/
  have : g.dual ≤ f.dual := h
  /-
    α : Type u
    inst✝ : CompleteLattice α
    f g : OrderHom α α
    h : LE.le f g
    this : LE.le (OrderHom.dual g) (OrderHom.dual f)
    ⊢ LE.le (OrdinalApprox.gfpApprox f) (OrdinalApprox.gfpApprox g)
  -/
  exact lfpApprox_mono_left this
  /-
    🎉 no goals
  -/


theorem gfpApprox_mono_mid : Monotone (gfpApprox f) :=
  fun _ _ h => lfpApprox_mono_mid f.dual h


/-- The approximations of the greatest fixed point stabilize at a fixed point of `f` -/
theorem gfpApprox_eq_of_mem_fixedPoints {a b : Ordinal} (h_init : f x ≤ x) (h_ab : a ≤ b)
    (h : gfpApprox f x a ∈ fixedPoints f) : gfpApprox f x b = gfpApprox f x a :=
  lfpApprox_eq_of_mem_fixedPoints f.dual x h_init h_ab h


/-- There are distinct indices smaller than the successor of the domain's cardinality
yielding the same value -/
theorem exists_gfpApprox_eq_gfpApprox : ∃ a < ord <| succ #α, ∃ b < ord <| succ #α,
    a ≠ b ∧ gfpApprox f x a = gfpApprox f x b :=
  exists_lfpApprox_eq_lfpApprox f.dual x


/-- The approximation at the index of the successor of the domain's cardinality is a fixed point -/
lemma gfpApprox_ord_mem_fixedPoint (h_init : f x ≤ x) :
    gfpApprox f x (ord <| succ #α) ∈ fixedPoints f :=
  lfpApprox_ord_mem_fixedPoint f.dual x h_init


/-- Every value of the approximation is greater or equal than every fixed point of `f`
less or equal than the initial value -/
lemma le_gfpApprox_of_mem_fixedPoints {a : α}
    (h_a : a ∈ fixedPoints f) (h_le_init : a ≤ x) (i : Ordinal) : a ≤ gfpApprox f x i :=
  lfpApprox_le_of_mem_fixedPoints f.dual x h_a h_le_init i


/-- The approximation sequence converges at the successor of the domain's cardinality
to the greatest fixed point if starting from `⊥` -/
theorem gfpApprox_ord_eq_gfp : gfpApprox f ⊤ (ord <| succ #α) = f.gfp :=
  lfpApprox_ord_eq_lfp f.dual


/-- Some approximation of the least fixed point starting from `⊤` is the greatest fixed point. -/
theorem gfp_mem_range_gfpApprox : f.gfp ∈ Set.range (gfpApprox f ⊤) :=
  lfp_mem_range_lfpApprox f.dual


