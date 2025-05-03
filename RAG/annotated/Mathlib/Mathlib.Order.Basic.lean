theorem le_trans' : b ≤ c → a ≤ b → a ≤ c :=
  flip le_trans


theorem lt_trans' : b < c → a < b → a < c :=
  flip lt_trans


theorem lt_of_le_of_lt' : b ≤ c → a < b → a < c :=
  flip lt_of_lt_of_le


theorem lt_of_lt_of_le' : b < c → a ≤ b → a < c :=
  flip lt_of_le_of_lt


theorem ge_antisymm : a ≤ b → b ≤ a → b = a :=
  flip le_antisymm


theorem lt_of_le_of_ne' : a ≤ b → b ≠ a → a < b := fun h₁ h₂ ↦ lt_of_le_of_ne h₁ h₂.symm


theorem Ne.lt_of_le : a ≠ b → a ≤ b → a < b :=
  flip lt_of_le_of_ne


theorem Ne.lt_of_le' : b ≠ a → a ≤ b → a < b :=
  flip lt_of_le_of_ne'


attribute [ext] LE


alias LE.le.trans := le_trans


alias LE.le.trans' := le_trans'


alias LE.le.trans_lt := lt_of_le_of_lt


alias LE.le.trans_lt' := lt_of_le_of_lt'


alias LE.le.antisymm := le_antisymm


alias LE.le.antisymm' := ge_antisymm


alias LE.le.lt_of_ne := lt_of_le_of_ne


alias LE.le.lt_of_ne' := lt_of_le_of_ne'


alias LE.le.lt_of_not_le := lt_of_le_not_le


alias LE.le.lt_or_eq := lt_or_eq_of_le


alias LE.le.lt_or_eq_dec := Decidable.lt_or_eq_of_le


alias LT.lt.le := le_of_lt


alias LT.lt.trans := lt_trans


alias LT.lt.trans' := lt_trans'


alias LT.lt.trans_le := lt_of_lt_of_le


alias LT.lt.trans_le' := lt_of_lt_of_le'


alias LT.lt.ne := ne_of_lt


alias LT.lt.asymm := lt_asymm


alias LT.lt.not_lt := lt_asymm


alias Eq.le := le_of_eq

-- Porting note: no `decidable_classical` linter
-- attribute [nolint decidable_classical] LE.le.lt_or_eq_dec


@[simp]
theorem lt_self_iff_false (x : α) : x < x ↔ False :=
  ⟨lt_irrefl x, False.elim⟩


theorem le_of_le_of_eq' : b ≤ c → a = b → a ≤ c :=
  flip le_of_eq_of_le


theorem le_of_eq_of_le' : b = c → a ≤ b → a ≤ c :=
  flip le_of_le_of_eq


theorem lt_of_lt_of_eq' : b < c → a = b → a < c :=
  flip lt_of_eq_of_lt


theorem lt_of_eq_of_lt' : b = c → a < b → a < c :=
  flip lt_of_lt_of_eq


alias LE.le.trans_eq := le_of_le_of_eq


alias LE.le.trans_eq' := le_of_le_of_eq'


alias LT.lt.trans_eq := lt_of_lt_of_eq


alias LT.lt.trans_eq' := lt_of_lt_of_eq'


alias Eq.trans_le := le_of_eq_of_le


alias Eq.trans_ge := le_of_eq_of_le'


alias Eq.trans_lt := lt_of_eq_of_lt


alias Eq.trans_gt := lt_of_eq_of_lt'


/-- If `x = y` then `y ≤ x`. Note: this lemma uses `y ≤ x` instead of `x ≥ y`, because `le` is used
almost exclusively in mathlib. -/
protected theorem ge (h : x = y) : y ≤ x :=
  h.symm.le


theorem not_lt (h : x = y) : ¬x < y := fun h' ↦ h'.ne h


theorem not_gt (h : x = y) : ¬y < x :=
  h.symm.not_lt


@[simp] lemma le_of_subsingleton [Subsingleton α] : a ≤ b := (Subsingleton.elim a b).le

-- Making this a @[simp] lemma causes confluences problems downstream.

lemma not_lt_of_subsingleton [Subsingleton α] : ¬a < b := (Subsingleton.elim a b).not_lt


protected theorem ge [LE α] {x y : α} (h : x ≤ y) : y ≥ x :=
  h


theorem lt_iff_ne (h : a ≤ b) : a < b ↔ a ≠ b :=
  ⟨fun h ↦ h.ne, h.lt_of_ne⟩


theorem gt_iff_ne (h : a ≤ b) : a < b ↔ b ≠ a :=
  ⟨fun h ↦ h.ne.symm, h.lt_of_ne'⟩


theorem not_lt_iff_eq (h : a ≤ b) : ¬a < b ↔ a = b :=
  h.lt_iff_ne.not_left


theorem not_gt_iff_eq (h : a ≤ b) : ¬a < b ↔ b = a :=
  h.gt_iff_ne.not_left


theorem le_iff_eq (h : a ≤ b) : b ≤ a ↔ b = a :=
  ⟨fun h' ↦ h'.antisymm h, Eq.le⟩


theorem ge_iff_eq (h : a ≤ b) : b ≤ a ↔ a = b :=
  ⟨h.antisymm, Eq.ge⟩


theorem lt_or_le [LinearOrder α] {a b : α} (h : a ≤ b) (c : α) : a < c ∨ c ≤ b :=
  ((lt_or_ge a c).imp id) fun hc ↦ le_trans hc h


theorem le_or_lt [LinearOrder α] {a b : α} (h : a ≤ b) (c : α) : a ≤ c ∨ c < b :=
  ((le_or_gt a c).imp id) fun hc ↦ lt_of_lt_of_le hc h


theorem le_or_le [LinearOrder α] {a b : α} (h : a ≤ b) (c : α) : a ≤ c ∨ c ≤ b :=
  (h.le_or_lt c).elim Or.inl fun h ↦ Or.inr <| le_of_lt h


protected theorem gt [LT α] {x y : α} (h : x < y) : y > x :=
  h


protected theorem false [Preorder α] {x : α} : x < x → False :=
  lt_irrefl x


theorem ne' [Preorder α] {x y : α} (h : x < y) : y ≠ x :=
  h.ne.symm


theorem lt_or_lt [LinearOrder α] {x y : α} (h : x < y) (z : α) : x < z ∨ z < y :=
  (lt_or_ge z y).elim Or.inr fun hz ↦ Or.inl <| h.trans_le hz


protected theorem GE.ge.le [LE α] {x y : α} (h : x ≥ y) : y ≤ x :=
  h

-- see Note [nolint_ge]
-- Porting note: linter not found @[nolint ge_or_gt]

protected theorem GT.gt.lt [LT α] {x y : α} (h : x > y) : y < x :=
  h

-- see Note [nolint_ge]
-- Porting note: linter not found @[nolint ge_or_gt]

theorem ge_of_eq [Preorder α] {a b : α} (h : a = b) : a ≥ b :=
  h.ge


theorem ne_of_not_le [Preorder α] {a b : α} (h : ¬a ≤ b) : a ≠ b := fun hab ↦ h (le_of_eq hab)


protected theorem Decidable.le_iff_eq_or_lt [DecidableRel (α := α) (· ≤ ·)] :
    a ≤ b ↔ a = b ∨ a < b :=
  Decidable.le_iff_lt_or_eq.trans or_comm


theorem le_iff_eq_or_lt : a ≤ b ↔ a = b ∨ a < b := le_iff_lt_or_eq.trans or_comm


theorem lt_iff_le_and_ne : a < b ↔ a ≤ b ∧ a ≠ b :=
  ⟨fun h ↦ ⟨le_of_lt h, ne_of_lt h⟩, fun ⟨h1, h2⟩ ↦ h1.lt_of_ne h2⟩


                                                                /-
                                                                  α : Type u_2
                                                                  inst✝ : PartialOrder α
                                                                  a b : α
                                                                  hab : LE.le a b
                                                                  ⊢ Iff (Eq a b) (Not (LT.lt a b))
                                                                -/
lemma eq_iff_not_lt_of_le (hab : a ≤ b) : a = b ↔ ¬ a < b := by simp [hab, lt_iff_le_and_ne]
                                                                /-
                                                                  🎉 no goals
                                                                -/


alias LE.le.eq_iff_not_lt := eq_iff_not_lt_of_le

-- See Note [decidable namespace]

protected theorem Decidable.eq_iff_le_not_lt [DecidableRel (α := α) (· ≤ ·)] :
    a = b ↔ a ≤ b ∧ ¬a < b :=
  ⟨fun h ↦ ⟨h.le, h ▸ lt_irrefl _⟩, fun ⟨h₁, h₂⟩ ↦
    h₁.antisymm <| Decidable.byContradiction fun h₃ ↦ h₂ (h₁.lt_of_not_le h₃)⟩


theorem eq_iff_le_not_lt : a = b ↔ a ≤ b ∧ ¬a < b :=
  haveI := Classical.dec
  Decidable.eq_iff_le_not_lt


theorem eq_or_lt_of_le (h : a ≤ b) : a = b ∨ a < b := h.lt_or_eq.symm

theorem eq_or_gt_of_le (h : a ≤ b) : b = a ∨ a < b := h.lt_or_eq.symm.imp Eq.symm id

theorem gt_or_eq_of_le (h : a ≤ b) : a < b ∨ b = a := (eq_or_gt_of_le h).symm


alias LE.le.eq_or_lt_dec := Decidable.eq_or_lt_of_le

alias LE.le.eq_or_lt := eq_or_lt_of_le

alias LE.le.eq_or_gt := eq_or_gt_of_le

alias LE.le.gt_or_eq := gt_or_eq_of_le

-- Porting note: no `decidable_classical` linter
-- attribute [nolint decidable_classical] LE.le.eq_or_lt_dec


theorem eq_of_le_of_not_lt (hab : a ≤ b) (hba : ¬a < b) : a = b := hab.eq_or_lt.resolve_right hba

theorem eq_of_ge_of_not_gt (hab : a ≤ b) (hba : ¬a < b) : b = a := (eq_of_le_of_not_lt hab hba).symm


alias LE.le.eq_of_not_lt := eq_of_le_of_not_lt

alias LE.le.eq_of_not_gt := eq_of_ge_of_not_gt


theorem Ne.le_iff_lt (h : a ≠ b) : a ≤ b ↔ a < b := ⟨fun h' ↦ lt_of_le_of_ne h' h, fun h ↦ h.le⟩


theorem Ne.not_le_or_not_le (h : a ≠ b) : ¬a ≤ b ∨ ¬b ≤ a := not_and_or.1 <| le_antisymm_iff.not.1 h

-- See Note [decidable namespace]

protected theorem Decidable.ne_iff_lt_iff_le [DecidableEq α] : (a ≠ b ↔ a < b) ↔ a ≤ b :=
  ⟨fun h ↦ Decidable.byCases le_of_eq (le_of_lt ∘ h.mp), fun h ↦ ⟨lt_of_le_of_ne h, ne_of_lt⟩⟩


@[simp]
theorem ne_iff_lt_iff_le : (a ≠ b ↔ a < b) ↔ a ≤ b :=
  haveI := Classical.dec
  Decidable.ne_iff_lt_iff_le


theorem min_def' [LinearOrder α] (a b : α) : min a b = if b ≤ a then b else a := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (Min.min a b) (ite (LE.le b a) b a)
  -/
  rw [min_def]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (ite (LE.le a b) a b) (ite (LE.le b a) b a)
  -/
  rcases lt_trichotomy a b with (lt | eq | gt)
    /-
      case inl
      α : Type u_2
      inst✝ : LinearOrder α
      a b : α
      lt : LT.lt a b
      ⊢ Eq (ite (LE.le a b) a b) (ite (LE.le b a) b a)
    -/
  · rw [if_pos lt.le, if_neg (not_le.mpr lt)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_2
      inst✝ : LinearOrder α
      a b : α
      eq : Eq a b
      ⊢ Eq (ite (LE.le a b) a b) (ite (LE.le b a) b a)
    -/
  · rw [if_pos eq.le, if_pos eq.ge, eq]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_2
      inst✝ : LinearOrder α
      a b : α
      gt : LT.lt b a
      ⊢ Eq (ite (LE.le a b) a b) (ite (LE.le b a) b a)
    -/
  · rw [if_neg (not_le.mpr gt.gt), if_pos gt.le]
    /-
      🎉 no goals
    -/

-- Variant of `min_def` with the branches reversed.
-- This is sometimes useful as it used to be the default.

theorem max_def' [LinearOrder α] (a b : α) : max a b = if b ≤ a then a else b := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (Max.max a b) (ite (LE.le b a) a b)
  -/
  rw [max_def]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (ite (LE.le a b) b a) (ite (LE.le b a) a b)
  -/
  rcases lt_trichotomy a b with (lt | eq | gt)
    /-
      case inl
      α : Type u_2
      inst✝ : LinearOrder α
      a b : α
      lt : LT.lt a b
      ⊢ Eq (ite (LE.le a b) b a) (ite (LE.le b a) a b)
    -/
  · rw [if_pos lt.le, if_neg (not_le.mpr lt)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_2
      inst✝ : LinearOrder α
      a b : α
      eq : Eq a b
      ⊢ Eq (ite (LE.le a b) b a) (ite (LE.le b a) a b)
    -/
  · rw [if_pos eq.le, if_pos eq.ge, eq]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_2
      inst✝ : LinearOrder α
      a b : α
      gt : LT.lt b a
      ⊢ Eq (ite (LE.le a b) b a) (ite (LE.le b a) a b)
    -/
  · rw [if_neg (not_le.mpr gt.gt), if_pos gt.le]
    /-
      🎉 no goals
    -/


theorem lt_of_not_le [LinearOrder α] {a b : α} (h : ¬b ≤ a) : a < b :=
  ((le_total _ _).resolve_right h).lt_of_not_le h


theorem lt_iff_not_le [LinearOrder α] {x y : α} : x < y ↔ ¬y ≤ x :=
  ⟨not_le_of_lt, lt_of_not_le⟩


theorem Ne.lt_or_lt [LinearOrder α] {x y : α} (h : x ≠ y) : x < y ∨ y < x :=
  lt_or_gt_of_ne h


/-- A version of `ne_iff_lt_or_gt` with LHS and RHS reversed. -/
@[simp]
theorem lt_or_lt_iff_ne [LinearOrder α] {x y : α} : x < y ∨ y < x ↔ x ≠ y :=
  ne_iff_lt_or_gt.symm


theorem not_lt_iff_eq_or_lt [LinearOrder α] {a b : α} : ¬a < b ↔ a = b ∨ b < a :=
  not_lt.trans <| Decidable.le_iff_eq_or_lt.trans <| or_congr eq_comm Iff.rfl


theorem exists_ge_of_linear [LinearOrder α] (a b : α) : ∃ c, a ≤ c ∧ b ≤ c :=
  match le_total a b with
  | Or.inl h => ⟨_, h, le_rfl⟩
  | Or.inr h => ⟨_, le_rfl, h⟩


lemma exists_forall_ge_and [LinearOrder α] {p q : α → Prop} :
    (∃ i, ∀ j ≥ i, p j) → (∃ i, ∀ j ≥ i, q j) → ∃ i, ∀ j ≥ i, p j ∧ q j
  | ⟨a, ha⟩, ⟨b, hb⟩ =>
    let ⟨c, hac, hbc⟩ := exists_ge_of_linear a b
    ⟨c, fun _d hcd ↦ ⟨ha _ <| hac.trans hcd, hb _ <| hbc.trans hcd⟩⟩


theorem lt_imp_lt_of_le_imp_le {β} [LinearOrder α] [Preorder β] {a b : α} {c d : β}
    (H : a ≤ b → c ≤ d) (h : d < c) : b < a :=
  lt_of_not_le fun h' ↦ (H h').not_lt h


theorem le_imp_le_iff_lt_imp_lt {β} [LinearOrder α] [LinearOrder β] {a b : α} {c d : β} :
    a ≤ b → c ≤ d ↔ d < c → b < a :=
  ⟨lt_imp_lt_of_le_imp_le, le_imp_le_of_lt_imp_lt⟩


theorem lt_iff_lt_of_le_iff_le' {β} [Preorder α] [Preorder β] {a b : α} {c d : β}
    (H : a ≤ b ↔ c ≤ d) (H' : b ≤ a ↔ d ≤ c) : b < a ↔ d < c :=
  lt_iff_le_not_le.trans <| (and_congr H' (not_congr H)).trans lt_iff_le_not_le.symm


theorem lt_iff_lt_of_le_iff_le {β} [LinearOrder α] [LinearOrder β] {a b : α} {c d : β}
    (H : a ≤ b ↔ c ≤ d) : b < a ↔ d < c :=
  not_le.symm.trans <| (not_congr H).trans <| not_le


theorem le_iff_le_iff_lt_iff_lt {β} [LinearOrder α] [LinearOrder β] {a b : α} {c d : β} :
    (a ≤ b ↔ c ≤ d) ↔ (b < a ↔ d < c) :=
  ⟨lt_iff_lt_of_le_iff_le, fun H ↦ not_lt.symm.trans <| (not_congr H).trans <| not_lt⟩


theorem eq_of_forall_le_iff [PartialOrder α] {a b : α} (H : ∀ c, c ≤ a ↔ c ≤ b) : a = b :=
  ((H _).1 le_rfl).antisymm ((H _).2 le_rfl)


theorem le_of_forall_le [Preorder α] {a b : α} (H : ∀ c, c ≤ a → c ≤ b) : a ≤ b :=
  H _ le_rfl


theorem le_of_forall_le' [Preorder α] {a b : α} (H : ∀ c, a ≤ c → b ≤ c) : b ≤ a :=
  H _ le_rfl


theorem le_of_forall_lt [LinearOrder α] {a b : α} (H : ∀ c, c < a → c < b) : a ≤ b :=
  le_of_not_lt fun h ↦ lt_irrefl _ (H _ h)


theorem forall_lt_iff_le [LinearOrder α] {a b : α} : (∀ ⦃c⦄, c < a → c < b) ↔ a ≤ b :=
  ⟨le_of_forall_lt, fun h _ hca ↦ lt_of_lt_of_le hca h⟩


theorem le_of_forall_lt' [LinearOrder α] {a b : α} (H : ∀ c, a < c → b < c) : b ≤ a :=
  le_of_not_lt fun h ↦ lt_irrefl _ (H _ h)


theorem forall_lt_iff_le' [LinearOrder α] {a b : α} : (∀ ⦃c⦄, a < c → b < c) ↔ b ≤ a :=
  ⟨le_of_forall_lt', fun h _ hac ↦ lt_of_le_of_lt h hac⟩


theorem eq_of_forall_ge_iff [PartialOrder α] {a b : α} (H : ∀ c, a ≤ c ↔ b ≤ c) : a = b :=
  ((H _).2 le_rfl).antisymm ((H _).1 le_rfl)


theorem eq_of_forall_lt_iff [LinearOrder α] {a b : α} (h : ∀ c, c < a ↔ c < b) : a = b :=
  (le_of_forall_lt fun _ ↦ (h _).1).antisymm <| le_of_forall_lt fun _ ↦ (h _).2


theorem eq_of_forall_gt_iff [LinearOrder α] {a b : α} (h : ∀ c, a < c ↔ b < c) : a = b :=
  (le_of_forall_lt' fun _ ↦ (h _).2).antisymm <| le_of_forall_lt' fun _ ↦ (h _).1


/-- A symmetric relation implies two values are equal, when it implies they're less-equal. -/
theorem rel_imp_eq_of_rel_imp_le [PartialOrder β] (r : α → α → Prop) [IsSymm α r] {f : α → β}
    (h : ∀ a b, r a b → f a ≤ f b) {a b : α} : r a b → f a = f b := fun hab ↦
  le_antisymm (h a b hab) (h b a <| symm hab)


/-- monotonicity of `≤` with respect to `→` -/
theorem le_implies_le_of_le_of_le {a b c d : α} [Preorder α] (hca : c ≤ a) (hbd : b ≤ d) :
    a ≤ b → c ≤ d :=
  fun hab ↦ (hca.trans hab).trans hbd


/-- To prove commutativity of a binary operation `○`, we only to check `a ○ b ≤ b ○ a` for all `a`,
`b`. -/
theorem commutative_of_le {f : β → β → α} (comm : ∀ a b, f a b ≤ f b a) : ∀ a b, f a b = f b a :=
  fun _ _ ↦ (comm _ _).antisymm <| comm _ _


/-- To prove associativity of a commutative binary operation `○`, we only to check
`(a ○ b) ○ c ≤ a ○ (b ○ c)` for all `a`, `b`, `c`. -/
theorem associative_of_commutative_of_le {f : α → α → α} (comm : Std.Commutative f)
    (assoc : ∀ a b c, f (f a b) c ≤ f a (f b c)) : Std.Associative f where
  assoc a b c :=
    le_antisymm (assoc _ _ _) <| by
      /-
        α : Type u_2
        inst✝ : PartialOrder α
        f : α → α → α
        comm : Std.Commutative f
        assoc : ∀ (a b c : α), LE.le (f (f a b) c) (f a (f b c))
        a b c : α
        ⊢ LE.le (f a (f b c)) (f (f a b) c)
      -/
      rw [comm.comm, comm.comm b, comm.comm _ c, comm.comm a]
      /-
        α : Type u_2
        inst✝ : PartialOrder α
        f : α → α → α
        comm : Std.Commutative f
        assoc : ∀ (a b c : α), LE.le (f (f a b) c) (f a (f b c))
        a b c : α
        ⊢ LE.le (f (f c b) a) (f c (f b a))
      -/
      exact assoc _ _ _
      /-
        🎉 no goals
      -/


@[ext]
theorem Preorder.toLE_injective : Function.Injective (@Preorder.toLE α) :=
  fun A B h ↦ match A, B with
  | { lt := A_lt, lt_iff_le_not_le := A_iff, .. },
    { lt := B_lt, lt_iff_le_not_le := B_iff, .. } => by
    /-
      α : Type u_2
      A B : Preorder α
      le✝¹ A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      A_iff : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      le✝ B_lt : α → α → Prop
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      B_iff : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      h : Eq Preorder.toLE Preorder.toLE
      ⊢ Eq (Preorder.mk le_refl✝¹ le_trans✝¹ A_iff) (Preorder.mk le_refl✝ le_trans✝  …
    -/
    cases h
    have : A_lt = B_lt := by
      funext a b
      show (LT.mk A_lt).lt a b = (LT.mk B_lt).lt a b
      rw [A_iff, B_iff]
    /-
      case refl
      α : Type u_2
      A B : Preorder α
      le✝ A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      A_iff : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      B_lt : α → α → Prop
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      B_iff : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      this : Eq A_lt B_lt
      ⊢ Eq (Preorder.mk le_refl✝¹ le_trans✝¹ A_iff) (Preorder.mk le_refl✝ le_trans✝  …
    -/
    cases this
    /-
      case refl.refl
      α : Type u_2
      A B : Preorder α
      le✝ A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      A_iff : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      B_iff : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
      ⊢ Eq (Preorder.mk le_refl✝¹ le_trans✝¹ A_iff) (Preorder.mk le_refl✝ le_trans✝  …
    -/
    congr
    /-
      🎉 no goals
    -/


@[ext]
theorem PartialOrder.toPreorder_injective :
    Function.Injective (@PartialOrder.toPreorder α) := fun A B h ↦ by
  /-
    α : Type u_2
    A B : PartialOrder α
    h : Eq PartialOrder.toPreorder PartialOrder.toPreorder
    ⊢ Eq A B
  -/
  cases A
  /-
    case mk
    α : Type u_2
    B : PartialOrder α
    toPreorder✝ : Preorder α
    le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
    h : Eq PartialOrder.toPreorder PartialOrder.toPreorder
    ⊢ Eq (PartialOrder.mk le_antisymm✝) B
  -/
  cases B
  /-
    case mk.mk
    α : Type u_2
    toPreorder✝¹ : Preorder α
    le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
    toPreorder✝ : Preorder α
    le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
    h : Eq PartialOrder.toPreorder PartialOrder.toPreorder
    ⊢ Eq (PartialOrder.mk le_antisymm✝¹) (PartialOrder.mk le_antisymm✝)
  -/
  cases h
  /-
    case mk.mk.refl
    α : Type u_2
    toPreorder✝ : Preorder α
    le_antisymm✝¹ le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
    ⊢ Eq (PartialOrder.mk le_antisymm✝¹) (PartialOrder.mk le_antisymm✝)
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext]
theorem LinearOrder.toPartialOrder_injective :
    Function.Injective (@LinearOrder.toPartialOrder α) :=
  fun A B h ↦ match A, B with
  | { le := A_le, lt := A_lt,
      decidableLE := A_decidableLE, decidableEq := A_decidableEq, decidableLT := A_decidableLT
      min := A_min, max := A_max, min_def := A_min_def, max_def := A_max_def,
      compare := A_compare, compare_eq_compareOfLessAndEq := A_compare_canonical, .. },
    { le := B_le, lt := B_lt,
      decidableLE := B_decidableLE, decidableEq := B_decidableEq, decidableLT := B_decidableLT
      min := B_min, max := B_max, min_def := B_min_def, max_def := B_max_def,
      compare := B_compare, compare_eq_compareOfLessAndEq := B_compare_canonical, .. } => by
    /-
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_le B_lt : α → α → Prop
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      B_min B_max : α → α → α
      B_compare : α → α → Ordering
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      B_decidableEq : DecidableEq α
      B_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      h : Eq LinearOrder.toPartialOrder LinearOrder.toPartialOrder
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    cases h
    /-
      case refl
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min B_max : α → α → α
      B_compare : α → α → Ordering
      B_decidableEq : DecidableEq α
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      B_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    obtain rfl : A_decidableLE = B_decidableLE := Subsingleton.elim _ _
    /-
      case refl
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min B_max : α → α → α
      B_compare : α → α → Ordering
      B_decidableEq : DecidableEq α
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    obtain rfl : A_decidableEq = B_decidableEq := Subsingleton.elim _ _
    /-
      case refl
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min B_max : α → α → α
      B_compare : α → α → Ordering
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    obtain rfl : A_decidableLT = B_decidableLT := Subsingleton.elim _ _
    have : A_min = B_min := by
      funext a b
      exact (A_min_def _ _).trans (B_min_def _ _).symm
    /-
      case refl
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min B_max : α → α → α
      B_compare : α → α → Ordering
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      this : Eq A_min B_min
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    cases this
    have : A_max = B_max := by
      funext a b
      exact (A_max_def _ _).trans (B_max_def _ _).symm
    /-
      case refl.refl
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_max : α → α → α
      B_compare : α → α → Ordering
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      this : Eq A_max B_max
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    cases this
    have : A_compare = B_compare := by
      funext a b
      exact (A_compare_canonical _ _).trans (B_compare_canonical _ _).symm
    /-
      case refl.refl.refl
      α : Type u_2
      A B : LinearOrder α
      A_le A_lt : α → α → Prop
      le_refl✝¹ : ∀ (a : α), LE.le a a
      le_trans✝¹ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝¹ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
      le_antisymm✝¹ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      A_min A_max : α → α → α
      A_compare : α → α → Ordering
      le_total✝¹ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      A_decidableLE : DecidableRel fun x1 x2 => LE.le x1 x2
      A_decidableEq : DecidableEq α
      A_decidableLT : DecidableRel fun x1 x2 => LT.lt x1 x2
      A_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      A_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      A_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_compare : α → α → Ordering
      le_refl✝ : ∀ (a : α), LE.le a a
      le_trans✝ : ∀ (a b c : α), LE.le a b → LE.le b c → LE.le a c
      lt_iff_le_not_le✝ : ∀ (a b : α), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le  …
      le_antisymm✝ : ∀ (a b : α), LE.le a b → LE.le b a → Eq a b
      le_total✝ : ∀ (a b : α), Or (LE.le a b) (LE.le b a)
      B_compare_canonical : ∀ (a b : α), Eq (Ord.compare a b) (compareOfLessAndEq a b)
      B_min_def : ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      B_max_def : ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      this : Eq A_compare B_compare
      ⊢ Eq (LinearOrder.mk le_total✝¹ A_decidableLE A_decidableEq A_decidableLT A_mi …
    -/
    congr
    /-
      🎉 no goals
    -/


theorem Preorder.ext {A B : Preorder α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y) : A = B := by
  /-
    α : Type u_2
    A B : Preorder α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq A B
  -/
  ext x y
  /-
    case a.le.h.h.a
    α : Type u_2
    A B : Preorder α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    x y : α
    ⊢ Iff (LE.le x y) (LE.le x y)
  -/
  exact H x y
  /-
    🎉 no goals
  -/


theorem PartialOrder.ext {A B : PartialOrder α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y) : A = B := by
  /-
    α : Type u_2
    A B : PartialOrder α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq A B
  -/
  ext x y
  /-
    case a.a.le.h.h.a
    α : Type u_2
    A B : PartialOrder α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    x y : α
    ⊢ Iff (LE.le x y) (LE.le x y)
  -/
  exact H x y
  /-
    🎉 no goals
  -/


theorem PartialOrder.ext_lt {A B : PartialOrder α}
    (H : ∀ x y : α, (haveI := A; x < y) ↔ x < y) : A = B := by
  /-
    α : Type u_2
    A B : PartialOrder α
    H : ∀ (x y : α), Iff (LT.lt x y) (LT.lt x y)
    ⊢ Eq A B
  -/
  ext x y
  /-
    case a.a.le.h.h.a
    α : Type u_2
    A B : PartialOrder α
    H : ∀ (x y : α), Iff (LT.lt x y) (LT.lt x y)
    x y : α
    ⊢ Iff (LE.le x y) (LE.le x y)
  -/
  rw [le_iff_lt_or_eq, @le_iff_lt_or_eq _ A, H]
  /-
    🎉 no goals
  -/


theorem LinearOrder.ext {A B : LinearOrder α}
    (H : ∀ x y : α, (haveI := A; x ≤ y) ↔ x ≤ y) : A = B := by
  /-
    α : Type u_2
    A B : LinearOrder α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    ⊢ Eq A B
  -/
  ext x y
  /-
    case a.a.a.le.h.h.a
    α : Type u_2
    A B : LinearOrder α
    H : ∀ (x y : α), Iff (LE.le x y) (LE.le x y)
    x y : α
    ⊢ Iff (LE.le x y) (LE.le x y)
  -/
  exact H x y
  /-
    🎉 no goals
  -/


theorem LinearOrder.ext_lt {A B : LinearOrder α}
    (H : ∀ x y : α, (haveI := A; x < y) ↔ x < y) : A = B :=
  LinearOrder.toPartialOrder_injective (PartialOrder.ext_lt H)


/-- Given a relation `R` on `β` and a function `f : α → β`, the preimage relation on `α` is defined
by `x ≤ y ↔ f x ≤ f y`. It is the unique relation on `α` making `f` a `RelEmbedding` (assuming `f`
is injective). -/
@[simp]
def Order.Preimage (f : α → β) (s : β → β → Prop) (x y : α) : Prop :=
  s (f x) (f y)


@[inherit_doc]
infixl:80 " ⁻¹'o " => Order.Preimage


/-- The preimage of a decidable order is decidable. -/
instance Order.Preimage.decidable (f : α → β) (s : β → β → Prop) [H : DecidableRel s] :
    DecidableRel (f ⁻¹'o s) := fun _ _ ↦ H _ _


@[simp]
lemma ltByCases_lt (h : x < y) {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P} :
    ltByCases x y h₁ h₂ h₃ = h₁ h := dif_pos h


@[simp]
lemma ltByCases_gt (h : y < x) {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P} :
    ltByCases x y h₁ h₂ h₃ = h₃ h := (dif_neg h.not_lt).trans (dif_pos h)


@[simp]
lemma ltByCases_eq (h : x = y) {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P} :
    ltByCases x y h₁ h₂ h₃ = h₂ h := (dif_neg h.not_lt).trans (dif_neg h.not_gt)


lemma ltByCases_not_lt (h : ¬ x < y) {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P}
    (p : ¬ y < x → x = y := fun h' => (le_antisymm (le_of_not_gt h') (le_of_not_gt h))) :
    ltByCases x y h₁ h₂ h₃ = if h' : y < x then h₃ h' else h₂ (p h') := dif_neg h


lemma ltByCases_not_gt (h : ¬ y < x) {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P}
    (p : ¬ x < y → x = y := fun h' => (le_antisymm (le_of_not_gt h) (le_of_not_gt h'))) :
    ltByCases x y h₁ h₂ h₃ = if h' : x < y then h₁ h' else h₂ (p h') :=
  dite_congr rfl (fun _ => rfl) (fun _ => dif_neg h)


lemma ltByCases_ne (h : x ≠ y) {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P}
    (p : ¬ x < y → y < x := fun h' => h.lt_or_lt.resolve_left h') :
    ltByCases x y h₁ h₂ h₃ = if h' : x < y then h₁ h' else h₃ (p h') :=
  dite_congr rfl (fun _ => rfl) (fun _ => dif_pos _)


lemma ltByCases_comm {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P}
    (p : y = x → x = y := fun h' => h'.symm) :
    ltByCases x y h₁ h₂ h₃ = ltByCases y x h₃ (h₂ ∘ p) h₁ := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    P : Sort u_5
    x y : α
    h₁ : LT.lt x y → P
    h₂ : Eq x y → P
    h₃ : LT.lt y x → P
    p : optParam (Eq y x → Eq x y) ⋯
    ⊢ Eq (ltByCases x y h₁ h₂ h₃) (ltByCases y x h₃ (Function.comp h₂ p) h₁)
  -/
  refine ltByCases x y (fun h => ?_) (fun h => ?_) (fun h => ?_)
    /-
      case refine_1
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      p : optParam (Eq y x → Eq x y) ⋯
      h : LT.lt x y
      ⊢ Eq (ltByCases x y h₁ h₂ h₃) (ltByCases y x h₃ (Function.comp h₂ p) h₁)
    -/
  · rw [ltByCases_lt h, ltByCases_gt h]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      p : optParam (Eq y x → Eq x y) ⋯
      h : Eq x y
      ⊢ Eq (ltByCases x y h₁ h₂ h₃) (ltByCases y x h₃ (Function.comp h₂ p) h₁)
    -/
  · rw [ltByCases_eq h, ltByCases_eq h.symm, comp_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      p : optParam (Eq y x → Eq x y) ⋯
      h : LT.lt y x
      ⊢ Eq (ltByCases x y h₁ h₂ h₃) (ltByCases y x h₃ (Function.comp h₂ p) h₁)
    -/
  · rw [ltByCases_lt h, ltByCases_gt h]
    /-
      🎉 no goals
    -/


lemma eq_iff_eq_of_lt_iff_lt_of_gt_iff_gt {x' y' : α}
    (ltc : (x < y) ↔ (x' < y')) (gtc : (y < x) ↔ (y' < x')) :
                          /-
                            α : Type u_2
                            inst✝ : LinearOrder α
                            x y x' y' : α
                            ltc : Iff (LT.lt x y) (LT.lt x' y')
                            gtc : Iff (LT.lt y x) (LT.lt y' x')
                            ⊢ Iff (Eq x y) (Eq x' y')
                          -/
    x = y ↔ x' = y' := by simp_rw [eq_iff_le_not_lt, ← not_lt, ltc, gtc]
                          /-
                            🎉 no goals
                          -/


lemma ltByCases_rec {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P} (p : P)
    (hlt : (h : x < y) → h₁ h = p) (heq : (h : x = y) → h₂ h = p)
    (hgt : (h : y < x) → h₃ h = p) :
    ltByCases x y h₁ h₂ h₃ = p :=
  ltByCases x y
    (fun h => ltByCases_lt h ▸ hlt h)
    (fun h => ltByCases_eq h ▸ heq h)
    (fun h => ltByCases_gt h ▸ hgt h)


lemma ltByCases_eq_iff {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P} {p : P} :
    ltByCases x y h₁ h₂ h₃ = p ↔ (∃ h, h₁ h = p) ∨ (∃ h, h₂ h = p) ∨ (∃ h, h₃ h = p) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    P : Sort u_5
    x y : α
    h₁ : LT.lt x y → P
    h₂ : Eq x y → P
    h₃ : LT.lt y x → P
    p : P
    ⊢ Iff (Eq (ltByCases x y h₁ h₂ h₃) p) (Or (Exists fun h => Eq (h₁ h) p) (Or (E …
  -/
  refine ltByCases x y (fun h => ?_) (fun h => ?_) (fun h => ?_)
  · simp only [ltByCases_lt, exists_prop_of_true, h, h.not_lt, not_false_eq_true,
    exists_prop_of_false, or_false, h.ne]
  · simp only [h, lt_self_iff_false, ltByCases_eq, not_false_eq_true,
    exists_prop_of_false, exists_prop_of_true, or_false, false_or]
  · simp only [ltByCases_gt, exists_prop_of_true, h, h.not_lt, not_false_eq_true,
    exists_prop_of_false, false_or, h.ne']


lemma ltByCases_congr {x' y' : α} {h₁ : x < y → P} {h₂ : x = y → P} {h₃ : y < x → P}
    {h₁' : x' < y' → P} {h₂' : x' = y' → P} {h₃' : y' < x' → P} (ltc : (x < y) ↔ (x' < y'))
    (gtc : (y < x) ↔ (y' < x')) (hh'₁ : ∀ (h : x' < y'), h₁ (ltc.mpr h) = h₁' h)
    (hh'₂ : ∀ (h : x' = y'), h₂ ((eq_iff_eq_of_lt_iff_lt_of_gt_iff_gt ltc gtc).mpr h) = h₂' h)
    (hh'₃ : ∀ (h : y' < x'), h₃ (gtc.mpr h) = h₃' h) :
    ltByCases x y h₁ h₂ h₃ = ltByCases x' y' h₁' h₂' h₃' := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    P : Sort u_5
    x y x' y' : α
    h₁ : LT.lt x y → P
    h₂ : Eq x y → P
    h₃ : LT.lt y x → P
    h₁' : LT.lt x' y' → P
    h₂' : Eq x' y' → P
    h₃' : LT.lt y' x' → P
    ltc : Iff (LT.lt x y) (LT.lt x' y')
    gtc : Iff (LT.lt y x) (LT.lt y' x')
    hh'₁ : ∀ (h : LT.lt x' y'), Eq (h₁ ⋯) (h₁' h)
    hh'₂ : ∀ (h : Eq x' y'), Eq (h₂ ⋯) (h₂' h)
    hh'₃ : ∀ (h : LT.lt y' x'), Eq (h₃ ⋯) (h₃' h)
    ⊢ Eq (ltByCases x y h₁ h₂ h₃) (ltByCases x' y' h₁' h₂' h₃')
  -/
  refine ltByCases_rec _ (fun h => ?_) (fun h => ?_) (fun h => ?_)
    /-
      case refine_1
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y x' y' : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      h₁' : LT.lt x' y' → P
      h₂' : Eq x' y' → P
      h₃' : LT.lt y' x' → P
      ltc : Iff (LT.lt x y) (LT.lt x' y')
      gtc : Iff (LT.lt y x) (LT.lt y' x')
      hh'₁ : ∀ (h : LT.lt x' y'), Eq (h₁ ⋯) (h₁' h)
      hh'₂ : ∀ (h : Eq x' y'), Eq (h₂ ⋯) (h₂' h)
      hh'₃ : ∀ (h : LT.lt y' x'), Eq (h₃ ⋯) (h₃' h)
      h : LT.lt x y
      ⊢ Eq (h₁ h) (ltByCases x' y' h₁' h₂' h₃')
    -/
  · rw [ltByCases_lt (ltc.mp h), hh'₁]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y x' y' : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      h₁' : LT.lt x' y' → P
      h₂' : Eq x' y' → P
      h₃' : LT.lt y' x' → P
      ltc : Iff (LT.lt x y) (LT.lt x' y')
      gtc : Iff (LT.lt y x) (LT.lt y' x')
      hh'₁ : ∀ (h : LT.lt x' y'), Eq (h₁ ⋯) (h₁' h)
      hh'₂ : ∀ (h : Eq x' y'), Eq (h₂ ⋯) (h₂' h)
      hh'₃ : ∀ (h : LT.lt y' x'), Eq (h₃ ⋯) (h₃' h)
      h : Eq x y
      ⊢ Eq (h₂ h) (ltByCases x' y' h₁' h₂' h₃')
    -/
  · rw [eq_iff_eq_of_lt_iff_lt_of_gt_iff_gt ltc gtc] at h
    /-
      case refine_2
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y x' y' : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      h₁' : LT.lt x' y' → P
      h₂' : Eq x' y' → P
      h₃' : LT.lt y' x' → P
      ltc : Iff (LT.lt x y) (LT.lt x' y')
      gtc : Iff (LT.lt y x) (LT.lt y' x')
      hh'₁ : ∀ (h : LT.lt x' y'), Eq (h₁ ⋯) (h₁' h)
      hh'₂ : ∀ (h : Eq x' y'), Eq (h₂ ⋯) (h₂' h)
      hh'₃ : ∀ (h : LT.lt y' x'), Eq (h₃ ⋯) (h₃' h)
      h✝ : Eq x y
      h : Eq x' y'
      ⊢ Eq (h₂ h✝) (ltByCases x' y' h₁' h₂' h₃')
    -/
    rw [ltByCases_eq h, hh'₂]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y x' y' : α
      h₁ : LT.lt x y → P
      h₂ : Eq x y → P
      h₃ : LT.lt y x → P
      h₁' : LT.lt x' y' → P
      h₂' : Eq x' y' → P
      h₃' : LT.lt y' x' → P
      ltc : Iff (LT.lt x y) (LT.lt x' y')
      gtc : Iff (LT.lt y x) (LT.lt y' x')
      hh'₁ : ∀ (h : LT.lt x' y'), Eq (h₁ ⋯) (h₁' h)
      hh'₂ : ∀ (h : Eq x' y'), Eq (h₂ ⋯) (h₂' h)
      hh'₃ : ∀ (h : LT.lt y' x'), Eq (h₃ ⋯) (h₃' h)
      h : LT.lt y x
      ⊢ Eq (h₃ h) (ltByCases x' y' h₁' h₂' h₃')
    -/
  · rw [ltByCases_gt (gtc.mp h), hh'₃]
    /-
      🎉 no goals
    -/


/-- Perform a case-split on the ordering of `x` and `y` in a decidable linear order,
non-dependently. -/
abbrev ltTrichotomy (x y : α) (p q r : P) := ltByCases x y (fun _ => p) (fun _ => q) (fun _ => r)


@[simp]
lemma ltTrichotomy_lt (h : x < y) : ltTrichotomy x y p q r = p := ltByCases_lt h


@[simp]
lemma ltTrichotomy_gt (h : y < x) : ltTrichotomy x y p q r = r := ltByCases_gt h


@[simp]
lemma ltTrichotomy_eq (h : x = y) : ltTrichotomy x y p q r = q := ltByCases_eq h


lemma ltTrichotomy_not_lt (h : ¬ x < y) :
    ltTrichotomy x y p q r = if y < x then r else q := ltByCases_not_lt h


lemma ltTrichotomy_not_gt (h : ¬ y < x) :
    ltTrichotomy x y p q r = if x < y then p else q := ltByCases_not_gt h


lemma ltTrichotomy_ne (h : x ≠ y) :
    ltTrichotomy x y p q r = if x < y then p else r := ltByCases_ne h


lemma ltTrichotomy_comm : ltTrichotomy x y p q r = ltTrichotomy y x r q p := ltByCases_comm


lemma ltTrichotomy_self {p : P} : ltTrichotomy x y p p p = p :=
  ltByCases_rec p (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)


lemma ltTrichotomy_eq_iff : ltTrichotomy x y p q r = s ↔
    (x < y ∧ p = s) ∨ (x = y ∧ q = s) ∨ (y < x ∧ r = s) := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    P : Sort u_5
    x y : α
    p q r s : P
    ⊢ Iff (Eq (ltTrichotomy x y p q r) s) (Or (And (LT.lt x y) (Eq p s)) (Or (And  …
  -/
  refine ltByCases x y (fun h => ?_) (fun h => ?_) (fun h => ?_)
    /-
      case refine_1
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y : α
      p q r s : P
      h : LT.lt x y
      ⊢ Iff (Eq (ltTrichotomy x y p q r) s) (Or (And (LT.lt x y) (Eq p s)) (Or (And  …
    -/
  · simp only [ltTrichotomy_lt, false_and, true_and, or_false, h, h.not_lt, h.ne]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y : α
      p q r s : P
      h : Eq x y
      ⊢ Iff (Eq (ltTrichotomy x y p q r) s) (Or (And (LT.lt x y) (Eq p s)) (Or (And  …
    -/
  · simp only [ltTrichotomy_eq, false_and, true_and, or_false, false_or, h, lt_irrefl]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_2
      inst✝ : LinearOrder α
      P : Sort u_5
      x y : α
      p q r s : P
      h : LT.lt y x
      ⊢ Iff (Eq (ltTrichotomy x y p q r) s) (Or (And (LT.lt x y) (Eq p s)) (Or (And  …
    -/
  · simp only [ltTrichotomy_gt, false_and, true_and, false_or, h, h.not_lt, h.ne']
    /-
      🎉 no goals
    -/


lemma ltTrichotomy_congr {x' y' : α} {p' q' r' : P} (ltc : (x < y) ↔ (x' < y'))
    (gtc : (y < x) ↔ (y' < x')) (hh'₁ : x' < y' → p = p')
    (hh'₂ : x' = y' → q = q') (hh'₃ : y' < x' → r = r') :
    ltTrichotomy x y p q r = ltTrichotomy x' y' p' q' r' :=
  ltByCases_congr ltc gtc hh'₁ hh'₂ hh'₃


/-- Type synonym to equip a type with the dual order: `≤` means `≥` and `<` means `>`. `αᵒᵈ` is
notation for `OrderDual α`. -/
def OrderDual (α : Type*) : Type _ :=
  α


@[inherit_doc]
notation:max α "ᵒᵈ" => OrderDual α


instance (α : Type*) [h : Nonempty α] : Nonempty αᵒᵈ :=
  h


instance (α : Type*) [h : Subsingleton α] : Subsingleton αᵒᵈ :=
  h


instance (α : Type*) [LE α] : LE αᵒᵈ :=
  ⟨fun x y : α ↦ y ≤ x⟩


instance (α : Type*) [LT α] : LT αᵒᵈ :=
  ⟨fun x y : α ↦ y < x⟩


instance instOrd (α : Type*) [Ord α] : Ord αᵒᵈ where
  compare := fun (a b : α) ↦ compare b a


instance instSup (α : Type*) [Min α] : Max αᵒᵈ :=
  ⟨((· ⊓ ·) : α → α → α)⟩


instance instInf (α : Type*) [Max α] : Min αᵒᵈ :=
  ⟨((· ⊔ ·) : α → α → α)⟩


instance instPreorder (α : Type*) [Preorder α] : Preorder αᵒᵈ where
  le_refl := fun _ ↦ le_refl _
  le_trans := fun _ _ _ hab hbc ↦ hbc.trans hab
  lt_iff_le_not_le := fun _ _ ↦ lt_iff_le_not_le


instance instPartialOrder (α : Type*) [PartialOrder α] : PartialOrder αᵒᵈ where
  __ := inferInstanceAs (Preorder αᵒᵈ)
  le_antisymm := fun a b hab hba ↦ @le_antisymm α _ a b hba hab


instance instLinearOrder (α : Type*) [LinearOrder α] : LinearOrder αᵒᵈ where
  __ := inferInstanceAs (PartialOrder αᵒᵈ)
  __ := inferInstanceAs (Ord αᵒᵈ)
  le_total := fun a b : α ↦ le_total b a
  max := fun a b ↦ (min a b : α)
  min := fun a b ↦ (max a b : α)
                                                /-
                                                  ι : Type u_1
                                                  α✝ : Type u_2
                                                  β : Type u_3
                                                  π : ι → Type u_4
                                                  α : Type u_5
                                                  inst✝ : LinearOrder α
                                                  a b : OrderDual α
                                                  ⊢ Eq (Max.max a b) (ite (LE.le a b) a b)
                                                -/
  min_def := fun a b ↦ show (max .. : α) = _ by rw [max_comm, max_def]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                /-
                                                  ι : Type u_1
                                                  α✝ : Type u_2
                                                  β : Type u_3
                                                  π : ι → Type u_4
                                                  α : Type u_5
                                                  inst✝ : LinearOrder α
                                                  a b : OrderDual α
                                                  ⊢ Eq (Min.min a b) (ite (LE.le a b) b a)
                                                -/
  max_def := fun a b ↦ show (min .. : α) = _ by rw [min_comm, min_def]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  decidableLE := (inferInstance : DecidableRel (fun a b : α ↦ b ≤ a))
  decidableLT := (inferInstance : DecidableRel (fun a b : α ↦ b < a))
  decidableEq := (inferInstance : DecidableEq α)
  compare_eq_compareOfLessAndEq a b := by
    /-
      ι : Type u_1
      α✝ : Type u_2
      β : Type u_3
      π : ι → Type u_4
      α : Type u_5
      inst✝ : LinearOrder α
      a b : OrderDual α
      ⊢ Eq (Ord.compare a b) (compareOfLessAndEq a b)
    -/
    simp only [compare, LinearOrder.compare_eq_compareOfLessAndEq, compareOfLessAndEq, eq_comm]
    /-
      ι : Type u_1
      α✝ : Type u_2
      β : Type u_3
      π : ι → Type u_4
      α : Type u_5
      inst✝ : LinearOrder α
      a b : OrderDual α
      ⊢ Eq (ite (LT.lt b a) Ordering.lt (ite (Eq a b) Ordering.eq Ordering.gt)) (ite …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The opposite linear order to a given linear order -/
def _root_.LinearOrder.swap (α : Type*) (_ : LinearOrder α) : LinearOrder α :=
  inferInstanceAs <| LinearOrder (OrderDual α)


instance : ∀ [Inhabited α], Inhabited αᵒᵈ := fun [x : Inhabited α] => x


theorem Ord.dual_dual (α : Type*) [H : Ord α] : OrderDual.instOrd αᵒᵈ = H :=
  rfl


theorem Preorder.dual_dual (α : Type*) [H : Preorder α] : OrderDual.instPreorder αᵒᵈ = H :=
  rfl


theorem instPartialOrder.dual_dual (α : Type*) [H : PartialOrder α] :
    OrderDual.instPartialOrder αᵒᵈ = H :=
  rfl


theorem instLinearOrder.dual_dual (α : Type*) [H : LinearOrder α] :
    OrderDual.instLinearOrder αᵒᵈ = H :=
  rfl


instance Prop.hasCompl : HasCompl Prop :=
  ⟨Not⟩


instance Pi.hasCompl [∀ i, HasCompl (π i)] : HasCompl (∀ i, π i) :=
  ⟨fun x i ↦ (x i)ᶜ⟩


theorem Pi.compl_def [∀ i, HasCompl (π i)] (x : ∀ i, π i) :
    xᶜ = fun i ↦ (x i)ᶜ :=
  rfl


@[simp]
theorem Pi.compl_apply [∀ i, HasCompl (π i)] (x : ∀ i, π i) (i : ι) :
    xᶜ i = (x i)ᶜ :=
  rfl


instance IsIrrefl.compl (r) [IsIrrefl α r] : IsRefl α rᶜ :=
  ⟨@irrefl α r _⟩


instance IsRefl.compl (r) [IsRefl α r] : IsIrrefl α rᶜ :=
  ⟨fun a ↦ not_not_intro (refl a)⟩


                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝ : LinearOrder α
                                                                          ⊢ Eq (HasCompl.compl fun x1 x2 => LT.lt x1 x2) fun x1 x2 => GE.ge x1 x2
                                                                        -/
theorem compl_lt [LinearOrder α] : (· < · : α → α → _)ᶜ = (· ≥ ·) := by ext; simp [compl]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝ : LinearOrder α
                                                                          ⊢ Eq (HasCompl.compl fun x1 x2 => LE.le x1 x2) fun x1 x2 => GT.gt x1 x2
                                                                        -/
theorem compl_le [LinearOrder α] : (· ≤ · : α → α → _)ᶜ = (· > ·) := by ext; simp [compl]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝ : LinearOrder α
                                                                          ⊢ Eq (HasCompl.compl fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LE.le x1 x2
                                                                        -/
theorem compl_gt [LinearOrder α] : (· > · : α → α → _)ᶜ = (· ≤ ·) := by ext; simp [compl]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝ : LinearOrder α
                                                                          ⊢ Eq (HasCompl.compl fun x1 x2 => GE.ge x1 x2) fun x1 x2 => LT.lt x1 x2
                                                                        -/
theorem compl_ge [LinearOrder α] : (· ≥ · : α → α → _)ᶜ = (· < ·) := by ext; simp [compl]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance Ne.instIsEquiv_compl : IsEquiv α (· ≠ ·)ᶜ := by
  /-
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    π : ι → Type u_4
    ⊢ IsEquiv α (HasCompl.compl fun x1 x2 => Ne x1 x2)
  -/
  convert eq_isEquiv α
  /-
    case h.e'_2.h.h.h.e
    ι : Type u_1
    α : Type u_2
    β : Type u_3
    π : ι → Type u_4
    x✝¹ x✝ : α
    ⊢ Eq (HasCompl.compl fun x1 x2 => Ne x1 x2) Eq
  -/
  simp [compl]
  /-
    🎉 no goals
  -/


instance Pi.hasLe [∀ i, LE (π i)] :
    LE (∀ i, π i) where le x y := ∀ i, x i ≤ y i


theorem Pi.le_def [∀ i, LE (π i)] {x y : ∀ i, π i} :
    x ≤ y ↔ ∀ i, x i ≤ y i :=
  Iff.rfl


instance Pi.preorder [∀ i, Preorder (π i)] : Preorder (∀ i, π i) where
  __ := inferInstanceAs (LE (∀ i, π i))
  le_refl := fun a i ↦ le_refl (a i)
  le_trans := fun _ _ _ h₁ h₂ i ↦ le_trans (h₁ i) (h₂ i)


theorem Pi.lt_def [∀ i, Preorder (π i)] {x y : ∀ i, π i} :
    x < y ↔ x ≤ y ∧ ∃ i, x i < y i := by
  /-
    ι : Type u_1
    π : ι → Type u_4
    inst✝ : (i : ι) → Preorder (π i)
    x y : (i : ι) → π i
    ⊢ Iff (LT.lt x y) (And (LE.le x y) (Exists fun i => LT.lt (x i) (y i)))
  -/
  simp +contextual [lt_iff_le_not_le, Pi.le_def]
  /-
    🎉 no goals
  -/


instance Pi.partialOrder [∀ i, PartialOrder (π i)] : PartialOrder (∀ i, π i) where
  __ := Pi.preorder
  le_antisymm := fun _ _ h1 h2 ↦ funext fun b ↦ (h1 b).antisymm (h2 b)


@[simp]
lemma elim_le_elim_iff {u₁ v₁ : α₁ → β} {u₂ v₂ : α₂ → β} :
    Sum.elim u₁ u₂ ≤ Sum.elim v₁ v₂ ↔ u₁ ≤ v₁ ∧ u₂ ≤ v₂ :=
  Sum.forall


lemma const_le_elim_iff {b : β} {v₁ : α₁ → β} {v₂ : α₂ → β} :
    Function.const _ b ≤ Sum.elim v₁ v₂ ↔ Function.const _ b ≤ v₁ ∧ Function.const _ b ≤ v₂ :=
  elim_const_const b ▸ elim_le_elim_iff ..


lemma elim_le_const_iff {b : β} {u₁ : α₁ → β} {u₂ : α₂ → β} :
    Sum.elim u₁ u₂ ≤ Function.const _ b ↔ u₁ ≤ Function.const _ b ∧ u₂ ≤ Function.const _ b :=
  elim_const_const b ▸ elim_le_elim_iff ..


/-- A function `a` is strongly less than a function `b` if `a i < b i` for all `i`. -/
def StrongLT [∀ i, LT (π i)] (a b : ∀ i, π i) : Prop :=
  ∀ i, a i < b i


@[inherit_doc]
local infixl:50 " ≺ " => StrongLT


theorem le_of_strongLT (h : a ≺ b) : a ≤ b := fun _ ↦ (h _).le


theorem lt_of_strongLT [Nonempty ι] (h : a ≺ b) : a < b := by
  /-
    ι : Type u_1
    π : ι → Type u_4
    inst✝¹ : (i : ι) → Preorder (π i)
    a b : (i : ι) → π i
    inst✝ : Nonempty ι
    h : StrongLT a b
    ⊢ LT.lt a b
  -/
  inhabit ι
  /-
    ι : Type u_1
    π : ι → Type u_4
    inst✝¹ : (i : ι) → Preorder (π i)
    a b : (i : ι) → π i
    inst✝ : Nonempty ι
    h : StrongLT a b
    inhabited_h : Inhabited ι
    ⊢ LT.lt a b
  -/
  exact Pi.lt_def.2 ⟨le_of_strongLT h, default, h _⟩
  /-
    🎉 no goals
  -/


theorem strongLT_of_strongLT_of_le (hab : a ≺ b) (hbc : b ≤ c) : a ≺ c := fun _ ↦
  (hab _).trans_le <| hbc _


theorem strongLT_of_le_of_strongLT (hab : a ≤ b) (hbc : b ≺ c) : a ≺ c := fun _ ↦
  (hab _).trans_lt <| hbc _


alias StrongLT.le := le_of_strongLT


alias StrongLT.lt := lt_of_strongLT


alias StrongLT.trans_le := strongLT_of_strongLT_of_le


alias LE.le.trans_strongLT := strongLT_of_le_of_strongLT


theorem le_update_iff : x ≤ Function.update y i a ↔ x i ≤ a ∧ ∀ (j) (_ : j ≠ i), x j ≤ y j :=
  Function.forall_update_iff _ fun j z ↦ x j ≤ z


theorem update_le_iff : Function.update x i a ≤ y ↔ a ≤ y i ∧ ∀ (j) (_ : j ≠ i), x j ≤ y j :=
  Function.forall_update_iff _ fun j z ↦ z ≤ y j


theorem update_le_update_iff :
    Function.update x i a ≤ Function.update y i b ↔ a ≤ b ∧ ∀ (j) (_ : j ≠ i), x j ≤ y j := by
  /-
    ι : Type u_1
    π : ι → Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → Preorder (π i)
    x y : (i : ι) → π i
    i : ι
    a b : π i
    ⊢ Iff (LE.le (Function.update x i a) (Function.update y i b)) (And (LE.le a b) …
  -/
  simp +contextual [update_le_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem update_le_update_iff' : update x i a ≤ update x i b ↔ a ≤ b := by
  /-
    ι : Type u_1
    π : ι → Type u_4
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → Preorder (π i)
    x : (i : ι) → π i
    i : ι
    a b : π i
    ⊢ Iff (LE.le (Function.update x i a) (Function.update x i b)) (LE.le a b)
  -/
  simp [update_le_update_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem update_lt_update_iff : update x i a < update x i b ↔ a < b :=
  lt_iff_lt_of_le_iff_le' update_le_update_iff' update_le_update_iff'


@[simp]
                                                              /-
                                                                ι : Type u_1
                                                                π : ι → Type u_4
                                                                inst✝¹ : DecidableEq ι
                                                                inst✝ : (i : ι) → Preorder (π i)
                                                                x : (i : ι) → π i
                                                                i : ι
                                                                a : π i
                                                                ⊢ Iff (LE.le x (Function.update x i a)) (LE.le (x i) a)
                                                              -/
theorem le_update_self_iff : x ≤ update x i a ↔ x i ≤ a := by simp [le_update_iff]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                              /-
                                                                ι : Type u_1
                                                                π : ι → Type u_4
                                                                inst✝¹ : DecidableEq ι
                                                                inst✝ : (i : ι) → Preorder (π i)
                                                                x : (i : ι) → π i
                                                                i : ι
                                                                a : π i
                                                                ⊢ Iff (LE.le (Function.update x i a) x) (LE.le a (x i))
                                                              -/
theorem update_le_self_iff : update x i a ≤ x ↔ a ≤ x i := by simp [update_le_iff]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                              /-
                                                                ι : Type u_1
                                                                π : ι → Type u_4
                                                                inst✝¹ : DecidableEq ι
                                                                inst✝ : (i : ι) → Preorder (π i)
                                                                x : (i : ι) → π i
                                                                i : ι
                                                                a : π i
                                                                ⊢ Iff (LT.lt x (Function.update x i a)) (LT.lt (x i) a)
                                                              -/
theorem lt_update_self_iff : x < update x i a ↔ x i < a := by simp [lt_iff_le_not_le]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                              /-
                                                                ι : Type u_1
                                                                π : ι → Type u_4
                                                                inst✝¹ : DecidableEq ι
                                                                inst✝ : (i : ι) → Preorder (π i)
                                                                x : (i : ι) → π i
                                                                i : ι
                                                                a : π i
                                                                ⊢ Iff (LT.lt (Function.update x i a) x) (LT.lt a (x i))
                                                              -/
theorem update_lt_self_iff : update x i a < x ↔ a < x i := by simp [lt_iff_le_not_le]
                                                              /-
                                                                🎉 no goals
                                                              -/


instance Pi.sdiff [∀ i, SDiff (π i)] : SDiff (∀ i, π i) :=
  ⟨fun x y i ↦ x i \ y i⟩


theorem Pi.sdiff_def [∀ i, SDiff (π i)] (x y : ∀ i, π i) :
    x \ y = fun i ↦ x i \ y i :=
  rfl


@[simp]
theorem Pi.sdiff_apply [∀ i, SDiff (π i)] (x y : ∀ i, π i) (i : ι) :
    (x \ y) i = x i \ y i :=
  rfl


@[simp]
                                                             /-
                                                               α : Type u_2
                                                               β : Type u_3
                                                               inst✝¹ : Preorder α
                                                               inst✝ : Nonempty β
                                                               a b : α
                                                               ⊢ Iff (LE.le (Function.const β a) (Function.const β b)) (LE.le a b)
                                                             -/
theorem const_le_const : const β a ≤ const β b ↔ a ≤ b := by simp [Pi.le_def]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                             /-
                                                               α : Type u_2
                                                               β : Type u_3
                                                               inst✝¹ : Preorder α
                                                               inst✝ : Nonempty β
                                                               a b : α
                                                               ⊢ Iff (LT.lt (Function.const β a) (Function.const β b)) (LT.lt a b)
                                                             -/
theorem const_lt_const : const β a < const β b ↔ a < b := by simpa [Pi.lt_def] using le_of_lt
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem min_rec (hx : x ≤ y → p x) (hy : y ≤ x → p y) : p (min x y) :=
  (le_total x y).rec (fun h ↦ (min_eq_left h).symm.subst (hx h)) fun h ↦
    (min_eq_right h).symm.subst (hy h)


theorem max_rec (hx : y ≤ x → p x) (hy : x ≤ y → p y) : p (max x y) :=
  @min_rec αᵒᵈ _ _ _ _ hx hy


theorem min_rec' (p : α → Prop) (hx : p x) (hy : p y) : p (min x y) :=
  min_rec (fun _ ↦ hx) fun _ ↦ hy


theorem max_rec' (p : α → Prop) (hx : p x) (hy : p y) : p (max x y) :=
  max_rec (fun _ ↦ hx) fun _ ↦ hy


theorem min_def_lt (x y : α) : min x y = if x < y then x else y := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    x y : α
    ⊢ Eq (Min.min x y) (ite (LT.lt x y) x y)
  -/
  rw [min_comm, min_def, ← ite_not]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    x y : α
    ⊢ Eq (ite (Not (LE.le y x)) x y) (ite (LT.lt x y) x y)
  -/
  simp only [not_le]
  /-
    🎉 no goals
  -/


theorem max_def_lt (x y : α) : max x y = if x < y then y else x := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    x y : α
    ⊢ Eq (Max.max x y) (ite (LT.lt x y) y x)
  -/
  rw [max_comm, max_def, ← ite_not]
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    x y : α
    ⊢ Eq (ite (Not (LE.le y x)) y x) (ite (LT.lt x y) y x)
  -/
  simp only [not_le]
  /-
    🎉 no goals
  -/


/-- Transfer a `Preorder` on `β` to a `Preorder` on `α` using a function `f : α → β`.
See note [reducible non-instances]. -/
abbrev Preorder.lift [Preorder β] (f : α → β) : Preorder α where
  le x y := f x ≤ f y
  le_refl _ := le_rfl
  le_trans _ _ _ := _root_.le_trans
  lt x y := f x < f y
  lt_iff_le_not_le _ _ := _root_.lt_iff_le_not_le


/-- Transfer a `PartialOrder` on `β` to a `PartialOrder` on `α` using an injective
function `f : α → β`. See note [reducible non-instances]. -/
abbrev PartialOrder.lift [PartialOrder β] (f : α → β) (inj : Injective f) : PartialOrder α :=
  { Preorder.lift f with le_antisymm := fun _ _ h₁ h₂ ↦ inj (h₁.antisymm h₂) }


theorem compare_of_injective_eq_compareOfLessAndEq (a b : α) [LinearOrder β]
    [DecidableEq α] (f : α → β) (inj : Injective f)
    [Decidable (LT.lt (self := PartialOrder.lift f inj |>.toLT) a b)] :
    compare (f a) (f b) =
      @compareOfLessAndEq _ a b (PartialOrder.lift f inj |>.toLT) _ _ := by
  /-
    α : Type u_2
    β : Type u_3
    a b : α
    inst✝² : LinearOrder β
    inst✝¹ : DecidableEq α
    f : α → β
    inj : Function.Injective f
    inst✝ : Decidable (LT.lt a b)
    ⊢ Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq a b)
  -/
  have h := LinearOrder.compare_eq_compareOfLessAndEq (f a) (f b)
  /-
    α : Type u_2
    β : Type u_3
    a b : α
    inst✝² : LinearOrder β
    inst✝¹ : DecidableEq α
    f : α → β
    inj : Function.Injective f
    inst✝ : Decidable (LT.lt a b)
    h : Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq (f a) (f b))
    ⊢ Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq a b)
  -/
  simp only [h, compareOfLessAndEq]
  /-
    α : Type u_2
    β : Type u_3
    a b : α
    inst✝² : LinearOrder β
    inst✝¹ : DecidableEq α
    f : α → β
    inj : Function.Injective f
    inst✝ : Decidable (LT.lt a b)
    h : Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq (f a) (f b))
    ⊢ Eq (ite (LT.lt (f a) (f b)) Ordering.lt (ite (Eq (f a) (f b)) Ordering.eq Or …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> try (first | rfl | contradiction)
                /-
                  🎉 no goals
                -/
    /-
      case neg
      α : Type u_2
      β : Type u_3
      a b : α
      inst✝² : LinearOrder β
      inst✝¹ : DecidableEq α
      f : α → β
      inj : Function.Injective f
      inst✝ : Decidable (LT.lt a b)
      h : Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq (f a) (f b))
      h✝³ : Not (LT.lt (f a) (f b))
      h✝² : Eq (f a) (f b)
      h✝¹ : Not (LT.lt a b)
      h✝ : Not (Eq a b)
      ⊢ False
    -/
  · have : ¬ f a = f b := by rename_i h; exact inj.ne h
    /-
      case neg
      α : Type u_2
      β : Type u_3
      a b : α
      inst✝² : LinearOrder β
      inst✝¹ : DecidableEq α
      f : α → β
      inj : Function.Injective f
      inst✝ : Decidable (LT.lt a b)
      h : Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq (f a) (f b))
      h✝³ : Not (LT.lt (f a) (f b))
      h✝² : Eq (f a) (f b)
      h✝¹ : Not (LT.lt a b)
      h✝ : Not (Eq a b)
      this : Not (Eq (f a) (f b))
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_2
      β : Type u_3
      a b : α
      inst✝² : LinearOrder β
      inst✝¹ : DecidableEq α
      f : α → β
      inj : Function.Injective f
      inst✝ : Decidable (LT.lt a b)
      h : Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq (f a) (f b))
      h✝³ : Not (LT.lt (f a) (f b))
      h✝² : Not (Eq (f a) (f b))
      h✝¹ : Not (LT.lt a b)
      h✝ : Eq a b
      ⊢ False
    -/
  · have : f a = f b := by rename_i h; exact congrArg f h
    /-
      case pos
      α : Type u_2
      β : Type u_3
      a b : α
      inst✝² : LinearOrder β
      inst✝¹ : DecidableEq α
      f : α → β
      inj : Function.Injective f
      inst✝ : Decidable (LT.lt a b)
      h : Eq (Ord.compare (f a) (f b)) (compareOfLessAndEq (f a) (f b))
      h✝³ : Not (LT.lt (f a) (f b))
      h✝² : Not (Eq (f a) (f b))
      h✝¹ : Not (LT.lt a b)
      h✝ : Eq a b
      this : Eq (f a) (f b)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


/-- Transfer a `LinearOrder` on `β` to a `LinearOrder` on `α` using an injective
function `f : α → β`. This version takes `[Max α]` and `[Min α]` as arguments, then uses
them for `max` and `min` fields. See `LinearOrder.lift'` for a version that autogenerates `min` and
`max` fields, and `LinearOrder.liftWithOrd` for one that does not auto-generate `compare`
fields. See note [reducible non-instances]. -/
abbrev LinearOrder.lift [LinearOrder β] [Max α] [Min α] (f : α → β) (inj : Injective f)
    (hsup : ∀ x y, f (x ⊔ y) = max (f x) (f y)) (hinf : ∀ x y, f (x ⊓ y) = min (f x) (f y)) :
    LinearOrder α :=
  letI instOrdα : Ord α := ⟨fun a b ↦ compare (f a) (f b)⟩
  letI decidableLE := fun x y ↦ (inferInstance : Decidable (f x ≤ f y))
  letI decidableLT := fun x y ↦ (inferInstance : Decidable (f x < f y))
  letI decidableEq := fun x y ↦ decidable_of_iff (f x = f y) inj.eq_iff
  { PartialOrder.lift f inj, instOrdα with
    le_total := fun x y ↦ le_total (f x) (f y)
    decidableLE := decidableLE
    decidableLT := decidableLT
    decidableEq := decidableEq
    min := (· ⊓ ·)
    max := (· ⊔ ·)
    min_def := by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        ⊢ ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      -/
      intros x y
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (Min.min x y) (ite (LE.le x y) x y)
      -/
      apply inj
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Min.min x y)) (f (ite (LE.le x y) x y))
      -/
      rw [apply_ite f]
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Min.min x y)) (ite (LE.le x y) (f x) (f y))
      -/
      exact (hinf _ _).trans (min_def _ _)
      /-
        🎉 no goals
      -/
    max_def := by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        ⊢ ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      -/
      intros x y
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (Max.max x y) (ite (LE.le x y) y x)
      -/
      apply inj
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Max.max x y)) (f (ite (LE.le x y) y x))
      -/
      rw [apply_ite f]
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝² : LinearOrder β
        inst✝¹ : Max α
        inst✝ : Min α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        instOrdα : Ord α := { compare := fun a b => Ord.compare (f a) (f b) }
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Max.max x y)) (ite (LE.le x y) (f y) (f x))
      -/
      exact (hsup _ _).trans (max_def _ _)
      /-
        🎉 no goals
      -/
    compare_eq_compareOfLessAndEq := fun a b ↦
      compare_of_injective_eq_compareOfLessAndEq a b f inj }


/-- Transfer a `LinearOrder` on `β` to a `LinearOrder` on `α` using an injective
function `f : α → β`. This version autogenerates `min` and `max` fields. See `LinearOrder.lift`
for a version that takes `[Max α]` and `[Min α]`, then uses them as `max` and `min`. See
`LinearOrder.liftWithOrd'` for a version which does not auto-generate `compare` fields.
See note [reducible non-instances]. -/
abbrev LinearOrder.lift' [LinearOrder β] (f : α → β) (inj : Injective f) : LinearOrder α :=
  @LinearOrder.lift α β _ ⟨fun x y ↦ if f x ≤ f y then y else x⟩
    ⟨fun x y ↦ if f x ≤ f y then x else y⟩ f inj
    (fun _ _ ↦ (apply_ite f _ _ _).trans (max_def _ _).symm) fun _ _ ↦
    (apply_ite f _ _ _).trans (min_def _ _).symm


/-- Transfer a `LinearOrder` on `β` to a `LinearOrder` on `α` using an injective
function `f : α → β`. This version takes `[Max α]` and `[Min α]` as arguments, then uses
them for `max` and `min` fields. It also takes `[Ord α]` as an argument and uses them for `compare`
fields. See `LinearOrder.lift` for a version that autogenerates `compare` fields, and
`LinearOrder.liftWithOrd'` for one that auto-generates `min` and `max` fields.
fields. See note [reducible non-instances]. -/
abbrev LinearOrder.liftWithOrd [LinearOrder β] [Max α] [Min α] [Ord α] (f : α → β)
    (inj : Injective f) (hsup : ∀ x y, f (x ⊔ y) = max (f x) (f y))
    (hinf : ∀ x y, f (x ⊓ y) = min (f x) (f y))
    (compare_f : ∀ a b : α, compare a b = compare (f a) (f b)) : LinearOrder α :=
  letI decidableLE := fun x y ↦ (inferInstance : Decidable (f x ≤ f y))
  letI decidableLT := fun x y ↦ (inferInstance : Decidable (f x < f y))
  letI decidableEq := fun x y ↦ decidable_of_iff (f x = f y) inj.eq_iff
  { PartialOrder.lift f inj with
    le_total := fun x y ↦ le_total (f x) (f y)
    decidableLE := decidableLE
    decidableLT := decidableLT
    decidableEq := decidableEq
    min := (· ⊓ ·)
    max := (· ⊔ ·)
    min_def := by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        ⊢ ∀ (a b : α), Eq (Min.min a b) (ite (LE.le a b) a b)
      -/
      intros x y
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (Min.min x y) (ite (LE.le x y) x y)
      -/
      apply inj
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Min.min x y)) (f (ite (LE.le x y) x y))
      -/
      rw [apply_ite f]
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Min.min x y)) (ite (LE.le x y) (f x) (f y))
      -/
      exact (hinf _ _).trans (min_def _ _)
      /-
        🎉 no goals
      -/
    max_def := by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        ⊢ ∀ (a b : α), Eq (Max.max a b) (ite (LE.le a b) b a)
      -/
      intros x y
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (Max.max x y) (ite (LE.le x y) y x)
      -/
      apply inj
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Max.max x y)) (f (ite (LE.le x y) y x))
      -/
      rw [apply_ite f]
      /-
        case a
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : LinearOrder β
        inst✝² : Max α
        inst✝¹ : Min α
        inst✝ : Ord α
        f : α → β
        inj : Function.Injective f
        hsup : ∀ (x y : α), Eq (f (Max.max x y)) (Max.max (f x) (f y))
        hinf : ∀ (x y : α), Eq (f (Min.min x y)) (Min.min (f x) (f y))
        compare_f : ∀ (a b : α), Eq (Ord.compare a b) (Ord.compare (f a) (f b))
        decidableLE : (x y : α) → Decidable (LE.le (f x) (f y)) := fun x y => inferIns …
        decidableLT : (x y : α) → Decidable (LT.lt (f x) (f y)) := fun x y => inferIns …
        decidableEq : (x y : α) → Decidable (Eq x y) := fun x y => decidable_of_iff (E …
        x y : α
        ⊢ Eq (f (Max.max x y)) (ite (LE.le x y) (f y) (f x))
      -/
      exact (hsup _ _).trans (max_def _ _)
      /-
        🎉 no goals
      -/
    compare_eq_compareOfLessAndEq := fun a b ↦
      (compare_f a b).trans <| compare_of_injective_eq_compareOfLessAndEq a b f inj }


/-- Transfer a `LinearOrder` on `β` to a `LinearOrder` on `α` using an injective
function `f : α → β`. This version auto-generates `min` and `max` fields. It also takes `[Ord α]`
as an argument and uses them for `compare` fields. See `LinearOrder.lift` for a version that
autogenerates `compare` fields, and `LinearOrder.liftWithOrd` for one that doesn't auto-generate
`min` and `max` fields. fields. See note [reducible non-instances]. -/
abbrev LinearOrder.liftWithOrd' [LinearOrder β] [Ord α] (f : α → β)
    (inj : Injective f)
    (compare_f : ∀ a b : α, compare a b = compare (f a) (f b)) : LinearOrder α :=
  @LinearOrder.liftWithOrd α β _ ⟨fun x y ↦ if f x ≤ f y then y else x⟩
    ⟨fun x y ↦ if f x ≤ f y then x else y⟩ _ f inj
    (fun _ _ ↦ (apply_ite f _ _ _).trans (max_def _ _).symm)
    (fun _ _ ↦ (apply_ite f _ _ _).trans (min_def _ _).symm)
    compare_f


instance le [LE α] {p : α → Prop} : LE (Subtype p) :=
  ⟨fun x y ↦ (x : α) ≤ y⟩


instance lt [LT α] {p : α → Prop} : LT (Subtype p) :=
  ⟨fun x y ↦ (x : α) < y⟩


@[simp]
theorem mk_le_mk [LE α] {p : α → Prop} {x y : α} {hx : p x} {hy : p y} :
    (⟨x, hx⟩ : Subtype p) ≤ ⟨y, hy⟩ ↔ x ≤ y :=
  Iff.rfl


@[simp]
theorem mk_lt_mk [LT α] {p : α → Prop} {x y : α} {hx : p x} {hy : p y} :
    (⟨x, hx⟩ : Subtype p) < ⟨y, hy⟩ ↔ x < y :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_le_coe [LE α] {p : α → Prop} {x y : Subtype p} : (x : α) ≤ y ↔ x ≤ y :=
  Iff.rfl


@[gcongr] alias ⟨_, GCongr.coe_le_coe⟩ := coe_le_coe


@[simp, norm_cast]
theorem coe_lt_coe [LT α] {p : α → Prop} {x y : Subtype p} : (x : α) < y ↔ x < y :=
  Iff.rfl


@[gcongr] alias ⟨_, GCongr.coe_lt_coe⟩ := coe_lt_coe


instance preorder [Preorder α] (p : α → Prop) : Preorder (Subtype p) :=
  Preorder.lift (fun (a : Subtype p) ↦ (a : α))


instance partialOrder [PartialOrder α] (p : α → Prop) : PartialOrder (Subtype p) :=
  PartialOrder.lift (fun (a : Subtype p) ↦ (a : α)) Subtype.coe_injective


instance decidableLE [Preorder α] [h : DecidableRel (α := α) (· ≤ ·)] {p : α → Prop} :
    DecidableRel (α := Subtype p) (· ≤ ·) := fun a b ↦ h a b


instance decidableLT [Preorder α] [h : DecidableRel (α := α) (· < ·)] {p : α → Prop} :
    DecidableRel (α := Subtype p) (· < ·) := fun a b ↦ h a b


/-- A subtype of a linear order is a linear order. We explicitly give the proofs of decidable
equality and decidable order in order to ensure the decidability instances are all definitionally
equal. -/
instance instLinearOrder [LinearOrder α] (p : α → Prop) : LinearOrder (Subtype p) :=
  @LinearOrder.lift (Subtype p) _ _ ⟨fun x y ↦ ⟨max x y, max_rec' _ x.2 y.2⟩⟩
    ⟨fun x y ↦ ⟨min x y, min_rec' _ x.2 y.2⟩⟩ (fun (a : Subtype p) ↦ (a : α))
    Subtype.coe_injective (fun _ _ ↦ rfl) fun _ _ ↦
    rfl


instance (α β : Type*) [LE α] [LE β] : LE (α × β) :=
  ⟨fun p q ↦ p.1 ≤ q.1 ∧ p.2 ≤ q.2⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance instDecidableLE (α β : Type*) [LE α] [LE β] (x y : α × β)
    [Decidable (x.1 ≤ y.1)] [Decidable (x.2 ≤ y.2)] : Decidable (x ≤ y) :=
  inferInstanceAs (Decidable (x.1 ≤ y.1 ∧ x.2 ≤ y.2))


theorem le_def [LE α] [LE β] {x y : α × β} : x ≤ y ↔ x.1 ≤ y.1 ∧ x.2 ≤ y.2 :=
  Iff.rfl


@[simp]
theorem mk_le_mk [LE α] [LE β] {x₁ x₂ : α} {y₁ y₂ : β} : (x₁, y₁) ≤ (x₂, y₂) ↔ x₁ ≤ x₂ ∧ y₁ ≤ y₂ :=
  Iff.rfl


@[simp]
theorem swap_le_swap [LE α] [LE β] {x y : α × β} : x.swap ≤ y.swap ↔ x ≤ y :=
  and_comm


instance (α β : Type*) [Preorder α] [Preorder β] : Preorder (α × β) where
  __ := inferInstanceAs (LE (α × β))
  le_refl := fun ⟨a, b⟩ ↦ ⟨le_refl a, le_refl b⟩
  le_trans := fun ⟨_, _⟩ ⟨_, _⟩ ⟨_, _⟩ ⟨hac, hbd⟩ ⟨hce, hdf⟩ ↦ ⟨le_trans hac hce, le_trans hbd hdf⟩


@[simp]
theorem swap_lt_swap : x.swap < y.swap ↔ x < y :=
  and_congr swap_le_swap (not_congr swap_le_swap)


theorem mk_le_mk_iff_left : (a₁, b) ≤ (a₂, b) ↔ a₁ ≤ a₂ :=
  and_iff_left le_rfl


theorem mk_le_mk_iff_right : (a, b₁) ≤ (a, b₂) ↔ b₁ ≤ b₂ :=
  and_iff_right le_rfl


theorem mk_lt_mk_iff_left : (a₁, b) < (a₂, b) ↔ a₁ < a₂ :=
  lt_iff_lt_of_le_iff_le' mk_le_mk_iff_left mk_le_mk_iff_left


theorem mk_lt_mk_iff_right : (a, b₁) < (a, b₂) ↔ b₁ < b₂ :=
  lt_iff_lt_of_le_iff_le' mk_le_mk_iff_right mk_le_mk_iff_right


theorem lt_iff : x < y ↔ x.1 < y.1 ∧ x.2 ≤ y.2 ∨ x.1 ≤ y.1 ∧ x.2 < y.2 := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    x y : Prod α β
    ⊢ Iff (LT.lt x y) (Or (And (LT.lt x.fst y.fst) (LE.le x.snd y.snd)) (And (LE.l …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      x y : Prod α β
      h : LT.lt x y
      ⊢ Or (And (LT.lt x.fst y.fst) (LE.le x.snd y.snd)) (And (LE.le x.fst y.fst) (L …
    -/
  · by_cases h₁ : y.1 ≤ x.1
      /-
        case pos
        α : Type u_2
        β : Type u_3
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        x y : Prod α β
        h : LT.lt x y
        h₁ : LE.le y.fst x.fst
        ⊢ Or (And (LT.lt x.fst y.fst) (LE.le x.snd y.snd)) (And (LE.le x.fst y.fst) (L …
      -/
    · exact Or.inr ⟨h.1.1, LE.le.lt_of_not_le h.1.2 fun h₂ ↦ h.2 ⟨h₁, h₂⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_2
        β : Type u_3
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        x y : Prod α β
        h : LT.lt x y
        h₁ : Not (LE.le y.fst x.fst)
        ⊢ Or (And (LT.lt x.fst y.fst) (LE.le x.snd y.snd)) (And (LE.le x.fst y.fst) (L …
      -/
    · exact Or.inl ⟨LE.le.lt_of_not_le h.1.1 h₁, h.1.2⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      x y : Prod α β
      ⊢ Or (And (LT.lt x.fst y.fst) (LE.le x.snd y.snd)) (And (LE.le x.fst y.fst) (L …
    -/
  · rintro (⟨h₁, h₂⟩ | ⟨h₁, h₂⟩)
      /-
        case refine_2.inl.intro
        α : Type u_2
        β : Type u_3
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        x y : Prod α β
        h₁ : LT.lt x.fst y.fst
        h₂ : LE.le x.snd y.snd
        ⊢ LT.lt x y
      -/
    · exact ⟨⟨h₁.le, h₂⟩, fun h ↦ h₁.not_le h.1⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro
        α : Type u_2
        β : Type u_3
        inst✝¹ : Preorder α
        inst✝ : Preorder β
        x y : Prod α β
        h₁ : LE.le x.fst y.fst
        h₂ : LT.lt x.snd y.snd
        ⊢ LT.lt x y
      -/
    · exact ⟨⟨h₁, h₂.le⟩, fun h ↦ h₂.not_le h.2⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem mk_lt_mk : (a₁, b₁) < (a₂, b₂) ↔ a₁ < a₂ ∧ b₁ ≤ b₂ ∨ a₁ ≤ a₂ ∧ b₁ < b₂ :=
  lt_iff


                                                                               /-
                                                                                 α : Type u_2
                                                                                 β : Type u_3
                                                                                 inst✝¹ : Preorder α
                                                                                 inst✝ : Preorder β
                                                                                 x y : Prod α β
                                                                                 h₁ : LT.lt x.fst y.fst
                                                                                 h₂ : LE.le x.snd y.snd
                                                                                 ⊢ LT.lt x y
                                                                               -/
protected lemma lt_of_lt_of_le (h₁ : x.1 < y.1) (h₂ : x.2 ≤ y.2) : x < y := by simp [lt_iff, *]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

                                                                               /-
                                                                                 α : Type u_2
                                                                                 β : Type u_3
                                                                                 inst✝¹ : Preorder α
                                                                                 inst✝ : Preorder β
                                                                                 x y : Prod α β
                                                                                 h₁ : LE.le x.fst y.fst
                                                                                 h₂ : LT.lt x.snd y.snd
                                                                                 ⊢ LT.lt x y
                                                                               -/
protected lemma lt_of_le_of_lt (h₁ : x.1 ≤ y.1) (h₂ : x.2 < y.2) : x < y := by simp [lt_iff, *]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma mk_lt_mk_of_lt_of_le (h₁ : a₁ < a₂) (h₂ : b₁ ≤ b₂) : (a₁, b₁) < (a₂, b₂) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a₁ a₂ : α
    b₁ b₂ : β
    h₁ : LT.lt a₁ a₂
    h₂ : LE.le b₁ b₂
    ⊢ LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
  -/
  simp [lt_iff, *]
  /-
    🎉 no goals
  -/


lemma mk_lt_mk_of_le_of_lt (h₁ : a₁ ≤ a₂) (h₂ : b₁ < b₂) : (a₁, b₁) < (a₂, b₂) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a₁ a₂ : α
    b₁ b₂ : β
    h₁ : LE.le a₁ a₂
    h₂ : LT.lt b₁ b₂
    ⊢ LT.lt { fst := a₁, snd := b₁ } { fst := a₂, snd := b₂ }
  -/
  simp [lt_iff, *]
  /-
    🎉 no goals
  -/


/-- The pointwise partial order on a product.
    (The lexicographic ordering is defined in `Order.Lexicographic`, and the instances are
    available via the type synonym `α ×ₗ β = α × β`.) -/
instance instPartialOrder (α β : Type*) [PartialOrder α] [PartialOrder β] :
    PartialOrder (α × β) where
  __ := inferInstanceAs (Preorder (α × β))
  le_antisymm := fun _ _ ⟨hac, hbd⟩ ⟨hca, hdb⟩ ↦ Prod.ext (hac.antisymm hca) (hbd.antisymm hdb)


/-- An order is dense if there is an element between any pair of distinct comparable elements. -/
class DenselyOrdered (α : Type*) [LT α] : Prop where
  /-- An order is dense if there is an element between any pair of distinct elements. -/
  dense : ∀ a₁ a₂ : α, a₁ < a₂ → ∃ a, a₁ < a ∧ a < a₂


theorem exists_between [LT α] [DenselyOrdered α] : ∀ {a₁ a₂ : α}, a₁ < a₂ → ∃ a, a₁ < a ∧ a < a₂ :=
  DenselyOrdered.dense _ _


instance OrderDual.denselyOrdered (α : Type*) [LT α] [h : DenselyOrdered α] :
    DenselyOrdered αᵒᵈ :=
  ⟨fun _ _ ha ↦ (@exists_between α _ h _ _ ha).imp fun _ ↦ And.symm⟩


@[simp]
theorem denselyOrdered_orderDual [LT α] : DenselyOrdered αᵒᵈ ↔ DenselyOrdered α :=
      /-
        α : Type u_2
        inst✝ : LT α
        ⊢ DenselyOrdered (OrderDual α) → DenselyOrdered α
      -/
  ⟨by convert @OrderDual.denselyOrdered αᵒᵈ _, @OrderDual.denselyOrdered α _⟩
      /-
        🎉 no goals
      -/


/-- Any ordered subsingleton is densely ordered. Not an instance to avoid a heavy subsingleton
typeclass search. -/
lemma Subsingleton.instDenselyOrdered {X : Type*} [Subsingleton X] [Preorder X] :
    DenselyOrdered X :=
  ⟨fun _ _ h ↦ (not_lt_of_subsingleton h).elim⟩


instance [Preorder α] [Preorder β] [DenselyOrdered α] [DenselyOrdered β] : DenselyOrdered (α × β) :=
  ⟨fun a b ↦ by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      π : ι → Type u_4
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : DenselyOrdered α
      inst✝ : DenselyOrdered β
      a b : Prod α β
      ⊢ LT.lt a b → Exists fun a_2 => And (LT.lt a a_2) (LT.lt a_2 b)
    -/
    simp_rw [Prod.lt_iff]
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      π : ι → Type u_4
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : DenselyOrdered α
      inst✝ : DenselyOrdered β
      a b : Prod α β
      ⊢ Or (And (LT.lt a.fst b.fst) (LE.le a.snd b.snd)) (And (LE.le a.fst b.fst) (L …
    -/
    rintro (⟨h₁, h₂⟩ | ⟨h₁, h₂⟩)
      /-
        case inl.intro
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a b : Prod α β
        h₁ : LT.lt a.fst b.fst
        h₂ : LE.le a.snd b.snd
        ⊢ Exists fun a_1 => And (Or (And (LT.lt a.fst a_1.fst) (LE.le a.snd a_1.snd))  …
      -/
    · obtain ⟨c, ha, hb⟩ := exists_between h₁
      /-
        case inl.intro.intro.intro
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a b : Prod α β
        h₁ : LT.lt a.fst b.fst
        h₂ : LE.le a.snd b.snd
        c : α
        ha : LT.lt a.fst c
        hb : LT.lt c b.fst
        ⊢ Exists fun a_1 => And (Or (And (LT.lt a.fst a_1.fst) (LE.le a.snd a_1.snd))  …
      -/
      exact ⟨(c, _), Or.inl ⟨ha, h₂⟩, Or.inl ⟨hb, le_rfl⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.intro
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a b : Prod α β
        h₁ : LE.le a.fst b.fst
        h₂ : LT.lt a.snd b.snd
        ⊢ Exists fun a_1 => And (Or (And (LT.lt a.fst a_1.fst) (LE.le a.snd a_1.snd))  …
      -/
    · obtain ⟨c, ha, hb⟩ := exists_between h₂
      /-
        case inr.intro.intro.intro
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        π : ι → Type u_4
        inst✝³ : Preorder α
        inst✝² : Preorder β
        inst✝¹ : DenselyOrdered α
        inst✝ : DenselyOrdered β
        a b : Prod α β
        h₁ : LE.le a.fst b.fst
        h₂ : LT.lt a.snd b.snd
        c : β
        ha : LT.lt a.snd c
        hb : LT.lt c b.snd
        ⊢ Exists fun a_1 => And (Or (And (LT.lt a.fst a_1.fst) (LE.le a.snd a_1.snd))  …
      -/
      exact ⟨(_, c), Or.inr ⟨h₁, ha⟩, Or.inr ⟨le_rfl, hb⟩⟩⟩
      /-
        🎉 no goals
      -/


instance [∀ i, Preorder (π i)] [∀ i, DenselyOrdered (π i)] :
    DenselyOrdered (∀ i, π i) :=
  ⟨fun a b ↦ by
    classical
      simp_rw [Pi.lt_def]
      rintro ⟨hab, i, hi⟩
      obtain ⟨c, ha, hb⟩ := exists_between hi
      exact
        ⟨Function.update a i c,
          ⟨le_update_iff.2 ⟨ha.le, fun _ _ ↦ le_rfl⟩, i, by rwa [update_self]⟩,
          update_le_iff.2 ⟨hb.le, fun _ _ ↦ hab _⟩, i, by rwa [update_self]⟩⟩


theorem le_of_forall_le_of_dense [LinearOrder α] [DenselyOrdered α] {a₁ a₂ : α}
    (h : ∀ a, a₂ < a → a₁ ≤ a) : a₁ ≤ a₂ :=
  le_of_not_gt fun ha ↦
    let ⟨a, ha₁, ha₂⟩ := exists_between ha
    lt_irrefl a <| lt_of_lt_of_le ‹a < a₁› (h _ ‹a₂ < a›)


theorem eq_of_le_of_forall_le_of_dense [LinearOrder α] [DenselyOrdered α] {a₁ a₂ : α} (h₁ : a₂ ≤ a₁)
    (h₂ : ∀ a, a₂ < a → a₁ ≤ a) : a₁ = a₂ :=
  le_antisymm (le_of_forall_le_of_dense h₂) h₁


theorem le_of_forall_ge_of_dense [LinearOrder α] [DenselyOrdered α] {a₁ a₂ : α}
    (h : ∀ a₃ < a₁, a₃ ≤ a₂) : a₁ ≤ a₂ :=
  le_of_not_gt fun ha ↦
    let ⟨a, ha₁, ha₂⟩ := exists_between ha
    lt_irrefl a <| lt_of_le_of_lt (h _ ‹a < a₁›) ‹a₂ < a›


theorem eq_of_le_of_forall_ge_of_dense [LinearOrder α] [DenselyOrdered α] {a₁ a₂ : α} (h₁ : a₂ ≤ a₁)
    (h₂ : ∀ a₃ < a₁, a₃ ≤ a₂) : a₁ = a₂ :=
  (le_of_forall_ge_of_dense h₂).antisymm h₁


theorem forall_lt_le_iff [LinearOrder α] [DenselyOrdered α] {a b : α} : (∀ c < a, c ≤ b) ↔ a ≤ b :=
  ⟨le_of_forall_ge_of_dense, fun hab _c hca ↦ hca.le.trans hab⟩


theorem forall_gt_ge_iff [LinearOrder α] [DenselyOrdered α] {a b : α} :
    (∀ c, a < c → b ≤ c) ↔ b ≤ a :=
  forall_lt_le_iff (α := αᵒᵈ)


theorem dense_or_discrete [LinearOrder α] (a₁ a₂ : α) :
    (∃ a, a₁ < a ∧ a < a₂) ∨ (∀ a, a₁ < a → a₂ ≤ a) ∧ ∀ a < a₂, a ≤ a₁ :=
  or_iff_not_imp_left.2 fun h ↦
    ⟨fun a ha₁ ↦ le_of_not_gt fun ha₂ ↦ h ⟨a, ha₁, ha₂⟩,
     fun a ha₂ ↦ le_of_not_gt fun ha₁ ↦ h ⟨a, ha₁, ha₂⟩⟩


/-- If a linear order has no elements `x < y < z`, then it has at most two elements. -/
lemma eq_or_eq_or_eq_of_forall_not_lt_lt [LinearOrder α]
    (h : ∀ ⦃x y z : α⦄, x < y → y < z → False) (x y z : α) : x = y ∨ y = z ∨ x = z := by
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    x y z : α
    ⊢ Or (Eq x y) (Or (Eq y z) (Eq x z))
  -/
  by_contra hne
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    x y z : α
    hne : Not (Or (Eq x y) (Or (Eq y z) (Eq x z)))
    ⊢ False
  -/
  simp only [not_or, ← Ne.eq_def] at hne
  /-
    α : Type u_2
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    x y z : α
    hne : And (Ne x y) (And (Ne y z) (Ne x z))
    ⊢ False
  -/
  rcases hne.1.lt_or_lt with h₁ | h₁ <;>
  /-
    case inl
    α : Type u_2
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    x y z : α
    hne : And (Ne x y) (And (Ne y z) (Ne x z))
    h₁ : LT.lt x y
    ⊢ False
  -/
  rcases hne.2.1.lt_or_lt with h₂ | h₂ <;>
  /-
    case inl.inl
    α : Type u_2
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    x y z : α
    hne : And (Ne x y) (And (Ne y z) (Ne x z))
    h₁ : LT.lt x y
    h₂ : LT.lt y z
    ⊢ False
  -/
  rcases hne.2.2.lt_or_lt with h₃ | h₃
  /-
    case inl.inl.inl
    α : Type u_2
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    x y z : α
    hne : And (Ne x y) (And (Ne y z) (Ne x z))
    h₁ : LT.lt x y
    h₂ : LT.lt y z
    h₃ : LT.lt x z
    ⊢ False
  -/
  exacts [h h₁ h₂, h h₂ h₃, h h₃ h₂, h h₃ h₁, h h₁ h₃, h h₂ h₃, h h₁ h₃, h h₂ h₁]
  /-
    🎉 no goals
  -/


instance instLinearOrder : LinearOrder PUnit where
  le  := fun _ _ ↦ True
  lt  := fun _ _ ↦ False
  max := fun _ _ ↦ unit
  min := fun _ _ ↦ unit
  decidableEq := inferInstance
  decidableLE := fun _ _ ↦ Decidable.isTrue trivial
  decidableLT := fun _ _ ↦ Decidable.isFalse id
                    /-
                      ι : Type u_1
                      α : Type u_2
                      β : Type u_3
                      π : ι → Type u_4
                      a : PUnit.{?u.82773}
                      b : PUnit.{?u.82776}
                      ⊢ ∀ (a : PUnit.{?u.82921 + 1}), LE.le a a
                    -/
  le_refl     := by intros; trivial
                            /-
                              🎉 no goals
                            -/
                    /-
                      ι : Type u_1
                      α : Type u_2
                      β : Type u_3
                      π : ι → Type u_4
                      a : PUnit.{?u.82773}
                      b : PUnit.{?u.82776}
                      ⊢ ∀ (a b c : PUnit.{?u.82921 + 1}), LE.le a b → LE.le b c → LE.le a c
                    -/
  le_trans    := by intros; trivial
                            /-
                              🎉 no goals
                            -/
                    /-
                      ι : Type u_1
                      α : Type u_2
                      β : Type u_3
                      π : ι → Type u_4
                      a : PUnit.{?u.82773}
                      b : PUnit.{?u.82776}
                      ⊢ ∀ (a b : PUnit.{?u.82921 + 1}), Or (LE.le a b) (LE.le b a)
                    -/
                    /-
                      ι : Type u_1
                      α : Type u_2
                      β : Type u_3
                      π : ι → Type u_4
                      a : PUnit.{?u.82773}
                      b : PUnit.{?u.82776}
                      ⊢ ∀ (a b : PUnit.{?u.82921 + 1}), LE.le a b → LE.le b a → Eq a b
                    -/
                         /-
                           ι : Type u_1
                           α : Type u_2
                           β : Type u_3
                           π : ι → Type u_4
                           a : PUnit.{?u.82773}
                           b : PUnit.{?u.82776}
                           ⊢ ∀ (a b : PUnit.{?u.82921 + 1}), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le …
                         -/
  le_total    := by intros; exact Or.inl trivial
                         /-
                           🎉 no goals
                         -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
  le_antisymm := by intros; rfl
  lt_iff_le_not_le := by simp only [not_true, and_false, forall_const]


theorem max_eq : max a b = unit :=
  rfl


theorem min_eq : min a b = unit :=
  rfl


protected theorem le : a ≤ b :=
  trivial


theorem not_lt : ¬a < b :=
  not_false


instance : DenselyOrdered PUnit :=
  ⟨fun _ _ ↦ False.elim⟩


/-- Propositions form a complete boolean algebra, where the `≤` relation is given by implication. -/
instance Prop.le : LE Prop :=
  ⟨(· → ·)⟩


@[simp]
theorem le_Prop_eq : ((· ≤ ·) : Prop → Prop → Prop) = (· → ·) :=
  rfl


theorem subrelation_iff_le {r s : α → α → Prop} : Subrelation r s ↔ r ≤ s :=
  Iff.rfl


instance Prop.partialOrder : PartialOrder Prop where
  __ := Prop.le
  le_refl _ := id
  le_trans _ _ _ f g := g ∘ f
  le_antisymm _ _ Hab Hba := propext ⟨Hab, Hba⟩


/-- Type synonym to create an instance of `LinearOrder` from a `PartialOrder` and `IsTotal α (≤)` -/
def AsLinearOrder (α : Type*)  :=
  α


instance [Inhabited α] : Inhabited (AsLinearOrder α) :=
  ⟨(default : α)⟩


noncomputable instance AsLinearOrder.linearOrder [PartialOrder α] [IsTotal α (· ≤ ·)] :
    LinearOrder (AsLinearOrder α) where
  __ := inferInstanceAs (PartialOrder α)
  le_total := @total_of α (· ≤ ·) _
  decidableLE := Classical.decRel _


@[to_additive dite_nonneg]
lemma one_le_dite [LE α] (ha : ∀ h, 1 ≤ a h) (hb : ∀ h, 1 ≤ b h) : 1 ≤ dite p a b := by
  /-
    α : Type u_2
    inst✝² : One α
    p : Prop
    inst✝¹ : Decidable p
    a : p → α
    b : Not p → α
    inst✝ : LE α
    ha : ∀ (h : p), LE.le 1 (a h)
    hb : ∀ (h : Not p), LE.le 1 (b h)
    ⊢ LE.le 1 (dite p a b)
  -/
  split; exacts [ha ‹_›, hb ‹_›]
         /-
           🎉 no goals
         -/


@[to_additive]
lemma dite_le_one [LE α] (ha : ∀ h, a h ≤ 1) (hb : ∀ h, b h ≤ 1) : dite p a b ≤ 1 := by
  /-
    α : Type u_2
    inst✝² : One α
    p : Prop
    inst✝¹ : Decidable p
    a : p → α
    b : Not p → α
    inst✝ : LE α
    ha : ∀ (h : p), LE.le (a h) 1
    hb : ∀ (h : Not p), LE.le (b h) 1
    ⊢ LE.le (dite p a b) 1
  -/
  split; exacts [ha ‹_›, hb ‹_›]
         /-
           🎉 no goals
         -/


@[to_additive dite_pos]
lemma one_lt_dite [LT α] (ha : ∀ h, 1 < a h) (hb : ∀ h, 1 < b h) : 1 < dite p a b := by
  /-
    α : Type u_2
    inst✝² : One α
    p : Prop
    inst✝¹ : Decidable p
    a : p → α
    b : Not p → α
    inst✝ : LT α
    ha : ∀ (h : p), LT.lt 1 (a h)
    hb : ∀ (h : Not p), LT.lt 1 (b h)
    ⊢ LT.lt 1 (dite p a b)
  -/
  split; exacts [ha ‹_›, hb ‹_›]
         /-
           🎉 no goals
         -/


@[to_additive]
lemma dite_lt_one [LT α] (ha : ∀ h, a h < 1) (hb : ∀ h, b h < 1) : dite p a b < 1 := by
  /-
    α : Type u_2
    inst✝² : One α
    p : Prop
    inst✝¹ : Decidable p
    a : p → α
    b : Not p → α
    inst✝ : LT α
    ha : ∀ (h : p), LT.lt (a h) 1
    hb : ∀ (h : Not p), LT.lt (b h) 1
    ⊢ LT.lt (dite p a b) 1
  -/
  split; exacts [ha ‹_›, hb ‹_›]
         /-
           🎉 no goals
         -/


@[to_additive ite_nonneg]
                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝² : One α
                                                                          p : Prop
                                                                          inst✝¹ : Decidable p
                                                                          a b : α
                                                                          inst✝ : LE α
                                                                          ha : LE.le 1 a
                                                                          hb : LE.le 1 b
                                                                          ⊢ LE.le 1 (ite p a b)
                                                                        -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma one_le_ite [LE α] (ha : 1 ≤ a) (hb : 1 ≤ b) : 1 ≤ ite p a b := by split <;> assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[to_additive]
                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝² : One α
                                                                          p : Prop
                                                                          inst✝¹ : Decidable p
                                                                          a b : α
                                                                          inst✝ : LE α
                                                                          ha : LE.le a 1
                                                                          hb : LE.le b 1
                                                                          ⊢ LE.le (ite p a b) 1
                                                                        -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma ite_le_one [LE α] (ha : a ≤ 1) (hb : b ≤ 1) : ite p a b ≤ 1 := by split <;> assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[to_additive ite_pos]
                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝² : One α
                                                                          p : Prop
                                                                          inst✝¹ : Decidable p
                                                                          a b : α
                                                                          inst✝ : LT α
                                                                          ha : LT.lt 1 a
                                                                          hb : LT.lt 1 b
                                                                          ⊢ LT.lt 1 (ite p a b)
                                                                        -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma one_lt_ite [LT α] (ha : 1 < a) (hb : 1 < b) : 1 < ite p a b := by split <;> assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[to_additive]
                                                                        /-
                                                                          α : Type u_2
                                                                          inst✝² : One α
                                                                          p : Prop
                                                                          inst✝¹ : Decidable p
                                                                          a b : α
                                                                          inst✝ : LT α
                                                                          ha : LT.lt a 1
                                                                          hb : LT.lt b 1
                                                                          ⊢ LT.lt (ite p a b) 1
                                                                        -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma ite_lt_one [LT α] (ha : a < 1) (hb : b < 1) : ite p a b < 1 := by split <;> assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


