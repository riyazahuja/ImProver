/-- Syntax typeclass for a betweenness relation. -/
class Btw (α : Type*) where
  /-- Betweenness for circular orders. `btw a b c` states that `b` is between `a` and `c` (in that
  order). -/
  btw : α → α → α → Prop


/-- Syntax typeclass for a strict betweenness relation. -/
class SBtw (α : Type*) where
  /-- Strict betweenness for circular orders. `sbtw a b c` states that `b` is strictly between `a`
  and `c` (in that order). -/
  sbtw : α → α → α → Prop


/-- A circular preorder is the analogue of a preorder where you can loop around. `≤` and `<` are
replaced by ternary relations `btw` and `sbtw`. `btw` is reflexive and cyclic. `sbtw` is transitive.
-/
class CircularPreorder (α : Type*) extends Btw α, SBtw α where
  /-- `a` is between `a` and `a`. -/
  btw_refl (a : α) : btw a a a
  /-- If `b` is between `a` and `c`, then `c` is between `b` and `a`.
  This is motivated by imagining three points on a circle. -/
  btw_cyclic_left {a b c : α} : btw a b c → btw b c a
  sbtw := fun a b c => btw a b c ∧ ¬btw c b a
  /-- Strict betweenness is given by betweenness in one direction and non-betweenness in the other.

  I.e., if `b` is between `a` and `c` but not between `c` and `a`, then we say `b` is strictly
  between `a` and `c`. -/
  sbtw_iff_btw_not_btw {a b c : α} : sbtw a b c ↔ btw a b c ∧ ¬btw c b a := by intros; rfl
  /-- For any fixed `c`, `fun a b ↦ sbtw a b c` is a transitive relation.

  I.e., given `a` `b` `d` `c` in that "order", if we have `b` strictly between `a` and `c`, and `d`
  strictly between `b` and `c`, then `d` is strictly between `a` and `c`. -/
  sbtw_trans_left {a b c d : α} : sbtw a b c → sbtw b d c → sbtw a d c


/-- A circular partial order is the analogue of a partial order where you can loop around. `≤` and
`<` are replaced by ternary relations `btw` and `sbtw`. `btw` is reflexive, cyclic and
antisymmetric. `sbtw` is transitive. -/
class CircularPartialOrder (α : Type*) extends CircularPreorder α where
  /-- If `b` is between `a` and `c` and also between `c` and `a`, then at least one pair of points
  among `a`, `b`, `c` are identical. -/
  btw_antisymm {a b c : α} : btw a b c → btw c b a → a = b ∨ b = c ∨ c = a


/-- A circular order is the analogue of a linear order where you can loop around. `≤` and `<` are
replaced by ternary relations `btw` and `sbtw`. `btw` is reflexive, cyclic, antisymmetric and total.
`sbtw` is transitive. -/
class CircularOrder (α : Type*) extends CircularPartialOrder α where
  /-- For any triple of points, the second is between the other two one way or another. -/
  btw_total : ∀ a b c : α, btw a b c ∨ btw c b a


theorem btw_rfl {a : α} : btw a a a :=
  btw_refl _

-- TODO: `alias` creates a def instead of a lemma (because `btw_cyclic_left` is a def).
-- alias btw_cyclic_left        ← Btw.btw.cyclic_left

theorem Btw.btw.cyclic_left {a b c : α} (h : btw a b c) : btw b c a :=
  btw_cyclic_left h


theorem btw_cyclic_right {a b c : α} (h : btw a b c) : btw c a b :=
  h.cyclic_left.cyclic_left


alias Btw.btw.cyclic_right := btw_cyclic_right


/-- The order of the `↔` has been chosen so that `rw [btw_cyclic]` cycles to the right while
`rw [← btw_cyclic]` cycles to the left (thus following the prepended arrow). -/
theorem btw_cyclic {a b c : α} : btw a b c ↔ btw c a b :=
  ⟨btw_cyclic_right, btw_cyclic_left⟩


theorem sbtw_iff_btw_not_btw {a b c : α} : sbtw a b c ↔ btw a b c ∧ ¬btw c b a :=
  CircularPreorder.sbtw_iff_btw_not_btw


theorem btw_of_sbtw {a b c : α} (h : sbtw a b c) : btw a b c :=
  (sbtw_iff_btw_not_btw.1 h).1


alias SBtw.sbtw.btw := btw_of_sbtw


theorem not_btw_of_sbtw {a b c : α} (h : sbtw a b c) : ¬btw c b a :=
  (sbtw_iff_btw_not_btw.1 h).2


alias SBtw.sbtw.not_btw := not_btw_of_sbtw


theorem not_sbtw_of_btw {a b c : α} (h : btw a b c) : ¬sbtw c b a := fun h' => h'.not_btw h


alias Btw.btw.not_sbtw := not_sbtw_of_btw


theorem sbtw_of_btw_not_btw {a b c : α} (habc : btw a b c) (hcba : ¬btw c b a) : sbtw a b c :=
  sbtw_iff_btw_not_btw.2 ⟨habc, hcba⟩


alias Btw.btw.sbtw_of_not_btw := sbtw_of_btw_not_btw


theorem sbtw_cyclic_left {a b c : α} (h : sbtw a b c) : sbtw b c a :=
  h.btw.cyclic_left.sbtw_of_not_btw fun h' => h.not_btw h'.cyclic_left


alias SBtw.sbtw.cyclic_left := sbtw_cyclic_left


theorem sbtw_cyclic_right {a b c : α} (h : sbtw a b c) : sbtw c a b :=
  h.cyclic_left.cyclic_left


alias SBtw.sbtw.cyclic_right := sbtw_cyclic_right


/-- The order of the `↔` has been chosen so that `rw [sbtw_cyclic]` cycles to the right while
`rw [← sbtw_cyclic]` cycles to the left (thus following the prepended arrow). -/
theorem sbtw_cyclic {a b c : α} : sbtw a b c ↔ sbtw c a b :=
  ⟨sbtw_cyclic_right, sbtw_cyclic_left⟩

-- TODO: `alias` creates a def instead of a lemma (because `sbtw_trans_left` is a def).
-- alias btw_trans_left        ← SBtw.sbtw.trans_left

theorem SBtw.sbtw.trans_left {a b c d : α} (h : sbtw a b c) : sbtw b d c → sbtw a d c :=
  sbtw_trans_left h


theorem sbtw_trans_right {a b c d : α} (hbc : sbtw a b c) (hcd : sbtw a c d) : sbtw a b d :=
  (hbc.cyclic_left.trans_left hcd.cyclic_left).cyclic_right


alias SBtw.sbtw.trans_right := sbtw_trans_right


theorem sbtw_asymm {a b c : α} (h : sbtw a b c) : ¬sbtw c b a :=
  h.btw.not_sbtw


alias SBtw.sbtw.not_sbtw := sbtw_asymm


theorem sbtw_irrefl_left_right {a b : α} : ¬sbtw a b a := fun h => h.not_btw h.btw


theorem sbtw_irrefl_left {a b : α} : ¬sbtw a a b := fun h => sbtw_irrefl_left_right h.cyclic_left


theorem sbtw_irrefl_right {a b : α} : ¬sbtw a b b := fun h => sbtw_irrefl_left_right h.cyclic_right


theorem sbtw_irrefl (a : α) : ¬sbtw a a a :=
  sbtw_irrefl_left_right


theorem Btw.btw.antisymm {a b c : α} (h : btw a b c) : btw c b a → a = b ∨ b = c ∨ c = a :=
  btw_antisymm h


theorem btw_refl_left_right (a b : α) : btw a b a :=
  or_self_iff.1 (btw_total a b a)


theorem btw_rfl_left_right {a b : α} : btw a b a :=
  btw_refl_left_right _ _


theorem btw_refl_left (a b : α) : btw a a b :=
  btw_rfl_left_right.cyclic_right


theorem btw_rfl_left {a b : α} : btw a a b :=
  btw_refl_left _ _


theorem btw_refl_right (a b : α) : btw a b b :=
  btw_rfl_left_right.cyclic_left


theorem btw_rfl_right {a b : α} : btw a b b :=
  btw_refl_right _ _


theorem sbtw_iff_not_btw {a b c : α} : sbtw a b c ↔ ¬btw c b a := by
  /-
    α : Type u_1
    inst✝ : CircularOrder α
    a b c : α
    ⊢ Iff (SBtw.sbtw a b c) (Not (Btw.btw c b a))
  -/
  rw [sbtw_iff_btw_not_btw]
  /-
    α : Type u_1
    inst✝ : CircularOrder α
    a b c : α
    ⊢ Iff (And (Btw.btw a b c) (Not (Btw.btw c b a))) (Not (Btw.btw c b a))
  -/
  exact and_iff_right_of_imp (btw_total _ _ _).resolve_left
  /-
    🎉 no goals
  -/


theorem btw_iff_not_sbtw {a b c : α} : btw a b c ↔ ¬sbtw c b a :=
  iff_not_comm.1 sbtw_iff_not_btw


/-- Closed-closed circular interval -/
def cIcc (a b : α) : Set α :=
  { x | btw a x b }


/-- Open-open circular interval -/
def cIoo (a b : α) : Set α :=
  { x | sbtw a x b }


@[simp]
theorem mem_cIcc {a b x : α} : x ∈ cIcc a b ↔ btw a x b :=
  Iff.rfl


@[simp]
theorem mem_cIoo {a b x : α} : x ∈ cIoo a b ↔ sbtw a x b :=
  Iff.rfl


theorem left_mem_cIcc (a b : α) : a ∈ cIcc a b :=
  btw_rfl_left


theorem right_mem_cIcc (a b : α) : b ∈ cIcc a b :=
  btw_rfl_right


theorem compl_cIcc {a b : α} : (cIcc a b)ᶜ = cIoo b a := by
  /-
    α : Type u_1
    inst✝ : CircularOrder α
    a b : α
    ⊢ Eq (HasCompl.compl (Set.cIcc a b)) (Set.cIoo b a)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : CircularOrder α
    a b x✝ : α
    ⊢ Iff (Membership.mem (HasCompl.compl (Set.cIcc a b)) x✝) (Membership.mem (Set …
  -/
  rw [Set.mem_cIoo, sbtw_iff_not_btw, cIcc, mem_compl_iff, mem_setOf]
  /-
    🎉 no goals
  -/


theorem compl_cIoo {a b : α} : (cIoo a b)ᶜ = cIcc b a := by
  /-
    α : Type u_1
    inst✝ : CircularOrder α
    a b : α
    ⊢ Eq (HasCompl.compl (Set.cIoo a b)) (Set.cIcc b a)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : CircularOrder α
    a b x✝ : α
    ⊢ Iff (Membership.mem (HasCompl.compl (Set.cIoo a b)) x✝) (Membership.mem (Set …
  -/
  rw [Set.mem_cIcc, btw_iff_not_sbtw, cIoo, mem_compl_iff, mem_setOf]
  /-
    🎉 no goals
  -/


/-- The betweenness relation obtained from "looping around" `≤`.
See note [reducible non-instances]. -/
abbrev LE.toBtw (α : Type*) [LE α] : Btw α where
  btw a b c := a ≤ b ∧ b ≤ c ∨ b ≤ c ∧ c ≤ a ∨ c ≤ a ∧ a ≤ b


/-- The strict betweenness relation obtained from "looping around" `<`.
See note [reducible non-instances]. -/
abbrev LT.toSBtw (α : Type*) [LT α] : SBtw α where
  sbtw a b c := a < b ∧ b < c ∨ b < c ∧ c < a ∨ c < a ∧ a < b


/-- The circular preorder obtained from "looping around" a preorder.
See note [reducible non-instances]. -/
abbrev Preorder.toCircularPreorder (α : Type*) [Preorder α] : CircularPreorder α where
  btw a b c := a ≤ b ∧ b ≤ c ∨ b ≤ c ∧ c ≤ a ∨ c ≤ a ∧ a ≤ b
  sbtw a b c := a < b ∧ b < c ∨ b < c ∧ c < a ∨ c < a ∧ a < b
  btw_refl _ := Or.inl ⟨le_rfl, le_rfl⟩
  btw_cyclic_left {a b c} h := by
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      h : Btw.btw a b c
      ⊢ Btw.btw b c a
    -/
    dsimp
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      h : Btw.btw a b c
      ⊢ Or (And (LE.le b c) (LE.le c a)) (Or (And (LE.le c a) (LE.le a b)) (And (LE. …
    -/
    rwa [← or_assoc, or_comm]
    /-
      🎉 no goals
    -/
  sbtw_trans_left {a b c d} := by
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c d : α
      ⊢ SBtw.sbtw a b c → SBtw.sbtw b d c → SBtw.sbtw a d c
    -/
    rintro (⟨hab, hbc⟩ | ⟨hbc, hca⟩ | ⟨hca, hab⟩) (⟨hbd, hdc⟩ | ⟨hdc, hcb⟩ | ⟨hcb, hbd⟩)
      /-
        case inl.intro.inl.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hab : LT.lt a b
        hbc : LT.lt b c
        hbd : LT.lt b d
        hdc : LT.lt d c
        ⊢ SBtw.sbtw a d c
      -/
    · exact Or.inl ⟨hab.trans hbd, hdc⟩
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.inr.inl.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hab : LT.lt a b
        hbc : LT.lt b c
        hdc : LT.lt d c
        hcb : LT.lt c b
        ⊢ SBtw.sbtw a d c
      -/
    · exact (hbc.not_lt hcb).elim
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.inr.inr.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hab : LT.lt a b
        hbc : LT.lt b c
        hcb : LT.lt c b
        hbd : LT.lt b d
        ⊢ SBtw.sbtw a d c
      -/
    · exact (hbc.not_lt hcb).elim
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      ⊢ Iff (SBtw.sbtw a b c) (And (Btw.btw a b c) (Not (Btw.btw c b a)))
    -/
      /-
        case inr.inl.intro.inl.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hbc : LT.lt b c
        hca : LT.lt c a
        hbd : LT.lt b d
        hdc : LT.lt d c
        ⊢ SBtw.sbtw a d c
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      ⊢ Iff (Or (And (And (LE.le a b) (Not (LE.le b a))) (And (LE.le b c) (Not (LE.l …
    -/
    · exact Or.inr (Or.inl ⟨hdc, hca⟩)
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      h1 : LE.le a b → LE.le b c → LE.le a c
      ⊢ Iff (Or (And (And (LE.le a b) (Not (LE.le b a))) (And (LE.le b c) (Not (LE.l …
    -/
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      h1 : LE.le a b → LE.le b c → LE.le a c
      h2 : LE.le b c → LE.le c a → LE.le b a
      ⊢ Iff (Or (And (And (LE.le a b) (Not (LE.le b a))) (And (LE.le b c) (Not (LE.l …
    -/
      /-
        case inr.inl.intro.inr.inl.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hbc : LT.lt b c
        hca : LT.lt c a
        hdc : LT.lt d c
        hcb : LT.lt c b
        ⊢ SBtw.sbtw a d c
      -/
    · exact Or.inr (Or.inl ⟨hdc, hca⟩)
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      h1 : LE.le a b → LE.le b c → LE.le a c
      h2 : LE.le b c → LE.le c a → LE.le b a
      h3 : LE.le c a → LE.le a b → LE.le c b
      ⊢ Iff (Or (And (And (LE.le a b) (Not (LE.le b a))) (And (LE.le b c) (Not (LE.l …
    -/
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      ⊢ (LE.le a b → LE.le b c → LE.le a c) → (LE.le b c → LE.le c a → LE.le b a) →  …
    -/
      /-
        case inr.inl.intro.inr.inr.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hbc : LT.lt b c
        hca : LT.lt c a
        hcb : LT.lt c b
        hbd : LT.lt b d
        ⊢ SBtw.sbtw a d c
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      p1 : Prop
      ⊢ (p1 → LE.le b c → LE.le a c) → (LE.le b c → LE.le c a → LE.le b a) → (LE.le  …
    -/
    · exact (hbc.not_lt hcb).elim
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      p1 p2 : Prop
      ⊢ (p1 → LE.le b c → LE.le a c) → (LE.le b c → LE.le c a → p2) → (LE.le c a → p …
    -/
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      p1 p2 p3 : Prop
      ⊢ (p1 → LE.le b c → p3) → (LE.le b c → LE.le c a → p2) → (LE.le c a → p1 → LE. …
    -/
      /-
        case inr.inr.intro.inl.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hca : LT.lt c a
        hab : LT.lt a b
        hbd : LT.lt b d
        hdc : LT.lt d c
        ⊢ SBtw.sbtw a d c
      -/
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      p1 p2 p3 p4 : Prop
      ⊢ (p1 → LE.le b c → p3) → (LE.le b c → p4 → p2) → (p4 → p1 → LE.le c b) → Iff  …
    -/
    · exact Or.inr (Or.inl ⟨hdc, hca⟩)
    /-
      α : Type u_1
      inst✝ : Preorder α
      a b c : α
      p1 p2 p3 p4 p5 : Prop
      ⊢ (p1 → p5 → p3) → (p5 → p4 → p2) → (p4 → p1 → LE.le c b) → Iff (Or (And (And  …
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
        case inr.inr.intro.inr.inl.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hca : LT.lt c a
        hab : LT.lt a b
        hdc : LT.lt d c
        hcb : LT.lt c b
        ⊢ SBtw.sbtw a d c
      -/
    · exact Or.inr (Or.inl ⟨hdc, hca⟩)
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.intro.inr.inr.intro
        α : Type u_1
        inst✝ : Preorder α
        a b c d : α
        hca : LT.lt c a
        hab : LT.lt a b
        hcb : LT.lt c b
        hbd : LT.lt b d
        ⊢ SBtw.sbtw a d c
      -/
    · exact Or.inr (Or.inr ⟨hca, hab.trans hbd⟩)
      /-
        🎉 no goals
      -/
  sbtw_iff_btw_not_btw {a b c} := by
    simp_rw [lt_iff_le_not_le]
    have h1 := le_trans a b c
    have h2 := le_trans b c a
    have h3 := le_trans c a b
    -- Porting note: was `tauto`, but this is a much faster tactic proof
    revert h1 h2 h3
    generalize (a ≤ b) = p1
    generalize (b ≤ a) = p2
    generalize (a ≤ c) = p3
    generalize (c ≤ a) = p4
    generalize (b ≤ c) = p5
    by_cases p1 <;> by_cases p2 <;> by_cases p3 <;> by_cases p4 <;> by_cases p5 <;> simp [*]


/-- The circular partial order obtained from "looping around" a partial order.
See note [reducible non-instances]. -/
abbrev PartialOrder.toCircularPartialOrder (α : Type*) [PartialOrder α] : CircularPartialOrder α :=
  { Preorder.toCircularPreorder α with
    btw_antisymm := fun {a b c} => by
      /-
        α : Type u_1
        inst✝ : PartialOrder α
        a b c : α
        ⊢ Btw.btw a b c → Btw.btw c b a → Or (Eq a b) (Or (Eq b c) (Eq c a))
      -/
      rintro (⟨hab, hbc⟩ | ⟨hbc, hca⟩ | ⟨hca, hab⟩) (⟨hcb, hba⟩ | ⟨hba, hac⟩ | ⟨hac, hcb⟩)
        /-
          case inl.intro.inl.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          hcb : LE.le c b
          hba : LE.le b a
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inl (hab.antisymm hba)
        /-
          🎉 no goals
        -/
        /-
          case inl.intro.inr.inl.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          hba : LE.le b a
          hac : LE.le a c
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inl (hab.antisymm hba)
        /-
          🎉 no goals
        -/
        /-
          case inl.intro.inr.inr.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          hac : LE.le a c
          hcb : LE.le c b
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inr (Or.inl <| hbc.antisymm hcb)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl.intro.inl.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hbc : LE.le b c
          hca : LE.le c a
          hcb : LE.le c b
          hba : LE.le b a
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inr (Or.inl <| hbc.antisymm hcb)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl.intro.inr.inl.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hbc : LE.le b c
          hca : LE.le c a
          hba : LE.le b a
          hac : LE.le a c
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inr (Or.inr <| hca.antisymm hac)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl.intro.inr.inr.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hbc : LE.le b c
          hca : LE.le c a
          hac : LE.le a c
          hcb : LE.le c b
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inr (Or.inl <| hbc.antisymm hcb)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.intro.inl.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hca : LE.le c a
          hab : LE.le a b
          hcb : LE.le c b
          hba : LE.le b a
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inl (hab.antisymm hba)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.intro.inr.inl.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hca : LE.le c a
          hab : LE.le a b
          hba : LE.le b a
          hac : LE.le a c
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inl (hab.antisymm hba)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.intro.inr.inr.intro
          α : Type u_1
          inst✝ : PartialOrder α
          a b c : α
          hca : LE.le c a
          hab : LE.le a b
          hac : LE.le a c
          hcb : LE.le c b
          ⊢ Or (Eq a b) (Or (Eq b c) (Eq c a))
        -/
      · exact Or.inr (Or.inr <| hca.antisymm hac) }
        /-
          🎉 no goals
        -/


/-- The circular order obtained from "looping around" a linear order.
See note [reducible non-instances]. -/
abbrev LinearOrder.toCircularOrder (α : Type*) [LinearOrder α] : CircularOrder α :=
  { PartialOrder.toCircularPartialOrder α with
    btw_total := fun a b c => by
      /-
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : α
        ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
      -/
      rcases le_total a b with hab | hba <;> rcases le_total b c with hbc | hcb <;>
        /-
          case inl.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
        rcases le_total c a with hca | hac
        /-
          case inl.inl.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          hca : LE.le c a
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inl (Or.inl ⟨hab, hbc⟩)
        /-
          🎉 no goals
        -/
        /-
          case inl.inl.inr
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hab : LE.le a b
          hbc : LE.le b c
          hac : LE.le a c
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inl (Or.inl ⟨hab, hbc⟩)
        /-
          🎉 no goals
        -/
        /-
          case inl.inr.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hab : LE.le a b
          hcb : LE.le c b
          hca : LE.le c a
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inl (Or.inr <| Or.inr ⟨hca, hab⟩)
        /-
          🎉 no goals
        -/
        /-
          case inl.inr.inr
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hab : LE.le a b
          hcb : LE.le c b
          hac : LE.le a c
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inr (Or.inr <| Or.inr ⟨hac, hcb⟩)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hba : LE.le b a
          hbc : LE.le b c
          hca : LE.le c a
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inl (Or.inr <| Or.inl ⟨hbc, hca⟩)
        /-
          🎉 no goals
        -/
        /-
          case inr.inl.inr
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hba : LE.le b a
          hbc : LE.le b c
          hac : LE.le a c
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inr (Or.inr <| Or.inl ⟨hba, hac⟩)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hba : LE.le b a
          hcb : LE.le c b
          hca : LE.le c a
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inr (Or.inl ⟨hcb, hba⟩)
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.inr
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : α
          hba : LE.le b a
          hcb : LE.le c b
          hac : LE.le a c
          ⊢ Or (Btw.btw a b c) (Btw.btw c b a)
        -/
      · exact Or.inr (Or.inr <| Or.inl ⟨hba, hac⟩) }
        /-
          🎉 no goals
        -/


instance btw (α : Type*) [Btw α] : Btw αᵒᵈ :=
  ⟨fun a b c : α => Btw.btw c b a⟩


instance sbtw (α : Type*) [SBtw α] : SBtw αᵒᵈ :=
  ⟨fun a b c : α => SBtw.sbtw c b a⟩


instance circularPreorder (α : Type*) [CircularPreorder α] : CircularPreorder αᵒᵈ :=
  { OrderDual.btw α,
    OrderDual.sbtw α with
    btw_refl := fun _ => @btw_refl α _ _
    btw_cyclic_left := fun {_ _ _} => @btw_cyclic_right α _ _ _ _
    sbtw_trans_left := fun {_ _ _ _} habc hbdc => hbdc.trans_right habc
    sbtw_iff_btw_not_btw := fun {a b c} => @sbtw_iff_btw_not_btw α _ c b a }


instance circularPartialOrder (α : Type*) [CircularPartialOrder α] : CircularPartialOrder αᵒᵈ :=
  { OrderDual.circularPreorder α with
    btw_antisymm := fun {_ _ _} habc hcba => @btw_antisymm α _ _ _ _ hcba habc }


instance (α : Type*) [CircularOrder α] : CircularOrder αᵒᵈ :=
  { OrderDual.circularPartialOrder α with
    btw_total := fun {a b c} => @btw_total α _ c b a }


