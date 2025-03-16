/-- The subtype of natural numbers `i` which have the property that
no `j` less than `i` satisfies `p`. This is an initial segment of the
natural numbers, up to and including the first value satisfying `p`.

We will be particularly interested in the case where there exists a value
satisfying `p`, because in this case the `>` relation is well-founded. -/
abbrev Upto (p : ℕ → Prop) : Type :=
  { i : ℕ // ∀ j < i, ¬p j }


/-- Lift the "greater than" relation on natural numbers to `Nat.Upto`. -/
protected def GT (p) (x y : Upto p) : Prop :=
  x.1 > y.1


instance : LT (Upto p) :=
  ⟨fun x y => x.1 < y.1⟩


/-- The "greater than" relation on `Upto p` is well founded if (and only if) there exists a value
satisfying `p`. -/
protected theorem wf : (∃ x, p x) → WellFounded (Upto.GT p)
  | ⟨x, h⟩ => by
    suffices Upto.GT p = InvImage (· < ·) fun y : Nat.Upto p => x - y.val by
      rw [this]
      exact (measure _).wf
    /-
      p : Nat → Prop
      x : Nat
      h : p x
      ⊢ Eq (Nat.Upto.GT p) (InvImage (fun x1 x2 => LT.lt x1 x2) fun y => HSub.hSub x …
    -/
    ext ⟨a, ha⟩ ⟨b, _⟩
    /-
      case h.mk.h.mk.a
      p : Nat → Prop
      x : Nat
      h : p x
      a : Nat
      ha : ∀ (j : Nat), LT.lt j a → Not (p j)
      b : Nat
      property✝ : ∀ (j : Nat), LT.lt j b → Not (p j)
      ⊢ Iff (Nat.Upto.GT p ⟨a, ha⟩ ⟨b, property✝⟩) (InvImage (fun x1 x2 => LT.lt x1  …
    -/
    dsimp [InvImage, Upto.GT]
    /-
      case h.mk.h.mk.a
      p : Nat → Prop
      x : Nat
      h : p x
      a : Nat
      ha : ∀ (j : Nat), LT.lt j a → Not (p j)
      b : Nat
      property✝ : ∀ (j : Nat), LT.lt j b → Not (p j)
      ⊢ Iff (GT.gt a b) (LT.lt (HSub.hSub x a) (HSub.hSub x b))
    -/
    rw [tsub_lt_tsub_iff_left_of_le (le_of_not_lt fun h' => ha _ h' h)]
    /-
      🎉 no goals
    -/


/-- Zero is always a member of `Nat.Upto p` because it has no predecessors. -/
def zero : Nat.Upto p :=
  ⟨0, fun _ h => False.elim (Nat.not_lt_zero _ h)⟩


/-- The successor of `n` is in `Nat.Upto p` provided that `n` doesn't satisfy `p`. -/
def succ (x : Nat.Upto p) (h : ¬p x.val) : Nat.Upto p :=
  ⟨x.val.succ, fun j h' => by
    /-
      p : Nat → Prop
      x : Nat.Upto p
      h : Not (p ↑x)
      j : Nat
      h' : LT.lt j (↑x).succ
      ⊢ Not (p j)
    -/
    rcases Nat.lt_succ_iff_lt_or_eq.1 h' with (h' | rfl) <;> [exact x.2 _ h'; exact h]⟩
    /-
      🎉 no goals
    -/


