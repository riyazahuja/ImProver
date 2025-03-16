/-- `Fix α` provides a `fix` operator to define recursive computation
via the fixed point of function of type `α → α`. -/
class Fix (α : Type*) where
  /-- `fix f` represents the computation of a fixed point for `f`. -/
  fix : (α → α) → α


/-- A series of successive, finite approximation of the fixed point of `f`, defined by
`approx f n = f^[n] ⊥`. The limit of this chain is the fixed point of `f`. -/
def Fix.approx : Stream' (∀ a, Part (β a))
  | 0 => ⊥
  | Nat.succ i => f (Fix.approx i)


/-- loop body for finding the fixed point of `f` -/
def fixAux {p : ℕ → Prop} (i : Nat.Upto p) (g : ∀ j : Nat.Upto p, i < j → ∀ a, Part (β a)) :
    ∀ a, Part (β a) :=
  f fun x : α => (assert ¬p i.val) fun h : ¬p i.val => g (i.succ h) (Nat.lt_succ_self _) x


/-- The least fixed point of `f`.

If `f` is a continuous function (according to complete partial orders),
it satisfies the equations:

  1. `fix f = f (fix f)`          (is a fixed point)
  2. `∀ X, f X ≤ X → fix f ≤ X`   (least fixed point)
-/
protected def fix (x : α) : Part (β x) :=
  (Part.assert (∃ i, (Fix.approx f i x).Dom)) fun h =>
    WellFounded.fix.{1} (Nat.Upto.wf h) (fixAux f) Nat.Upto.zero x


open Classical in
protected theorem fix_def {x : α} (h' : ∃ i, (Fix.approx f i x).Dom) :
    Part.fix f x = Fix.approx f (Nat.succ (Nat.find h')) x := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    ⊢ Eq (Part.fix f x) (Part.Fix.approx f (Nat.find h').succ x)
  -/
  let p := fun i : ℕ => (Fix.approx f i x).Dom
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    ⊢ Eq (Part.fix f x) (Part.Fix.approx f (Nat.find h').succ x)
  -/
  have : p (Nat.find h') := Nat.find_spec h'
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    this : p (Nat.find h')
    ⊢ Eq (Part.fix f x) (Part.Fix.approx f (Nat.find h').succ x)
  -/
  generalize hk : Nat.find h' = k
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    this : p (Nat.find h')
    k : Nat
    hk : Eq (Nat.find h') k
    ⊢ Eq (Part.fix f x) (Part.Fix.approx f k.succ x)
  -/
  replace hk : Nat.find h' = k + (@Upto.zero p).val := hk
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    this : p (Nat.find h')
    k : Nat
    hk : Eq (Nat.find h') (HAdd.hAdd k ↑Nat.Upto.zero)
    ⊢ Eq (Part.fix f x) (Part.Fix.approx f k.succ x)
  -/
  rw [hk] at this
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    k : Nat
    this : p (HAdd.hAdd k ↑Nat.Upto.zero)
    hk : Eq (Nat.find h') (HAdd.hAdd k ↑Nat.Upto.zero)
    ⊢ Eq (Part.fix f x) (Part.Fix.approx f k.succ x)
  -/
  revert hk
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    k : Nat
    this : p (HAdd.hAdd k ↑Nat.Upto.zero)
    ⊢ Eq (Nat.find h') (HAdd.hAdd k ↑Nat.Upto.zero) → Eq (Part.fix f x) (Part.Fix. …
  -/
  dsimp [Part.fix]; rw [assert_pos h']; revert this
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Exists fun i => (Part.Fix.approx f i x).Dom
    p : Nat → Prop := fun i => (Part.Fix.approx f i x).Dom
    k : Nat
    ⊢ p (HAdd.hAdd k ↑Nat.Upto.zero) → Eq (Nat.find h') (HAdd.hAdd k ↑Nat.Upto.zer …
  -/
  generalize Upto.zero = z; intro _this hk
  suffices ∀ x',
    WellFounded.fix (Part.fix.proof_1 f x h') (fixAux f) z x' = Fix.approx f (succ k) x'
    from this _
  induction k generalizing z with
  | zero =>
    intro x'
    rw [Fix.approx, WellFounded.fix_eq, fixAux]
    congr
    ext x : 1
    rw [assert_neg]
    · rfl
    · rw [Nat.zero_add] at _this
      simpa only [not_not, Coe]
  | succ n n_ih =>
    intro x'
    rw [Fix.approx, WellFounded.fix_eq, fixAux]
    congr
    ext : 1
    have hh : ¬(Fix.approx f z.val x).Dom := by
      apply Nat.find_min h'
      rw [hk, Nat.succ_add_eq_add_succ]
      apply Nat.lt_of_succ_le
      apply Nat.le_add_left
    rw [succ_add_eq_add_succ] at _this hk
    rw [assert_pos hh, n_ih (Upto.succ z hh) _this hk]


theorem fix_def' {x : α} (h' : ¬∃ i, (Fix.approx f i x).Dom) : Part.fix f x = none := by
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Not (Exists fun i => (Part.Fix.approx f i x).Dom)
    ⊢ Eq (Part.fix f x) Part.none
  -/
  dsimp [Part.fix]
  /-
    α : Type u_1
    β : α → Type u_2
    f : ((a : α) → Part (β a)) → (a : α) → Part (β a)
    x : α
    h' : Not (Exists fun i => (Part.Fix.approx f i x).Dom)
    ⊢ Eq (Part.assert (Exists fun i => (Part.Fix.approx f i x).Dom) fun h => ⋯.fix …
  -/
  rw [assert_neg h']
  /-
    🎉 no goals
  -/


instance hasFix : Fix (Part α) :=
  ⟨fun f => Part.fix (fun x u => f (x u)) ()⟩


instance Part.hasFix {β} : Fix (α → Part β) :=
  ⟨Part.fix⟩


