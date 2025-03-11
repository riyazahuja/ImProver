@[simp] theorem get_pure {α} (x : α) : (Thunk.pure x).get = x := rfl

@[simp] theorem get_mk {α} (f : Unit → α) : (Thunk.mk f).get = f () := rfl


instance [DecidableEq α] : DecidableEq (Thunk α) := by
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    ⊢ DecidableEq (Thunk α)
  -/
  intro a b
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    a b : Thunk α
    ⊢ Decidable (Eq a b)
  -/
  have : a = b ↔ a.get = b.get := ⟨by intro x; rw [x], by intro; ext; assumption⟩
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    a b : Thunk α
    this : Iff (Eq a b) (Eq a.get b.get)
    ⊢ Decidable (Eq a b)
  -/
  rw [this]
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq α
    a b : Thunk α
    this : Iff (Eq a b) (Eq a.get b.get)
    ⊢ Decidable (Eq a.get b.get)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The cartesian product of two thunks. -/
def prod (a : Thunk α) (b : Thunk β) : Thunk (α × β) := Thunk.mk fun _ => (a.get, b.get)


@[simp] theorem prod_get_fst {a : Thunk α} {b : Thunk β} : (prod a b).get.1 = a.get := rfl

@[simp] theorem prod_get_snd {a : Thunk α} {b : Thunk β} : (prod a b).get.2 = b.get := rfl


/-- The sum of two thunks. -/
def add [Add α] (a b : Thunk α) : Thunk α := Thunk.mk fun _ => a.get + b.get


instance [Add α] : Add (Thunk α) := ⟨add⟩


@[simp] theorem add_get [Add α] {a b : Thunk α} : (a + b).get = a.get + b.get := rfl


