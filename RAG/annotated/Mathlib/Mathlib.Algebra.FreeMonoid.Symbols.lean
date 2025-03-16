/-- the set of unique symbols in a free monoid element -/
@[to_additive "The set of unique symbols in an additive free monoid element"]
def symbols (a : FreeMonoid α) : Finset α := List.toFinset a


@[to_additive (attr := simp)]
theorem symbols_one : symbols (1 : FreeMonoid α) = ∅ := rfl


@[to_additive (attr := simp)]
theorem symbols_of {m : α} : symbols (of m) = {m} := rfl


@[to_additive (attr := simp)]
theorem symbols_mul {a b : FreeMonoid α} : symbols (a * b) = symbols a ∪ symbols b := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : FreeMonoid α
    ⊢ Eq (HMul.hMul a b).symbols (Union.union a.symbols b.symbols)
  -/
  simp only [symbols, List.mem_toFinset, Finset.mem_union]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : FreeMonoid α
    ⊢ Eq (List.toFinset (HMul.hMul a b)) (Union.union (List.toFinset a) (List.toFi …
  -/
  apply List.toFinset_append
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mem_symbols {m : α} {a : FreeMonoid α} : m ∈ symbols a ↔ m ∈ a :=
  List.mem_toFinset


