instance instOrderBot : OrderBot ℕ where
  bot := 0
  bot_le := zero_le


instance instNoMaxOrder : NoMaxOrder ℕ where
  exists_gt n := ⟨n + 1, n.lt_succ_self⟩


@[simp, nolint simpNF] protected lemma bot_eq_zero : ⊥ = 0 := rfl


/-- `Nat.find` is the minimum natural number satisfying a predicate `p`. -/
lemma isLeast_find {p : ℕ → Prop} [DecidablePred p] (hp : ∃ n, p n) :
    IsLeast {n | p n} (Nat.find hp) :=
  ⟨Nat.find_spec hp, fun _ ↦ Nat.find_min' hp⟩


/-- `Nat.find` is the minimum element of a nonempty set of natural numbers. -/
lemma Set.Nonempty.isLeast_natFind {s : Set ℕ} [DecidablePred (· ∈ s)] (hs : s.Nonempty) :
    IsLeast s (Nat.find hs) :=
  Nat.isLeast_find hs

