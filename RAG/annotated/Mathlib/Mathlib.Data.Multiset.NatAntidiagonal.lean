/-- The antidiagonal of a natural number `n` is
    the multiset of pairs `(i, j)` such that `i + j = n`. -/
def antidiagonal (n : ℕ) : Multiset (ℕ × ℕ) :=
  List.Nat.antidiagonal n


/-- A pair (i, j) is contained in the antidiagonal of `n` if and only if `i + j = n`. -/
@[simp]
theorem mem_antidiagonal {n : ℕ} {x : ℕ × ℕ} : x ∈ antidiagonal n ↔ x.1 + x.2 = n := by
  /-
    n : Nat
    x : Prod Nat Nat
    ⊢ Iff (Membership.mem (Multiset.Nat.antidiagonal n) x) (Eq (HAdd.hAdd x.1 x.2) …
  -/
  rw [antidiagonal, mem_coe, List.Nat.mem_antidiagonal]
  /-
    🎉 no goals
  -/


/-- The cardinality of the antidiagonal of `n` is `n+1`. -/
@[simp]
theorem card_antidiagonal (n : ℕ) : card (antidiagonal n) = n + 1 := by
  /-
    n : Nat
    ⊢ Eq (Multiset.Nat.antidiagonal n).card (HAdd.hAdd n 1)
  -/
  rw [antidiagonal, coe_card, List.Nat.length_antidiagonal]
  /-
    🎉 no goals
  -/


/-- The antidiagonal of `0` is the list `[(0, 0)]` -/
@[simp]
theorem antidiagonal_zero : antidiagonal 0 = {(0, 0)} :=
  rfl


/-- The antidiagonal of `n` does not contain duplicate entries. -/
@[simp]
theorem nodup_antidiagonal (n : ℕ) : Nodup (antidiagonal n) :=
  coe_nodup.2 <| List.Nat.nodup_antidiagonal n


@[simp]
theorem antidiagonal_succ {n : ℕ} :
    antidiagonal (n + 1) = (0, n + 1) ::ₘ (antidiagonal n).map (Prod.map Nat.succ id) := by
  /-
    n : Nat
    ⊢ Eq (Multiset.Nat.antidiagonal (HAdd.hAdd n 1)) (Multiset.cons { fst := 0, sn …
  -/
  simp only [antidiagonal, List.Nat.antidiagonal_succ, map_coe, cons_coe]
  /-
    🎉 no goals
  -/


theorem antidiagonal_succ' {n : ℕ} :
    antidiagonal (n + 1) = (n + 1, 0) ::ₘ (antidiagonal n).map (Prod.map id Nat.succ) := by
  rw [antidiagonal, List.Nat.antidiagonal_succ', ← coe_add, add_comm, antidiagonal, map_coe,
    coe_add, List.singleton_append, cons_coe]


theorem antidiagonal_succ_succ' {n : ℕ} :
    antidiagonal (n + 2) =
      (0, n + 2) ::ₘ (n + 2, 0) ::ₘ (antidiagonal n).map (Prod.map Nat.succ Nat.succ) := by
  /-
    n : Nat
    ⊢ Eq (Multiset.Nat.antidiagonal (HAdd.hAdd n 2)) (Multiset.cons { fst := 0, sn …
  -/
  rw [antidiagonal_succ, antidiagonal_succ', map_cons, map_map, Prod.map_apply]
  /-
    n : Nat
    ⊢ Eq (Multiset.cons { fst := 0, snd := HAdd.hAdd (HAdd.hAdd n 1) 1 } (Multiset …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_swap_antidiagonal {n : ℕ} : (antidiagonal n).map Prod.swap = antidiagonal n := by
  /-
    n : Nat
    ⊢ Eq (Multiset.map Prod.swap (Multiset.Nat.antidiagonal n)) (Multiset.Nat.anti …
  -/
  rw [antidiagonal, map_coe, List.Nat.map_swap_antidiagonal, coe_reverse]
  /-
    🎉 no goals
  -/


