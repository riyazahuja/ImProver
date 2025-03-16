theorem cast_descFactorial {n p : ℕ} (h : n ≤ p) :
    (descFactorial (p - 1) n : ZMod p) = (-1) ^ n * n ! := by
  /-
    n p : Nat
    h : LE.le n p
    ⊢ Eq (↑((HSub.hSub p 1).descFactorial n)) (HMul.hMul (HPow.hPow (-1) n) ↑n.fac …
  -/
  rw [descFactorial_eq_prod_range, ← prod_range_add_one_eq_factorial]
  /-
    n p : Nat
    h : LE.le n p
    ⊢ Eq (↑((Finset.range n).prod fun i => HSub.hSub (HSub.hSub p 1) i)) (HMul.hMu …
  -/
  simp only [cast_prod]
  /-
    n p : Nat
    h : LE.le n p
    ⊢ Eq ((Finset.range n).prod fun i => ↑(HSub.hSub (HSub.hSub p 1) i)) (HMul.hMu …
  -/
  nth_rw 2 [← card_range n]
  /-
    n p : Nat
    h : LE.le n p
    ⊢ Eq ((Finset.range n).prod fun i => ↑(HSub.hSub (HSub.hSub p 1) i)) (HMul.hMu …
  -/
  rw [pow_card_mul_prod]
  /-
    n p : Nat
    h : LE.le n p
    ⊢ Eq ((Finset.range n).prod fun i => ↑(HSub.hSub (HSub.hSub p 1) i)) ((Finset. …
  -/
  refine prod_congr rfl ?_
  /-
    n p : Nat
    h : LE.le n p
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → Eq (↑(HSub.hSub (HSub.hSub  …
  -/
  intro x hx
  rw [← tsub_add_eq_tsub_tsub_swap,
    Nat.cast_sub <| Nat.le_trans (Nat.add_one_le_iff.mpr (List.mem_range.mp hx)) h,
    CharP.cast_eq_zero, zero_sub, cast_succ, neg_add_rev, mul_add, neg_mul, one_mul,
    mul_one, add_comm]


