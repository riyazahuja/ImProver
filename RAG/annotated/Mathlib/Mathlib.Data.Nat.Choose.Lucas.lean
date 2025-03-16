/-- For primes `p`, `choose n k` is congruent to `choose (n % p) (k % p) * choose (n / p) (k / p)`
modulo `p`. Also see `choose_modEq_choose_mod_mul_choose_div_nat` for the version with `MOD`. -/
theorem choose_modEq_choose_mod_mul_choose_div :
    choose n k ≡ choose (n % p) (k % p) * choose (n / p) (k / p) [ZMOD p] := by
  have decompose : ((X : (ZMod p)[X]) + 1) ^ n = (X + 1) ^ (n % p) * (X ^ p + 1) ^ (n / p) := by
    simpa using add_pow_eq_mul_pow_add_pow_div_char (X : (ZMod p)[X]) 1 p _
  simp only [← ZMod.intCast_eq_intCast_iff, Int.cast_mul, Int.cast_ofNat,
    ← coeff_X_add_one_pow _ n k, ← eq_intCast (Int.castRingHom (ZMod p)), ← coeff_map,
    Polynomial.map_pow, Polynomial.map_add, Polynomial.map_one, map_X, decompose]
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
    ⊢ Eq ((HMul.hMul (HPow.hPow (HAdd.hAdd Polynomial.X 1) (HMod.hMod n p)) (HPow. …
  -/
  simp only [add_pow, one_pow, mul_one, ← pow_mul, sum_mul_sum]
  conv_lhs =>
    enter [1, 2, k, 2, k']
    rw [← mul_assoc, mul_right_comm _ _ (X ^ (p * k')), ← pow_add, mul_assoc, ← cast_mul]
  have h_iff : ∀ x ∈ range (n % p + 1) ×ˢ range (n / p + 1),
      k = x.1 + p * x.2 ↔ (k % p, k / p) = x := by
    intro ⟨x₁, x₂⟩ hx
    rw [Prod.mk.injEq]
    constructor <;> intro h
    · simp only [mem_product, mem_range] at hx
      have h' : x₁ < p := lt_of_lt_of_le hx.left <| mod_lt _ Fin.pos'
      rw [h, add_mul_mod_self_left, add_mul_div_left _ _ Fin.pos', eq_comm (b := x₂)]
      exact ⟨mod_eq_of_lt h', self_eq_add_left.mpr (div_eq_of_lt h')⟩
    · rw [← h.left, ← h.right, mod_add_div]
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
    h_iff : ∀ (x : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd. …
    ⊢ Eq (((Finset.range (HAdd.hAdd (HMod.hMod n p) 1)).sum fun k => (Finset.range …
  -/
  simp only [finset_sum_coeff, coeff_mul_natCast, coeff_X_pow, ite_mul, zero_mul, ← cast_mul]
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
    h_iff : ∀ (x : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd. …
    ⊢ Eq ((Finset.range (HAdd.hAdd (HMod.hMod n p) 1)).sum fun x => (Finset.range  …
  -/
  rw [← sum_product', sum_congr rfl (fun a ha ↦ if_congr (h_iff a ha) rfl rfl), sum_ite_eq]
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
    h_iff : ∀ (x : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd. …
    ⊢ Eq (ite (Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd (HMod.hMod n p …
  -/
  split_ifs with h
    /-
      case pos
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
      h_iff : ∀ (x : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd. …
      h : Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd (HMod.hMod n p) 1)) ( …
      ⊢ Eq (HMul.hMul 1 ↑(HMul.hMul ((HMod.hMod n p).choose { fst := HMod.hMod k p,  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
      h_iff : ∀ (x : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd. …
      h : Not (Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd (HMod.hMod n p)  …
      ⊢ Eq 0 ((Int.castRingHom (ZMod p)) ↑(HMul.hMul ((HMod.hMod n p).choose (HMod.h …
    -/
  · rw [mem_product, mem_range, mem_range, not_and_or, lt_succ, not_le, not_lt] at h
    /-
      case neg
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      decompose : Eq (HPow.hPow (HAdd.hAdd Polynomial.X 1) n) (HMul.hMul (HPow.hPow  …
      h_iff : ∀ (x : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd. …
      h : Or (LT.lt (HMod.hMod n p) { fst := HMod.hMod k p, snd := HDiv.hDiv k p }.1 …
      ⊢ Eq 0 ((Int.castRingHom (ZMod p)) ↑(HMul.hMul ((HMod.hMod n p).choose (HMod.h …
    -/
                /-
                  🎉 no goals
                -/
    cases h <;> simp [choose_eq_zero_of_lt (by tauto)]
                /-
                  🎉 no goals
                -/


/-- For primes `p`, `choose n k` is congruent to `choose (n % p) (k % p) * choose (n / p) (k / p)`
modulo `p`. Also see `choose_modEq_choose_mod_mul_choose_div` for the version with `ZMOD`. -/
theorem choose_modEq_choose_mod_mul_choose_div_nat :
    choose n k ≡ choose (n % p) (k % p) * choose (n / p) (k / p) [MOD p] := by
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ p.ModEq (n.choose k) (HMul.hMul ((HMod.hMod n p).choose (HMod.hMod k p)) ((H …
  -/
  rw [← Int.natCast_modEq_iff]
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ (↑p).ModEq ↑(n.choose k) ↑(HMul.hMul ((HMod.hMod n p).choose (HMod.hMod k p) …
  -/
  exact_mod_cast choose_modEq_choose_mod_mul_choose_div
  /-
    🎉 no goals
  -/


/-- For primes `p`, `choose n k` is congruent to the product of `choose (⌊n / p ^ i⌋ % p)
(⌊k / p ^ i⌋ % p)` over i < a, multiplied by `choose (⌊n / p ^ a⌋) (⌊k / p ^ a⌋)`, modulo `p`. -/
theorem choose_modEq_choose_mul_prod_range_choose (a : ℕ) :
    choose n k ≡ choose (n / p ^ a) (k / p ^ a) *
      ∏ i in range a, choose (n / p ^ i % p) (k / p ^ i % p) [ZMOD p] :=
  match a with
                   /-
                     n k p : Nat
                     inst✝ : Fact (Nat.Prime p)
                     a : Nat
                     ⊢ (↑p).ModEq (↑(n.choose k)) (HMul.hMul ↑((HDiv.hDiv n (HPow.hPow p Nat.zero)) …
                   -/
  | Nat.zero => by simp
                   /-
                     🎉 no goals
                   -/
  | Nat.succ a => (choose_modEq_choose_mul_prod_range_choose a).trans <| by
    /-
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      a✝ a : Nat
      ⊢ (↑p).ModEq (HMul.hMul ↑((HDiv.hDiv n (HPow.hPow p a)).choose (HDiv.hDiv k (H …
    -/
    rw [prod_range_succ, cast_mul, ← mul_assoc, mul_right_comm]
    /-
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      a✝ a : Nat
      ⊢ (↑p).ModEq (HMul.hMul ↑((HDiv.hDiv n (HPow.hPow p a)).choose (HDiv.hDiv k (H …
    -/
    gcongr
    /-
      case h
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      a✝ a : Nat
      ⊢ (↑p).ModEq (↑((HDiv.hDiv n (HPow.hPow p a)).choose (HDiv.hDiv k (HPow.hPow p …
    -/
    apply choose_modEq_choose_mod_mul_choose_div.trans
    /-
      case h
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      a✝ a : Nat
      ⊢ (↑p).ModEq (HMul.hMul ↑((HMod.hMod (HDiv.hDiv n (HPow.hPow p a)) p).choose ( …
    -/
    simp_rw [pow_succ, Nat.div_div_eq_div_mul, mul_comm]
    /-
      case h
      n k p : Nat
      inst✝ : Fact (Nat.Prime p)
      a✝ a : Nat
      ⊢ (↑p).ModEq (HMul.hMul ↑((HMod.hMod (HDiv.hDiv n (HPow.hPow p a)) p).choose ( …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- **Lucas's Theorem**: For primes `p`, `choose n k` is congruent to the product of
`choose (⌊n / p ^ i⌋ % p) (⌊k / p ^ i⌋ % p)` over `i` modulo `p`. -/
theorem choose_modEq_prod_range_choose {a : ℕ} (ha₁ : n < p ^ a) (ha₂ : k < p ^ a) :
    choose n k ≡ ∏ i in range a, choose (n / p ^ i % p) (k / p ^ i % p) [ZMOD p] := by
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Nat
    ha₁ : LT.lt n (HPow.hPow p a)
    ha₂ : LT.lt k (HPow.hPow p a)
    ⊢ (↑p).ModEq (↑(n.choose k)) ((Finset.range a).prod fun i => ↑((HMod.hMod (HDi …
  -/
  apply (choose_modEq_choose_mul_prod_range_choose a).trans
  simp_rw [Nat.div_eq_of_lt ha₁, Nat.div_eq_of_lt ha₂, choose, cast_one, one_mul, cast_prod,
    Int.ModEq.refl]


/-- **Lucas's Theorem**: For primes `p`, `choose n k` is congruent to the product of
`choose (⌊n / p ^ i⌋ % p) (⌊k / p ^ i⌋ % p)` over `i` modulo `p`. -/
theorem choose_modEq_prod_range_choose_nat {a : ℕ} (ha₁ : n < p ^ a) (ha₂ : k < p ^ a) :
    choose n k ≡ ∏ i in range a, choose (n / p ^ i % p) (k / p ^ i % p) [MOD p] := by
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Nat
    ha₁ : LT.lt n (HPow.hPow p a)
    ha₂ : LT.lt k (HPow.hPow p a)
    ⊢ p.ModEq (n.choose k) ((Finset.range a).prod fun i => (HMod.hMod (HDiv.hDiv n …
  -/
  rw [← Int.natCast_modEq_iff]
  /-
    n k p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : Nat
    ha₁ : LT.lt n (HPow.hPow p a)
    ha₂ : LT.lt k (HPow.hPow p a)
    ⊢ (↑p).ModEq ↑(n.choose k) ↑((Finset.range a).prod fun i => (HMod.hMod (HDiv.h …
  -/
  exact_mod_cast choose_modEq_prod_range_choose ha₁ ha₂
  /-
    🎉 no goals
  -/


alias lucas_theorem := choose_modEq_prod_range_choose

alias lucas_theorem_nat := choose_modEq_prod_range_choose_nat


