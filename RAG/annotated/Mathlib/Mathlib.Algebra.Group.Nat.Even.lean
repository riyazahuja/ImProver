lemma even_iff : Even n ↔ n % 2 = 0 where
                         /-
                           n : Nat
                           x✝ : Even n
                           m : Nat
                           hm : Eq n (HAdd.hAdd m m)
                           ⊢ Eq (HMod.hMod n 2) 0
                         -/
  mp := fun ⟨m, hm⟩ ↦ by simp [← Nat.two_mul, hm]
                         /-
                           🎉 no goals
                         -/
                                                    /-
                                                      n : Nat
                                                      h : Eq (HMod.hMod n 2) 0
                                                      ⊢ Eq (HAdd.hAdd (HMod.hMod n 2) (HMul.hMul 2 (HDiv.hDiv n 2))) (HAdd.hAdd (HDi …
                                                    -/
  mpr h := ⟨n / 2, (mod_add_div n 2).symm.trans (by simp [← Nat.two_mul, h])⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


instance : DecidablePred (Even : ℕ → Prop) := fun _ ↦ decidable_of_iff _ even_iff.symm


/-- `IsSquare` can be decided on `ℕ` by checking against the square root. -/
instance : DecidablePred (IsSquare : ℕ → Prop) :=
  fun m ↦ decidable_of_iff' (Nat.sqrt m * Nat.sqrt m = m) <| by
    /-
      m✝ n m : Nat
      ⊢ Iff (IsSquare m) (Eq (HMul.hMul m.sqrt m.sqrt) m)
    -/
    simp_rw [← Nat.exists_mul_self m, IsSquare, eq_comm]
    /-
      🎉 no goals
    -/


                                                /-
                                                  n : Nat
                                                  ⊢ Iff (Not (Even n)) (Eq (HMod.hMod n 2) 1)
                                                -/
lemma not_even_iff : ¬ Even n ↔ n % 2 = 1 := by rw [even_iff, mod_two_not_eq_zero]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma two_dvd_ne_zero : ¬2 ∣ n ↔ n % 2 = 1 :=
  (even_iff_exists_two_nsmul _).symm.not.trans not_even_iff


                                           /-
                                             ⊢ Not (Even 1)
                                           -/
@[simp] lemma not_even_one : ¬Even 1 := by simp [even_iff]
                                           /-
                                             🎉 no goals
                                           -/


@[parity_simps] lemma even_add : Even (m + n) ↔ (Even m ↔ Even n) := by
  /-
    m n : Nat
    ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Even m) (Even n))
  -/
  rcases mod_two_eq_zero_or_one m with h₁ | h₁ <;> rcases mod_two_eq_zero_or_one n with h₂ | h₂ <;>
    /-
      case inl.inl
      m n : Nat
      h₁ : Eq (HMod.hMod m 2) 0
      h₂ : Eq (HMod.hMod n 2) 0
      ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Even m) (Even n))
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
    simp [even_iff, h₁, h₂, Nat.add_mod]
    /-
      🎉 no goals
    -/


                                                                  /-
                                                                    n : Nat
                                                                    ⊢ Iff (Even (HAdd.hAdd n 1)) (Not (Even n))
                                                                  -/
@[parity_simps] lemma even_add_one : Even (n + 1) ↔ ¬Even n := by simp [even_add]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma succ_mod_two_eq_zero_iff : (m + 1) % 2 = 0 ↔ m % 2 = 1 := by
  /-
    m : Nat
    ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd m 1) 2) 0) (Eq (HMod.hMod m 2) 1)
  -/
  simp [← Nat.even_iff, ← Nat.not_even_iff, parity_simps]
  /-
    🎉 no goals
  -/


lemma succ_mod_two_eq_one_iff : (m + 1) % 2 = 1 ↔ m % 2 = 0 := by
  /-
    m : Nat
    ⊢ Iff (Eq (HMod.hMod (HAdd.hAdd m 1) 2) 1) (Eq (HMod.hMod m 2) 0)
  -/
  simp [← Nat.even_iff, ← Nat.not_even_iff, parity_simps]
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   n : Nat
                                                                   ⊢ Not (Dvd.dvd 2 (HAdd.hAdd (HMul.hMul 2 n) 1))
                                                                 -/
lemma two_not_dvd_two_mul_add_one (n : ℕ) : ¬2 ∣ 2 * n + 1 := by simp [add_mod]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma two_not_dvd_two_mul_sub_one : ∀ {n}, 0 < n → ¬2 ∣ 2 * n - 1
  | n + 1, _ => two_not_dvd_two_mul_add_one n


@[parity_simps] lemma even_sub (h : n ≤ m) : Even (m - n) ↔ (Even m ↔ Even n) := by
  /-
    m n : Nat
    h : LE.le n m
    ⊢ Iff (Even (HSub.hSub m n)) (Iff (Even m) (Even n))
  -/
  conv_rhs => rw [← Nat.sub_add_cancel h, even_add]
  /-
    m n : Nat
    h : LE.le n m
    ⊢ Iff (Even (HSub.hSub m n)) (Iff (Iff (Even (HSub.hSub m n)) (Even n)) (Even  …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases h : Even n <;> simp [h]
                          /-
                            🎉 no goals
                          -/


@[parity_simps] lemma even_mul : Even (m * n) ↔ Even m ∨ Even n := by
  /-
    m n : Nat
    ⊢ Iff (Even (HMul.hMul m n)) (Or (Even m) (Even n))
  -/
  rcases mod_two_eq_zero_or_one m with h₁ | h₁ <;> rcases mod_two_eq_zero_or_one n with h₂ | h₂ <;>
    /-
      case inl.inl
      m n : Nat
      h₁ : Eq (HMod.hMod m 2) 0
      h₂ : Eq (HMod.hMod n 2) 0
      ⊢ Iff (Even (HMul.hMul m n)) (Or (Even m) (Even n))
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
    simp [even_iff, h₁, h₂, Nat.mul_mod]
    /-
      🎉 no goals
    -/


/-- If `m` and `n` are natural numbers, then the natural number `m^n` is even
if and only if `m` is even and `n` is positive. -/
@[parity_simps] lemma even_pow : Even (m ^ n) ↔ Even m ∧ n ≠ 0 := by
  /-
    m n : Nat
    ⊢ Iff (Even (HPow.hPow m n)) (And (Even m) (Ne n 0))
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp +contextual [*, pow_succ', even_mul]
                  /-
                    🎉 no goals
                  -/


lemma even_pow' (h : n ≠ 0) : Even (m ^ n) ↔ Even m := even_pow.trans <| and_iff_left h


                                                            /-
                                                              n : Nat
                                                              ⊢ Even (HMul.hMul n (HAdd.hAdd n 1))
                                                            -/
lemma even_mul_succ_self (n : ℕ) : Even (n * (n + 1)) := by rw [even_mul, even_add_one]; exact em _
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


lemma even_mul_pred_self : ∀ n : ℕ, Even (n * (n - 1))
  | 0 => even_zero
  | (n + 1) => mul_comm (n + 1 - 1) (n + 1) ▸ even_mul_succ_self n


lemma two_mul_div_two_of_even : Even n → 2 * (n / 2) = n := fun h ↦
  Nat.mul_div_cancel_left' ((even_iff_exists_two_nsmul _).1 h)


lemma div_two_mul_two_of_even : Even n → n / 2 * 2 = n :=
  fun h ↦ Nat.div_mul_cancel ((even_iff_exists_two_nsmul _).1 h)


theorem one_lt_of_ne_zero_of_even (h0 : n ≠ 0) (hn : Even n) : 1 < n := by
  /-
    n : Nat
    h0 : Ne n 0
    hn : Even n
    ⊢ LT.lt 1 n
  -/
  refine Nat.one_lt_iff_ne_zero_and_ne_one.mpr (And.intro h0 ?_)
  /-
    n : Nat
    h0 : Ne n 0
    hn : Even n
    ⊢ Ne n 1
  -/
  intro h
  /-
    n : Nat
    h0 : Ne n 0
    hn : Even n
    h : Eq n 1
    ⊢ False
  -/
  rw [h] at hn
  /-
    n : Nat
    h0 : Ne n 0
    hn : Even 1
    h : Eq n 1
    ⊢ False
  -/
  exact Nat.not_even_one hn
  /-
    🎉 no goals
  -/


theorem add_one_lt_of_even (hn : Even n) (hm : Even m) (hnm : n < m) :
    n + 1 < m := by
  /-
    m n : Nat
    hn : Even n
    hm : Even m
    hnm : LT.lt n m
    ⊢ LT.lt (HAdd.hAdd n 1) m
  -/
  rcases hn with ⟨n, rfl⟩
  /-
    case intro
    m : Nat
    hm : Even m
    n : Nat
    hnm : LT.lt (HAdd.hAdd n n) m
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd n n) 1) m
  -/
  rcases hm with ⟨m, rfl⟩
  /-
    case intro.intro
    n m : Nat
    hnm : LT.lt (HAdd.hAdd n n) (HAdd.hAdd m m)
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd n n) 1) (HAdd.hAdd m m)
  -/
  omega
  /-
    🎉 no goals
  -/

-- Here are examples of how `parity_simps` can be used with `Nat`.

