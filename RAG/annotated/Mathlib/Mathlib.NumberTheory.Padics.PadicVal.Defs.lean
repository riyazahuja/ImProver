/-- For `p ≠ 1`, the `p`-adic valuation of a natural `n ≠ 0` is the largest natural number `k` such
that `p^k` divides `n`. If `n = 0` or `p = 1`, then `padicValNat p q` defaults to `0`. -/
def padicValNat (p : ℕ) (n : ℕ) : ℕ :=
  if h : p ≠ 1 ∧ 0 < n then Nat.find (finiteMultiplicity_iff.2 h) else 0


theorem padicValNat_def' {n : ℕ} (hp : p ≠ 1) (hn : 0 < n) :
    padicValNat p n = multiplicity p n := by
  /-
    p n : Nat
    hp : Ne p 1
    hn : LT.lt 0 n
    ⊢ Eq (padicValNat p n) (multiplicity p n)
  -/
  simp [padicValNat, hp, hn, multiplicity, emultiplicity, finiteMultiplicity_iff.2 ⟨hp, hn⟩]
  /-
    p n : Nat
    hp : Ne p 1
    hn : LT.lt 0 n
    ⊢ Eq (Nat.find ⋯) (WithTop.untop' 1 ↑(Nat.find ⋯))
  -/
  convert (WithTop.untop'_coe ..).symm
  /-
    🎉 no goals
  -/


/-- A simplification of `padicValNat` when one input is prime, by analogy with
`padicValRat_def`. -/
theorem padicValNat_def [hp : Fact p.Prime] {n : ℕ} (hn : 0 < n) :
    padicValNat p n = multiplicity p n :=
  padicValNat_def' hp.out.ne_one hn


/-- A simplification of `padicValNat` when one input is prime, by analogy with
`padicValRat_def`. -/
theorem padicValNat_eq_emultiplicity [hp : Fact p.Prime] {n : ℕ} (hn : 0 < n) :
    padicValNat p n = emultiplicity p n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (↑(padicValNat p n)) (emultiplicity p n)
  -/
  rw [(finiteMultiplicity_iff.2 ⟨hp.out.ne_one, hn⟩).emultiplicity_eq_multiplicity]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq ↑(padicValNat p n) ↑(multiplicity p n)
  -/
  exact_mod_cast padicValNat_def hn
  /-
    🎉 no goals
  -/


/-- `padicValNat p 0` is `0` for any `p`. -/
@[simp]
                                                   /-
                                                     p : Nat
                                                     ⊢ Eq (padicValNat p 0) 0
                                                   -/
protected theorem zero : padicValNat p 0 = 0 := by simp [padicValNat]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `padicValNat p 1` is `0` for any `p`. -/
@[simp]
                                                  /-
                                                    p : Nat
                                                    ⊢ Eq (padicValNat p 1) 0
                                                  -/
protected theorem one : padicValNat p 1 = 0 := by simp [padicValNat]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem eq_zero_iff {n : ℕ} : padicValNat p n = 0 ↔ p = 1 ∨ n = 0 ∨ ¬p ∣ n := by
  simp only [padicValNat, ne_eq, pos_iff_ne_zero, dite_eq_right_iff, find_eq_zero, zero_add,
    pow_one, and_imp, ← or_iff_not_imp_left]


theorem le_emultiplicity_iff_replicate_subperm_primeFactorsList {a b : ℕ} {n : ℕ} (ha : a.Prime)
    (hb : b ≠ 0) :
    ↑n ≤ emultiplicity a b ↔ replicate n a <+~ b.primeFactorsList :=
  (replicate_subperm_primeFactorsList_iff ha hb).trans
    pow_dvd_iff_le_emultiplicity |>.symm


@[deprecated (since := "2024-07-17")]
alias le_multiplicity_iff_replicate_subperm_factors :=
  le_emultiplicity_iff_replicate_subperm_primeFactorsList


theorem le_padicValNat_iff_replicate_subperm_primeFactorsList {a b : ℕ} {n : ℕ} (ha : a.Prime)
    (hb : b ≠ 0) :
    n ≤ padicValNat a b ↔ replicate n a <+~ b.primeFactorsList := by
  rw [← le_emultiplicity_iff_replicate_subperm_primeFactorsList ha hb,
    Nat.finiteMultiplicity_iff.2 ⟨ha.ne_one, Nat.pos_of_ne_zero hb⟩
      |>.emultiplicity_eq_multiplicity,     ← padicValNat_def' ha.ne_one (Nat.pos_of_ne_zero hb),
    Nat.cast_le]


@[deprecated (since := "2024-07-17")]
alias le_padicValNat_iff_replicate_subperm_factors :=
  le_padicValNat_iff_replicate_subperm_primeFactorsList

