local notation "𝕎" => WittVector p -- type as `\bbW`


/-- The rational polynomials that give the coefficients of `frobenius x`,
in terms of the coefficients of `x`.
These polynomials actually have integral coefficients,
see `frobeniusPoly` and `map_frobeniusPoly`. -/
def frobeniusPolyRat (n : ℕ) : MvPolynomial ℕ ℚ :=
  bind₁ (wittPolynomial p ℚ ∘ fun n => n + 1) (xInTermsOfW p ℚ n)


theorem bind₁_frobeniusPolyRat_wittPolynomial (n : ℕ) :
    bind₁ (frobeniusPolyRat p) (wittPolynomial p ℚ n) = wittPolynomial p ℚ (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ (WittVector.frobeniusPolyRat p)) (wittPolynomial p R …
  -/
  delta frobeniusPolyRat
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ fun n => (MvPolynomial.bind₁ (Function.comp (wittPol …
  -/
  rw [← bind₁_bind₁, bind₁_xInTermsOfW_wittPolynomial, bind₁_X_right, Function.comp_apply]
  /-
    🎉 no goals
  -/


local notation "v" => multiplicity


/-- An auxiliary polynomial over the integers, that satisfies
`p * (frobeniusPolyAux p n) + X n ^ p = frobeniusPoly p n`.
This makes it easy to show that `frobeniusPoly p n` is congruent to `X n ^ p`
modulo `p`. -/
noncomputable def frobeniusPolyAux : ℕ → MvPolynomial ℕ ℤ
  | n => X (n + 1) -  ∑ i : Fin n, have _ := i.is_lt
      ∑ j ∈ range (p ^ (n - i)),
        (((X (i : ℕ) ^ p) ^ (p ^ (n - (i : ℕ)) - (j + 1)) : MvPolynomial ℕ ℤ) *
        (frobeniusPolyAux i) ^ (j + 1)) *
        C (((p ^ (n - i)).choose (j + 1) / (p ^ (n - i - v p (j + 1)))
          * ↑p ^ (j - v p (j + 1)) : ℕ) : ℤ)


omit hp in
theorem frobeniusPolyAux_eq (n : ℕ) :
    frobeniusPolyAux p n =
      X (n + 1) - ∑ i ∈ range n,
          ∑ j ∈ range (p ^ (n - i)),
            (X i ^ p) ^ (p ^ (n - i) - (j + 1)) * frobeniusPolyAux p i ^ (j + 1) *
              C ↑((p ^ (n - i)).choose (j + 1) / p ^ (n - i - v p (j + 1)) *
                ↑p ^ (j - v p (j + 1)) : ℕ) := by
  /-
    p n : Nat
    ⊢ Eq (WittVector.frobeniusPolyAux p n) (HSub.hSub (MvPolynomial.X (HAdd.hAdd n …
  -/
  rw [frobeniusPolyAux, ← Fin.sum_univ_eq_sum_range]
  /-
    🎉 no goals
  -/


/-- The polynomials that give the coefficients of `frobenius x`,
in terms of the coefficients of `x`. -/
def frobeniusPoly (n : ℕ) : MvPolynomial ℕ ℤ :=
  X n ^ p + C (p : ℤ) * frobeniusPolyAux p n

/-
Our next goal is to prove
```
lemma map_frobeniusPoly (n : ℕ) :
    MvPolynomial.map (Int.castRingHom ℚ) (frobeniusPoly p n) = frobeniusPolyRat p n
```
This lemma has a rather long proof, but it mostly boils down to applying induction,
and then using the following two key facts at the right point.
-/

/-- A key divisibility fact for the proof of `WittVector.map_frobeniusPoly`. -/
theorem map_frobeniusPoly.key₁ (n j : ℕ) (hj : j < p ^ n) :
    p ^ (n - v p (j + 1)) ∣ (p ^ n).choose (j + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n j : Nat
    hj : LT.lt j (HPow.hPow p n)
    ⊢ Dvd.dvd (HPow.hPow p (HSub.hSub n (multiplicity p (HAdd.hAdd j 1)))) ((HPow. …
  -/
  apply pow_dvd_of_le_emultiplicity
  /-
    case hk
    p : Nat
    hp : Fact (Nat.Prime p)
    n j : Nat
    hj : LT.lt j (HPow.hPow p n)
    ⊢ LE.le (↑(HSub.hSub n (multiplicity p (HAdd.hAdd j 1)))) (emultiplicity p ((H …
  -/
  rw [hp.out.emultiplicity_choose_prime_pow hj j.succ_ne_zero]
  /-
    🎉 no goals
  -/


/-- A key numerical identity needed for the proof of `WittVector.map_frobeniusPoly`. -/
theorem map_frobeniusPoly.key₂ {n i j : ℕ} (hi : i ≤ n) (hj : j < p ^ (n - i)) :
    j - v p (j + 1) + n = i + j + (n - i - v p (j + 1)) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n i j : Nat
    hi : LE.le i n
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HAdd.hAdd (HSub.hSub j (multiplicity p (HAdd.hAdd j 1))) n) (HAdd.hAdd ( …
  -/
  generalize h : v p (j + 1) = m
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n i j : Nat
    hi : LE.le i n
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    m : Nat
    h : Eq (multiplicity p (HAdd.hAdd j 1)) m
    ⊢ Eq (HAdd.hAdd (HSub.hSub j m) n) (HAdd.hAdd (HAdd.hAdd i j) (HSub.hSub (HSub …
  -/
  rsuffices ⟨h₁, h₂⟩ : m ≤ n - i ∧ m ≤ j
  · rw [tsub_add_eq_add_tsub h₂, add_comm i j, add_tsub_assoc_of_le (h₁.trans (Nat.sub_le n i)),
      add_assoc, tsub_right_comm, add_comm i,
      tsub_add_cancel_of_le (le_tsub_of_add_le_right ((le_tsub_iff_left hi).mp h₁))]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n i j : Nat
    hi : LE.le i n
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    m : Nat
    h : Eq (multiplicity p (HAdd.hAdd j 1)) m
    ⊢ And (LE.le m (HSub.hSub n i)) (LE.le m j)
  -/
  have hle : p ^ m ≤ j + 1 := h ▸ Nat.le_of_dvd j.succ_pos (pow_multiplicity_dvd _ _)
  exact ⟨(Nat.pow_le_pow_iff_right hp.1.one_lt).1 (hle.trans hj),
     Nat.le_of_lt_succ ((m.lt_pow_self hp.1.one_lt).trans_le hle)⟩


theorem map_frobeniusPoly (n : ℕ) :
    MvPolynomial.map (Int.castRingHom ℚ) (frobeniusPoly p n) = frobeniusPolyRat p n := by
  rw [frobeniusPoly, RingHom.map_add, RingHom.map_mul, RingHom.map_pow, map_C, map_X, eq_intCast,
    Int.cast_natCast, frobeniusPolyRat]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X n) p) (HMul.hMul (MvPolynomial.C ↑p …
  -/
  refine Nat.strong_induction_on n ?_; clear n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ ∀ (n : Nat), (∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomia …
  -/
  intro n IH
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    ⊢ Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X n) p) (HMul.hMul (MvPolynomial.C ↑p …
  -/
  rw [xInTermsOfW_eq]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    ⊢ Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X n) p) (HMul.hMul (MvPolynomial.C ↑p …
  -/
  simp only [map_sum, map_sub, map_mul, map_pow (bind₁ _), bind₁_C_right]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    ⊢ Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X n) p) (HMul.hMul (MvPolynomial.C ↑p …
  -/
  have h1 : (p : ℚ) ^ n * ⅟ (p : ℚ) ^ n = 1 := by rw [← mul_pow, mul_invOf_self, one_pow]
  rw [bind₁_X_right, Function.comp_apply, wittPolynomial_eq_sum_C_mul_X_pow, sum_range_succ,
    sum_range_succ, tsub_self, add_tsub_cancel_left, pow_zero, pow_one, pow_one, sub_mul, add_mul,
    add_mul, mul_right_comm, mul_right_comm (C ((p : ℚ) ^ (n + 1))), ← C_mul, ← C_mul, pow_succ',
    mul_assoc (p : ℚ) ((p : ℚ) ^ n), h1, mul_one, C_1, one_mul, add_comm _ (X n ^ p), add_assoc,
    ← add_sub, add_right_inj, frobeniusPolyAux_eq, RingHom.map_sub, map_X, mul_sub, sub_eq_add_neg,
    add_comm _ (C (p : ℚ) * X (n + 1)), ← add_sub,
    add_right_inj, neg_eq_iff_eq_neg, neg_sub, eq_comm]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    ⊢ Eq (HSub.hSub (HMul.hMul ((Finset.range n).sum fun x => HMul.hMul (MvPolynom …
  -/
  simp only [map_sum, mul_sum, sum_mul, ← sum_sub_distrib]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    ⊢ Eq ((Finset.range n).sum fun x => HSub.hSub (HMul.hMul (HMul.hMul (MvPolynom …
  -/
  apply sum_congr rfl
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range n) x → Eq (HSub.hSub (HMul.hMul (H …
  -/
  intro i hi
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : Membership.mem (Finset.range n) i
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) i)) (HPo …
  -/
  rw [mem_range] at hi
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) i)) (HPo …
  -/
  rw [← IH i hi]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X m) p) ( …
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) i)) (HPo …
  -/
  clear IH
  rw [add_comm (X i ^ p), add_pow, sum_range_succ', pow_zero, tsub_zero, Nat.choose_zero_right,
    one_mul, Nat.cast_one, mul_one, mul_add, add_mul, Nat.succ_sub (le_of_lt hi),
    Nat.succ_eq_add_one (n - i), pow_succ', pow_mul, add_sub_cancel_right, mul_sum, sum_mul]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    ⊢ Eq ((Finset.range (HPow.hPow p (HSub.hSub n i))).sum fun i_1 => HMul.hMul (H …
  -/
  apply sum_congr rfl
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HPow.hPow p (HSub.hSub n i))) x → …
  -/
  intro j hj
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : Membership.mem (Finset.range (HPow.hPow p (HSub.hSub n i))) j
    ⊢ Eq (HMul.hMul (HMul.hMul (MvPolynomial.C (HPow.hPow (↑p) i)) (HMul.hMul (HMu …
  -/
  rw [mem_range] at hj
  rw [RingHom.map_mul, RingHom.map_mul, RingHom.map_pow, RingHom.map_pow, RingHom.map_pow,
    RingHom.map_pow, RingHom.map_pow, map_C, map_X, mul_pow]
  rw [mul_comm (C (p : ℚ) ^ i), mul_comm _ ((X i ^ p) ^ _), mul_comm (C (p : ℚ) ^ (j + 1)),
    mul_comm (C (p : ℚ))]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HPow.hPow (MvPoly …
  -/
  simp only [mul_assoc]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow (MvPolynomial.X i) p) (HSub.hSub (HPow.h …
  -/
  apply congr_arg
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow ((MvPolynomial.map (Int.castRingHom Rat)) (WittVect …
  -/
  apply congr_arg
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow (MvPolynomial.C ↑p) (HAdd.hAdd j 1)) (HMul.hMul (↑( …
  -/
  rw [← C_eq_coe_nat]
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow (MvPolynomial.C ↑p) (HAdd.hAdd j 1)) (HMul.hMul (Mv …
  -/
  simp only [← RingHom.map_pow, ← C_mul]
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (MvPolynomial.C (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd j 1)) (HMul.hMul (↑ …
  -/
  rw [C_inj]
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd j 1)) (HMul.hMul (↑((HPow.hPow p (H …
  -/
  simp only [invOf_eq_inv, eq_intCast, inv_pow, Int.cast_natCast, Nat.cast_mul, Int.cast_mul]
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd j 1)) (HMul.hMul (↑((HPow.hPow p (H …
  -/
  rw [Rat.natCast_div _ _ (map_frobeniusPoly.key₁ p (n - i) j hj)]
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) (HAdd.hAdd j 1)) (HMul.hMul (↑((HPow.hPow p (H …
  -/
  simp only [Nat.cast_pow, pow_add, pow_one]
  suffices
    (((p ^ (n - i)).choose (j + 1) : ℚ) * (p : ℚ) ^ (j - v p (j + 1)) * p * (p ^ n : ℚ))
      = (p : ℚ) ^ j * p * ↑((p ^ (n - i)).choose (j + 1) * p ^ i) *
        (p : ℚ) ^ (n - i - v p (j + 1)) by
    have aux : ∀ k : ℕ, (p : ℚ)^ k ≠ 0 := by
      intro; apply pow_ne_zero; exact mod_cast hp.1.ne_zero
    simpa [aux, -one_div, -pow_eq_zero_iff', field_simps] using this.symm
  rw [mul_comm _ (p : ℚ), mul_assoc, mul_assoc, ← pow_add,
    map_frobeniusPoly.key₂ p hi.le hj, Nat.cast_mul, Nat.cast_pow]
  /-
    case h.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    h1 : Eq (HMul.hMul (HPow.hPow (↑p) n) (HPow.hPow (Invertible.invOf ↑p) n)) 1
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j (HPow.hPow p (HSub.hSub n i))
    ⊢ Eq (HMul.hMul (↑p) (HMul.hMul (↑((HPow.hPow p (HSub.hSub n i)).choose (HAdd. …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem frobeniusPoly_zmod (n : ℕ) :
    MvPolynomial.map (Int.castRingHom (ZMod p)) (frobeniusPoly p n) = X n ^ p := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.map (Int.castRingHom (ZMod p))) (WittVector.frobeniusPoly  …
  -/
  rw [frobeniusPoly, RingHom.map_add, RingHom.map_pow, RingHom.map_mul, map_X, map_C]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (HAdd.hAdd (HPow.hPow (MvPolynomial.X n) p) (HMul.hMul (MvPolynomial.C (( …
  -/
  simp only [Int.cast_natCast, add_zero, eq_intCast, ZMod.natCast_self, zero_mul, C_0]
  /-
    🎉 no goals
  -/


@[simp]
theorem bind₁_frobeniusPoly_wittPolynomial (n : ℕ) :
    bind₁ (frobeniusPoly p) (wittPolynomial p ℤ n) = wittPolynomial p ℤ (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((MvPolynomial.bind₁ (WittVector.frobeniusPoly p)) (wittPolynomial p Int  …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [map_bind₁, map_frobeniusPoly, bind₁_frobeniusPolyRat_wittPolynomial,
    map_wittPolynomial]


/-- `frobeniusFun` is the function underlying the ring endomorphism
`frobenius : 𝕎 R →+* frobenius 𝕎 R`. -/
def frobeniusFun (x : 𝕎 R) : 𝕎 R :=
  mk p fun n => MvPolynomial.aeval x.coeff (frobeniusPoly p n)


omit hp in
theorem coeff_frobeniusFun (x : 𝕎 R) (n : ℕ) :
    coeff (frobeniusFun x) n = MvPolynomial.aeval x.coeff (frobeniusPoly p n) := by
  /-
    p : Nat
    R : Type u_1
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq (x.frobeniusFun.coeff n) ((MvPolynomial.aeval x.coeff) (WittVector.froben …
  -/
  rw [frobeniusFun, coeff_mk]
  /-
    🎉 no goals
  -/


/-- `frobeniusFun` is tautologically a polynomial function.

See also `frobenius_isPoly`. -/
-- Porting note: replaced `@[is_poly]` with `instance`.
instance frobeniusFun_isPoly : IsPoly p fun R _ Rcr => @frobeniusFun p R _ Rcr :=
                        /-
                          p : Nat
                          R : Type u_1
                          hp : Fact (Nat.Prime p)
                          inst✝ : CommRing R
                          ⊢ ∀ ⦃R : Type u_2⦄ [inst : CommRing R] (x : WittVector p R), Eq x.frobeniusFun …
                        -/
  ⟨⟨frobeniusPoly p, by intros; funext n; apply coeff_frobeniusFun⟩⟩
                                          /-
                                            🎉 no goals
                                          -/


@[ghost_simps]
theorem ghostComponent_frobeniusFun (n : ℕ) (x : 𝕎 R) :
    ghostComponent n (frobeniusFun x) = ghostComponent (n + 1) x := by
  simp only [ghostComponent_apply, frobeniusFun, coeff_mk, ← bind₁_frobeniusPoly_wittPolynomial,
    aeval_bind₁]


/-- If `R` has characteristic `p`, then there is a ring endomorphism
that raises `r : R` to the power `p`.
By applying `WittVector.map` to this endomorphism,
we obtain a ring endomorphism `frobenius R p : 𝕎 R →+* 𝕎 R`.

The underlying function of this morphism is `WittVector.frobeniusFun`.
-/
def frobenius : 𝕎 R →+* 𝕎 R where
  toFun := frobeniusFun
  map_zero' := by
    -- Porting note: removing the placeholders give an error
    refine IsPoly.ext (@IsPoly.comp p _ _ (frobeniusFun_isPoly p) WittVector.zeroIsPoly)
      (@IsPoly.comp p _ _ WittVector.zeroIsPoly
      (frobeniusFun_isPoly p)) ?_ _ 0
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      ⊢ ∀ (R : Type u_1) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    -/
    simp only [Function.comp_apply, map_zero, forall_const]
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      ⊢ ∀ (R : Type u_1) [_Rcr : CommRing R] (n : Nat), Eq ((WittVector.ghostCompone …
    -/
    ghost_simp
    /-
      🎉 no goals
    -/
  map_one' := by
    refine
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      ⊢ ∀ (R : Type u_1) [_Rcr : CommRing R] (x : WittVector p R) (n : Nat), Eq ((Wi …
    -/
      -- Porting note: removing the placeholders give an error
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      ⊢ ∀ (R : Type u_1) [_Rcr : CommRing R] (n : Nat), Eq ((WittVector.ghostCompone …
    -/
      IsPoly.ext (@IsPoly.comp p _ _ (frobeniusFun_isPoly p) WittVector.oneIsPoly)
    /-
      🎉 no goals
    -/
        (@IsPoly.comp p _ _ WittVector.oneIsPoly (frobeniusFun_isPoly p)) ?_ _ 0
                 /-
                   p : Nat
                   R : Type u_1
                   hp : Fact (Nat.Prime p)
                   inst✝ : CommRing R
                   ⊢ ∀ (x y : WittVector p R), Eq ({ toFun := WittVector.frobeniusFun, map_one' : …
                 -/
    simp only [Function.comp_apply, map_one, forall_const]
                                             /-
                                               🎉 no goals
                                             -/
    ghost_simp
                 /-
                   p : Nat
                   R : Type u_1
                   hp : Fact (Nat.Prime p)
                   inst✝ : CommRing R
                   ⊢ ∀ (x y : WittVector p R), Eq ((↑{ toFun := WittVector.frobeniusFun, map_one' …
                 -/
  map_add' := by dsimp only; ghost_calc _ _; ghost_simp
                                             /-
                                               🎉 no goals
                                             -/
  map_mul' := by dsimp only; ghost_calc _ _; ghost_simp


theorem coeff_frobenius (x : 𝕎 R) (n : ℕ) :
    coeff (frobenius x) n = MvPolynomial.aeval x.coeff (frobeniusPoly p n) :=
  coeff_frobeniusFun _ _


@[ghost_simps]
theorem ghostComponent_frobenius (n : ℕ) (x : 𝕎 R) :
    ghostComponent n (frobenius x) = ghostComponent (n + 1) x :=
  ghostComponent_frobeniusFun _ _


/-- `frobenius` is tautologically a polynomial function. -/
-- Porting note: replaced `@[is_poly]` with `instance`.
instance frobenius_isPoly : IsPoly p fun R _Rcr => @frobenius p R _ _Rcr :=
  frobeniusFun_isPoly _


@[simp]
theorem coeff_frobenius_charP (x : 𝕎 R) (n : ℕ) : coeff (frobenius x) n = x.coeff n ^ p := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    n : Nat
    ⊢ Eq ((WittVector.frobenius x).coeff n) (HPow.hPow (x.coeff n) p)
  -/
  rw [coeff_frobenius]
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    n : Nat
    ⊢ Eq ((MvPolynomial.aeval x.coeff) (WittVector.frobeniusPoly p n)) (HPow.hPow  …
  -/
  letI : Algebra (ZMod p) R := ZMod.algebra _ _
  -- outline of the calculation, proofs follow below
  calc
    aeval (fun k => x.coeff k) (frobeniusPoly p n) =
        aeval (fun k => x.coeff k)
          (MvPolynomial.map (Int.castRingHom (ZMod p)) (frobeniusPoly p n)) := ?_
    _ = aeval (fun k => x.coeff k) (X n ^ p : MvPolynomial ℕ (ZMod p)) := ?_
    _ = x.coeff n ^ p := ?_
    /-
      case calc_1
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      n : Nat
      this : Algebra (ZMod p) R := ZMod.algebra R p
      ⊢ Eq ((MvPolynomial.aeval fun k => x.coeff k) (WittVector.frobeniusPoly p n))  …
    -/
  · conv_rhs => rw [aeval_eq_eval₂Hom, eval₂Hom_map_hom]
    /-
      case calc_1
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      n : Nat
      this : Algebra (ZMod p) R := ZMod.algebra R p
      ⊢ Eq ((MvPolynomial.aeval fun k => x.coeff k) (WittVector.frobeniusPoly p n))  …
    -/
    apply eval₂Hom_congr (RingHom.ext_int _ _) rfl rfl
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      n : Nat
      this : Algebra (ZMod p) R := ZMod.algebra R p
      ⊢ Eq ((MvPolynomial.aeval fun k => x.coeff k) ((MvPolynomial.map (Int.castRing …
    -/
  · rw [frobeniusPoly_zmod]
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      n : Nat
      this : Algebra (ZMod p) R := ZMod.algebra R p
      ⊢ Eq ((MvPolynomial.aeval fun k => x.coeff k) (HPow.hPow (MvPolynomial.X n) p) …
    -/
  · rw [map_pow, aeval_X]
    /-
      🎉 no goals
    -/


theorem frobenius_eq_map_frobenius : @frobenius p R _ _ = map (_root_.frobenius R p) := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    ⊢ Eq WittVector.frobenius (WittVector.map (_root_.frobenius R p))
  -/
  ext (x n)
  /-
    case a.h
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    n : Nat
    ⊢ Eq ((WittVector.frobenius x).coeff n) (((WittVector.map (_root_.frobenius R  …
  -/
  simp only [coeff_frobenius_charP, map_coeff, frobenius_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem frobenius_zmodp (x : 𝕎 (ZMod p)) : frobenius x = x := by
  simp only [WittVector.ext_iff, coeff_frobenius_charP, ZMod.pow_card, eq_self_iff_true,
    forall_const]


/-- `WittVector.frobenius` as an equiv. -/
@[simps (config := .asFn)]
def frobeniusEquiv [PerfectRing R p] : WittVector p R ≃+* WittVector p R :=
  { (WittVector.frobenius : WittVector p R →+* WittVector p R) with
    toFun := WittVector.frobenius
    invFun := map (_root_.frobeniusEquiv R p).symm
    left_inv := fun f => ext fun n => by
      /-
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝² : CommRing R
        inst✝¹ : CharP R p
        inst✝ : PerfectRing R p
        f : WittVector p R
        n : Nat
        ⊢ Eq (((WittVector.map ↑(_root_.frobeniusEquiv R p).symm) (WittVector.frobeniu …
      -/
      rw [frobenius_eq_map_frobenius]
      /-
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝² : CommRing R
        inst✝¹ : CharP R p
        inst✝ : PerfectRing R p
        f : WittVector p R
        n : Nat
        ⊢ Eq (((WittVector.map ↑(_root_.frobeniusEquiv R p).symm) ((WittVector.map (_r …
      -/
      exact frobeniusEquiv_symm_apply_frobenius R p _
      /-
        🎉 no goals
      -/
    right_inv := fun f => ext fun n => by
      /-
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝² : CommRing R
        inst✝¹ : CharP R p
        inst✝ : PerfectRing R p
        f : WittVector p R
        n : Nat
        ⊢ Eq ((WittVector.frobenius ((WittVector.map ↑(_root_.frobeniusEquiv R p).symm …
      -/
      rw [frobenius_eq_map_frobenius]
      /-
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝² : CommRing R
        inst✝¹ : CharP R p
        inst✝ : PerfectRing R p
        f : WittVector p R
        n : Nat
        ⊢ Eq (((WittVector.map (_root_.frobenius R p)) ((WittVector.map ↑(_root_.frobe …
      -/
      exact frobenius_apply_frobeniusEquiv_symm R p _ }
      /-
        🎉 no goals
      -/


theorem frobenius_bijective [PerfectRing R p] :
    Function.Bijective (@WittVector.frobenius p R _ _) :=
  (frobeniusEquiv p R).bijective


