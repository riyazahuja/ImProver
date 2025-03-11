/-- Express the value of `cosZeta` at a positive even integer as a value
of the Bernoulli polynomial. -/
theorem cosZeta_two_mul_nat (hk : k ≠ 0) (hx : x ∈ Icc 0 1) :
    cosZeta x (2 * k) = (-1) ^ (k + 1) * (2 * π) ^ (2 * k) / 2 / (2 * k)! *
      ((Polynomial.bernoulli (2 * k)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.cosZeta (↑x) (HMul.hMul 2 ↑k)) (HMul.hMul (HDiv.hDiv (HDiv.h …
  -/
  rw [← (hasSum_nat_cosZeta x (?_ : 1 < re (2 * k))).tsum_eq]
  · refine Eq.trans ?_ <|
      (congr_arg ofReal (hasSum_one_div_nat_pow_mul_cos hk hx).tsum_eq).trans ?_
      /-
        case refine_1
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Eq (tsum fun b => HDiv.hDiv (↑(Real.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 R …
      -/
    · rw [ofReal_tsum]
      /-
        case refine_1
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Eq (tsum fun b => HDiv.hDiv (↑(Real.cos (HMul.hMul (HMul.hMul (HMul.hMul 2 R …
      -/
      refine tsum_congr fun n ↦ ?_
      rw [mul_comm (1 / _), mul_one_div, ofReal_div, mul_assoc (2 * π), mul_comm x n, ← mul_assoc,
        ← Nat.cast_ofNat (R := ℂ), ← Nat.cast_mul, cpow_natCast, ofReal_pow, ofReal_natCast]
    · simp only [ofReal_mul, ofReal_div, ofReal_pow, ofReal_natCast, ofReal_ofNat,
        ofReal_neg, ofReal_one]
      /-
        case refine_2
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
      -/
      congr 1
      have : (Polynomial.bernoulli (2 * k)).map (algebraMap ℚ ℂ) = _ :=
        (Polynomial.map_map (algebraMap ℚ ℝ) ofRealHom _).symm
      /-
        case refine_2.e_a
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        this : Eq (Polynomial.map (algebraMap Rat Complex) (Polynomial.bernoulli (HMul …
        ⊢ Eq (↑(Polynomial.eval x (Polynomial.map (algebraMap Rat Real) (Polynomial.be …
      -/
      rw [this, ← ofRealHom_eq_coe, ← ofRealHom_eq_coe]
      /-
        case refine_2.e_a
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        this : Eq (Polynomial.map (algebraMap Rat Complex) (Polynomial.bernoulli (HMul …
        ⊢ Eq (Complex.ofRealHom (Polynomial.eval x (Polynomial.map (algebraMap Rat Rea …
      -/
      apply Polynomial.map_aeval_eq_aeval_map
      /-
        case refine_2.e_a.h
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        this : Eq (Polynomial.map (algebraMap Rat Complex) (Polynomial.bernoulli (HMul …
        ⊢ Eq ((algebraMap Complex Complex).comp Complex.ofRealHom) (Complex.ofRealHom. …
      -/
      simp only [Algebra.id.map_eq_id, RingHomCompTriple.comp_eq]
      /-
        🎉 no goals
      -/
    /-
      k : Nat
      x : Real
      hk : Ne k 0
      hx : Membership.mem (Set.Icc 0 1) x
      ⊢ LT.lt 1 (HMul.hMul 2 ↑k).re
    -/
  · rw [← Nat.cast_ofNat, ← Nat.cast_one, ← Nat.cast_mul, natCast_re, Nat.cast_lt]
    /-
      k : Nat
      x : Real
      hk : Ne k 0
      hx : Membership.mem (Set.Icc 0 1) x
      ⊢ LT.lt 1 (HMul.hMul 2 k)
    -/
    omega
    /-
      🎉 no goals
    -/


/--
Express the value of `sinZeta` at an odd integer `> 1` as a value of the Bernoulli polynomial.

Note that this formula is also correct for `k = 0` (i.e. for the value at `s = 1`), but we do not
prove it in this case, owing to the additional difficulty of working with series that are only
conditionally convergent.
-/
theorem sinZeta_two_mul_nat_add_one (hk : k ≠ 0) (hx : x ∈ Icc 0 1) :
    sinZeta x (2 * k + 1) = (-1) ^ (k + 1) * (2 * π) ^ (2 * k + 1) / 2 / (2 * k + 1)! *
      ((Polynomial.bernoulli (2 * k + 1)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.sinZeta (↑x) (HAdd.hAdd (HMul.hMul 2 ↑k) 1)) (HMul.hMul (HDi …
  -/
  rw [← (hasSum_nat_sinZeta x (?_ : 1 < re (2 * k + 1))).tsum_eq]
  · refine Eq.trans ?_ <|
      (congr_arg ofReal (hasSum_one_div_nat_pow_mul_sin hk hx).tsum_eq).trans ?_
      /-
        case refine_1
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Eq (tsum fun b => HDiv.hDiv (↑(Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 R …
      -/
    · rw [ofReal_tsum]
      /-
        case refine_1
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Eq (tsum fun b => HDiv.hDiv (↑(Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 R …
      -/
      refine tsum_congr fun n ↦ ?_
      /-
        case refine_1
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        n : Nat
        ⊢ Eq (HDiv.hDiv (↑(Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) x) ↑n) …
      -/
      rw [mul_comm (1 / _), mul_one_div, ofReal_div, mul_assoc (2 * π), mul_comm x n, ← mul_assoc]
      /-
        case refine_1
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        n : Nat
        ⊢ Eq (HDiv.hDiv (↑(Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) ↑n) x) …
      -/
      congr 1
      rw [← Nat.cast_ofNat, ← Nat.cast_mul, ← Nat.cast_add_one, cpow_natCast, ofReal_pow,
        ofReal_natCast]
    · simp only [ofReal_mul, ofReal_div, ofReal_pow, ofReal_natCast, ofReal_ofNat,
        ofReal_neg, ofReal_one]
      /-
        case refine_2
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
      -/
      congr 1
      have : (Polynomial.bernoulli (2 * k + 1)).map (algebraMap ℚ ℂ) = _ :=
        (Polynomial.map_map (algebraMap ℚ ℝ) ofRealHom _).symm
      /-
        case refine_2.e_a
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        this : Eq (Polynomial.map (algebraMap Rat Complex) (Polynomial.bernoulli (HAdd …
        ⊢ Eq (↑(Polynomial.eval x (Polynomial.map (algebraMap Rat Real) (Polynomial.be …
      -/
      rw [this, ← ofRealHom_eq_coe, ← ofRealHom_eq_coe]
      /-
        case refine_2.e_a
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        this : Eq (Polynomial.map (algebraMap Rat Complex) (Polynomial.bernoulli (HAdd …
        ⊢ Eq (Complex.ofRealHom (Polynomial.eval x (Polynomial.map (algebraMap Rat Rea …
      -/
      apply Polynomial.map_aeval_eq_aeval_map
      /-
        case refine_2.e_a.h
        k : Nat
        x : Real
        hk : Ne k 0
        hx : Membership.mem (Set.Icc 0 1) x
        this : Eq (Polynomial.map (algebraMap Rat Complex) (Polynomial.bernoulli (HAdd …
        ⊢ Eq ((algebraMap Complex Complex).comp Complex.ofRealHom) (Complex.ofRealHom. …
      -/
      simp only [Algebra.id.map_eq_id, RingHomCompTriple.comp_eq]
      /-
        🎉 no goals
      -/
  · rw [← Nat.cast_ofNat, ← Nat.cast_one, ← Nat.cast_mul, ← Nat.cast_add_one, natCast_re,
      Nat.cast_lt, lt_add_iff_pos_left]
    /-
      k : Nat
      x : Real
      hk : Ne k 0
      hx : Membership.mem (Set.Icc 0 1) x
      ⊢ LT.lt 0 (HMul.hMul 2 k)
    -/
    exact mul_pos two_pos (Nat.pos_of_ne_zero hk)
    /-
      🎉 no goals
    -/


/-- Reformulation of `cosZeta_two_mul_nat` using `Gammaℂ`. -/
theorem cosZeta_two_mul_nat' (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    cosZeta x (2 * k) = (-1) ^ (k + 1) / (2 * k) / Gammaℂ (2 * k) *
      ((Polynomial.bernoulli (2 * k)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.cosZeta (↑x) (HMul.hMul 2 ↑k)) (HMul.hMul (HDiv.hDiv (HDiv.h …
  -/
  rw [cosZeta_two_mul_nat hk hx]
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
  -/
  congr 1
  have : (2 * k)! = (2 * k) * Complex.Gamma (2 * k) := by
    rw [(by { norm_cast; omega } : 2 * (k : ℂ) = ↑(2 * k - 1) + 1), Complex.Gamma_nat_eq_factorial,
      ← Nat.cast_add_one, ← Nat.cast_mul, ← Nat.factorial_succ, Nat.sub_add_cancel (by omega)]
  simp_rw [this, Gammaℂ, cpow_neg, ← div_div, div_inv_eq_mul, div_mul_eq_mul_div, div_div,
    mul_right_comm (2 : ℂ) (k : ℂ)]
  /-
    case e_a
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    this : Eq (↑(HMul.hMul 2 k).factorial) (HMul.hMul (HMul.hMul 2 ↑k) (Complex.Ga …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k 1)) (HPow.hPow (HMul.h …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- Reformulation of `sinZeta_two_mul_nat_add_one` using `Gammaℂ`. -/
theorem sinZeta_two_mul_nat_add_one' (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    sinZeta x (2 * k + 1) = (-1) ^ (k + 1) / (2 * k + 1) / Gammaℂ (2 * k + 1) *
      ((Polynomial.bernoulli (2 * k + 1)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.sinZeta (↑x) (HAdd.hAdd (HMul.hMul 2 ↑k) 1)) (HMul.hMul (HDi …
  -/
  rw [sinZeta_two_mul_nat_add_one hk hx]
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
  -/
  congr 1
  have : (2 * k + 1)! = (2 * k + 1) * Complex.Gamma (2 * k + 1) := by
    rw [(by simp : Complex.Gamma (2 * k + 1) = Complex.Gamma (↑(2 * k) + 1)),
       Complex.Gamma_nat_eq_factorial, ← Nat.cast_ofNat (R := ℂ), ← Nat.cast_mul,
      ← Nat.cast_add_one, ← Nat.cast_mul, ← Nat.factorial_succ]
  /-
    case e_a
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    this : Eq (↑(HAdd.hAdd (HMul.hMul 2 k) 1).factorial) (HMul.hMul (HAdd.hAdd (HM …
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k 1)) (HPow.h …
  -/
  simp_rw [this, Gammaℂ, cpow_neg, ← div_div, div_inv_eq_mul, div_mul_eq_mul_div, div_div]
  /-
    case e_a
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    this : Eq (↑(HAdd.hAdd (HMul.hMul 2 k) 1).factorial) (HMul.hMul (HAdd.hAdd (HM …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k 1)) (HPow.hPow (HMul.h …
  -/
  rw [(by simp : 2 * (k : ℂ) + 1 = ↑(2 * k + 1)), cpow_natCast]
  /-
    case e_a
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    this : Eq (↑(HAdd.hAdd (HMul.hMul 2 k) 1).factorial) (HMul.hMul (HAdd.hAdd (HM …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k 1)) (HPow.hPow (HMul.h …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem hurwitzZetaEven_one_sub_two_mul_nat (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    hurwitzZetaEven x (1 - 2 * k) =
      -1 / (2 * k) * ((Polynomial.bernoulli (2 * k)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  have h1 (n : ℕ) : (2 * k : ℂ) ≠ -n := by
    rw [← Int.cast_ofNat, ← Int.cast_natCast, ← Int.cast_mul, ← Int.cast_natCast n, ← Int.cast_neg,
      Ne, Int.cast_inj, ← Ne]
    refine ne_of_gt ((neg_nonpos_of_nonneg n.cast_nonneg).trans_lt (mul_pos two_pos ?_))
    exact Nat.cast_pos.mpr (Nat.pos_of_ne_zero hk)
  have h2 : (2 * k : ℂ) ≠ 1 := by norm_cast; simp only [mul_eq_one, OfNat.ofNat_ne_one,
    false_and, not_false_eq_true]
  have h3 : Gammaℂ (2 * k) ≠ 0 := by
    refine mul_ne_zero (mul_ne_zero two_ne_zero ?_) (Gamma_ne_zero h1)
    simp only [ne_eq, cpow_eq_zero_iff, mul_eq_zero, OfNat.ofNat_ne_zero, ofReal_eq_zero,
      pi_ne_zero, Nat.cast_eq_zero, false_or, false_and, not_false_eq_true]
  rw [hurwitzZetaEven_one_sub _ h1 (Or.inr h2), ← Gammaℂ, cosZeta_two_mul_nat' hk hx, ← mul_assoc,
    ← mul_div_assoc, mul_assoc, mul_div_cancel_left₀ _ h3, ← mul_div_assoc]
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    h1 : ∀ (n : Nat), Ne (HMul.hMul 2 ↑k) (Neg.neg ↑n)
    h2 : Ne (HMul.hMul 2 ↑k) 1
    h3 : Ne (HMul.hMul 2 ↑k).Gammaℂ 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HMul.hMul (Complex.cos (HDiv.hDiv (HMul.hMul (↑Rea …
  -/
  congr 2
  rw [mul_div_assoc, mul_div_cancel_left₀ _ two_ne_zero, ← ofReal_natCast, ← ofReal_mul,
    ← ofReal_cos, mul_comm π, ← sub_zero (k * π), cos_nat_mul_pi_sub, Real.cos_zero, mul_one,
    ofReal_pow, ofReal_neg, ofReal_one, pow_succ, mul_neg_one, mul_neg, ← mul_pow, neg_one_mul,
    neg_neg, one_pow]


theorem hurwitzZetaOdd_neg_two_mul_nat (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    hurwitzZetaOdd x (-(2 * k)) =
    -1 / (2 * k + 1) * ((Polynomial.bernoulli (2 * k + 1)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  have h1 (n : ℕ) : (2 * k + 1 : ℂ) ≠ -n := by
    rw [← Int.cast_ofNat, ← Int.cast_natCast, ← Int.cast_mul, ← Int.cast_natCast n, ← Int.cast_neg,
      ← Int.cast_one, ← Int.cast_add, Ne, Int.cast_inj, ← Ne]
    refine ne_of_gt ((neg_nonpos_of_nonneg n.cast_nonneg).trans_lt ?_)
    positivity
  have h3 : Gammaℂ (2 * k + 1) ≠ 0 := by
    refine mul_ne_zero (mul_ne_zero two_ne_zero ?_) (Gamma_ne_zero h1)
    simp only [ne_eq, cpow_eq_zero_iff, mul_eq_zero, OfNat.ofNat_ne_zero, ofReal_eq_zero,
      pi_ne_zero, Nat.cast_eq_zero, false_or, false_and, not_false_eq_true]
  rw [(by simp : -(2 * k : ℂ) = 1 - (2 * k + 1)),
    hurwitzZetaOdd_one_sub _ h1, ← Gammaℂ, sinZeta_two_mul_nat_add_one' hk hx, ← mul_assoc,
    ← mul_div_assoc, mul_assoc, mul_div_cancel_left₀ _ h3, ← mul_div_assoc]
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    h1 : ∀ (n : Nat), Ne (HAdd.hAdd (HMul.hMul 2 ↑k) 1) (Neg.neg ↑n)
    h3 : Ne (HAdd.hAdd (HMul.hMul 2 ↑k) 1).Gammaℂ 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HMul.hMul (Complex.sin (HDiv.hDiv (HMul.hMul (↑Rea …
  -/
  congr 2
  rw [mul_div_assoc, add_div, mul_div_cancel_left₀ _ two_ne_zero, ← ofReal_natCast,
    ← ofReal_one, ← ofReal_ofNat, ← ofReal_div, ← ofReal_add, ← ofReal_mul,
    ← ofReal_sin, mul_comm π, add_mul, mul_comm (1 / 2), mul_one_div, Real.sin_add_pi_div_two,
    ← sub_zero (k * π), cos_nat_mul_pi_sub, Real.cos_zero, mul_one,
    ofReal_pow, ofReal_neg, ofReal_one, pow_succ, mul_neg_one, mul_neg, ← mul_pow, neg_one_mul,
    neg_neg, one_pow]

-- private because it is superseded by `hurwitzZeta_neg_nat` below

private lemma hurwitzZeta_one_sub_two_mul_nat (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    hurwitzZeta x (1 - 2 * k) =
      -1 / (2 * k) * ((Polynomial.bernoulli (2 * k)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  suffices hurwitzZetaOdd x (1 - 2 * k) = 0 by
    rw [hurwitzZeta, this, add_zero, hurwitzZetaEven_one_sub_two_mul_nat hk hx]
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.hurwitzZetaOdd (↑x) (HSub.hSub 1 (HMul.hMul 2 ↑k))) 0
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hk
  rw [Nat.cast_succ, show (1 : ℂ) - 2 * (k + 1) = - 2 * k - 1 by ring,
    hurwitzZetaOdd_neg_two_mul_nat_sub_one]

-- private because it is superseded by `hurwitzZeta_neg_nat` below

private lemma hurwitzZeta_neg_two_mul_nat (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    hurwitzZeta x (-(2 * k)) = -1 / (2 * k + 1) *
      ((Polynomial.bernoulli (2 * k + 1)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  suffices hurwitzZetaEven x (-(2 * k)) = 0 by
    rw [hurwitzZeta, this, zero_add, hurwitzZetaOdd_neg_two_mul_nat hk hx]
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.hurwitzZetaEven (↑x) (Neg.neg (HMul.hMul 2 ↑k))) 0
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hk
  /-
    case intro
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    k : Nat
    hk : Ne k.succ 0
    ⊢ Eq (HurwitzZeta.hurwitzZetaEven (↑x) (Neg.neg (HMul.hMul 2 ↑k.succ))) 0
  -/
  simpa only [Nat.cast_succ, ← neg_mul] using hurwitzZetaEven_neg_two_mul_nat_add_one x k
  /-
    🎉 no goals
  -/


/-- Values of Hurwitz zeta functions at (strictly) negative integers.

TODO: This formula is also correct for `k = 0`; but our current proof does not work in this
case. -/
theorem hurwitzZeta_neg_nat (hk : k ≠ 0) (hx : x ∈ Icc (0 : ℝ) 1) :
    hurwitzZeta x (-k) =
    -1 / (k + 1) * ((Polynomial.bernoulli (k + 1)).map (algebraMap ℚ ℂ)).eval (x : ℂ) := by
  /-
    k : Nat
    x : Real
    hk : Ne k 0
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ Eq (HurwitzZeta.hurwitzZeta (↑x) (Neg.neg ↑k)) (HMul.hMul (HDiv.hDiv (-1) (H …
  -/
  rcases Nat.even_or_odd' k with ⟨n, (rfl | rfl)⟩
    /-
      case intro.inl
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      hk : Ne (HMul.hMul 2 n) 0
      ⊢ Eq (HurwitzZeta.hurwitzZeta (↑x) (Neg.neg ↑(HMul.hMul 2 n))) (HMul.hMul (HDi …
    -/
  · exact_mod_cast hurwitzZeta_neg_two_mul_nat (by omega : n ≠ 0) hx
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      hk : Ne (HAdd.hAdd (HMul.hMul 2 n) 1) 0
      ⊢ Eq (HurwitzZeta.hurwitzZeta (↑x) (Neg.neg ↑(HAdd.hAdd (HMul.hMul 2 n) 1))) ( …
    -/
  · exact_mod_cast hurwitzZeta_one_sub_two_mul_nat (by omega : n + 1 ≠ 0) hx
    /-
      🎉 no goals
    -/


/-- Explicit formula for `ζ (2 * k)`, for `k ∈ ℕ` with `k ≠ 0`, in terms of the Bernoulli number
`bernoulli (2 * k)`.

Compare `hasSum_zeta_nat` for a version formulated in terms of a sum over `1 / n ^ (2 * k)`, and
`riemannZeta_neg_nat_eq_bernoulli` for values at negative integers (equivalent to the above via
the functional equation). -/
theorem riemannZeta_two_mul_nat {k : ℕ} (hk : k ≠ 0) :
    riemannZeta (2 * k) = (-1) ^ (k + 1) * (2 : ℂ) ^ (2 * k - 1)
      * (π : ℂ) ^ (2 * k) * bernoulli (2 * k) / (2 * k)! := by
  /-
    k : Nat
    hk : Ne k 0
    ⊢ Eq (riemannZeta (HMul.hMul 2 ↑k)) (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMu …
  -/
  convert congr_arg ((↑) : ℝ → ℂ) (hasSum_zeta_nat hk).tsum_eq
    /-
      case h.e'_2
      k : Nat
      hk : Ne k 0
      ⊢ Eq (riemannZeta (HMul.hMul 2 ↑k)) ↑(tsum fun b => HDiv.hDiv 1 (HPow.hPow (↑b …
    -/
  · rw [← Nat.cast_two, ← Nat.cast_mul, zeta_nat_eq_tsum_of_gt_one (by omega)]
    /-
      case h.e'_2
      k : Nat
      hk : Ne k 0
      ⊢ Eq (tsum fun n => HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) ↑(tsum fun b …
    -/
    simp only [push_cast]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      k : Nat
      hk : Ne k 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


theorem riemannZeta_two : riemannZeta 2 = (π : ℂ) ^ 2 / 6 := by
  /-
    ⊢ Eq (riemannZeta 2) (HDiv.hDiv (HPow.hPow (↑Real.pi) 2) 6)
  -/
  convert congr_arg ((↑) : ℝ → ℂ) hasSum_zeta_two.tsum_eq
    /-
      case h.e'_2
      ⊢ Eq (riemannZeta 2) ↑(tsum fun b => HDiv.hDiv 1 (HPow.hPow (↑b) 2))
    -/
  · rw [← Nat.cast_two, zeta_nat_eq_tsum_of_gt_one one_lt_two]
    /-
      case h.e'_2
      ⊢ Eq (tsum fun n => HDiv.hDiv 1 (HPow.hPow (↑n) 2)) ↑(tsum fun b => HDiv.hDiv  …
    -/
    simp only [push_cast]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      ⊢ Eq (HDiv.hDiv (HPow.hPow (↑Real.pi) 2) 6) ↑(HDiv.hDiv (HPow.hPow Real.pi 2) 6)
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


theorem riemannZeta_four : riemannZeta 4 = π ^ 4 / 90 := by
  /-
    ⊢ Eq (riemannZeta 4) (HDiv.hDiv (HPow.hPow (↑Real.pi) 4) 90)
  -/
  convert congr_arg ((↑) : ℝ → ℂ) hasSum_zeta_four.tsum_eq
  · rw [← Nat.cast_one, show (4 : ℂ) = (4 : ℕ) by norm_num,
      zeta_nat_eq_tsum_of_gt_one (by norm_num : 1 < 4)]
    /-
      case h.e'_2
      ⊢ Eq (tsum fun n => HDiv.hDiv 1 (HPow.hPow (↑n) 4)) ↑(tsum fun b => HDiv.hDiv  …
    -/
    simp only [push_cast]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      ⊢ Eq (HDiv.hDiv (HPow.hPow (↑Real.pi) 4) 90) ↑(HDiv.hDiv (HPow.hPow Real.pi 4) …
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


/-- Value of Riemann zeta at `-ℕ` in terms of `bernoulli'`. -/
theorem riemannZeta_neg_nat_eq_bernoulli' (k : ℕ) :
    riemannZeta (-k) = -bernoulli' (k + 1) / (k + 1) := by
  /-
    k : Nat
    ⊢ Eq (riemannZeta (Neg.neg ↑k)) (HDiv.hDiv (Neg.neg ↑(bernoulli' (HAdd.hAdd k  …
  -/
  rcases eq_or_ne k 0 with rfl | hk
  · rw [Nat.cast_zero, neg_zero, riemannZeta_zero, zero_add, zero_add, div_one,
      bernoulli'_one, Rat.cast_div, Rat.cast_one, Rat.cast_ofNat, neg_div]
  · rw [← hurwitzZeta_zero, ← QuotientAddGroup.mk_zero, hurwitzZeta_neg_nat hk
      (left_mem_Icc.mpr zero_le_one), ofReal_zero, Polynomial.eval_zero_map,
      Polynomial.bernoulli_eval_zero, Algebra.algebraMap_eq_smul_one, Rat.smul_one_eq_cast,
      div_mul_eq_mul_div, neg_one_mul, bernoulli_eq_bernoulli'_of_ne_one (by simp [hk])]


/-- Value of Riemann zeta at `-ℕ` in terms of `bernoulli`. -/
theorem riemannZeta_neg_nat_eq_bernoulli (k : ℕ) :
    riemannZeta (-k) = (-1 : ℂ) ^ k * bernoulli (k + 1) / (k + 1) := by
  rw [riemannZeta_neg_nat_eq_bernoulli', bernoulli, Rat.cast_mul, Rat.cast_pow, Rat.cast_neg,
    Rat.cast_one, ← neg_one_mul, ← mul_assoc, pow_succ, ← mul_assoc, ← mul_pow, neg_one_mul (-1),
    neg_neg, one_pow, one_mul]

