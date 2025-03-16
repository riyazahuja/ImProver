theorem Complex.hasSum_cos' (z : ℂ) :
    HasSum (fun n : ℕ => (z * Complex.I) ^ (2 * n) / ↑(2 * n)!) (Complex.cos z) := by
  /-
    z : Complex
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HMul.hMul 2 n …
  -/
  rw [Complex.cos, Complex.exp_eq_exp_ℂ]
  have := ((expSeries_div_hasSum_exp ℂ (z * Complex.I)).add
    (expSeries_div_hasSum_exp ℂ (-z * Complex.I))).div_const 2
  /-
    z : Complex
    this : HasSum (fun i => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul  …
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HMul.hMul 2 n …
  -/
  replace := (Nat.divModEquiv 2).symm.hasSum_iff.mpr this
  /-
    z : Complex
    this : HasSum (Function.comp (fun i => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.h …
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HMul.hMul 2 n …
  -/
  dsimp [Function.comp_def] at this
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul  …
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HMul.hMul 2 n …
  -/
  simp_rw [← mul_comm 2 _] at this
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul  …
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HMul.hMul 2 n …
  -/
  refine this.prod_fiberwise fun k => ?_
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul  …
    k : Nat
    ⊢ HasSum (fun c => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul z Com …
  -/
  dsimp only
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul  …
    k : Nat
    ⊢ HasSum (fun c => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul z Com …
  -/
  convert hasSum_fintype (_ : Fin 2 → ℂ) using 1
  /-
    case h.e'_6
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HPow.hPow (HMul.hMul  …
    k : Nat
    ⊢ Eq (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HMul.hMul 2 k)) ↑(HMul.hMu …
  -/
  rw [Fin.sum_univ_two]
  simp_rw [Fin.val_zero, Fin.val_one, add_zero, pow_succ, pow_mul, mul_pow, neg_sq, ← two_mul,
    neg_mul, mul_neg, neg_div, add_neg_cancel, zero_div, add_zero,
    mul_div_cancel_left₀ _ (two_ne_zero : (2 : ℂ) ≠ 0)]


theorem Complex.hasSum_sin' (z : ℂ) :
    HasSum (fun n : ℕ => (z * Complex.I) ^ (2 * n + 1) / ↑(2 * n + 1)! / Complex.I)
      (Complex.sin z) := by
  /-
    z : Complex
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HA …
  -/
  rw [Complex.sin, Complex.exp_eq_exp_ℂ]
  have := (((expSeries_div_hasSum_exp ℂ (-z * Complex.I)).sub
    (expSeries_div_hasSum_exp ℂ (z * Complex.I))).mul_right Complex.I).div_const 2
  /-
    z : Complex
    this : HasSum (fun i => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow  …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HA …
  -/
  replace := (Nat.divModEquiv 2).symm.hasSum_iff.mpr this
  /-
    z : Complex
    this : HasSum (Function.comp (fun i => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.h …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HA …
  -/
  dsimp [Function.comp_def] at this
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow  …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HA …
  -/
  simp_rw [← mul_comm 2 _] at this
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow  …
    ⊢ HasSum (fun n => HDiv.hDiv (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HA …
  -/
  refine this.prod_fiberwise fun k => ?_
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow  …
    k : Nat
    ⊢ HasSum (fun c => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow (HMul …
  -/
  dsimp only
  /-
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow  …
    k : Nat
    ⊢ HasSum (fun c => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow (HMul …
  -/
  convert hasSum_fintype (_ : Fin 2 → ℂ) using 1
  /-
    case h.e'_6
    z : Complex
    this : HasSum (fun x => HDiv.hDiv (HMul.hMul (HSub.hSub (HDiv.hDiv (HPow.hPow  …
    k : Nat
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HPow.hPow (HMul.hMul z Complex.I) (HAdd.hAdd (HMul …
  -/
  rw [Fin.sum_univ_two]
  simp_rw [Fin.val_zero, Fin.val_one, add_zero, pow_succ, pow_mul, mul_pow, neg_sq, sub_self,
    zero_mul, zero_div, zero_add, neg_mul, mul_neg, neg_div, ← neg_add', ← two_mul,
    neg_mul, neg_div, mul_assoc, mul_div_cancel_left₀ _ (two_ne_zero : (2 : ℂ) ≠ 0), Complex.div_I]


/-- The power series expansion of `Complex.cos`. -/
theorem Complex.hasSum_cos (z : ℂ) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n) / ↑(2 * n)!) (Complex.cos z) := by
  /-
    z : Complex
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HMul. …
  -/
  convert Complex.hasSum_cos' z using 1
  /-
    case h.e'_5
    z : Complex
    ⊢ Eq (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HMul.hMul …
  -/
  simp_rw [mul_pow, pow_mul, Complex.I_sq, mul_comm]
  /-
    🎉 no goals
  -/


/-- The power series expansion of `Complex.sin`. -/
theorem Complex.hasSum_sin (z : ℂ) :
    HasSum (fun n : ℕ => (-1) ^ n * z ^ (2 * n + 1) / ↑(2 * n + 1)!) (Complex.sin z) := by
  /-
    z : Complex
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow z (HAdd. …
  -/
  convert Complex.hasSum_sin' z using 1
  simp_rw [mul_pow, pow_succ, pow_mul, Complex.I_sq, ← mul_assoc, mul_div_assoc, div_right_comm,
    div_self Complex.I_ne_zero, mul_comm _ ((-1 : ℂ) ^ _), mul_one_div, mul_div_assoc, mul_assoc]


theorem Complex.cos_eq_tsum' (z : ℂ) :
    Complex.cos z = ∑' n : ℕ, (z * Complex.I) ^ (2 * n) / ↑(2 * n)! :=
  (Complex.hasSum_cos' z).tsum_eq.symm


theorem Complex.sin_eq_tsum' (z : ℂ) :
    Complex.sin z = ∑' n : ℕ, (z * Complex.I) ^ (2 * n + 1) / ↑(2 * n + 1)! / Complex.I :=
  (Complex.hasSum_sin' z).tsum_eq.symm


theorem Complex.cos_eq_tsum (z : ℂ) :
    Complex.cos z = ∑' n : ℕ, (-1) ^ n * z ^ (2 * n) / ↑(2 * n)! :=
  (Complex.hasSum_cos z).tsum_eq.symm


theorem Complex.sin_eq_tsum (z : ℂ) :
    Complex.sin z = ∑' n : ℕ, (-1) ^ n * z ^ (2 * n + 1) / ↑(2 * n + 1)! :=
  (Complex.hasSum_sin z).tsum_eq.symm


/-- The power series expansion of `Real.cos`. -/
theorem Real.hasSum_cos (r : ℝ) :
    HasSum (fun n : ℕ => (-1) ^ n * r ^ (2 * n) / ↑(2 * n)!) (Real.cos r) :=
  mod_cast Complex.hasSum_cos r


/-- The power series expansion of `Real.sin`. -/
theorem Real.hasSum_sin (r : ℝ) :
    HasSum (fun n : ℕ => (-1) ^ n * r ^ (2 * n + 1) / ↑(2 * n + 1)!) (Real.sin r) :=
  mod_cast Complex.hasSum_sin r


theorem Real.cos_eq_tsum (r : ℝ) : Real.cos r = ∑' n : ℕ, (-1) ^ n * r ^ (2 * n) / ↑(2 * n)! :=
  (Real.hasSum_cos r).tsum_eq.symm


theorem Real.sin_eq_tsum (r : ℝ) :
    Real.sin r = ∑' n : ℕ, (-1) ^ n * r ^ (2 * n + 1) / ↑(2 * n + 1)! :=
  (Real.hasSum_sin r).tsum_eq.symm


/-- The power series expansion of `Complex.cosh`. -/
lemma hasSum_cosh (z : ℂ) : HasSum (fun n ↦ z ^ (2 * n) / ↑(2 * n)!) (cosh z) := by
  /-
    z : Complex
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow z (HMul.hMul 2 n)) ↑(HMul.hMul 2 n).fa …
  -/
  simpa [mul_assoc, cos_mul_I] using hasSum_cos' (z * I)
  /-
    🎉 no goals
  -/


/-- The power series expansion of `Complex.sinh`. -/
lemma hasSum_sinh (z : ℂ) : HasSum (fun n ↦ z ^ (2 * n + 1) / ↑(2 * n + 1)!) (sinh z) := by
  simpa [mul_assoc, sin_mul_I, neg_pow z, pow_add, pow_mul, neg_mul, neg_div]
    using (hasSum_sin' (z * I)).mul_right (-I)


lemma cosh_eq_tsum (z : ℂ) : cosh z = ∑' n, z ^ (2 * n) / ↑(2 * n)! := z.hasSum_cosh.tsum_eq.symm


lemma sinh_eq_tsum (z : ℂ) : sinh z = ∑' n, z ^ (2 * n + 1) / ↑(2 * n + 1)! :=
  z.hasSum_sinh.tsum_eq.symm


/-- The power series expansion of `Real.cosh`. -/
lemma hasSum_cosh (r : ℝ) : HasSum (fun n  ↦ r ^ (2 * n) / ↑(2 * n)!) (cosh r) :=
  mod_cast Complex.hasSum_cosh r


/-- The power series expansion of `Real.sinh`. -/
lemma hasSum_sinh (r : ℝ) : HasSum (fun n ↦ r ^ (2 * n + 1) / ↑(2 * n + 1)!) (sinh r) :=
  mod_cast Complex.hasSum_sinh r


lemma cosh_eq_tsum (r : ℝ) : cosh r = ∑' n, r ^ (2 * n) / ↑(2 * n)! := r.hasSum_cosh.tsum_eq.symm


lemma sinh_eq_tsum (r : ℝ) : sinh r = ∑' n, r ^ (2 * n + 1) / ↑(2 * n + 1)! :=
  r.hasSum_sinh.tsum_eq.symm


lemma cosh_le_exp_half_sq (x : ℝ) : cosh x ≤ exp (x ^ 2 / 2) := by
  /-
    x : Real
    ⊢ LE.le (Real.cosh x) (Real.exp (HDiv.hDiv (HPow.hPow x 2) 2))
  -/
  rw [cosh_eq_tsum, exp_eq_exp_ℝ, exp_eq_tsum]
  /-
    x : Real
    ⊢ LE.le (tsum fun n => HDiv.hDiv (HPow.hPow x (HMul.hMul 2 n)) ↑(HMul.hMul 2 n …
  -/
  refine tsum_le_tsum (fun i ↦ ?_) x.hasSum_cosh.summable <| expSeries_summable' (x ^ 2 / 2)
  /-
    x : Real
    i : Nat
    ⊢ LE.le (HDiv.hDiv (HPow.hPow x (HMul.hMul 2 i)) ↑(HMul.hMul 2 i).factorial) ( …
  -/
  simp only [div_pow, pow_mul, smul_eq_mul, inv_mul_eq_div, div_div]
  /-
    x : Real
    i : Nat
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (HPow.hPow x 2) i) ↑(HMul.hMul 2 i).factorial) ( …
  -/
  gcongr
  /-
    case h
    x : Real
    i : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 i) ↑i.factorial) ↑(HMul.hMul 2 i).factorial
  -/
  norm_cast
  /-
    case h
    x : Real
    i : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow 2 i) i.factorial) (HMul.hMul 2 i).factorial
  -/
  exact Nat.two_pow_mul_factorial_le_factorial_two_mul _
  /-
    🎉 no goals
  -/


