lemma Complex.cot_eq_exp_ratio (z : ℂ) :
    cot z = (Complex.exp (2 * I * z) + 1) / (I * (1 - Complex.exp (2 * I * z))) := by
  /-
    z : Complex
    ⊢ Eq z.cot (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMul 2 Complex. …
  -/
  rw [Complex.cot, Complex.sin, Complex.cos]
  /-
    z : Complex
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul z Complex.I)) (C …
  -/
  field_simp
  have h1 : exp (z * I) + exp (-(z * I)) = exp (-(z * I)) * (exp (2 * I * z) + 1) := by
    rw [mul_add, ← Complex.exp_add]
    simp only [mul_one, add_left_inj]
    ring_nf
  have h2 : (exp (-(z * I)) - exp (z * I)) * I = exp (-(z * I)) * (I * (1 - exp (2 * I * z))) := by
    ring_nf
    rw [mul_assoc, ← Complex.exp_add]
    ring_nf
  /-
    z : Complex
    h1 : Eq (HAdd.hAdd (Complex.exp (HMul.hMul z Complex.I)) (Complex.exp (Neg.neg …
    h2 : Eq (HMul.hMul (HSub.hSub (Complex.exp (Neg.neg (HMul.hMul z Complex.I)))  …
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul z Complex.I)) (Complex.exp  …
  -/
  rw [h1, h2, mul_div_mul_left _ _ (Complex.exp_ne_zero _)]
  /-
    🎉 no goals
  -/

/- The version one probably wants to use more. -/

lemma Complex.cot_pi_eq_exp_ratio (z : ℂ) :
    cot (π * z) = (Complex.exp (2 * π * I * z) + 1) / (I * (1 - Complex.exp (2 * π * I * z))) := by
  /-
    z : Complex
    ⊢ Eq (HMul.hMul (↑Real.pi) z).cot (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMu …
  -/
  rw [cot_eq_exp_ratio (π * z)]
  /-
    z : Complex
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMul 2 Complex.I) (HM …
  -/
  ring_nf
  /-
    🎉 no goals
  -/

/- This is the version one probably wants, which is why the pi's are there. -/

theorem pi_mul_cot_pi_q_exp (z : ℍ) :
    π * cot (π * z) = π * I - 2 * π * I * ∑' n : ℕ, Complex.exp (2 * π * I * z) ^ n := by
  have h1 : π * ((exp (2 * π * I * z) + 1) / (I * (1 - exp (2 * π * I * z)))) =
      -π * I * ((exp (2 * π * I * z) + 1) * (1 / (1 - exp (2 * π * I * z)))) := by
    simp only [div_mul_eq_div_mul_one_div, div_I, one_div, neg_mul, mul_neg, neg_inj]
    ring
  rw [cot_pi_eq_exp_ratio, h1, one_div, (tsum_geometric_of_norm_lt_one
    (UpperHalfPlane.abs_exp_two_pi_I_lt_one z)).symm, add_comm, geom_series_mul_one_add
      (Complex.exp (2 * π * I * (z : ℂ))) (UpperHalfPlane.abs_exp_two_pi_I_lt_one _)]
  /-
    z : UpperHalfPlane
    h1 : Eq (HMul.hMul (↑Real.pi) (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul (H …
    ⊢ Eq (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) Complex.I) (HSub.hSub (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/

