@[simp, norm_cast]
theorem exp_coe (r : ℝ) : exp ℝ (r : ℍ[ℝ]) = ↑(exp ℝ r) :=
  (map_exp ℝ (algebraMap ℝ ℍ[ℝ]) (continuous_algebraMap _ _) _).symm


/-- The even terms of `expSeries` are real, and correspond to the series for $\cos ‖q‖$. -/
theorem expSeries_even_of_imaginary {q : Quaternion ℝ} (hq : q.re = 0) (n : ℕ) :
    expSeries ℝ (Quaternion ℝ) (2 * n) (fun _ => q) =
      ↑((-1 : ℝ) ^ n * ‖q‖ ^ (2 * n) / (2 * n)!) := by
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    ⊢ Eq ((NormedSpace.expSeries Real (Quaternion Real) (HMul.hMul 2 n)) fun x =>  …
  -/
  rw [expSeries_apply_eq]
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HMul.hMul 2 n).factorial) (HPow.hPow q (HMul.hMul …
  -/
  have hq2 : q ^ 2 = -normSq q := sq_eq_neg_normSq.mpr hq
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HMul.hMul 2 n).factorial) (HPow.hPow q (HMul.hMul …
  -/
  letI k : ℝ := ↑(2 * n)!
  calc
    k⁻¹ • q ^ (2 * n) = k⁻¹ • (-normSq q) ^ n := by rw [pow_mul, hq2]
    _ = k⁻¹ • ↑((-1 : ℝ) ^ n * ‖q‖ ^ (2 * n)) := ?_
    _ = ↑((-1 : ℝ) ^ n * ‖q‖ ^ (2 * n) / k) := ?_
    /-
      case calc_1
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq (HSMul.hSMul (Inv.inv k) (HPow.hPow (Neg.neg ↑(Quaternion.normSq q)) n))  …
    -/
  · congr 1
    /-
      case calc_1.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq (HPow.hPow (Neg.neg ↑(Quaternion.normSq q)) n) ↑(HMul.hMul (HPow.hPow (-1 …
    -/
    rw [neg_pow, normSq_eq_norm_mul_self, pow_mul, sq]
    /-
      case calc_1.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (↑(HMul.hMul (Norm.norm q) (Norm …
    -/
    push_cast
    /-
      case calc_1.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (HMul.hMul ↑(Norm.norm q) ↑(Norm …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq (HSMul.hSMul (Inv.inv k) ↑(HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm. …
    -/
  · rw [← coe_mul_eq_smul, div_eq_mul_inv]
    /-
      case calc_2
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq (HMul.hMul ↑(Inv.inv k) ↑(HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm.n …
    -/
    norm_cast
    /-
      case calc_2
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      k : Real := ↑(HMul.hMul 2 n).factorial
      ⊢ Eq ↑(HMul.hMul (Inv.inv k) (HMul.hMul (↑(HPow.hPow (Int.negSucc 0) n)) (HPow …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


/-- The odd terms of `expSeries` are real, and correspond to the series for
$\frac{q}{‖q‖} \sin ‖q‖$. -/
theorem expSeries_odd_of_imaginary {q : Quaternion ℝ} (hq : q.re = 0) (n : ℕ) :
    expSeries ℝ (Quaternion ℝ) (2 * n + 1) (fun _ => q) =
      (((-1 : ℝ) ^ n * ‖q‖ ^ (2 * n + 1) / (2 * n + 1)!) / ‖q‖) • q := by
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    ⊢ Eq ((NormedSpace.expSeries Real (Quaternion Real) (HAdd.hAdd (HMul.hMul 2 n) …
  -/
  rw [expSeries_apply_eq]
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial) (HPow.hPo …
  -/
  obtain rfl | hq0 := eq_or_ne q 0
    /-
      case inl
      n : Nat
      hq : Eq (QuaternionAlgebra.re 0) 0
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial) (HPow.hPo …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    hq0 : Ne q 0
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial) (HPow.hPo …
  -/
  have hq2 : q ^ 2 = -normSq q := sq_eq_neg_normSq.mpr hq
  /-
    case inr
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    hq0 : Ne q 0
    hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial) (HPow.hPo …
  -/
  have hqn := norm_ne_zero_iff.mpr hq0
  /-
    case inr
    q : Quaternion Real
    hq : Eq q.re 0
    n : Nat
    hq0 : Ne q 0
    hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
    hqn : Ne (Norm.norm q) 0
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial) (HPow.hPo …
  -/
  let k : ℝ := ↑(2 * n + 1)!
  calc
    k⁻¹ • q ^ (2 * n + 1) = k⁻¹ • ((-normSq q) ^ n * q) := by rw [pow_succ, pow_mul, hq2]
    _ = k⁻¹ • ((-1 : ℝ) ^ n * ‖q‖ ^ (2 * n)) • q := ?_
    _ = ((-1 : ℝ) ^ n * ‖q‖ ^ (2 * n + 1) / k / ‖q‖) • q := ?_
    /-
      case inr.calc_1
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HSMul.hSMul (Inv.inv k) (HMul.hMul (HPow.hPow (Neg.neg ↑(Quaternion.norm …
    -/
  · congr 1
    /-
      case inr.calc_1.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HMul.hMul (HPow.hPow (Neg.neg ↑(Quaternion.normSq q)) n) q) (HSMul.hSMul …
    -/
    rw [neg_pow, normSq_eq_norm_mul_self, pow_mul, sq, ← coe_mul_eq_smul]
    /-
      case inr.calc_1.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (↑(HMul.hMul (Norm.no …
    -/
    norm_cast
    /-
      🎉 no goals
    -/
    /-
      case inr.calc_2
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HSMul.hSMul (Inv.inv k) (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) n) (HPow …
    -/
  · rw [smul_smul]
    /-
      case inr.calc_2
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv k) (HMul.hMul (HPow.hPow (-1) n) (HPow.h …
    -/
    congr 1
    /-
      case inr.calc_2.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HMul.hMul (Inv.inv k) (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm.nor …
    -/
    simp_rw [pow_succ, mul_div_assoc, div_div_cancel_left' hqn]
    /-
      case inr.calc_2.e_a
      q : Quaternion Real
      hq : Eq q.re 0
      n : Nat
      hq0 : Ne q 0
      hq2 : Eq (HPow.hPow q 2) (Neg.neg ↑(Quaternion.normSq q))
      hqn : Ne (Norm.norm q) 0
      k : Real := ↑(HAdd.hAdd (HMul.hMul 2 n) 1).factorial
      ⊢ Eq (HMul.hMul (Inv.inv k) (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm.nor …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Auxiliary result; if the power series corresponding to `Real.cos` and `Real.sin` evaluated
at `‖q‖` tend to `c` and `s`, then the exponential series tends to `c + (s / ‖q‖)`. -/
theorem hasSum_expSeries_of_imaginary {q : Quaternion ℝ} (hq : q.re = 0) {c s : ℝ}
    (hc : HasSum (fun n => (-1 : ℝ) ^ n * ‖q‖ ^ (2 * n) / (2 * n)!) c)
    (hs : HasSum (fun n => (-1 : ℝ) ^ n * ‖q‖ ^ (2 * n + 1) / (2 * n + 1)!) s) :
    HasSum (fun n => expSeries ℝ (Quaternion ℝ) n fun _ => q) (↑c + (s / ‖q‖) • q) := by
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    c s : Real
    hc : HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm …
    hs : HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm …
    ⊢ HasSum (fun n => (NormedSpace.expSeries Real (Quaternion Real) n) fun x => q …
  -/
  replace hc := hasSum_coe.mpr hc
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    c s : Real
    hs : HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (HPow.hPow (Norm …
    hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
    ⊢ HasSum (fun n => (NormedSpace.expSeries Real (Quaternion Real) n) fun x => q …
  -/
  replace hs := (hs.div_const ‖q‖).smul_const q
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    c s : Real
    hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
    hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
    ⊢ HasSum (fun n => (NormedSpace.expSeries Real (Quaternion Real) n) fun x => q …
  -/
  refine HasSum.even_add_odd ?_ ?_
    /-
      case refine_1
      q : Quaternion Real
      hq : Eq q.re 0
      c s : Real
      hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
      hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
      ⊢ HasSum (fun k => (NormedSpace.expSeries Real (Quaternion Real) (HMul.hMul 2  …
    -/
  · convert hc using 1
    /-
      case h.e'_5
      q : Quaternion Real
      hq : Eq q.re 0
      c s : Real
      hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
      hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
      ⊢ Eq (fun k => (NormedSpace.expSeries Real (Quaternion Real) (HMul.hMul 2 k))  …
    -/
    ext n : 1
    /-
      case h.e'_5.h
      q : Quaternion Real
      hq : Eq q.re 0
      c s : Real
      hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
      hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
      n : Nat
      ⊢ Eq ((NormedSpace.expSeries Real (Quaternion Real) (HMul.hMul 2 n)) fun x =>  …
    -/
    rw [expSeries_even_of_imaginary hq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      q : Quaternion Real
      hq : Eq q.re 0
      c s : Real
      hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
      hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
      ⊢ HasSum (fun k => (NormedSpace.expSeries Real (Quaternion Real) (HAdd.hAdd (H …
    -/
  · convert hs using 1
    /-
      case h.e'_5
      q : Quaternion Real
      hq : Eq q.re 0
      c s : Real
      hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
      hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
      ⊢ Eq (fun k => (NormedSpace.expSeries Real (Quaternion Real) (HAdd.hAdd (HMul. …
    -/
    ext n : 1
    /-
      case h.e'_5.h
      q : Quaternion Real
      hq : Eq q.re 0
      c s : Real
      hc : HasSum (fun a => ↑(HDiv.hDiv (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (No …
      hs : HasSum (fun z => HSMul.hSMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow  …
      n : Nat
      ⊢ Eq ((NormedSpace.expSeries Real (Quaternion Real) (HAdd.hAdd (HMul.hMul 2 n) …
    -/
    rw [expSeries_odd_of_imaginary hq]
    /-
      🎉 no goals
    -/


/-- The closed form for the quaternion exponential on imaginary quaternions. -/
theorem exp_of_re_eq_zero (q : Quaternion ℝ) (hq : q.re = 0) :
    exp ℝ q = ↑(Real.cos ‖q‖) + (Real.sin ‖q‖ / ‖q‖) • q := by
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    ⊢ Eq (NormedSpace.exp Real q) (HAdd.hAdd (↑(Real.cos (Norm.norm q))) (HSMul.hS …
  -/
  rw [exp_eq_tsum]
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    ⊢ Eq ((fun x => tsum fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow x  …
  -/
  refine HasSum.tsum_eq ?_
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow q n)) (HAdd.h …
  -/
  simp_rw [← expSeries_apply_eq]
  /-
    q : Quaternion Real
    hq : Eq q.re 0
    ⊢ HasSum (fun n => (NormedSpace.expSeries Real (Quaternion Real) n) fun x => q …
  -/
  exact hasSum_expSeries_of_imaginary hq (Real.hasSum_cos _) (Real.hasSum_sin _)
  /-
    🎉 no goals
  -/


/-- The closed form for the quaternion exponential on arbitrary quaternions. -/
theorem exp_eq (q : Quaternion ℝ) :
    exp ℝ q = exp ℝ q.re • (↑(Real.cos ‖q.im‖) + (Real.sin ‖q.im‖ / ‖q.im‖) • q.im) := by
  rw [← exp_of_re_eq_zero q.im q.im_re, ← coe_mul_eq_smul, ← exp_coe, ← exp_add_of_commute,
    re_add_im]
  /-
    q : Quaternion Real
    ⊢ Commute (↑q.re) q.im
  -/
  exact Algebra.commutes q.re (_ : ℍ[ℝ])
  /-
    🎉 no goals
  -/


                                                                                  /-
                                                                                    q : Quaternion Real
                                                                                    ⊢ Eq (NormedSpace.exp Real q).re (HMul.hMul (NormedSpace.exp Real q.re) (Real. …
                                                                                  -/
theorem re_exp (q : ℍ[ℝ]) : (exp ℝ q).re = exp ℝ q.re * Real.cos ‖q - q.re‖ := by simp [exp_eq]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem im_exp (q : ℍ[ℝ]) : (exp ℝ q).im = (exp ℝ q.re * (Real.sin ‖q.im‖ / ‖q.im‖)) • q.im := by
  /-
    q : Quaternion Real
    ⊢ Eq (NormedSpace.exp Real q).im (HSMul.hSMul (HMul.hMul (NormedSpace.exp Real …
  -/
  simp [exp_eq, smul_smul]
  /-
    🎉 no goals
  -/


theorem normSq_exp (q : ℍ[ℝ]) : normSq (exp ℝ q) = exp ℝ q.re ^ 2 :=
  calc
    normSq (exp ℝ q) =
        normSq (exp ℝ q.re • (↑(Real.cos ‖q.im‖) + (Real.sin ‖q.im‖ / ‖q.im‖) • q.im)) := by
      /-
        q : Quaternion Real
        ⊢ Eq (Quaternion.normSq (NormedSpace.exp Real q)) (Quaternion.normSq (HSMul.hS …
      -/
      rw [exp_eq]
      /-
        🎉 no goals
      -/
    _ = exp ℝ q.re ^ 2 * normSq (↑(Real.cos ‖q.im‖) + (Real.sin ‖q.im‖ / ‖q.im‖) • q.im) := by
      /-
        q : Quaternion Real
        ⊢ Eq (Quaternion.normSq (HSMul.hSMul (NormedSpace.exp Real q.re) (HAdd.hAdd (↑ …
      -/
      rw [normSq_smul]
      /-
        🎉 no goals
      -/
    _ = exp ℝ q.re ^ 2 * (Real.cos ‖q.im‖ ^ 2 + Real.sin ‖q.im‖ ^ 2) := by
      /-
        q : Quaternion Real
        ⊢ Eq (HMul.hMul (HPow.hPow (NormedSpace.exp Real q.re) 2) (Quaternion.normSq ( …
      -/
      congr 1
      /-
        case e_a
        q : Quaternion Real
        ⊢ Eq (Quaternion.normSq (HAdd.hAdd (↑(Real.cos (Norm.norm q.im))) (HSMul.hSMul …
      -/
      obtain hv | hv := eq_or_ne ‖q.im‖ 0
        /-
          case e_a.inl
          q : Quaternion Real
          hv : Eq (Norm.norm q.im) 0
          ⊢ Eq (Quaternion.normSq (HAdd.hAdd (↑(Real.cos (Norm.norm q.im))) (HSMul.hSMul …
        -/
      · simp [hv]
        /-
          🎉 no goals
        -/
      rw [normSq_add, normSq_smul, star_smul, coe_mul_eq_smul, smul_re, smul_re, star_re, im_re,
        smul_zero, smul_zero, mul_zero, add_zero, div_pow, normSq_coe,
        normSq_eq_norm_mul_self, ← sq, div_mul_cancel₀ _ (pow_ne_zero _ hv)]
                             /-
                               q : Quaternion Real
                               ⊢ Eq (HMul.hMul (HPow.hPow (NormedSpace.exp Real q.re) 2) (HAdd.hAdd (HPow.hPo …
                             -/
    _ = exp ℝ q.re ^ 2 := by rw [Real.cos_sq_add_sin_sq, mul_one]
                             /-
                               🎉 no goals
                             -/


/-- Note that this implies that exponentials of pure imaginary quaternions are unit quaternions
since in that case the RHS is `1` via `NormedSpace.exp_zero` and `norm_one`. -/
@[simp]
theorem norm_exp (q : ℍ[ℝ]) : ‖exp ℝ q‖ = ‖exp ℝ q.re‖ := by
  rw [norm_eq_sqrt_real_inner (exp ℝ q), inner_self, normSq_exp, Real.sqrt_sq_eq_abs,
    Real.norm_eq_abs]


