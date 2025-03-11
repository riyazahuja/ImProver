/-- If `IsCyclotomicExtension {3} ℚ K` then `𝓞 K` is a principal ideal domain. -/
theorem three_pid [IsCyclotomicExtension {3} ℚ K] : IsPrincipalIdealRing (𝓞 K) := by
  /-
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ IsPrincipalIdealRing (NumberField.RingOfIntegers K)
  -/
  apply RingOfIntegers.isPrincipalIdealRing_of_abs_discr_lt
  rw [absdiscr_prime 3 K, IsCyclotomicExtension.finrank (n := 3) K
    (irreducible_rat (by norm_num)), nrComplexPlaces_eq_totient_div_two 3, totient_prime
      PNat.prime_three]
  simp only [Int.reduceNeg, PNat.val_ofNat, succ_sub_succ_eq_sub, tsub_zero, zero_lt_two,
    Nat.div_self, pow_one, cast_ofNat, neg_mul, one_mul, abs_neg, Int.cast_abs, Int.cast_ofNat,
    factorial_two, gt_iff_lt, abs_of_pos (show (0 : ℝ) < 3 by norm_num)]
  suffices (2 * (3 / 4) * (2 ^ 2 / 2)) ^ 2 < (2 * (π / 4) * (2 ^ 2 / 2)) ^ 2 from
    lt_trans (by norm_num) this
  /-
    case h
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ LT.lt (HPow.hPow (HMul.hMul (HMul.hMul 2 (3 / 4)) (HDiv.hDiv (HPow.hPow 2 2) …
  -/
  gcongr
  /-
    case h.hab.bc.bc.h
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ LT.lt 3 Real.pi
  -/
  exact pi_gt_three
  /-
    🎉 no goals
  -/


/-- If `IsCyclotomicExtension {5} ℚ K` then `𝓞 K` is a principal ideal domain. -/
theorem five_pid [IsCyclotomicExtension {5} ℚ K] : IsPrincipalIdealRing (𝓞 K) := by
  /-
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 5) Rat K
    ⊢ IsPrincipalIdealRing (NumberField.RingOfIntegers K)
  -/
  apply RingOfIntegers.isPrincipalIdealRing_of_abs_discr_lt
  rw [absdiscr_prime 5 K, IsCyclotomicExtension.finrank (n := 5) K
    (irreducible_rat (by norm_num)), nrComplexPlaces_eq_totient_div_two 5, totient_prime
      PNat.prime_five]
  simp only [Int.reduceNeg, PNat.val_ofNat, succ_sub_succ_eq_sub, tsub_zero, reduceDiv, even_two,
    Even.neg_pow, one_pow, cast_ofNat, Int.reducePow, one_mul, Int.cast_abs, Int.cast_ofNat,
    div_pow, gt_iff_lt, show 4! = 24 by rfl, abs_of_pos (show (0 : ℝ) < 125 by norm_num)]
  suffices (2 * (3 ^ 2 / 4 ^ 2) * (4 ^ 4 / 24)) ^ 2 < (2 * (π ^ 2 / 4 ^ 2) * (4 ^ 4 / 24)) ^ 2 from
    lt_trans (by norm_num) this
  /-
    case h
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 5) Rat K
    ⊢ LT.lt (HPow.hPow (HMul.hMul (HMul.hMul 2 (HDiv.hDiv (HPow.hPow 3 2) (HPow.hP …
  -/
  gcongr
  /-
    case h.hab.bc.bc.h.hab
    K : Type u
    inst✝² : Field K
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 5) Rat K
    ⊢ LT.lt 3 Real.pi
  -/
  exact pi_gt_three
  /-
    🎉 no goals
  -/


