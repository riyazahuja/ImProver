/-- `hermite n` is (up to sign) the factor appearing in `deriv^[n]` of a gaussian -/
theorem deriv_gaussian_eq_hermite_mul_gaussian (n : ℕ) (x : ℝ) :
    deriv^[n] (fun y => Real.exp (-(y ^ 2 / 2))) x =
    (-1 : ℝ) ^ n * aeval x (hermite n) * Real.exp (-(x ^ 2 / 2)) := by
  /-
    n : Nat
    x : Real
    ⊢ Eq (Nat.iterate deriv n (fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow y  …
  -/
  rw [mul_assoc]
  /-
    n : Nat
    x : Real
    ⊢ Eq (Nat.iterate deriv n (fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow y  …
  -/
  induction' n with n ih generalizing x
    /-
      case zero
      x : Real
      ⊢ Eq (Nat.iterate deriv 0 (fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow y  …
    -/
  · rw [Function.iterate_zero_apply, pow_zero, one_mul, hermite_zero, C_1, map_one, one_mul]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : ∀ (x : Real), Eq (Nat.iterate deriv n (fun y => Real.exp (Neg.neg (HDiv.h …
      x : Real
      ⊢ Eq (Nat.iterate deriv (HAdd.hAdd n 1) (fun y => Real.exp (Neg.neg (HDiv.hDiv …
    -/
  · replace ih : deriv^[n] _ = _ := _root_.funext ih
    have deriv_gaussian :
      deriv (fun y => Real.exp (-(y ^ 2 / 2))) x = -x * Real.exp (-(x ^ 2 / 2)) := by
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [mul_comm, ← neg_mul]`
      rw [deriv_exp (by simp)]; simp; ring
    rw [Function.iterate_succ_apply', ih, deriv_const_mul_field, deriv_mul, pow_succ (-1 : ℝ),
      deriv_gaussian, hermite_succ, map_sub, map_mul, aeval_X, Polynomial.deriv_aeval]
      /-
        case succ
        n : Nat
        x : Real
        ih : Eq (Nat.iterate deriv n fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow  …
        deriv_gaussian : Eq (deriv (fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow y …
        ⊢ Eq (HMul.hMul (HPow.hPow (-1) n) (HAdd.hAdd (HMul.hMul ((Polynomial.aeval x) …
      -/
    · ring
      /-
        🎉 no goals
      -/
      /-
        case succ.hc
        n : Nat
        x : Real
        ih : Eq (Nat.iterate deriv n fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow  …
        deriv_gaussian : Eq (deriv (fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow y …
        ⊢ DifferentiableAt Real (fun x => (Polynomial.aeval x) (Polynomial.hermite n)) x
      -/
    · apply Polynomial.differentiable_aeval
      /-
        🎉 no goals
      -/
      /-
        case succ.hd
        n : Nat
        x : Real
        ih : Eq (Nat.iterate deriv n fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow  …
        deriv_gaussian : Eq (deriv (fun y => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow y …
        ⊢ DifferentiableAt Real (fun x => Real.exp (Neg.neg (HDiv.hDiv (HPow.hPow x 2) …
      -/
    · apply DifferentiableAt.exp; simp -- Porting note: was just `simp`
                                  /-
                                    🎉 no goals
                                  -/


theorem hermite_eq_deriv_gaussian (n : ℕ) (x : ℝ) : aeval x (hermite n) =
    (-1 : ℝ) ^ n * deriv^[n] (fun y => Real.exp (-(y ^ 2 / 2))) x / Real.exp (-(x ^ 2 / 2)) := by
  /-
    n : Nat
    x : Real
    ⊢ Eq ((Polynomial.aeval x) (Polynomial.hermite n)) (HDiv.hDiv (HMul.hMul (HPow …
  -/
  rw [deriv_gaussian_eq_hermite_mul_gaussian]
  /-
    n : Nat
    x : Real
    ⊢ Eq ((Polynomial.aeval x) (Polynomial.hermite n)) (HDiv.hDiv (HMul.hMul (HPow …
  -/
  field_simp [Real.exp_ne_zero]
  rw [← @smul_eq_mul ℝ _ ((-1) ^ n), ← inv_smul_eq_iff₀, mul_assoc, smul_eq_mul, ← inv_pow, ←
    neg_inv, inv_one]
  /-
    case ha
    n : Nat
    x : Real
    ⊢ Ne (HPow.hPow (-1) n) 0
  -/
  exact pow_ne_zero _ (by norm_num)
  /-
    🎉 no goals
  -/


theorem hermite_eq_deriv_gaussian' (n : ℕ) (x : ℝ) : aeval x (hermite n) =
    (-1 : ℝ) ^ n * deriv^[n] (fun y => Real.exp (-(y ^ 2 / 2))) x * Real.exp (x ^ 2 / 2) := by
  /-
    n : Nat
    x : Real
    ⊢ Eq ((Polynomial.aeval x) (Polynomial.hermite n)) (HMul.hMul (HMul.hMul (HPow …
  -/
  rw [hermite_eq_deriv_gaussian, Real.exp_neg]
  /-
    n : Nat
    x : Real
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) n) (Nat.iterate deriv n (fun y => R …
  -/
  field_simp [Real.exp_ne_zero]
  /-
    🎉 no goals
  -/


