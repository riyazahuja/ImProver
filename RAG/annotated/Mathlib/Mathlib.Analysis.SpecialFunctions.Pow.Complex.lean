/-- The complex power function `x ^ y`, given by `x ^ y = exp(y log x)` (where `log` is the
principal determination of the logarithm), unless `x = 0` where one sets `0 ^ 0 = 1` and
`0 ^ y = 0` for `y ≠ 0`. -/
noncomputable def cpow (x y : ℂ) : ℂ :=
  if x = 0 then if y = 0 then 1 else 0 else exp (log x * y)


noncomputable instance : Pow ℂ ℂ :=
  ⟨cpow⟩


@[simp]
theorem cpow_eq_pow (x y : ℂ) : cpow x y = x ^ y :=
  rfl


theorem cpow_def (x y : ℂ) : x ^ y = if x = 0 then if y = 0 then 1 else 0 else exp (log x * y) :=
  rfl


theorem cpow_def_of_ne_zero {x : ℂ} (hx : x ≠ 0) (y : ℂ) : x ^ y = exp (log x * y) :=
  if_neg hx


@[simp]
                                                  /-
                                                    x : Complex
                                                    ⊢ Eq (HPow.hPow x 0) 1
                                                  -/
theorem cpow_zero (x : ℂ) : x ^ (0 : ℂ) = 1 := by simp [cpow_def]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem cpow_eq_zero_iff (x y : ℂ) : x ^ y = 0 ↔ x = 0 ∧ y ≠ 0 := by
  /-
    x y : Complex
    ⊢ Iff (Eq (HPow.hPow x y) 0) (And (Eq x 0) (Ne y 0))
  -/
  simp only [cpow_def]
  /-
    x y : Complex
    ⊢ Iff (Eq (ite (Eq x 0) (ite (Eq y 0) 1 0) (Complex.exp (HMul.hMul (Complex.lo …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [*, exp_ne_zero]
                /-
                  🎉 no goals
                -/


@[simp]
                                                              /-
                                                                x : Complex
                                                                h : Ne x 0
                                                                ⊢ Eq (HPow.hPow 0 x) 0
                                                              -/
theorem zero_cpow {x : ℂ} (h : x ≠ 0) : (0 : ℂ) ^ x = 0 := by simp [cpow_def, *]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem zero_cpow_eq_iff {x : ℂ} {a : ℂ} : (0 : ℂ) ^ x = a ↔ x ≠ 0 ∧ a = 0 ∨ x = 0 ∧ a = 1 := by
  /-
    x a : Complex
    ⊢ Iff (Eq (HPow.hPow 0 x) a) (Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1 …
  -/
  constructor
    /-
      case mp
      x a : Complex
      ⊢ Eq (HPow.hPow 0 x) a → Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
    -/
  · intro hyp
    /-
      case mp
      x a : Complex
      hyp : Eq (HPow.hPow 0 x) a
      ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
    -/
    simp only [cpow_def, eq_self_iff_true, if_true] at hyp
    /-
      case mp
      x a : Complex
      hyp : Eq (ite (Eq x 0) 1 0) a
      ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
    -/
    by_cases h : x = 0
      /-
        case pos
        x a : Complex
        hyp : Eq (ite (Eq x 0) 1 0) a
        h : Eq x 0
        ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
      -/
    · subst h
      /-
        case pos
        a : Complex
        hyp : Eq (ite (Eq 0 0) 1 0) a
        ⊢ Or (And (Ne 0 0) (Eq a 0)) (And (Eq 0 0) (Eq a 1))
      -/
      simp only [if_true, eq_self_iff_true] at hyp
      /-
        case pos
        a : Complex
        hyp : Eq 1 a
        ⊢ Or (And (Ne 0 0) (Eq a 0)) (And (Eq 0 0) (Eq a 1))
      -/
      right
      /-
        case pos.h
        a : Complex
        hyp : Eq 1 a
        ⊢ And (Eq 0 0) (Eq a 1)
      -/
      exact ⟨rfl, hyp.symm⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        x a : Complex
        hyp : Eq (ite (Eq x 0) 1 0) a
        h : Not (Eq x 0)
        ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
      -/
    · rw [if_neg h] at hyp
      /-
        case neg
        x a : Complex
        hyp : Eq 0 a
        h : Not (Eq x 0)
        ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
      -/
      left
      /-
        case neg.h
        x a : Complex
        hyp : Eq 0 a
        h : Not (Eq x 0)
        ⊢ And (Ne x 0) (Eq a 0)
      -/
      exact ⟨h, hyp.symm⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x a : Complex
      ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1)) → Eq (HPow.hPow 0 x) a
    -/
  · rintro (⟨h, rfl⟩ | ⟨rfl, rfl⟩)
      /-
        case mpr.inl.intro
        x : Complex
        h : Ne x 0
        ⊢ Eq (HPow.hPow 0 x) 0
      -/
    · exact zero_cpow h
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        ⊢ Eq (HPow.hPow 0 0) 1
      -/
    · exact cpow_zero _
      /-
        🎉 no goals
      -/


theorem eq_zero_cpow_iff {x : ℂ} {a : ℂ} : a = (0 : ℂ) ^ x ↔ x ≠ 0 ∧ a = 0 ∨ x = 0 ∧ a = 1 := by
  /-
    x a : Complex
    ⊢ Iff (Eq a (HPow.hPow 0 x)) (Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1 …
  -/
  rw [← zero_cpow_eq_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem cpow_one (x : ℂ) : x ^ (1 : ℂ) = x :=
                        /-
                          x : Complex
                          hx : Eq x 0
                          ⊢ Eq (HPow.hPow x 1) x
                        -/
  if hx : x = 0 then by simp [hx, cpow_def]
                        /-
                          🎉 no goals
                        -/
          /-
            x : Complex
            hx : Not (Eq x 0)
            ⊢ Eq (HPow.hPow x 1) x
          -/
  else by rw [cpow_def, if_neg (one_ne_zero : (1 : ℂ) ≠ 0), if_neg hx, mul_one, exp_log hx]
          /-
            🎉 no goals
          -/


@[simp]
theorem one_cpow (x : ℂ) : (1 : ℂ) ^ x = 1 := by
  /-
    x : Complex
    ⊢ Eq (HPow.hPow 1 x) 1
  -/
  rw [cpow_def]
  /-
    x : Complex
    ⊢ Eq (ite (Eq 1 0) (ite (Eq x 0) 1 0) (Complex.exp (HMul.hMul (Complex.log 1)  …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all [one_ne_zero]
                /-
                  🎉 no goals
                -/


theorem cpow_add {x : ℂ} (y z : ℂ) (hx : x ≠ 0) : x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x y z : Complex
    hx : Ne x 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  simp only [cpow_def, ite_mul, boole_mul, mul_ite, mul_boole]
  /-
    x y z : Complex
    hx : Ne x 0
    ⊢ Eq (ite (Eq x 0) (ite (Eq (HAdd.hAdd y z) 0) 1 0) (Complex.exp (HMul.hMul (C …
  -/
  simp_all [exp_add, mul_add]
  /-
    🎉 no goals
  -/


theorem cpow_mul {x y : ℂ} (z : ℂ) (h₁ : -π < (log x * y).im) (h₂ : (log x * y).im ≤ π) :
    x ^ (y * z) = (x ^ y) ^ z := by
  /-
    x y z : Complex
    h₁ : LT.lt (Neg.neg Real.pi) (HMul.hMul (Complex.log x) y).im
    h₂ : LE.le (HMul.hMul (Complex.log x) y).im Real.pi
    ⊢ Eq (HPow.hPow x (HMul.hMul y z)) (HPow.hPow (HPow.hPow x y) z)
  -/
  simp only [cpow_def]
  /-
    x y z : Complex
    h₁ : LT.lt (Neg.neg Real.pi) (HMul.hMul (Complex.log x) y).im
    h₂ : LE.le (HMul.hMul (Complex.log x) y).im Real.pi
    ⊢ Eq (ite (Eq x 0) (ite (Eq (HMul.hMul y z) 0) 1 0) (Complex.exp (HMul.hMul (C …
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
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
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all [exp_ne_zero, log_exp h₁ h₂, mul_assoc]
                /-
                  🎉 no goals
                -/


theorem cpow_neg (x y : ℂ) : x ^ (-y) = (x ^ y)⁻¹ := by
  /-
    x y : Complex
    ⊢ Eq (HPow.hPow x (Neg.neg y)) (Inv.inv (HPow.hPow x y))
  -/
  simp only [cpow_def, neg_eq_zero, mul_neg]
  /-
    x y : Complex
    ⊢ Eq (ite (Eq x 0) (ite (Eq y 0) 1 0) (Complex.exp (Neg.neg (HMul.hMul (Comple …
  -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [exp_neg]
                /-
                  🎉 no goals
                -/


theorem cpow_sub {x : ℂ} (y z : ℂ) (hx : x ≠ 0) : x ^ (y - z) = x ^ y / x ^ z := by
  /-
    x y z : Complex
    hx : Ne x 0
    ⊢ Eq (HPow.hPow x (HSub.hSub y z)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x z))
  -/
  rw [sub_eq_add_neg, cpow_add _ _ hx, cpow_neg, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


                                                        /-
                                                          x : Complex
                                                          ⊢ Eq (HPow.hPow x (-1)) (Inv.inv x)
                                                        -/
theorem cpow_neg_one (x : ℂ) : x ^ (-1 : ℂ) = x⁻¹ := by simpa using cpow_neg x 1
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- See also `Complex.cpow_int_mul'`. -/
lemma cpow_int_mul (x : ℂ) (n : ℤ) (y : ℂ) : x ^ (n * y) = (x ^ y) ^ n := by
  /-
    x : Complex
    n : Int
    y : Complex
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) y)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      n : Int
      y : Complex
      ⊢ Eq (HPow.hPow 0 (HMul.hMul (↑n) y)) (HPow.hPow (HPow.hPow 0 y) n)
    -/
  · rcases eq_or_ne n 0 with rfl | hn
      /-
        case inl.inl
        y : Complex
        ⊢ Eq (HPow.hPow 0 (HMul.hMul (↑0) y)) (HPow.hPow (HPow.hPow 0 y) 0)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        n : Int
        y : Complex
        hn : Ne n 0
        ⊢ Eq (HPow.hPow 0 (HMul.hMul (↑n) y)) (HPow.hPow (HPow.hPow 0 y) n)
      -/
                                            /-
                                              🎉 no goals
                                            -/
    · rcases eq_or_ne y 0 with rfl | hy <;> simp [*, zero_zpow]
                                            /-
                                              🎉 no goals
                                            -/
    /-
      case inr
      x : Complex
      n : Int
      y : Complex
      hx : Ne x 0
      ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) y)) (HPow.hPow (HPow.hPow x y) n)
    -/
  · rw [cpow_def_of_ne_zero hx, cpow_def_of_ne_zero hx, mul_left_comm, exp_int_mul]
    /-
      🎉 no goals
    -/


                                                                       /-
                                                                         x y : Complex
                                                                         n : Int
                                                                         ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
                                                                       -/
lemma cpow_mul_int (x y : ℂ) (n : ℤ) : x ^ (y * n) = (x ^ y) ^ n := by rw [mul_comm, cpow_int_mul]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma cpow_nat_mul (x : ℂ) (n : ℕ) (y : ℂ) : x ^ (n * y) = (x ^ y) ^ n :=
  mod_cast cpow_int_mul x n y


lemma cpow_ofNat_mul (x : ℂ) (n : ℕ) [n.AtLeastTwo] (y : ℂ) :
    x ^ (ofNat(n) * y) = (x ^ y) ^ ofNat(n) :=
  cpow_nat_mul x n y


lemma cpow_mul_nat (x y : ℂ) (n : ℕ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x y : Complex
    n : Nat
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [mul_comm, cpow_nat_mul]
  /-
    🎉 no goals
  -/


lemma cpow_mul_ofNat (x y : ℂ) (n : ℕ) [n.AtLeastTwo] :
    x ^ (y * ofNat(n)) = (x ^ y) ^ ofNat(n) :=
  cpow_mul_nat x y n


@[simp, norm_cast]
                                                                 /-
                                                                   x : Complex
                                                                   n : Nat
                                                                   ⊢ Eq (HPow.hPow x ↑n) (HPow.hPow x n)
                                                                 -/
theorem cpow_natCast (x : ℂ) (n : ℕ) : x ^ (n : ℂ) = x ^ n := by simpa using cpow_nat_mul x n 1
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated (since := "2024-04-17")]
alias cpow_nat_cast := cpow_natCast


@[simp]
lemma cpow_ofNat (x : ℂ) (n : ℕ) [n.AtLeastTwo] :
    x ^ (ofNat(n) : ℂ) = x ^ ofNat(n) :=
  cpow_natCast x n


theorem cpow_two (x : ℂ) : x ^ (2 : ℂ) = x ^ (2 : ℕ) := cpow_ofNat x 2


@[simp, norm_cast]
                                                                 /-
                                                                   x : Complex
                                                                   n : Int
                                                                   ⊢ Eq (HPow.hPow x ↑n) (HPow.hPow x n)
                                                                 -/
theorem cpow_intCast (x : ℂ) (n : ℤ) : x ^ (n : ℂ) = x ^ n := by simpa using cpow_int_mul x n 1
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated (since := "2024-04-17")]
alias cpow_int_cast := cpow_intCast


@[simp]
theorem cpow_nat_inv_pow (x : ℂ) {n : ℕ} (hn : n ≠ 0) : (x ^ (n⁻¹ : ℂ)) ^ n = x := by
  /-
    x : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv ↑n)) n) x
  -/
  rw [← cpow_nat_mul, mul_inv_cancel₀, cpow_one]
  /-
    x : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Ne (↑n) 0
  -/
  assumption_mod_cast
  /-
    🎉 no goals
  -/


@[simp]
lemma cpow_ofNat_inv_pow (x : ℂ) (n : ℕ) [n.AtLeastTwo] :
    (x ^ ((ofNat(n) : ℂ)⁻¹)) ^ (ofNat(n) : ℕ) = x :=
  cpow_nat_inv_pow _ (NeZero.ne n)


/-- A version of `Complex.cpow_int_mul` with RHS that matches `Complex.cpow_mul`.

The assumptions on the arguments are needed
because the equality fails, e.g., for `x = -I`, `n = 2`, `y = 1/2`. -/
lemma cpow_int_mul' {x : ℂ} {n : ℤ} (hlt : -π < n * x.arg) (hle : n * x.arg ≤ π) (y : ℂ) :
    x ^ (n * y) = (x ^ n) ^ y := by
  /-
    x : Complex
    n : Int
    hlt : LT.lt (Neg.neg Real.pi) (HMul.hMul (↑n) x.arg)
    hle : LE.le (HMul.hMul (↑n) x.arg) Real.pi
    y : Complex
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) y)) (HPow.hPow (HPow.hPow x n) y)
  -/
  rw [mul_comm] at hlt hle
  /-
    x : Complex
    n : Int
    hlt : LT.lt (Neg.neg Real.pi) (HMul.hMul x.arg ↑n)
    hle : LE.le (HMul.hMul x.arg ↑n) Real.pi
    y : Complex
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) y)) (HPow.hPow (HPow.hPow x n) y)
  -/
                                  /-
                                    🎉 no goals
                                  -/
  rw [cpow_mul, cpow_intCast] <;> simpa [log_im]
                                  /-
                                    🎉 no goals
                                  -/


/-- A version of `Complex.cpow_nat_mul` with RHS that matches `Complex.cpow_mul`.

The assumptions on the arguments are needed
because the equality fails, e.g., for `x = -I`, `n = 2`, `y = 1/2`. -/
lemma cpow_nat_mul' {x : ℂ} {n : ℕ} (hlt : -π < n * x.arg) (hle : n * x.arg ≤ π) (y : ℂ) :
    x ^ (n * y) = (x ^ n) ^ y :=
  cpow_int_mul' hlt hle y


lemma cpow_ofNat_mul' {x : ℂ} {n : ℕ} [n.AtLeastTwo] (hlt : -π < OfNat.ofNat n * x.arg)
    (hle : OfNat.ofNat n * x.arg ≤ π) (y : ℂ) :
    x ^ (OfNat.ofNat n * y) = (x ^ ofNat(n)) ^ y :=
  cpow_nat_mul' hlt hle y


lemma pow_cpow_nat_inv {x : ℂ} {n : ℕ} (h₀ : n ≠ 0) (hlt : -(π / n) < x.arg) (hle : x.arg ≤ π / n) :
    (x ^ n) ^ (n⁻¹ : ℂ) = x := by
  /-
    x : Complex
    n : Nat
    h₀ : Ne n 0
    hlt : LT.lt (Neg.neg (HDiv.hDiv Real.pi ↑n)) x.arg
    hle : LE.le x.arg (HDiv.hDiv Real.pi ↑n)
    ⊢ Eq (HPow.hPow (HPow.hPow x n) (Inv.inv ↑n)) x
  -/
  rw [← cpow_nat_mul', mul_inv_cancel₀ (Nat.cast_ne_zero.2 h₀), cpow_one]
    /-
      case hlt
      x : Complex
      n : Nat
      h₀ : Ne n 0
      hlt : LT.lt (Neg.neg (HDiv.hDiv Real.pi ↑n)) x.arg
      hle : LE.le x.arg (HDiv.hDiv Real.pi ↑n)
      ⊢ LT.lt (Neg.neg Real.pi) (HMul.hMul (↑n) x.arg)
    -/
  · rwa [← div_lt_iff₀' (Nat.cast_pos.2 h₀.bot_lt), neg_div]
    /-
      🎉 no goals
    -/
    /-
      case hle
      x : Complex
      n : Nat
      h₀ : Ne n 0
      hlt : LT.lt (Neg.neg (HDiv.hDiv Real.pi ↑n)) x.arg
      hle : LE.le x.arg (HDiv.hDiv Real.pi ↑n)
      ⊢ LE.le (HMul.hMul (↑n) x.arg) Real.pi
    -/
  · rwa [← le_div_iff₀' (Nat.cast_pos.2 h₀.bot_lt)]
    /-
      🎉 no goals
    -/


lemma pow_cpow_ofNat_inv {x : ℂ} {n : ℕ} [n.AtLeastTwo] (hlt : -(π / OfNat.ofNat n) < x.arg)
    (hle : x.arg ≤ π / OfNat.ofNat n) :
    (x ^ ofNat(n)) ^ ((OfNat.ofNat n : ℂ)⁻¹) = x :=
  pow_cpow_nat_inv (NeZero.ne n) hlt hle


/-- See also `Complex.pow_cpow_ofNat_inv` for a version that also works for `x * I`, `0 ≤ x`. -/
lemma sq_cpow_two_inv {x : ℂ} (hx : 0 < x.re) : (x ^ (2 : ℕ)) ^ (2⁻¹ : ℂ) = x :=
  pow_cpow_ofNat_inv (neg_pi_div_two_lt_arg_iff.2 <| .inl hx)
    (arg_le_pi_div_two_iff.2 <| .inl hx.le)


theorem mul_cpow_ofReal_nonneg {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (r : ℂ) :
    ((a : ℂ) * (b : ℂ)) ^ r = (a : ℂ) ^ r * (b : ℂ) ^ r := by
  /-
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    r : Complex
    ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑b) r) (HMul.hMul (HPow.hPow (↑a) r) (HPow.hPow  …
  -/
  rcases eq_or_ne r 0 with (rfl | hr)
    /-
      case inl
      a b : Real
      ha : LE.le 0 a
      hb : LE.le 0 b
      ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑b) 0) (HMul.hMul (HPow.hPow (↑a) 0) (HPow.hPow  …
    -/
  · simp only [cpow_zero, mul_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    r : Complex
    hr : Ne r 0
    ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑b) r) (HMul.hMul (HPow.hPow (↑a) r) (HPow.hPow  …
  -/
  rcases eq_or_lt_of_le ha with (rfl | ha')
    /-
      case inr.inl
      b : Real
      hb : LE.le 0 b
      r : Complex
      hr : Ne r 0
      ha : LE.le 0 0
      ⊢ Eq (HPow.hPow (HMul.hMul ↑0 ↑b) r) (HMul.hMul (HPow.hPow (↑0) r) (HPow.hPow  …
    -/
  · rw [ofReal_zero, zero_mul, zero_cpow hr, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    r : Complex
    hr : Ne r 0
    ha' : LT.lt 0 a
    ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑b) r) (HMul.hMul (HPow.hPow (↑a) r) (HPow.hPow  …
  -/
  rcases eq_or_lt_of_le hb with (rfl | hb')
    /-
      case inr.inr.inl
      a : Real
      ha : LE.le 0 a
      r : Complex
      hr : Ne r 0
      ha' : LT.lt 0 a
      hb : LE.le 0 0
      ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑0) r) (HMul.hMul (HPow.hPow (↑a) r) (HPow.hPow  …
    -/
  · rw [ofReal_zero, mul_zero, zero_cpow hr, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.inr
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    r : Complex
    hr : Ne r 0
    ha' : LT.lt 0 a
    hb' : LT.lt 0 b
    ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑b) r) (HMul.hMul (HPow.hPow (↑a) r) (HPow.hPow  …
  -/
  have ha'' : (a : ℂ) ≠ 0 := ofReal_ne_zero.mpr ha'.ne'
  /-
    case inr.inr.inr
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    r : Complex
    hr : Ne r 0
    ha' : LT.lt 0 a
    hb' : LT.lt 0 b
    ha'' : Ne (↑a) 0
    ⊢ Eq (HPow.hPow (HMul.hMul ↑a ↑b) r) (HMul.hMul (HPow.hPow (↑a) r) (HPow.hPow  …
  -/
  have hb'' : (b : ℂ) ≠ 0 := ofReal_ne_zero.mpr hb'.ne'
  rw [cpow_def_of_ne_zero (mul_ne_zero ha'' hb''), log_ofReal_mul ha' hb'', ofReal_log ha,
    add_mul, exp_add, ← cpow_def_of_ne_zero ha'', ← cpow_def_of_ne_zero hb'']


lemma natCast_mul_natCast_cpow (m n : ℕ) (s : ℂ) : (m * n : ℂ) ^ s = m ^ s * n ^ s :=
  ofReal_natCast m ▸ ofReal_natCast n ▸ mul_cpow_ofReal_nonneg m.cast_nonneg n.cast_nonneg s


lemma natCast_cpow_natCast_mul (n m : ℕ) (z : ℂ) : (n : ℂ) ^ (m * z) = ((n : ℂ) ^ m) ^ z := by
  /-
    n m : Nat
    z : Complex
    ⊢ Eq (HPow.hPow (↑n) (HMul.hMul (↑m) z)) (HPow.hPow (HPow.hPow (↑n) m) z)
  -/
  refine cpow_nat_mul' (x := n) (n := m) ?_ ?_ z
    /-
      case refine_1
      n m : Nat
      z : Complex
      ⊢ LT.lt (Neg.neg Real.pi) (HMul.hMul (↑m) (↑n).arg)
    -/
  · simp only [natCast_arg, mul_zero, Left.neg_neg_iff, pi_pos]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n m : Nat
      z : Complex
      ⊢ LE.le (HMul.hMul (↑m) (↑n).arg) Real.pi
    -/
  · simp only [natCast_arg, mul_zero, pi_pos.le]
    /-
      🎉 no goals
    -/


theorem inv_cpow_eq_ite (x : ℂ) (n : ℂ) :
    x⁻¹ ^ n = if x.arg = π then conj (x ^ conj n)⁻¹ else (x ^ n)⁻¹ := by
  simp_rw [Complex.cpow_def, log_inv_eq_ite, inv_eq_zero, map_eq_zero, ite_mul, neg_mul,
    RCLike.conj_inv, apply_ite conj, apply_ite exp, apply_ite Inv.inv, map_zero, map_one, exp_neg,
    inv_one, inv_zero, ← exp_conj, map_mul, conj_conj]
  /-
    x n : Complex
    ⊢ Eq (ite (Eq x 0) (ite (Eq n 0) 1 0) (ite (Eq x.arg Real.pi) (Inv.inv (Comple …
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
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  split_ifs with hx hn ha ha <;> rfl
                                 /-
                                   🎉 no goals
                                 -/


theorem inv_cpow (x : ℂ) (n : ℂ) (hx : x.arg ≠ π) : x⁻¹ ^ n = (x ^ n)⁻¹ := by
  /-
    x n : Complex
    hx : Ne x.arg Real.pi
    ⊢ Eq (HPow.hPow (Inv.inv x) n) (Inv.inv (HPow.hPow x n))
  -/
  rw [inv_cpow_eq_ite, if_neg hx]
  /-
    🎉 no goals
  -/


/-- `Complex.inv_cpow_eq_ite` with the `ite` on the other side. -/
theorem inv_cpow_eq_ite' (x : ℂ) (n : ℂ) :
    (x ^ n)⁻¹ = if x.arg = π then conj (x⁻¹ ^ conj n) else x⁻¹ ^ n := by
  /-
    x n : Complex
    ⊢ Eq (Inv.inv (HPow.hPow x n)) (ite (Eq x.arg Real.pi) ((starRingEnd Complex)  …
  -/
  rw [inv_cpow_eq_ite, apply_ite conj, conj_conj, conj_conj]
  /-
    x n : Complex
    ⊢ Eq (Inv.inv (HPow.hPow x n)) (ite (Eq x.arg Real.pi) (ite (Eq x.arg Real.pi) …
  -/
  split_ifs with h
    /-
      case pos
      x n : Complex
      h : Eq x.arg Real.pi
      ⊢ Eq (Inv.inv (HPow.hPow x n)) (Inv.inv (HPow.hPow x n))
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      x n : Complex
      h : Not (Eq x.arg Real.pi)
      ⊢ Eq (Inv.inv (HPow.hPow x n)) (HPow.hPow (Inv.inv x) n)
    -/
  · rw [inv_cpow _ _ h]
    /-
      🎉 no goals
    -/


theorem conj_cpow_eq_ite (x : ℂ) (n : ℂ) :
    conj x ^ n = if x.arg = π then x ^ n else conj (x ^ conj n) := by
  simp_rw [cpow_def, map_eq_zero, apply_ite conj, map_one, map_zero, ← exp_conj, map_mul, conj_conj,
    log_conj_eq_ite]
  /-
    x n : Complex
    ⊢ Eq (ite (Eq x 0) (ite (Eq n 0) 1 0) (Complex.exp (HMul.hMul (ite (Eq x.arg R …
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
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  split_ifs with hcx hn hx <;> rfl
                               /-
                                 🎉 no goals
                               -/


theorem conj_cpow (x : ℂ) (n : ℂ) (hx : x.arg ≠ π) : conj x ^ n = conj (x ^ conj n) := by
  /-
    x n : Complex
    hx : Ne x.arg Real.pi
    ⊢ Eq (HPow.hPow ((starRingEnd Complex) x) n) ((starRingEnd Complex) (HPow.hPow …
  -/
  rw [conj_cpow_eq_ite, if_neg hx]
  /-
    🎉 no goals
  -/


theorem cpow_conj (x : ℂ) (n : ℂ) (hx : x.arg ≠ π) : x ^ conj n = conj (conj x ^ n) := by
  /-
    x n : Complex
    hx : Ne x.arg Real.pi
    ⊢ Eq (HPow.hPow x ((starRingEnd Complex) n)) ((starRingEnd Complex) (HPow.hPow …
  -/
  rw [conj_cpow _ _ hx, conj_conj]
  /-
    🎉 no goals
  -/


lemma natCast_add_one_cpow_ne_zero (n : ℕ) (z : ℂ) : (n + 1 : ℂ) ^ z ≠ 0 :=
                                         /-
                                           n : Nat
                                           z : Complex
                                           H : And (Eq (HAdd.hAdd (↑n) 1) 0) (Ne z 0)
                                           ⊢ False
                                         -/
  mt (cpow_eq_zero_iff ..).mp fun H ↦ by norm_cast at H; exact H.1
                                                         /-
                                                           🎉 no goals
                                                         -/


