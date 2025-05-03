/-- The real power function `x ^ y`, defined as the real part of the complex power function.
For `x > 0`, it is equal to `exp (y log x)`. For `x = 0`, one sets `0 ^ 0=1` and `0 ^ y=0` for
`y ≠ 0`. For `x < 0`, the definition is somewhat arbitrary as it depends on the choice of a complex
determination of the logarithm. With our conventions, it is equal to `exp (y log x) cos (π y)`. -/
noncomputable def rpow (x y : ℝ) :=
  ((x : ℂ) ^ (y : ℂ)).re


noncomputable instance : Pow ℝ ℝ := ⟨rpow⟩


@[simp]
theorem rpow_eq_pow (x y : ℝ) : rpow x y = x ^ y := rfl


theorem rpow_def (x y : ℝ) : x ^ y = ((x : ℂ) ^ (y : ℂ)).re := rfl


theorem rpow_def_of_nonneg {x : ℝ} (hx : 0 ≤ x) (y : ℝ) :
    x ^ y = if x = 0 then if y = 0 then 1 else 0 else exp (log x * y) := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ Eq (HPow.hPow x y) (ite (Eq x 0) (ite (Eq y 0) 1 0) (Real.exp (HMul.hMul (Re …
  -/
  simp only [rpow_def, Complex.cpow_def]; split_ifs <;>
  simp_all [(Complex.ofReal_log hx).symm, -Complex.ofReal_mul,
      (Complex.ofReal_mul _ _).symm, Complex.exp_ofReal_re, Complex.ofReal_eq_zero]


theorem rpow_def_of_pos {x : ℝ} (hx : 0 < x) (y : ℝ) : x ^ y = exp (log x * y) := by
  /-
    x : Real
    hx : LT.lt 0 x
    y : Real
    ⊢ Eq (HPow.hPow x y) (Real.exp (HMul.hMul (Real.log x) y))
  -/
  rw [rpow_def_of_nonneg (le_of_lt hx), if_neg (ne_of_gt hx)]
  /-
    🎉 no goals
  -/


                                                          /-
                                                            x y : Real
                                                            ⊢ Eq (Real.exp (HMul.hMul x y)) (HPow.hPow (Real.exp x) y)
                                                          -/
theorem exp_mul (x y : ℝ) : exp (x * y) = exp x ^ y := by rw [rpow_def_of_pos (exp_pos _), log_exp]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp, norm_cast]
theorem rpow_intCast (x : ℝ) (n : ℤ) : x ^ (n : ℝ) = x ^ n := by
  simp only [rpow_def, ← Complex.ofReal_zpow, Complex.cpow_intCast, Complex.ofReal_intCast,
    Complex.ofReal_re]


@[deprecated (since := "2024-04-17")]
alias rpow_int_cast := rpow_intCast


@[simp, norm_cast]
                                                                 /-
                                                                   x : Real
                                                                   n : Nat
                                                                   ⊢ Eq (HPow.hPow x ↑n) (HPow.hPow x n)
                                                                 -/
theorem rpow_natCast (x : ℝ) (n : ℕ) : x ^ (n : ℝ) = x ^ n := by simpa using rpow_intCast x n
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated (since := "2024-04-17")]
alias rpow_nat_cast := rpow_natCast


@[simp]
                                                       /-
                                                         x : Real
                                                         ⊢ Eq (HPow.hPow (Real.exp 1) x) (Real.exp x)
                                                       -/
theorem exp_one_rpow (x : ℝ) : exp 1 ^ x = exp x := by rw [← exp_mul, one_mul]
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                            /-
                                                              n : Nat
                                                              ⊢ Eq (HPow.hPow (Real.exp 1) n) (Real.exp ↑n)
                                                            -/
@[simp] lemma exp_one_pow (n : ℕ) : exp 1 ^ n = exp n := by rw [← rpow_natCast, exp_one_rpow]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem rpow_eq_zero_iff_of_nonneg (hx : 0 ≤ x) : x ^ y = 0 ↔ x = 0 ∧ y ≠ 0 := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Iff (Eq (HPow.hPow x y) 0) (And (Eq x 0) (Ne y 0))
  -/
  simp only [rpow_def_of_nonneg hx]
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Iff (Eq (ite (Eq x 0) (ite (Eq y 0) 1 0) (Real.exp (HMul.hMul (Real.log x) y …
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
lemma rpow_eq_zero (hx : 0 ≤ x) (hy : y ≠ 0) : x ^ y = 0 ↔ x = 0 := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : Ne y 0
    ⊢ Iff (Eq (HPow.hPow x y) 0) (Eq x 0)
  -/
  simp [rpow_eq_zero_iff_of_nonneg, *]
  /-
    🎉 no goals
  -/


@[simp]
lemma rpow_ne_zero (hx : 0 ≤ x) (hy : y ≠ 0) : x ^ y ≠ 0 ↔ x ≠ 0 :=
  Real.rpow_eq_zero hx hy |>.not


theorem rpow_def_of_neg {x : ℝ} (hx : x < 0) (y : ℝ) : x ^ y = exp (log x * y) * cos (y * π) := by
  /-
    x : Real
    hx : LT.lt x 0
    y : Real
    ⊢ Eq (HPow.hPow x y) (HMul.hMul (Real.exp (HMul.hMul (Real.log x) y)) (Real.co …
  -/
  rw [rpow_def, Complex.cpow_def, if_neg]
  · have : Complex.log x * y = ↑(log (-x) * y) + ↑(y * π) * Complex.I := by
      simp only [Complex.log, abs_of_neg hx, Complex.arg_ofReal_of_neg hx, Complex.abs_ofReal,
        Complex.ofReal_mul]
      ring
    rw [this, Complex.exp_add_mul_I, ← Complex.ofReal_exp, ← Complex.ofReal_cos, ←
      Complex.ofReal_sin, mul_add, ← Complex.ofReal_mul, ← mul_assoc, ← Complex.ofReal_mul,
      Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.I_re, Complex.ofReal_im,
      Real.log_neg_eq_log]
    /-
      x : Real
      hx : LT.lt x 0
      y : Real
      this : Eq (HMul.hMul (Complex.log ↑x) ↑y) (HAdd.hAdd (↑(HMul.hMul (Real.log (N …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Real.exp (HMul.hMul (Real.log x) y)) (Real.cos (HM …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case hnc
      x : Real
      hx : LT.lt x 0
      y : Real
      ⊢ Not (Eq (↑x) 0)
    -/
  · rw [Complex.ofReal_eq_zero]
    /-
      case hnc
      x : Real
      hx : LT.lt x 0
      y : Real
      ⊢ Not (Eq x 0)
    -/
    exact ne_of_lt hx
    /-
      🎉 no goals
    -/


theorem rpow_def_of_nonpos {x : ℝ} (hx : x ≤ 0) (y : ℝ) :
    x ^ y = if x = 0 then if y = 0 then 1 else 0 else exp (log x * y) * cos (y * π) := by
  /-
    x : Real
    hx : LE.le x 0
    y : Real
    ⊢ Eq (HPow.hPow x y) (ite (Eq x 0) (ite (Eq y 0) 1 0) (HMul.hMul (Real.exp (HM …
  -/
                       /-
                         🎉 no goals
                       -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [rpow_def, *]; exact rpow_def_of_neg (lt_of_le_of_ne hx h) _
                                           /-
                                             🎉 no goals
                                           -/


@[bound]
theorem rpow_pos_of_pos {x : ℝ} (hx : 0 < x) (y : ℝ) : 0 < x ^ y := by
  /-
    x : Real
    hx : LT.lt 0 x
    y : Real
    ⊢ LT.lt 0 (HPow.hPow x y)
  -/
  rw [rpow_def_of_pos hx]; apply exp_pos
                           /-
                             🎉 no goals
                           -/


@[simp]
                                                  /-
                                                    x : Real
                                                    ⊢ Eq (HPow.hPow x 0) 1
                                                  -/
theorem rpow_zero (x : ℝ) : x ^ (0 : ℝ) = 1 := by simp [rpow_def]
                                                  /-
                                                    🎉 no goals
                                                  -/


                                                      /-
                                                        x : Real
                                                        ⊢ LT.lt 0 (HPow.hPow x 0)
                                                      -/
theorem rpow_zero_pos (x : ℝ) : 0 < x ^ (0 : ℝ) := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                              /-
                                                                x : Real
                                                                h : Ne x 0
                                                                ⊢ Eq (HPow.hPow 0 x) 0
                                                              -/
theorem zero_rpow {x : ℝ} (h : x ≠ 0) : (0 : ℝ) ^ x = 0 := by simp [rpow_def, *]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem zero_rpow_eq_iff {x : ℝ} {a : ℝ} : 0 ^ x = a ↔ x ≠ 0 ∧ a = 0 ∨ x = 0 ∧ a = 1 := by
  /-
    x a : Real
    ⊢ Iff (Eq (HPow.hPow 0 x) a) (Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1 …
  -/
  constructor
    /-
      case mp
      x a : Real
      ⊢ Eq (HPow.hPow 0 x) a → Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
    -/
  · intro hyp
    /-
      case mp
      x a : Real
      hyp : Eq (HPow.hPow 0 x) a
      ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
    -/
    simp only [rpow_def, Complex.ofReal_zero] at hyp
    /-
      case mp
      x a : Real
      hyp : Eq (HPow.hPow 0 ↑x).re a
      ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
    -/
    by_cases h : x = 0
      /-
        case pos
        x a : Real
        hyp : Eq (HPow.hPow 0 ↑x).re a
        h : Eq x 0
        ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
      -/
    · subst h
      /-
        case pos
        a : Real
        hyp : Eq (HPow.hPow 0 ↑0).re a
        ⊢ Or (And (Ne 0 0) (Eq a 0)) (And (Eq 0 0) (Eq a 1))
      -/
      simp only [Complex.one_re, Complex.ofReal_zero, Complex.cpow_zero] at hyp
      /-
        case pos
        a : Real
        hyp : Eq 1 a
        ⊢ Or (And (Ne 0 0) (Eq a 0)) (And (Eq 0 0) (Eq a 1))
      -/
      exact Or.inr ⟨rfl, hyp.symm⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        x a : Real
        hyp : Eq (HPow.hPow 0 ↑x).re a
        h : Not (Eq x 0)
        ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
      -/
    · rw [Complex.zero_cpow (Complex.ofReal_ne_zero.mpr h)] at hyp
      /-
        case neg
        x a : Real
        hyp : Eq (Complex.re 0) a
        h : Not (Eq x 0)
        ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1))
      -/
      exact Or.inl ⟨h, hyp.symm⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x a : Real
      ⊢ Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1)) → Eq (HPow.hPow 0 x) a
    -/
  · rintro (⟨h, rfl⟩ | ⟨rfl, rfl⟩)
      /-
        case mpr.inl.intro
        x : Real
        h : Ne x 0
        ⊢ Eq (HPow.hPow 0 x) 0
      -/
    · exact zero_rpow h
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        ⊢ Eq (HPow.hPow 0 0) 1
      -/
    · exact rpow_zero _
      /-
        🎉 no goals
      -/


theorem eq_zero_rpow_iff {x : ℝ} {a : ℝ} : a = 0 ^ x ↔ x ≠ 0 ∧ a = 0 ∨ x = 0 ∧ a = 1 := by
  /-
    x a : Real
    ⊢ Iff (Eq a (HPow.hPow 0 x)) (Or (And (Ne x 0) (Eq a 0)) (And (Eq x 0) (Eq a 1 …
  -/
  rw [← zero_rpow_eq_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                                 /-
                                                   x : Real
                                                   ⊢ Eq (HPow.hPow x 1) x
                                                 -/
theorem rpow_one (x : ℝ) : x ^ (1 : ℝ) = x := by simp [rpow_def]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                 /-
                                                   x : Real
                                                   ⊢ Eq (HPow.hPow 1 x) 1
                                                 -/
theorem one_rpow (x : ℝ) : (1 : ℝ) ^ x = 1 := by simp [rpow_def]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem zero_rpow_le_one (x : ℝ) : (0 : ℝ) ^ x ≤ 1 := by
  /-
    x : Real
    ⊢ LE.le (HPow.hPow 0 x) 1
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : x = 0 <;> simp [h, zero_le_one]
                         /-
                           🎉 no goals
                         -/


theorem zero_rpow_nonneg (x : ℝ) : 0 ≤ (0 : ℝ) ^ x := by
  /-
    x : Real
    ⊢ LE.le 0 (HPow.hPow 0 x)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : x = 0 <;> simp [h, zero_le_one]
                         /-
                           🎉 no goals
                         -/


@[bound]
theorem rpow_nonneg {x : ℝ} (hx : 0 ≤ x) (y : ℝ) : 0 ≤ x ^ y := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ LE.le 0 (HPow.hPow x y)
  -/
  rw [rpow_def_of_nonneg hx]; split_ifs <;>
    /-
      case pos
      x : Real
      hx : LE.le 0 x
      y : Real
      h✝¹ : Eq x 0
      h✝ : Eq y 0
      ⊢ LE.le 0 1
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp only [zero_le_one, le_refl, le_of_lt (exp_pos _)]
    /-
      🎉 no goals
    -/


theorem abs_rpow_of_nonneg {x y : ℝ} (hx_nonneg : 0 ≤ x) : |x ^ y| = |x| ^ y := by
  /-
    x y : Real
    hx_nonneg : LE.le 0 x
    ⊢ Eq (abs (HPow.hPow x y)) (HPow.hPow (abs x) y)
  -/
  have h_rpow_nonneg : 0 ≤ x ^ y := Real.rpow_nonneg hx_nonneg _
  /-
    x y : Real
    hx_nonneg : LE.le 0 x
    h_rpow_nonneg : LE.le 0 (HPow.hPow x y)
    ⊢ Eq (abs (HPow.hPow x y)) (HPow.hPow (abs x) y)
  -/
  rw [abs_eq_self.mpr hx_nonneg, abs_eq_self.mpr h_rpow_nonneg]
  /-
    🎉 no goals
  -/


@[bound]
theorem abs_rpow_le_abs_rpow (x y : ℝ) : |x ^ y| ≤ |x| ^ y := by
  /-
    x y : Real
    ⊢ LE.le (abs (HPow.hPow x y)) (HPow.hPow (abs x) y)
  -/
  rcases le_or_lt 0 x with hx | hx
    /-
      case inl
      x y : Real
      hx : LE.le 0 x
      ⊢ LE.le (abs (HPow.hPow x y)) (HPow.hPow (abs x) y)
    -/
  · rw [abs_rpow_of_nonneg hx]
    /-
      🎉 no goals
    -/
  · rw [abs_of_neg hx, rpow_def_of_neg hx, rpow_def_of_pos (neg_pos.2 hx), log_neg_eq_log, abs_mul,
      abs_of_pos (exp_pos _)]
    /-
      case inr
      x y : Real
      hx : LT.lt x 0
      ⊢ LE.le (HMul.hMul (Real.exp (HMul.hMul (Real.log x) y)) (abs (Real.cos (HMul. …
    -/
    exact mul_le_of_le_one_right (exp_pos _).le (abs_cos_le_one _)
    /-
      🎉 no goals
    -/


theorem abs_rpow_le_exp_log_mul (x y : ℝ) : |x ^ y| ≤ exp (log x * y) := by
  /-
    x y : Real
    ⊢ LE.le (abs (HPow.hPow x y)) (Real.exp (HMul.hMul (Real.log x) y))
  -/
  refine (abs_rpow_le_abs_rpow x y).trans ?_
  /-
    x y : Real
    ⊢ LE.le (HPow.hPow (abs x) y) (Real.exp (HMul.hMul (Real.log x) y))
  -/
  by_cases hx : x = 0
    /-
      case pos
      x y : Real
      hx : Eq x 0
      ⊢ LE.le (HPow.hPow (abs x) y) (Real.exp (HMul.hMul (Real.log x) y))
    -/
                            /-
                              🎉 no goals
                            -/
  · by_cases hy : y = 0 <;> simp [hx, hy, zero_le_one]
                            /-
                              🎉 no goals
                            -/
    /-
      case neg
      x y : Real
      hx : Not (Eq x 0)
      ⊢ LE.le (HPow.hPow (abs x) y) (Real.exp (HMul.hMul (Real.log x) y))
    -/
  · rw [rpow_def_of_pos (abs_pos.2 hx), log_abs]
    /-
      🎉 no goals
    -/


lemma rpow_inv_log (hx₀ : 0 < x) (hx₁ : x ≠ 1) : x ^ (log x)⁻¹ = exp 1 := by
  /-
    x : Real
    hx₀ : LT.lt 0 x
    hx₁ : Ne x 1
    ⊢ Eq (HPow.hPow x (Inv.inv (Real.log x))) (Real.exp 1)
  -/
  rw [rpow_def_of_pos hx₀, mul_inv_cancel₀]
  /-
    x : Real
    hx₀ : LT.lt 0 x
    hx₁ : Ne x 1
    ⊢ Ne (Real.log x) 0
  -/
  exact log_ne_zero.2 ⟨hx₀.ne', hx₁, (hx₀.trans' <| by norm_num).ne'⟩
  /-
    🎉 no goals
  -/


/-- See `Real.rpow_inv_log` for the equality when `x ≠ 1` is strictly positive. -/
lemma rpow_inv_log_le_exp_one : x ^ (log x)⁻¹ ≤ exp 1 := by
  calc
    _ ≤ |x ^ (log x)⁻¹| := le_abs_self _
    _ ≤ |x| ^ (log x)⁻¹ := abs_rpow_le_abs_rpow ..
  /-
    case calc.step
    x : Real
    ⊢ LE.le (HPow.hPow (abs x) (Inv.inv (Real.log x))) (Real.exp 1)
  -/
  rw [← log_abs]
  /-
    case calc.step
    x : Real
    ⊢ LE.le (HPow.hPow (abs x) (Inv.inv (Real.log (abs x)))) (Real.exp 1)
  -/
  obtain hx | hx := (abs_nonneg x).eq_or_gt
    /-
      case calc.step.inl
      x : Real
      hx : Eq (abs x) 0
      ⊢ LE.le (HPow.hPow (abs x) (Inv.inv (Real.log (abs x)))) (Real.exp 1)
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case calc.step.inr
      x : Real
      hx : LT.lt 0 (abs x)
      ⊢ LE.le (HPow.hPow (abs x) (Inv.inv (Real.log (abs x)))) (Real.exp 1)
    -/
  · rw [rpow_def_of_pos hx]
    /-
      case calc.step.inr
      x : Real
      hx : LT.lt 0 (abs x)
      ⊢ LE.le (Real.exp (HMul.hMul (Real.log (abs x)) (Inv.inv (Real.log (abs x))))) …
    -/
    gcongr
    /-
      case calc.step.inr.h
      x : Real
      hx : LT.lt 0 (abs x)
      ⊢ LE.le (HMul.hMul (Real.log (abs x)) (Inv.inv (Real.log (abs x)))) 1
    -/
    exact mul_inv_le_one
    /-
      🎉 no goals
    -/


theorem norm_rpow_of_nonneg {x y : ℝ} (hx_nonneg : 0 ≤ x) : ‖x ^ y‖ = ‖x‖ ^ y := by
  /-
    x y : Real
    hx_nonneg : LE.le 0 x
    ⊢ Eq (Norm.norm (HPow.hPow x y)) (HPow.hPow (Norm.norm x) y)
  -/
  simp_rw [Real.norm_eq_abs]
  /-
    x y : Real
    hx_nonneg : LE.le 0 x
    ⊢ Eq (abs (HPow.hPow x y)) (HPow.hPow (abs x) y)
  -/
  exact abs_rpow_of_nonneg hx_nonneg
  /-
    🎉 no goals
  -/


theorem rpow_add (hx : 0 < x) (y z : ℝ) : x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x : Real
    hx : LT.lt 0 x
    y z : Real
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  simp only [rpow_def_of_pos hx, mul_add, exp_add]
  /-
    🎉 no goals
  -/


theorem rpow_add' (hx : 0 ≤ x) (h : y + z ≠ 0) : x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x y z : Real
    hx : LE.le 0 x
    h : Ne (HAdd.hAdd y z) 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  rcases hx.eq_or_lt with (rfl | pos)
    /-
      case inl
      y z : Real
      h : Ne (HAdd.hAdd y z) 0
      hx : LE.le 0 0
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow 0 y) (HPow.hPow 0 z))
    -/
  · rw [zero_rpow h, zero_eq_mul]
    /-
      case inl
      y z : Real
      h : Ne (HAdd.hAdd y z) 0
      hx : LE.le 0 0
      ⊢ Or (Eq (HPow.hPow 0 y) 0) (Eq (HPow.hPow 0 z) 0)
    -/
    have : y ≠ 0 ∨ z ≠ 0 := not_and_or.1 fun ⟨hy, hz⟩ => h <| hy.symm ▸ hz.symm ▸ zero_add 0
    /-
      case inl
      y z : Real
      h : Ne (HAdd.hAdd y z) 0
      hx : LE.le 0 0
      this : Or (Ne y 0) (Ne z 0)
      ⊢ Or (Eq (HPow.hPow 0 y) 0) (Eq (HPow.hPow 0 z) 0)
    -/
    exact this.imp zero_rpow zero_rpow
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Real
      hx : LE.le 0 x
      h : Ne (HAdd.hAdd y z) 0
      pos : LT.lt 0 x
      ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
    -/
  · exact rpow_add pos _ _
    /-
      🎉 no goals
    -/


/-- Variant of `Real.rpow_add'` that avoids having to prove `y + z = w` twice. -/
lemma rpow_of_add_eq (hx : 0 ≤ x) (hw : w ≠ 0) (h : y + z = w) : x ^ w = x ^ y * x ^ z := by
  /-
    w x y z : Real
    hx : LE.le 0 x
    hw : Ne w 0
    h : Eq (HAdd.hAdd y z) w
    ⊢ Eq (HPow.hPow x w) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  rw [← h, rpow_add' hx]; rwa [h]
                          /-
                            🎉 no goals
                          -/


theorem rpow_add_of_nonneg (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
    x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    hz : LE.le 0 z
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  rcases hy.eq_or_lt with (rfl | hy)
    /-
      case inl
      x z : Real
      hx : LE.le 0 x
      hz : LE.le 0 z
      hy : LE.le 0 0
      ⊢ Eq (HPow.hPow x (HAdd.hAdd 0 z)) (HMul.hMul (HPow.hPow x 0) (HPow.hPow x z))
    -/
  · rw [zero_add, rpow_zero, one_mul]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x y z : Real
    hx : LE.le 0 x
    hy✝ : LE.le 0 y
    hz : LE.le 0 z
    hy : LT.lt 0 y
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  exact rpow_add' hx (ne_of_gt <| add_pos_of_pos_of_nonneg hy hz)
  /-
    🎉 no goals
  -/


/-- For `0 ≤ x`, the only problematic case in the equality `x ^ y * x ^ z = x ^ (y + z)` is for
`x = 0` and `y + z = 0`, where the right hand side is `1` while the left hand side can vanish.
The inequality is always true, though, and given in this lemma. -/
theorem le_rpow_add {x : ℝ} (hx : 0 ≤ x) (y z : ℝ) : x ^ y * x ^ z ≤ x ^ (y + z) := by
  /-
    x : Real
    hx : LE.le 0 x
    y z : Real
    ⊢ LE.le (HMul.hMul (HPow.hPow x y) (HPow.hPow x z)) (HPow.hPow x (HAdd.hAdd y  …
  -/
  rcases le_iff_eq_or_lt.1 hx with (H | pos)
    /-
      case inl
      x : Real
      hx : LE.le 0 x
      y z : Real
      H : Eq 0 x
      ⊢ LE.le (HMul.hMul (HPow.hPow x y) (HPow.hPow x z)) (HPow.hPow x (HAdd.hAdd y  …
    -/
  · by_cases h : y + z = 0
      /-
        case pos
        x : Real
        hx : LE.le 0 x
        y z : Real
        H : Eq 0 x
        h : Eq (HAdd.hAdd y z) 0
        ⊢ LE.le (HMul.hMul (HPow.hPow x y) (HPow.hPow x z)) (HPow.hPow x (HAdd.hAdd y  …
      -/
    · simp only [H.symm, h, rpow_zero]
      calc
        (0 : ℝ) ^ y * 0 ^ z ≤ 1 * 1 :=
          mul_le_mul (zero_rpow_le_one y) (zero_rpow_le_one z) (zero_rpow_nonneg z) zero_le_one
        _ = 1 := by simp

      /-
        case neg
        x : Real
        hx : LE.le 0 x
        y z : Real
        H : Eq 0 x
        h : Not (Eq (HAdd.hAdd y z) 0)
        ⊢ LE.le (HMul.hMul (HPow.hPow x y) (HPow.hPow x z)) (HPow.hPow x (HAdd.hAdd y  …
      -/
    · simp [rpow_add', ← H, h]
      /-
        🎉 no goals
      -/
    /-
      case inr
      x : Real
      hx : LE.le 0 x
      y z : Real
      pos : LT.lt 0 x
      ⊢ LE.le (HMul.hMul (HPow.hPow x y) (HPow.hPow x z)) (HPow.hPow x (HAdd.hAdd y  …
    -/
  · simp [rpow_add pos]
    /-
      🎉 no goals
    -/


theorem rpow_sum_of_pos {ι : Type*} {a : ℝ} (ha : 0 < a) (f : ι → ℝ) (s : Finset ι) :
    (a ^ ∑ x ∈ s, f x) = ∏ x ∈ s, a ^ f x :=
  map_sum (⟨⟨fun (x : ℝ) => (a ^ x : ℝ), rpow_zero a⟩, rpow_add ha⟩ : ℝ →+ (Additive ℝ)) f s


theorem rpow_sum_of_nonneg {ι : Type*} {a : ℝ} (ha : 0 ≤ a) {s : Finset ι} {f : ι → ℝ}
    (h : ∀ x ∈ s, 0 ≤ f x) : (a ^ ∑ x ∈ s, f x) = ∏ x ∈ s, a ^ f x := by
  /-
    ι : Type u_1
    a : Real
    ha : LE.le 0 a
    s : Finset ι
    f : ι → Real
    h : ∀ (x : ι), Membership.mem s x → LE.le 0 (f x)
    ⊢ Eq (HPow.hPow a (s.sum fun x => f x)) (s.prod fun x => HPow.hPow a (f x))
  -/
  induction' s using Finset.cons_induction with i s hi ihs
    /-
      case empty
      ι : Type u_1
      a : Real
      ha : LE.le 0 a
      f : ι → Real
      h : ∀ (x : ι), Membership.mem EmptyCollection.emptyCollection x → LE.le 0 (f x)
      ⊢ Eq (HPow.hPow a (EmptyCollection.emptyCollection.sum fun x => f x)) (EmptyCo …
    -/
  · rw [sum_empty, Finset.prod_empty, rpow_zero]
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      a : Real
      ha : LE.le 0 a
      f : ι → Real
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ihs : (∀ (x : ι), Membership.mem s x → LE.le 0 (f x)) → Eq (HPow.hPow a (s.sum …
      h : ∀ (x : ι), Membership.mem (Finset.cons i s hi) x → LE.le 0 (f x)
      ⊢ Eq (HPow.hPow a ((Finset.cons i s hi).sum fun x => f x)) ((Finset.cons i s h …
    -/
  · rw [forall_mem_cons] at h
    /-
      case cons
      ι : Type u_1
      a : Real
      ha : LE.le 0 a
      f : ι → Real
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ihs : (∀ (x : ι), Membership.mem s x → LE.le 0 (f x)) → Eq (HPow.hPow a (s.sum …
      h : And (LE.le 0 (f i)) (∀ (x : ι), Membership.mem s x → LE.le 0 (f x))
      ⊢ Eq (HPow.hPow a ((Finset.cons i s hi).sum fun x => f x)) ((Finset.cons i s h …
    -/
    rw [sum_cons, prod_cons, ← ihs h.2, rpow_add_of_nonneg ha h.1 (sum_nonneg h.2)]
    /-
      🎉 no goals
    -/


theorem rpow_neg {x : ℝ} (hx : 0 ≤ x) (y : ℝ) : x ^ (-y) = (x ^ y)⁻¹ := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ Eq (HPow.hPow x (Neg.neg y)) (Inv.inv (HPow.hPow x y))
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
  simp only [rpow_def_of_nonneg hx]; split_ifs <;> simp_all [exp_neg]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem rpow_sub {x : ℝ} (hx : 0 < x) (y z : ℝ) : x ^ (y - z) = x ^ y / x ^ z := by
  /-
    x : Real
    hx : LT.lt 0 x
    y z : Real
    ⊢ Eq (HPow.hPow x (HSub.hSub y z)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x z))
  -/
  simp only [sub_eq_add_neg, rpow_add hx, rpow_neg (le_of_lt hx), div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem rpow_sub' {x : ℝ} (hx : 0 ≤ x) {y z : ℝ} (h : y - z ≠ 0) : x ^ (y - z) = x ^ y / x ^ z := by
  /-
    x : Real
    hx : LE.le 0 x
    y z : Real
    h : Ne (HSub.hSub y z) 0
    ⊢ Eq (HPow.hPow x (HSub.hSub y z)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x z))
  -/
  simp only [sub_eq_add_neg] at h ⊢
  /-
    x : Real
    hx : LE.le 0 x
    y z : Real
    h : Ne (HAdd.hAdd y (Neg.neg z)) 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y (Neg.neg z))) (HDiv.hDiv (HPow.hPow x y) (HPow. …
  -/
  simp only [rpow_add' hx h, rpow_neg hx, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


protected theorem _root_.HasCompactSupport.rpow_const {α : Type*} [TopologicalSpace α] {f : α → ℝ}
    (hf : HasCompactSupport f) {r : ℝ} (hr : r ≠ 0) : HasCompactSupport (fun x ↦ f x ^ r) :=
  hf.comp_left (g := (· ^ r)) (Real.zero_rpow hr)


theorem ofReal_cpow {x : ℝ} (hx : 0 ≤ x) (y : ℝ) : ((x ^ y : ℝ) : ℂ) = (x : ℂ) ^ (y : ℂ) := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow ↑x ↑y)
  -/
  simp only [Real.rpow_def_of_nonneg hx, Complex.cpow_def, ofReal_eq_zero]; split_ifs <;>
    /-
      case pos
      x : Real
      hx : LE.le 0 x
      y : Real
      h✝¹ : Eq x 0
      h✝ : Eq y 0
      ⊢ Eq (↑1) 1
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [Complex.ofReal_log hx]
    /-
      🎉 no goals
    -/


theorem ofReal_cpow_of_nonpos {x : ℝ} (hx : x ≤ 0) (y : ℂ) :
    (x : ℂ) ^ y = (-x : ℂ) ^ y * exp (π * I * y) := by
  /-
    x : Real
    hx : LE.le x 0
    y : Complex
    ⊢ Eq (HPow.hPow (↑x) y) (HMul.hMul (HPow.hPow (Neg.neg ↑x) y) (Complex.exp (HM …
  -/
  rcases hx.eq_or_lt with (rfl | hlt)
    /-
      case inl
      y : Complex
      hx : LE.le 0 0
      ⊢ Eq (HPow.hPow (↑0) y) (HMul.hMul (HPow.hPow (Neg.neg ↑0) y) (Complex.exp (HM …
    -/
                                            /-
                                              🎉 no goals
                                            -/
  · rcases eq_or_ne y 0 with (rfl | hy) <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/
  /-
    case inr
    x : Real
    hx : LE.le x 0
    y : Complex
    hlt : LT.lt x 0
    ⊢ Eq (HPow.hPow (↑x) y) (HMul.hMul (HPow.hPow (Neg.neg ↑x) y) (Complex.exp (HM …
  -/
  have hne : (x : ℂ) ≠ 0 := ofReal_ne_zero.mpr hlt.ne
  rw [cpow_def_of_ne_zero hne, cpow_def_of_ne_zero (neg_ne_zero.2 hne), ← exp_add, ← add_mul, log,
    log, abs.map_neg, arg_ofReal_of_neg hlt, ← ofReal_neg,
    arg_ofReal_of_nonneg (neg_nonneg.2 hx), ofReal_zero, zero_mul, add_zero]


lemma cpow_ofReal (x : ℂ) (y : ℝ) :
    x ^ (y : ℂ) = ↑(abs x ^ y) * (Real.cos (arg x * y) + Real.sin (arg x * y) * I) := by
  /-
    x : Complex
    y : Real
    ⊢ Eq (HPow.hPow x ↑y) (HMul.hMul (↑(HPow.hPow (Complex.abs x) y)) (HAdd.hAdd ( …
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      y : Real
      ⊢ Eq (HPow.hPow 0 ↑y) (HMul.hMul (↑(HPow.hPow (Complex.abs 0) y)) (HAdd.hAdd ( …
    -/
  · simp [ofReal_cpow le_rfl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Complex
      y : Real
      hx : Ne x 0
      ⊢ Eq (HPow.hPow x ↑y) (HMul.hMul (↑(HPow.hPow (Complex.abs x) y)) (HAdd.hAdd ( …
    -/
  · rw [cpow_def_of_ne_zero hx, exp_eq_exp_re_mul_sin_add_cos, mul_comm (log x)]
    /-
      case inr
      x : Complex
      y : Real
      hx : Ne x 0
      ⊢ Eq (HMul.hMul (Complex.exp ↑(HMul.hMul (↑y) (Complex.log x)).re) (HAdd.hAdd  …
    -/
    norm_cast
    rw [re_ofReal_mul, im_ofReal_mul, log_re, log_im, mul_comm y, mul_comm y, Real.exp_mul,
      Real.exp_log]
    /-
      case inr
      x : Complex
      y : Real
      hx : Ne x 0
      ⊢ LT.lt 0 (Complex.abs x)
    -/
    rwa [abs.pos_iff]
    /-
      🎉 no goals
    -/


lemma cpow_ofReal_re (x : ℂ) (y : ℝ) : (x ^ (y : ℂ)).re = (abs x) ^ y * Real.cos (arg x * y) := by
  /-
    x : Complex
    y : Real
    ⊢ Eq (HPow.hPow x ↑y).re (HMul.hMul (HPow.hPow (Complex.abs x) y) (Real.cos (H …
  -/
  rw [cpow_ofReal]; generalize arg x * y = z; simp [Real.cos]
                                              /-
                                                🎉 no goals
                                              -/


lemma cpow_ofReal_im (x : ℂ) (y : ℝ) : (x ^ (y : ℂ)).im = (abs x) ^ y * Real.sin (arg x * y) := by
  /-
    x : Complex
    y : Real
    ⊢ Eq (HPow.hPow x ↑y).im (HMul.hMul (HPow.hPow (Complex.abs x) y) (Real.sin (H …
  -/
  rw [cpow_ofReal]; generalize arg x * y = z; simp [Real.sin]
                                              /-
                                                🎉 no goals
                                              -/


theorem abs_cpow_of_ne_zero {z : ℂ} (hz : z ≠ 0) (w : ℂ) :
    abs (z ^ w) = abs z ^ w.re / Real.exp (arg z * im w) := by
  rw [cpow_def_of_ne_zero hz, abs_exp, mul_re, log_re, log_im, Real.exp_sub,
    Real.rpow_def_of_pos (abs.pos hz)]


theorem abs_cpow_of_imp {z w : ℂ} (h : z = 0 → w.re = 0 → w = 0) :
    abs (z ^ w) = abs z ^ w.re / Real.exp (arg z * im w) := by
  /-
    z w : Complex
    h : Eq z 0 → Eq w.re 0 → Eq w 0
    ⊢ Eq (Complex.abs (HPow.hPow z w)) (HDiv.hDiv (HPow.hPow (Complex.abs z) w.re) …
  -/
  rcases ne_or_eq z 0 with (hz | rfl) <;> [exact abs_cpow_of_ne_zero hz w; rw [map_zero]]
  /-
    case inr
    w : Complex
    h : Eq 0 0 → Eq w.re 0 → Eq w 0
    ⊢ Eq (Complex.abs (HPow.hPow 0 w)) (HDiv.hDiv (HPow.hPow 0 w.re) (Real.exp (HM …
  -/
  rcases eq_or_ne w.re 0 with hw | hw
    /-
      case inr.inl
      w : Complex
      h : Eq 0 0 → Eq w.re 0 → Eq w 0
      hw : Eq w.re 0
      ⊢ Eq (Complex.abs (HPow.hPow 0 w)) (HDiv.hDiv (HPow.hPow 0 w.re) (Real.exp (HM …
    -/
  · simp [hw, h rfl hw]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      w : Complex
      h : Eq 0 0 → Eq w.re 0 → Eq w 0
      hw : Ne w.re 0
      ⊢ Eq (Complex.abs (HPow.hPow 0 w)) (HDiv.hDiv (HPow.hPow 0 w.re) (Real.exp (HM …
    -/
  · rw [Real.zero_rpow hw, zero_div, zero_cpow, map_zero]
    /-
      case inr.inr
      w : Complex
      h : Eq 0 0 → Eq w.re 0 → Eq w 0
      hw : Ne w.re 0
      ⊢ Ne w 0
    -/
    exact ne_of_apply_ne re hw
    /-
      🎉 no goals
    -/


theorem abs_cpow_le (z w : ℂ) : abs (z ^ w) ≤ abs z ^ w.re / Real.exp (arg z * im w) := by
  /-
    z w : Complex
    ⊢ LE.le (Complex.abs (HPow.hPow z w)) (HDiv.hDiv (HPow.hPow (Complex.abs z) w. …
  -/
  by_cases h : z = 0 → w.re = 0 → w = 0
    /-
      case pos
      z w : Complex
      h : Eq z 0 → Eq w.re 0 → Eq w 0
      ⊢ LE.le (Complex.abs (HPow.hPow z w)) (HDiv.hDiv (HPow.hPow (Complex.abs z) w. …
    -/
  · exact (abs_cpow_of_imp h).le
    /-
      🎉 no goals
    -/
    /-
      case neg
      z w : Complex
      h : Not (Eq z 0 → Eq w.re 0 → Eq w 0)
      ⊢ LE.le (Complex.abs (HPow.hPow z w)) (HDiv.hDiv (HPow.hPow (Complex.abs z) w. …
    -/
  · push_neg at h
    /-
      case neg
      z w : Complex
      h : And (Eq z 0) (And (Eq w.re 0) (Ne w 0))
      ⊢ LE.le (Complex.abs (HPow.hPow z w)) (HDiv.hDiv (HPow.hPow (Complex.abs z) w. …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


@[simp]
theorem abs_cpow_real (x : ℂ) (y : ℝ) : abs (x ^ (y : ℂ)) = Complex.abs x ^ y := by
  /-
    x : Complex
    y : Real
    ⊢ Eq (Complex.abs (HPow.hPow x ↑y)) (HPow.hPow (Complex.abs x) y)
  -/
                           /-
                             🎉 no goals
                           -/
  rw [abs_cpow_of_imp] <;> simp
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem abs_cpow_inv_nat (x : ℂ) (n : ℕ) : abs (x ^ (n⁻¹ : ℂ)) = Complex.abs x ^ (n⁻¹ : ℝ) := by
  /-
    x : Complex
    n : Nat
    ⊢ Eq (Complex.abs (HPow.hPow x (Inv.inv ↑n))) (HPow.hPow (Complex.abs x) (Inv. …
  -/
  rw [← abs_cpow_real]; simp [-abs_cpow_real]
                        /-
                          🎉 no goals
                        -/


theorem abs_cpow_eq_rpow_re_of_pos {x : ℝ} (hx : 0 < x) (y : ℂ) : abs (x ^ y) = x ^ y.re := by
  rw [abs_cpow_of_ne_zero (ofReal_ne_zero.mpr hx.ne'), arg_ofReal_of_nonneg hx.le,
    zero_mul, Real.exp_zero, div_one, abs_of_nonneg hx.le]


theorem abs_cpow_eq_rpow_re_of_nonneg {x : ℝ} (hx : 0 ≤ x) {y : ℂ} (hy : re y ≠ 0) :
    abs (x ^ y) = x ^ re y := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Complex
    hy : Ne y.re 0
    ⊢ Eq (Complex.abs (HPow.hPow (↑x) y)) (HPow.hPow x y.re)
  -/
                           /-
                             🎉 no goals
                           -/
  rw [abs_cpow_of_imp] <;> simp [*, arg_ofReal_of_nonneg, _root_.abs_of_nonneg]
                           /-
                             🎉 no goals
                           -/


lemma norm_natCast_cpow_of_re_ne_zero (n : ℕ) {s : ℂ} (hs : s.re ≠ 0) :
    ‖(n : ℂ) ^ s‖ = (n : ℝ) ^ (s.re) := by
  /-
    n : Nat
    s : Complex
    hs : Ne s.re 0
    ⊢ Eq (Norm.norm (HPow.hPow (↑n) s)) (HPow.hPow (↑n) s.re)
  -/
  rw [norm_eq_abs, ← ofReal_natCast, abs_cpow_eq_rpow_re_of_nonneg n.cast_nonneg hs]
  /-
    🎉 no goals
  -/


lemma norm_natCast_cpow_of_pos {n : ℕ} (hn : 0 < n) (s : ℂ) :
    ‖(n : ℂ) ^ s‖ = (n : ℝ) ^ (s.re) := by
  /-
    n : Nat
    hn : LT.lt 0 n
    s : Complex
    ⊢ Eq (Norm.norm (HPow.hPow (↑n) s)) (HPow.hPow (↑n) s.re)
  -/
  rw [norm_eq_abs, ← ofReal_natCast, abs_cpow_eq_rpow_re_of_pos (Nat.cast_pos.mpr hn) _]
  /-
    🎉 no goals
  -/


lemma norm_natCast_cpow_pos_of_pos {n : ℕ} (hn : 0 < n) (s : ℂ) : 0 < ‖(n : ℂ) ^ s‖ :=
  (norm_natCast_cpow_of_pos hn _).symm ▸ Real.rpow_pos_of_pos (Nat.cast_pos.mpr hn) _


theorem cpow_mul_ofReal_nonneg {x : ℝ} (hx : 0 ≤ x) (y : ℝ) (z : ℂ) :
    (x : ℂ) ^ (↑y * z) = (↑(x ^ y) : ℂ) ^ z := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    z : Complex
    ⊢ Eq (HPow.hPow (↑x) (HMul.hMul (↑y) z)) (HPow.hPow (↑(HPow.hPow x y)) z)
  -/
  rw [cpow_mul, ofReal_cpow hx]
    /-
      case h₁
      x : Real
      hx : LE.le 0 x
      y : Real
      z : Complex
      ⊢ LT.lt (Neg.neg Real.pi) (HMul.hMul (Complex.log ↑x) ↑y).im
    -/
  · rw [← ofReal_log hx, ← ofReal_mul, ofReal_im, neg_lt_zero]; exact Real.pi_pos
                                                                /-
                                                                  🎉 no goals
                                                                -/
    /-
      case h₂
      x : Real
      hx : LE.le 0 x
      y : Real
      z : Complex
      ⊢ LE.le (HMul.hMul (Complex.log ↑x) ↑y).im Real.pi
    -/
  · rw [← ofReal_log hx, ← ofReal_mul, ofReal_im]; exact Real.pi_pos.le
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Extension for the `positivity` tactic: exponentiation by a real number is positive (namely 1)
when the exponent is zero. The other cases are done in `evalRpow`. -/
@[positivity (_ : ℝ) ^ (0 : ℝ)]
def evalRpowZero : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q($a ^ (0 : ℝ)) =>
    assertInstancesCommute
    pure (.positive q(Real.rpow_zero_pos $a))
  | _, _, _ => throwError "not Real.rpow"


/-- Extension for the `positivity` tactic: exponentiation by a real number is nonnegative when
the base is nonnegative and positive when the base is positive. -/
@[positivity (_ : ℝ) ^ (_ : ℝ)]
def evalRpow : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q($a ^ ($b : ℝ)) =>
    let ra ← core q(inferInstance) q(inferInstance) a
    assertInstancesCommute
    match ra with
    | .positive pa =>
        pure (.positive q(Real.rpow_pos_of_pos $pa $b))
    | .nonnegative pa =>
        pure (.nonnegative q(Real.rpow_nonneg $pa $b))
    | _ => pure .none
  | _, _, _ => throwError "not Real.rpow"


theorem rpow_mul {x : ℝ} (hx : 0 ≤ x) (y z : ℝ) : x ^ (y * z) = (x ^ y) ^ z := by
  rw [← Complex.ofReal_inj, Complex.ofReal_cpow (rpow_nonneg hx _),
      Complex.ofReal_cpow hx, Complex.ofReal_mul, Complex.cpow_mul, Complex.ofReal_cpow hx] <;>
    simp only [(Complex.ofReal_mul _ _).symm, (Complex.ofReal_log hx).symm, Complex.ofReal_im,
      neg_lt_zero, pi_pos, le_of_lt pi_pos]


lemma rpow_add_intCast {x : ℝ} (hx : x ≠ 0) (y : ℝ) (n : ℤ) : x ^ (y + n) = x ^ y * x ^ n := by
  rw [rpow_def, rpow_def, Complex.ofReal_add,
    Complex.cpow_add _ _ (Complex.ofReal_ne_zero.mpr hx), Complex.ofReal_intCast,
    Complex.cpow_intCast, ← Complex.ofReal_zpow, mul_comm, Complex.re_ofReal_mul, mul_comm]


lemma rpow_add_natCast {x : ℝ} (hx : x ≠ 0) (y : ℝ) (n : ℕ) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    x : Real
    hx : Ne x 0
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  simpa using rpow_add_intCast hx y n
  /-
    🎉 no goals
  -/


lemma rpow_sub_intCast {x : ℝ} (hx : x ≠ 0) (y : ℝ) (n : ℕ) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    x : Real
    hx : Ne x 0
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  simpa using rpow_add_intCast hx y (-n)
  /-
    🎉 no goals
  -/


lemma rpow_sub_natCast {x : ℝ} (hx : x ≠ 0) (y : ℝ) (n : ℕ) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    x : Real
    hx : Ne x 0
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  simpa using rpow_sub_intCast hx y n
  /-
    🎉 no goals
  -/


lemma rpow_add_intCast' (hx : 0 ≤ x) {n : ℤ} (h : y + n ≠ 0) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    x y : Real
    hx : LE.le 0 x
    n : Int
    h : Ne (HAdd.hAdd y ↑n) 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  rw [rpow_add' hx h, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma rpow_add_natCast' (hx : 0 ≤ x) (h : y + n ≠ 0) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    x y : Real
    n : Nat
    hx : LE.le 0 x
    h : Ne (HAdd.hAdd y ↑n) 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  rw [rpow_add' hx h, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_sub_intCast' (hx : 0 ≤ x) {n : ℤ} (h : y - n ≠ 0) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    x y : Real
    hx : LE.le 0 x
    n : Int
    h : Ne (HSub.hSub y ↑n) 0
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  rw [rpow_sub' hx h, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma rpow_sub_natCast' (hx : 0 ≤ x) (h : y - n ≠ 0) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    x y : Real
    n : Nat
    hx : LE.le 0 x
    h : Ne (HSub.hSub y ↑n) 0
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  rw [rpow_sub' hx h, rpow_natCast]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-28")] alias rpow_add_int := rpow_add_intCast

@[deprecated (since := "2024-08-28")] alias rpow_add_nat := rpow_add_natCast

@[deprecated (since := "2024-08-28")] alias rpow_sub_int := rpow_sub_intCast

@[deprecated (since := "2024-08-28")] alias rpow_sub_nat := rpow_sub_natCast

@[deprecated (since := "2024-08-28")] alias rpow_add_int' := rpow_add_intCast'

@[deprecated (since := "2024-08-28")] alias rpow_add_nat' := rpow_add_natCast'

@[deprecated (since := "2024-08-28")] alias rpow_sub_int' := rpow_sub_intCast'

@[deprecated (since := "2024-08-28")] alias rpow_sub_nat' := rpow_sub_natCast'


theorem rpow_add_one {x : ℝ} (hx : x ≠ 0) (y : ℝ) : x ^ (y + 1) = x ^ y * x := by
  /-
    x : Real
    hx : Ne x 0
    y : Real
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y 1)) (HMul.hMul (HPow.hPow x y) x)
  -/
  simpa using rpow_add_natCast hx y 1
  /-
    🎉 no goals
  -/


theorem rpow_sub_one {x : ℝ} (hx : x ≠ 0) (y : ℝ) : x ^ (y - 1) = x ^ y / x := by
  /-
    x : Real
    hx : Ne x 0
    y : Real
    ⊢ Eq (HPow.hPow x (HSub.hSub y 1)) (HDiv.hDiv (HPow.hPow x y) x)
  -/
  simpa using rpow_sub_natCast hx y 1
  /-
    🎉 no goals
  -/


lemma rpow_add_one' (hx : 0 ≤ x) (h : y + 1 ≠ 0) : x ^ (y + 1) = x ^ y * x := by
  /-
    x y : Real
    hx : LE.le 0 x
    h : Ne (HAdd.hAdd y 1) 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y 1)) (HMul.hMul (HPow.hPow x y) x)
  -/
  rw [rpow_add' hx h, rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_one_add' (hx : 0 ≤ x) (h : 1 + y ≠ 0) : x ^ (1 + y) = x * x ^ y := by
  /-
    x y : Real
    hx : LE.le 0 x
    h : Ne (HAdd.hAdd 1 y) 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd 1 y)) (HMul.hMul x (HPow.hPow x y))
  -/
  rw [rpow_add' hx h, rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_sub_one' (hx : 0 ≤ x) (h : y - 1 ≠ 0) : x ^ (y - 1) = x ^ y / x := by
  /-
    x y : Real
    hx : LE.le 0 x
    h : Ne (HSub.hSub y 1) 0
    ⊢ Eq (HPow.hPow x (HSub.hSub y 1)) (HDiv.hDiv (HPow.hPow x y) x)
  -/
  rw [rpow_sub' hx h, rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_one_sub' (hx : 0 ≤ x) (h : 1 - y ≠ 0) : x ^ (1 - y) = x / x ^ y := by
  /-
    x y : Real
    hx : LE.le 0 x
    h : Ne (HSub.hSub 1 y) 0
    ⊢ Eq (HPow.hPow x (HSub.hSub 1 y)) (HDiv.hDiv x (HPow.hPow x y))
  -/
  rw [rpow_sub' hx h, rpow_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem rpow_two (x : ℝ) : x ^ (2 : ℝ) = x ^ 2 := by
  /-
    x : Real
    ⊢ Eq (HPow.hPow x 2) (HPow.hPow x 2)
  -/
  rw [← rpow_natCast]
  /-
    x : Real
    ⊢ Eq (HPow.hPow x 2) (HPow.hPow x ↑2)
  -/
  simp only [Nat.cast_ofNat]
  /-
    🎉 no goals
  -/


theorem rpow_neg_one (x : ℝ) : x ^ (-1 : ℝ) = x⁻¹ := by
  /-
    x : Real
    ⊢ Eq (HPow.hPow x (-1)) (Inv.inv x)
  -/
  suffices H : x ^ ((-1 : ℤ) : ℝ) = x⁻¹ by rwa [Int.cast_neg, Int.cast_one] at H
  /-
    x : Real
    ⊢ Eq (HPow.hPow x ↑(-1)) (Inv.inv x)
  -/
  simp only [rpow_intCast, zpow_one, zpow_neg]
  /-
    🎉 no goals
  -/


theorem mul_rpow (hx : 0 ≤ x) (hy : 0 ≤ y) : (x * y) ^ z = x ^ z * y ^ z := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (HMul.hMul (HPow.hPow x z) (HPow.hPow y z))
  -/
  iterate 2 rw [Real.rpow_def_of_nonneg]; split_ifs with h_ifs <;> simp_all
    /-
      case neg
      x y z : Real
      hx : LE.le 0 x
      hy : LE.le 0 y
      h_ifs✝ : Not (Eq y 0)
      h_ifs : Not (Eq x 0)
      ⊢ Eq (Real.exp (HMul.hMul (Real.log (HMul.hMul x y)) z)) (HMul.hMul (Real.exp  …
    -/
  · rw [log_mul ‹_› ‹_›, add_mul, exp_add, rpow_def_of_pos (hy.lt_of_ne' ‹_›)]
    /-
      🎉 no goals
    -/
  /-
    case neg.hx
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    h_ifs : And (Not (Eq x 0)) (Not (Eq y 0))
    ⊢ LE.le 0 x
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


theorem inv_rpow (hx : 0 ≤ x) (y : ℝ) : x⁻¹ ^ y = (x ^ y)⁻¹ := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    ⊢ Eq (HPow.hPow (Inv.inv x) y) (Inv.inv (HPow.hPow x y))
  -/
  simp only [← rpow_neg_one, ← rpow_mul hx, mul_comm]
  /-
    🎉 no goals
  -/


theorem div_rpow (hx : 0 ≤ x) (hy : 0 ≤ y) (z : ℝ) : (x / y) ^ z = x ^ z / y ^ z := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    z : Real
    ⊢ Eq (HPow.hPow (HDiv.hDiv x y) z) (HDiv.hDiv (HPow.hPow x z) (HPow.hPow y z))
  -/
  simp only [div_eq_mul_inv, mul_rpow hx (inv_nonneg.2 hy), inv_rpow hy]
  /-
    🎉 no goals
  -/


theorem log_rpow {x : ℝ} (hx : 0 < x) (y : ℝ) : log (x ^ y) = y * log x := by
  /-
    x : Real
    hx : LT.lt 0 x
    y : Real
    ⊢ Eq (Real.log (HPow.hPow x y)) (HMul.hMul y (Real.log x))
  -/
  apply exp_injective
  /-
    case a
    x : Real
    hx : LT.lt 0 x
    y : Real
    ⊢ Eq (Real.exp (Real.log (HPow.hPow x y))) (Real.exp (HMul.hMul y (Real.log x)))
  -/
  rw [exp_log (rpow_pos_of_pos hx y), ← exp_log hx, mul_comm, rpow_def_of_pos (exp_pos (log x)) y]
  /-
    🎉 no goals
  -/


theorem mul_log_eq_log_iff {x y z : ℝ} (hx : 0 < x) (hz : 0 < z) :
    y * log x = log z ↔ x ^ y = z :=
  ⟨fun h ↦ log_injOn_pos (rpow_pos_of_pos hx _) hz <| log_rpow hx _ |>.trans h,
     /-
       x y z : Real
       hx : LT.lt 0 x
       hz : LT.lt 0 z
       ⊢ Eq (HPow.hPow x y) z → Eq (HMul.hMul y (Real.log x)) (Real.log z)
     -/
  by rintro rfl; rw [log_rpow hx]⟩
                 /-
                   🎉 no goals
                 -/


@[simp] lemma rpow_rpow_inv (hx : 0 ≤ x) (hy : y ≠ 0) : (x ^ y) ^ y⁻¹ = x := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : Ne y 0
    ⊢ Eq (HPow.hPow (HPow.hPow x y) (Inv.inv y)) x
  -/
  rw [← rpow_mul hx, mul_inv_cancel₀ hy, rpow_one]
  /-
    🎉 no goals
  -/


@[simp] lemma rpow_inv_rpow (hx : 0 ≤ x) (hy : y ≠ 0) : (x ^ y⁻¹) ^ y = x := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : Ne y 0
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv y)) y) x
  -/
  rw [← rpow_mul hx, inv_mul_cancel₀ hy, rpow_one]
  /-
    🎉 no goals
  -/


theorem pow_rpow_inv_natCast (hx : 0 ≤ x) (hn : n ≠ 0) : (x ^ n) ^ (n⁻¹ : ℝ) = x := by
  /-
    x : Real
    n : Nat
    hx : LE.le 0 x
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow x n) (Inv.inv ↑n)) x
  -/
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn
  /-
    x : Real
    n : Nat
    hx : LE.le 0 x
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    ⊢ Eq (HPow.hPow (HPow.hPow x n) (Inv.inv ↑n)) x
  -/
  rw [← rpow_natCast, ← rpow_mul hx, mul_inv_cancel₀ hn0, rpow_one]
  /-
    🎉 no goals
  -/


theorem rpow_inv_natCast_pow (hx : 0 ≤ x) (hn : n ≠ 0) : (x ^ (n⁻¹ : ℝ)) ^ n = x := by
  /-
    x : Real
    n : Nat
    hx : LE.le 0 x
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv ↑n)) n) x
  -/
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn
  /-
    x : Real
    n : Nat
    hx : LE.le 0 x
    hn : Ne n 0
    hn0 : Ne (↑n) 0
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv ↑n)) n) x
  -/
  rw [← rpow_natCast, ← rpow_mul hx, inv_mul_cancel₀ hn0, rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_natCast_mul (hx : 0 ≤ x) (n : ℕ) (z : ℝ) : x ^ (n * z) = (x ^ n) ^ z := by
  /-
    x : Real
    hx : LE.le 0 x
    n : Nat
    z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) z)) (HPow.hPow (HPow.hPow x n) z)
  -/
  rw [rpow_mul hx, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_mul_natCast (hx : 0 ≤ x) (y : ℝ) (n : ℕ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [rpow_mul hx, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_intCast_mul (hx : 0 ≤ x) (n : ℤ) (z : ℝ) : x ^ (n * z) = (x ^ n) ^ z := by
  /-
    x : Real
    hx : LE.le 0 x
    n : Int
    z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) z)) (HPow.hPow (HPow.hPow x n) z)
  -/
  rw [rpow_mul hx, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma rpow_mul_intCast (hx : 0 ≤ x) (y : ℝ) (n : ℤ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x : Real
    hx : LE.le 0 x
    y : Real
    n : Int
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [rpow_mul hx, rpow_intCast]
  /-
    🎉 no goals
  -/


@[gcongr, bound]
theorem rpow_lt_rpow (hx : 0 ≤ x) (hxy : x < y) (hz : 0 < z) : x ^ z < y ^ z := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hxy : LT.lt x y
    hz : LT.lt 0 z
    ⊢ LT.lt (HPow.hPow x z) (HPow.hPow y z)
  -/
  rw [le_iff_eq_or_lt] at hx; cases' hx with hx hx
    /-
      case inl
      x y z : Real
      hxy : LT.lt x y
      hz : LT.lt 0 z
      hx : Eq 0 x
      ⊢ LT.lt (HPow.hPow x z) (HPow.hPow y z)
    -/
  · rw [← hx, zero_rpow (ne_of_gt hz)]
    /-
      case inl
      x y z : Real
      hxy : LT.lt x y
      hz : LT.lt 0 z
      hx : Eq 0 x
      ⊢ LT.lt 0 (HPow.hPow y z)
    -/
    exact rpow_pos_of_pos (by rwa [← hx] at hxy) _
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Real
      hxy : LT.lt x y
      hz : LT.lt 0 z
      hx : LT.lt 0 x
      ⊢ LT.lt (HPow.hPow x z) (HPow.hPow y z)
    -/
  · rw [rpow_def_of_pos hx, rpow_def_of_pos (lt_trans hx hxy), exp_lt_exp]
    /-
      case inr
      x y z : Real
      hxy : LT.lt x y
      hz : LT.lt 0 z
      hx : LT.lt 0 x
      ⊢ LT.lt (HMul.hMul (Real.log x) z) (HMul.hMul (Real.log y) z)
    -/
    exact mul_lt_mul_of_pos_right (log_lt_log hx hxy) hz
    /-
      🎉 no goals
    -/


theorem strictMonoOn_rpow_Ici_of_exponent_pos {r : ℝ} (hr : 0 < r) :
    StrictMonoOn (fun (x : ℝ) => x ^ r) (Set.Ici 0) :=
  fun _ ha _ _ hab => rpow_lt_rpow ha hab hr


@[gcongr, bound]
theorem rpow_le_rpow {x y z : ℝ} (h : 0 ≤ x) (h₁ : x ≤ y) (h₂ : 0 ≤ z) : x ^ z ≤ y ^ z := by
  /-
    x y z : Real
    h : LE.le 0 x
    h₁ : LE.le x y
    h₂ : LE.le 0 z
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow y z)
  -/
  rcases eq_or_lt_of_le h₁ with (rfl | h₁'); · rfl
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case inr
    x y z : Real
    h : LE.le 0 x
    h₁ : LE.le x y
    h₂ : LE.le 0 z
    h₁' : LT.lt x y
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow y z)
  -/
  rcases eq_or_lt_of_le h₂ with (rfl | h₂'); · simp
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case inr.inr
    x y z : Real
    h : LE.le 0 x
    h₁ : LE.le x y
    h₂ : LE.le 0 z
    h₁' : LT.lt x y
    h₂' : LT.lt 0 z
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow y z)
  -/
  exact le_of_lt (rpow_lt_rpow h h₁' h₂')
  /-
    🎉 no goals
  -/


theorem monotoneOn_rpow_Ici_of_exponent_nonneg {r : ℝ} (hr : 0 ≤ r) :
    MonotoneOn (fun (x : ℝ) => x ^ r) (Set.Ici 0) :=
  fun _ ha _ _ hab => rpow_le_rpow ha hab hr


lemma rpow_lt_rpow_of_neg (hx : 0 < x) (hxy : x < y) (hz : z < 0) : y ^ z < x ^ z := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LT.lt x y
    hz : LT.lt z 0
    ⊢ LT.lt (HPow.hPow y z) (HPow.hPow x z)
  -/
  have := hx.trans hxy
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LT.lt x y
    hz : LT.lt z 0
    this : LT.lt 0 y
    ⊢ LT.lt (HPow.hPow y z) (HPow.hPow x z)
  -/
  rw [← inv_lt_inv₀, ← rpow_neg, ← rpow_neg]
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LT.lt x y
    hz : LT.lt z 0
    this : LT.lt 0 y
    ⊢ LT.lt (HPow.hPow x (Neg.neg z)) (HPow.hPow y (Neg.neg z))
  -/
  on_goal 1 => refine rpow_lt_rpow ?_ hxy (neg_pos.2 hz)
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LT.lt x y
    hz : LT.lt z 0
    this : LT.lt 0 y
    ⊢ LE.le 0 x
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


lemma rpow_le_rpow_of_nonpos (hx : 0 < x) (hxy : x ≤ y) (hz : z ≤ 0) : y ^ z ≤ x ^ z := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LE.le x y
    hz : LE.le z 0
    ⊢ LE.le (HPow.hPow y z) (HPow.hPow x z)
  -/
  have := hx.trans_le hxy
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LE.le x y
    hz : LE.le z 0
    this : LT.lt 0 y
    ⊢ LE.le (HPow.hPow y z) (HPow.hPow x z)
  -/
  rw [← inv_le_inv₀, ← rpow_neg, ← rpow_neg]
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LE.le x y
    hz : LE.le z 0
    this : LT.lt 0 y
    ⊢ LE.le (HPow.hPow x (Neg.neg z)) (HPow.hPow y (Neg.neg z))
  -/
  on_goal 1 => refine rpow_le_rpow ?_ hxy (neg_nonneg.2 hz)
  /-
    x y z : Real
    hx : LT.lt 0 x
    hxy : LE.le x y
    hz : LE.le z 0
    this : LT.lt 0 y
    ⊢ LE.le 0 x
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/


theorem rpow_lt_rpow_iff (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) : x ^ z < y ^ z ↔ x < y :=
  ⟨lt_imp_lt_of_le_imp_le fun h => rpow_le_rpow hy h (le_of_lt hz), fun h => rpow_lt_rpow hx h hz⟩


theorem rpow_le_rpow_iff (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) : x ^ z ≤ y ^ z ↔ x ≤ y :=
  le_iff_le_iff_lt_iff_lt.2 <| rpow_lt_rpow_iff hy hx hz


lemma rpow_lt_rpow_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ^ z < y ^ z ↔ y < x :=
  ⟨lt_imp_lt_of_le_imp_le fun h ↦ rpow_le_rpow_of_nonpos hx h hz.le,
    fun h ↦ rpow_lt_rpow_of_neg hy h hz⟩


lemma rpow_le_rpow_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ^ z ≤ y ^ z ↔ y ≤ x :=
  le_iff_le_iff_lt_iff_lt.2 <| rpow_lt_rpow_iff_of_neg hy hx hz


lemma le_rpow_inv_iff_of_pos (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) : x ≤ y ^ z⁻¹ ↔ x ^ z ≤ y := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    hz : LT.lt 0 z
    ⊢ Iff (LE.le x (HPow.hPow y (Inv.inv z))) (LE.le (HPow.hPow x z) y)
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  rw [← rpow_le_rpow_iff hx _ hz, rpow_inv_rpow] <;> positivity
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma rpow_inv_le_iff_of_pos (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) : x ^ z⁻¹ ≤ y ↔ x ≤ y ^ z := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (Inv.inv z)) y) (LE.le x (HPow.hPow y z))
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  rw [← rpow_le_rpow_iff _ hy hz, rpow_inv_rpow] <;> positivity
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma lt_rpow_inv_iff_of_pos (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) : x < y ^ z⁻¹ ↔ x ^ z < y :=
  lt_iff_lt_of_le_iff_le <| rpow_inv_le_iff_of_pos hy hx hz


lemma rpow_inv_lt_iff_of_pos (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) : x ^ z⁻¹ < y ↔ x < y ^ z :=
  lt_iff_lt_of_le_iff_le <| le_rpow_inv_iff_of_pos hy hx hz


theorem le_rpow_inv_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) :
    x ≤ y ^ z⁻¹ ↔ y ≤ x ^ z := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    hz : LT.lt z 0
    ⊢ Iff (LE.le x (HPow.hPow y (Inv.inv z))) (LE.le y (HPow.hPow x z))
  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  rw [← rpow_le_rpow_iff_of_neg _ hx hz, rpow_inv_rpow _ hz.ne] <;> positivity
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem lt_rpow_inv_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) :
    x < y ^ z⁻¹ ↔ y < x ^ z := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    hz : LT.lt z 0
    ⊢ Iff (LT.lt x (HPow.hPow y (Inv.inv z))) (LT.lt y (HPow.hPow x z))
  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  rw [← rpow_lt_rpow_iff_of_neg _ hx hz, rpow_inv_rpow _ hz.ne] <;> positivity
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem rpow_inv_lt_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) :
    x ^ z⁻¹ < y ↔ y ^ z < x := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    hz : LT.lt z 0
    ⊢ Iff (LT.lt (HPow.hPow x (Inv.inv z)) y) (LT.lt (HPow.hPow y z) x)
  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  rw [← rpow_lt_rpow_iff_of_neg hy _ hz, rpow_inv_rpow _ hz.ne] <;> positivity
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem rpow_inv_le_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) :
    x ^ z⁻¹ ≤ y ↔ y ^ z ≤ x := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    hz : LT.lt z 0
    ⊢ Iff (LE.le (HPow.hPow x (Inv.inv z)) y) (LE.le (HPow.hPow y z) x)
  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  rw [← rpow_le_rpow_iff_of_neg hy _ hz, rpow_inv_rpow _ hz.ne] <;> positivity
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem rpow_lt_rpow_of_exponent_lt (hx : 1 < x) (hyz : y < z) : x ^ y < x ^ z := by
  /-
    x y z : Real
    hx : LT.lt 1 x
    hyz : LT.lt y z
    ⊢ LT.lt (HPow.hPow x y) (HPow.hPow x z)
  -/
  repeat' rw [rpow_def_of_pos (lt_trans zero_lt_one hx)]
  /-
    x y z : Real
    hx : LT.lt 1 x
    hyz : LT.lt y z
    ⊢ LT.lt (Real.exp (HMul.hMul (Real.log x) y)) (Real.exp (HMul.hMul (Real.log x …
  -/
  rw [exp_lt_exp]; exact mul_lt_mul_of_pos_left hyz (log_pos hx)
                   /-
                     🎉 no goals
                   -/


@[gcongr]
theorem rpow_le_rpow_of_exponent_le (hx : 1 ≤ x) (hyz : y ≤ z) : x ^ y ≤ x ^ z := by
  /-
    x y z : Real
    hx : LE.le 1 x
    hyz : LE.le y z
    ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
  -/
  repeat' rw [rpow_def_of_pos (lt_of_lt_of_le zero_lt_one hx)]
  /-
    x y z : Real
    hx : LE.le 1 x
    hyz : LE.le y z
    ⊢ LE.le (Real.exp (HMul.hMul (Real.log x) y)) (Real.exp (HMul.hMul (Real.log x …
  -/
  rw [exp_le_exp]; exact mul_le_mul_of_nonneg_left hyz (log_nonneg hx)
                   /-
                     🎉 no goals
                   -/


theorem rpow_lt_rpow_of_exponent_neg {x y z : ℝ} (hy : 0 < y) (hxy : y < x) (hz : z < 0) :
    x ^ z < y ^ z := by
  /-
    x y z : Real
    hy : LT.lt 0 y
    hxy : LT.lt y x
    hz : LT.lt z 0
    ⊢ LT.lt (HPow.hPow x z) (HPow.hPow y z)
  -/
  have hx : 0 < x := hy.trans hxy
  rw [← neg_neg z, Real.rpow_neg (le_of_lt hx) (-z), Real.rpow_neg (le_of_lt hy) (-z),
      inv_lt_inv₀ (rpow_pos_of_pos hx _) (rpow_pos_of_pos hy _)]
  /-
    x y z : Real
    hy : LT.lt 0 y
    hxy : LT.lt y x
    hz : LT.lt z 0
    hx : LT.lt 0 x
    ⊢ LT.lt (HPow.hPow y (Neg.neg z)) (HPow.hPow x (Neg.neg z))
  -/
  exact Real.rpow_lt_rpow (by positivity) hxy <| neg_pos_of_neg hz
  /-
    🎉 no goals
  -/


theorem strictAntiOn_rpow_Ioi_of_exponent_neg {r : ℝ} (hr : r < 0) :
    StrictAntiOn (fun (x : ℝ) => x ^ r) (Set.Ioi 0) :=
  fun _ ha _ _ hab => rpow_lt_rpow_of_exponent_neg ha hab hr


theorem rpow_le_rpow_of_exponent_nonpos {x y : ℝ} (hy : 0 < y) (hxy : y ≤ x) (hz : z ≤ 0) :
    x ^ z ≤ y ^ z := by
  /-
    z x y : Real
    hy : LT.lt 0 y
    hxy : LE.le y x
    hz : LE.le z 0
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow y z)
  -/
  rcases ne_or_eq z 0 with hz_zero | rfl
  case inl =>
    rcases ne_or_eq x y with hxy' | rfl
    case inl =>
      exact le_of_lt <| rpow_lt_rpow_of_exponent_neg hy (Ne.lt_of_le (id (Ne.symm hxy')) hxy)
        (Ne.lt_of_le hz_zero hz)
    case inr => simp
  /-
    case inr
    x y : Real
    hy : LT.lt 0 y
    hxy : LE.le y x
    hz : LE.le 0 0
    ⊢ LE.le (HPow.hPow x 0) (HPow.hPow y 0)
  -/
  case inr => simp
  /-
    🎉 no goals
  -/


theorem antitoneOn_rpow_Ioi_of_exponent_nonpos {r : ℝ} (hr : r ≤ 0) :
    AntitoneOn (fun (x : ℝ) => x ^ r) (Set.Ioi 0) :=
  fun _ ha _ _ hab => rpow_le_rpow_of_exponent_nonpos ha hab hr


@[simp]
theorem rpow_le_rpow_left_iff (hx : 1 < x) : x ^ y ≤ x ^ z ↔ y ≤ z := by
  /-
    x y z : Real
    hx : LT.lt 1 x
    ⊢ Iff (LE.le (HPow.hPow x y) (HPow.hPow x z)) (LE.le y z)
  -/
  have x_pos : 0 < x := lt_trans zero_lt_one hx
  rw [← log_le_log_iff (rpow_pos_of_pos x_pos y) (rpow_pos_of_pos x_pos z), log_rpow x_pos,
    log_rpow x_pos, mul_le_mul_right (log_pos hx)]


@[simp]
theorem rpow_lt_rpow_left_iff (hx : 1 < x) : x ^ y < x ^ z ↔ y < z := by
  /-
    x y z : Real
    hx : LT.lt 1 x
    ⊢ Iff (LT.lt (HPow.hPow x y) (HPow.hPow x z)) (LT.lt y z)
  -/
  rw [lt_iff_not_le, rpow_le_rpow_left_iff hx, lt_iff_not_le]
  /-
    🎉 no goals
  -/


theorem rpow_lt_rpow_of_exponent_gt (hx0 : 0 < x) (hx1 : x < 1) (hyz : z < y) : x ^ y < x ^ z := by
  /-
    x y z : Real
    hx0 : LT.lt 0 x
    hx1 : LT.lt x 1
    hyz : LT.lt z y
    ⊢ LT.lt (HPow.hPow x y) (HPow.hPow x z)
  -/
  repeat' rw [rpow_def_of_pos hx0]
  /-
    x y z : Real
    hx0 : LT.lt 0 x
    hx1 : LT.lt x 1
    hyz : LT.lt z y
    ⊢ LT.lt (Real.exp (HMul.hMul (Real.log x) y)) (Real.exp (HMul.hMul (Real.log x …
  -/
  rw [exp_lt_exp]; exact mul_lt_mul_of_neg_left hyz (log_neg hx0 hx1)
                   /-
                     🎉 no goals
                   -/


theorem rpow_le_rpow_of_exponent_ge (hx0 : 0 < x) (hx1 : x ≤ 1) (hyz : z ≤ y) : x ^ y ≤ x ^ z := by
  /-
    x y z : Real
    hx0 : LT.lt 0 x
    hx1 : LE.le x 1
    hyz : LE.le z y
    ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
  -/
  repeat' rw [rpow_def_of_pos hx0]
  /-
    x y z : Real
    hx0 : LT.lt 0 x
    hx1 : LE.le x 1
    hyz : LE.le z y
    ⊢ LE.le (Real.exp (HMul.hMul (Real.log x) y)) (Real.exp (HMul.hMul (Real.log x …
  -/
  rw [exp_le_exp]; exact mul_le_mul_of_nonpos_left hyz (log_nonpos (le_of_lt hx0) hx1)
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem rpow_le_rpow_left_iff_of_base_lt_one (hx0 : 0 < x) (hx1 : x < 1) :
    x ^ y ≤ x ^ z ↔ z ≤ y := by
  rw [← log_le_log_iff (rpow_pos_of_pos hx0 y) (rpow_pos_of_pos hx0 z), log_rpow hx0, log_rpow hx0,
    mul_le_mul_right_of_neg (log_neg hx0 hx1)]


@[simp]
theorem rpow_lt_rpow_left_iff_of_base_lt_one (hx0 : 0 < x) (hx1 : x < 1) :
    x ^ y < x ^ z ↔ z < y := by
  /-
    x y z : Real
    hx0 : LT.lt 0 x
    hx1 : LT.lt x 1
    ⊢ Iff (LT.lt (HPow.hPow x y) (HPow.hPow x z)) (LT.lt z y)
  -/
  rw [lt_iff_not_le, rpow_le_rpow_left_iff_of_base_lt_one hx0 hx1, lt_iff_not_le]
  /-
    🎉 no goals
  -/


theorem rpow_lt_one {x z : ℝ} (hx1 : 0 ≤ x) (hx2 : x < 1) (hz : 0 < z) : x ^ z < 1 := by
  /-
    x z : Real
    hx1 : LE.le 0 x
    hx2 : LT.lt x 1
    hz : LT.lt 0 z
    ⊢ LT.lt (HPow.hPow x z) 1
  -/
  rw [← one_rpow z]
  /-
    x z : Real
    hx1 : LE.le 0 x
    hx2 : LT.lt x 1
    hz : LT.lt 0 z
    ⊢ LT.lt (HPow.hPow x z) (HPow.hPow 1 z)
  -/
  exact rpow_lt_rpow hx1 hx2 hz
  /-
    🎉 no goals
  -/


theorem rpow_le_one {x z : ℝ} (hx1 : 0 ≤ x) (hx2 : x ≤ 1) (hz : 0 ≤ z) : x ^ z ≤ 1 := by
  /-
    x z : Real
    hx1 : LE.le 0 x
    hx2 : LE.le x 1
    hz : LE.le 0 z
    ⊢ LE.le (HPow.hPow x z) 1
  -/
  rw [← one_rpow z]
  /-
    x z : Real
    hx1 : LE.le 0 x
    hx2 : LE.le x 1
    hz : LE.le 0 z
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow 1 z)
  -/
  exact rpow_le_rpow hx1 hx2 hz
  /-
    🎉 no goals
  -/


theorem rpow_lt_one_of_one_lt_of_neg {x z : ℝ} (hx : 1 < x) (hz : z < 0) : x ^ z < 1 := by
  /-
    x z : Real
    hx : LT.lt 1 x
    hz : LT.lt z 0
    ⊢ LT.lt (HPow.hPow x z) 1
  -/
  convert rpow_lt_rpow_of_exponent_lt hx hz
  /-
    case h.e'_4
    x z : Real
    hx : LT.lt 1 x
    hz : LT.lt z 0
    ⊢ Eq 1 (HPow.hPow x 0)
  -/
  exact (rpow_zero x).symm
  /-
    🎉 no goals
  -/


theorem rpow_le_one_of_one_le_of_nonpos {x z : ℝ} (hx : 1 ≤ x) (hz : z ≤ 0) : x ^ z ≤ 1 := by
  /-
    x z : Real
    hx : LE.le 1 x
    hz : LE.le z 0
    ⊢ LE.le (HPow.hPow x z) 1
  -/
  convert rpow_le_rpow_of_exponent_le hx hz
  /-
    case h.e'_4
    x z : Real
    hx : LE.le 1 x
    hz : LE.le z 0
    ⊢ Eq 1 (HPow.hPow x 0)
  -/
  exact (rpow_zero x).symm
  /-
    🎉 no goals
  -/


theorem one_lt_rpow {x z : ℝ} (hx : 1 < x) (hz : 0 < z) : 1 < x ^ z := by
  /-
    x z : Real
    hx : LT.lt 1 x
    hz : LT.lt 0 z
    ⊢ LT.lt 1 (HPow.hPow x z)
  -/
  rw [← one_rpow z]
  /-
    x z : Real
    hx : LT.lt 1 x
    hz : LT.lt 0 z
    ⊢ LT.lt (HPow.hPow 1 z) (HPow.hPow x z)
  -/
  exact rpow_lt_rpow zero_le_one hx hz
  /-
    🎉 no goals
  -/


theorem one_le_rpow {x z : ℝ} (hx : 1 ≤ x) (hz : 0 ≤ z) : 1 ≤ x ^ z := by
  /-
    x z : Real
    hx : LE.le 1 x
    hz : LE.le 0 z
    ⊢ LE.le 1 (HPow.hPow x z)
  -/
  rw [← one_rpow z]
  /-
    x z : Real
    hx : LE.le 1 x
    hz : LE.le 0 z
    ⊢ LE.le (HPow.hPow 1 z) (HPow.hPow x z)
  -/
  exact rpow_le_rpow zero_le_one hx hz
  /-
    🎉 no goals
  -/


theorem one_lt_rpow_of_pos_of_lt_one_of_neg (hx1 : 0 < x) (hx2 : x < 1) (hz : z < 0) :
    1 < x ^ z := by
  /-
    x z : Real
    hx1 : LT.lt 0 x
    hx2 : LT.lt x 1
    hz : LT.lt z 0
    ⊢ LT.lt 1 (HPow.hPow x z)
  -/
  convert rpow_lt_rpow_of_exponent_gt hx1 hx2 hz
  /-
    case h.e'_3
    x z : Real
    hx1 : LT.lt 0 x
    hx2 : LT.lt x 1
    hz : LT.lt z 0
    ⊢ Eq 1 (HPow.hPow x 0)
  -/
  exact (rpow_zero x).symm
  /-
    🎉 no goals
  -/


theorem one_le_rpow_of_pos_of_le_one_of_nonpos (hx1 : 0 < x) (hx2 : x ≤ 1) (hz : z ≤ 0) :
    1 ≤ x ^ z := by
  /-
    x z : Real
    hx1 : LT.lt 0 x
    hx2 : LE.le x 1
    hz : LE.le z 0
    ⊢ LE.le 1 (HPow.hPow x z)
  -/
  convert rpow_le_rpow_of_exponent_ge hx1 hx2 hz
  /-
    case h.e'_3
    x z : Real
    hx1 : LT.lt 0 x
    hx2 : LE.le x 1
    hz : LE.le z 0
    ⊢ Eq 1 (HPow.hPow x 0)
  -/
  exact (rpow_zero x).symm
  /-
    🎉 no goals
  -/


theorem rpow_lt_one_iff_of_pos (hx : 0 < x) : x ^ y < 1 ↔ 1 < x ∧ y < 0 ∨ x < 1 ∧ 0 < y := by
  /-
    x y : Real
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt (HPow.hPow x y) 1) (Or (And (LT.lt 1 x) (LT.lt y 0)) (And (LT.lt  …
  -/
  rw [rpow_def_of_pos hx, exp_lt_one_iff, mul_neg_iff, log_pos_iff hx, log_neg_iff hx]
  /-
    🎉 no goals
  -/


theorem rpow_lt_one_iff (hx : 0 ≤ x) :
    x ^ y < 1 ↔ x = 0 ∧ y ≠ 0 ∨ 1 < x ∧ y < 0 ∨ x < 1 ∧ 0 < y := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Iff (LT.lt (HPow.hPow x y) 1) (Or (And (Eq x 0) (Ne y 0)) (Or (And (LT.lt 1  …
  -/
  rcases hx.eq_or_lt with (rfl | hx)
    /-
      case inl
      y : Real
      hx : LE.le 0 0
      ⊢ Iff (LT.lt (HPow.hPow 0 y) 1) (Or (And (Eq 0 0) (Ne y 0)) (Or (And (LT.lt 1  …
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  · rcases _root_.em (y = 0) with (rfl | hy) <;> simp [*, lt_irrefl, zero_lt_one]
                                                 /-
                                                   🎉 no goals
                                                 -/
    /-
      case inr
      x y : Real
      hx✝ : LE.le 0 x
      hx : LT.lt 0 x
      ⊢ Iff (LT.lt (HPow.hPow x y) 1) (Or (And (Eq x 0) (Ne y 0)) (Or (And (LT.lt 1  …
    -/
  · simp [rpow_lt_one_iff_of_pos hx, hx.ne.symm]
    /-
      🎉 no goals
    -/


theorem rpow_lt_one_iff' {x y : ℝ} (hx : 0 ≤ x) (hy : 0 < y) :
    x ^ y < 1 ↔ x < 1 := by
  /-
    x y : Real
    hx : LE.le 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (HPow.hPow x y) 1) (LT.lt x 1)
  -/
  rw [← Real.rpow_lt_rpow_iff hx zero_le_one hy, Real.one_rpow]
  /-
    🎉 no goals
  -/


theorem one_lt_rpow_iff_of_pos (hx : 0 < x) : 1 < x ^ y ↔ 1 < x ∧ 0 < y ∨ x < 1 ∧ y < 0 := by
  /-
    x y : Real
    hx : LT.lt 0 x
    ⊢ Iff (LT.lt 1 (HPow.hPow x y)) (Or (And (LT.lt 1 x) (LT.lt 0 y)) (And (LT.lt  …
  -/
  rw [rpow_def_of_pos hx, one_lt_exp_iff, mul_pos_iff, log_pos_iff hx, log_neg_iff hx]
  /-
    🎉 no goals
  -/


theorem one_lt_rpow_iff (hx : 0 ≤ x) : 1 < x ^ y ↔ 1 < x ∧ 0 < y ∨ 0 < x ∧ x < 1 ∧ y < 0 := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Iff (LT.lt 1 (HPow.hPow x y)) (Or (And (LT.lt 1 x) (LT.lt 0 y)) (And (LT.lt  …
  -/
  rcases hx.eq_or_lt with (rfl | hx)
    /-
      case inl
      y : Real
      hx : LE.le 0 0
      ⊢ Iff (LT.lt 1 (HPow.hPow 0 y)) (Or (And (LT.lt 1 0) (LT.lt 0 y)) (And (LT.lt  …
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  · rcases _root_.em (y = 0) with (rfl | hy) <;> simp [*, lt_irrefl, (zero_lt_one' ℝ).not_lt]
                                                 /-
                                                   🎉 no goals
                                                 -/
    /-
      case inr
      x y : Real
      hx✝ : LE.le 0 x
      hx : LT.lt 0 x
      ⊢ Iff (LT.lt 1 (HPow.hPow x y)) (Or (And (LT.lt 1 x) (LT.lt 0 y)) (And (LT.lt  …
    -/
  · simp [one_lt_rpow_iff_of_pos hx, hx]
    /-
      🎉 no goals
    -/


/-- This is a more general but less convenient version of `rpow_le_rpow_of_exponent_ge`.
This version allows `x = 0`, so it explicitly forbids `x = y = 0`, `z ≠ 0`. -/
theorem rpow_le_rpow_of_exponent_ge_of_imp (hx0 : 0 ≤ x) (hx1 : x ≤ 1) (hyz : z ≤ y)
    (h : x = 0 → y = 0 → z = 0) :
    x ^ y ≤ x ^ z := by
  /-
    x y z : Real
    hx0 : LE.le 0 x
    hx1 : LE.le x 1
    hyz : LE.le z y
    h : Eq x 0 → Eq y 0 → Eq z 0
    ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
  -/
  rcases eq_or_lt_of_le hx0 with (rfl | hx0')
    /-
      case inl
      y z : Real
      hyz : LE.le z y
      hx0 : LE.le 0 0
      hx1 : LE.le 0 1
      h : Eq 0 0 → Eq y 0 → Eq z 0
      ⊢ LE.le (HPow.hPow 0 y) (HPow.hPow 0 z)
    -/
  · rcases eq_or_ne y 0 with rfl | hy0
      /-
        case inl.inl
        z : Real
        hx0 : LE.le 0 0
        hx1 : LE.le 0 1
        hyz : LE.le z 0
        h : Eq 0 0 → Eq 0 0 → Eq z 0
        ⊢ LE.le (HPow.hPow 0 0) (HPow.hPow 0 z)
      -/
    · rw [h rfl rfl]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        y z : Real
        hyz : LE.le z y
        hx0 : LE.le 0 0
        hx1 : LE.le 0 1
        h : Eq 0 0 → Eq y 0 → Eq z 0
        hy0 : Ne y 0
        ⊢ LE.le (HPow.hPow 0 y) (HPow.hPow 0 z)
      -/
    · rw [zero_rpow hy0]
      /-
        case inl.inr
        y z : Real
        hyz : LE.le z y
        hx0 : LE.le 0 0
        hx1 : LE.le 0 1
        h : Eq 0 0 → Eq y 0 → Eq z 0
        hy0 : Ne y 0
        ⊢ LE.le 0 (HPow.hPow 0 z)
      -/
      apply zero_rpow_nonneg
      /-
        🎉 no goals
      -/
    /-
      case inr
      x y z : Real
      hx0 : LE.le 0 x
      hx1 : LE.le x 1
      hyz : LE.le z y
      h : Eq x 0 → Eq y 0 → Eq z 0
      hx0' : LT.lt 0 x
      ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
    -/
  · exact rpow_le_rpow_of_exponent_ge hx0' hx1 hyz
    /-
      🎉 no goals
    -/


/-- This version of `rpow_le_rpow_of_exponent_ge` allows `x = 0` but requires `0 ≤ z`.
See also `rpow_le_rpow_of_exponent_ge_of_imp` for the most general version. -/
theorem rpow_le_rpow_of_exponent_ge' (hx0 : 0 ≤ x) (hx1 : x ≤ 1) (hz : 0 ≤ z) (hyz : z ≤ y) :
    x ^ y ≤ x ^ z :=
  rpow_le_rpow_of_exponent_ge_of_imp hx0 hx1 hyz fun _ hy ↦ le_antisymm (hyz.trans_eq hy) hz


theorem self_le_rpow_of_le_one (h₁ : 0 ≤ x) (h₂ : x ≤ 1) (h₃ : y ≤ 1) : x ≤ x ^ y := by
  simpa only [rpow_one]
    using rpow_le_rpow_of_exponent_ge_of_imp h₁ h₂ h₃ fun _ ↦ (absurd · one_ne_zero)


theorem self_le_rpow_of_one_le (h₁ : 1 ≤ x) (h₂ : 1 ≤ y) : x ≤ x ^ y := by
  /-
    x y : Real
    h₁ : LE.le 1 x
    h₂ : LE.le 1 y
    ⊢ LE.le x (HPow.hPow x y)
  -/
  simpa only [rpow_one] using rpow_le_rpow_of_exponent_le h₁ h₂
  /-
    🎉 no goals
  -/


theorem rpow_le_self_of_le_one (h₁ : 0 ≤ x) (h₂ : x ≤ 1) (h₃ : 1 ≤ y) : x ^ y ≤ x := by
  simpa only [rpow_one]
    using rpow_le_rpow_of_exponent_ge_of_imp h₁ h₂ h₃ fun _ ↦ (absurd · (one_pos.trans_le h₃).ne')


theorem rpow_le_self_of_one_le (h₁ : 1 ≤ x) (h₂ : y ≤ 1) : x ^ y ≤ x := by
  /-
    x y : Real
    h₁ : LE.le 1 x
    h₂ : LE.le y 1
    ⊢ LE.le (HPow.hPow x y) x
  -/
  simpa only [rpow_one] using rpow_le_rpow_of_exponent_le h₁ h₂
  /-
    🎉 no goals
  -/


theorem self_lt_rpow_of_lt_one (h₁ : 0 < x) (h₂ : x < 1) (h₃ : y < 1) : x < x ^ y := by
  /-
    x y : Real
    h₁ : LT.lt 0 x
    h₂ : LT.lt x 1
    h₃ : LT.lt y 1
    ⊢ LT.lt x (HPow.hPow x y)
  -/
  simpa only [rpow_one] using rpow_lt_rpow_of_exponent_gt h₁ h₂ h₃
  /-
    🎉 no goals
  -/


theorem self_lt_rpow_of_one_lt (h₁ : 1 < x) (h₂ : 1 < y) : x < x ^ y := by
  /-
    x y : Real
    h₁ : LT.lt 1 x
    h₂ : LT.lt 1 y
    ⊢ LT.lt x (HPow.hPow x y)
  -/
  simpa only [rpow_one] using rpow_lt_rpow_of_exponent_lt h₁ h₂
  /-
    🎉 no goals
  -/


theorem rpow_lt_self_of_lt_one (h₁ : 0 < x) (h₂ : x < 1) (h₃ : 1 < y) : x ^ y < x := by
  /-
    x y : Real
    h₁ : LT.lt 0 x
    h₂ : LT.lt x 1
    h₃ : LT.lt 1 y
    ⊢ LT.lt (HPow.hPow x y) x
  -/
  simpa only [rpow_one] using rpow_lt_rpow_of_exponent_gt h₁ h₂ h₃
  /-
    🎉 no goals
  -/


theorem rpow_lt_self_of_one_lt (h₁ : 1 < x) (h₂ : y < 1) : x ^ y < x := by
  /-
    x y : Real
    h₁ : LT.lt 1 x
    h₂ : LT.lt y 1
    ⊢ LT.lt (HPow.hPow x y) x
  -/
  simpa only [rpow_one] using rpow_lt_rpow_of_exponent_lt h₁ h₂
  /-
    🎉 no goals
  -/


theorem rpow_left_injOn {x : ℝ} (hx : x ≠ 0) : InjOn (fun y : ℝ => y ^ x) { y : ℝ | 0 ≤ y } := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ Set.InjOn (fun y => HPow.hPow y x) (setOf fun y => LE.le 0 y)
  -/
  rintro y hy z hz (hyz : y ^ x = z ^ x)
  /-
    x : Real
    hx : Ne x 0
    y : Real
    hy : Membership.mem (setOf fun y => LE.le 0 y) y
    z : Real
    hz : Membership.mem (setOf fun y => LE.le 0 y) z
    hyz : Eq (HPow.hPow y x) (HPow.hPow z x)
    ⊢ Eq y z
  -/
  rw [← rpow_one y, ← rpow_one z, ← mul_inv_cancel₀ hx, rpow_mul hy, rpow_mul hz, hyz]
  /-
    🎉 no goals
  -/


lemma rpow_left_inj (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : z ≠ 0) : x ^ z = y ^ z ↔ x = y :=
  (rpow_left_injOn hz).eq_iff hx hy


lemma rpow_inv_eq (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : z ≠ 0) : x ^ z⁻¹ = y ↔ x = y ^ z := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    hz : Ne z 0
    ⊢ Iff (Eq (HPow.hPow x (Inv.inv z)) y) (Eq x (HPow.hPow y z))
  -/
  rw [← rpow_left_inj _ hy hz, rpow_inv_rpow hx hz]; positivity
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma eq_rpow_inv (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : z ≠ 0) : x = y ^ z⁻¹ ↔ x ^ z = y := by
  /-
    x y z : Real
    hx : LE.le 0 x
    hy : LE.le 0 y
    hz : Ne z 0
    ⊢ Iff (Eq x (HPow.hPow y (Inv.inv z))) (Eq (HPow.hPow x z) y)
  -/
  rw [← rpow_left_inj hx _ hz, rpow_inv_rpow hy hz]; positivity
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem le_rpow_iff_log_le (hx : 0 < x) (hy : 0 < y) : x ≤ y ^ z ↔ log x ≤ z * log y := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LE.le x (HPow.hPow y z)) (LE.le (Real.log x) (HMul.hMul z (Real.log y)))
  -/
  rw [← log_le_log_iff hx (rpow_pos_of_pos hy z), log_rpow hy]
  /-
    🎉 no goals
  -/


lemma le_pow_iff_log_le (hx : 0 < x) (hy : 0 < y) : x ≤ y ^ n ↔ log x ≤ n * log y :=
  rpow_natCast _ _ ▸ le_rpow_iff_log_le hx hy


lemma le_zpow_iff_log_le {n : ℤ} (hx : 0 < x) (hy : 0 < y) : x ≤ y ^ n ↔ log x ≤ n * log y :=
  rpow_intCast _ _ ▸ le_rpow_iff_log_le hx hy


lemma le_rpow_of_log_le (hy : 0 < y) (h : log x ≤ z * log y) : x ≤ y ^ z := by
  /-
    x y z : Real
    hy : LT.lt 0 y
    h : LE.le (Real.log x) (HMul.hMul z (Real.log y))
    ⊢ LE.le x (HPow.hPow y z)
  -/
  obtain hx | hx := le_or_lt x 0
    /-
      case inl
      x y z : Real
      hy : LT.lt 0 y
      h : LE.le (Real.log x) (HMul.hMul z (Real.log y))
      hx : LE.le x 0
      ⊢ LE.le x (HPow.hPow y z)
    -/
  · exact hx.trans (rpow_pos_of_pos hy _).le
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Real
      hy : LT.lt 0 y
      h : LE.le (Real.log x) (HMul.hMul z (Real.log y))
      hx : LT.lt 0 x
      ⊢ LE.le x (HPow.hPow y z)
    -/
  · exact (le_rpow_iff_log_le hx hy).2 h
    /-
      🎉 no goals
    -/


lemma le_pow_of_log_le (hy : 0 < y) (h : log x ≤ n * log y) : x ≤ y ^ n :=
  rpow_natCast _ _ ▸ le_rpow_of_log_le hy h


lemma le_zpow_of_log_le {n : ℤ} (hy : 0 < y) (h : log x ≤ n * log y) : x ≤ y ^ n :=
  rpow_intCast _ _ ▸ le_rpow_of_log_le hy h


theorem lt_rpow_iff_log_lt (hx : 0 < x) (hy : 0 < y) : x < y ^ z ↔ log x < z * log y := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt x (HPow.hPow y z)) (LT.lt (Real.log x) (HMul.hMul z (Real.log y)))
  -/
  rw [← log_lt_log_iff hx (rpow_pos_of_pos hy z), log_rpow hy]
  /-
    🎉 no goals
  -/


lemma lt_pow_iff_log_lt (hx : 0 < x) (hy : 0 < y) : x < y ^ n ↔ log x < n * log y :=
  rpow_natCast _ _ ▸ lt_rpow_iff_log_lt hx hy


lemma lt_zpow_iff_log_lt {n : ℤ} (hx : 0 < x) (hy : 0 < y) : x < y ^ n ↔ log x < n * log y :=
  rpow_intCast _ _ ▸ lt_rpow_iff_log_lt hx hy


lemma lt_rpow_of_log_lt (hy : 0 < y) (h : log x < z * log y) : x < y ^ z := by
  /-
    x y z : Real
    hy : LT.lt 0 y
    h : LT.lt (Real.log x) (HMul.hMul z (Real.log y))
    ⊢ LT.lt x (HPow.hPow y z)
  -/
  obtain hx | hx := le_or_lt x 0
    /-
      case inl
      x y z : Real
      hy : LT.lt 0 y
      h : LT.lt (Real.log x) (HMul.hMul z (Real.log y))
      hx : LE.le x 0
      ⊢ LT.lt x (HPow.hPow y z)
    -/
  · exact hx.trans_lt (rpow_pos_of_pos hy _)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Real
      hy : LT.lt 0 y
      h : LT.lt (Real.log x) (HMul.hMul z (Real.log y))
      hx : LT.lt 0 x
      ⊢ LT.lt x (HPow.hPow y z)
    -/
  · exact (lt_rpow_iff_log_lt hx hy).2 h
    /-
      🎉 no goals
    -/


lemma lt_pow_of_log_lt (hy : 0 < y) (h : log x < n * log y) : x < y ^ n :=
  rpow_natCast _ _ ▸ lt_rpow_of_log_lt hy h


lemma lt_zpow_of_log_lt {n : ℤ} (hy : 0 < y) (h : log x < n * log y) : x < y ^ n :=
  rpow_intCast _ _ ▸ lt_rpow_of_log_lt hy h


lemma rpow_le_iff_le_log (hx : 0 < x) (hy : 0 < y) : x ^ z ≤ y ↔ z * log x ≤ log y := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LE.le (HPow.hPow x z) y) (LE.le (HMul.hMul z (Real.log x)) (Real.log y))
  -/
  rw [← log_le_log_iff (rpow_pos_of_pos hx _) hy, log_rpow hx]
  /-
    🎉 no goals
  -/


lemma pow_le_iff_le_log (hx : 0 < x) (hy : 0 < y) : x ^ n ≤ y ↔ n * log x ≤ log y := by
  /-
    x y : Real
    n : Nat
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LE.le (HPow.hPow x n) y) (LE.le (HMul.hMul (↑n) (Real.log x)) (Real.log …
  -/
  rw [← rpow_le_iff_le_log hx hy, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma zpow_le_iff_le_log {n : ℤ} (hx : 0 < x) (hy : 0 < y) : x ^ n ≤ y ↔ n * log x ≤ log y := by
  /-
    x y : Real
    n : Int
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LE.le (HPow.hPow x n) y) (LE.le (HMul.hMul (↑n) (Real.log x)) (Real.log …
  -/
  rw [← rpow_le_iff_le_log hx hy, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma le_log_of_rpow_le (hx : 0 < x) (h : x ^ z ≤ y) : z * log x ≤ log y :=
                                 /-
                                   x y z : Real
                                   hx : LT.lt 0 x
                                   h : LE.le (HPow.hPow x z) y
                                   ⊢ LT.lt 0 (HPow.hPow x z)
                                 -/
  log_rpow hx _ ▸ log_le_log (by positivity) h
                                 /-
                                   🎉 no goals
                                 -/


lemma le_log_of_pow_le (hx : 0 < x) (h : x ^ n ≤ y) : n * log x ≤ log y :=
  le_log_of_rpow_le hx (rpow_natCast _ _ ▸ h)


lemma le_log_of_zpow_le {n : ℤ} (hx : 0 < x) (h : x ^ n ≤ y) : n * log x ≤ log y :=
  le_log_of_rpow_le hx (rpow_intCast _ _ ▸ h)


lemma rpow_le_of_le_log (hy : 0 < y) (h : log x ≤ z * log y) : x ≤ y ^ z := by
  /-
    x y z : Real
    hy : LT.lt 0 y
    h : LE.le (Real.log x) (HMul.hMul z (Real.log y))
    ⊢ LE.le x (HPow.hPow y z)
  -/
  obtain hx | hx := le_or_lt x 0
    /-
      case inl
      x y z : Real
      hy : LT.lt 0 y
      h : LE.le (Real.log x) (HMul.hMul z (Real.log y))
      hx : LE.le x 0
      ⊢ LE.le x (HPow.hPow y z)
    -/
  · exact hx.trans (rpow_pos_of_pos hy _).le
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Real
      hy : LT.lt 0 y
      h : LE.le (Real.log x) (HMul.hMul z (Real.log y))
      hx : LT.lt 0 x
      ⊢ LE.le x (HPow.hPow y z)
    -/
  · exact (le_rpow_iff_log_le hx hy).2 h
    /-
      🎉 no goals
    -/


lemma pow_le_of_le_log (hy : 0 < y) (h : log x ≤ n * log y) : x ≤ y ^ n :=
  rpow_natCast _ _ ▸ rpow_le_of_le_log hy h


lemma zpow_le_of_le_log {n : ℤ} (hy : 0 < y) (h : log x ≤ n * log y) : x ≤ y ^ n :=
  rpow_intCast _ _ ▸ rpow_le_of_le_log hy h


lemma rpow_lt_iff_lt_log (hx : 0 < x) (hy : 0 < y) : x ^ z < y ↔ z * log x < log y := by
  /-
    x y z : Real
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (HPow.hPow x z) y) (LT.lt (HMul.hMul z (Real.log x)) (Real.log y))
  -/
  rw [← log_lt_log_iff (rpow_pos_of_pos hx _) hy, log_rpow hx]
  /-
    🎉 no goals
  -/


lemma pow_lt_iff_lt_log (hx : 0 < x) (hy : 0 < y) : x ^ n < y ↔ n * log x < log y := by
  /-
    x y : Real
    n : Nat
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (HPow.hPow x n) y) (LT.lt (HMul.hMul (↑n) (Real.log x)) (Real.log …
  -/
  rw [← rpow_lt_iff_lt_log hx hy, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma zpow_lt_iff_lt_log {n : ℤ} (hx : 0 < x) (hy : 0 < y) : x ^ n < y ↔ n * log x < log y := by
  /-
    x y : Real
    n : Int
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (HPow.hPow x n) y) (LT.lt (HMul.hMul (↑n) (Real.log x)) (Real.log …
  -/
  rw [← rpow_lt_iff_lt_log hx hy, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma lt_log_of_rpow_lt (hx : 0 < x) (h : x ^ z < y) : z * log x < log y :=
                                 /-
                                   x y z : Real
                                   hx : LT.lt 0 x
                                   h : LT.lt (HPow.hPow x z) y
                                   ⊢ LT.lt 0 (HPow.hPow x z)
                                 -/
  log_rpow hx _ ▸ log_lt_log (by positivity) h
                                 /-
                                   🎉 no goals
                                 -/


lemma lt_log_of_pow_lt (hx : 0 < x) (h : x ^ n < y) : n * log x < log y :=
  lt_log_of_rpow_lt hx (rpow_natCast _ _ ▸ h)


lemma lt_log_of_zpow_lt {n : ℤ} (hx : 0 < x) (h : x ^ n < y) : n * log x < log y :=
  lt_log_of_rpow_lt hx (rpow_intCast _ _ ▸ h)


lemma rpow_lt_of_lt_log (hy : 0 < y) (h : log x < z * log y) : x < y ^ z := by
  /-
    x y z : Real
    hy : LT.lt 0 y
    h : LT.lt (Real.log x) (HMul.hMul z (Real.log y))
    ⊢ LT.lt x (HPow.hPow y z)
  -/
  obtain hx | hx := le_or_lt x 0
    /-
      case inl
      x y z : Real
      hy : LT.lt 0 y
      h : LT.lt (Real.log x) (HMul.hMul z (Real.log y))
      hx : LE.le x 0
      ⊢ LT.lt x (HPow.hPow y z)
    -/
  · exact hx.trans_lt (rpow_pos_of_pos hy _)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Real
      hy : LT.lt 0 y
      h : LT.lt (Real.log x) (HMul.hMul z (Real.log y))
      hx : LT.lt 0 x
      ⊢ LT.lt x (HPow.hPow y z)
    -/
  · exact (lt_rpow_iff_log_lt hx hy).2 h
    /-
      🎉 no goals
    -/


lemma pow_lt_of_lt_log (hy : 0 < y) (h : log x < n * log y) : x < y ^ n :=
  rpow_natCast _ _ ▸ rpow_lt_of_lt_log hy h


lemma zpow_lt_of_lt_log {n : ℤ} (hy : 0 < y) (h : log x < n * log y) : x < y ^ n :=
  rpow_intCast _ _ ▸ rpow_lt_of_lt_log hy h


theorem rpow_le_one_iff_of_pos (hx : 0 < x) : x ^ y ≤ 1 ↔ 1 ≤ x ∧ y ≤ 0 ∨ x ≤ 1 ∧ 0 ≤ y := by
  /-
    x y : Real
    hx : LT.lt 0 x
    ⊢ Iff (LE.le (HPow.hPow x y) 1) (Or (And (LE.le 1 x) (LE.le y 0)) (And (LE.le  …
  -/
  rw [rpow_def_of_pos hx, exp_le_one_iff, mul_nonpos_iff, log_nonneg_iff hx, log_nonpos_iff hx]
  /-
    🎉 no goals
  -/


/-- Bound for `|log x * x ^ t|` in the interval `(0, 1]`, for positive real `t`. -/
theorem abs_log_mul_self_rpow_lt (x t : ℝ) (h1 : 0 < x) (h2 : x ≤ 1) (ht : 0 < t) :
    |log x * x ^ t| < 1 / t := by
  /-
    x t : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    ht : LT.lt 0 t
    ⊢ LT.lt (abs (HMul.hMul (Real.log x) (HPow.hPow x t))) (HDiv.hDiv 1 t)
  -/
  rw [lt_div_iff₀ ht]
  /-
    x t : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    ht : LT.lt 0 t
    ⊢ LT.lt (HMul.hMul (abs (HMul.hMul (Real.log x) (HPow.hPow x t))) t) 1
  -/
  have := abs_log_mul_self_lt (x ^ t) (rpow_pos_of_pos h1 t) (rpow_le_one h1.le h2 ht.le)
  /-
    x t : Real
    h1 : LT.lt 0 x
    h2 : LE.le x 1
    ht : LT.lt 0 t
    this : LT.lt (abs (HMul.hMul (Real.log (HPow.hPow x t)) (HPow.hPow x t))) 1
    ⊢ LT.lt (HMul.hMul (abs (HMul.hMul (Real.log x) (HPow.hPow x t))) t) 1
  -/
  rwa [log_rpow h1, mul_assoc, abs_mul, abs_of_pos ht, mul_comm] at this
  /-
    🎉 no goals
  -/


/-- `log x` is bounded above by a multiple of every power of `x` with positive exponent. -/
lemma log_le_rpow_div {x ε : ℝ} (hx : 0 ≤ x) (hε : 0 < ε) : log x ≤ x ^ ε / ε := by
  /-
    x ε : Real
    hx : LE.le 0 x
    hε : LT.lt 0 ε
    ⊢ LE.le (Real.log x) (HDiv.hDiv (HPow.hPow x ε) ε)
  -/
  rcases hx.eq_or_lt with rfl | h
    /-
      case inl
      ε : Real
      hε : LT.lt 0 ε
      hx : LE.le 0 0
      ⊢ LE.le (Real.log 0) (HDiv.hDiv (HPow.hPow 0 ε) ε)
    -/
  · rw [log_zero, zero_rpow hε.ne', zero_div]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x ε : Real
    hx : LE.le 0 x
    hε : LT.lt 0 ε
    h : LT.lt 0 x
    ⊢ LE.le (Real.log x) (HDiv.hDiv (HPow.hPow x ε) ε)
  -/
  rw [le_div_iff₀' hε]
  exact (log_rpow h ε).symm.trans_le <| (log_le_sub_one_of_pos <| rpow_pos_of_pos h ε).trans
    (sub_one_lt _).le


/-- The (real) logarithm of a natural number `n` is bounded by a multiple of every power of `n`
with positive exponent. -/
lemma log_natCast_le_rpow_div (n : ℕ) {ε : ℝ} (hε : 0 < ε) : log n ≤ n ^ ε / ε :=
  log_le_rpow_div n.cast_nonneg hε


lemma strictMono_rpow_of_base_gt_one {b : ℝ} (hb : 1 < b) :
    StrictMono (b ^ · : ℝ → ℝ) := by
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ StrictMono fun x => HPow.hPow b x
  -/
  simp_rw [Real.rpow_def_of_pos (zero_lt_one.trans hb)]
  /-
    b : Real
    hb : LT.lt 1 b
    ⊢ StrictMono fun x => Real.exp (HMul.hMul (Real.log b) x)
  -/
  exact exp_strictMono.comp <| StrictMono.const_mul strictMono_id <| Real.log_pos hb
  /-
    🎉 no goals
  -/


lemma monotone_rpow_of_base_ge_one {b : ℝ} (hb : 1 ≤ b) :
    Monotone (b ^ · : ℝ → ℝ) := by
  /-
    b : Real
    hb : LE.le 1 b
    ⊢ Monotone fun x => HPow.hPow b x
  -/
  rcases lt_or_eq_of_le hb with hb | rfl
  /-
    case inl
    b : Real
    hb✝ : LE.le 1 b
    hb : LT.lt 1 b
    ⊢ Monotone fun x => HPow.hPow b x
  -/
  case inl => exact (strictMono_rpow_of_base_gt_one hb).monotone
  /-
    case inr
    hb : LE.le 1 1
    ⊢ Monotone fun x => HPow.hPow 1 x
  -/
  case inr => intro _ _ _; simp
  /-
    🎉 no goals
  -/


lemma strictAnti_rpow_of_base_lt_one {b : ℝ} (hb₀ : 0 < b) (hb₁ : b < 1) :
    StrictAnti (b ^ · : ℝ → ℝ) := by
  /-
    b : Real
    hb₀ : LT.lt 0 b
    hb₁ : LT.lt b 1
    ⊢ StrictAnti fun x => HPow.hPow b x
  -/
  simp_rw [Real.rpow_def_of_pos hb₀]
  exact exp_strictMono.comp_strictAnti <| StrictMono.const_mul_of_neg strictMono_id
      <| Real.log_neg hb₀ hb₁


lemma antitone_rpow_of_base_le_one {b : ℝ} (hb₀ : 0 < b) (hb₁ : b ≤ 1) :
    Antitone (b ^ · : ℝ → ℝ) := by
  /-
    b : Real
    hb₀ : LT.lt 0 b
    hb₁ : LE.le b 1
    ⊢ Antitone fun x => HPow.hPow b x
  -/
  rcases lt_or_eq_of_le hb₁ with hb₁ | rfl
  /-
    case inl
    b : Real
    hb₀ : LT.lt 0 b
    hb₁✝ : LE.le b 1
    hb₁ : LT.lt b 1
    ⊢ Antitone fun x => HPow.hPow b x
  -/
  case inl => exact (strictAnti_rpow_of_base_lt_one hb₀ hb₁).antitone
  /-
    case inr
    hb₀ : LT.lt 0 1
    hb₁ : LE.le 1 1
    ⊢ Antitone fun x => HPow.hPow 1 x
  -/
  case inr => intro _ _ _; simp
  /-
    🎉 no goals
  -/


/-- Guessing rule for the `bound` tactic: when trying to prove `x ^ y ≤ x ^ z`, we can either assume
`1 ≤ x` or `0 < x ≤ 1`. -/
@[bound] lemma rpow_le_rpow_of_exponent_le_or_ge {x y z : ℝ}
    (h : 1 ≤ x ∧ y ≤ z ∨ 0 < x ∧ x ≤ 1 ∧ z ≤ y) : x ^ y ≤ x ^ z := by
  /-
    x y z : Real
    h : Or (And (LE.le 1 x) (LE.le y z)) (And (LT.lt 0 x) (And (LE.le x 1) (LE.le  …
    ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
  -/
  rcases h with ⟨x1, yz⟩ | ⟨x0, x1, zy⟩
    /-
      case inl.intro
      x y z : Real
      x1 : LE.le 1 x
      yz : LE.le y z
      ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
    -/
  · exact Real.rpow_le_rpow_of_exponent_le x1 yz
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro
      x y z : Real
      x0 : LT.lt 0 x
      x1 : LE.le x 1
      zy : LE.le z y
      ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
    -/
  · exact Real.rpow_le_rpow_of_exponent_ge x0 x1 zy
    /-
      🎉 no goals
    -/


lemma norm_prime_cpow_le_one_half (p : Nat.Primes) {s : ℂ} (hs : 1 < s.re) :
    ‖(p : ℂ) ^ (-s)‖ ≤ 1 / 2 := by
  /-
    p : Nat.Primes
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ LE.le (Norm.norm (HPow.hPow (↑↑p) (Neg.neg s))) (1 / 2)
  -/
  rw [norm_natCast_cpow_of_re_ne_zero p <| by rw [neg_re]; linarith only [hs]]
  refine (Real.rpow_le_rpow_of_nonpos zero_lt_two (Nat.cast_le.mpr p.prop.two_le) <|
    by rw [neg_re]; linarith only [hs]).trans ?_
  /-
    p : Nat.Primes
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ LE.le (HPow.hPow 2 (Neg.neg s).re) (1 / 2)
  -/
  rw [one_div, ← Real.rpow_neg_one]
  /-
    p : Nat.Primes
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ LE.le (HPow.hPow 2 (Neg.neg s).re) (HPow.hPow 2 (-1))
  -/
  exact Real.rpow_le_rpow_of_exponent_le one_le_two <| (neg_lt_neg hs).le
  /-
    🎉 no goals
  -/


lemma one_sub_prime_cpow_ne_zero {p : ℕ} (hp : p.Prime) {s : ℂ} (hs : 1 < s.re) :
    1 - (p : ℂ) ^ (-s) ≠ 0 := by
  /-
    p : Nat
    hp : Nat.Prime p
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Ne (HSub.hSub 1 (HPow.hPow (↑p) (Neg.neg s))) 0
  -/
  refine sub_ne_zero_of_ne fun H ↦ ?_
  /-
    p : Nat
    hp : Nat.Prime p
    s : Complex
    hs : LT.lt 1 s.re
    H : Eq 1 (HPow.hPow (↑p) (Neg.neg s))
    ⊢ False
  -/
  have := norm_prime_cpow_le_one_half ⟨p, hp⟩ hs
  /-
    p : Nat
    hp : Nat.Prime p
    s : Complex
    hs : LT.lt 1 s.re
    H : Eq 1 (HPow.hPow (↑p) (Neg.neg s))
    this : LE.le (Norm.norm (HPow.hPow (↑↑⟨p, hp⟩) (Neg.neg s))) (1 / 2)
    ⊢ False
  -/
  simp only at this
  /-
    p : Nat
    hp : Nat.Prime p
    s : Complex
    hs : LT.lt 1 s.re
    H : Eq 1 (HPow.hPow (↑p) (Neg.neg s))
    this : LE.le (Norm.norm (HPow.hPow (↑p) (Neg.neg s))) (1 / 2)
    ⊢ False
  -/
  rw [← H, norm_one] at this
  /-
    p : Nat
    hp : Nat.Prime p
    s : Complex
    hs : LT.lt 1 s.re
    H : Eq 1 (HPow.hPow (↑p) (Neg.neg s))
    this : LE.le 1 (1 / 2)
    ⊢ False
  -/
  norm_num at this
  /-
    🎉 no goals
  -/


lemma norm_natCast_cpow_le_norm_natCast_cpow_of_pos {n : ℕ} (hn : 0 < n) {w z : ℂ}
    (h : w.re ≤ z.re) :
    ‖(n : ℂ) ^ w‖ ≤ ‖(n : ℂ) ^ z‖ := by
  /-
    n : Nat
    hn : LT.lt 0 n
    w z : Complex
    h : LE.le w.re z.re
    ⊢ LE.le (Norm.norm (HPow.hPow (↑n) w)) (Norm.norm (HPow.hPow (↑n) z))
  -/
  simp_rw [norm_natCast_cpow_of_pos hn]
  /-
    n : Nat
    hn : LT.lt 0 n
    w z : Complex
    h : LE.le w.re z.re
    ⊢ LE.le (HPow.hPow (↑n) w.re) (HPow.hPow (↑n) z.re)
  -/
  exact Real.rpow_le_rpow_of_exponent_le (by exact_mod_cast hn) h
  /-
    🎉 no goals
  -/


lemma norm_natCast_cpow_le_norm_natCast_cpow_iff {n : ℕ} (hn : 1 < n) {w z : ℂ} :
    ‖(n : ℂ) ^ w‖ ≤ ‖(n : ℂ) ^ z‖ ↔ w.re ≤ z.re := by
  simp_rw [norm_natCast_cpow_of_pos (Nat.zero_lt_of_lt hn),
    Real.rpow_le_rpow_left_iff (Nat.one_lt_cast.mpr hn)]


lemma norm_log_natCast_le_rpow_div (n : ℕ) {ε : ℝ} (hε : 0 < ε) : ‖log n‖ ≤ n ^ ε / ε := by
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    ⊢ LE.le (Norm.norm (Complex.log ↑n)) (HDiv.hDiv (HPow.hPow (↑n) ε) ε)
  -/
  rcases n.eq_zero_or_pos with rfl | h
    /-
      case inl
      ε : Real
      hε : LT.lt 0 ε
      ⊢ LE.le (Norm.norm (Complex.log ↑0)) (HDiv.hDiv (HPow.hPow (↑0) ε) ε)
    -/
  · rw [Nat.cast_zero, Nat.cast_zero, log_zero, norm_zero, Real.zero_rpow hε.ne', zero_div]
    /-
      🎉 no goals
    -/
  rw [norm_eq_abs, ← natCast_log, abs_ofReal,
    _root_.abs_of_nonneg <| Real.log_nonneg <| by exact_mod_cast Nat.one_le_of_lt h.lt]
  /-
    case inr
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    h : GT.gt n 0
    ⊢ LE.le (Real.log ↑n) (HDiv.hDiv (HPow.hPow (↑n) ε) ε)
  -/
  exact Real.log_natCast_le_rpow_div n hε
  /-
    🎉 no goals
  -/


theorem sqrt_eq_rpow (x : ℝ) : √x = x ^ (1 / (2 : ℝ)) := by
  /-
    x : Real
    ⊢ Eq x.sqrt (HPow.hPow x (1 / 2))
  -/
  obtain h | h := le_or_lt 0 x
  · rw [← mul_self_inj_of_nonneg (sqrt_nonneg _) (rpow_nonneg h _), mul_self_sqrt h, ← sq,
      ← rpow_natCast, ← rpow_mul h]
    /-
      case inl
      x : Real
      h : LE.le 0 x
      ⊢ Eq x (HPow.hPow x (HMul.hMul (1 / 2) ↑2))
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      h : LT.lt x 0
      ⊢ Eq x.sqrt (HPow.hPow x (1 / 2))
    -/
  · have : 1 / (2 : ℝ) * π = π / (2 : ℝ) := by ring
    /-
      case inr
      x : Real
      h : LT.lt x 0
      this : Eq (HMul.hMul (1 / 2) Real.pi) (HDiv.hDiv Real.pi 2)
      ⊢ Eq x.sqrt (HPow.hPow x (1 / 2))
    -/
    rw [sqrt_eq_zero_of_nonpos h.le, rpow_def_of_neg h, this, cos_pi_div_two, mul_zero]
    /-
      🎉 no goals
    -/


theorem rpow_div_two_eq_sqrt {x : ℝ} (r : ℝ) (hx : 0 ≤ x) : x ^ (r / 2) = √x ^ r := by
  /-
    x r : Real
    hx : LE.le 0 x
    ⊢ Eq (HPow.hPow x (HDiv.hDiv r 2)) (HPow.hPow x.sqrt r)
  -/
  rw [sqrt_eq_rpow, ← rpow_mul hx]
  /-
    x r : Real
    hx : LE.le 0 x
    ⊢ Eq (HPow.hPow x (HDiv.hDiv r 2)) (HPow.hPow x (HMul.hMul (1 / 2) r))
  -/
  congr
  /-
    case e_a
    x r : Real
    hx : LE.le 0 x
    ⊢ Eq (HDiv.hDiv r 2) (HMul.hMul (1 / 2) r)
  -/
  ring
  /-
    🎉 no goals
  -/


lemma cpow_inv_two_re (x : ℂ) : (x ^ (2⁻¹ : ℂ)).re = sqrt ((abs x + x.re) / 2) := by
  rw [← ofReal_ofNat, ← ofReal_inv, cpow_ofReal_re, ← div_eq_mul_inv, ← one_div,
    ← Real.sqrt_eq_rpow, cos_half, ← sqrt_mul, ← mul_div_assoc, mul_add, mul_one, abs_mul_cos_arg]
  /-
    case hx
    x : Complex
    ⊢ LE.le 0 (Complex.abs x)
  -/
  exacts [abs.nonneg _, (neg_pi_lt_arg _).le, arg_le_pi _]
  /-
    🎉 no goals
  -/


lemma cpow_inv_two_im_eq_sqrt {x : ℂ} (hx : 0 ≤ x.im) :
    (x ^ (2⁻¹ : ℂ)).im = sqrt ((abs x - x.re) / 2) := by
  rw [← ofReal_ofNat, ← ofReal_inv, cpow_ofReal_im, ← div_eq_mul_inv, ← one_div,
    ← Real.sqrt_eq_rpow, sin_half_eq_sqrt, ← sqrt_mul (abs.nonneg _), ← mul_div_assoc, mul_sub,
    mul_one, abs_mul_cos_arg]
    /-
      case hl
      x : Complex
      hx : LE.le 0 x.im
      ⊢ LE.le 0 x.arg
    -/
  · rwa [arg_nonneg_iff]
    /-
      🎉 no goals
    -/
    /-
      case hr
      x : Complex
      hx : LE.le 0 x.im
      ⊢ LE.le x.arg (HMul.hMul 2 Real.pi)
    -/
  · linarith [pi_pos, arg_le_pi x]
    /-
      🎉 no goals
    -/


lemma cpow_inv_two_im_eq_neg_sqrt {x : ℂ} (hx : x.im < 0) :
    (x ^ (2⁻¹ : ℂ)).im = -sqrt ((abs x - x.re) / 2) := by
  rw [← ofReal_ofNat, ← ofReal_inv, cpow_ofReal_im, ← div_eq_mul_inv, ← one_div,
    ← Real.sqrt_eq_rpow, sin_half_eq_neg_sqrt, mul_neg, ← sqrt_mul (abs.nonneg _),
    ← mul_div_assoc, mul_sub, mul_one, abs_mul_cos_arg]
    /-
      case hl
      x : Complex
      hx : LT.lt x.im 0
      ⊢ LE.le (Neg.neg (HMul.hMul 2 Real.pi)) x.arg
    -/
  · linarith [pi_pos, neg_pi_lt_arg x]
    /-
      🎉 no goals
    -/
    /-
      case hr
      x : Complex
      hx : LT.lt x.im 0
      ⊢ LE.le x.arg 0
    -/
  · exact (arg_neg_iff.2 hx).le
    /-
      🎉 no goals
    -/


lemma abs_cpow_inv_two_im (x : ℂ) : |(x ^ (2⁻¹ : ℂ)).im| = sqrt ((abs x - x.re) / 2) := by
  rw [← ofReal_ofNat, ← ofReal_inv, cpow_ofReal_im, ← div_eq_mul_inv, ← one_div,
    ← Real.sqrt_eq_rpow, _root_.abs_mul, _root_.abs_of_nonneg (sqrt_nonneg _), abs_sin_half,
    ← sqrt_mul (abs.nonneg _), ← mul_div_assoc, mul_sub, mul_one, abs_mul_cos_arg]


open scoped ComplexOrder in
lemma inv_natCast_cpow_ofReal_pos {n : ℕ} (hn : n ≠ 0) (x : ℝ) :
    0 < ((n : ℂ) ^ (x : ℂ))⁻¹ := by
  /-
    n : Nat
    hn : Ne n 0
    x : Real
    ⊢ LT.lt 0 (Inv.inv (HPow.hPow ↑n ↑x))
  -/
  refine RCLike.inv_pos_of_pos ?_
  /-
    n : Nat
    hn : Ne n 0
    x : Real
    ⊢ LT.lt 0 (HPow.hPow ↑n ↑x)
  -/
  rw [show (n : ℂ) ^ (x : ℂ) = (n : ℝ) ^ (x : ℂ) from rfl, ← ofReal_cpow n.cast_nonneg']
  /-
    n : Nat
    hn : Ne n 0
    x : Real
    ⊢ LT.lt 0 ↑(HPow.hPow (↑n) x)
  -/
  positivity
  /-
    🎉 no goals
  -/


theorem isNat_rpow_pos {a b : ℝ} {nb ne : ℕ}
    (pb : IsNat b nb) (pe' : IsNat (a ^ nb) ne) :
    IsNat (a ^ b) ne := by
  /-
    a b : Real
    nb ne : Nat
    pb : Mathlib.Meta.NormNum.IsNat b nb
    pe' : Mathlib.Meta.NormNum.IsNat (HPow.hPow a nb) ne
    ⊢ Mathlib.Meta.NormNum.IsNat (HPow.hPow a b) ne
  -/
  rwa [pb.out, rpow_natCast]
  /-
    🎉 no goals
  -/


theorem isNat_rpow_neg {a b : ℝ} {nb ne : ℕ}
    (pb : IsInt b (Int.negOfNat nb)) (pe' : IsNat (a ^ (Int.negOfNat nb)) ne) :
    IsNat (a ^ b) ne := by
  /-
    a b : Real
    nb ne : Nat
    pb : Mathlib.Meta.NormNum.IsInt b (Int.negOfNat nb)
    pe' : Mathlib.Meta.NormNum.IsNat (HPow.hPow a (Int.negOfNat nb)) ne
    ⊢ Mathlib.Meta.NormNum.IsNat (HPow.hPow a b) ne
  -/
  rwa [pb.out, Real.rpow_intCast]
  /-
    🎉 no goals
  -/


theorem isInt_rpow_pos {a b : ℝ} {nb ne : ℕ}
    (pb : IsNat b nb) (pe' : IsInt (a ^ nb) (Int.negOfNat ne)) :
    IsInt (a ^ b) (Int.negOfNat ne) := by
  /-
    a b : Real
    nb ne : Nat
    pb : Mathlib.Meta.NormNum.IsNat b nb
    pe' : Mathlib.Meta.NormNum.IsInt (HPow.hPow a nb) (Int.negOfNat ne)
    ⊢ Mathlib.Meta.NormNum.IsInt (HPow.hPow a b) (Int.negOfNat ne)
  -/
  rwa [pb.out, rpow_natCast]
  /-
    🎉 no goals
  -/


theorem isInt_rpow_neg {a b : ℝ} {nb ne : ℕ}
    (pb : IsInt b (Int.negOfNat nb)) (pe' : IsInt (a ^ (Int.negOfNat nb)) (Int.negOfNat ne)) :
    IsInt (a ^ b) (Int.negOfNat ne) := by
  /-
    a b : Real
    nb ne : Nat
    pb : Mathlib.Meta.NormNum.IsInt b (Int.negOfNat nb)
    pe' : Mathlib.Meta.NormNum.IsInt (HPow.hPow a (Int.negOfNat nb)) (Int.negOfNat …
    ⊢ Mathlib.Meta.NormNum.IsInt (HPow.hPow a b) (Int.negOfNat ne)
  -/
  rwa [pb.out, Real.rpow_intCast]
  /-
    🎉 no goals
  -/


theorem isRat_rpow_pos {a b : ℝ} {nb : ℕ}
    {num : ℤ} {den : ℕ}
    (pb : IsNat b nb) (pe' : IsRat (a ^ nb) num den) :
    IsRat (a ^ b) num den := by
  /-
    a b : Real
    nb : Nat
    num : Int
    den : Nat
    pb : Mathlib.Meta.NormNum.IsNat b nb
    pe' : Mathlib.Meta.NormNum.IsRat (HPow.hPow a nb) num den
    ⊢ Mathlib.Meta.NormNum.IsRat (HPow.hPow a b) num den
  -/
  rwa [pb.out, rpow_natCast]
  /-
    🎉 no goals
  -/


theorem isRat_rpow_neg {a b : ℝ} {nb : ℕ}
    {num : ℤ} {den : ℕ}
    (pb : IsInt b (Int.negOfNat nb)) (pe' : IsRat (a ^ (Int.negOfNat nb)) num den) :
    IsRat (a ^ b) num den := by
  /-
    a b : Real
    nb : Nat
    num : Int
    den : Nat
    pb : Mathlib.Meta.NormNum.IsInt b (Int.negOfNat nb)
    pe' : Mathlib.Meta.NormNum.IsRat (HPow.hPow a (Int.negOfNat nb)) num den
    ⊢ Mathlib.Meta.NormNum.IsRat (HPow.hPow a b) num den
  -/
  rwa [pb.out, Real.rpow_intCast]
  /-
    🎉 no goals
  -/


/-- Evaluates expressions of the form `a ^ b` when `a` and `b` are both reals. -/
@[norm_num (_ : ℝ) ^ (_ : ℝ)]
def evalRPow : NormNumExt where eval {u α} e := do
  let .app (.app f (a : Q(ℝ))) (b : Q(ℝ)) ← Lean.Meta.whnfR e | failure
  guard <|← withNewMCtxDepth <| isDefEq f q(HPow.hPow (α := ℝ) (β := ℝ))
  haveI' : u =QL 0 := ⟨⟩
  haveI' : $α =Q ℝ := ⟨⟩
  haveI' h : $e =Q $a ^ $b := ⟨⟩
  h.check
  let (rb : Result b) ← derive (α := q(ℝ)) b
  match rb with
  | .isBool .. | .isRat _ .. => failure
  | .isNat sβ nb pb =>
    match ← derive q($a ^ $nb) with
    | .isBool .. => failure
    | .isNat sα' ne' pe' =>
      assumeInstancesCommute
      haveI' : $sα' =Q AddGroupWithOne.toAddMonoidWithOne := ⟨⟩
      return .isNat sα' ne' q(isNat_rpow_pos $pb $pe')
    | .isNegNat sα' ne' pe' =>
      assumeInstancesCommute
      return .isNegNat sα' ne' q(isInt_rpow_pos $pb $pe')
    | .isRat sα' qe' nume' dene' pe' =>
      assumeInstancesCommute
      return .isRat sα' qe' nume' dene' q(isRat_rpow_pos $pb $pe')
  | .isNegNat sβ nb pb =>
    match ← derive q($a ^ (-($nb : ℤ))) with
    | .isBool .. => failure
    | .isNat sα' ne' pe' =>
      assumeInstancesCommute
      return .isNat sα' ne' q(isNat_rpow_neg $pb $pe')
    | .isNegNat sα' ne' pe' =>
      let _ := q(Real.instRing)
      assumeInstancesCommute
      return .isNegNat sα' ne' q(isInt_rpow_neg $pb $pe')
    | .isRat sα' qe' nume' dene' pe' =>
      assumeInstancesCommute
      return .isRat sα' qe' nume' dene' q(isRat_rpow_neg $pb $pe')


