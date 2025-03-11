theorem antideriv_cos_comp_const_mul (hz : z ≠ 0) (x : ℝ) :
    HasDerivAt (fun y : ℝ => Complex.sin (2 * z * y) / (2 * z)) (Complex.cos (2 * z * x)) x := by
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑y))  …
  -/
  have a : HasDerivAt (fun y : ℂ => y * (2 * z)) _ x := hasDerivAt_mul_const _
  have b : HasDerivAt (Complex.sin ∘ fun y : ℂ => (y * (2 * z))) _ x :=
    HasDerivAt.comp (x : ℂ) (Complex.hasDerivAt_sin (x * (2 * z))) a
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    a : HasDerivAt (fun y => HMul.hMul y (HMul.hMul 2 z)) (HMul.hMul 2 z) ↑x
    b : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y (HMul.hMul 2 z) …
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑y))  …
  -/
  have c := b.comp_ofReal.div_const (2 * z)
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    a : HasDerivAt (fun y => HMul.hMul y (HMul.hMul 2 z)) (HMul.hMul 2 z) ↑x
    b : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y (HMul.hMul 2 z) …
    c : HasDerivAt (fun x => HDiv.hDiv (Function.comp Complex.sin (fun y => HMul.h …
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑y))  …
  -/
  field_simp at c; simp only [fun y => mul_comm y (2 * z)] at c
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    a : HasDerivAt (fun y => HMul.hMul y (HMul.hMul 2 z)) (HMul.hMul 2 z) ↑x
    b : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y (HMul.hMul 2 z) …
    c : HasDerivAt (fun x => HDiv.hDiv (Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑x) …
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑y))  …
  -/
  exact c
  /-
    🎉 no goals
  -/


theorem antideriv_sin_comp_const_mul (hz : z ≠ 0) (x : ℝ) :
    HasDerivAt (fun y : ℝ => -Complex.cos (2 * z * y) / (2 * z)) (Complex.sin (2 * z * x)) x := by
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Neg.neg (Complex.cos (HMul.hMul (HMul.hMul 2 …
  -/
  have a : HasDerivAt (fun y : ℂ => y * (2 * z)) _ x := hasDerivAt_mul_const _
  have b : HasDerivAt (Complex.cos ∘ fun y : ℂ => (y * (2 * z))) _ x :=
    HasDerivAt.comp (x : ℂ) (Complex.hasDerivAt_cos (x * (2 * z))) a
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    a : HasDerivAt (fun y => HMul.hMul y (HMul.hMul 2 z)) (HMul.hMul 2 z) ↑x
    b : HasDerivAt (Function.comp Complex.cos fun y => HMul.hMul y (HMul.hMul 2 z) …
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Neg.neg (Complex.cos (HMul.hMul (HMul.hMul 2 …
  -/
  have c := (b.comp_ofReal.div_const (2 * z)).neg
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    a : HasDerivAt (fun y => HMul.hMul y (HMul.hMul 2 z)) (HMul.hMul 2 z) ↑x
    b : HasDerivAt (Function.comp Complex.cos fun y => HMul.hMul y (HMul.hMul 2 z) …
    c : HasDerivAt (fun x => Neg.neg (HDiv.hDiv (Function.comp Complex.cos (fun y  …
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Neg.neg (Complex.cos (HMul.hMul (HMul.hMul 2 …
  -/
  field_simp at c; simp only [fun y => mul_comm y (2 * z)] at c
  /-
    z : Complex
    hz : Ne z 0
    x : Real
    a : HasDerivAt (fun y => HMul.hMul y (HMul.hMul 2 z)) (HMul.hMul 2 z) ↑x
    b : HasDerivAt (Function.comp Complex.cos fun y => HMul.hMul y (HMul.hMul 2 z) …
    c : HasDerivAt (fun x => HDiv.hDiv (Neg.neg (Complex.cos (HMul.hMul (HMul.hMul …
    ⊢ HasDerivAt (fun y => HDiv.hDiv (Neg.neg (Complex.cos (HMul.hMul (HMul.hMul 2 …
  -/
  exact c
  /-
    🎉 no goals
  -/


theorem integral_cos_mul_cos_pow_aux (hn : 2 ≤ n) (hz : z ≠ 0) :
    (∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ n) =
      n / (2 * z) *
        ∫ x in (0 : ℝ)..π / 2, Complex.sin (2 * z * x) * sin x * (cos x : ℂ) ^ (n - 1) := by
  have der1 :
    ∀ x : ℝ,
      x ∈ uIcc 0 (π / 2) →
        HasDerivAt (fun y : ℝ => (cos y : ℂ) ^ n) (-n * sin x * (cos x : ℂ) ^ (n - 1)) x := by
    intro x _
    have b : HasDerivAt (fun y : ℝ => (cos y : ℂ)) (-sin x) x := by
      simpa using (hasDerivAt_cos x).ofReal_comp
    convert HasDerivAt.comp x (hasDerivAt_pow _ _) b using 1
    ring
  convert (config := { sameFun := true })
    integral_mul_deriv_eq_deriv_mul der1 (fun x _ => antideriv_cos_comp_const_mul hz x) _ _ using 2
    /-
      case h.e'_2.h.e'_4
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Eq (fun x => HMul.hMul (Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑x)) (HPow.hP …
    -/
  · ext1 x; rw [mul_comm]
            /-
              🎉 no goals
            -/
  · rw [Complex.ofReal_zero, mul_zero, Complex.sin_zero, zero_div, mul_zero, sub_zero,
      cos_pi_div_two, Complex.ofReal_zero, zero_pow (by positivity : n ≠ 0), zero_mul, zero_sub,
      ← integral_neg, ← integral_const_mul]
    /-
      case h.e'_3
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HDiv.hDiv (↑n) (HMul.hMul 2 z)) (H …
    -/
    refine integral_congr fun x _ => ?_
    /-
      case h.e'_3
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      ⊢ Eq (HMul.hMul (HDiv.hDiv (↑n) (HMul.hMul 2 z)) (HMul.hMul (HMul.hMul (Comple …
    -/
    field_simp; ring
                /-
                  🎉 no goals
                -/
    /-
      case convert_1
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ IntervalIntegrable (fun x => HMul.hMul (HMul.hMul (Neg.neg ↑n) ↑(Real.sin x) …
    -/
  · apply Continuous.intervalIntegrable
    exact
      (continuous_const.mul (Complex.continuous_ofReal.comp continuous_sin)).mul
        ((Complex.continuous_ofReal.comp continuous_cos).pow (n - 1))
    /-
      case convert_2
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ IntervalIntegrable (fun x => Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑x)) Mea …
    -/
  · apply Continuous.intervalIntegrable
    /-
      case convert_2.hu
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Continuous fun x => Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑x)
    -/
    exact Complex.continuous_cos.comp (continuous_const.mul Complex.continuous_ofReal)
    /-
      🎉 no goals
    -/


theorem integral_sin_mul_sin_mul_cos_pow_eq (hn : 2 ≤ n) (hz : z ≠ 0) :
    (∫ x in (0 : ℝ)..π / 2, Complex.sin (2 * z * x) * sin x * (cos x : ℂ) ^ (n - 1)) =
      (n / (2 * z) * ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ n) -
        (n - 1) / (2 * z) *
          ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (n - 2) := by
  have der1 :
    ∀ x : ℝ,
      x ∈ uIcc 0 (π / 2) →
        HasDerivAt (fun y : ℝ => sin y * (cos y : ℂ) ^ (n - 1))
          ((cos x : ℂ) ^ n - (n - 1) * (sin x : ℂ) ^ 2 * (cos x : ℂ) ^ (n - 2)) x := by
    intro x _
    have c := HasDerivAt.comp (x : ℂ) (hasDerivAt_pow (n - 1) _) (Complex.hasDerivAt_cos x)
    convert ((Complex.hasDerivAt_sin x).mul c).comp_ofReal using 1
    · ext1 y; simp only [Complex.ofReal_sin, Complex.ofReal_cos, Function.comp]
    · simp only [Complex.ofReal_cos, Complex.ofReal_sin]
      rw [mul_neg, mul_neg, ← sub_eq_add_neg, Function.comp_apply]
      congr 1
      · rw [← pow_succ', Nat.sub_add_cancel (by omega : 1 ≤ n)]
      · have : ((n - 1 : ℕ) : ℂ) = (n : ℂ) - 1 := by
          rw [Nat.cast_sub (one_le_two.trans hn), Nat.cast_one]
        rw [Nat.sub_sub, this]
        ring
  convert
    integral_mul_deriv_eq_deriv_mul der1 (fun x _ => antideriv_sin_comp_const_mul hz x) _ _ using 1
    /-
      case h.e'_2
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HMul.hMul (Complex.sin (HMul.hMul  …
    -/
  · refine integral_congr fun x _ => ?_
    /-
      case h.e'_2
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      ⊢ Eq (HMul.hMul (HMul.hMul (Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑x)) ↑(Real …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
  · -- now a tedious rearrangement of terms
    -- gather into a single integral, and deal with continuity subgoals:
    rw [sin_zero, cos_pi_div_two, Complex.ofReal_zero, zero_pow, zero_mul,
      mul_zero, zero_mul, zero_mul, sub_zero, zero_sub, ←
      integral_neg, ← integral_const_mul, ← integral_const_mul, ← integral_sub]
    /-
      case h.e'_3
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Eq (intervalIntegral (fun x => HSub.hSub (HMul.hMul (HDiv.hDiv (↑n) (HMul.hM …
    -/
    rotate_left
      /-
        case h.e'_3.hf
        z : Complex
        n : Nat
        hn : LE.le 2 n
        hz : Ne z 0
        der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
        ⊢ IntervalIntegrable (fun x => HMul.hMul (HDiv.hDiv (↑n) (HMul.hMul 2 z)) (HMu …
      -/
    · apply Continuous.intervalIntegrable
      exact
        continuous_const.mul
          ((Complex.continuous_cos.comp (continuous_const.mul Complex.continuous_ofReal)).mul
            ((Complex.continuous_ofReal.comp continuous_cos).pow n))
      /-
        case h.e'_3.hg
        z : Complex
        n : Nat
        hn : LE.le 2 n
        hz : Ne z 0
        der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
        ⊢ IntervalIntegrable (fun x => HMul.hMul (HDiv.hDiv (HSub.hSub (↑n) 1) (HMul.h …
      -/
    · apply Continuous.intervalIntegrable
      exact
        continuous_const.mul
          ((Complex.continuous_cos.comp (continuous_const.mul Complex.continuous_ofReal)).mul
            ((Complex.continuous_ofReal.comp continuous_cos).pow (n - 2)))
      /-
        case h.e'_3
        z : Complex
        n : Nat
        hn : LE.le 2 n
        hz : Ne z 0
        der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
        ⊢ Ne (HSub.hSub n 1) 0
      -/
    · exact Nat.sub_ne_zero_of_lt hn
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Eq (intervalIntegral (fun x => HSub.hSub (HMul.hMul (HDiv.hDiv (↑n) (HMul.hM …
    -/
    refine integral_congr fun x _ => ?_
    /-
      case h.e'_3
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      ⊢ Eq (HSub.hSub (HMul.hMul (HDiv.hDiv (↑n) (HMul.hMul 2 z)) (HMul.hMul (Comple …
    -/
    dsimp only
    -- get rid of real trig functions and divisions by 2 * z:
    rw [Complex.ofReal_cos, Complex.ofReal_sin, Complex.sin_sq, ← mul_div_right_comm, ←
      mul_div_right_comm, ← sub_div, mul_div, ← neg_div]
    /-
      case h.e'_3
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HMul.hMul (↑n) (HMul.hMul (Complex.cos (HMul.hMul  …
    -/
    congr 1
    have : Complex.cos x ^ n = Complex.cos x ^ (n - 2) * Complex.cos x ^ 2 := by
      conv_lhs => rw [← Nat.sub_add_cancel hn, pow_add]
    /-
      case h.e'_3.e_a
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      this : Eq (HPow.hPow (Complex.cos ↑x) n) (HMul.hMul (HPow.hPow (Complex.cos ↑x …
      ⊢ Eq (HSub.hSub (HMul.hMul (↑n) (HMul.hMul (Complex.cos (HMul.hMul (HMul.hMul  …
    -/
    rw [this]
    /-
      case h.e'_3.e_a
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      this : Eq (HPow.hPow (Complex.cos ↑x) n) (HMul.hMul (HPow.hPow (Complex.cos ↑x …
      ⊢ Eq (HSub.hSub (HMul.hMul (↑n) (HMul.hMul (Complex.cos (HMul.hMul (HMul.hMul  …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case convert_1
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ IntervalIntegrable (fun x => HSub.hSub (HPow.hPow (↑(Real.cos x)) n) (HMul.h …
    -/
  · apply Continuous.intervalIntegrable
    exact
      ((Complex.continuous_ofReal.comp continuous_cos).pow n).sub
        ((continuous_const.mul ((Complex.continuous_ofReal.comp continuous_sin).pow 2)).mul
          ((Complex.continuous_ofReal.comp continuous_cos).pow (n - 2)))
    /-
      case convert_2
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ IntervalIntegrable (fun x => Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑x)) Mea …
    -/
  · apply Continuous.intervalIntegrable
    /-
      case convert_2.hu
      z : Complex
      n : Nat
      hn : LE.le 2 n
      hz : Ne z 0
      der1 : ∀ (x : Real), Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x → Has …
      ⊢ Continuous fun x => Complex.sin (HMul.hMul (HMul.hMul 2 z) ↑x)
    -/
    exact Complex.continuous_sin.comp (continuous_const.mul Complex.continuous_ofReal)
    /-
      🎉 no goals
    -/


/-- Note this also holds for `z = 0`, but we do not need this case for `sin_pi_mul_eq`. -/
theorem integral_cos_mul_cos_pow (hn : 2 ≤ n) (hz : z ≠ 0) :
    (((1 : ℂ) - (4 : ℂ) * z ^ 2 / (n : ℂ) ^ 2) *
      ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ n) =
      (n - 1 : ℂ) / n *
        ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (n - 2) := by
  have nne : (n : ℂ) ≠ 0 := by
    contrapose! hn; rw [Nat.cast_eq_zero] at hn; rw [hn]; exact zero_lt_two
  /-
    z : Complex
    n : Nat
    hn : LE.le 2 n
    hz : Ne z 0
    nne : Ne (↑n) 0
    ⊢ Eq (HMul.hMul (HSub.hSub 1 (HDiv.hDiv (HMul.hMul 4 (HPow.hPow z 2)) (HPow.hP …
  -/
  have := integral_cos_mul_cos_pow_aux hn hz
  rw [integral_sin_mul_sin_mul_cos_pow_eq hn hz, sub_eq_neg_add, mul_add, ← sub_eq_iff_eq_add]
    at this
  /-
    z : Complex
    n : Nat
    hn : LE.le 2 n
    hz : Ne z 0
    nne : Ne (↑n) 0
    this : Eq (HSub.hSub (intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul. …
    ⊢ Eq (HMul.hMul (HSub.hSub 1 (HDiv.hDiv (HMul.hMul 4 (HPow.hPow z 2)) (HPow.hP …
  -/
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
  convert congr_arg (fun u : ℂ => -u * (2 * z) ^ 2 / n ^ 2) this using 1 <;> field_simp <;> ring
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


/-- Note this also holds for `z = 0`, but we do not need this case for `sin_pi_mul_eq`. -/
theorem integral_cos_mul_cos_pow_even (n : ℕ) (hz : z ≠ 0) :
    (((1 : ℂ) - z ^ 2 / ((n : ℂ) + 1) ^ 2) *
        ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n + 2)) =
      (2 * n + 1 : ℂ) / (2 * n + 2) *
        ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n) := by
  /-
    z : Complex
    n : Nat
    hz : Ne z 0
    ⊢ Eq (HMul.hMul (HSub.hSub 1 (HDiv.hDiv (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd  …
  -/
  convert integral_cos_mul_cos_pow (by omega : 2 ≤ 2 * n + 2) hz using 3
    /-
      case h.e'_2.h.e'_5.h.e'_6
      z : Complex
      n : Nat
      hz : Ne z 0
      ⊢ Eq (HDiv.hDiv (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (HDiv.hDiv ( …
    -/
  · simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_two]
    /-
      case h.e'_2.h.e'_5.h.e'_6
      z : Complex
      n : Nat
      hz : Ne z 0
      ⊢ Eq (HDiv.hDiv (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (HDiv.hDiv ( …
    -/
    nth_rw 2 [← mul_one (2 : ℂ)]
    /-
      case h.e'_2.h.e'_5.h.e'_6
      z : Complex
      n : Nat
      hz : Ne z 0
      ⊢ Eq (HDiv.hDiv (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (HDiv.hDiv ( …
    -/
    rw [← mul_add, mul_pow, ← div_div]
    /-
      case h.e'_2.h.e'_5.h.e'_6
      z : Complex
      n : Nat
      hz : Ne z 0
      ⊢ Eq (HDiv.hDiv (HPow.hPow z 2) (HPow.hPow (HAdd.hAdd (↑n) 1) 2)) (HDiv.hDiv ( …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_5.h.e'_5
      z : Complex
      n : Nat
      hz : Ne z 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul 2 ↑n) 1) (HSub.hSub (↑(HAdd.hAdd (HMul.hMul 2 n) 2) …
    -/
  · push_cast; ring
               /-
                 🎉 no goals
               -/
    /-
      case h.e'_3.h.e'_5.h.e'_6
      z : Complex
      n : Nat
      hz : Ne z 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul 2 ↑n) 2) ↑(HAdd.hAdd (HMul.hMul 2 n) 2)
    -/
  · push_cast; ring
               /-
                 🎉 no goals
               -/


/-- Relate the integral `cos x ^ n` over `[0, π/2]` to the integral of `sin x ^ n` over `[0, π]`,
which is studied in `Data.Real.Pi.Wallis` and other places. -/
theorem integral_cos_pow_eq (n : ℕ) :
    (∫ x in (0 : ℝ)..π / 2, cos x ^ n) = 1 / 2 * ∫ x in (0 : ℝ)..π, sin x ^ n := by
  rw [mul_comm (1 / 2 : ℝ), ← div_eq_iff (one_div_ne_zero (two_ne_zero' ℝ)), ← div_mul, div_one,
    mul_two]
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv. …
  -/
  have L : IntervalIntegrable _ volume 0 (π / 2) := (continuous_sin.pow n).intervalIntegrable _ _
  /-
    n : Nat
    L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
    ⊢ Eq (HAdd.hAdd (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv. …
  -/
  have R : IntervalIntegrable _ volume (π / 2) π := (continuous_sin.pow n).intervalIntegrable _ _
  /-
    n : Nat
    L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
    R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
    ⊢ Eq (HAdd.hAdd (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv. …
  -/
  rw [← integral_add_adjacent_intervals L R]
  /-
    n : Nat
    L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
    R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
    ⊢ Eq (HAdd.hAdd (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv. …
  -/
  congr 1
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv.hDiv Real.p …
    -/
  · nth_rw 1 [(by ring : 0 = π / 2 - π / 2)]
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) (HSub.hSub (HDiv.hD …
    -/
    nth_rw 3 [(by ring : π / 2 = π / 2 - 0)]
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) (HSub.hSub (HDiv.hD …
    -/
    rw [← integral_comp_sub_left]
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos (HSub.hSub (HDiv.hDiv Rea …
    -/
    refine integral_congr fun x _ => ?_
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      ⊢ Eq (HPow.hPow (Real.cos (HSub.hSub (HDiv.hDiv Real.pi 2) x)) n) (HPow.hPow ( …
    -/
    rw [cos_pi_div_two_sub]
    /-
      🎉 no goals
    -/
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv.hDiv Real.p …
    -/
  · nth_rw 3 [(by ring : π = π / 2 + π / 2)]
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv.hDiv Real.p …
    -/
    nth_rw 2 [(by ring : π / 2 = 0 + π / 2)]
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv.hDiv Real.p …
    -/
    rw [← integral_comp_add_right]
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) n) 0 (HDiv.hDiv Real.p …
    -/
    refine integral_congr fun x _ => ?_
    /-
      case e_a
      n : Nat
      L : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      R : IntervalIntegrable (fun b => HPow.hPow (Real.sin b) n) MeasureTheory.Measu …
      x : Real
      x✝ : Membership.mem (Set.uIcc 0 (HDiv.hDiv Real.pi 2)) x
      ⊢ Eq (HPow.hPow (Real.cos x) n) (HPow.hPow (Real.sin (HAdd.hAdd x (HDiv.hDiv R …
    -/
    rw [sin_add_pi_div_two]
    /-
      🎉 no goals
    -/


theorem integral_cos_pow_pos (n : ℕ) : 0 < ∫ x in (0 : ℝ)..π / 2, cos x ^ n :=
  (integral_cos_pow_eq n).symm ▸ mul_pos one_half_pos (integral_sin_pow_pos _)


/-- Finite form of Euler's sine product, with remainder term expressed as a ratio of cosine
integrals. -/
theorem sin_pi_mul_eq (z : ℂ) (n : ℕ) :
    Complex.sin (π * z) =
      ((π * z * ∏ j ∈ Finset.range n, ((1 : ℂ) - z ^ 2 / ((j : ℂ) + 1) ^ 2)) *
          ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n)) /
        (∫ x in (0 : ℝ)..π / 2, cos x ^ (2 * n) : ℝ) := by
  /-
    z : Complex
    n : Nat
    ⊢ Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMul ( …
  -/
  rcases eq_or_ne z 0 with (rfl | hz)
    /-
      case inl
      n : Nat
      ⊢ Eq (Complex.sin (HMul.hMul (↑Real.pi) 0)) (HDiv.hDiv (HMul.hMul (HMul.hMul ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    z : Complex
    n : Nat
    hz : Ne z 0
    ⊢ Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMul ( …
  -/
  induction' n with n hn
  · simp_rw [mul_zero, pow_zero, mul_one, Finset.prod_range_zero, mul_one,
      integral_one, sub_zero]
    rw [integral_cos_mul_complex (mul_ne_zero two_ne_zero hz), Complex.ofReal_zero,
      mul_zero, Complex.sin_zero, zero_div, sub_zero,
      (by push_cast; field_simp; ring : 2 * z * ↑(π / 2) = π * z)]
    /-
      case inr.zero
      z : Complex
      hz : Ne z 0
      ⊢ Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMul ( …
    -/
    field_simp [Complex.ofReal_ne_zero.mpr pi_pos.ne']
    /-
      case inr.zero
      z : Complex
      hz : Ne z 0
      ⊢ Eq (HMul.hMul (Complex.sin (HMul.hMul (↑Real.pi) z)) (HMul.hMul (HMul.hMul 2 …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      ⊢ Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMul ( …
    -/
  · rw [hn, Finset.prod_range_succ]
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) ((Finset.range  …
    -/
    set A := ∏ j ∈ Finset.range n, ((1 : ℂ) - z ^ 2 / ((j : ℂ) + 1) ^ 2)
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) A) (intervalInt …
    -/
    set B := ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n)
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      B : Complex := intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul.hMul (H …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) A) B) ↑(interva …
    -/
    set C := ∫ x in (0 : ℝ)..π / 2, cos x ^ (2 * n)
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      B : Complex := intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul.hMul (H …
      C : Real := intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n)) …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) A) B) ↑C) (HDiv …
    -/
    have aux' : 2 * n.succ = 2 * n + 2 := by rw [Nat.succ_eq_add_one, mul_add, mul_one]
    have : (∫ x in (0 : ℝ)..π / 2, cos x ^ (2 * n.succ)) = (2 * (n : ℝ) + 1) / (2 * n + 2) * C := by
      rw [integral_cos_pow_eq]
      dsimp only [C]
      rw [integral_cos_pow_eq, aux', integral_sin_pow, sin_zero, sin_pi, pow_succ',
        zero_mul, zero_mul, zero_mul, sub_zero, zero_div,
        zero_add, ← mul_assoc, ← mul_assoc, mul_comm (1 / 2 : ℝ) _, Nat.cast_mul, Nat.cast_ofNat]
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      B : Complex := intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul.hMul (H …
      C : Real := intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n)) …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      aux' : Eq (HMul.hMul 2 n.succ) (HAdd.hAdd (HMul.hMul 2 n) 2)
      this : Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n.su …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) A) B) ↑C) (HDiv …
    -/
    rw [this]
    change
      π * z * A * B / C =
        (π * z * (A * ((1 : ℂ) - z ^ 2 / ((n : ℂ) + 1) ^ 2)) *
            ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n.succ)) /
          ((2 * n + 1) / (2 * n + 2) * C : ℝ)
    have :
      (π * z * (A * ((1 : ℂ) - z ^ 2 / ((n : ℂ) + 1) ^ 2)) *
          ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n.succ)) =
        π * z * A *
          (((1 : ℂ) - z ^ 2 / (n.succ : ℂ) ^ 2) *
            ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n.succ)) := by
      nth_rw 2 [Nat.succ_eq_add_one]
      rw [Nat.cast_add_one]
      ring
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      B : Complex := intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul.hMul (H …
      C : Real := intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n)) …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      aux' : Eq (HMul.hMul 2 n.succ) (HAdd.hAdd (HMul.hMul 2 n) 2)
      this✝ : Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n.s …
      this : Eq (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) (HMul.hMul A (HSub.hS …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) A) B) ↑C) (HDiv …
    -/
    rw [this]
    suffices
      (((1 : ℂ) - z ^ 2 / (n.succ : ℂ) ^ 2) *
          ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n.succ)) =
        (2 * n + 1) / (2 * n + 2) * B by
      rw [this, Complex.ofReal_mul, Complex.ofReal_div]
      have : (C : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr (integral_cos_pow_pos _).ne'
      have : 2 * (n : ℂ) + 1 ≠ 0 := by
        convert (Nat.cast_add_one_ne_zero (2 * n) : (↑(2 * n) + 1 : ℂ) ≠ 0)
        simp
      have : 2 * (n : ℂ) + 2 ≠ 0 := by
        convert (Nat.cast_add_one_ne_zero (2 * n + 1) : (↑(2 * n + 1) + 1 : ℂ) ≠ 0) using 1
        push_cast; ring
      field_simp; ring
    /-
      case inr.succ
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      B : Complex := intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul.hMul (H …
      C : Real := intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n)) …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      aux' : Eq (HMul.hMul 2 n.succ) (HAdd.hAdd (HMul.hMul 2 n) 2)
      this✝ : Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n.s …
      this : Eq (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) (HMul.hMul A (HSub.hS …
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (HDiv.hDiv (HPow.hPow z 2) (HPow.hPow (↑n.succ) 2 …
    -/
    convert integral_cos_mul_cos_pow_even n hz
    /-
      case h.e'_2.h.e'_5.h.e'_6.h.e'_6.h.e'_5
      z : Complex
      hz : Ne z 0
      n : Nat
      A : Complex := (Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPo …
      B : Complex := intervalIntegral (fun x => HMul.hMul (Complex.cos (HMul.hMul (H …
      C : Real := intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n)) …
      hn : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) (HDiv.hDiv (HMul.hMul (HMul.hMu …
      aux' : Eq (HMul.hMul 2 n.succ) (HAdd.hAdd (HMul.hMul 2 n) 2)
      this✝ : Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HMul.hMul 2 n.s …
      this : Eq (HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) (HMul.hMul A (HSub.hS …
      ⊢ Eq (↑n.succ) (HAdd.hAdd (↑n) 1)
    -/
    rw [Nat.cast_succ]
    /-
      🎉 no goals
    -/


theorem tendsto_integral_cos_pow_mul_div {f : ℝ → ℂ} (hf : ContinuousOn f (Icc 0 (π / 2))) :
    Tendsto
      (fun n : ℕ => (∫ x in (0 : ℝ)..π / 2, (cos x : ℂ) ^ n * f x) /
        (∫ x in (0 : ℝ)..π / 2, cos x ^ n : ℝ))
      atTop (𝓝 <| f 0) := by
  simp_rw [div_eq_inv_mul (α := ℂ), ← Complex.ofReal_inv, integral_of_le pi_div_two_pos.le,
    ← MeasureTheory.integral_Icc_eq_integral_Ioc, ← Complex.ofReal_pow, ← Complex.real_smul]
  have c_lt : ∀ y : ℝ, y ∈ Icc 0 (π / 2) → y ≠ 0 → cos y < cos 0 := fun y hy hy' =>
    cos_lt_cos_of_nonneg_of_le_pi_div_two (le_refl 0) hy.2 (lt_of_le_of_ne hy.1 hy'.symm)
  have c_nonneg : ∀ x : ℝ, x ∈ Icc 0 (π / 2) → 0 ≤ cos x := fun x hx =>
    cos_nonneg_of_mem_Icc ((Icc_subset_Icc_left (neg_nonpos_of_nonneg pi_div_two_pos.le)) hx)
  /-
    f : Real → Complex
    hf : ContinuousOn f (Set.Icc 0 (HDiv.hDiv Real.pi 2))
    c_lt : ∀ (y : Real), Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) y → Ne y …
    c_nonneg : ∀ (x : Real), Membership.mem (Set.Icc 0 (HDiv.hDiv Real.pi 2)) x →  …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv (MeasureTheory.integral (Measu …
  -/
  have c_zero_pos : 0 < cos 0 := by rw [cos_zero]; exact zero_lt_one
  have zero_mem : (0 : ℝ) ∈ closure (interior (Icc 0 (π / 2))) := by
    rw [interior_Icc, closure_Ioo pi_div_two_pos.ne, left_mem_Icc]
    exact pi_div_two_pos.le
  exact
    tendsto_setIntegral_pow_smul_of_unique_maximum_of_isCompact_of_continuousOn isCompact_Icc
      continuousOn_cos c_lt c_nonneg c_zero_pos zero_mem hf


/-- Euler's infinite product formula for the complex sine function. -/
theorem _root_.Complex.tendsto_euler_sin_prod (z : ℂ) :
    Tendsto (fun n : ℕ => π * z * ∏ j ∈ Finset.range n, ((1 : ℂ) - z ^ 2 / ((j : ℂ) + 1) ^ 2))
      atTop (𝓝 <| Complex.sin (π * z)) := by
  have A :
    Tendsto
      (fun n : ℕ =>
        ((π * z * ∏ j ∈ Finset.range n, ((1 : ℂ) - z ^ 2 / ((j : ℂ) + 1) ^ 2)) *
            ∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ (2 * n)) /
          (∫ x in (0 : ℝ)..π / 2, cos x ^ (2 * n) : ℝ))
      atTop (𝓝 <| _) :=
    Tendsto.congr (fun n => sin_pi_mul_eq z n) tendsto_const_nhds
  /-
    z : Complex
    A : Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (↑Real.pi) z) ((Finset.range n …
  -/
  have : 𝓝 (Complex.sin (π * z)) = 𝓝 (Complex.sin (π * z) * 1) := by rw [mul_one]
  /-
    z : Complex
    A : Filter.Tendsto (fun n => HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (↑Real …
    this : Eq (nhds (Complex.sin (HMul.hMul (↑Real.pi) z))) (nhds (HMul.hMul (Comp …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (↑Real.pi) z) ((Finset.range n …
  -/
  simp_rw [this, mul_div_assoc] at A
  /-
    z : Complex
    this : Eq (nhds (Complex.sin (HMul.hMul (↑Real.pi) z))) (nhds (HMul.hMul (Comp …
    A : Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) ((F …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (↑Real.pi) z) ((Finset.range n …
  -/
  convert (tendsto_mul_iff_of_ne_zero _ one_ne_zero).mp A
  suffices Tendsto (fun n : ℕ =>
        (∫ x in (0 : ℝ)..π / 2, Complex.cos (2 * z * x) * (cos x : ℂ) ^ n) /
          (∫ x in (0 : ℝ)..π / 2, cos x ^ n : ℝ)) atTop (𝓝 1) from
    this.comp (tendsto_id.const_mul_atTop' zero_lt_two)
  have : ContinuousOn (fun x : ℝ => Complex.cos (2 * z * x)) (Icc 0 (π / 2)) :=
    (Complex.continuous_cos.comp (continuous_const.mul Complex.continuous_ofReal)).continuousOn
  /-
    z : Complex
    this✝ : Eq (nhds (Complex.sin (HMul.hMul (↑Real.pi) z))) (nhds (HMul.hMul (Com …
    A : Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) ((F …
    this : ContinuousOn (fun x => Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑x)) (Set …
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (intervalIntegral (fun x => HMul.hMul (Co …
  -/
  convert tendsto_integral_cos_pow_mul_div this using 1
    /-
      case h.e'_3
      z : Complex
      this✝ : Eq (nhds (Complex.sin (HMul.hMul (↑Real.pi) z))) (nhds (HMul.hMul (Com …
      A : Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) ((F …
      this : ContinuousOn (fun x => Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑x)) (Set …
      ⊢ Eq (fun n => HDiv.hDiv (intervalIntegral (fun x => HMul.hMul (Complex.cos (H …
    -/
  · ext1 n; congr 2 with x : 1; rw [mul_comm]
                                /-
                                  🎉 no goals
                                -/
    /-
      case h.e'_5
      z : Complex
      this✝ : Eq (nhds (Complex.sin (HMul.hMul (↑Real.pi) z))) (nhds (HMul.hMul (Com …
      A : Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (HMul.hMul (↑Real.pi) z) ((F …
      this : ContinuousOn (fun x => Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑x)) (Set …
      ⊢ Eq (nhds 1) (nhds (Complex.cos (HMul.hMul (HMul.hMul 2 z) ↑0)))
    -/
  · rw [Complex.ofReal_zero, mul_zero, Complex.cos_zero]
    /-
      🎉 no goals
    -/


/-- Euler's infinite product formula for the real sine function. -/
theorem _root_.Real.tendsto_euler_sin_prod (x : ℝ) :
    Tendsto (fun n : ℕ => π * x * ∏ j ∈ Finset.range n, ((1 : ℝ) - x ^ 2 / ((j : ℝ) + 1) ^ 2))
      atTop (𝓝 <| sin (π * x)) := by
  /-
    x : Real
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HMul.hMul Real.pi x) ((Finset.range n).p …
  -/
  convert (Complex.continuous_re.tendsto _).comp (Complex.tendsto_euler_sin_prod x) using 1
    /-
      case h.e'_3
      x : Real
      ⊢ Eq (fun n => HMul.hMul (HMul.hMul Real.pi x) ((Finset.range n).prod fun j => …
    -/
  · ext1 n
    /-
      case h.e'_3.h
      x : Real
      n : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul Real.pi x) ((Finset.range n).prod fun j => HSub.hSu …
    -/
    rw [Function.comp_apply, ← Complex.ofReal_mul, Complex.re_ofReal_mul]
    suffices
      (∏ j ∈ Finset.range n, (1 - x ^ 2 / (j + 1) ^ 2) : ℂ) =
        (∏ j ∈ Finset.range n, (1 - x ^ 2 / (j + 1) ^ 2) : ℝ) by
      rw [this, Complex.ofReal_re]
    /-
      case h.e'_3.h
      x : Real
      n : Nat
      ⊢ Eq ((Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPow (↑x) 2) …
    -/
    rw [Complex.ofReal_prod]
    /-
      case h.e'_3.h
      x : Real
      n : Nat
      ⊢ Eq ((Finset.range n).prod fun j => HSub.hSub 1 (HDiv.hDiv (HPow.hPow (↑x) 2) …
    -/
    refine Finset.prod_congr (by rfl) fun n _ => ?_
    /-
      case h.e'_3.h
      x : Real
      n✝ n : Nat
      x✝ : Membership.mem (Finset.range n✝) n
      ⊢ Eq (HSub.hSub 1 (HDiv.hDiv (HPow.hPow (↑x) 2) (HPow.hPow (HAdd.hAdd (↑n) 1)  …
    -/
    norm_cast
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5
      x : Real
      ⊢ Eq (nhds (Real.sin (HMul.hMul Real.pi x))) (nhds (Complex.sin (HMul.hMul ↑Re …
    -/
  · rw [← Complex.ofReal_mul, ← Complex.ofReal_sin, Complex.ofReal_re]
    /-
      🎉 no goals
    -/


