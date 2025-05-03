lemma continuousOn_one_add_mul_inv {z : ℂ} (hz : 1 + z ∈ slitPlane) :
    ContinuousOn (fun t : ℝ ↦ (1 + t • z)⁻¹) (Set.Icc 0 1) :=
                        /-
                          z : Complex
                          hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 z)
                          ⊢ ContinuousOn (fun t => HAdd.hAdd 1 (HSMul.hSMul t z)) (Set.Icc 0 1)
                        -/
  ContinuousOn.inv₀ (by fun_prop)
                        /-
                          🎉 no goals
                        -/
    (fun _ ht ↦ slitPlane_ne_zero <| StarConvex.add_smul_mem starConvex_one_slitPlane hz ht.1 ht.2)


open intervalIntegral in
/-- Represent `log (1 + z)` as an integral over the unit interval -/
lemma log_eq_integral {z : ℂ} (hz : 1 + z ∈ slitPlane) :
    log (1 + z) = z * ∫ (t : ℝ) in (0 : ℝ)..1, (1 + t • z)⁻¹ := by
  convert (integral_unitInterval_deriv_eq_sub (continuousOn_one_add_mul_inv hz)
    (fun _ ht ↦ hasDerivAt_log <|
      StarConvex.add_smul_mem starConvex_one_slitPlane hz ht.1 ht.2)).symm using 1
  /-
    case h.e'_2
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 z)
    ⊢ Eq (Complex.log (HAdd.hAdd 1 z)) (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (C …
  -/
  simp only [log_one, sub_zero]
  /-
    🎉 no goals
  -/


/-- Represent `log (1 - z)⁻¹` as an integral over the unit interval -/
lemma log_inv_eq_integral {z : ℂ} (hz : 1 - z ∈ slitPlane) :
    log (1 - z)⁻¹ = z * ∫ (t : ℝ) in (0 : ℝ)..1, (1 - t • z)⁻¹ := by
  /-
    z : Complex
    hz : Membership.mem Complex.slitPlane (HSub.hSub 1 z)
    ⊢ Eq (Complex.log (Inv.inv (HSub.hSub 1 z))) (HMul.hMul z (intervalIntegral (f …
  -/
  rw [sub_eq_add_neg 1 z] at hz ⊢
  /-
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 (Neg.neg z))
    ⊢ Eq (Complex.log (Inv.inv (HAdd.hAdd 1 (Neg.neg z)))) (HMul.hMul z (intervalI …
  -/
  rw [log_inv _ <| slitPlane_arg_ne_pi hz, neg_eq_iff_eq_neg, ← neg_mul]
  /-
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 (Neg.neg z))
    ⊢ Eq (Complex.log (HAdd.hAdd 1 (Neg.neg z))) (HMul.hMul (Neg.neg z) (intervalI …
  -/
  convert log_eq_integral hz using 5
  /-
    case h.e'_3.h.e'_6.h.e'_4.h.h.e'_3
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 (Neg.neg z))
    x✝ : Real
    ⊢ Eq (HSub.hSub 1 (HSMul.hSMul x✝ z)) (HAdd.hAdd 1 (HSMul.hSMul x✝ (Neg.neg z)))
  -/
  rw [sub_eq_add_neg, smul_neg]
  /-
    🎉 no goals
  -/


/-- The `n`th Taylor polynomial of `log` at `1`, as a function `ℂ → ℂ` -/
noncomputable
def logTaylor (n : ℕ) : ℂ → ℂ := fun z ↦ ∑ j ∈ Finset.range n, (-1) ^ (j + 1) * z ^ j / j


lemma logTaylor_zero : logTaylor 0 = fun _ ↦ 0 := by
  /-
    ⊢ Eq (Complex.logTaylor 0) fun x => 0
  -/
  funext
  simp only [logTaylor, Finset.range_zero, ← Nat.not_even_iff_odd, Int.cast_pow, Int.cast_neg,
    Int.cast_one, Finset.sum_empty]


lemma logTaylor_succ (n : ℕ) :
    logTaylor (n + 1) = logTaylor n + (fun z : ℂ ↦ (-1) ^ (n + 1) * z ^ n / n) := by
  /-
    n : Nat
    ⊢ Eq (Complex.logTaylor (HAdd.hAdd n 1)) (HAdd.hAdd (Complex.logTaylor n) fun  …
  -/
  funext
  /-
    case h
    n : Nat
    x✝ : Complex
    ⊢ Eq (Complex.logTaylor (HAdd.hAdd n 1) x✝) (HAdd.hAdd (Complex.logTaylor n) ( …
  -/
  simpa only [logTaylor] using Finset.sum_range_succ ..
  /-
    🎉 no goals
  -/


lemma logTaylor_at_zero (n : ℕ) : logTaylor n 0 = 0 := by
  induction n with
  | zero => simp [logTaylor_zero]
  | succ n ih => simpa [logTaylor_succ, ih] using ne_or_eq n 0


lemma hasDerivAt_logTaylor (n : ℕ) (z : ℂ) :
    HasDerivAt (logTaylor (n + 1)) (∑ j ∈ Finset.range n, (-1) ^ j * z ^ j) z := by
  induction n with
  | zero => simp [logTaylor_succ, logTaylor_zero, Pi.add_def, hasDerivAt_const]
  | succ n ih =>
    rw [logTaylor_succ]
    simp only [cpow_natCast, Nat.cast_add, Nat.cast_one, ← Nat.not_even_iff_odd,
      Finset.sum_range_succ, (show (-1) ^ (n + 1 + 1) = (-1) ^ n by ring)]
    refine HasDerivAt.add ih ?_
    simp only [← Nat.not_even_iff_odd, Int.cast_pow, Int.cast_neg, Int.cast_one, mul_div_assoc]
    have : HasDerivAt (fun x : ℂ ↦ (x ^ (n + 1) / (n + 1))) (z ^ n) z := by
      simp_rw [div_eq_mul_inv]
      convert HasDerivAt.mul_const (hasDerivAt_pow (n + 1) z) (((n : ℂ) + 1)⁻¹) using 1
      field_simp [Nat.cast_add_one_ne_zero n]
    convert HasDerivAt.const_mul _ this using 2
    ring


lemma hasDerivAt_log_sub_logTaylor (n : ℕ) {z : ℂ} (hz : 1 + z ∈ slitPlane) :
    HasDerivAt (fun z : ℂ ↦ log (1 + z) - logTaylor (n + 1) z) ((-z) ^ n * (1 + z)⁻¹) z := by
  /-
    n : Nat
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 z)
    ⊢ HasDerivAt (fun z => HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTay …
  -/
  convert ((hasDerivAt_log hz).comp_const_add 1 z).sub (hasDerivAt_logTaylor n z) using 1
  have hz' : -z ≠ 1 := by
    intro H
    rw [neg_eq_iff_eq_neg] at H
    simp only [H, add_neg_cancel] at hz
    exact slitPlane_ne_zero hz rfl
  /-
    case h.e'_9
    n : Nat
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 z)
    hz' : Ne (Neg.neg z) 1
    ⊢ Eq (HMul.hMul (HPow.hPow (Neg.neg z) n) (Inv.inv (HAdd.hAdd 1 z))) (HSub.hSu …
  -/
  simp_rw [← mul_pow, neg_one_mul, geom_sum_eq hz', ← neg_add', div_neg, add_comm z]
  /-
    case h.e'_9
    n : Nat
    z : Complex
    hz : Membership.mem Complex.slitPlane (HAdd.hAdd 1 z)
    hz' : Ne (Neg.neg z) 1
    ⊢ Eq (HMul.hMul (HPow.hPow (Neg.neg z) n) (Inv.inv (HAdd.hAdd 1 z))) (HSub.hSu …
  -/
  field_simp [slitPlane_ne_zero hz]
  /-
    🎉 no goals
  -/


/-- Give a bound on `‖(1 + t * z)⁻¹‖` for `0 ≤ t ≤ 1` and `‖z‖ < 1`. -/
lemma norm_one_add_mul_inv_le {t : ℝ} (ht : t ∈ Set.Icc 0 1) {z : ℂ} (hz : ‖z‖ < 1) :
    ‖(1 + t * z)⁻¹‖ ≤ (1 - ‖z‖)⁻¹ := by
  /-
    t : Real
    ht : Membership.mem (Set.Icc 0 1) t
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (Inv.inv (HAdd.hAdd 1 (HMul.hMul (↑t) z)))) (Inv.inv (HSub. …
  -/
  rw [Set.mem_Icc] at ht
  /-
    t : Real
    ht : And (LE.le 0 t) (LE.le t 1)
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (Inv.inv (HAdd.hAdd 1 (HMul.hMul (↑t) z)))) (Inv.inv (HSub. …
  -/
  rw [norm_inv, norm_eq_abs]
  /-
    t : Real
    ht : And (LE.le 0 t) (LE.le t 1)
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Inv.inv (Complex.abs (HAdd.hAdd 1 (HMul.hMul (↑t) z)))) (Inv.inv (HSu …
  -/
  refine inv_anti₀ (by linarith) ?_
  calc 1 - ‖z‖
    _ ≤ 1 - t * ‖z‖ := by
      nlinarith [norm_nonneg z]
    _ = 1 - ‖t * z‖ := by
      rw [norm_mul, norm_eq_abs (t : ℂ), abs_of_nonneg ht.1]
    _ ≤ ‖1 + t * z‖ := by
      rw [← norm_neg (t * z), ← sub_neg_eq_add]
      convert norm_sub_norm_le 1 (-(t * z))
      exact norm_one.symm


lemma integrable_pow_mul_norm_one_add_mul_inv (n : ℕ) {z : ℂ} (hz : ‖z‖ < 1) :
    IntervalIntegrable (fun t : ℝ ↦ t ^ n * ‖(1 + t * z)⁻¹‖) MeasureTheory.volume 0 1 := by
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Norm.norm (Inv.inv ( …
  -/
  have := continuousOn_one_add_mul_inv <| mem_slitPlane_of_norm_lt_one hz
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : ContinuousOn (fun t => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul t z))) (Set.Ic …
    ⊢ IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Norm.norm (Inv.inv ( …
  -/
  rw [← Set.uIcc_of_le zero_le_one] at this
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    this : ContinuousOn (fun t => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul t z))) (Set.uI …
    ⊢ IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Norm.norm (Inv.inv ( …
  -/
  exact ContinuousOn.intervalIntegrable (by fun_prop)
  /-
    🎉 no goals
  -/


open intervalIntegral in
/-- The difference of `log (1+z)` and its `(n+1)`st Taylor polynomial can be bounded in
terms of `‖z‖`. -/
lemma norm_log_sub_logTaylor_le (n : ℕ) {z : ℂ} (hz : ‖z‖ < 1) :
    ‖log (1 + z) - logTaylor (n + 1) z‖ ≤ ‖z‖ ^ (n + 1) * (1 - ‖z‖)⁻¹ / (n + 1) := by
  have help : IntervalIntegrable (fun t : ℝ ↦ t ^ n * (1 - ‖z‖)⁻¹) MeasureTheory.volume 0 1 :=
    IntervalIntegrable.mul_const (Continuous.intervalIntegrable (by fun_prop) 0 1) (1 - ‖z‖)⁻¹
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    help : IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Inv.inv (HSub.h …
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTaylor …
  -/
  let f (z : ℂ) : ℂ := log (1 + z) - logTaylor (n + 1) z
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    help : IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Inv.inv (HSub.h …
    f : Complex → Complex := fun z => HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Com …
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTaylor …
  -/
  let f' (z : ℂ) : ℂ := (-z) ^ n * (1 + z)⁻¹
  have hderiv : ∀ t ∈ Set.Icc (0 : ℝ) 1, HasDerivAt f (f' (0 + t * z)) (0 + t * z) := by
    intro t ht
    rw [zero_add]
    exact hasDerivAt_log_sub_logTaylor n <|
      StarConvex.add_smul_mem starConvex_one_slitPlane (mem_slitPlane_of_norm_lt_one hz) ht.1 ht.2
  have hcont : ContinuousOn (fun t : ℝ ↦ f' (0 + t * z)) (Set.Icc 0 1) := by
    simp only [zero_add, zero_le_one, not_true_eq_false]
    exact (Continuous.continuousOn (by fun_prop)).mul <|
      continuousOn_one_add_mul_inv <| mem_slitPlane_of_norm_lt_one hz
  have H : f z = z * ∫ t in (0 : ℝ)..1, (-(t * z)) ^ n * (1 + t * z)⁻¹ := by
    convert (integral_unitInterval_deriv_eq_sub hcont hderiv).symm using 1
    · simp only [f, zero_add, add_zero, log_one, logTaylor_at_zero, sub_self, sub_zero]
    · simp only [f', add_zero, log_one, logTaylor_at_zero, sub_self, real_smul, zero_add,
        smul_eq_mul]
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    help : IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Inv.inv (HSub.h …
    f : Complex → Complex := fun z => HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Com …
    f' : Complex → Complex := fun z => HMul.hMul (HPow.hPow (Neg.neg z) n) (Inv.in …
    hderiv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivAt f (f' (HAdd …
    hcont : ContinuousOn (fun t => f' (HAdd.hAdd 0 (HMul.hMul (↑t) z))) (Set.Icc 0 …
    H : Eq (f z) (HMul.hMul z (intervalIntegral (fun t => HMul.hMul (HPow.hPow (Ne …
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTaylor …
  -/
  unfold f at H
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    help : IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Inv.inv (HSub.h …
    f : Complex → Complex := fun z => HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Com …
    f' : Complex → Complex := fun z => HMul.hMul (HPow.hPow (Neg.neg z) n) (Inv.in …
    hderiv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivAt f (f' (HAdd …
    hcont : ContinuousOn (fun t => f' (HAdd.hAdd 0 (HMul.hMul (↑t) z))) (Set.Icc 0 …
    H : Eq (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTaylor (HAdd.hAdd  …
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTaylor …
  -/
  simp only [H, norm_mul]
  simp_rw [neg_pow (_ * z) n, mul_assoc, intervalIntegral.integral_const_mul, mul_pow,
    mul_comm _ (z ^ n), mul_assoc, intervalIntegral.integral_const_mul, norm_mul, norm_pow,
    norm_neg, norm_one, one_pow, one_mul, ← mul_assoc, ← pow_succ', mul_div_assoc]
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    help : IntervalIntegrable (fun t => HMul.hMul (HPow.hPow t n) (Inv.inv (HSub.h …
    f : Complex → Complex := fun z => HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Com …
    f' : Complex → Complex := fun z => HMul.hMul (HPow.hPow (Neg.neg z) n) (Inv.in …
    hderiv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivAt f (f' (HAdd …
    hcont : ContinuousOn (fun t => f' (HAdd.hAdd 0 (HMul.hMul (↑t) z))) (Set.Icc 0 …
    H : Eq (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) (Complex.logTaylor (HAdd.hAdd  …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm z) (HAdd.hAdd n 1)) (Norm.norm (inter …
  -/
  refine mul_le_mul_of_nonneg_left ?_ (pow_nonneg (norm_nonneg z) (n + 1))
  calc ‖∫ t in (0 : ℝ)..1, (t : ℂ) ^ n * (1 + t * z)⁻¹‖
    _ ≤ ∫ t in (0 : ℝ)..1, ‖(t : ℂ) ^ n * (1 + t * z)⁻¹‖ :=
        intervalIntegral.norm_integral_le_integral_norm zero_le_one
    _ = ∫ t in (0 : ℝ)..1, t ^ n * ‖(1 + t * z)⁻¹‖ := by
        refine intervalIntegral.integral_congr <| fun t ht ↦ ?_
        rw [Set.uIcc_of_le zero_le_one, Set.mem_Icc] at ht
        simp_rw [norm_mul, norm_pow, norm_eq_abs, abs_of_nonneg ht.1]
    _ ≤ ∫ t in (0 : ℝ)..1, t ^ n * (1 - ‖z‖)⁻¹ :=
        intervalIntegral.integral_mono_on zero_le_one
          (integrable_pow_mul_norm_one_add_mul_inv n hz) help <|
          fun t ht ↦ mul_le_mul_of_nonneg_left (norm_one_add_mul_inv_le ht hz)
                       (pow_nonneg ((Set.mem_Icc.mp ht).1) _)
    _ = (1 - ‖z‖)⁻¹ / (n + 1) := by
        rw [intervalIntegral.integral_mul_const, mul_comm, integral_pow]
        field_simp


/-- The difference `log (1+z) - z` is bounded by `‖z‖^2/(2*(1-‖z‖))` when `‖z‖ < 1`. -/
lemma norm_log_one_add_sub_self_le {z : ℂ} (hz : ‖z‖ < 1) :
    ‖log (1 + z) - z‖ ≤ ‖z‖ ^ 2 * (1 - ‖z‖)⁻¹ / 2 := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) z)) (HDiv.hDiv (HM …
  -/
  convert norm_log_sub_logTaylor_le 1 hz using 2
    /-
      case h.e'_3.h.e'_3
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Eq (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) z) (HSub.hSub (Complex.log (HAdd …
    -/
  · simp [logTaylor_succ, logTaylor_zero, sub_eq_add_neg]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_6
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Eq 2 (HAdd.hAdd (↑1) 1)
    -/
  · norm_num
    /-
      🎉 no goals
    -/


lemma norm_log_one_add_le {z : ℂ} (hz : ‖z‖ < 1) :
    ‖log (1 + z)‖ ≤ ‖z‖ ^ 2 * (1 - ‖z‖)⁻¹ / 2 + ‖z‖ := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (Complex.log (HAdd.hAdd 1 z))) (HAdd.hAdd (HDiv.hDiv (HMul. …
  -/
  rw [← sub_add_cancel (log (1 + z)) z]
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (HAdd.hAdd (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) z) z))  …
  -/
  apply le_trans (norm_add_le _ _)
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (HAdd.hAdd (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 z)) z)) (No …
  -/
  exact add_le_add_right (Complex.norm_log_one_add_sub_self_le hz) ‖z‖
  /-
    🎉 no goals
  -/


/--For `‖z‖ ≤ 1/2`, the complex logarithm is bounded by `(3/2) * ‖z‖`. -/
lemma norm_log_one_add_half_le_self {z : ℂ} (hz : ‖z‖ ≤ 1/2) : ‖(log (1 + z))‖ ≤ (3/2) * ‖z‖ := by
  /-
    z : Complex
    hz : LE.le (Norm.norm z) (1 / 2)
    ⊢ LE.le (Norm.norm (Complex.log (HAdd.hAdd 1 z))) (HMul.hMul (3 / 2) (Norm.nor …
  -/
  apply le_trans (norm_log_one_add_le (lt_of_le_of_lt hz one_half_lt_one))
  have hz3 : (1 - ‖z‖)⁻¹ ≤ 2 := by
    rw [inv_eq_one_div, div_le_iff₀]
    · linarith
    · linarith
  have hz4 : ‖z‖^2 * (1 - ‖z‖)⁻¹ / 2 ≤ ‖z‖/2 * 2 / 2 := by
    gcongr
    · rw [inv_nonneg]
      linarith
    · rw [sq, div_eq_mul_one_div]
      apply mul_le_mul (by simp only [norm_eq_abs, mul_one, le_refl])
        (by simpa only [norm_eq_abs, one_div] using hz) (norm_nonneg z)
        (by simp only [norm_eq_abs, mul_one, apply_nonneg])
  simp only [isUnit_iff_ne_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    IsUnit.div_mul_cancel] at hz4
  /-
    z : Complex
    hz : LE.le (Norm.norm z) (1 / 2)
    hz3 : LE.le (Inv.inv (HSub.hSub 1 (Norm.norm z))) 2
    hz4 : LE.le (HDiv.hDiv (HMul.hMul (HPow.hPow (Norm.norm z) 2) (Inv.inv (HSub.h …
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv (HMul.hMul (HPow.hPow (Norm.norm z) 2) (Inv.inv  …
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- The difference of `log (1-z)⁻¹` and its `(n+1)`st Taylor polynomial can be bounded in
terms of `‖z‖`. -/
lemma norm_log_one_sub_inv_add_logTaylor_neg_le (n : ℕ) {z : ℂ} (hz : ‖z‖ < 1) :
    ‖log (1 - z)⁻¹ + logTaylor (n + 1) (-z)‖ ≤ ‖z‖ ^ (n + 1) * (1 - ‖z‖)⁻¹ / (n + 1) := by
  rw [sub_eq_add_neg,
    log_inv _ <| slitPlane_arg_ne_pi <| mem_slitPlane_of_norm_lt_one <| (norm_neg z).symm ▸ hz,
    ← sub_neg_eq_add, ← neg_sub', norm_neg]
  /-
    n : Nat
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (HAdd.hAdd 1 (Neg.neg z))) (Complex …
  -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  convert norm_log_sub_logTaylor_le n <| (norm_neg z).symm ▸ hz using 4 <;> rw [norm_neg]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The difference `log (1-z)⁻¹ - z` is bounded by `‖z‖^2/(2*(1-‖z‖))` when `‖z‖ < 1`. -/
lemma norm_log_one_sub_inv_sub_self_le {z : ℂ} (hz : ‖z‖ < 1) :
    ‖log (1 - z)⁻¹ - z‖ ≤ ‖z‖ ^ 2 * (1 - ‖z‖)⁻¹ / 2 := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ LE.le (Norm.norm (HSub.hSub (Complex.log (Inv.inv (HSub.hSub 1 z))) z)) (HDi …
  -/
  convert norm_log_one_sub_inv_add_logTaylor_neg_le 1 hz using 2
    /-
      case h.e'_3.h.e'_3
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Eq (HSub.hSub (Complex.log (Inv.inv (HSub.hSub 1 z))) z) (HAdd.hAdd (Complex …
    -/
  · simp [logTaylor_succ, logTaylor_zero, sub_eq_add_neg]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_6
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Eq 2 (HAdd.hAdd (↑1) 1)
    -/
  · norm_num
    /-
      🎉 no goals
    -/


open Filter Asymptotics in
/-- The Taylor series of the complex logarithm at `1` converges to the logarithm in the
open unit disk. -/
lemma hasSum_taylorSeries_log {z : ℂ} (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ ↦ (-1) ^ (n + 1) * z ^ n / n) (log (1 + z)) := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd n 1)) (HPow …
  -/
  refine (hasSum_iff_tendsto_nat_of_summable_norm ?_).mpr ?_
    /-
      case refine_1
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Summable fun i => Norm.norm (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd …
    -/
  · refine (summable_geometric_of_norm_lt_one hz).norm.of_nonneg_of_le (fun _ ↦ norm_nonneg _) ?_
    /-
      case refine_1
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ ∀ (b : Nat), LE.le (Norm.norm (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hA …
    -/
    intro n
    /-
      case refine_1
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      n : Nat
      ⊢ LE.le (Norm.norm (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd n 1)) (HPo …
    -/
    simp only [norm_div, norm_mul, norm_pow, norm_neg, norm_one, one_pow, one_mul, norm_natCast]
    /-
      case refine_1
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      n : Nat
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (Norm.norm z) n) ↑n) (HPow.hPow (Norm.norm z) n)
    -/
    rcases n.eq_zero_or_pos with rfl | hn
      /-
        case refine_1.inl
        z : Complex
        hz : LT.lt (Norm.norm z) 1
        ⊢ LE.le (HDiv.hDiv (HPow.hPow (Norm.norm z) 0) ↑0) (HPow.hPow (Norm.norm z) 0)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (Norm.norm z) n) ↑n) (HPow.hPow (Norm.norm z) n)
    -/
    conv => enter [2]; rw [← div_one (‖z‖ ^ n)]
    /-
      case refine_1.inr
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le (HDiv.hDiv (HPow.hPow (Norm.norm z) n) ↑n) (HDiv.hDiv (HPow.hPow (Norm …
    -/
    gcongr
    /-
      case refine_1.inr.h
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le 1 ↑n
    -/
    norm_cast
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HDiv.hDiv (HMul.hMul  …
    -/
  · rw [← tendsto_sub_nhds_zero_iff]
    /-
      case refine_2
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Filter.Tendsto (fun x => HSub.hSub ((Finset.range x).sum fun i => HDiv.hDiv  …
    -/
    conv => enter [1, x]; rw [← div_one (_ - _), ← logTaylor]
    /-
      case refine_2
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HSub.hSub (Complex.logTaylor x z) (Compl …
    -/
    rw [← isLittleO_iff_tendsto fun _ h ↦ (one_ne_zero h).elim]
    /-
      case refine_2
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HSub.hSub (Complex.logTaylor x  …
    -/
    refine IsLittleO.trans_isBigO ?_ <| isBigO_const_one ℂ (1 : ℝ) atTop
    have H : (fun n ↦ logTaylor n z - log (1 + z)) =O[atTop] (fun n : ℕ ↦ ‖z‖ ^ n) := by
      have (n : ℕ) : ‖logTaylor n z - log (1 + z)‖
          ≤ (max ‖log (1 + z)‖ (1 - ‖z‖)⁻¹) * ‖(‖z‖ ^ n)‖ := by
        rw [norm_sub_rev, norm_pow, norm_norm]
        cases n with
        | zero => simp [logTaylor_zero]
        | succ n =>
            refine (norm_log_sub_logTaylor_le n hz).trans ?_
            rw [mul_comm, ← div_one ((max _ _) * _)]
            gcongr
            · exact le_max_right ..
            · linarith
      exact (isBigOWith_of_le' atTop this).isBigO
    /-
      case refine_2
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      H : Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (Complex.logTaylor n z …
      ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HSub.hSub (Complex.logTaylor x  …
    -/
    refine IsBigO.trans_isLittleO H ?_
    /-
      case refine_2
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      H : Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (Complex.logTaylor n z …
      ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HPow.hPow (Norm.norm z) n) fun  …
    -/
    convert isLittleO_pow_pow_of_lt_left (norm_nonneg z) hz
    /-
      case h.e'_8.h
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      H : Asymptotics.IsBigO Filter.atTop (fun n => HSub.hSub (Complex.logTaylor n z …
      x✝ : Nat
      ⊢ Eq 1 (HPow.hPow 1 x✝)
    -/
    exact (one_pow _).symm
    /-
      🎉 no goals
    -/


/-- The series `∑ z^n/n` converges to `-log (1-z)` on the open unit disk. -/
lemma hasSum_taylorSeries_neg_log {z : ℂ} (hz : ‖z‖ < 1) :
    HasSum (fun n : ℕ ↦ z ^ n / n) (-log (1 - z)) := by
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow z n) ↑n) (Neg.neg (Complex.log (HSub.h …
  -/
  conv => enter [1, n]; rw [← neg_neg (z ^ n / n)]
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ HasSum (fun n => Neg.neg (Neg.neg (HDiv.hDiv (HPow.hPow z n) ↑n))) (Neg.neg  …
  -/
  refine HasSum.neg ?_
  /-
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    ⊢ HasSum (fun n => Neg.neg (HDiv.hDiv (HPow.hPow z n) ↑n)) (Complex.log (HSub. …
  -/
  convert hasSum_taylorSeries_log (z := -z) (norm_neg z ▸ hz) using 2 with n
  /-
    case h.e'_5.h
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    n : Nat
    ⊢ Eq (Neg.neg (HDiv.hDiv (HPow.hPow z n) ↑n)) (HDiv.hDiv (HMul.hMul (HPow.hPow …
  -/
  rcases n.eq_zero_or_pos with rfl | hn
    /-
      case h.e'_5.h.inl
      z : Complex
      hz : LT.lt (Norm.norm z) 1
      ⊢ Eq (Neg.neg (HDiv.hDiv (HPow.hPow z 0) ↑0)) (HDiv.hDiv (HMul.hMul (HPow.hPow …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case h.e'_5.h.inr
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    n : Nat
    hn : GT.gt n 0
    ⊢ Eq (Neg.neg (HDiv.hDiv (HPow.hPow z n) ↑n)) (HDiv.hDiv (HMul.hMul (HPow.hPow …
  -/
  field_simp
  /-
    case h.e'_5.h.inr
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    n : Nat
    hn : GT.gt n 0
    ⊢ Eq (HDiv.hDiv (Neg.neg (HPow.hPow z n)) ↑n) (HDiv.hDiv (HMul.hMul (HPow.hPow …
  -/
  rw [div_eq_div_iff, pow_succ', mul_assoc (-1), ← mul_pow, neg_mul_neg, neg_one_mul, one_mul]
  /-
    case h.e'_5.h.inr.hb
    z : Complex
    hz : LT.lt (Norm.norm z) 1
    n : Nat
    hn : GT.gt n 0
    ⊢ Ne (↑n) 0
  -/
  all_goals {norm_cast; exact hn.ne'}
  /-
    🎉 no goals
  -/


