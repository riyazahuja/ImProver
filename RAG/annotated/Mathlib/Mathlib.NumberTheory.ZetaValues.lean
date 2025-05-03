local notation "𝕌" => UnitAddCircle


/-- The function `x ↦ Bₖ(x) : ℝ → ℝ`. -/
def bernoulliFun (k : ℕ) (x : ℝ) : ℝ :=
  (Polynomial.map (algebraMap ℚ ℝ) (Polynomial.bernoulli k)).eval x


theorem bernoulliFun_eval_zero (k : ℕ) : bernoulliFun k 0 = bernoulli k := by
  /-
    k : Nat
    ⊢ Eq (bernoulliFun k 0) ↑(bernoulli k)
  -/
  rw [bernoulliFun, Polynomial.eval_zero_map, Polynomial.bernoulli_eval_zero, eq_ratCast]
  /-
    🎉 no goals
  -/


theorem bernoulliFun_endpoints_eq_of_ne_one {k : ℕ} (hk : k ≠ 1) :
    bernoulliFun k 1 = bernoulliFun k 0 := by
  rw [bernoulliFun_eval_zero, bernoulliFun, Polynomial.eval_one_map, Polynomial.bernoulli_eval_one,
    bernoulli_eq_bernoulli'_of_ne_one hk, eq_ratCast]


theorem bernoulliFun_eval_one (k : ℕ) : bernoulliFun k 1 = bernoulliFun k 0 + ite (k = 1) 1 0 := by
  /-
    k : Nat
    ⊢ Eq (bernoulliFun k 1) (HAdd.hAdd (bernoulliFun k 0) (ite (Eq k 1) 1 0))
  -/
  rw [bernoulliFun, bernoulliFun_eval_zero, Polynomial.eval_one_map, Polynomial.bernoulli_eval_one]
  /-
    k : Nat
    ⊢ Eq ((algebraMap Rat Real) (bernoulli' k)) (HAdd.hAdd (↑(bernoulli k)) (ite ( …
  -/
  split_ifs with h
    /-
      case pos
      k : Nat
      h : Eq k 1
      ⊢ Eq ((algebraMap Rat Real) (bernoulli' k)) (HAdd.hAdd (↑(bernoulli k)) 1)
    -/
  · rw [h, bernoulli_one, bernoulli'_one, eq_ratCast]
    /-
      case pos
      k : Nat
      h : Eq k 1
      ⊢ Eq (↑(1 / 2)) (HAdd.hAdd (↑(-1 / 2)) 1)
    -/
    push_cast; ring
               /-
                 🎉 no goals
               -/
    /-
      case neg
      k : Nat
      h : Not (Eq k 1)
      ⊢ Eq ((algebraMap Rat Real) (bernoulli' k)) (HAdd.hAdd (↑(bernoulli k)) 0)
    -/
  · rw [bernoulli_eq_bernoulli'_of_ne_one h, add_zero, eq_ratCast]
    /-
      🎉 no goals
    -/


theorem hasDerivAt_bernoulliFun (k : ℕ) (x : ℝ) :
    HasDerivAt (bernoulliFun k) (k * bernoulliFun (k - 1) x) x := by
  /-
    k : Nat
    x : Real
    ⊢ HasDerivAt (bernoulliFun k) (HMul.hMul (↑k) (bernoulliFun (HSub.hSub k 1) x) …
  -/
  convert ((Polynomial.bernoulli k).map <| algebraMap ℚ ℝ).hasDerivAt x using 1
  simp only [bernoulliFun, Polynomial.derivative_map, Polynomial.derivative_bernoulli k,
    Polynomial.map_mul, Polynomial.map_natCast, Polynomial.eval_mul, Polynomial.eval_natCast]


theorem antideriv_bernoulliFun (k : ℕ) (x : ℝ) :
    HasDerivAt (fun x => bernoulliFun (k + 1) x / (k + 1)) (bernoulliFun k x) x := by
  /-
    k : Nat
    x : Real
    ⊢ HasDerivAt (fun x => HDiv.hDiv (bernoulliFun (HAdd.hAdd k 1) x) (HAdd.hAdd ( …
  -/
  convert (hasDerivAt_bernoulliFun (k + 1) x).div_const _ using 1
  /-
    case h.e'_9
    k : Nat
    x : Real
    ⊢ Eq (bernoulliFun k x) (HDiv.hDiv (HMul.hMul (↑(HAdd.hAdd k 1)) (bernoulliFun …
  -/
  field_simp [Nat.cast_add_one_ne_zero k]
  /-
    🎉 no goals
  -/


theorem integral_bernoulliFun_eq_zero {k : ℕ} (hk : k ≠ 0) :
    ∫ x : ℝ in (0)..1, bernoulliFun k x = 0 := by
  rw [integral_eq_sub_of_hasDerivAt (fun x _ => antideriv_bernoulliFun k x)
      ((Polynomial.continuous _).intervalIntegrable _ _)]
  /-
    k : Nat
    hk : Ne k 0
    ⊢ Eq (HSub.hSub (HDiv.hDiv (bernoulliFun (HAdd.hAdd k 1) 1) (HAdd.hAdd (↑k) 1) …
  -/
  rw [bernoulliFun_eval_one]
  /-
    k : Nat
    hk : Ne k 0
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd (bernoulliFun (HAdd.hAdd k 1) 0) (ite (E …
  -/
  split_ifs with h
    /-
      case pos
      k : Nat
      hk : Ne k 0
      h : Eq (HAdd.hAdd k 1) 1
      ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd (bernoulliFun (HAdd.hAdd k 1) 0) 1) (HAd …
    -/
  · exfalso; exact hk (Nat.succ_inj'.mp h)
             /-
               🎉 no goals
             -/
    /-
      case neg
      k : Nat
      hk : Ne k 0
      h : Not (Eq (HAdd.hAdd k 1) 1)
      ⊢ Eq (HSub.hSub (HDiv.hDiv (HAdd.hAdd (bernoulliFun (HAdd.hAdd k 1) 0) 0) (HAd …
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- The `n`-th Fourier coefficient of the `k`-th Bernoulli function on the interval `[0, 1]`. -/
def bernoulliFourierCoeff (k : ℕ) (n : ℤ) : ℂ :=
  fourierCoeffOn zero_lt_one (fun x => bernoulliFun k x) n


/-- Recurrence relation (in `k`) for the `n`-th Fourier coefficient of `Bₖ`. -/
theorem bernoulliFourierCoeff_recurrence (k : ℕ) {n : ℤ} (hn : n ≠ 0) :
    bernoulliFourierCoeff k n =
      1 / (-2 * π * I * n) * (ite (k = 1) 1 0 - k * bernoulliFourierCoeff (k - 1) n) := by
  /-
    k : Nat
    n : Int
    hn : Ne n 0
    ⊢ Eq (bernoulliFourierCoeff k n) (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul …
  -/
  unfold bernoulliFourierCoeff
  rw [fourierCoeffOn_of_hasDerivAt zero_lt_one hn
      (fun x _ => (hasDerivAt_bernoulliFun k x).ofReal_comp)
      ((continuous_ofReal.comp <|
            continuous_const.mul <| Polynomial.continuous _).intervalIntegrable
        _ _)]
  /-
    k : Nat
    n : Int
    hn : Ne n 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (HMul.hMul (-2) ↑Real.pi) C …
  -/
  simp_rw [ofReal_one, ofReal_zero, sub_zero, one_mul]
  rw [QuotientAddGroup.mk_zero, fourier_eval_zero, one_mul, ← ofReal_sub, bernoulliFun_eval_one,
    add_sub_cancel_left]
  /-
    k : Nat
    n : Int
    hn : Ne n 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (HMul.hMul (-2) ↑Real.pi) C …
  -/
  congr 2
    /-
      case e_a.e_a
      k : Nat
      n : Int
      hn : Ne n 0
      ⊢ Eq (↑(ite (Eq k 1) 1 0)) (ite (Eq k 1) 1 0)
    -/
                  /-
                    🎉 no goals
                  -/
  · split_ifs <;> simp only [ofReal_one, ofReal_zero, one_mul]
                  /-
                    🎉 no goals
                  -/
    /-
      case e_a.e_a
      k : Nat
      n : Int
      hn : Ne n 0
      ⊢ Eq (fourierCoeffOn ⋯ (fun x => ↑(HMul.hMul (↑k) (bernoulliFun (HSub.hSub k 1 …
    -/
  · simp_rw [ofReal_mul, ofReal_natCast, fourierCoeffOn.const_mul]
    /-
      🎉 no goals
    -/


/-- The Fourier coefficients of `B₀(x) = 1`. -/
theorem bernoulli_zero_fourier_coeff {n : ℤ} (hn : n ≠ 0) : bernoulliFourierCoeff 0 n = 0 := by
  /-
    n : Int
    hn : Ne n 0
    ⊢ Eq (bernoulliFourierCoeff 0 n) 0
  -/
  simpa using bernoulliFourierCoeff_recurrence 0 hn
  /-
    🎉 no goals
  -/


/-- The `0`-th Fourier coefficient of `Bₖ(x)`. -/
theorem bernoulliFourierCoeff_zero {k : ℕ} (hk : k ≠ 0) : bernoulliFourierCoeff k 0 = 0 := by
  simp_rw [bernoulliFourierCoeff, fourierCoeffOn_eq_integral, neg_zero, fourier_zero, sub_zero,
    div_one, one_smul, intervalIntegral.integral_ofReal, integral_bernoulliFun_eq_zero hk,
    ofReal_zero]


theorem bernoulliFourierCoeff_eq {k : ℕ} (hk : k ≠ 0) (n : ℤ) :
    bernoulliFourierCoeff k n = -k ! / (2 * π * I * n) ^ k := by
  /-
    k : Nat
    hk : Ne k 0
    n : Int
    ⊢ Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow  …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
  · rw [bernoulliFourierCoeff_zero hk, Int.cast_zero, mul_zero, zero_pow hk,
      div_zero]
  /-
    case inr
    k : Nat
    hk : Ne k 0
    n : Int
    hn : Ne n 0
    ⊢ Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow  …
  -/
  refine Nat.le_induction ?_ (fun k hk h'k => ?_) k (Nat.one_le_iff_ne_zero.mpr hk)
    /-
      case inr.refine_1
      k : Nat
      hk : Ne k 0
      n : Int
      hn : Ne n 0
      ⊢ Eq (bernoulliFourierCoeff 1 n) (HDiv.hDiv (Neg.neg ↑(Nat.factorial 1)) (HPow …
    -/
  · rw [bernoulliFourierCoeff_recurrence 1 hn]
    simp only [Nat.cast_one, tsub_self, neg_mul, one_mul, eq_self_iff_true, if_true,
      Nat.factorial_one, pow_one, inv_I, mul_neg]
    /-
      case inr.refine_1
      k : Nat
      hk : Ne k 0
      n : Int
      hn : Ne n 0
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (Neg.neg (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real …
    -/
    rw [bernoulli_zero_fourier_coeff hn, sub_zero, mul_one, div_neg, neg_div]
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      k✝ : Nat
      hk✝ : Ne k✝ 0
      n : Int
      hn : Ne n 0
      k : Nat
      hk : LE.le 1 k
      h'k : Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
      ⊢ Eq (bernoulliFourierCoeff (HAdd.hAdd k 1) n) (HDiv.hDiv (Neg.neg ↑(HAdd.hAdd …
    -/
  · rw [bernoulliFourierCoeff_recurrence (k + 1) hn, Nat.add_sub_cancel k 1]
    /-
      case inr.refine_2
      k✝ : Nat
      hk✝ : Ne k✝ 0
      n : Int
      hn : Ne n 0
      k : Nat
      hk : LE.le 1 k
      h'k : Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (HMul.hMul (-2) ↑Real.pi) C …
    -/
    split_ifs with h
      /-
        case pos
        k✝ : Nat
        hk✝ : Ne k✝ 0
        n : Int
        hn : Ne n 0
        k : Nat
        hk : LE.le 1 k
        h'k : Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
        h : Eq (HAdd.hAdd k 1) 1
        ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (HMul.hMul (-2) ↑Real.pi) C …
      -/
    · exfalso; exact (ne_of_gt (Nat.lt_succ_iff.mpr hk)) h
               /-
                 🎉 no goals
               -/
    · rw [h'k, Nat.factorial_succ, zero_sub, Nat.cast_mul, pow_add, pow_one, neg_div, mul_neg,
        mul_neg, mul_neg, neg_neg, neg_mul, neg_mul, neg_mul, div_neg]
      /-
        case neg
        k✝ : Nat
        hk✝ : Ne k✝ 0
        n : Int
        hn : Ne n 0
        k : Nat
        hk : LE.le 1 k
        h'k : Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
        h : Not (Eq (HAdd.hAdd k 1) 1)
        ⊢ Eq (HMul.hMul (Neg.neg (HDiv.hDiv 1 (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real …
      -/
      field_simp [Int.cast_ne_zero.mpr hn, I_ne_zero]
      /-
        case neg
        k✝ : Nat
        hk✝ : Ne k✝ 0
        n : Int
        hn : Ne n 0
        k : Nat
        hk : LE.le 1 k
        h'k : Eq (bernoulliFourierCoeff k n) (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
        h : Not (Eq (HAdd.hAdd k 1) 1)
        ⊢ Eq (HDiv.hDiv (Neg.neg (HMul.hMul (HAdd.hAdd (↑k) 1) ↑k.factorial)) (HMul.hM …
      -/
      ring_nf
      /-
        🎉 no goals
      -/


/-- The Bernoulli polynomial, extended from `[0, 1)` to the unit circle. -/
def periodizedBernoulli (k : ℕ) : 𝕌 → ℝ :=
  AddCircle.liftIco 1 0 (bernoulliFun k)


theorem periodizedBernoulli.continuous {k : ℕ} (hk : k ≠ 1) : Continuous (periodizedBernoulli k) :=
  AddCircle.liftIco_zero_continuous
    (mod_cast (bernoulliFun_endpoints_eq_of_ne_one hk).symm)
    (Polynomial.continuous _).continuousOn


theorem fourierCoeff_bernoulli_eq {k : ℕ} (hk : k ≠ 0) (n : ℤ) :
    fourierCoeff ((↑) ∘ periodizedBernoulli k : 𝕌 → ℂ) n = -k ! / (2 * π * I * n) ^ k := by
  have : ((↑) ∘ periodizedBernoulli k : 𝕌 → ℂ) = AddCircle.liftIco 1 0 ((↑) ∘ bernoulliFun k) := by
    ext1 x; rfl
  /-
    k : Nat
    hk : Ne k 0
    n : Int
    this : Eq (Function.comp Complex.ofReal (periodizedBernoulli k)) (AddCircle.li …
    ⊢ Eq (fourierCoeff (Function.comp Complex.ofReal (periodizedBernoulli k)) n) ( …
  -/
  rw [this, fourierCoeff_liftIco_eq]
  /-
    k : Nat
    hk : Ne k 0
    n : Int
    this : Eq (Function.comp Complex.ofReal (periodizedBernoulli k)) (AddCircle.li …
    ⊢ Eq (fourierCoeffOn ⋯ (Function.comp Complex.ofReal (bernoulliFun k)) n) (HDi …
  -/
  simpa only [zero_add] using bernoulliFourierCoeff_eq hk n
  /-
    🎉 no goals
  -/


theorem summable_bernoulli_fourier {k : ℕ} (hk : 2 ≤ k) :
    Summable (fun n => -k ! / (2 * π * I * n) ^ k : ℤ → ℂ) := by
  have :
      ∀ n : ℤ, -(k ! : ℂ) / (2 * π * I * n) ^ k = -k ! / (2 * π * I) ^ k * (1 / (n : ℂ) ^ k) := by
    intro n; rw [mul_one_div, div_div, ← mul_pow]
  /-
    k : Nat
    hk : LE.le 2 k
    this : ∀ (n : Int), Eq (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow (HMul.hMul …
    ⊢ Summable fun n => HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow (HMul.hMul (HM …
  -/
  simp_rw [this]
  /-
    k : Nat
    hk : LE.le 2 k
    this : ∀ (n : Int), Eq (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow (HMul.hMul …
    ⊢ Summable fun n => HMul.hMul (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow (HM …
  -/
  refine Summable.mul_left _ <| .of_norm ?_
  have : (fun x : ℤ => ‖1 / (x : ℂ) ^ k‖) = fun x : ℤ => |1 / (x : ℝ) ^ k| := by
    ext1 x
    rw [norm_eq_abs, ← Complex.abs_ofReal]
    congr 1
    norm_cast
  /-
    k : Nat
    hk : LE.le 2 k
    this✝ : ∀ (n : Int), Eq (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow (HMul.hMu …
    this : Eq (fun x => Norm.norm (HDiv.hDiv 1 (HPow.hPow (↑x) k))) fun x => abs ( …
    ⊢ Summable fun a => Norm.norm (HDiv.hDiv 1 (HPow.hPow (↑a) k))
  -/
  simp_rw [this]
  /-
    k : Nat
    hk : LE.le 2 k
    this✝ : ∀ (n : Int), Eq (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.hPow (HMul.hMu …
    this : Eq (fun x => Norm.norm (HDiv.hDiv 1 (HPow.hPow (↑x) k))) fun x => abs ( …
    ⊢ Summable fun x => abs (HDiv.hDiv 1 (HPow.hPow (↑x) k))
  -/
  rwa [summable_abs_iff, Real.summable_one_div_int_pow]
  /-
    🎉 no goals
  -/


theorem hasSum_one_div_pow_mul_fourier_mul_bernoulliFun {k : ℕ} (hk : 2 ≤ k) {x : ℝ}
    (hx : x ∈ Icc (0 : ℝ) 1) :
    HasSum (fun n : ℤ => 1 / (n : ℂ) ^ k * fourier n (x : 𝕌))
      (-(2 * π * I) ^ k / k ! * bernoulliFun k x) := by
  -- first show it suffices to prove result for `Ico 0 1`
  suffices ∀ {y : ℝ}, y ∈ Ico (0 : ℝ) 1 →
      HasSum (fun (n : ℤ) ↦ 1 / (n : ℂ) ^ k * fourier n y)
        (-(2 * (π : ℂ) * I) ^ k / k ! * bernoulliFun k y) by
    rw [← Ico_insert_right (zero_le_one' ℝ), mem_insert_iff, or_comm] at hx
    rcases hx with (hx | rfl)
    · exact this hx
    · convert this (left_mem_Ico.mpr zero_lt_one) using 1
      · rw [AddCircle.coe_period, QuotientAddGroup.mk_zero]
      · rw [bernoulliFun_endpoints_eq_of_ne_one (by omega : k ≠ 1)]
  /-
    k : Nat
    hk : LE.le 2 k
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ ∀ {y : Real}, Membership.mem (Set.Ico 0 1) y → HasSum (fun n => HMul.hMul (H …
  -/
  intro y hy
  let B : C(𝕌, ℂ) :=
    ContinuousMap.mk ((↑) ∘ periodizedBernoulli k)
      (continuous_ofReal.comp (periodizedBernoulli.continuous (by omega)))
  have step1 : ∀ n : ℤ, fourierCoeff B n = -k ! / (2 * π * I * n) ^ k := by
    rw [ContinuousMap.coe_mk]; exact fourierCoeff_bernoulli_eq (by omega : k ≠ 0)
  have step2 :=
    has_pointwise_sum_fourier_series_of_summable
      ((summable_bernoulli_fourier hk).congr fun n => (step1 n).symm) y
  /-
    k : Nat
    hk : LE.le 2 k
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    y : Real
    hy : Membership.mem (Set.Ico 0 1) y
    B : ContinuousMap UnitAddCircle Complex := { toFun := Function.comp Complex.of …
    step1 : ∀ (n : Int), Eq (fourierCoeff (⇑B) n) (HDiv.hDiv (Neg.neg ↑k.factorial …
    step2 : HasSum (fun i => HSMul.hSMul (fourierCoeff (⇑B) i) ((fourier i) ↑y)) ( …
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) ((fourier n) ↑y) …
  -/
  simp_rw [step1] at step2
  /-
    k : Nat
    hk : LE.le 2 k
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    y : Real
    hy : Membership.mem (Set.Ico 0 1) y
    B : ContinuousMap UnitAddCircle Complex := { toFun := Function.comp Complex.of …
    step1 : ∀ (n : Int), Eq (fourierCoeff (⇑B) n) (HDiv.hDiv (Neg.neg ↑k.factorial …
    step2 : HasSum (fun i => HSMul.hSMul (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) ((fourier n) ↑y) …
  -/
  convert step2.mul_left (-(2 * ↑π * I) ^ k / (k ! : ℂ)) using 2 with n
  · rw [smul_eq_mul, ← mul_assoc, mul_div, mul_neg, div_mul_cancel₀, neg_neg, mul_pow _ (n : ℂ),
      ← div_div, div_self]
      /-
        case h.e'_5.h
        k : Nat
        hk : LE.le 2 k
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        y : Real
        hy : Membership.mem (Set.Ico 0 1) y
        B : ContinuousMap UnitAddCircle Complex := { toFun := Function.comp Complex.of …
        step1 : ∀ (n : Int), Eq (fourierCoeff (⇑B) n) (HDiv.hDiv (Neg.neg ↑k.factorial …
        step2 : HasSum (fun i => HSMul.hSMul (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
        n : Int
        ⊢ Ne (HPow.hPow (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) k) 0
      -/
    · rw [Ne, pow_eq_zero_iff', not_and_or]
      /-
        case h.e'_5.h
        k : Nat
        hk : LE.le 2 k
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        y : Real
        hy : Membership.mem (Set.Ico 0 1) y
        B : ContinuousMap UnitAddCircle Complex := { toFun := Function.comp Complex.of …
        step1 : ∀ (n : Int), Eq (fourierCoeff (⇑B) n) (HDiv.hDiv (Neg.neg ↑k.factorial …
        step2 : HasSum (fun i => HSMul.hSMul (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
        n : Int
        ⊢ Or (Not (Eq (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) 0)) (Not (Ne k 0))
      -/
      exact Or.inl two_pi_I_ne_zero
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.h
        k : Nat
        hk : LE.le 2 k
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        y : Real
        hy : Membership.mem (Set.Ico 0 1) y
        B : ContinuousMap UnitAddCircle Complex := { toFun := Function.comp Complex.of …
        step1 : ∀ (n : Int), Eq (fourierCoeff (⇑B) n) (HDiv.hDiv (Neg.neg ↑k.factorial …
        step2 : HasSum (fun i => HSMul.hSMul (HDiv.hDiv (Neg.neg ↑k.factorial) (HPow.h …
        n : Int
        ⊢ Ne (↑k.factorial) 0
      -/
    · exact Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero _)
      /-
        🎉 no goals
      -/
  · rw [ContinuousMap.coe_mk, Function.comp_apply, ofReal_inj, periodizedBernoulli,
      AddCircle.liftIco_coe_apply (show y ∈ Ico 0 (0 + 1) by rwa [zero_add])]


theorem hasSum_one_div_nat_pow_mul_fourier {k : ℕ} (hk : 2 ≤ k) {x : ℝ} (hx : x ∈ Icc (0 : ℝ) 1) :
    HasSum
      (fun n : ℕ =>
        (1 : ℂ) / (n : ℂ) ^ k * (fourier n (x : 𝕌) + (-1 : ℂ) ^ k * fourier (-n) (x : 𝕌)))
      (-(2 * π * I) ^ k / k ! * bernoulliFun k x) := by
  /-
    k : Nat
    hk : LE.le 2 k
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) (HAdd.hAdd ((fou …
  -/
  convert (hasSum_one_div_pow_mul_fourier_mul_bernoulliFun hk hx).nat_add_neg using 1
    /-
      case h.e'_5
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      ⊢ Eq (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) (HAdd.hAdd ((fourier …
    -/
  · ext1 n
    /-
      case h.e'_5.h
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) (HAdd.hAdd ((fourier ↑n) ↑x)  …
    -/
    rw [Int.cast_neg, mul_add, ← mul_assoc]
    /-
      case h.e'_5.h
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) ((fourier ↑n) ↑x)) …
    -/
    conv_rhs => rw [neg_eq_neg_one_mul, mul_pow, ← div_div]
    /-
      case h.e'_5.h
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) ((fourier ↑n) ↑x)) …
    -/
    congr 2
    /-
      case h.e'_5.h.e_a.e_a
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) k)) (HPow.hPow (-1) k)) (HDiv.hDi …
    -/
    rw [div_mul_eq_mul_div₀, one_mul]
    /-
      case h.e'_5.h.e_a.e_a
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Eq (HDiv.hDiv (HPow.hPow (-1) k) (HPow.hPow (↑n) k)) (HDiv.hDiv (HDiv.hDiv 1 …
    -/
    congr 1
    /-
      case h.e'_5.h.e_a.e_a.e_a
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Eq (HPow.hPow (-1) k) (HDiv.hDiv 1 (HPow.hPow (-1) k))
    -/
    rw [eq_div_iff, ← mul_pow, ← neg_eq_neg_one_mul, neg_neg, one_pow]
    /-
      case h.e'_5.h.e_a.e_a.e_a
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      n : Nat
      ⊢ Ne (HPow.hPow (-1) k) 0
    -/
    apply pow_ne_zero; rw [neg_ne_zero]; exact one_ne_zero
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case h.e'_6
      k : Nat
      hk : LE.le 2 k
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      ⊢ Eq (HMul.hMul (HDiv.hDiv (Neg.neg (HPow.hPow (HMul.hMul (HMul.hMul 2 ↑Real.p …
    -/
  · rw [Int.cast_zero, zero_pow (by positivity : k ≠ 0), div_zero, zero_mul, add_zero]
    /-
      🎉 no goals
    -/


theorem hasSum_one_div_nat_pow_mul_cos {k : ℕ} (hk : k ≠ 0) {x : ℝ} (hx : x ∈ Icc (0 : ℝ) 1) :
    HasSum (fun n : ℕ => 1 / (n : ℝ) ^ (2 * k) * Real.cos (2 * π * n * x))
      ((-1 : ℝ) ^ (k + 1) * (2 * π) ^ (2 * k) / 2 / (2 * k)! *
        (Polynomial.map (algebraMap ℚ ℝ) (Polynomial.bernoulli (2 * k))).eval x) := by
  have :
    HasSum (fun n : ℕ => 1 / (n : ℂ) ^ (2 * k) * (fourier n (x : 𝕌) + fourier (-n) (x : 𝕌)))
      ((-1 : ℂ) ^ (k + 1) * (2 * (π : ℂ)) ^ (2 * k) / (2 * k)! * bernoulliFun (2 * k) x) := by
    convert
      hasSum_one_div_nat_pow_mul_fourier (by omega : 2 ≤ 2 * k)
        hx using 3
    · rw [pow_mul (-1 : ℂ), neg_one_sq, one_pow, one_mul]
    · rw [pow_add, pow_one]
      conv_rhs =>
        rw [mul_pow]
        congr
        congr
        · skip
        · rw [pow_mul, I_sq]
      ring
  /-
    k : Nat
    hk : Ne k 0
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) (R …
  -/
  have ofReal_two : ((2 : ℝ) : ℂ) = 2 := by norm_cast
  /-
    k : Nat
    hk : Ne k 0
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
    ofReal_two : Eq (↑2) 2
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) (R …
  -/
  convert ((hasSum_iff _ _).mp (this.div_const 2)).1 with n
    /-
      case h.e'_5.h
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
      ofReal_two : Eq (↑2) 2
      n : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) (Real.cos (HMul …
    -/
  · convert (ofReal_re _).symm
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
      ofReal_two : Eq (↑2) 2
      n : Nat
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) (HAd …
    -/
    rw [ofReal_mul]; rw [← mul_div]; congr
      /-
        case h.e'_3.h.e'_1.e_a
        k : Nat
        hk : Ne k 0
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
        ofReal_two : Eq (↑2) 2
        n : Nat
        ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) ↑(HDiv.hDiv 1 (HPow.hPow ( …
      -/
    · rw [ofReal_div, ofReal_one, ofReal_pow]; rfl
                                               /-
                                                 🎉 no goals
                                               -/
    · rw [ofReal_cos, ofReal_mul, fourier_coe_apply, fourier_coe_apply, cos, ofReal_one, div_one,
        div_one, ofReal_mul, ofReal_mul, ofReal_two, Int.cast_neg, Int.cast_natCast,
        ofReal_natCast]
      /-
        case h.e'_3.h.e'_1.e_a
        k : Nat
        hk : Ne k 0
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
        ofReal_two : Eq (↑2) 2
        n : Nat
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd (Complex.exp (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      -/
      congr 3
        /-
          case h.e'_3.h.e'_1.e_a.e_a.e_a.e_z
          k : Nat
          hk : Ne k 0
          x : Real
          hx : Membership.mem (Set.Icc 0 1) x
          this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
          ofReal_two : Eq (↑2) 2
          n : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ↑n) ↑x …
        -/
      · ring
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3.h.e'_1.e_a.e_a.e_a.e_z
          k : Nat
          hk : Ne k 0
          x : Real
          hx : Membership.mem (Set.Icc 0 1) x
          this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
          ofReal_two : Eq (↑2) 2
          n : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (Neg.n …
        -/
      · ring
        /-
          🎉 no goals
        -/
    /-
      case h.e'_6
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
      ofReal_two : Eq (↑2) 2
      ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
    -/
  · convert (ofReal_re _).symm
    rw [ofReal_mul, ofReal_div, ofReal_div, ofReal_mul, ofReal_pow, ofReal_pow, ofReal_neg,
      ofReal_natCast, ofReal_mul, ofReal_two, ofReal_one]
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
      ofReal_two : Eq (↑2) 2
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
    -/
    rw [bernoulliFun]
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k) …
      ofReal_two : Eq (↑2) 2
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem hasSum_one_div_nat_pow_mul_sin {k : ℕ} (hk : k ≠ 0) {x : ℝ} (hx : x ∈ Icc (0 : ℝ) 1) :
    HasSum (fun n : ℕ => 1 / (n : ℝ) ^ (2 * k + 1) * Real.sin (2 * π * n * x))
      ((-1 : ℝ) ^ (k + 1) * (2 * π) ^ (2 * k + 1) / 2 / (2 * k + 1)! *
        (Polynomial.map (algebraMap ℚ ℝ) (Polynomial.bernoulli (2 * k + 1))).eval x) := by
  have :
    HasSum (fun n : ℕ => 1 / (n : ℂ) ^ (2 * k + 1) * (fourier n (x : 𝕌) - fourier (-n) (x : 𝕌)))
      ((-1 : ℂ) ^ (k + 1) * I * (2 * π : ℂ) ^ (2 * k + 1) / (2 * k + 1)! *
        bernoulliFun (2 * k + 1) x) := by
    convert
      hasSum_one_div_nat_pow_mul_fourier
        (by omega : 2 ≤ 2 * k + 1) hx using 1
    · ext1 n
      rw [pow_add (-1 : ℂ), pow_mul (-1 : ℂ), neg_one_sq, one_pow, one_mul, pow_one, ←
        neg_eq_neg_one_mul, ← sub_eq_add_neg]
    · congr
      rw [pow_add, pow_one]
      conv_rhs =>
        rw [mul_pow]
        congr
        congr
        · skip
        · rw [pow_add, pow_one, pow_mul, I_sq]
      ring
  /-
    k : Nat
    hk : Ne k 0
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMul.hMu …
  -/
  have ofReal_two : ((2 : ℝ) : ℂ) = 2 := by norm_cast
  /-
    k : Nat
    hk : Ne k 0
    x : Real
    hx : Membership.mem (Set.Icc 0 1) x
    this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
    ofReal_two : Eq (↑2) 2
    ⊢ HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMul.hMu …
  -/
  convert ((hasSum_iff _ _).mp (this.div_const (2 * I))).1
    /-
      case h.e'_5.h
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
      ofReal_two : Eq (↑2) 2
      x✝ : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑x✝) (HAdd.hAdd (HMul.hMul 2 k) 1)))  …
    -/
  · convert (ofReal_re _).symm
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
      ofReal_two : Eq (↑2) 2
      x✝ : Nat
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑x✝) (HAdd.hAdd (HMul.hMul …
    -/
    rw [ofReal_mul]; rw [← mul_div]; congr
      /-
        case h.e'_3.h.e'_1.e_a
        k : Nat
        hk : Ne k 0
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
        ofReal_two : Eq (↑2) 2
        x✝ : Nat
        ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (↑x✝) (HAdd.hAdd (HMul.hMul 2 k) 1))) ↑(HDiv.hDiv …
      -/
    · rw [ofReal_div, ofReal_one, ofReal_pow]; rfl
                                               /-
                                                 🎉 no goals
                                               -/
    · rw [ofReal_sin, ofReal_mul, fourier_coe_apply, fourier_coe_apply, sin, ofReal_one, div_one,
        div_one, ofReal_mul, ofReal_mul, ofReal_two, Int.cast_neg, Int.cast_natCast,
        ofReal_natCast, ← div_div, div_I, div_mul_eq_mul_div₀, ← neg_div, ← neg_mul, neg_sub]
      /-
        case h.e'_3.h.e'_1.e_a
        k : Nat
        hk : Ne k 0
        x : Real
        hx : Membership.mem (Set.Icc 0 1) x
        this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
        ofReal_two : Eq (↑2) 2
        x✝ : Nat
        ⊢ Eq (HDiv.hDiv (HMul.hMul (HSub.hSub (Complex.exp (HMul.hMul (HMul.hMul (HMul …
      -/
      congr 4
        /-
          case h.e'_3.h.e'_1.e_a.e_a.e_a.e_a.e_z
          k : Nat
          hk : Ne k 0
          x : Real
          hx : Membership.mem (Set.Icc 0 1) x
          this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
          ofReal_two : Eq (↑2) 2
          x✝ : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) (Neg.n …
        -/
      · ring
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3.h.e'_1.e_a.e_a.e_a.e_a.e_z
          k : Nat
          hk : Ne k 0
          x : Real
          hx : Membership.mem (Set.Icc 0 1) x
          this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
          ofReal_two : Eq (↑2) 2
          x✝ : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ↑x✝) ↑ …
        -/
      · ring
        /-
          🎉 no goals
        -/
    /-
      case h.e'_6
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
      ofReal_two : Eq (↑2) 2
      ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
    -/
  · convert (ofReal_re _).symm
    rw [ofReal_mul, ofReal_div, ofReal_div, ofReal_mul, ofReal_pow, ofReal_pow, ofReal_neg,
      ofReal_natCast, ofReal_mul, ofReal_two, ofReal_one, ← div_div, div_I,
      div_mul_eq_mul_div₀]
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMu …
      ofReal_two : Eq (↑2) 2
      ⊢ Eq (Neg.neg (HDiv.hDiv (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul.hMu …
    -/
    have : ∀ α β γ δ : ℂ, α * I * β / γ * δ * I = I ^ 2 * α * β / γ * δ := by intros; ring
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this✝ : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HM …
      ofReal_two : Eq (↑2) 2
      this : ∀ (α β γ δ : Complex), Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul ( …
      ⊢ Eq (Neg.neg (HDiv.hDiv (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul.hMu …
    -/
    rw [this, I_sq]
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this✝ : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HM …
      ofReal_two : Eq (↑2) 2
      this : ∀ (α β γ δ : Complex), Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul ( …
      ⊢ Eq (Neg.neg (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul.hMul (-1) (HPo …
    -/
    rw [bernoulliFun]
    /-
      case h.e'_3.h.e'_1
      k : Nat
      hk : Ne k 0
      x : Real
      hx : Membership.mem (Set.Icc 0 1) x
      this✝ : HasSum (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HM …
      ofReal_two : Eq (↑2) 2
      this : ∀ (α β γ δ : Complex), Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul ( …
      ⊢ Eq (Neg.neg (HDiv.hDiv (HMul.hMul (HDiv.hDiv (HMul.hMul (HMul.hMul (-1) (HPo …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem hasSum_zeta_nat {k : ℕ} (hk : k ≠ 0) :
    HasSum (fun n : ℕ => 1 / (n : ℝ) ^ (2 * k))
      ((-1 : ℝ) ^ (k + 1) * (2 : ℝ) ^ (2 * k - 1) * π ^ (2 * k) *
        bernoulli (2 * k) / (2 * k)!) := by
  /-
    k : Nat
    hk : Ne k 0
    ⊢ HasSum (fun n => HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) (HDiv.hDiv (H …
  -/
  convert hasSum_one_div_nat_pow_mul_cos hk (left_mem_Icc.mpr zero_le_one) using 1
    /-
      case h.e'_5
      k : Nat
      hk : Ne k 0
      ⊢ Eq (fun n => HDiv.hDiv 1 (HPow.hPow (↑n) (HMul.hMul 2 k))) fun n => HMul.hMu …
    -/
  · ext1 n; rw [mul_zero, Real.cos_zero, mul_one]
            /-
              🎉 no goals
            -/
  /-
    case h.e'_6
    k : Nat
    hk : Ne k 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
  -/
  rw [Polynomial.eval_zero_map, Polynomial.bernoulli_eval_zero, eq_ratCast]
  have : (2 : ℝ) ^ (2 * k - 1) = (2 : ℝ) ^ (2 * k) / 2 := by
    rw [eq_div_iff (two_ne_zero' ℝ)]
    conv_lhs =>
      congr
      · skip
      · rw [← pow_one (2 : ℝ)]
    rw [← pow_add, Nat.sub_add_cancel]
    omega
  /-
    case h.e'_6
    k : Nat
    hk : Ne k 0
    this : Eq (HPow.hPow 2 (HSub.hSub (HMul.hMul 2 k) 1)) (HDiv.hDiv (HPow.hPow 2  …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
  -/
  rw [this, mul_pow]
  /-
    case h.e'_6
    k : Nat
    hk : Ne k 0
    this : Eq (HPow.hPow 2 (HSub.hSub (HMul.hMul 2 k) 1)) (HDiv.hDiv (HPow.hPow 2  …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd k  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem hasSum_zeta_two : HasSum (fun n : ℕ => (1 : ℝ) / (n : ℝ) ^ 2) (π ^ 2 / 6) := by
  /-
    ⊢ HasSum (fun n => HDiv.hDiv 1 (HPow.hPow (↑n) 2)) (HDiv.hDiv (HPow.hPow Real. …
  -/
  convert hasSum_zeta_nat one_ne_zero using 1; rw [mul_one]
  /-
    case h.e'_6
    ⊢ Eq (HDiv.hDiv (HPow.hPow Real.pi 2) 6) (HDiv.hDiv (HMul.hMul (HMul.hMul (HMu …
  -/
  rw [bernoulli_eq_bernoulli'_of_ne_one (by decide : 2 ≠ 1), bernoulli'_two]
  /-
    case h.e'_6
    ⊢ Eq (HDiv.hDiv (HPow.hPow Real.pi 2) 6) (HDiv.hDiv (HMul.hMul (HMul.hMul (HMu …
  -/
  norm_num [Nat.factorial]; field_simp; ring
                                        /-
                                          🎉 no goals
                                        -/


theorem hasSum_zeta_four : HasSum (fun n : ℕ => (1 : ℝ) / (n : ℝ) ^ 4) (π ^ 4 / 90) := by
  /-
    ⊢ HasSum (fun n => HDiv.hDiv 1 (HPow.hPow (↑n) 4)) (HDiv.hDiv (HPow.hPow Real. …
  -/
  convert hasSum_zeta_nat two_ne_zero using 1; norm_num
  /-
    case h.e'_6
    ⊢ Eq (HDiv.hDiv (HPow.hPow Real.pi 4) 90) (HDiv.hDiv (Neg.neg (HMul.hMul (HMul …
  -/
  rw [bernoulli_eq_bernoulli'_of_ne_one, bernoulli'_four]
    /-
      case h.e'_6
      ⊢ Eq (HDiv.hDiv (HPow.hPow Real.pi 4) 90) (HDiv.hDiv (Neg.neg (HMul.hMul (HMul …
    -/
  · norm_num [Nat.factorial]; field_simp; ring
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case h.e'_6
      ⊢ Ne 4 1
    -/
  · decide
    /-
      🎉 no goals
    -/


theorem Polynomial.bernoulli_three_eval_one_quarter :
    (Polynomial.bernoulli 3).eval (1 / 4) = 3 / 64 := by
  simp_rw [Polynomial.bernoulli, Finset.sum_range_succ, Polynomial.eval_add,
    Polynomial.eval_monomial]
  /-
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Polynomial.eval (1 / 4) ((Fi …
  -/
  rw [Finset.sum_range_zero, Polynomial.eval_zero, zero_add, bernoulli_one]
  rw [bernoulli_eq_bernoulli'_of_ne_one zero_ne_one, bernoulli'_zero,
    bernoulli_eq_bernoulli'_of_ne_one (by decide : 2 ≠ 1), bernoulli'_two,
    bernoulli_eq_bernoulli'_of_ne_one (by decide : 3 ≠ 1), bernoulli'_three]
  /-
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul 1 ↑(Nat.choose 3 0 …
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- Explicit formula for `L(χ, 3)`, where `χ` is the unique nontrivial Dirichlet character modulo 4.
-/
theorem hasSum_L_function_mod_four_eval_three :
    HasSum (fun n : ℕ => (1 : ℝ) / (n : ℝ) ^ 3 * Real.sin (π * n / 2)) (π ^ 3 / 32) := by
  -- Porting note: times out with
  -- convert hasSum_one_div_nat_pow_mul_sin one_ne_zero (_ : 1 / 4 ∈ Icc (0 : ℝ) 1)
  apply (congr_arg₂ HasSum ?_ ?_).to_iff.mp <|
    hasSum_one_div_nat_pow_mul_sin one_ne_zero (?_ : 1 / 4 ∈ Icc (0 : ℝ) 1)
    /-
      ⊢ Eq (fun n => HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMul.hMul 2  …
    -/
  · ext1 n
    /-
      case h
      n : Nat
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑n) (HAdd.hAdd (HMul.hMul 2 1) 1))) ( …
    -/
    norm_num
    /-
      case h
      n : Nat
      ⊢ Or (Eq (Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) ↑n) (1 / 4))) ( …
    -/
    left
    /-
      case h.h
      n : Nat
      ⊢ Eq (Real.sin (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) ↑n) (1 / 4))) (Real …
    -/
    congr 1
    /-
      case h.h.e_x
      n : Nat
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) ↑n) (1 / 4)) (HDiv.hDiv (HMul …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd 1  …
    -/
  · have : (1 / 4 : ℝ) = (algebraMap ℚ ℝ) (1 / 4 : ℚ) := by norm_num
    rw [this, mul_pow, Polynomial.eval_map, Polynomial.eval₂_at_apply, (by decide : 2 * 1 + 1 = 3),
      Polynomial.bernoulli_three_eval_one_quarter]
    /-
      this : Eq (1 / 4) ((algebraMap Rat Real) (1 / 4))
      ⊢ Eq (HMul.hMul (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd 1  …
    -/
    norm_num [Nat.factorial]; field_simp; ring
                                          /-
                                            🎉 no goals
                                          -/
    /-
      ⊢ Membership.mem (Set.Icc 0 1) (1 / 4)
    -/
  · rw [mem_Icc]; constructor
      /-
        case left
        ⊢ LE.le 0 (1 / 4)
      -/
    · linarith
      /-
        🎉 no goals
      -/
      /-
        case right
        ⊢ LE.le (1 / 4) 1
      -/
    · linarith
      /-
        🎉 no goals
      -/


