lemma rexp_neg_quadratic_isLittleO_rpow_atTop {a : ℝ} (ha : a < 0) (b s : ℝ) :
    (fun x ↦ rexp (a * x ^ 2 + b * x)) =o[atTop] (· ^ s) := by
  suffices (fun x ↦ rexp (a * x ^ 2 + b * x)) =o[atTop] (fun x ↦ rexp (-x)) by
    refine this.trans ?_
    simpa only [neg_one_mul] using isLittleO_exp_neg_mul_rpow_atTop zero_lt_one s
  /-
    a : Real
    ha : LT.lt a 0
    b s : Real
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Real.exp (HAdd.hAdd (HMul.hMul  …
  -/
  rw [isLittleO_exp_comp_exp_comp]
  have : (fun x ↦ -x - (a * x ^ 2 + b * x)) = fun x ↦ x * (-a * x - (b + 1)) := by
    ext1 x; ring_nf
  /-
    a : Real
    ha : LT.lt a 0
    b s : Real
    this : Eq (fun x => HSub.hSub (Neg.neg x) (HAdd.hAdd (HMul.hMul a (HPow.hPow x …
    ⊢ Filter.Tendsto (fun x => HSub.hSub (Neg.neg x) (HAdd.hAdd (HMul.hMul a (HPow …
  -/
  rw [this]
  exact tendsto_id.atTop_mul_atTop <|
    Filter.tendsto_atTop_add_const_right _ _ <| tendsto_id.const_mul_atTop (neg_pos.mpr ha)


lemma cexp_neg_quadratic_isLittleO_rpow_atTop {a : ℂ} (ha : a.re < 0) (b : ℂ) (s : ℝ) :
    (fun x : ℝ ↦ cexp (a * x ^ 2 + b * x)) =o[atTop] (· ^ s) := by
  /-
    a : Complex
    ha : LT.lt a.re 0
    b : Complex
    s : Real
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Complex.exp (HAdd.hAdd (HMul.hM …
  -/
  apply Asymptotics.IsLittleO.of_norm_left
  /-
    case a
    a : Complex
    ha : LT.lt a.re 0
    b : Complex
    s : Real
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Norm.norm (Complex.exp (HAdd.hA …
  -/
  convert rexp_neg_quadratic_isLittleO_rpow_atTop ha b.re s with x
  simp_rw [Complex.norm_eq_abs, Complex.abs_exp, add_re, ← ofReal_pow, mul_comm (_ : ℂ) ↑(_ : ℝ),
      re_ofReal_mul, mul_comm _ (re _)]


lemma cexp_neg_quadratic_isLittleO_abs_rpow_cocompact {a : ℂ} (ha : a.re < 0) (b : ℂ) (s : ℝ) :
    (fun x : ℝ ↦ cexp (a * x ^ 2 + b * x)) =o[cocompact ℝ] (|·| ^ s) := by
  /-
    a : Complex
    ha : LT.lt a.re 0
    b : Complex
    s : Real
    ⊢ Asymptotics.IsLittleO (Filter.cocompact Real) (fun x => Complex.exp (HAdd.hA …
  -/
  rw [cocompact_eq_atBot_atTop, isLittleO_sup]
  /-
    a : Complex
    ha : LT.lt a.re 0
    b : Complex
    s : Real
    ⊢ And (Asymptotics.IsLittleO Filter.atBot (fun x => Complex.exp (HAdd.hAdd (HM …
  -/
  constructor
  · refine ((cexp_neg_quadratic_isLittleO_rpow_atTop ha (-b) s).comp_tendsto
      Filter.tendsto_neg_atBot_atTop).congr' (Eventually.of_forall fun x ↦ ?_) ?_
      /-
        case left.refine_1
        a : Complex
        ha : LT.lt a.re 0
        b : Complex
        s x : Real
        ⊢ Eq (Function.comp (fun x => Complex.exp (HAdd.hAdd (HMul.hMul a (HPow.hPow ( …
      -/
    · simp only [neg_mul, Function.comp_apply, ofReal_neg, neg_sq, mul_neg, neg_neg]
      /-
        🎉 no goals
      -/
      /-
        case left.refine_2
        a : Complex
        ha : LT.lt a.re 0
        b : Complex
        s : Real
        ⊢ Filter.atBot.EventuallyEq (Function.comp (fun x => HPow.hPow x s) Neg.neg) f …
      -/
    · refine (eventually_lt_atBot 0).mp (Eventually.of_forall fun x hx ↦ ?_)
      /-
        case left.refine_2
        a : Complex
        ha : LT.lt a.re 0
        b : Complex
        s x : Real
        hx : LT.lt x 0
        ⊢ Eq (Function.comp (fun x => HPow.hPow x s) Neg.neg x) ((fun x => HPow.hPow ( …
      -/
      simp only [Function.comp_apply, abs_of_neg hx]
      /-
        🎉 no goals
      -/
    /-
      case right
      a : Complex
      ha : LT.lt a.re 0
      b : Complex
      s : Real
      ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => Complex.exp (HAdd.hAdd (HMul.hM …
    -/
  · refine (cexp_neg_quadratic_isLittleO_rpow_atTop ha b s).congr' EventuallyEq.rfl ?_
    /-
      case right
      a : Complex
      ha : LT.lt a.re 0
      b : Complex
      s : Real
      ⊢ Filter.atTop.EventuallyEq (fun x => HPow.hPow x s) fun x => HPow.hPow (abs x …
    -/
    refine (eventually_gt_atTop 0).mp (Eventually.of_forall fun x hx ↦ ?_)
    /-
      case right
      a : Complex
      ha : LT.lt a.re 0
      b : Complex
      s x : Real
      hx : LT.lt 0 x
      ⊢ Eq ((fun x => HPow.hPow x s) x) ((fun x => HPow.hPow (abs x) s) x)
    -/
    simp_rw [abs_of_pos hx]
    /-
      🎉 no goals
    -/


theorem tendsto_rpow_abs_mul_exp_neg_mul_sq_cocompact {a : ℝ} (ha : 0 < a) (s : ℝ) :
    Tendsto (fun x : ℝ => |x| ^ s * rexp (-a * x ^ 2)) (cocompact ℝ) (𝓝 0) := by
  /-
    a : Real
    ha : LT.lt 0 a
    s : Real
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (abs x) s) (Real.exp (HMul.hMu …
  -/
  conv in rexp _ => rw [← sq_abs]
  erw [cocompact_eq_atBot_atTop, ← comap_abs_atTop,
    @tendsto_comap'_iff _ _ _ (fun y => y ^ s * rexp (-a * y ^ 2)) _ _ _
      (mem_atTop_sets.mpr ⟨0, fun b hb => ⟨b, abs_of_nonneg hb⟩⟩)]
  exact
    (rpow_mul_exp_neg_mul_sq_isLittleO_exp_neg ha s).tendsto_zero_of_tendsto
      (tendsto_exp_atBot.comp <| tendsto_id.const_mul_atTop_of_neg (neg_lt_zero.mpr one_half_pos))


theorem isLittleO_exp_neg_mul_sq_cocompact {a : ℂ} (ha : 0 < a.re) (s : ℝ) :
    (fun x : ℝ => Complex.exp (-a * x ^ 2)) =o[cocompact ℝ] fun x : ℝ => |x| ^ s := by
  /-
    a : Complex
    ha : LT.lt 0 a.re
    s : Real
    ⊢ Asymptotics.IsLittleO (Filter.cocompact Real) (fun x => Complex.exp (HMul.hM …
  -/
  convert cexp_neg_quadratic_isLittleO_abs_rpow_cocompact (?_ : (-a).re < 0) 0 s using 1
    /-
      case h.e'_7
      a : Complex
      ha : LT.lt 0 a.re
      s : Real
      ⊢ Eq (fun x => Complex.exp (HMul.hMul (Neg.neg a) (HPow.hPow (↑x) 2))) fun x = …
    -/
  · simp_rw [zero_mul, add_zero]
    /-
      🎉 no goals
    -/
    /-
      a : Complex
      ha : LT.lt 0 a.re
      s : Real
      ⊢ LT.lt (Neg.neg a).re 0
    -/
  · rwa [neg_re, neg_lt_zero]
    /-
      🎉 no goals
    -/


/-- Jacobi's theta-function transformation formula for the sum of `exp -Q(x)`, where `Q` is a
negative definite quadratic form. -/
theorem Complex.tsum_exp_neg_quadratic {a : ℂ} (ha : 0 < a.re) (b : ℂ) :
    (∑' n : ℤ, cexp (-π * a * n ^ 2 + 2 * π * b * n)) =
      1 / a ^ (1 / 2 : ℂ) * ∑' n : ℤ, cexp (-π / a * (n + I * b) ^ 2) := by
  /-
    a : Complex
    ha : LT.lt 0 a.re
    b : Complex
    ⊢ Eq (tsum fun n => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (Neg.neg ↑Rea …
  -/
  let f : ℝ → ℂ := fun x ↦ cexp (-π * a * x ^ 2 + 2 * π * b * x)
  have hCf : Continuous f := by
    refine Complex.continuous_exp.comp (Continuous.add ?_ ?_)
    · exact continuous_const.mul (Complex.continuous_ofReal.pow 2)
    · exact continuous_const.mul Complex.continuous_ofReal
  have hFf : 𝓕 f = fun x : ℝ ↦ 1 / a ^ (1 / 2 : ℂ) * cexp (-π / a * (x + I * b) ^ 2) :=
    fourierIntegral_gaussian_pi' ha b
  have h1 : 0 < (↑π * a).re := by
    rw [re_ofReal_mul]
    exact mul_pos pi_pos ha
  have h2 : 0 < (↑π / a).re := by
    rw [div_eq_mul_inv, re_ofReal_mul, inv_re]
    refine mul_pos pi_pos (div_pos ha <| normSq_pos.mpr ?_)
    contrapose! ha
    rw [ha, zero_re]
  have f_bd : f =O[cocompact ℝ] (fun x => |x| ^ (-2 : ℝ)) := by
    convert (cexp_neg_quadratic_isLittleO_abs_rpow_cocompact ?_ _ (-2)).isBigO
    rwa [neg_mul, neg_re, neg_lt_zero]
  have Ff_bd : (𝓕 f) =O[cocompact ℝ] (fun x => |x| ^ (-2 : ℝ)) := by
    rw [hFf]
    have : ∀ (x : ℝ), -↑π / a * (↑x + I * b) ^ 2 =
        -↑π / a * x ^ 2 + (-2 * π * I * b) / a * x + π * b ^ 2 / a := by
      intro x; ring_nf; rw [I_sq]; ring
    simp_rw [this]
    conv => enter [2, x]; rw [Complex.exp_add, ← mul_assoc _ _ (Complex.exp _), mul_comm]
    refine ((cexp_neg_quadratic_isLittleO_abs_rpow_cocompact
      (?_) (-2 * ↑π * I * b / a) (-2)).isBigO.const_mul_left _).const_mul_left _
    rwa [neg_div, neg_re, neg_lt_zero]
  /-
    a : Complex
    ha : LT.lt 0 a.re
    b : Complex
    f : Real → Complex := fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (N …
    hCf : Continuous f
    hFf : Eq (Real.fourierIntegral f) fun x => HMul.hMul (HDiv.hDiv 1 (HPow.hPow a …
    h1 : LT.lt 0 (HMul.hMul (↑Real.pi) a).re
    h2 : LT.lt 0 (HDiv.hDiv (↑Real.pi) a).re
    f_bd : Asymptotics.IsBigO (Filter.cocompact Real) f fun x => HPow.hPow (_root_ …
    Ff_bd : Asymptotics.IsBigO (Filter.cocompact Real) (Real.fourierIntegral f) fu …
    ⊢ Eq (tsum fun n => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (Neg.neg ↑Rea …
  -/
  convert Real.tsum_eq_tsum_fourierIntegral_of_rpow_decay hCf one_lt_two f_bd Ff_bd 0 using 1
    /-
      case h.e'_2
      a : Complex
      ha : LT.lt 0 a.re
      b : Complex
      f : Real → Complex := fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (N …
      hCf : Continuous f
      hFf : Eq (Real.fourierIntegral f) fun x => HMul.hMul (HDiv.hDiv 1 (HPow.hPow a …
      h1 : LT.lt 0 (HMul.hMul (↑Real.pi) a).re
      h2 : LT.lt 0 (HDiv.hDiv (↑Real.pi) a).re
      f_bd : Asymptotics.IsBigO (Filter.cocompact Real) f fun x => HPow.hPow (_root_ …
      Ff_bd : Asymptotics.IsBigO (Filter.cocompact Real) (Real.fourierIntegral f) fu …
      ⊢ Eq (tsum fun n => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (Neg.neg ↑Rea …
    -/
  · simp only [f, zero_add, ofReal_intCast]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      a : Complex
      ha : LT.lt 0 a.re
      b : Complex
      f : Real → Complex := fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (N …
      hCf : Continuous f
      hFf : Eq (Real.fourierIntegral f) fun x => HMul.hMul (HDiv.hDiv 1 (HPow.hPow a …
      h1 : LT.lt 0 (HMul.hMul (↑Real.pi) a).re
      h2 : LT.lt 0 (HDiv.hDiv (↑Real.pi) a).re
      f_bd : Asymptotics.IsBigO (Filter.cocompact Real) f fun x => HPow.hPow (_root_ …
      Ff_bd : Asymptotics.IsBigO (Filter.cocompact Real) (Real.fourierIntegral f) fu …
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow a (1 / 2))) (tsum fun n => Complex.exp …
    -/
  · rw [← tsum_mul_left]
    /-
      case h.e'_3
      a : Complex
      ha : LT.lt 0 a.re
      b : Complex
      f : Real → Complex := fun x => Complex.exp (HAdd.hAdd (HMul.hMul (HMul.hMul (N …
      hCf : Continuous f
      hFf : Eq (Real.fourierIntegral f) fun x => HMul.hMul (HDiv.hDiv 1 (HPow.hPow a …
      h1 : LT.lt 0 (HMul.hMul (↑Real.pi) a).re
      h2 : LT.lt 0 (HDiv.hDiv (↑Real.pi) a).re
      f_bd : Asymptotics.IsBigO (Filter.cocompact Real) f fun x => HPow.hPow (_root_ …
      Ff_bd : Asymptotics.IsBigO (Filter.cocompact Real) (Real.fourierIntegral f) fu …
      ⊢ Eq (tsum fun x => HMul.hMul (HDiv.hDiv 1 (HPow.hPow a (1 / 2))) (Complex.exp …
    -/
    simp only [QuotientAddGroup.mk_zero, fourier_eval_zero, mul_one, hFf, ofReal_intCast]
    /-
      🎉 no goals
    -/


theorem Complex.tsum_exp_neg_mul_int_sq {a : ℂ} (ha : 0 < a.re) :
    (∑' n : ℤ, cexp (-π * a * (n : ℂ) ^ 2)) =
      1 / a ^ (1 / 2 : ℂ) * ∑' n : ℤ, cexp (-π / a * (n : ℂ) ^ 2) := by
  /-
    a : Complex
    ha : LT.lt 0 a.re
    ⊢ Eq (tsum fun n => Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑Real.pi) a) (H …
  -/
  simpa only [mul_zero, zero_mul, add_zero] using Complex.tsum_exp_neg_quadratic ha 0
  /-
    🎉 no goals
  -/


theorem Real.tsum_exp_neg_mul_int_sq {a : ℝ} (ha : 0 < a) :
    (∑' n : ℤ, exp (-π * a * (n : ℝ) ^ 2)) =
      (1 : ℝ) / a ^ (1 / 2 : ℝ) * (∑' n : ℤ, exp (-π / a * (n : ℝ) ^ 2)) := by
  simpa only [← ofReal_inj, ofReal_tsum, ofReal_exp, ofReal_mul, ofReal_neg, ofReal_pow,
    ofReal_intCast, ofReal_div, ofReal_one, ofReal_cpow ha.le, ofReal_ofNat, mul_zero, zero_mul,
    add_zero] using Complex.tsum_exp_neg_quadratic (by rwa [ofReal_re] : 0 < (a : ℂ).re) 0


