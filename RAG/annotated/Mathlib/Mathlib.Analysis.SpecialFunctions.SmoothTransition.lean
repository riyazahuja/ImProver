/-- `expNegInvGlue` is the real function given by `x ↦ exp (-1/x)` for `x > 0` and `0`
for `x ≤ 0`. It is a basic building block to construct smooth partitions of unity. Its main property
is that it vanishes for `x ≤ 0`, it is positive for `x > 0`, and the junction between the two
behaviors is flat enough to retain smoothness. The fact that this function is `C^∞` is proved in
`expNegInvGlue.contDiff`. -/
def expNegInvGlue (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else exp (-x⁻¹)


/-- The function `expNegInvGlue` vanishes on `(-∞, 0]`. -/
                                                                        /-
                                                                          x : Real
                                                                          hx : LE.le x 0
                                                                          ⊢ Eq (expNegInvGlue x) 0
                                                                        -/
theorem zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : expNegInvGlue x = 0 := by simp [expNegInvGlue, hx]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
protected theorem zero : expNegInvGlue 0 = 0 := zero_of_nonpos le_rfl


/-- The function `expNegInvGlue` is positive on `(0, +∞)`. -/
theorem pos_of_pos {x : ℝ} (hx : 0 < x) : 0 < expNegInvGlue x := by
  /-
    x : Real
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (expNegInvGlue x)
  -/
  simp [expNegInvGlue, not_le.2 hx, exp_pos]
  /-
    🎉 no goals
  -/


/-- The function `expNegInvGlue` is nonnegative. -/
theorem nonneg (x : ℝ) : 0 ≤ expNegInvGlue x := by
  cases le_or_gt x 0 with
  | inl h => exact ge_of_eq (zero_of_nonpos h)
  | inr h => exact le_of_lt (pos_of_pos h)


@[simp] theorem zero_iff_nonpos {x : ℝ} : expNegInvGlue x = 0 ↔ x ≤ 0 :=
  ⟨fun h ↦ not_lt.mp fun h' ↦ (pos_of_pos h').ne' h, zero_of_nonpos⟩


/-- Our function tends to zero at zero faster than any $P(x^{-1})$, $P∈ℝ[X]$, tends to infinity. -/
theorem tendsto_polynomial_inv_mul_zero (p : ℝ[X]) :
    Tendsto (fun x ↦ p.eval x⁻¹ * expNegInvGlue x) (𝓝 0) (𝓝 0) := by
  /-
    p : Polynomial Real
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegIn …
  -/
  simp only [expNegInvGlue, mul_ite, mul_zero]
  /-
    p : Polynomial Real
    ⊢ Filter.Tendsto (fun x => ite (LE.le x 0) 0 (HMul.hMul (Polynomial.eval (Inv. …
  -/
  refine tendsto_const_nhds.if ?_
  /-
    p : Polynomial Real
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (Real.exp …
  -/
  simp only [not_le]
  have : Tendsto (fun x ↦ p.eval x⁻¹ / exp x⁻¹) (𝓝[>] 0) (𝓝 0) :=
    p.tendsto_div_exp_atTop.comp tendsto_inv_nhdsGT_zero
  /-
    p : Polynomial Real
    this : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval (Inv.inv x) p) (Rea …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (Real.exp …
  -/
  refine this.congr' <| mem_of_superset self_mem_nhdsWithin fun x hx ↦ ?_
  /-
    p : Polynomial Real
    this : Filter.Tendsto (fun x => HDiv.hDiv (Polynomial.eval (Inv.inv x) p) (Rea …
    x : Real
    hx : Membership.mem (setOf fun x => LT.lt 0 x) x
    ⊢ Membership.mem (setOf fun x => (fun x => Eq (HDiv.hDiv (Polynomial.eval (Inv …
  -/
  simp [expNegInvGlue, hx.out.not_le, exp_neg, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem hasDerivAt_polynomial_eval_inv_mul (p : ℝ[X]) (x : ℝ) :
    HasDerivAt (fun x ↦ p.eval x⁻¹ * expNegInvGlue x)
      ((X ^ 2 * (p - derivative (R := ℝ) p)).eval x⁻¹ * expNegInvGlue x) x := by
  /-
    p : Polynomial Real
    x : Real
    ⊢ HasDerivAt (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegInvGlu …
  -/
  rcases lt_trichotomy x 0 with hx | rfl | hx
    /-
      case inl
      p : Polynomial Real
      x : Real
      hx : LT.lt x 0
      ⊢ HasDerivAt (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegInvGlu …
    -/
  · rw [zero_of_nonpos hx.le, mul_zero]
    /-
      case inl
      p : Polynomial Real
      x : Real
      hx : LT.lt x 0
      ⊢ HasDerivAt (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegInvGlu …
    -/
    refine (hasDerivAt_const _ 0).congr_of_eventuallyEq ?_
    /-
      case inl
      p : Polynomial Real
      x : Real
      hx : LT.lt x 0
      ⊢ (nhds x).EventuallyEq (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (e …
    -/
    filter_upwards [gt_mem_nhds hx] with y hy
    /-
      case h
      p : Polynomial Real
      x : Real
      hx : LT.lt x 0
      y : Real
      hy : LT.lt y 0
      ⊢ Eq (HMul.hMul (Polynomial.eval (Inv.inv y) p) (expNegInvGlue y)) 0
    -/
    rw [zero_of_nonpos hy.le, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      p : Polynomial Real
      ⊢ HasDerivAt (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegInvGlu …
    -/
  · rw [expNegInvGlue.zero, mul_zero, hasDerivAt_iff_tendsto_slope]
    /-
      case inr.inl
      p : Polynomial Real
      ⊢ Filter.Tendsto (slope (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (e …
    -/
    refine ((tendsto_polynomial_inv_mul_zero (p * X)).mono_left inf_le_left).congr fun x ↦ ?_
    /-
      case inr.inl
      p : Polynomial Real
      x : Real
      ⊢ Eq (HMul.hMul (Polynomial.eval (Inv.inv x) (HMul.hMul p Polynomial.X)) (expN …
    -/
    simp [slope_def_field, div_eq_mul_inv, mul_right_comm]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p : Polynomial Real
      x : Real
      hx : LT.lt 0 x
      ⊢ HasDerivAt (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegInvGlu …
    -/
  · have := ((p.hasDerivAt x⁻¹).mul (hasDerivAt_neg _).exp).comp x (hasDerivAt_inv hx.ne')
    /-
      case inr.inr
      p : Polynomial Real
      x : Real
      hx : LT.lt 0 x
      this : HasDerivAt (Function.comp (fun y => HMul.hMul (Polynomial.eval y p) (Re …
      ⊢ HasDerivAt (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegInvGlu …
    -/
    convert this.congr_of_eventuallyEq _ using 1
      /-
        case h.e'_9
        p : Polynomial Real
        x : Real
        hx : LT.lt 0 x
        this : HasDerivAt (Function.comp (fun y => HMul.hMul (Polynomial.eval y p) (Re …
        ⊢ Eq (HMul.hMul (Polynomial.eval (Inv.inv x) (HMul.hMul (HPow.hPow Polynomial. …
      -/
    · simp [expNegInvGlue, hx.not_le]
      /-
        case h.e'_9
        p : Polynomial Real
        x : Real
        hx : LT.lt 0 x
        this : HasDerivAt (Function.comp (fun y => HMul.hMul (Polynomial.eval y p) (Re …
        ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv (HPow.hPow x 2)) (HSub.hSub (Polynomial.ev …
      -/
      ring
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.convert_2
        p : Polynomial Real
        x : Real
        hx : LT.lt 0 x
        this : HasDerivAt (Function.comp (fun y => HMul.hMul (Polynomial.eval y p) (Re …
        ⊢ (nhds x).EventuallyEq (fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (e …
      -/
    · filter_upwards [lt_mem_nhds hx] with y hy
      /-
        case h
        p : Polynomial Real
        x : Real
        hx : LT.lt 0 x
        this : HasDerivAt (Function.comp (fun y => HMul.hMul (Polynomial.eval y p) (Re …
        y : Real
        hy : LT.lt 0 y
        ⊢ Eq (HMul.hMul (Polynomial.eval (Inv.inv y) p) (expNegInvGlue y)) (Function.c …
      -/
      simp [expNegInvGlue, hy.not_le]
      /-
        🎉 no goals
      -/


theorem differentiable_polynomial_eval_inv_mul (p : ℝ[X]) :
    Differentiable ℝ (fun x ↦ p.eval x⁻¹ * expNegInvGlue x) := fun x ↦
  (hasDerivAt_polynomial_eval_inv_mul p x).differentiableAt


theorem continuous_polynomial_eval_inv_mul (p : ℝ[X]) :
    Continuous (fun x ↦ p.eval x⁻¹ * expNegInvGlue x) :=
  (differentiable_polynomial_eval_inv_mul p).continuous


theorem contDiff_polynomial_eval_inv_mul {n : ℕ∞} (p : ℝ[X]) :
    ContDiff ℝ n (fun x ↦ p.eval x⁻¹ * expNegInvGlue x) := by
  /-
    n : ENat
    p : Polynomial Real
    ⊢ ContDiff Real ↑n fun x => HMul.hMul (Polynomial.eval (Inv.inv x) p) (expNegI …
  -/
  apply contDiff_all_iff_nat.2 (fun m => ?_) n
  induction m generalizing p with
  | zero => exact contDiff_zero.2 <| continuous_polynomial_eval_inv_mul _
  | succ m ihm =>
    rw [show ((m + 1 : ℕ) : WithTop ℕ∞) = m + 1 from rfl]
    refine contDiff_succ_iff_deriv.2 ⟨differentiable_polynomial_eval_inv_mul _, by simp, ?_⟩
    convert ihm (X ^ 2 * (p - derivative (R := ℝ) p)) using 2
    exact (hasDerivAt_polynomial_eval_inv_mul p _).deriv


/-- The function `expNegInvGlue` is smooth. -/
protected theorem contDiff {n : ℕ∞} : ContDiff ℝ n expNegInvGlue := by
  /-
    n : ENat
    ⊢ ContDiff Real (↑n) expNegInvGlue
  -/
  simpa using contDiff_polynomial_eval_inv_mul 1
  /-
    🎉 no goals
  -/


/-- An infinitely smooth function `f : ℝ → ℝ` such that `f x = 0` for `x ≤ 0`,
`f x = 1` for `1 ≤ x`, and `0 < f x < 1` for `0 < x < 1`. -/
def Real.smoothTransition (x : ℝ) : ℝ :=
  expNegInvGlue x / (expNegInvGlue x + expNegInvGlue (1 - x))


theorem pos_denom (x) : 0 < expNegInvGlue x + expNegInvGlue (1 - x) :=
  (zero_lt_one.lt_or_lt x).elim (fun hx => add_pos_of_pos_of_nonneg (pos_of_pos hx) (nonneg _))
    fun hx => add_pos_of_nonneg_of_pos (nonneg _) (pos_of_pos <| sub_pos.2 hx)


theorem one_of_one_le (h : 1 ≤ x) : smoothTransition x = 1 :=
                                                   /-
                                                     x : Real
                                                     h : LE.le 1 x
                                                     ⊢ Eq (expNegInvGlue x) (HAdd.hAdd (expNegInvGlue x) (expNegInvGlue (HSub.hSub  …
                                                   -/
  (div_eq_one_iff_eq <| (pos_denom x).ne').2 <| by rw [zero_of_nonpos (sub_nonpos.2 h), add_zero]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
nonrec theorem zero_iff_nonpos : smoothTransition x = 0 ↔ x ≤ 0 := by
  /-
    x : Real
    ⊢ Iff (Eq x.smoothTransition 0) (LE.le x 0)
  -/
  simp only [smoothTransition, _root_.div_eq_zero_iff, (pos_denom x).ne', zero_iff_nonpos, or_false]
  /-
    🎉 no goals
  -/


theorem zero_of_nonpos (h : x ≤ 0) : smoothTransition x = 0 := zero_iff_nonpos.2 h


@[simp]
protected theorem zero : smoothTransition 0 = 0 :=
  zero_of_nonpos le_rfl


@[simp]
protected theorem one : smoothTransition 1 = 1 :=
  one_of_one_le le_rfl


/-- Since `Real.smoothTransition` is constant on $(-∞, 0]$ and $[1, ∞)$, applying it to the
projection of `x : ℝ` to $[0, 1]$ gives the same result as applying it to `x`. -/
@[simp]
protected theorem projIcc :
    smoothTransition (projIcc (0 : ℝ) 1 zero_le_one x) = smoothTransition x := by
  refine congr_fun
    (IccExtend_eq_self zero_le_one smoothTransition (fun x hx => ?_) fun x hx => ?_) x
    /-
      case refine_1
      x✝ x : Real
      hx : LT.lt x 0
      ⊢ Eq x.smoothTransition (Real.smoothTransition 0)
    -/
  · rw [smoothTransition.zero, zero_of_nonpos hx.le]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x✝ x : Real
      hx : LT.lt 1 x
      ⊢ Eq x.smoothTransition (Real.smoothTransition 1)
    -/
  · rw [smoothTransition.one, one_of_one_le hx.le]
    /-
      🎉 no goals
    -/


theorem le_one (x : ℝ) : smoothTransition x ≤ 1 :=
  (div_le_one (pos_denom x)).2 <| le_add_of_nonneg_right (nonneg _)


theorem nonneg (x : ℝ) : 0 ≤ smoothTransition x :=
  div_nonneg (expNegInvGlue.nonneg _) (pos_denom x).le


theorem lt_one_of_lt_one (h : x < 1) : smoothTransition x < 1 :=
  (div_lt_one <| pos_denom x).2 <| lt_add_of_pos_right _ <| pos_of_pos <| sub_pos.2 h


theorem pos_of_pos (h : 0 < x) : 0 < smoothTransition x :=
  div_pos (expNegInvGlue.pos_of_pos h) (pos_denom x)


protected theorem contDiff {n : ℕ∞} : ContDiff ℝ n smoothTransition :=
  expNegInvGlue.contDiff.div
    (expNegInvGlue.contDiff.add <| expNegInvGlue.contDiff.comp <| contDiff_const.sub contDiff_id)
    fun x => (pos_denom x).ne'


protected theorem contDiffAt {x : ℝ} {n : ℕ∞} : ContDiffAt ℝ n smoothTransition x :=
  smoothTransition.contDiff.contDiffAt


protected theorem continuous : Continuous smoothTransition :=
  (@smoothTransition.contDiff 0).continuous


protected theorem continuousAt : ContinuousAt smoothTransition x :=
  smoothTransition.continuous.continuousAt


