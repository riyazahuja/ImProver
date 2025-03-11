/-- `exp (-b * x)` is integrable on `(a, ∞)`. -/
theorem exp_neg_integrableOn_Ioi (a : ℝ) {b : ℝ} (h : 0 < b) :
    /-
      a b : Real
      h : LT.lt 0 b
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x : ℝ => exp (-b * x)) (Ioi a) := by
    /-
      🎉 no goals
    -/
  have : Tendsto (fun x => -exp (-b * x) / b) atTop (𝓝 (-0 / b)) := by
    refine Tendsto.div_const (Tendsto.neg ?_) _
    exact tendsto_exp_atBot.comp (tendsto_id.const_mul_atTop_of_neg (neg_neg_iff_pos.2 h))
  /-
    a b : Real
    h : LT.lt 0 b
    this : Filter.Tendsto (fun x => HDiv.hDiv (Neg.neg (Real.exp (HMul.hMul (Neg.n …
    ⊢ MeasureTheory.IntegrableOn (fun x => Real.exp (HMul.hMul (Neg.neg b) x)) (Se …
  -/
  refine integrableOn_Ioi_deriv_of_nonneg' (fun x _ => ?_) (fun x _ => (exp_pos _).le) this
  /-
    a b : Real
    h : LT.lt 0 b
    this : Filter.Tendsto (fun x => HDiv.hDiv (Neg.neg (Real.exp (HMul.hMul (Neg.n …
    x : Real
    x✝ : Membership.mem (Set.Ici a) x
    ⊢ HasDerivAt (fun x => HDiv.hDiv (Neg.neg (Real.exp (HMul.hMul (Neg.neg b) x)) …
  -/
  simpa [h.ne'] using ((hasDerivAt_id x).const_mul b).neg.exp.neg.div_const b
  /-
    🎉 no goals
  -/


/-- If `f` is continuous on `[a, ∞)`, and is `O (exp (-b * x))` at `∞` for some `b > 0`, then
`f` is integrable on `(a, ∞)`. -/
theorem integrable_of_isBigO_exp_neg {f : ℝ → ℝ} {a b : ℝ} (h0 : 0 < b)
    (hf : ContinuousOn f (Ici a)) (ho : f =O[atTop] fun x => exp (-b * x)) :
    /-
      f : Real → Real
      a b : Real
      h0 : LT.lt 0 b
      hf : ContinuousOn f (Set.Ici a)
      ho : Asymptotics.IsBigO Filter.atTop f fun x => Real.exp (HMul.hMul (Neg.neg b …
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn f (Ioi a) :=
    /-
      🎉 no goals
    -/
  integrableOn_Ici_iff_integrableOn_Ioi.mp <|
    (hf.locallyIntegrableOn measurableSet_Ici).integrableOn_of_isBigO_atTop
    ho ⟨Ioi b, Ioi_mem_atTop b, exp_neg_integrableOn_Ioi b h0⟩

