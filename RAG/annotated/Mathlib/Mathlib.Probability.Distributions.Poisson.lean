/-- The pmf of the Poisson distribution depending on its rate, as a function to ℝ -/
noncomputable
def poissonPMFReal (r : ℝ≥0) (n : ℕ) : ℝ := exp (- r) * r ^ n / n !


lemma poissonPMFRealSum (r : ℝ≥0) : HasSum (fun n ↦ poissonPMFReal r n) 1 := by
  /-
    r : NNReal
    ⊢ HasSum (fun n => ProbabilityTheory.poissonPMFReal r n) 1
  -/
  let r := r.toReal
  /-
    r✝ : NNReal
    r : Real := ↑r✝
    ⊢ HasSum (fun n => ProbabilityTheory.poissonPMFReal r✝ n) 1
  -/
  unfold poissonPMFReal
  /-
    r✝ : NNReal
    r : Real := ↑r✝
    ⊢ HasSum (fun n => HDiv.hDiv (HMul.hMul (Real.exp (Neg.neg ↑r✝)) (HPow.hPow (↑ …
  -/
  apply (hasSum_mul_left_iff (exp_ne_zero r)).mp
  /-
    r✝ : NNReal
    r : Real := ↑r✝
    ⊢ HasSum (fun i => HMul.hMul (Real.exp r) (HDiv.hDiv (HMul.hMul (Real.exp (Neg …
  -/
  simp only [mul_one]
  have : (fun i ↦ rexp r * (rexp (-r) * r ^ i / ↑(Nat.factorial i))) =
      fun i ↦ r ^ i / ↑(Nat.factorial i) := by
    ext n
    rw [mul_div_assoc, exp_neg, ← mul_assoc, ← div_eq_mul_inv, div_self (exp_ne_zero r), one_mul]
  /-
    r✝ : NNReal
    r : Real := ↑r✝
    this : Eq (fun i => HMul.hMul (Real.exp r) (HDiv.hDiv (HMul.hMul (Real.exp (Ne …
    ⊢ HasSum (fun i => HMul.hMul (Real.exp r) (HDiv.hDiv (HMul.hMul (Real.exp (Neg …
  -/
  rw [this, exp_eq_exp_ℝ]
  /-
    r✝ : NNReal
    r : Real := ↑r✝
    this : Eq (fun i => HMul.hMul (Real.exp r) (HDiv.hDiv (HMul.hMul (Real.exp (Ne …
    ⊢ HasSum (fun i => HDiv.hDiv (HPow.hPow r i) ↑i.factorial) (NormedSpace.exp Re …
  -/
  exact NormedSpace.expSeries_div_hasSum_exp ℝ r
  /-
    🎉 no goals
  -/


/-- The Poisson pmf is positive for all natural numbers -/
lemma poissonPMFReal_pos {r : ℝ≥0} {n : ℕ} (hr : 0 < r) : 0 < poissonPMFReal r n := by
  /-
    r : NNReal
    n : Nat
    hr : LT.lt 0 r
    ⊢ LT.lt 0 (ProbabilityTheory.poissonPMFReal r n)
  -/
  rw [poissonPMFReal]
  /-
    r : NNReal
    n : Nat
    hr : LT.lt 0 r
    ⊢ LT.lt 0 (HDiv.hDiv (HMul.hMul (Real.exp (Neg.neg ↑r)) (HPow.hPow (↑r) n)) ↑n …
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma poissonPMFReal_nonneg {r : ℝ≥0} {n : ℕ} : 0 ≤ poissonPMFReal r n := by
  /-
    r : NNReal
    n : Nat
    ⊢ LE.le 0 (ProbabilityTheory.poissonPMFReal r n)
  -/
  unfold poissonPMFReal
  /-
    r : NNReal
    n : Nat
    ⊢ LE.le 0 (HDiv.hDiv (HMul.hMul (Real.exp (Neg.neg ↑r)) (HPow.hPow (↑r) n)) ↑n …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- The pmf of the Poisson distribution depending on its rate, as a PMF. -/
noncomputable
def poissonPMF (r : ℝ≥0) : PMF ℕ := by
  /-
    r : NNReal
    ⊢ PMF Nat
  -/
  refine ⟨fun n ↦ ENNReal.ofReal (poissonPMFReal r n), ?_⟩
  /-
    r : NNReal
    ⊢ HasSum (fun n => ENNReal.ofReal (ProbabilityTheory.poissonPMFReal r n)) 1
  -/
  apply ENNReal.hasSum_coe.mpr
  /-
    r : NNReal
    ⊢ HasSum (fun a => (ProbabilityTheory.poissonPMFReal r a).toNNReal) 1
  -/
  rw [← toNNReal_one]
  /-
    r : NNReal
    ⊢ HasSum (fun a => (ProbabilityTheory.poissonPMFReal r a).toNNReal) (Real.toNN …
  -/
  exact (poissonPMFRealSum r).toNNReal (fun n ↦ poissonPMFReal_nonneg)
  /-
    🎉 no goals
  -/


/-- The Poisson pmf is measurable. -/
@[measurability]
                                                                                /-
                                                                                  r : NNReal
                                                                                  ⊢ Measurable (ProbabilityTheory.poissonPMFReal r)
                                                                                -/
lemma measurable_poissonPMFReal (r : ℝ≥0) : Measurable (poissonPMFReal r) := by measurability
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[measurability]
lemma stronglyMeasurable_poissonPMFReal (r : ℝ≥0) : StronglyMeasurable (poissonPMFReal r) :=
  stronglyMeasurable_iff_measurable.mpr (measurable_poissonPMFReal r)


/-- Measure defined by the Poisson distribution -/
noncomputable
def poissonMeasure (r : ℝ≥0) : Measure ℕ := (poissonPMF r).toMeasure


instance isProbabilityMeasurePoisson (r : ℝ≥0) :
    IsProbabilityMeasure (poissonMeasure r) := PMF.toMeasure.isProbabilityMeasure (poissonPMF r)


