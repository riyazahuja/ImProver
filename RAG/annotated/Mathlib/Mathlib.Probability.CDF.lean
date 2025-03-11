/-- Cumulative distribution function of a real measure. The definition currently makes sense only
for probability measures. In that case, it satisfies `cdf μ x = (μ (Iic x)).toReal` (see
`ProbabilityTheory.cdf_eq_toReal`). -/
noncomputable
def cdf (μ : Measure ℝ) : StieltjesFunction :=
  condCDF ((Measure.dirac Unit.unit).prod μ) Unit.unit


/-- The cdf is non-negative. -/
lemma cdf_nonneg (x : ℝ) : 0 ≤ cdf μ x := condCDF_nonneg _ _ _


/-- The cdf is lower or equal to 1. -/
lemma cdf_le_one (x : ℝ) : cdf μ x ≤ 1 := condCDF_le_one _ _ _


/-- The cdf is monotone. -/
lemma monotone_cdf : Monotone (cdf μ) := (condCDF _ _).mono


/-- The cdf tends to 0 at -∞. -/
lemma tendsto_cdf_atBot : Tendsto (cdf μ) atBot (𝓝 0) := tendsto_condCDF_atBot _ _


/-- The cdf tends to 1 at +∞. -/
lemma tendsto_cdf_atTop : Tendsto (cdf μ) atTop (𝓝 1) := tendsto_condCDF_atTop _ _


lemma ofReal_cdf [IsProbabilityMeasure μ] (x : ℝ) : ENNReal.ofReal (cdf μ x) = μ (Iic x) := by
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    x : Real
    ⊢ Eq (ENNReal.ofReal (↑(ProbabilityTheory.cdf μ) x)) (μ (Set.Iic x))
  -/
  have h := lintegral_condCDF ((Measure.dirac Unit.unit).prod μ) x
  simpa only [MeasureTheory.Measure.fst_prod, Measure.prod_prod, measure_univ, one_mul,
    lintegral_dirac] using h


lemma cdf_eq_toReal [IsProbabilityMeasure μ] (x : ℝ) : cdf μ x = (μ (Iic x)).toReal := by
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    x : Real
    ⊢ Eq (↑(ProbabilityTheory.cdf μ) x) (μ (Set.Iic x)).toReal
  -/
  rw [← ofReal_cdf μ x, ENNReal.toReal_ofReal (cdf_nonneg μ x)]
  /-
    🎉 no goals
  -/


instance instIsProbabilityMeasurecdf : IsProbabilityMeasure (cdf μ).measure := by
  /-
    μ : MeasureTheory.Measure Real
    ⊢ MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.cdf μ).measure
  -/
  constructor
  simp only [StieltjesFunction.measure_univ _ (tendsto_cdf_atBot μ) (tendsto_cdf_atTop μ), sub_zero,
    ENNReal.ofReal_one]


/-- The measure associated to the cdf of a probability measure is the same probability measure. -/
lemma measure_cdf [IsProbabilityMeasure μ] : (cdf μ).measure = μ := by
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Eq (ProbabilityTheory.cdf μ).measure μ
  -/
  refine Measure.ext_of_Iic (cdf μ).measure μ (fun a ↦ ?_)
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsProbabilityMeasure μ
    a : Real
    ⊢ Eq ((ProbabilityTheory.cdf μ).measure (Set.Iic a)) (μ (Set.Iic a))
  -/
  rw [StieltjesFunction.measure_Iic _ (tendsto_cdf_atBot μ), sub_zero, ofReal_cdf]
  /-
    🎉 no goals
  -/


lemma cdf_measure_stieltjesFunction (f : StieltjesFunction) (hf0 : Tendsto f atBot (𝓝 0))
    (hf1 : Tendsto f atTop (𝓝 1)) :
    cdf f.measure = f := by
  /-
    f : StieltjesFunction
    hf0 : Filter.Tendsto (↑f) Filter.atBot (nhds 0)
    hf1 : Filter.Tendsto (↑f) Filter.atTop (nhds 1)
    ⊢ Eq (ProbabilityTheory.cdf f.measure) f
  -/
  refine (cdf f.measure).eq_of_measure_of_tendsto_atBot f ?_ (tendsto_cdf_atBot _) hf0
  have h_prob : IsProbabilityMeasure f.measure :=
    ⟨by rw [f.measure_univ hf0 hf1, sub_zero, ENNReal.ofReal_one]⟩
  /-
    f : StieltjesFunction
    hf0 : Filter.Tendsto (↑f) Filter.atBot (nhds 0)
    hf1 : Filter.Tendsto (↑f) Filter.atTop (nhds 1)
    h_prob : MeasureTheory.IsProbabilityMeasure f.measure
    ⊢ Eq (ProbabilityTheory.cdf f.measure).measure f.measure
  -/
  exact measure_cdf f.measure
  /-
    🎉 no goals
  -/


/-- If two real probability distributions have the same cdf, they are equal. -/
lemma MeasureTheory.Measure.eq_of_cdf (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (h : cdf μ = cdf ν) : μ = ν := by
  /-
    μ ν : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
    inst✝ : MeasureTheory.IsProbabilityMeasure ν
    h : Eq (ProbabilityTheory.cdf μ) (ProbabilityTheory.cdf ν)
    ⊢ Eq μ ν
  -/
  rw [← measure_cdf μ, ← measure_cdf ν, h]
  /-
    🎉 no goals
  -/


@[simp] lemma MeasureTheory.Measure.cdf_eq_iff (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] :
    cdf μ = cdf ν ↔ μ = ν :=
                                                 /-
                                                   μ ν : MeasureTheory.Measure Real
                                                   inst✝¹ : MeasureTheory.IsProbabilityMeasure μ
                                                   inst✝ : MeasureTheory.IsProbabilityMeasure ν
                                                   h : Eq μ ν
                                                   ⊢ Eq (ProbabilityTheory.cdf μ) (ProbabilityTheory.cdf ν)
                                                 -/
⟨MeasureTheory.Measure.eq_of_cdf μ ν, fun h ↦ by rw [h]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/

