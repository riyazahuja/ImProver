/-- The pdf of the exponential distribution depending on its rate -/
noncomputable
def exponentialPDFReal (r x : ℝ) : ℝ :=
  gammaPDFReal 1 r x


/-- The pdf of the exponential distribution, as a function valued in `ℝ≥0∞` -/
noncomputable
def exponentialPDF (r x : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (exponentialPDFReal r x)


lemma exponentialPDF_eq (r x : ℝ) :
    exponentialPDF r x = ENNReal.ofReal (if 0 ≤ x then r * exp (-(r * x)) else 0) := by
  /-
    r x : Real
    ⊢ Eq (ProbabilityTheory.exponentialPDF r x) (ENNReal.ofReal (ite (LE.le 0 x) ( …
  -/
  rw [exponentialPDF, exponentialPDFReal, gammaPDFReal]
  /-
    r x : Real
    ⊢ Eq (ENNReal.ofReal (ite (LE.le 0 x) (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.h …
  -/
  simp only [rpow_one, Gamma_one, div_one, sub_self, rpow_zero, mul_one]
  /-
    🎉 no goals
  -/


lemma exponentialPDF_of_neg {r x : ℝ} (hx : x < 0) : exponentialPDF r x = 0 := gammaPDF_of_neg hx


lemma exponentialPDF_of_nonneg {r x : ℝ} (hx : 0 ≤ x) :
    exponentialPDF r x = ENNReal.ofReal (r * rexp (-(r * x))) := by
  /-
    r x : Real
    hx : LE.le 0 x
    ⊢ Eq (ProbabilityTheory.exponentialPDF r x) (ENNReal.ofReal (HMul.hMul r (Real …
  -/
  simp only [exponentialPDF_eq, if_pos hx]
  /-
    🎉 no goals
  -/


/-- The Lebesgue integral of the exponential pdf over nonpositive reals equals 0-/
lemma lintegral_exponentialPDF_of_nonpos {x r : ℝ} (hx : x ≤ 0) :
    ∫⁻ y in Iio x, exponentialPDF r y = 0 := lintegral_gammaPDF_of_nonpos hx


/-- The exponential pdf is measurable. -/
@[measurability]
lemma measurable_exponentialPDFReal (r : ℝ) : Measurable (exponentialPDFReal r) :=
  measurable_gammaPDFReal 1 r

-- The exponential pdf is strongly measurable -/

@[measurability]
 lemma stronglyMeasurable_exponentialPDFReal (r : ℝ) :
     StronglyMeasurable (exponentialPDFReal r) := stronglyMeasurable_gammaPDFReal 1 r


/-- The exponential pdf is positive for all positive reals -/
lemma exponentialPDFReal_pos {x r : ℝ} (hr : 0 < r) (hx : 0 < x) :
    0 < exponentialPDFReal r x := gammaPDFReal_pos zero_lt_one hr hx


/-- The exponential pdf is nonnegative -/
lemma exponentialPDFReal_nonneg {r : ℝ} (hr : 0 < r) (x : ℝ) :
    0 ≤ exponentialPDFReal r x := gammaPDFReal_nonneg zero_lt_one hr x


/-- The pdf of the exponential distribution integrates to 1 -/
@[simp]
lemma lintegral_exponentialPDF_eq_one {r : ℝ} (hr : 0 < r) : ∫⁻ x, exponentialPDF r x = 1 :=
  lintegral_gammaPDF_eq_one zero_lt_one hr


/-- Measure defined by the exponential distribution -/
noncomputable
def expMeasure (r : ℝ) : Measure ℝ := gammaMeasure 1 r


lemma isProbabilityMeasureExponential {r : ℝ} (hr : 0 < r) :
    IsProbabilityMeasure (expMeasure r) := isProbabilityMeasureGamma zero_lt_one hr


/-- CDF of the exponential distribution -/
noncomputable
def exponentialCDFReal (r : ℝ) : StieltjesFunction :=
  cdf (expMeasure r)


lemma exponentialCDFReal_eq_integral {r : ℝ} (hr : 0 < r) (x : ℝ) :
    exponentialCDFReal r x = ∫ x in Iic x, exponentialPDFReal r x :=
  gammaCDFReal_eq_integral zero_lt_one hr x


lemma exponentialCDFReal_eq_lintegral {r : ℝ} (hr : 0 < r) (x : ℝ) :
    exponentialCDFReal r x = ENNReal.toReal (∫⁻ x in Iic x, exponentialPDF r x) :=
  gammaCDFReal_eq_lintegral zero_lt_one hr x


lemma hasDerivAt_neg_exp_mul_exp {r x : ℝ} :
    HasDerivAt (fun a ↦ -exp (-(r * a))) (r * exp (-(r * x))) x := by
  /-
    r x : Real
    ⊢ HasDerivAt (fun a => Neg.neg (Real.exp (Neg.neg (HMul.hMul r a)))) (HMul.hMu …
  -/
  convert (((hasDerivAt_id x).const_mul (-r)).exp.const_mul (-1)) using 1
    /-
      case h.e'_8
      r x : Real
      ⊢ Eq (fun a => Neg.neg (Real.exp (Neg.neg (HMul.hMul r a)))) fun y => HMul.hMu …
    -/
  · simp only [one_mul, id_eq, neg_mul]
    /-
      🎉 no goals
    -/
  /-
    case h.e'_9
    r x : Real
    ⊢ Eq (HMul.hMul r (Real.exp (Neg.neg (HMul.hMul r x)))) (HMul.hMul (-1) (HMul. …
  -/
  simp only [id_eq, neg_mul, mul_one, mul_neg, one_mul, neg_neg, mul_comm]
  /-
    🎉 no goals
  -/


/-- A negative exponential function is integrable on intervals in `R≥0` -/
lemma exp_neg_integrableOn_Ioc {b x : ℝ} (hb : 0 < b) :
    /-
      b x : Real
      hb : LT.lt 0 b
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x ↦ rexp (-(b * x))) (Ioc 0 x) := by
    /-
      🎉 no goals
    -/
  /-
    b x : Real
    hb : LT.lt 0 b
    ⊢ MeasureTheory.IntegrableOn (fun x => Real.exp (Neg.neg (HMul.hMul b x))) (Se …
  -/
  simp only [neg_mul_eq_neg_mul]
  /-
    b x : Real
    hb : LT.lt 0 b
    ⊢ MeasureTheory.IntegrableOn (fun x => Real.exp (HMul.hMul (Neg.neg b) x)) (Se …
  -/
  exact (exp_neg_integrableOn_Ioi _ hb).mono_set Ioc_subset_Ioi_self
  /-
    🎉 no goals
  -/


lemma lintegral_exponentialPDF_eq_antiDeriv {r : ℝ} (hr : 0 < r) (x : ℝ) :
    ∫⁻ y in Iic x, exponentialPDF r y
    = ENNReal.ofReal (if 0 ≤ x then 1 - exp (-(r * x)) else 0) := by
  /-
    r : Real
    hr : LT.lt 0 r
    x : Real
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  split_ifs with h
  case neg =>
    simp only [exponentialPDF_eq]
    rw [setLIntegral_congr_fun measurableSet_Iic, lintegral_zero, ENNReal.ofReal_zero]
    exact ae_of_all _ fun a (_ : a ≤ _) ↦ by rw [if_neg (by linarith), ENNReal.ofReal_eq_zero]
  case pos =>
    rw [lintegral_Iic_eq_lintegral_Iio_add_Icc _ h, lintegral_exponentialPDF_of_nonpos (le_refl 0),
      zero_add]
    simp only [exponentialPDF_eq]
    rw [setLIntegral_congr_fun measurableSet_Icc (ae_of_all _
        (by intro a ⟨(hle : _ ≤ a), _⟩; rw [if_pos hle]))]
    rw [← ENNReal.toReal_eq_toReal _ ENNReal.ofReal_ne_top, ← integral_eq_lintegral_of_nonneg_ae
        (Eventually.of_forall fun _ ↦ le_of_lt (mul_pos hr (exp_pos _)))]
    · have : ∫ a in uIoc 0 x, r * rexp (-(r * a)) = ∫ a in (0)..x, r * rexp (-(r * a)) := by
        rw [intervalIntegral.intervalIntegral_eq_integral_uIoc, smul_eq_mul, if_pos h, one_mul]
      rw [integral_Icc_eq_integral_Ioc, ← uIoc_of_le h, this]
      rw [intervalIntegral.integral_eq_sub_of_hasDeriv_right_of_le h
        (f := fun a ↦ -1 * rexp (-(r * a))) _ _]
      · rw [ENNReal.toReal_ofReal_eq_iff.2 (by norm_num; positivity)]
        norm_num; ring
      · simp only [intervalIntegrable_iff, uIoc_of_le h]
        exact Integrable.const_mul (exp_neg_integrableOn_Ioc hr) _
      · have : Continuous (fun a ↦ rexp (-(r * a))) := by
          simp only [← neg_mul]; exact (continuous_mul_left (-r)).rexp
        exact Continuous.continuousOn (Continuous.comp' (continuous_mul_left (-1)) this)
      · simp only [neg_mul, one_mul]
        exact fun _ _ ↦ HasDerivAt.hasDerivWithinAt hasDerivAt_neg_exp_mul_exp
    · refine Integrable.aestronglyMeasurable (Integrable.const_mul ?_ _)
      rw [← IntegrableOn, integrableOn_Icc_iff_integrableOn_Ioc]
      exact exp_neg_integrableOn_Ioc hr
    · refine ne_of_lt (IntegrableOn.setLIntegral_lt_top ?_)
      rw [integrableOn_Icc_iff_integrableOn_Ioc]
      exact Integrable.const_mul (exp_neg_integrableOn_Ioc hr) _


/-- The CDF of the exponential distribution equals ``1 - exp (-(r * x))``-/
lemma exponentialCDFReal_eq {r : ℝ} (hr : 0 < r) (x : ℝ) :
    exponentialCDFReal r x = if 0 ≤ x then 1 - exp (-(r * x)) else 0 := by
  rw [exponentialCDFReal_eq_lintegral hr, lintegral_exponentialPDF_eq_antiDeriv hr x,
    ENNReal.toReal_ofReal_eq_iff]
  /-
    r : Real
    hr : LT.lt 0 r
    x : Real
    ⊢ LE.le 0 (ite (LE.le 0 x) (HSub.hSub 1 (Real.exp (Neg.neg (HMul.hMul r x)))) 0)
  -/
  split_ifs with h
    /-
      case pos
      r : Real
      hr : LT.lt 0 r
      x : Real
      h : LE.le 0 x
      ⊢ LE.le 0 (HSub.hSub 1 (Real.exp (Neg.neg (HMul.hMul r x))))
    -/
  · simp only [sub_nonneg, exp_le_one_iff, Left.neg_nonpos_iff]
    /-
      case pos
      r : Real
      hr : LT.lt 0 r
      x : Real
      h : LE.le 0 x
      ⊢ LE.le 0 (HMul.hMul r x)
    -/
    exact mul_nonneg hr.le h
    /-
      🎉 no goals
    -/
    /-
      case neg
      r : Real
      hr : LT.lt 0 r
      x : Real
      h : Not (LE.le 0 x)
      ⊢ LE.le 0 0
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


