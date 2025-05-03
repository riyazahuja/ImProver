/-- A Lebesgue Integral from -∞ to y can be expressed as the sum of one from -∞ to 0 and 0 to x -/
lemma lintegral_Iic_eq_lintegral_Iio_add_Icc {y z : ℝ} (f : ℝ → ℝ≥0∞) (hzy : z ≤ y) :
    ∫⁻ x in Iic y, f x = (∫⁻ x in Iio z, f x) + ∫⁻ x in Icc z y, f x := by
  /-
    y z : Real
    f : Real → ENNReal
    hzy : LE.le z y
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  rw [← Iio_union_Icc_eq_Iic hzy, lintegral_union measurableSet_Icc]
  /-
    y z : Real
    f : Real → ENNReal
    hzy : LE.le z y
    ⊢ Disjoint (Set.Iio z) (Set.Icc z y)
  -/
  simp_rw [Set.disjoint_iff_forall_ne, mem_Iio, mem_Icc]
  /-
    y z : Real
    f : Real → ENNReal
    hzy : LE.le z y
    ⊢ ∀ ⦃a : Real⦄, LT.lt a z → ∀ ⦃b : Real⦄, And (LE.le z b) (LE.le b y) → Ne a b
  -/
  intros
  /-
    y z : Real
    f : Real → ENNReal
    hzy : LE.le z y
    a✝² : Real
    a✝¹ : LT.lt a✝² z
    b✝ : Real
    a✝ : And (LE.le z b✝) (LE.le b✝ y)
    ⊢ Ne a✝² b✝
  -/
  linarith
  /-
    🎉 no goals
  -/


/-- The pdf of the gamma distribution depending on its scale and rate -/
noncomputable
def gammaPDFReal (a r x : ℝ) : ℝ :=
  if 0 ≤ x then r ^ a / (Gamma a) * x ^ (a-1) * exp (-(r * x)) else 0


/-- The pdf of the gamma distribution, as a function valued in `ℝ≥0∞` -/
noncomputable
def gammaPDF (a r x : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (gammaPDFReal a r x)


lemma gammaPDF_eq (a r x : ℝ) :
    gammaPDF a r x =
      ENNReal.ofReal (if 0 ≤ x then r ^ a / (Gamma a) * x ^ (a-1) * exp (-(r * x)) else 0) :=
  rfl


lemma gammaPDF_of_neg {a r x : ℝ} (hx : x < 0) : gammaPDF a r x = 0 := by
  /-
    a r x : Real
    hx : LT.lt x 0
    ⊢ Eq (ProbabilityTheory.gammaPDF a r x) 0
  -/
  simp only [gammaPDF_eq, if_neg (not_le.mpr hx), ENNReal.ofReal_zero]
  /-
    🎉 no goals
  -/


lemma gammaPDF_of_nonneg {a r x : ℝ} (hx : 0 ≤ x) :
    gammaPDF a r x = ENNReal.ofReal (r ^ a / (Gamma a) * x ^ (a-1) * exp (-(r * x))) := by
  /-
    a r x : Real
    hx : LE.le 0 x
    ⊢ Eq (ProbabilityTheory.gammaPDF a r x) (ENNReal.ofReal (HMul.hMul (HMul.hMul  …
  -/
  simp only [gammaPDF_eq, if_pos hx]
  /-
    🎉 no goals
  -/


/-- The Lebesgue integral of the gamma pdf over nonpositive reals equals 0 -/
lemma lintegral_gammaPDF_of_nonpos {x a r : ℝ} (hx : x ≤ 0) :
    ∫⁻ y in Iio x, gammaPDF a r y = 0 := by
  /-
    x a r : Real
    hx : LE.le x 0
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  rw [setLIntegral_congr_fun (g := fun _ ↦ 0) measurableSet_Iio]
    /-
      x a r : Real
      hx : LE.le x 0
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
    -/
  · rw [lintegral_zero, ← ENNReal.ofReal_zero]
    /-
      🎉 no goals
    -/
    /-
      x a r : Real
      hx : LE.le x 0
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Iio x) x_1 → Eq (Probabili …
    -/
  · simp only [gammaPDF_eq, ENNReal.ofReal_eq_zero]
    /-
      x a r : Real
      hx : LE.le x 0
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Iio x) x_1 → LE.le (ite (L …
    -/
    filter_upwards with a (_ : a < _)
    /-
      case h
      x a✝ r : Real
      hx : LE.le x 0
      a : Real
      x✝ : LT.lt a x
      ⊢ LE.le (ite (LE.le 0 a) (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.hPow r a✝) (Re …
    -/
    rw [if_neg (by linarith)]
    /-
      🎉 no goals
    -/


/-- The gamma pdf is measurable. -/
@[measurability]
lemma measurable_gammaPDFReal (a r : ℝ) : Measurable (gammaPDFReal a r) :=
  Measurable.ite measurableSet_Ici (((measurable_id'.pow_const _).const_mul _).mul
    (measurable_id'.const_mul _).neg.exp) measurable_const


/-- The gamma pdf is strongly measurable -/
@[measurability]
 lemma stronglyMeasurable_gammaPDFReal (a r : ℝ) :
     StronglyMeasurable (gammaPDFReal a r) :=
   (measurable_gammaPDFReal a r).stronglyMeasurable


/-- The gamma pdf is positive for all positive reals -/
lemma gammaPDFReal_pos {x a r : ℝ} (ha : 0 < a) (hr : 0 < r) (hx : 0 < x) :
    0 < gammaPDFReal a r x := by
  /-
    x a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (ProbabilityTheory.gammaPDFReal a r x)
  -/
  simp only [gammaPDFReal, if_pos hx.le]
  /-
    x a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    hx : LT.lt 0 x
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.hPow r a) (Real.Gamma a)) (HP …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- The gamma pdf is nonnegative -/
lemma gammaPDFReal_nonneg {a r : ℝ} (ha : 0 < a) (hr : 0 < r) (x : ℝ) :
    0 ≤ gammaPDFReal a r x := by
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    ⊢ LE.le 0 (ProbabilityTheory.gammaPDFReal a r x)
  -/
  unfold gammaPDFReal
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    ⊢ LE.le 0 (ite (LE.le 0 x) (HMul.hMul (HMul.hMul (HDiv.hDiv (HPow.hPow r a) (R …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> positivity
                /-
                  🎉 no goals
                -/


/-- The pdf of the gamma distribution integrates to 1 -/
@[simp]
lemma lintegral_gammaPDF_eq_one {a r : ℝ} (ha : 0 < a) (hr : 0 < r) :
    ∫⁻ x, gammaPDF a r x = 1 := by
  have leftSide : ∫⁻ x in Iio 0, gammaPDF a r x = 0 := by
    rw [setLIntegral_congr_fun measurableSet_Iio
      (ae_of_all _ (fun x (hx : x < 0) ↦ gammaPDF_of_neg hx)), lintegral_zero]
  have rightSide : ∫⁻ x in Ici 0, gammaPDF a r x =
      ∫⁻ x in Ici 0, ENNReal.ofReal (r ^ a / Gamma a * x ^ (a - 1) * exp (-(r * x))) :=
    setLIntegral_congr_fun measurableSet_Ici (ae_of_all _ (fun _ ↦ gammaPDF_of_nonneg))
  rw [← ENNReal.toReal_eq_one_iff, ← lintegral_add_compl _ measurableSet_Ici, compl_Ici,
    leftSide, rightSide, add_zero, ← integral_eq_lintegral_of_nonneg_ae]
    /-
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
  · simp_rw [integral_Ici_eq_integral_Ioi, mul_assoc]
    rw [integral_mul_left, integral_rpow_mul_exp_neg_mul_Ioi ha hr, div_mul_eq_mul_div,
      ← mul_assoc, mul_div_assoc, div_self (Gamma_pos_of_pos ha).ne', mul_one,
      div_rpow zero_le_one hr.le, one_rpow, mul_one_div, div_self (rpow_pos_of_pos hr _).ne']
    /-
      case hf
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ici 0))). …
    -/
  · rw [EventuallyLE, ae_restrict_iff' measurableSet_Ici]
    /-
      case hf
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ici 0) x → LE.le (0 x) (HMul …
    -/
    exact ae_of_all _ (fun x (hx : 0 ≤ x) ↦ by positivity)
    /-
      🎉 no goals
    -/
    /-
      case hfm
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HMul.hMul (HMul.hMul (HDiv.hDiv …
    -/
  · apply (measurable_gammaPDFReal a r).aestronglyMeasurable.congr
    /-
      case hfm
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ici 0))). …
    -/
    refine (ae_restrict_iff' measurableSet_Ici).mpr <| ae_of_all _ fun x (hx : 0 ≤ x) ↦ ?_
    /-
      case hfm
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      x : Real
      hx : LE.le 0 x
      ⊢ Eq (ProbabilityTheory.gammaPDFReal a r x) ((fun x => HMul.hMul (HMul.hMul (H …
    -/
    simp_rw [gammaPDFReal, eq_true_intro hx, ite_true]
    /-
      🎉 no goals
    -/


/-- Measure defined by the gamma distribution -/
noncomputable
def gammaMeasure (a r : ℝ) : Measure ℝ :=
  volume.withDensity (gammaPDF a r)


lemma isProbabilityMeasureGamma {a r : ℝ} (ha : 0 < a) (hr : 0 < r) :
    IsProbabilityMeasure (gammaMeasure a r) where
                     /-
                       a r : Real
                       ha : LT.lt 0 a
                       hr : LT.lt 0 r
                       ⊢ Eq ((ProbabilityTheory.gammaMeasure a r) Set.univ) 1
                     -/
  measure_univ := by simp [gammaMeasure, lintegral_gammaPDF_eq_one ha hr]
                     /-
                       🎉 no goals
                     -/


/-- CDF of the gamma distribution -/
noncomputable
def gammaCDFReal (a r : ℝ) : StieltjesFunction :=
  cdf (gammaMeasure a r)


lemma gammaCDFReal_eq_integral {a r : ℝ} (ha : 0 < a) (hr : 0 < r) (x : ℝ) :
    gammaCDFReal a r x = ∫ x in Iic x, gammaPDFReal a r x := by
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    ⊢ Eq (↑(ProbabilityTheory.gammaCDFReal a r) x) (MeasureTheory.integral (Measur …
  -/
  have : IsProbabilityMeasure (gammaMeasure a r) := isProbabilityMeasureGamma ha hr
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.gammaMeasure a r)
    ⊢ Eq (↑(ProbabilityTheory.gammaCDFReal a r) x) (MeasureTheory.integral (Measur …
  -/
  rw [gammaCDFReal, cdf_eq_toReal, gammaMeasure, withDensity_apply _ measurableSet_Iic]
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.gammaMeasure a r)
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  refine (integral_eq_lintegral_of_nonneg_ae ?_ ?_).symm
    /-
      case refine_1
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      x : Real
      this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.gammaMeasure a r)
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Iic x))). …
    -/
  · exact ae_of_all _ fun b ↦ by simp only [Pi.zero_apply, gammaPDFReal_nonneg ha hr]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a r : Real
      ha : LT.lt 0 a
      hr : LT.lt 0 r
      x : Real
      this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.gammaMeasure a r)
      ⊢ MeasureTheory.AEStronglyMeasurable (ProbabilityTheory.gammaPDFReal a r) (Mea …
    -/
  · exact (measurable_gammaPDFReal a r).aestronglyMeasurable.restrict
    /-
      🎉 no goals
    -/


lemma gammaCDFReal_eq_lintegral {a r : ℝ} (ha : 0 < a) (hr : 0 < r) (x : ℝ) :
    gammaCDFReal a r x = ENNReal.toReal (∫⁻ x in Iic x, gammaPDF a r x) := by
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    ⊢ Eq (↑(ProbabilityTheory.gammaCDFReal a r) x) (MeasureTheory.lintegral (Measu …
  -/
  have : IsProbabilityMeasure (gammaMeasure a r) := isProbabilityMeasureGamma ha hr
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.gammaMeasure a r)
    ⊢ Eq (↑(ProbabilityTheory.gammaCDFReal a r) x) (MeasureTheory.lintegral (Measu …
  -/
  simp only [gammaPDF, gammaCDFReal, cdf_eq_toReal]
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.gammaMeasure a r)
    ⊢ Eq ((ProbabilityTheory.gammaMeasure a r) (Set.Iic x)).toReal (MeasureTheory. …
  -/
  simp only [gammaMeasure, measurableSet_Iic, withDensity_apply, gammaPDF]
  /-
    🎉 no goals
  -/


