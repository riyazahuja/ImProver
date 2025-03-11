/-- The pdf of the Pareto distribution depending on its scale `t` and rate `r`. -/
noncomputable def paretoPDFReal (t r x : ℝ) : ℝ :=
  if t ≤ x then r * t ^ r * x ^ (-(r + 1)) else 0


/-- The pdf of the Pareto distribution, as a function valued in `ℝ≥0∞`. -/
noncomputable def paretoPDF (t r x : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (paretoPDFReal t r x)


lemma paretoPDF_eq (t r x : ℝ) :
    paretoPDF t r x = ENNReal.ofReal (if t ≤ x then r * t ^ r * x ^ (-(r + 1)) else 0) := rfl


lemma paretoPDF_of_lt (hx : x < t) : paretoPDF t r x = 0 := by
  /-
    t r x : Real
    hx : LT.lt x t
    ⊢ Eq (ProbabilityTheory.paretoPDF t r x) 0
  -/
  simp only [paretoPDF_eq, if_neg (not_le.mpr hx), ENNReal.ofReal_zero]
  /-
    🎉 no goals
  -/


lemma paretoPDF_of_le (hx : t ≤ x) :
    paretoPDF t r x = ENNReal.ofReal (r * t ^ r * x ^ (-(r + 1))) := by
  /-
    t r x : Real
    hx : LE.le t x
    ⊢ Eq (ProbabilityTheory.paretoPDF t r x) (ENNReal.ofReal (HMul.hMul (HMul.hMul …
  -/
  simp only [paretoPDF_eq, if_pos hx]
  /-
    🎉 no goals
  -/


/-- The Lebesgue integral of the Pareto pdf over reals `≤ t` equals `0`. -/
lemma lintegral_paretoPDF_of_le (hx : x ≤ t) :
    ∫⁻ y in Iio x, paretoPDF t r y = 0 := by
  /-
    t r x : Real
    hx : LE.le x t
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  rw [setLIntegral_congr_fun (g := fun _ ↦ 0) measurableSet_Iio]
    /-
      t r x : Real
      hx : LE.le x t
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
    -/
  · rw [lintegral_zero, ← ENNReal.ofReal_zero]
    /-
      🎉 no goals
    -/
    /-
      t r x : Real
      hx : LE.le x t
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Iio x) x_1 → Eq (Probabili …
    -/
  · simp only [paretoPDF_eq, ge_iff_le, ENNReal.ofReal_eq_zero]
    /-
      t r x : Real
      hx : LE.le x t
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Iio x) x_1 → LE.le (ite (L …
    -/
    filter_upwards with a (_ : a < _)
    /-
      case h
      t r x : Real
      hx : LE.le x t
      a : Real
      x✝ : LT.lt a x
      ⊢ LE.le (ite (LE.le t a) (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow a …
    -/
    rw [if_neg (by linarith)]
    /-
      🎉 no goals
    -/


/-- The Pareto pdf is measurable. -/
@[measurability, fun_prop]
lemma measurable_paretoPDFReal (t r : ℝ) : Measurable (paretoPDFReal t r) :=
  Measurable.ite measurableSet_Ici ((measurable_id.pow_const _).const_mul _) measurable_const


/-- The Pareto pdf is strongly measurable. -/
@[measurability]
lemma stronglyMeasurable_paretoPDFReal (t r : ℝ) :
    StronglyMeasurable (paretoPDFReal t r) :=
  (measurable_paretoPDFReal t r).stronglyMeasurable


/-- The Pareto pdf is positive for all reals `>= t`. -/
lemma paretoPDFReal_pos (ht : 0 < t) (hr : 0 < r) (hx : t ≤ x) :
    0 < paretoPDFReal t r x := by
  /-
    t r x : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    hx : LE.le t x
    ⊢ LT.lt 0 (ProbabilityTheory.paretoPDFReal t r x)
  -/
  rw [paretoPDFReal, if_pos hx]
  /-
    t r x : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    hx : LE.le t x
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow x (Neg.neg (HAdd …
  -/
  have _ : 0 < x := by linarith
  /-
    t r x : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    hx : LE.le t x
    x✝ : LT.lt 0 x
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow x (Neg.neg (HAdd …
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- The Pareto pdf is nonnegative. -/
lemma paretoPDFReal_nonneg (ht : 0 ≤ t) (hr : 0 ≤ r) (x : ℝ) :
    0 ≤ paretoPDFReal t r x := by
  /-
    t r : Real
    ht : LE.le 0 t
    hr : LE.le 0 r
    x : Real
    ⊢ LE.le 0 (ProbabilityTheory.paretoPDFReal t r x)
  -/
  unfold paretoPDFReal
  /-
    t r : Real
    ht : LE.le 0 t
    hr : LE.le 0 r
    x : Real
    ⊢ LE.le 0 (ite (LE.le t x) (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow …
  -/
  split_ifs with h
  · cases le_iff_eq_or_lt.1 ht with
    | inl ht0 =>
      rw [← ht0] at h
      positivity
    | inr htp =>
      have := lt_of_lt_of_le htp h
      positivity
    /-
      case neg
      t r : Real
      ht : LE.le 0 t
      hr : LE.le 0 r
      x : Real
      h : Not (LE.le t x)
      ⊢ LE.le 0 0
    -/
  · positivity
    /-
      🎉 no goals
    -/


/-- The pdf of the Pareto distribution integrates to `1`. -/
@[simp]
lemma lintegral_paretoPDF_eq_one (ht : 0 < t) (hr : 0 < r) :
    ∫⁻ x, paretoPDF t r x = 1 := by
  /-
    t r : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    ⊢ Eq (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun x => Proba …
  -/
  have leftSide : ∫⁻ x in Iio t, paretoPDF t r x = 0 := lintegral_paretoPDF_of_le (le_refl t)
  have rightSide : ∫⁻ x in Ici t, paretoPDF t r x =
      ∫⁻ x in Ici t, ENNReal.ofReal (r * t ^ r * x ^ (-(r + 1))) :=
    setLIntegral_congr_fun measurableSet_Ici (ae_of_all _ (fun _ ↦ paretoPDF_of_le))
  rw [← ENNReal.toReal_eq_one_iff, ← lintegral_add_compl _ measurableSet_Ici, compl_Ici,
    leftSide, rightSide, add_zero, ← integral_eq_lintegral_of_nonneg_ae]
    /-
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
  · rw [integral_Ici_eq_integral_Ioi, integral_mul_left, integral_Ioi_rpow_of_lt _ ht]
      /-
        t r : Real
        ht : LT.lt 0 t
        hr : LT.lt 0 r
        leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
        rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
        ⊢ Eq (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HDiv.hDiv (Neg.neg (HPow.hPow t …
      -/
    · field_simp [hr]
      /-
        t r : Real
        ht : LT.lt 0 t
        hr : LT.lt 0 r
        leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
        rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
        ⊢ Eq (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow t (Neg.neg r))) r
      -/
      rw [mul_assoc, ← rpow_add ht]
      /-
        t r : Real
        ht : LT.lt 0 t
        hr : LT.lt 0 r
        leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
        rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
        ⊢ Eq (HMul.hMul r (HPow.hPow t (HAdd.hAdd r (Neg.neg r)))) r
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ LT.lt (Neg.neg (HAdd.hAdd r 1)) (-1)
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case hf
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ici t))). …
    -/
  · rw [EventuallyLE, ae_restrict_iff' measurableSet_Ici]
    /-
      case hf
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ici t) x → LE.le (0 x) (HMul …
    -/
    refine ae_of_all _ fun x (hx : t ≤ x) ↦ ?_
    /-
      case hf
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      x : Real
      hx : LE.le t x
      ⊢ LE.le (0 x) (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow x (Neg.neg ( …
    -/
    have := lt_of_lt_of_le ht hx
    /-
      case hf
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      x : Real
      hx : LE.le t x
      this : LT.lt 0 x
      ⊢ LE.le (0 x) (HMul.hMul (HMul.hMul r (HPow.hPow t r)) (HPow.hPow x (Neg.neg ( …
    -/
    positivity
    /-
      🎉 no goals
    -/
    /-
      case hfm
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HMul.hMul (HMul.hMul r (HPow.hP …
    -/
  · apply (measurable_paretoPDFReal t r).aestronglyMeasurable.congr
    /-
      case hfm
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ici t))). …
    -/
    refine (ae_restrict_iff' measurableSet_Ici).mpr <| ae_of_all _ fun x (hx : t ≤ x) ↦ ?_
    /-
      case hfm
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      leftSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.rest …
      rightSide : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.res …
      x : Real
      hx : LE.le t x
      ⊢ Eq (ProbabilityTheory.paretoPDFReal t r x) ((fun x => HMul.hMul (HMul.hMul r …
    -/
    simp_rw [paretoPDFReal, eq_true_intro hx, ite_true]
    /-
      🎉 no goals
    -/


/-- Measure defined by the Pareto distribution. -/
noncomputable def paretoMeasure (t r : ℝ) : Measure ℝ :=
  volume.withDensity (paretoPDF t r)


lemma isProbabilityMeasure_paretoMeasure (ht : 0 < t) (hr : 0 < r) :
    IsProbabilityMeasure (paretoMeasure t r) where
                     /-
                       t r : Real
                       ht : LT.lt 0 t
                       hr : LT.lt 0 r
                       ⊢ Eq ((ProbabilityTheory.paretoMeasure t r) Set.univ) 1
                     -/
  measure_univ := by simp [paretoMeasure, lintegral_paretoPDF_eq_one ht hr]
                     /-
                       🎉 no goals
                     -/


/-- CDF of the Pareto distribution equals the integral of the PDF. -/
lemma paretoCDFReal_eq_integral (ht : 0 < t) (hr : 0 < r) (x : ℝ) :
    cdf (paretoMeasure t r) x = ∫ x in Iic x, paretoPDFReal t r x := by
  /-
    t r : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    x : Real
    ⊢ Eq (↑(ProbabilityTheory.cdf (ProbabilityTheory.paretoMeasure t r)) x) (Measu …
  -/
  have : IsProbabilityMeasure (paretoMeasure t r) := isProbabilityMeasure_paretoMeasure ht hr
  /-
    t r : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.paretoMeasure t r)
    ⊢ Eq (↑(ProbabilityTheory.cdf (ProbabilityTheory.paretoMeasure t r)) x) (Measu …
  -/
  rw [cdf_eq_toReal, paretoMeasure, withDensity_apply _ measurableSet_Iic]
  /-
    t r : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.paretoMeasure t r)
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  refine (integral_eq_lintegral_of_nonneg_ae ?_ ?_).symm
    /-
      case refine_1
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      x : Real
      this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.paretoMeasure t r)
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Iic x))). …
    -/
  · exact ae_of_all _ fun _ ↦ by simp only [Pi.zero_apply, paretoPDFReal_nonneg ht.le hr.le]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      t r : Real
      ht : LT.lt 0 t
      hr : LT.lt 0 r
      x : Real
      this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.paretoMeasure t r)
      ⊢ MeasureTheory.AEStronglyMeasurable (ProbabilityTheory.paretoPDFReal t r) (Me …
    -/
  · exact (measurable_paretoPDFReal t r).aestronglyMeasurable.restrict
    /-
      🎉 no goals
    -/


lemma paretoCDFReal_eq_lintegral (ht : 0 < t) (hr : 0 < r) (x : ℝ) :
    cdf (paretoMeasure t r) x = ENNReal.toReal (∫⁻ x in Iic x, paretoPDF t r x) := by
  /-
    t r : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    x : Real
    ⊢ Eq (↑(ProbabilityTheory.cdf (ProbabilityTheory.paretoMeasure t r)) x) (Measu …
  -/
  have : IsProbabilityMeasure (paretoMeasure t r) := isProbabilityMeasure_paretoMeasure ht hr
  /-
    t r : Real
    ht : LT.lt 0 t
    hr : LT.lt 0 r
    x : Real
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.paretoMeasure t r)
    ⊢ Eq (↑(ProbabilityTheory.cdf (ProbabilityTheory.paretoMeasure t r)) x) (Measu …
  -/
  rw [cdf_eq_toReal, paretoMeasure, withDensity_apply _ measurableSet_Iic]
  /-
    🎉 no goals
  -/


