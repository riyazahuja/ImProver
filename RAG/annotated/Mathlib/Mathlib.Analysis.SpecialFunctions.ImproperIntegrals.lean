                                       /-
                                         c : Real
                                         ⊢ MeasureTheory.Measure Real
                                       -/
theorem integrableOn_exp_Iic (c : ℝ) : IntegrableOn exp (Iic c) := by
                                       /-
                                         🎉 no goals
                                       -/
  refine
    integrableOn_Iic_of_intervalIntegral_norm_bounded (exp c) c
      (fun y => intervalIntegrable_exp.1) tendsto_id
      (eventually_of_mem (Iic_mem_atBot 0) fun y _ => ?_)
  /-
    c y : Real
    x✝ : Membership.mem (Set.Iic 0) y
    ⊢ LE.le (intervalIntegral (fun x => Norm.norm (Real.exp x)) (id y) c MeasureTh …
  -/
  simp_rw [norm_of_nonneg (exp_pos _).le, integral_exp, sub_le_self_iff]
  /-
    c y : Real
    x✝ : Membership.mem (Set.Iic 0) y
    ⊢ LE.le 0 (Real.exp (id y))
  -/
  exact (exp_pos _).le
  /-
    🎉 no goals
  -/


theorem integral_exp_Iic (c : ℝ) : ∫ x : ℝ in Iic c, exp x = exp c := by
  refine
    tendsto_nhds_unique
      (intervalIntegral_tendsto_integral_Iic _ (integrableOn_exp_Iic _) tendsto_id) ?_
  /-
    c : Real
    ⊢ Filter.Tendsto (fun i => intervalIntegral (fun x => Real.exp x) (id i) c Mea …
  -/
  simp_rw [integral_exp, show 𝓝 (exp c) = 𝓝 (exp c - 0) by rw [sub_zero]]
  /-
    c : Real
    ⊢ Filter.Tendsto (fun i => HSub.hSub (Real.exp c) (Real.exp (id i))) Filter.at …
  -/
  exact tendsto_exp_atBot.const_sub _
  /-
    🎉 no goals
  -/


theorem integral_exp_Iic_zero : ∫ x : ℝ in Iic 0, exp x = 1 :=
  exp_zero ▸ integral_exp_Iic 0


theorem integral_exp_neg_Ioi (c : ℝ) : (∫ x : ℝ in Ioi c, exp (-x)) = exp (-c) := by
  /-
    c : Real
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simpa only [integral_comp_neg_Ioi] using integral_exp_Iic (-c)
  /-
    🎉 no goals
  -/


theorem integral_exp_neg_Ioi_zero : (∫ x : ℝ in Ioi 0, exp (-x)) = 1 := by
  /-
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simpa only [neg_zero, exp_zero] using integral_exp_neg_Ioi 0
  /-
    🎉 no goals
  -/


/-- If `0 < c`, then `(fun t : ℝ ↦ t ^ a)` is integrable on `(c, ∞)` for all `a < -1`. -/
theorem integrableOn_Ioi_rpow_of_lt {a : ℝ} (ha : a < -1) {c : ℝ} (hc : 0 < c) :
    /-
      a : Real
      ha : LT.lt a (-1)
      c : Real
      hc : LT.lt 0 c
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun t : ℝ => t ^ a) (Ioi c) := by
    /-
      🎉 no goals
    -/
  have hd : ∀ x ∈ Ici c, HasDerivAt (fun t => t ^ (a + 1) / (a + 1)) (x ^ a) x := by
    intro x hx
    -- Porting note: helped `convert` with explicit arguments
    convert (hasDerivAt_rpow_const (p := a + 1) (Or.inl (hc.trans_le hx).ne')).div_const _ using 1
    field_simp [show a + 1 ≠ 0 from ne_of_lt (by linarith), mul_comm]
  have ht : Tendsto (fun t => t ^ (a + 1) / (a + 1)) atTop (𝓝 (0 / (a + 1))) := by
    apply Tendsto.div_const
    simpa only [neg_neg] using tendsto_rpow_neg_atTop (by linarith : 0 < -(a + 1))
  exact
    integrableOn_Ioi_deriv_of_nonneg' hd (fun t ht => rpow_nonneg (hc.trans ht).le a) ht


theorem integrableOn_Ioi_rpow_iff {s t : ℝ} (ht : 0 < t) :
    /-
      s t : Real
      ht : LT.lt 0 t
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x ↦ x ^ s) (Ioi t) ↔ s < -1 := by
    /-
      🎉 no goals
    -/
  /-
    s t : Real
    ht : LT.lt 0 t
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi t) Measure …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ integrableOn_Ioi_rpow_of_lt h ht⟩
  /-
    s t : Real
    ht : LT.lt 0 t
    h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi t) MeasureThe …
    ⊢ LT.lt s (-1)
  -/
  contrapose! h
  /-
    s t : Real
    ht : LT.lt 0 t
    h : LE.le (-1) s
    ⊢ Not (MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi t) Measure …
  -/
  intro H
  have H' : IntegrableOn (fun x ↦ x ^ s) (Ioi (max 1 t)) :=
    H.mono (Set.Ioi_subset_Ioi (le_max_right _ _)) le_rfl
  have : IntegrableOn (fun x ↦ x⁻¹) (Ioi (max 1 t)) := by
    apply H'.mono' measurable_inv.aestronglyMeasurable
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx
    have x_one : 1 ≤ x := ((le_max_left _ _).trans_lt (mem_Ioi.1 hx)).le
    simp only [norm_inv, Real.norm_eq_abs, abs_of_nonneg (zero_le_one.trans x_one)]
    rw [← Real.rpow_neg_one x]
    exact Real.rpow_le_rpow_of_exponent_le x_one h
  /-
    s t : Real
    ht : LT.lt 0 t
    h : LE.le (-1) s
    H : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi t) MeasureThe …
    H' : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi (Max.max 1 t …
    this : MeasureTheory.IntegrableOn (fun x => Inv.inv x) (Set.Ioi (Max.max 1 t)) …
    ⊢ False
  -/
  exact not_IntegrableOn_Ioi_inv this
  /-
    🎉 no goals
  -/


/-- The real power function with any exponent is not integrable on `(0, +∞)`. -/
                                              /-
                                                s : Real
                                                ⊢ MeasureTheory.Measure Real
                                              -/
theorem not_integrableOn_Ioi_rpow (s : ℝ) : ¬ IntegrableOn (fun x ↦ x ^ s) (Ioi (0 : ℝ)) := by
                                              /-
                                                🎉 no goals
                                              -/
  /-
    s : Real
    ⊢ Not (MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) Measure …
  -/
  intro h
  /-
    s : Real
    h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
    ⊢ False
  -/
  rcases le_or_lt s (-1) with hs|hs
    /-
      case inl
      s : Real
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
      hs : LE.le s (-1)
      ⊢ False
    -/
  · have : IntegrableOn (fun x ↦ x ^ s) (Ioo (0 : ℝ) 1) := h.mono Ioo_subset_Ioi_self le_rfl
    /-
      case inl
      s : Real
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
      hs : LE.le s (-1)
      this : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioo 0 1) Measu …
      ⊢ False
    -/
    rw [integrableOn_Ioo_rpow_iff zero_lt_one] at this
    /-
      case inl
      s : Real
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
      hs : LE.le s (-1)
      this : LT.lt (-1) s
      ⊢ False
    -/
    exact hs.not_lt this
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Real
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
      hs : LT.lt (-1) s
      ⊢ False
    -/
  · have : IntegrableOn (fun x ↦ x ^ s) (Ioi (1 : ℝ)) := h.mono (Ioi_subset_Ioi zero_le_one) le_rfl
    /-
      case inr
      s : Real
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
      hs : LT.lt (-1) s
      this : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 1) Measure …
      ⊢ False
    -/
    rw [integrableOn_Ioi_rpow_iff zero_lt_one] at this
    /-
      case inr
      s : Real
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioi 0) MeasureThe …
      hs : LT.lt (-1) s
      this : LT.lt s (-1)
      ⊢ False
    -/
    exact hs.not_lt this
    /-
      🎉 no goals
    -/


theorem setIntegral_Ioi_zero_rpow (s : ℝ) : ∫ x in Ioi (0 : ℝ), x ^ s = 0 :=
  MeasureTheory.integral_undef (not_integrableOn_Ioi_rpow s)


theorem integral_Ioi_rpow_of_lt {a : ℝ} (ha : a < -1) {c : ℝ} (hc : 0 < c) :
    ∫ t : ℝ in Ioi c, t ^ a = -c ^ (a + 1) / (a + 1) := by
  have hd : ∀ x ∈ Ici c, HasDerivAt (fun t => t ^ (a + 1) / (a + 1)) (x ^ a) x := by
    intro x hx
    convert (hasDerivAt_rpow_const (p := a + 1) (Or.inl (hc.trans_le hx).ne')).div_const _ using 1
    field_simp [show a + 1 ≠ 0 from ne_of_lt (by linarith), mul_comm]
  have ht : Tendsto (fun t => t ^ (a + 1) / (a + 1)) atTop (𝓝 (0 / (a + 1))) := by
    apply Tendsto.div_const
    simpa only [neg_neg] using tendsto_rpow_neg_atTop (by linarith : 0 < -(a + 1))
  /-
    a : Real
    ha : LT.lt a (-1)
    c : Real
    hc : LT.lt 0 c
    hd : ∀ (x : Real), Membership.mem (Set.Ici c) x → HasDerivAt (fun t => HDiv.hD …
    ht : Filter.Tendsto (fun t => HDiv.hDiv (HPow.hPow t (HAdd.hAdd a 1)) (HAdd.hA …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  convert integral_Ioi_of_hasDerivAt_of_tendsto' hd (integrableOn_Ioi_rpow_of_lt ha hc) ht using 1
  /-
    case h.e'_3
    a : Real
    ha : LT.lt a (-1)
    c : Real
    hc : LT.lt 0 c
    hd : ∀ (x : Real), Membership.mem (Set.Ici c) x → HasDerivAt (fun t => HDiv.hD …
    ht : Filter.Tendsto (fun t => HDiv.hDiv (HPow.hPow t (HAdd.hAdd a 1)) (HAdd.hA …
    ⊢ Eq (HDiv.hDiv (Neg.neg (HPow.hPow c (HAdd.hAdd a 1))) (HAdd.hAdd a 1)) (HSub …
  -/
  simp only [neg_div, zero_div, zero_sub]
  /-
    🎉 no goals
  -/


theorem integrableOn_Ioi_cpow_of_lt {a : ℂ} (ha : a.re < -1) {c : ℝ} (hc : 0 < c) :
    /-
      a : Complex
      ha : LT.lt a.re (-1)
      c : Real
      hc : LT.lt 0 c
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun t : ℝ => (t : ℂ) ^ a) (Ioi c) := by
    /-
      🎉 no goals
    -/
  /-
    a : Complex
    ha : LT.lt a.re (-1)
    c : Real
    hc : LT.lt 0 c
    ⊢ MeasureTheory.IntegrableOn (fun t => HPow.hPow (↑t) a) (Set.Ioi c) MeasureTh …
  -/
  rw [IntegrableOn, ← integrable_norm_iff, ← IntegrableOn]
    /-
      a : Complex
      ha : LT.lt a.re (-1)
      c : Real
      hc : LT.lt 0 c
      ⊢ MeasureTheory.IntegrableOn (fun a_1 => Norm.norm (HPow.hPow (↑a_1) a)) (Set. …
    -/
  · refine (integrableOn_Ioi_rpow_of_lt ha hc).congr_fun (fun x hx => ?_) measurableSet_Ioi
      /-
        a : Complex
        ha : LT.lt a.re (-1)
        c : Real
        hc : LT.lt 0 c
        x : Real
        hx : Membership.mem (Set.Ioi c) x
        ⊢ Eq (HPow.hPow x a.re) (Norm.norm (HPow.hPow (↑x) a))
      -/
    · dsimp only
      /-
        a : Complex
        ha : LT.lt a.re (-1)
        c : Real
        hc : LT.lt 0 c
        x : Real
        hx : Membership.mem (Set.Ioi c) x
        ⊢ Eq (HPow.hPow x a.re) (Norm.norm (HPow.hPow (↑x) a))
      -/
      rw [Complex.norm_eq_abs, Complex.abs_cpow_eq_rpow_re_of_pos (hc.trans hx)]
      /-
        🎉 no goals
      -/
    /-
      a : Complex
      ha : LT.lt a.re (-1)
      c : Real
      hc : LT.lt 0 c
      ⊢ MeasureTheory.AEStronglyMeasurable (fun t => HPow.hPow (↑t) a) (MeasureTheor …
    -/
  · refine ContinuousOn.aestronglyMeasurable (fun t ht => ?_) measurableSet_Ioi
    exact
      (Complex.continuousAt_ofReal_cpow_const _ _ (Or.inr (hc.trans ht).ne')).continuousWithinAt


theorem integrableOn_Ioi_cpow_iff {s : ℂ} {t : ℝ} (ht : 0 < t) :
    /-
      s : Complex
      t : Real
      ht : LT.lt 0 t
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x : ℝ ↦ (x : ℂ) ^ s) (Ioi t) ↔ s.re < -1 := by
    /-
      🎉 no goals
    -/
  /-
    s : Complex
    t : Real
    ht : LT.lt 0 t
    ⊢ Iff (MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi t) Meas …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ integrableOn_Ioi_cpow_of_lt h ht⟩
  have B : IntegrableOn (fun a ↦ a ^ s.re) (Ioi t) := by
    apply (integrableOn_congr_fun _ measurableSet_Ioi).1 h.norm
    intro a ha
    have : 0 < a := ht.trans ha
    simp [Complex.abs_cpow_eq_rpow_re_of_pos this]
  /-
    s : Complex
    t : Real
    ht : LT.lt 0 t
    h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi t) Measure …
    B : MeasureTheory.IntegrableOn (fun a => HPow.hPow a s.re) (Set.Ioi t) Measure …
    ⊢ LT.lt s.re (-1)
  -/
  rwa [integrableOn_Ioi_rpow_iff ht] at B
  /-
    🎉 no goals
  -/


/-- The complex power function with any exponent is not integrable on `(0, +∞)`. -/
theorem not_integrableOn_Ioi_cpow (s : ℂ) :
      /-
        s : Complex
        ⊢ MeasureTheory.Measure Real
      -/
    ¬ IntegrableOn (fun x : ℝ ↦ (x : ℂ) ^ s) (Ioi (0 : ℝ)) := by
      /-
        🎉 no goals
      -/
  /-
    s : Complex
    ⊢ Not (MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 0) Meas …
  -/
  intro h
  /-
    s : Complex
    h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 0) Measure …
    ⊢ False
  -/
  rcases le_or_lt s.re (-1) with hs|hs
  · have : IntegrableOn (fun x : ℝ ↦ (x : ℂ) ^ s) (Ioo (0 : ℝ) 1) :=
      h.mono Ioo_subset_Ioi_self le_rfl
    /-
      case inl
      s : Complex
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 0) Measure …
      hs : LE.le s.re (-1)
      this : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioo 0 1) Me …
      ⊢ False
    -/
    rw [integrableOn_Ioo_cpow_iff zero_lt_one] at this
    /-
      case inl
      s : Complex
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 0) Measure …
      hs : LE.le s.re (-1)
      this : LT.lt (-1) s.re
      ⊢ False
    -/
    exact hs.not_lt this
    /-
      🎉 no goals
    -/
  · have : IntegrableOn (fun x : ℝ ↦ (x : ℂ) ^ s) (Ioi 1) :=
      h.mono (Ioi_subset_Ioi zero_le_one) le_rfl
    /-
      case inr
      s : Complex
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 0) Measure …
      hs : LT.lt (-1) s.re
      this : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 1) Meas …
      ⊢ False
    -/
    rw [integrableOn_Ioi_cpow_iff zero_lt_one] at this
    /-
      case inr
      s : Complex
      h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioi 0) Measure …
      hs : LT.lt (-1) s.re
      this : LT.lt s.re (-1)
      ⊢ False
    -/
    exact hs.not_lt this
    /-
      🎉 no goals
    -/


theorem setIntegral_Ioi_zero_cpow (s : ℂ) : ∫ x in Ioi (0 : ℝ), (x : ℂ) ^ s = 0 :=
  MeasureTheory.integral_undef (not_integrableOn_Ioi_cpow s)


theorem integral_Ioi_cpow_of_lt {a : ℂ} (ha : a.re < -1) {c : ℝ} (hc : 0 < c) :
    (∫ t : ℝ in Ioi c, (t : ℂ) ^ a) = -(c : ℂ) ^ (a + 1) / (a + 1) := by
  refine
    tendsto_nhds_unique
      (intervalIntegral_tendsto_integral_Ioi c (integrableOn_Ioi_cpow_of_lt ha hc) tendsto_id) ?_
  suffices
    Tendsto (fun x : ℝ => ((x : ℂ) ^ (a + 1) - (c : ℂ) ^ (a + 1)) / (a + 1)) atTop
      (𝓝 <| -c ^ (a + 1) / (a + 1)) by
    refine this.congr' ((eventually_gt_atTop 0).mp (Eventually.of_forall fun x hx => ?_))
    dsimp only
    rw [integral_cpow, id]
    refine Or.inr ⟨?_, not_mem_uIcc_of_lt hc hx⟩
    apply_fun Complex.re
    rw [Complex.neg_re, Complex.one_re]
    exact ha.ne
  /-
    a : Complex
    ha : LT.lt a.re (-1)
    c : Real
    hc : LT.lt 0 c
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HSub.hSub (HPow.hPow (↑x) (HAdd.hAdd a 1 …
  -/
  simp_rw [← zero_sub, sub_div]
  /-
    a : Complex
    ha : LT.lt a.re (-1)
    c : Real
    hc : LT.lt 0 c
    ⊢ Filter.Tendsto (fun x => HSub.hSub (HDiv.hDiv (HPow.hPow (↑x) (HAdd.hAdd a 1 …
  -/
  refine (Tendsto.div_const ?_ _).sub_const _
  /-
    a : Complex
    ha : LT.lt a.re (-1)
    c : Real
    hc : LT.lt 0 c
    ⊢ Filter.Tendsto (fun x => HPow.hPow (↑x) (HAdd.hAdd a 1)) Filter.atTop (nhds 0)
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  refine
    (tendsto_rpow_neg_atTop (by linarith : 0 < -(a.re + 1))).congr'
      ((eventually_gt_atTop 0).mp (Eventually.of_forall fun x hx => ?_))
  simp_rw [neg_neg, Complex.norm_eq_abs, Complex.abs_cpow_eq_rpow_re_of_pos hx, Complex.add_re,
    Complex.one_re]


                                    /-
                                      ⊢ MeasureTheory.Measure Real
                                    -/
theorem integrable_inv_one_add_sq : Integrable fun (x : ℝ) ↦ (1 + x ^ 2)⁻¹ := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    ⊢ MeasureTheory.Integrable (fun x => Inv.inv (HAdd.hAdd 1 (HPow.hPow x 2))) Me …
  -/
  suffices Integrable fun (x : ℝ) ↦ (1 + ‖x‖ ^ 2) ^ ((-2 : ℝ) / 2) by simpa [rpow_neg_one]
  /-
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (HPow.hPow (Norm.n …
  -/
  exact integrable_rpow_neg_one_add_norm_sq (by simp)
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_Iic_inv_one_add_sq {i : ℝ} :
    ∫ (x : ℝ) in Set.Iic i, (1 + x ^ 2)⁻¹ = arctan i + (π / 2) :=
  integral_Iic_of_hasDerivAt_of_tendsto' (fun x _ => hasDerivAt_arctan' x)
    integrable_inv_one_add_sq.integrableOn (tendsto_nhds_of_tendsto_nhdsWithin tendsto_arctan_atBot)
    |>.trans (sub_neg_eq_add _ _)


@[simp]
theorem integral_Ioi_inv_one_add_sq {i : ℝ} :
    ∫ (x : ℝ) in Set.Ioi i, (1 + x ^ 2)⁻¹ = (π / 2) - arctan i :=
  integral_Ioi_of_hasDerivAt_of_tendsto' (fun x _ => hasDerivAt_arctan' x)
    integrable_inv_one_add_sq.integrableOn (tendsto_nhds_of_tendsto_nhdsWithin tendsto_arctan_atTop)


@[simp]
theorem integral_univ_inv_one_add_sq : ∫ (x : ℝ), (1 + x ^ 2)⁻¹ = π :=
      /-
        ⊢ Eq Real.pi (HSub.hSub (HDiv.hDiv Real.pi 2) (Neg.neg (HDiv.hDiv Real.pi 2)))
      -/
  (by ring : π = (π / 2) - (-(π / 2))) ▸ integral_of_hasDerivAt_of_tendsto hasDerivAt_arctan'
      /-
        🎉 no goals
      -/
    integrable_inv_one_add_sq (tendsto_nhds_of_tendsto_nhdsWithin tendsto_arctan_atBot)
    (tendsto_nhds_of_tendsto_nhdsWithin tendsto_arctan_atTop)

