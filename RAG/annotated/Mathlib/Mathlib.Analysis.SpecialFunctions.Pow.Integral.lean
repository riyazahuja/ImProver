include f_nn f_mble p_pos in
/-- An application of the layer cake formula / Cavalieri's principle / tail probability formula:

For a nonnegative function `f` on a measure space, the Lebesgue integral of `f` can
be written (roughly speaking) as: `∫⁻ f^p ∂μ = p * ∫⁻ t in 0..∞, t^(p-1) * μ {ω | f(ω) ≥ t}`.

See `MeasureTheory.lintegral_rpow_eq_lintegral_meas_lt_mul` for a version with sets of the form
`{ω | f(ω) > t}` instead. -/
theorem lintegral_rpow_eq_lintegral_meas_le_mul :
    ∫⁻ ω, ENNReal.ofReal (f ω ^ p) ∂μ =
      ENNReal.ofReal p * ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (t ^ (p - 1)) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (HPow.hPow (f ω) p)) ( …
  -/
  have one_lt_p : -1 < p - 1 := by linarith
  have obs : ∀ x : ℝ, ∫ t : ℝ in (0)..x, t ^ (p - 1) = x ^ p / p := by
    intro x
    rw [integral_rpow (Or.inl one_lt_p)]
    simp [Real.zero_rpow p_pos.ne.symm]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    one_lt_p : LT.lt (-1) (HSub.hSub p 1)
    obs : ∀ (x : Real), Eq (intervalIntegral (fun t => HPow.hPow t (HSub.hSub p 1) …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (HPow.hPow (f ω) p)) ( …
  -/
  set g := fun t : ℝ => t ^ (p - 1)
  have g_nn : ∀ᵐ t ∂volume.restrict (Ioi (0 : ℝ)), 0 ≤ g t := by
    filter_upwards [self_mem_ae_restrict (measurableSet_Ioi : MeasurableSet (Ioi (0 : ℝ)))]
    intro t t_pos
    exact Real.rpow_nonneg (mem_Ioi.mp t_pos).le (p - 1)
  have g_intble : ∀ t > 0, IntervalIntegrable g volume 0 t := fun _ _ =>
    intervalIntegral.intervalIntegrable_rpow' one_lt_p
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    one_lt_p : LT.lt (-1) (HSub.hSub p 1)
    g : Real → Real := fun t => HPow.hPow t (HSub.hSub p 1)
    obs : ∀ (x : Real), Eq (intervalIntegral g 0 x MeasureTheory.MeasureSpace.volu …
    g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (HPow.hPow (f ω) p)) ( …
  -/
  have key := lintegral_comp_eq_lintegral_meas_le_mul μ f_nn f_mble g_intble g_nn
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    one_lt_p : LT.lt (-1) (HSub.hSub p 1)
    g : Real → Real := fun t => HPow.hPow t (HSub.hSub p 1)
    obs : ∀ (x : Real), Eq (intervalIntegral g 0 x MeasureTheory.MeasureSpace.volu …
    g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (HPow.hPow (f ω) p)) ( …
  -/
  rw [← key, ← lintegral_const_mul'' (ENNReal.ofReal p)] <;> simp_rw [obs]
    /-
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Real
      μ : MeasureTheory.Measure α
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
      f_mble : AEMeasurable f μ
      p : Real
      p_pos : LT.lt 0 p
      one_lt_p : LT.lt (-1) (HSub.hSub p 1)
      g : Real → Real := fun t => HPow.hPow t (HSub.hSub p 1)
      obs : ∀ (x : Real), Eq (intervalIntegral g 0 x MeasureTheory.MeasureSpace.volu …
      g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (HPow.hPow (f ω) p)) ( …
    -/
  · congr with ω
    /-
      case e_f.h
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Real
      μ : MeasureTheory.Measure α
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
      f_mble : AEMeasurable f μ
      p : Real
      p_pos : LT.lt 0 p
      one_lt_p : LT.lt (-1) (HSub.hSub p 1)
      g : Real → Real := fun t => HPow.hPow t (HSub.hSub p 1)
      obs : ∀ (x : Real), Eq (intervalIntegral g 0 x MeasureTheory.MeasureSpace.volu …
      g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
      ω : α
      ⊢ Eq (ENNReal.ofReal (HPow.hPow (f ω) p)) (HMul.hMul (ENNReal.ofReal p) (ENNRe …
    -/
    rw [← ENNReal.ofReal_mul p_pos.le, mul_div_cancel₀ (f ω ^ p) p_pos.ne.symm]
    /-
      🎉 no goals
    -/
  · have aux := (@measurable_const ℝ α (by infer_instance) (by infer_instance) p).aemeasurable
                  (μ := μ)
    exact (Measurable.ennreal_ofReal (hf := measurable_id)).comp_aemeasurable
      ((f_mble.pow aux).div_const p)


include f_nn f_mble p_pos in
/-- An application of the layer cake formula / Cavalieri's principle / tail probability formula:

For a nonnegative function `f` on a measure space, the Lebesgue integral of `f` can
be written (roughly speaking) as: `∫⁻ f^p ∂μ = p * ∫⁻ t in 0..∞, t^(p-1) * μ {ω | f(ω) > t}`.

See `MeasureTheory.lintegral_rpow_eq_lintegral_meas_le_mul` for a version with sets of the form
`{ω | f(ω) ≥ t}` instead. -/
theorem lintegral_rpow_eq_lintegral_meas_lt_mul :
    ∫⁻ ω, ENNReal.ofReal (f ω ^ p) ∂μ =
      ENNReal.ofReal p * ∫⁻ t in Ioi 0, μ {a : α | t < f a} * ENNReal.ofReal (t ^ (p - 1)) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (HPow.hPow (f ω) p)) ( …
  -/
  rw [lintegral_rpow_eq_lintegral_meas_le_mul μ f_nn f_mble p_pos]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    ⊢ Eq (HMul.hMul (ENNReal.ofReal p) (MeasureTheory.lintegral (MeasureTheory.Mea …
  -/
  apply congr_arg fun z => ENNReal.ofReal p * z
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
  -/
  apply lintegral_congr_ae
  filter_upwards [meas_le_ae_eq_meas_lt μ (volume.restrict (Ioi 0)) f]
    with t ht
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    p : Real
    p_pos : LT.lt 0 p
    t : Real
    ht : Eq (μ (setOf fun a => LE.le t (f a))) (μ (setOf fun a => LT.lt t (f a)))
    ⊢ Eq (HMul.hMul (μ (setOf fun a => LE.le t (f a))) (ENNReal.ofReal (HPow.hPow  …
  -/
  rw [ht]
  /-
    🎉 no goals
  -/


