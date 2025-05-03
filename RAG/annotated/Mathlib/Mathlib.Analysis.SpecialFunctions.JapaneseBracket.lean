theorem sqrt_one_add_norm_sq_le (x : E) : √((1 : ℝ) + ‖x‖ ^ 2) ≤ 1 + ‖x‖ := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    x : E
    ⊢ LE.le (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)).sqrt (HAdd.hAdd 1 (Norm.norm …
  -/
  rw [sqrt_le_left (by positivity)]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    x : E
    ⊢ LE.le (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)) (HPow.hPow (HAdd.hAdd 1 (Nor …
  -/
  simp [add_sq]
  /-
    🎉 no goals
  -/


theorem one_add_norm_le_sqrt_two_mul_sqrt (x : E) :
    (1 : ℝ) + ‖x‖ ≤ √2 * √(1 + ‖x‖ ^ 2) := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    x : E
    ⊢ LE.le (HAdd.hAdd 1 (Norm.norm x)) (HMul.hMul (Real.sqrt 2) (HAdd.hAdd 1 (HPo …
  -/
  rw [← sqrt_mul zero_le_two]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    x : E
    ⊢ LE.le (HAdd.hAdd 1 (Norm.norm x)) (HMul.hMul 2 (HAdd.hAdd 1 (HPow.hPow (Norm …
  -/
  have := sq_nonneg (‖x‖ - 1)
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    x : E
    this : LE.le 0 (HPow.hPow (HSub.hSub (Norm.norm x) 1) 2)
    ⊢ LE.le (HAdd.hAdd 1 (Norm.norm x)) (HMul.hMul 2 (HAdd.hAdd 1 (HPow.hPow (Norm …
  -/
  apply le_sqrt_of_sq_le
  /-
    case h
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    x : E
    this : LE.le 0 (HPow.hPow (HSub.hSub (Norm.norm x) 1) 2)
    ⊢ LE.le (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) 2) (HMul.hMul 2 (HAdd.hAdd 1 (H …
  -/
  linarith
  /-
    🎉 no goals
  -/


theorem rpow_neg_one_add_norm_sq_le {r : ℝ} (x : E) (hr : 0 < r) :
    ((1 : ℝ) + ‖x‖ ^ 2) ^ (-r / 2) ≤ (2 : ℝ) ^ (r / 2) * (1 + ‖x‖) ^ (-r) :=
  calc
    ((1 : ℝ) + ‖x‖ ^ 2) ^ (-r / 2)
      = (2 : ℝ) ^ (r / 2) * ((√2 * √((1 : ℝ) + ‖x‖ ^ 2)) ^ r)⁻¹ := by
      rw [rpow_div_two_eq_sqrt, rpow_div_two_eq_sqrt, mul_rpow, mul_inv, rpow_neg,
                                  /-
                                    case h
                                    E : Type u_1
                                    inst✝ : NormedAddCommGroup E
                                    r : Real
                                    x : E
                                    hr : LT.lt 0 r
                                    ⊢ Ne (HPow.hPow (Real.sqrt 2) r) 0
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
        mul_inv_cancel_left₀] <;> positivity
                                  /-
                                    🎉 no goals
                                  -/
    _ ≤ (2 : ℝ) ^ (r / 2) * ((1 + ‖x‖) ^ r)⁻¹ := by
      /-
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        r : Real
        x : E
        hr : LT.lt 0 r
        ⊢ LE.le (HMul.hMul (HPow.hPow 2 (HDiv.hDiv r 2)) (Inv.inv (HPow.hPow (HMul.hMu …
      -/
      gcongr
      /-
        case h.hba.h₁
        E : Type u_1
        inst✝ : NormedAddCommGroup E
        r : Real
        x : E
        hr : LT.lt 0 r
        ⊢ LE.le (HAdd.hAdd 1 (Norm.norm x)) (HMul.hMul (Real.sqrt 2) (HAdd.hAdd 1 (HPo …
      -/
      apply one_add_norm_le_sqrt_two_mul_sqrt
      /-
        🎉 no goals
      -/
                                                   /-
                                                     E : Type u_1
                                                     inst✝ : NormedAddCommGroup E
                                                     r : Real
                                                     x : E
                                                     hr : LT.lt 0 r
                                                     ⊢ Eq (HMul.hMul (HPow.hPow 2 (HDiv.hDiv r 2)) (Inv.inv (HPow.hPow (HAdd.hAdd 1 …
                                                   -/
    _ = (2 : ℝ) ^ (r / 2) * (1 + ‖x‖) ^ (-r) := by rw [rpow_neg]; positivity
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem le_rpow_one_add_norm_iff_norm_le {r t : ℝ} (hr : 0 < r) (ht : 0 < t) (x : E) :
    t ≤ (1 + ‖x‖) ^ (-r) ↔ ‖x‖ ≤ t ^ (-r⁻¹) - 1 := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    r t : Real
    hr : LT.lt 0 r
    ht : LT.lt 0 t
    x : E
    ⊢ Iff (LE.le t (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))) (LE.le (No …
  -/
  rw [le_sub_iff_add_le', neg_inv]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    r t : Real
    hr : LT.lt 0 r
    ht : LT.lt 0 t
    x : E
    ⊢ Iff (LE.le t (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))) (LE.le (HA …
  -/
  exact (Real.le_rpow_inv_iff_of_neg (by positivity) ht (neg_lt_zero.mpr hr)).symm
  /-
    🎉 no goals
  -/


theorem closedBall_rpow_sub_one_eq_empty_aux {r t : ℝ} (hr : 0 < r) (ht : 1 < t) :
    Metric.closedBall (0 : E) (t ^ (-r⁻¹) - 1) = ∅ := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    r t : Real
    hr : LT.lt 0 r
    ht : LT.lt 1 t
    ⊢ Eq (Metric.closedBall 0 (HSub.hSub (HPow.hPow t (Neg.neg (Inv.inv r))) 1)) E …
  -/
  rw [Metric.closedBall_eq_empty, sub_neg]
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    r t : Real
    hr : LT.lt 0 r
    ht : LT.lt 1 t
    ⊢ LT.lt (HPow.hPow t (Neg.neg (Inv.inv r))) 1
  -/
  exact Real.rpow_lt_one_of_one_lt_of_neg ht (by simp only [hr, Right.neg_neg_iff, inv_pos])
  /-
    🎉 no goals
  -/


theorem finite_integral_rpow_sub_one_pow_aux {r : ℝ} (n : ℕ) (hnr : (n : ℝ) < r) :
    (∫⁻ x : ℝ in Ioc 0 1, ENNReal.ofReal ((x ^ (-r⁻¹) - 1) ^ n)) < ∞ := by
  /-
    r : Real
    n : Nat
    hnr : LT.lt (↑n) r
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  have hr : 0 < r := lt_of_le_of_lt n.cast_nonneg hnr
  have h_int x (hx : x ∈ Ioc (0 : ℝ) 1) := by
    calc
      ENNReal.ofReal ((x ^ (-r⁻¹) - 1) ^ n) ≤ .ofReal ((x ^ (-r⁻¹) - 0) ^ n) := by
        gcongr
        · rw [sub_nonneg]
          exact Real.one_le_rpow_of_pos_of_le_one_of_nonpos hx.1 hx.2 (by simpa using hr.le)
        · norm_num
      _ = .ofReal (x ^ (-(r⁻¹ * n))) := by simp [rpow_mul hx.1.le, ← neg_mul]
  /-
    r : Real
    n : Nat
    hnr : LT.lt (↑n) r
    hr : LT.lt 0 r
    h_int : ∀ (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (ENNReal.ofReal ( …
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  refine lt_of_le_of_lt (setLIntegral_mono' measurableSet_Ioc h_int) ?_
  /-
    r : Real
    n : Nat
    hnr : LT.lt (↑n) r
    hr : LT.lt 0 r
    h_int : ∀ (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (ENNReal.ofReal ( …
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  refine IntegrableOn.setLIntegral_lt_top ?_
  /-
    r : Real
    n : Nat
    hnr : LT.lt (↑n) r
    hr : LT.lt 0 r
    h_int : ∀ (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (ENNReal.ofReal ( …
    ⊢ MeasureTheory.IntegrableOn (fun x => HPow.hPow x (Neg.neg (HMul.hMul (Inv.in …
  -/
  rw [← intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one]
  /-
    r : Real
    n : Nat
    hnr : LT.lt (↑n) r
    hr : LT.lt 0 r
    h_int : ∀ (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (ENNReal.ofReal ( …
    ⊢ IntervalIntegrable (fun x => HPow.hPow x (Neg.neg (HMul.hMul (Inv.inv r) ↑n) …
  -/
  apply intervalIntegral.intervalIntegrable_rpow'
  /-
    case h
    r : Real
    n : Nat
    hnr : LT.lt (↑n) r
    hr : LT.lt 0 r
    h_int : ∀ (x : Real), Membership.mem (Set.Ioc 0 1) x → LE.le (ENNReal.ofReal ( …
    ⊢ LT.lt (-1) (Neg.neg (HMul.hMul (Inv.inv r) ↑n))
  -/
  rwa [neg_lt_neg_iff, inv_mul_lt_iff₀' hr, one_mul]
  /-
    🎉 no goals
  -/


theorem finite_integral_one_add_norm {r : ℝ} (hnr : (finrank ℝ E : ℝ) < r) :
    (∫⁻ x : E, ENNReal.ofReal ((1 + ‖x‖) ^ (-r)) ∂μ) < ∞ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (HPow.hPow (HAdd.hA …
  -/
  have hr : 0 < r := lt_of_le_of_lt (finrank ℝ E).cast_nonneg hnr
  -- We start by applying the layer cake formula
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (HPow.hPow (HAdd.hA …
  -/
  have h_meas : Measurable fun ω : E => (1 + ‖ω‖) ^ (-r) := by fun_prop
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    h_meas : Measurable fun ω => HPow.hPow (HAdd.hAdd 1 (Norm.norm ω)) (Neg.neg r)
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (HPow.hPow (HAdd.hA …
  -/
  have h_pos : ∀ x : E, 0 ≤ (1 + ‖x‖) ^ (-r) := fun x ↦ by positivity
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    h_meas : Measurable fun ω => HPow.hPow (HAdd.hAdd 1 (Norm.norm ω)) (Neg.neg r)
    h_pos : ∀ (x : E), LE.le 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (HPow.hPow (HAdd.hA …
  -/
  rw [lintegral_eq_lintegral_meas_le μ (Eventually.of_forall h_pos) h_meas.aemeasurable]
  have h_int : ∀ t, 0 < t → μ {a : E | t ≤ (1 + ‖a‖) ^ (-r)} =
      μ (Metric.closedBall (0 : E) (t ^ (-r⁻¹) - 1)) := fun t ht ↦ by
    congr 1
    ext x
    simp only [mem_setOf_eq, mem_closedBall_zero_iff]
    exact le_rpow_one_add_norm_iff_norm_le hr (mem_Ioi.mp ht) x
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    h_meas : Measurable fun ω => HPow.hPow (HAdd.hAdd 1 (Norm.norm ω)) (Neg.neg r)
    h_pos : ∀ (x : E), LE.le 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))
    h_int : ∀ (t : Real), LT.lt 0 t → Eq (μ (setOf fun a => LE.le t (HPow.hPow (HA …
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  rw [setLIntegral_congr_fun measurableSet_Ioi (Eventually.of_forall h_int)]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    h_meas : Measurable fun ω => HPow.hPow (HAdd.hAdd 1 (Norm.norm ω)) (Neg.neg r)
    h_pos : ∀ (x : E), LE.le 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))
    h_int : ∀ (t : Real), LT.lt 0 t → Eq (μ (setOf fun a => LE.le t (HPow.hPow (HA …
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  set f := fun t : ℝ ↦ μ (Metric.closedBall (0 : E) (t ^ (-r⁻¹) - 1))
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    h_meas : Measurable fun ω => HPow.hPow (HAdd.hAdd 1 (Norm.norm ω)) (Neg.neg r)
    h_pos : ∀ (x : E), LE.le 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))
    h_int : ∀ (t : Real), LT.lt 0 t → Eq (μ (setOf fun a => LE.le t (HPow.hPow (HA …
    f : Real → ENNReal := fun t => μ (Metric.closedBall 0 (HSub.hSub (HPow.hPow t  …
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict ( …
  -/
  set mB := μ (Metric.ball (0 : E) 1)
  -- the next two inequalities are in fact equalities but we don't need that
  calc
    ∫⁻ t in Ioi 0, f t ≤ ∫⁻ t in Ioc 0 1 ∪ Ioi 1, f t := lintegral_mono_set Ioi_subset_Ioc_union_Ioi
    _ ≤ (∫⁻ t in Ioc 0 1, f t) + ∫⁻ t in Ioi 1, f t := lintegral_union_le _ _ _
    _ < ∞ := ENNReal.add_lt_top.2 ⟨?_, ?_⟩
  · -- We use estimates from auxiliary lemmas to deal with integral from `0` to `1`
    have h_int' : ∀ t ∈ Ioc (0 : ℝ) 1,
        f t = ENNReal.ofReal ((t ^ (-r⁻¹) - 1) ^ finrank ℝ E) * mB := fun t ht ↦ by
      refine μ.addHaar_closedBall (0 : E) ?_
      rw [sub_nonneg]
      exact Real.one_le_rpow_of_pos_of_le_one_of_nonpos ht.1 ht.2 (by simp [hr.le])
    rw [setLIntegral_congr_fun measurableSet_Ioc (ae_of_all _ h_int'),
      lintegral_mul_const' _ _ measure_ball_lt_top.ne]
    exact ENNReal.mul_lt_top
      (finite_integral_rpow_sub_one_pow_aux (finrank ℝ E) hnr) measure_ball_lt_top
  · -- The integral from 1 to ∞ is zero:
    have h_int'' : ∀ t ∈ Ioi (1 : ℝ), f t = 0 := fun t ht => by
      simp only [f, closedBall_rpow_sub_one_eq_empty_aux E hr ht, measure_empty]
    -- The integral over the constant zero function is finite:
    rw [setLIntegral_congr_fun measurableSet_Ioi (ae_of_all volume <| h_int''), lintegral_const 0,
      zero_mul]
    /-
      case calc_2
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      r : Real
      hnr : LT.lt (↑(Module.finrank Real E)) r
      hr : LT.lt 0 r
      h_meas : Measurable fun ω => HPow.hPow (HAdd.hAdd 1 (Norm.norm ω)) (Neg.neg r)
      h_pos : ∀ (x : E), LE.le 0 (HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Neg.neg r))
      h_int : ∀ (t : Real), LT.lt 0 t → Eq (μ (setOf fun a => LE.le t (HPow.hPow (HA …
      f : Real → ENNReal := fun t => μ (Metric.closedBall 0 (HSub.hSub (HPow.hPow t  …
      mB : ENNReal := μ (Metric.ball 0 1)
      h_int'' : ∀ (t : Real), Membership.mem (Set.Ioi 1) t → Eq (f t) 0
      ⊢ LT.lt 0 Top.top
    -/
    exact WithTop.top_pos
    /-
      🎉 no goals
    -/


theorem integrable_one_add_norm {r : ℝ} (hnr : (finrank ℝ E : ℝ) < r) :
    Integrable (fun x ↦ (1 + ‖x‖) ^ (-r)) μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm x)) (Ne …
  -/
  constructor
    /-
      case left
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      r : Real
      hnr : LT.lt (↑(Module.finrank Real E)) r
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.no …
    -/
  · apply Measurable.aestronglyMeasurable (by fun_prop)
    /-
      🎉 no goals
    -/
  -- Lower Lebesgue integral
  have : (∫⁻ a : E, ‖(1 + ‖a‖) ^ (-r)‖₊ ∂μ) = ∫⁻ a : E, ENNReal.ofReal ((1 + ‖a‖) ^ (-r)) ∂μ :=
    lintegral_nnnorm_eq_of_nonneg fun _ => rpow_nonneg (by positivity) _
  /-
    case right
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    this : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (HPow.hPow (HAdd …
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => HPow.hPow (HAdd.hAdd 1 (Norm.norm  …
  -/
  rw [hasFiniteIntegral_iff_nnnorm, this]
  /-
    case right
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    this : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (HPow.hPow (HAdd …
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (HAdd.hA …
  -/
  exact finite_integral_one_add_norm hnr
  /-
    🎉 no goals
  -/


theorem integrable_rpow_neg_one_add_norm_sq {r : ℝ} (hnr : (finrank ℝ E : ℝ) < r) :
    Integrable (fun x ↦ ((1 : ℝ) + ‖x‖ ^ 2) ^ (-r / 2)) μ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (HAdd.hAdd 1 (HPow.hPow (Norm.n …
  -/
  have hr : 0 < r := lt_of_le_of_lt (finrank ℝ E).cast_nonneg hnr
  refine ((integrable_one_add_norm hnr).const_mul <| (2 : ℝ) ^ (r / 2)).mono'
    ?_ (Eventually.of_forall fun x => ?_)
    /-
      case refine_1
      E : Type u_1
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      μ : MeasureTheory.Measure E
      inst✝ : μ.IsAddHaarMeasure
      r : Real
      hnr : LT.lt (↑(Module.finrank Real E)) r
      hr : LT.lt 0 r
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HPow.hPow (HAdd.hAdd 1 (HPow.hP …
    -/
  · apply Measurable.aestronglyMeasurable (by fun_prop)
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    x : E
    ⊢ LE.le (Norm.norm (HPow.hPow (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)) (HDiv. …
  -/
  refine (abs_of_pos ?_).trans_le (rpow_neg_one_add_norm_sq_le x hr)
  /-
    case refine_2
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FiniteDimensional Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    r : Real
    hnr : LT.lt (↑(Module.finrank Real E)) r
    hr : LT.lt 0 r
    x : E
    ⊢ LT.lt 0 (HPow.hPow (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2)) (HDiv.hDiv (Neg …
  -/
  positivity
  /-
    🎉 no goals
  -/

