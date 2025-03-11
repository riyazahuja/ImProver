theorem countable_meas_le_ne_meas_lt (g : α → R) :
    {t : R | μ {a : α | t ≤ g a} ≠ μ {a : α | t < g a}}.Countable := by
  -- the target set is contained in the set of points where the function `t ↦ μ {a : α | t ≤ g a}`
  -- jumps down on the right of `t`. This jump set is countable for any function.
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder R
    g : α → R
    ⊢ (setOf fun t => Ne (μ (setOf fun a => LE.le t (g a))) (μ (setOf fun a => LT. …
  -/
  let F : R → ℝ≥0∞ := fun t ↦ μ {a : α | t ≤ g a}
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder R
    g : α → R
    F : R → ENNReal := fun t => μ (setOf fun a => LE.le t (g a))
    ⊢ (setOf fun t => Ne (μ (setOf fun a => LE.le t (g a))) (μ (setOf fun a => LT. …
  -/
  apply (countable_image_gt_image_Ioi F).mono
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder R
    g : α → R
    F : R → ENNReal := fun t => μ (setOf fun a => LE.le t (g a))
    ⊢ HasSubset.Subset (setOf fun t => Ne (μ (setOf fun a => LE.le t (g a))) (μ (s …
  -/
  intro t ht
  have : μ {a | t < g a} < μ {a | t ≤ g a} :=
    lt_of_le_of_ne (measure_mono (fun a ha ↦ le_of_lt ha)) (Ne.symm ht)
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : LinearOrder R
    g : α → R
    F : R → ENNReal := fun t => μ (setOf fun a => LE.le t (g a))
    t : R
    ht : Membership.mem (setOf fun t => Ne (μ (setOf fun a => LE.le t (g a))) (μ ( …
    this : LT.lt (μ (setOf fun a => LT.lt t (g a))) (μ (setOf fun a => LE.le t (g  …
    ⊢ Membership.mem (setOf fun x => Exists fun z => And (LT.lt z (F x)) (∀ (y : R …
  -/
  exact ⟨μ {a | t < g a}, this, fun s hs ↦ measure_mono (fun a ha ↦ hs.trans_le ha)⟩
  /-
    🎉 no goals
  -/


theorem meas_le_ae_eq_meas_lt {R : Type*} [LinearOrder R] [MeasurableSpace R]
    (ν : Measure R) [NoAtoms ν] (g : α → R) :
    (fun t => μ {a : α | t ≤ g a}) =ᵐ[ν] fun t => μ {a : α | t < g a} :=
  Set.Countable.measure_zero (countable_meas_le_ne_meas_lt μ g) _


/-- An auxiliary version of the layer cake formula (Cavalieri's principle, tail probability
formula), with a measurability assumption that would also essentially follow from the
integrability assumptions, and a sigma-finiteness assumption.

See `MeasureTheory.lintegral_comp_eq_lintegral_meas_le_mul` and
`MeasureTheory.lintegral_comp_eq_lintegral_meas_lt_mul` for the main formulations of the layer
cake formula. -/
theorem lintegral_comp_eq_lintegral_meas_le_mul_of_measurable_of_sigmaFinite
    (μ : Measure α) [SFinite μ]
    (f_nn : 0 ≤ f) (f_mble : Measurable f)
    (g_intble : ∀ t > 0, IntervalIntegrable g volume 0 t) (g_mble : Measurable g)
    (g_nn : ∀ t > 0, 0 ≤ g t) :
    ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ =
      ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
  have g_intble' : ∀ t : ℝ, 0 ≤ t → IntervalIntegrable g volume 0 t := by
    intro t ht
    cases' eq_or_lt_of_le ht with h h
    · simp [← h]
    · exact g_intble t h
  have integrand_eq : ∀ ω,
      ENNReal.ofReal (∫ t in (0)..f ω, g t) = ∫⁻ t in Ioc 0 (f ω), ENNReal.ofReal (g t) := by
    intro ω
    have g_ae_nn : 0 ≤ᵐ[volume.restrict (Ioc 0 (f ω))] g := by
      filter_upwards [self_mem_ae_restrict (measurableSet_Ioc : MeasurableSet (Ioc 0 (f ω)))]
        with x hx using g_nn x hx.1
    rw [← ofReal_integral_eq_lintegral_ofReal (g_intble' (f ω) (f_nn ω)).1 g_ae_nn]
    congr
    exact intervalIntegral.integral_of_le (f_nn ω)
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
    integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  rw [lintegral_congr integrand_eq]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
    integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
    ⊢ Eq (MeasureTheory.lintegral μ fun a => MeasureTheory.lintegral (MeasureTheor …
  -/
  simp_rw [← lintegral_indicator measurableSet_Ioc]
  -- Porting note: was part of `simp_rw` on the previous line, but didn't trigger.
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
    integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
    ⊢ Eq (MeasureTheory.lintegral μ fun a => MeasureTheory.lintegral MeasureTheory …
  -/
  rw [← lintegral_indicator measurableSet_Ioi, lintegral_lintegral_swap]
    /-
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      ⊢ Eq (MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun y => Measu …
    -/
  · apply congr_arg
    /-
      case h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      ⊢ Eq (fun y => MeasureTheory.lintegral μ fun x => (Set.Ioc 0 (f x)).indicator  …
    -/
    funext s
    have aux₁ :
      (fun x => (Ioc 0 (f x)).indicator (fun t : ℝ => ENNReal.ofReal (g t)) s) = fun x =>
        ENNReal.ofReal (g s) * (Ioi (0 : ℝ)).indicator (fun _ => 1) s *
          (Ici s).indicator (fun _ : ℝ => (1 : ℝ≥0∞)) (f x) := by
      funext a
      by_cases h : s ∈ Ioc (0 : ℝ) (f a)
      · simp only [h, show s ∈ Ioi (0 : ℝ) from h.1, show f a ∈ Ici s from h.2, indicator_of_mem,
          mul_one]
      · have h_copy := h
        simp only [mem_Ioc, not_and, not_le] at h
        by_cases h' : 0 < s
        · simp only [h_copy, h h', indicator_of_not_mem, not_false_iff, mem_Ici, not_le, mul_zero]
        · have : s ∉ Ioi (0 : ℝ) := h'
          simp only [this, h', indicator_of_not_mem, not_false_iff, mul_zero,
            zero_mul, mem_Ioc, false_and]
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (MeasureTheory.lintegral μ fun x => (Set.Ioc 0 (f x)).indicator (fun t => …
    -/
    simp_rw [aux₁]
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (MeasureTheory.lintegral μ fun x => HMul.hMul (HMul.hMul (ENNReal.ofReal  …
    -/
    rw [lintegral_const_mul']
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (HMul.hMul (HMul.hMul (ENNReal.ofReal (g s)) ((Set.Ioi 0).indicator (fun  …
    -/
    swap
      /-
        case h.h.hr
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        f : α → Real
        g : Real → Real
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f_nn : LE.le 0 f
        f_mble : Measurable f
        g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
        g_mble : Measurable g
        g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
        g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
        integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
        s : Real
        aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
        ⊢ Ne (HMul.hMul (ENNReal.ofReal (g s)) ((Set.Ioi 0).indicator (fun x => 1) s)) …
      -/
    · apply ENNReal.mul_ne_top ENNReal.ofReal_ne_top
      /-
        case h.h.hr
        α : Type u_1
        inst✝¹ : MeasurableSpace α
        f : α → Real
        g : Real → Real
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        f_nn : LE.le 0 f
        f_mble : Measurable f
        g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
        g_mble : Measurable g
        g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
        g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
        integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
        s : Real
        aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
        ⊢ Ne ((Set.Ioi 0).indicator (fun x => 1) s) Top.top
      -/
                                     /-
                                       🎉 no goals
                                     -/
      by_cases h : (0 : ℝ) < s <;> · simp [h]
                                     /-
                                       🎉 no goals
                                     -/
    simp_rw [show
        (fun a => (Ici s).indicator (fun _ : ℝ => (1 : ℝ≥0∞)) (f a)) = fun a =>
          {a : α | s ≤ f a}.indicator (fun _ => 1) a
        by funext a; by_cases h : s ≤ f a <;> simp [h]]
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (HMul.hMul (HMul.hMul (ENNReal.ofReal (g s)) ((Set.Ioi 0).indicator (fun  …
    -/
    rw [lintegral_indicator₀]
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (HMul.hMul (HMul.hMul (ENNReal.ofReal (g s)) ((Set.Ioi 0).indicator (fun  …
    -/
    swap; · exact f_mble.nullMeasurable measurableSet_Ici
            /-
              🎉 no goals
            -/
    rw [lintegral_one, Measure.restrict_apply MeasurableSet.univ, univ_inter, indicator_mul_left,
      mul_assoc,
      show
        (Ioi 0).indicator (fun _x : ℝ => (1 : ℝ≥0∞)) s * μ {a : α | s ≤ f a} =
          (Ioi 0).indicator (fun _x : ℝ => 1 * μ {a : α | s ≤ f a}) s
        by by_cases h : 0 < s <;> simp [h]]
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (HMul.hMul (ENNReal.ofReal (g s)) ((Set.Ioi 0).indicator (fun _x => HMul. …
    -/
    simp_rw [mul_comm _ (ENNReal.ofReal _), one_mul]
    /-
      case h.h
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
      integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
      s : Real
      aux₁ : Eq (fun x => (Set.Ioc 0 (f x)).indicator (fun t => ENNReal.ofReal (g t) …
      ⊢ Eq (HMul.hMul (ENNReal.ofReal (g s)) ((Set.Ioi 0).indicator (fun _x => μ (se …
    -/
    rfl
    /-
      🎉 no goals
    -/
  have aux₂ :
    (Function.uncurry fun (x : α) (y : ℝ) =>
        (Ioc 0 (f x)).indicator (fun t : ℝ => ENNReal.ofReal (g t)) y) =
      {p : α × ℝ | p.2 ∈ Ioc 0 (f p.1)}.indicator fun p => ENNReal.ofReal (g p.2) := by
    funext p
    cases p with | mk p_fst p_snd => ?_
    rw [Function.uncurry_apply_pair]
    by_cases h : p_snd ∈ Ioc 0 (f p_fst)
    · have h' : (p_fst, p_snd) ∈ {p : α × ℝ | p.snd ∈ Ioc 0 (f p.fst)} := h
      rw [Set.indicator_of_mem h', Set.indicator_of_mem h]
    · have h' : (p_fst, p_snd) ∉ {p : α × ℝ | p.snd ∈ Ioc 0 (f p.fst)} := h
      rw [Set.indicator_of_not_mem h', Set.indicator_of_not_mem h]
  /-
    case hf
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    g_intble' : ∀ (t : Real), LE.le 0 t → IntervalIntegrable g MeasureTheory.Measu …
    integrand_eq : ∀ (ω : α), Eq (ENNReal.ofReal (intervalIntegral (fun t => g t)  …
    aux₂ : Eq (Function.uncurry fun x y => (Set.Ioc 0 (f x)).indicator (fun t => E …
    ⊢ AEMeasurable (Function.uncurry fun a => (Set.Ioc 0 (f a)).indicator fun t => …
  -/
  rw [aux₂]
  have mble₀ : MeasurableSet {p : α × ℝ | p.snd ∈ Ioc 0 (f p.fst)} := by
    simpa only [mem_univ, Pi.zero_apply, true_and] using
      measurableSet_region_between_oc measurable_zero f_mble MeasurableSet.univ
  exact (ENNReal.measurable_ofReal.comp (g_mble.comp measurable_snd)).aemeasurable.indicator₀
    mble₀.nullMeasurableSet


/-- An auxiliary version of the layer cake formula (Cavalieri's principle, tail probability
formula), with a measurability assumption that would also essentially follow from the
integrability assumptions.
Compared to `lintegral_comp_eq_lintegral_meas_le_mul_of_measurable_of_sigmaFinite`, we remove
the sigma-finite assumption.

See `MeasureTheory.lintegral_comp_eq_lintegral_meas_le_mul` and
`MeasureTheory.lintegral_comp_eq_lintegral_meas_lt_mul` for the main formulations of the layer
cake formula. -/
theorem lintegral_comp_eq_lintegral_meas_le_mul_of_measurable (μ : Measure α)
    (f_nn : 0 ≤ f) (f_mble : Measurable f)
    (g_intble : ∀ t > 0, IntervalIntegrable g volume 0 t) (g_mble : Measurable g)
    (g_nn : ∀ t > 0, 0 ≤ g t) :
    ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ =
      ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
  /- We will reduce to the sigma-finite case, after excluding two easy cases where the result
  is more or less obvious. -/
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  have f_nonneg : ∀ ω, 0 ≤ f ω := fun ω ↦ f_nn ω
  -- trivial case where `g` is ae zero. Then both integrals vanish.
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  by_cases H1 : g =ᵐ[volume.restrict (Ioi (0 : ℝ))] 0
  · have A : ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ = 0 := by
      have : ∀ ω, ∫ t in (0)..f ω, g t = ∫ t in (0)..f ω, 0 := by
        intro ω
        simp_rw [intervalIntegral.integral_of_le (f_nonneg ω)]
        apply integral_congr_ae
        exact ae_restrict_of_ae_restrict_of_subset Ioc_subset_Ioi_self H1
      simp [this]
    have B : ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) = 0 := by
      have : (fun t ↦ μ {a : α | t ≤ f a} * ENNReal.ofReal (g t))
        =ᵐ[volume.restrict (Ioi (0 : ℝ))] 0 := by
          filter_upwards [H1] with t ht using by simp [ht]
      simp [lintegral_congr_ae this]
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
      H1 : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioi 0) …
      A : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (f …
      B : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (S …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
    -/
    rw [A, B]
    /-
      🎉 no goals
    -/
  -- easy case where both sides are obviously infinite: for some `s`, one has
  -- `μ {a : α | s < f a} = ∞` and moreover `g` is not ae zero on `[0, s]`.
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  by_cases H2 : ∃ s > 0, 0 < ∫ t in (0)..s, g t ∧ μ {a : α | s < f a} = ∞
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
      H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
      H2 : Exists fun s => And (GT.gt s 0) (And (LT.lt 0 (intervalIntegral (fun t => …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
    -/
  · rcases H2 with ⟨s, s_pos, hs, h's⟩
    /-
      case pos.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
      H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
      s : Real
      s_pos : GT.gt s 0
      hs : LT.lt 0 (intervalIntegral (fun t => g t) 0 s MeasureTheory.MeasureSpace.v …
      h's : Eq (μ (setOf fun a => LT.lt s (f a))) Top.top
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
    -/
    rw [intervalIntegral.integral_of_le s_pos.le] at hs
    /- The first integral is infinite, as for `t ∈ [0, s]` one has `μ {a : α | t ≤ f a} = ∞`,
    and moreover the additional integral `g` is not uniformly zero. -/
    have A : ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) = ∞ := by
      rw [eq_top_iff]
      calc
      ∞ = ∫⁻ t in Ioc 0 s, ∞ * ENNReal.ofReal (g t) := by
          have I_pos : ∫⁻ (a : ℝ) in Ioc 0 s, ENNReal.ofReal (g a) ≠ 0 := by
            rw [← ofReal_integral_eq_lintegral_ofReal (g_intble s s_pos).1]
            · simpa only [not_lt, ne_eq, ENNReal.ofReal_eq_zero, not_le] using hs
            · filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht using g_nn _ ht.1
          rw [lintegral_const_mul, ENNReal.top_mul I_pos]
          exact ENNReal.measurable_ofReal.comp g_mble
      _ ≤ ∫⁻ t in Ioc 0 s, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
          apply setLIntegral_mono' measurableSet_Ioc (fun x hx ↦ ?_)
          rw [← h's]
          gcongr
          exact fun a ha ↦ hx.2.trans (le_of_lt ha)
      _ ≤ ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) :=
          lintegral_mono_set Ioc_subset_Ioi_self
    /- The second integral is infinite, as one integrates among other things on those `ω` where
    `f ω > s`: this is an infinite measure set, and on it the integrand is bounded below
    by `∫ t in 0..s, g t` which is positive. -/
    have B : ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ = ∞ := by
      rw [eq_top_iff]
      calc
      ∞ = ∫⁻ _ in {a | s < f a}, ENNReal.ofReal (∫ t in (0)..s, g t) ∂μ := by
          simp only [lintegral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter,
            h's, ne_eq, ENNReal.ofReal_eq_zero, not_le]
          rw [ENNReal.mul_top]
          simpa [intervalIntegral.integral_of_le s_pos.le] using hs
      _ ≤ ∫⁻ ω in {a | s < f a}, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ := by
          apply setLIntegral_mono' (measurableSet_lt measurable_const f_mble) (fun a ha ↦ ?_)
          apply ENNReal.ofReal_le_ofReal
          apply intervalIntegral.integral_mono_interval le_rfl s_pos.le (le_of_lt ha)
          · filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht using g_nn _ ht.1
          · exact g_intble _ (s_pos.trans ha)
      _ ≤ ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ := setLIntegral_le_lintegral _ _
    /-
      case pos.intro.intro.intro
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Real
      g : Real → Real
      μ : MeasureTheory.Measure α
      f_nn : LE.le 0 f
      f_mble : Measurable f
      g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
      g_mble : Measurable g
      g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
      f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
      H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
      s : Real
      s_pos : GT.gt s 0
      hs : LT.lt 0 (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restri …
      h's : Eq (μ (setOf fun a => LT.lt s (f a))) Top.top
      A : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (S …
      B : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (f …
      ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
    -/
    rw [A, B]
    /-
      🎉 no goals
    -/
  /- It remains to handle the interesting case, where `g` is not zero, but both integrals are
  not obviously infinite. Let `M` be the largest number such that `g = 0` on `[0, M]`. Then we
  may restrict `μ` to the points where `f ω > M` (as the other ones do not contribute to the
  integral). The restricted measure `ν` is sigma-finite, as `μ` gives finite measure to
  `{ω | f ω > a}` for any `a > M` (otherwise, we would be in the easy case above), so that
  one can write (a full measure subset of) the space as the countable union of the finite measure
  sets `{ω | f ω > uₙ}` for `uₙ` a sequence decreasing to `M`. Therefore,
  this case follows from the case where the measure is sigma-finite, applied to `ν`. -/
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    H2 : Not (Exists fun s => And (GT.gt s 0) (And (LT.lt 0 (intervalIntegral (fun …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  push_neg at H2
  have M_bdd : BddAbove {s : ℝ | g =ᵐ[volume.restrict (Ioc (0 : ℝ) s)] 0} := by
    contrapose! H1
    have : ∀ (n : ℕ), g =ᵐ[volume.restrict (Ioc (0 : ℝ) n)] 0 := by
      intro n
      rcases not_bddAbove_iff.1 H1 n with ⟨s, hs, ns⟩
      exact ae_restrict_of_ae_restrict_of_subset (Ioc_subset_Ioc_right ns.le) hs
    have Hg : g =ᵐ[volume.restrict (⋃ (n : ℕ), (Ioc (0 : ℝ) n))] 0 :=
      (ae_restrict_iUnion_iff _ _).2 this
    have : (⋃ (n : ℕ), (Ioc (0 : ℝ) n)) = Ioi 0 :=
      iUnion_Ioc_eq_Ioi_self_iff.2 (fun x _ ↦ exists_nat_ge x)
    rwa [this] at Hg
  -- let `M` be the largest number such that `g` vanishes ae on `(0, M]`.
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    H2 : ∀ (s : Real), GT.gt s 0 → LT.lt 0 (intervalIntegral (fun t => g t) 0 s Me …
    M_bdd : BddAbove (setOf fun s => (MeasureTheory.ae (MeasureTheory.MeasureSpace …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  let M : ℝ := sSup {s : ℝ | g =ᵐ[volume.restrict (Ioc (0 : ℝ) s)] 0}
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    H2 : ∀ (s : Real), GT.gt s 0 → LT.lt 0 (intervalIntegral (fun t => g t) 0 s Me …
    M_bdd : BddAbove (setOf fun s => (MeasureTheory.ae (MeasureTheory.MeasureSpace …
    M : Real := SupSet.sSup (setOf fun s => (MeasureTheory.ae (MeasureTheory.Measu …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  have zero_mem : 0 ∈ {s : ℝ | g =ᵐ[volume.restrict (Ioc (0 : ℝ) s)] 0} := by simpa using trivial
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    H2 : ∀ (s : Real), GT.gt s 0 → LT.lt 0 (intervalIntegral (fun t => g t) 0 s Me …
    M_bdd : BddAbove (setOf fun s => (MeasureTheory.ae (MeasureTheory.MeasureSpace …
    M : Real := SupSet.sSup (setOf fun s => (MeasureTheory.ae (MeasureTheory.Measu …
    zero_mem : Membership.mem (setOf fun s => (MeasureTheory.ae (MeasureTheory.Mea …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  have M_nonneg : 0 ≤ M := le_csSup M_bdd zero_mem
  -- Then the function `g` indeed vanishes ae on `(0, M]`.
  have hgM : g =ᵐ[volume.restrict (Ioc (0 : ℝ) M)] 0 := by
    rw [← restrict_Ioo_eq_restrict_Ioc]
    obtain ⟨u, -, uM, ulim⟩ : ∃ u, StrictMono u ∧ (∀ (n : ℕ), u n < M) ∧ Tendsto u atTop (𝓝 M) :=
      exists_seq_strictMono_tendsto M
    have I : ∀ n, g =ᵐ[volume.restrict (Ioc (0 : ℝ) (u n))] 0 := by
      intro n
      obtain ⟨s, hs, uns⟩ : ∃ s, g =ᶠ[ae (Measure.restrict volume (Ioc 0 s))] 0 ∧ u n < s :=
        exists_lt_of_lt_csSup (Set.nonempty_of_mem zero_mem) (uM n)
      exact ae_restrict_of_ae_restrict_of_subset (Ioc_subset_Ioc_right uns.le) hs
    have : g =ᵐ[volume.restrict (⋃ n, Ioc (0 : ℝ) (u n))] 0 := (ae_restrict_iUnion_iff _ _).2 I
    apply ae_restrict_of_ae_restrict_of_subset _ this
    rintro x ⟨x_pos, xM⟩
    obtain ⟨n, hn⟩ : ∃ n, x < u n := ((tendsto_order.1 ulim).1 _ xM).exists
    exact mem_iUnion.2 ⟨n, ⟨x_pos, hn.le⟩⟩
  -- Let `ν` be the restriction of `μ` to those points where `f a > M`.
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    H2 : ∀ (s : Real), GT.gt s 0 → LT.lt 0 (intervalIntegral (fun t => g t) 0 s Me …
    M_bdd : BddAbove (setOf fun s => (MeasureTheory.ae (MeasureTheory.MeasureSpace …
    M : Real := SupSet.sSup (setOf fun s => (MeasureTheory.ae (MeasureTheory.Measu …
    zero_mem : Membership.mem (setOf fun s => (MeasureTheory.ae (MeasureTheory.Mea …
    M_nonneg : LE.le 0 M
    hgM : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc 0 …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  let ν := μ.restrict {a : α | M < f a}
  -- This measure is sigma-finite (this is the whole point of the argument).
  have : SigmaFinite ν := by
    obtain ⟨u, -, uM, ulim⟩ : ∃ u, StrictAnti u ∧ (∀ (n : ℕ), M < u n) ∧ Tendsto u atTop (𝓝 M) :=
      exists_seq_strictAnti_tendsto M
    let s : ν.FiniteSpanningSetsIn univ :=
    { set := fun n ↦ {a | f a ≤ M} ∪ {a | u n < f a}
      set_mem := fun _ ↦ trivial
      finite := by
        intro n
        have I : ν {a | f a ≤ M} = 0 := by
          rw [Measure.restrict_apply (measurableSet_le f_mble measurable_const)]
          convert measure_empty (μ := μ)
          rw [← disjoint_iff_inter_eq_empty]
          exact disjoint_left.mpr (fun a ha ↦ by simpa using ha)
        have J : μ {a | u n < f a} < ∞ := by
          rw [lt_top_iff_ne_top]
          apply H2 _ (M_nonneg.trans_lt (uM n))
          by_contra H3
          rw [not_lt, intervalIntegral.integral_of_le (M_nonneg.trans (uM n).le)] at H3
          have g_nn_ae : ∀ᵐ t ∂(volume.restrict (Ioc 0 (u n))), 0 ≤ g t := by
            filter_upwards [ae_restrict_mem measurableSet_Ioc] with s hs using g_nn _ hs.1
          have Ig : ∫ (t : ℝ) in Ioc 0 (u n), g t = 0 :=
            le_antisymm H3 (integral_nonneg_of_ae g_nn_ae)
          have J : ∀ᵐ t ∂(volume.restrict (Ioc 0 (u n))), g t = 0 :=
            (integral_eq_zero_iff_of_nonneg_ae g_nn_ae
              (g_intble (u n) (M_nonneg.trans_lt (uM n))).1).1 Ig
          have : u n ≤ M := le_csSup M_bdd J
          exact lt_irrefl _ (this.trans_lt (uM n))
        refine lt_of_le_of_lt (measure_union_le _ _) ?_
        rw [I, zero_add]
        apply lt_of_le_of_lt _ J
        exact restrict_le_self _
      spanning := by
        apply eq_univ_iff_forall.2 (fun a ↦ ?_)
        rcases le_or_lt (f a) M with ha|ha
        · exact mem_iUnion.2 ⟨0, Or.inl ha⟩
        · obtain ⟨n, hn⟩ : ∃ n, u n < f a := ((tendsto_order.1 ulim).2 _ ha).exists
          exact mem_iUnion.2 ⟨n, Or.inr hn⟩ }
    exact ⟨⟨s⟩⟩
  -- the first integrals with respect to `μ` and to `ν` coincide, as points with `f a ≤ M` are
  -- weighted by zero as `g` vanishes there.
  have A : ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ
         = ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂ν := by
    have meas : MeasurableSet {a | M < f a} := measurableSet_lt measurable_const f_mble
    have I : ∫⁻ ω in {a | M < f a}ᶜ, ENNReal.ofReal (∫ t in (0).. f ω, g t) ∂μ
             = ∫⁻ _ in {a | M < f a}ᶜ, 0 ∂μ := by
      apply setLIntegral_congr_fun meas.compl (Eventually.of_forall (fun s hs ↦ ?_))
      have : ∫ (t : ℝ) in (0)..f s, g t = ∫ (t : ℝ) in (0)..f s, 0 := by
        simp_rw [intervalIntegral.integral_of_le (f_nonneg s)]
        apply integral_congr_ae
        apply ae_mono (restrict_mono ?_ le_rfl) hgM
        apply Ioc_subset_Ioc_right
        simpa using hs
      simp [this]
    simp only [lintegral_const, zero_mul] at I
    rw [← lintegral_add_compl _ meas, I, add_zero]
  -- the second integrals with respect to `μ` and to `ν` coincide, as points with `f a ≤ M` do not
  -- contribute to either integral since the weight `g` vanishes.
  have B : ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t)
           = ∫⁻ t in Ioi 0, ν {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
    have B1 : ∫⁻ t in Ioc 0 M, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t)
         = ∫⁻ t in Ioc 0 M, ν {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
      apply lintegral_congr_ae
      filter_upwards [hgM] with t ht
      simp [ht]
    have B2 : ∫⁻ t in Ioi M, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t)
              = ∫⁻ t in Ioi M, ν {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
      apply setLIntegral_congr_fun measurableSet_Ioi (Eventually.of_forall (fun t ht ↦ ?_))
      rw [Measure.restrict_apply (measurableSet_le measurable_const f_mble)]
      congr 3
      exact (inter_eq_left.2 (fun a ha ↦ (mem_Ioi.1 ht).trans_le ha)).symm
    have I : Ioi (0 : ℝ) = Ioc (0 : ℝ) M ∪ Ioi M := (Ioc_union_Ioi_eq_Ioi M_nonneg).symm
    have J : Disjoint (Ioc 0 M) (Ioi M) := Ioc_disjoint_Ioi le_rfl
    rw [I, lintegral_union measurableSet_Ioi J, lintegral_union measurableSet_Ioi J, B1, B2]
  -- therefore, we may replace the integrals wrt `μ` with integrals wrt `ν`, and apply the
  -- result for sigma-finite measures.
  /-
    case neg
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : LE.le 0 f
    f_mble : Measurable f
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_mble : Measurable g
    g_nn : ∀ (t : Real), GT.gt t 0 → LE.le 0 (g t)
    f_nonneg : ∀ (ω : α), LE.le 0 (f ω)
    H1 : Not ((MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.I …
    H2 : ∀ (s : Real), GT.gt s 0 → LT.lt 0 (intervalIntegral (fun t => g t) 0 s Me …
    M_bdd : BddAbove (setOf fun s => (MeasureTheory.ae (MeasureTheory.MeasureSpace …
    M : Real := SupSet.sSup (setOf fun s => (MeasureTheory.ae (MeasureTheory.Measu …
    zero_mem : Membership.mem (setOf fun s => (MeasureTheory.ae (MeasureTheory.Mea …
    M_nonneg : LE.le 0 M
    hgM : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc 0 …
    ν : MeasureTheory.Measure α := μ.restrict (setOf fun a => LT.lt M (f a))
    this : MeasureTheory.SigmaFinite ν
    A : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (f …
    B : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (S …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  rw [A, B]
  exact lintegral_comp_eq_lintegral_meas_le_mul_of_measurable_of_sigmaFinite
    ν f_nn f_mble g_intble g_mble g_nn


/-- The layer cake formula / **Cavalieri's principle** / tail probability formula:

Let `f` be a non-negative measurable function on a measure space. Let `G` be an
increasing absolutely continuous function on the positive real line, vanishing at the origin,
with derivative `G' = g`. Then the integral of the composition `G ∘ f` can be written as
the integral over the positive real line of the "tail measures" `μ {ω | f(ω) ≥ t}` of `f`
weighted by `g`.

Roughly speaking, the statement is: `∫⁻ (G ∘ f) ∂μ = ∫⁻ t in 0..∞, g(t) * μ {ω | f(ω) ≥ t}`.

See `MeasureTheory.lintegral_comp_eq_lintegral_meas_lt_mul` for a version with sets of the form
`{ω | f(ω) > t}` instead. -/
theorem lintegral_comp_eq_lintegral_meas_le_mul (μ : Measure α) (f_nn : 0 ≤ᵐ[μ] f)
    (f_mble : AEMeasurable f μ) (g_intble : ∀ t > 0, IntervalIntegrable g volume 0 t)
    (g_nn : ∀ᵐ t ∂volume.restrict (Ioi 0), 0 ≤ g t) :
    ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ =
      ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) := by
  obtain ⟨G, G_mble, G_nn, g_eq_G⟩ : ∃ G : ℝ → ℝ, Measurable G ∧ 0 ≤ G
      ∧ g =ᵐ[volume.restrict (Ioi 0)] G := by
    refine AEMeasurable.exists_measurable_nonneg ?_ g_nn
    exact aemeasurable_Ioi_of_forall_Ioc fun t ht => (g_intble t ht).1.1.aemeasurable
  have g_eq_G_on : ∀ t, g =ᵐ[volume.restrict (Ioc 0 t)] G := fun t =>
    ae_mono (Measure.restrict_mono Ioc_subset_Ioi_self le_rfl) g_eq_G
  have G_intble : ∀ t > 0, IntervalIntegrable G volume 0 t := by
    refine fun t t_pos => ⟨(g_intble t t_pos).1.congr_fun_ae (g_eq_G_on t), ?_⟩
    rw [Ioc_eq_empty_of_le t_pos.lt.le]
    exact integrableOn_empty
  obtain ⟨F, F_mble, F_nn, f_eq_F⟩ : ∃ F : α → ℝ, Measurable F ∧ 0 ≤ F ∧ f =ᵐ[μ] F := by
    refine ⟨fun ω ↦ max (f_mble.mk f ω) 0, f_mble.measurable_mk.max measurable_const,
        fun ω ↦ le_max_right _ _, ?_⟩
    filter_upwards [f_mble.ae_eq_mk, f_nn] with ω hω h'ω
    rw [← hω]
    exact (max_eq_left h'ω).symm
  have eq₁ :
    ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} * ENNReal.ofReal (g t) =
      ∫⁻ t in Ioi 0, μ {a : α | t ≤ F a} * ENNReal.ofReal (G t) := by
    apply lintegral_congr_ae
    filter_upwards [g_eq_G] with t ht
    rw [ht]
    congr 1
    apply measure_congr
    filter_upwards [f_eq_F] with a ha using by simp [setOf, ha]
  have eq₂ : ∀ᵐ ω ∂μ,
      ENNReal.ofReal (∫ t in (0)..f ω, g t) = ENNReal.ofReal (∫ t in (0)..F ω, G t) := by
    filter_upwards [f_eq_F] with ω fω_nn
    rw [fω_nn]
    congr 1
    refine intervalIntegral.integral_congr_ae ?_
    have fω_nn : 0 ≤ F ω := F_nn ω
    rw [uIoc_of_le fω_nn, ←
      ae_restrict_iff' (measurableSet_Ioc : MeasurableSet (Ioc (0 : ℝ) (F ω)))]
    exact g_eq_G_on (F ω)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
    G : Real → Real
    G_mble : Measurable G
    G_nn : LE.le 0 G
    g_eq_G : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Io …
    g_eq_G_on : ∀ (t : Real), (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume …
    G_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable G MeasureTheory.Measur …
    F : α → Real
    F_mble : Measurable F
    F_nn : LE.le 0 F
    f_eq_F : (MeasureTheory.ae μ).EventuallyEq f F
    eq₁ : Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict  …
    eq₂ : Filter.Eventually (fun ω => Eq (ENNReal.ofReal (intervalIntegral (fun t  …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  simp_rw [lintegral_congr_ae eq₂, eq₁]
  exact lintegral_comp_eq_lintegral_meas_le_mul_of_measurable μ F_nn F_mble
          G_intble G_mble (fun t _ => G_nn t)


/-- The standard case of the layer cake formula / Cavalieri's principle / tail probability formula:

For a nonnegative function `f` on a measure space, the Lebesgue integral of `f` can
be written (roughly speaking) as: `∫⁻ f ∂μ = ∫⁻ t in 0..∞, μ {ω | f(ω) ≥ t}`.

See `MeasureTheory.lintegral_eq_lintegral_meas_lt` for a version with sets of the form
`{ω | f(ω) > t}` instead. -/
theorem lintegral_eq_lintegral_meas_le (μ : Measure α) (f_nn : 0 ≤ᵐ[μ] f)
    (f_mble : AEMeasurable f μ) :
    ∫⁻ ω, ENNReal.ofReal (f ω) ∂μ = ∫⁻ t in Ioi 0, μ {a : α | t ≤ f a} := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureTheory. …
  -/
  set cst := fun _ : ℝ => (1 : ℝ)
  have cst_intble : ∀ t > 0, IntervalIntegrable cst volume 0 t := fun _ _ =>
    intervalIntegrable_const
  have key :=
    lintegral_comp_eq_lintegral_meas_le_mul μ f_nn f_mble cst_intble
      (Eventually.of_forall fun _ => zero_le_one)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    cst : Real → Real := fun x => 1
    cst_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable cst MeasureTheory.Me …
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureTheory. …
  -/
  simp_rw [cst, ENNReal.ofReal_one, mul_one] at key
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    cst : Real → Real := fun x => 1
    cst_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable cst MeasureTheory.Me …
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureTheory. …
  -/
  rw [← key]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    cst : Real → Real := fun x => 1
    cst_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable cst MeasureTheory.Me …
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureTheory. …
  -/
  congr with ω
  /-
    case e_f.h
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    cst : Real → Real := fun x => 1
    cst_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable cst MeasureTheory.Me …
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral  …
    ω : α
    ⊢ Eq (ENNReal.ofReal (f ω)) (ENNReal.ofReal (intervalIntegral (fun t => 1) 0 ( …
  -/
  simp only [intervalIntegral.integral_const, sub_zero, Algebra.id.smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


/-- The layer cake formula / Cavalieri's principle / tail probability formula:

Let `f` be a non-negative measurable function on a measure space. Let `G` be an
increasing absolutely continuous function on the positive real line, vanishing at the origin,
with derivative `G' = g`. Then the integral of the composition `G ∘ f` can be written as
the integral over the positive real line of the "tail measures" `μ {ω | f(ω) > t}` of `f`
weighted by `g`.

Roughly speaking, the statement is: `∫⁻ (G ∘ f) ∂μ = ∫⁻ t in 0..∞, g(t) * μ {ω | f(ω) > t}`.

See `lintegral_comp_eq_lintegral_meas_le_mul` for a version with sets of the form `{ω | f(ω) ≥ t}`
instead. -/
theorem lintegral_comp_eq_lintegral_meas_lt_mul (μ : Measure α) (f_nn : 0 ≤ᵐ[μ] f)
    (f_mble : AEMeasurable f μ) (g_intble : ∀ t > 0, IntervalIntegrable g volume 0 t)
    (g_nn : ∀ᵐ t ∂volume.restrict (Ioi 0), 0 ≤ g t) :
    ∫⁻ ω, ENNReal.ofReal (∫ t in (0)..f ω, g t) ∂μ =
      ∫⁻ t in Ioi 0, μ {a : α | t < f a} * ENNReal.ofReal (g t) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (intervalIntegral (fun …
  -/
  rw [lintegral_comp_eq_lintegral_meas_le_mul μ f_nn f_mble g_intble g_nn]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
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
    g : Real → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    g_intble : ∀ (t : Real), GT.gt t 0 → IntervalIntegrable g MeasureTheory.Measur …
    g_nn : Filter.Eventually (fun t => LE.le 0 (g t)) (MeasureTheory.ae (MeasureTh …
    t : Real
    ht : Eq (μ (setOf fun a => LE.le t (f a))) (μ (setOf fun a => LT.lt t (f a)))
    ⊢ Eq (HMul.hMul (μ (setOf fun a => LE.le t (f a))) (ENNReal.ofReal (g t))) (HM …
  -/
  rw [ht]
  /-
    🎉 no goals
  -/


/-- The standard case of the layer cake formula / Cavalieri's principle / tail probability formula:

For a nonnegative function `f` on a measure space, the Lebesgue integral of `f` can
be written (roughly speaking) as: `∫⁻ f ∂μ = ∫⁻ t in 0..∞, μ {ω | f(ω) > t}`.

See `lintegral_eq_lintegral_meas_le` for a version with sets of the form `{ω | f(ω) ≥ t}`
instead. -/
theorem lintegral_eq_lintegral_meas_lt (μ : Measure α)
    (f_nn : 0 ≤ᵐ[μ] f) (f_mble : AEMeasurable f μ) :
    ∫⁻ ω, ENNReal.ofReal (f ω) ∂μ = ∫⁻ t in Ioi 0, μ {a : α | t < f a} := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
    ⊢ Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureTheory. …
  -/
  rw [lintegral_eq_lintegral_meas_le μ f_nn f_mble]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    μ : MeasureTheory.Measure α
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_mble : AEMeasurable f μ
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
    t : Real
    ht : Eq (μ (setOf fun a => LE.le t (f a))) (μ (setOf fun a => LT.lt t (f a)))
    ⊢ Eq (μ (setOf fun a => LE.le t (f a))) (μ (setOf fun a => LT.lt t (f a)))
  -/
  rw [ht]
  /-
    🎉 no goals
  -/


/-- The standard case of the layer cake formula / Cavalieri's principle / tail probability formula:

For an integrable a.e.-nonnegative real-valued function `f`, the Bochner integral of `f` can be
written (roughly speaking) as: `∫ f ∂μ = ∫ t in 0..∞, μ {ω | f(ω) > t}`.

See `MeasureTheory.lintegral_eq_lintegral_meas_lt` for a version with Lebesgue integral `∫⁻`
instead. -/
theorem Integrable.integral_eq_integral_meas_lt
    (f_intble : Integrable f μ) (f_nn : 0 ≤ᵐ[μ] f) :
    ∫ ω, f ω ∂μ = ∫ t in Set.Ioi 0, ENNReal.toReal (μ {a : α | t < f a}) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  have key := lintegral_eq_lintegral_meas_lt μ f_nn f_intble.aemeasurable
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  have lhs_finite : ∫⁻ (ω : α), ENNReal.ofReal (f ω) ∂μ < ∞ := Integrable.lintegral_lt_top f_intble
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
    lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  have rhs_finite : ∫⁻ (t : ℝ) in Set.Ioi 0, μ {a | t < f a} < ∞ := by simp only [← key, lhs_finite]
  have rhs_integrand_finite : ∀ (t : ℝ), t > 0 → μ {a | t < f a} < ∞ :=
    fun t ht ↦ measure_gt_lt_top f_intble ht
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
    lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
    rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
    rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  convert (ENNReal.toReal_eq_toReal lhs_finite.ne rhs_finite.ne).mpr key
    /-
      case h.e'_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      f_intble : MeasureTheory.Integrable f μ
      f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
      key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
      lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
      rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
      rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
      ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.lintegral μ fun ω  …
    -/
  · exact integral_eq_lintegral_of_nonneg_ae f_nn f_intble.aestronglyMeasurable
    /-
      🎉 no goals
    -/
  · have aux := @integral_eq_lintegral_of_nonneg_ae _ _ ((volume : Measure ℝ).restrict (Set.Ioi 0))
      (fun t ↦ ENNReal.toReal (μ {a : α | t < f a})) ?_ ?_
      /-
        case h.e'_3.refine_3
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        aux : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ( …
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
      -/
    · rw [aux]
      /-
        case h.e'_3.refine_3
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        aux : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ( …
        ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
      -/
      congr 1
      /-
        case h.e'_3.refine_3.e_a
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        aux : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ( …
        ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume.restrict (Set …
      -/
      apply setLIntegral_congr_fun measurableSet_Ioi (Eventually.of_forall _)
      /-
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        aux : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ( …
        ⊢ ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Eq (ENNReal.ofReal (μ (setOf fu …
      -/
      exact fun t t_pos ↦ ENNReal.ofReal_toReal (rhs_integrand_finite t t_pos).ne
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.refine_1
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioi 0))). …
      -/
    · exact Eventually.of_forall (fun x ↦ by simp only [Pi.zero_apply, ENNReal.toReal_nonneg])
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.refine_2
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        ⊢ MeasureTheory.AEStronglyMeasurable (fun t => (μ (setOf fun a => LT.lt t (f a …
      -/
    · apply Measurable.aestronglyMeasurable
      /-
        case h.e'_3.refine_2.hf
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        ⊢ Measurable fun t => (μ (setOf fun a => LT.lt t (f a))).toReal
      -/
      refine Measurable.ennreal_toReal ?_
      /-
        case h.e'_3.refine_2.hf
        α : Type u_1
        inst✝ : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        f_intble : MeasureTheory.Integrable f μ
        f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
        key : Eq (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) (MeasureThe …
        lhs_finite : LT.lt (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (f ω)) T …
        rhs_finite : LT.lt (MeasureTheory.lintegral (MeasureTheory.MeasureSpace.volume …
        rhs_integrand_finite : ∀ (t : Real), GT.gt t 0 → LT.lt (μ (setOf fun a => LT.l …
        ⊢ Measurable fun t => μ (setOf fun a => LT.lt t (f a))
      -/
      exact Antitone.measurable (fun _ _ hst ↦ measure_mono (fun _ h ↦ lt_of_le_of_lt hst h))
      /-
        🎉 no goals
      -/


theorem Integrable.integral_eq_integral_meas_le
    (f_intble : Integrable f μ) (f_nn : 0 ≤ᵐ[μ] f) :
    ∫ ω, f ω ∂μ = ∫ t in Set.Ioi 0, ENNReal.toReal (μ {a : α | t ≤ f a}) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  rw [Integrable.integral_eq_integral_meas_lt f_intble f_nn]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  apply integral_congr_ae
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioi 0))). …
  -/
  filter_upwards [meas_le_ae_eq_meas_lt μ (volume.restrict (Ioi 0)) f] with t ht
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    t : Real
    ht : Eq (μ (setOf fun a => LE.le t (f a))) (μ (setOf fun a => LT.lt t (f a)))
    ⊢ Eq (μ (setOf fun a => LT.lt t (f a))).toReal (μ (setOf fun a => LE.le t (f a …
  -/
  exact congrArg ENNReal.toReal ht.symm
  /-
    🎉 no goals
  -/


lemma Integrable.integral_eq_integral_Ioc_meas_le {f : α → ℝ} {M : ℝ}
    (f_intble : Integrable f μ) (f_nn : 0 ≤ᵐ[μ] f) (f_bdd : f ≤ᵐ[μ] (fun _ ↦ M)) :
    ∫ ω, f ω ∂μ = ∫ t in Ioc 0 M, ENNReal.toReal (μ {a : α | t ≤ f a}) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    M : Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_bdd : (MeasureTheory.ae μ).EventuallyLE f fun x => M
    ⊢ Eq (MeasureTheory.integral μ fun ω => f ω) (MeasureTheory.integral (MeasureT …
  -/
  rw [f_intble.integral_eq_integral_meas_le f_nn]
  rw [setIntegral_eq_of_subset_of_ae_diff_eq_zero
      nullMeasurableSet_Ioi Ioc_subset_Ioi_self ?_]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    M : Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_bdd : (MeasureTheory.ae μ).EventuallyLE f fun x => M
    ⊢ Filter.Eventually (fun x => Membership.mem (SDiff.sdiff (Set.Ioi 0) (Set.Ioc …
  -/
  apply Eventually.of_forall (fun t ht ↦ ?_)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    M : Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_bdd : (MeasureTheory.ae μ).EventuallyLE f fun x => M
    t : Real
    ht : Membership.mem (SDiff.sdiff (Set.Ioi 0) (Set.Ioc 0 M)) t
    ⊢ Eq (μ (setOf fun a => LE.le t (f a))).toReal 0
  -/
  have htM : M < t := by simp_all only [mem_diff, mem_Ioi, mem_Ioc, not_and, not_le]
  have obs : μ {a | M < f a} = 0 := by
    rw [measure_zero_iff_ae_nmem]
    filter_upwards [f_bdd] with a ha using not_lt.mpr ha
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    M : Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_bdd : (MeasureTheory.ae μ).EventuallyLE f fun x => M
    t : Real
    ht : Membership.mem (SDiff.sdiff (Set.Ioi 0) (Set.Ioc 0 M)) t
    htM : LT.lt M t
    obs : Eq (μ (setOf fun a => LT.lt M (f a))) 0
    ⊢ Eq (μ (setOf fun a => LE.le t (f a))).toReal 0
  -/
  rw [ENNReal.toReal_eq_zero_iff]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    M : Real
    f_intble : MeasureTheory.Integrable f μ
    f_nn : (MeasureTheory.ae μ).EventuallyLE 0 f
    f_bdd : (MeasureTheory.ae μ).EventuallyLE f fun x => M
    t : Real
    ht : Membership.mem (SDiff.sdiff (Set.Ioi 0) (Set.Ioc 0 M)) t
    htM : LT.lt M t
    obs : Eq (μ (setOf fun a => LT.lt M (f a))) 0
    ⊢ Or (Eq (μ (setOf fun a => LE.le t (f a))) 0) (Eq (μ (setOf fun a => LE.le t  …
  -/
  exact Or.inl <| measure_mono_null (fun a ha ↦ lt_of_lt_of_le htM ha) obs
  /-
    🎉 no goals
  -/


