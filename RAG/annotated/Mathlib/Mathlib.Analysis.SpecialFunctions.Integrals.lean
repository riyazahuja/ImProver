@[simp]
theorem intervalIntegrable_pow : IntervalIntegrable (fun x => x ^ n) μ a b :=
  (continuous_pow n).intervalIntegrable a b


theorem intervalIntegrable_zpow {n : ℤ} (h : 0 ≤ n ∨ (0 : ℝ) ∉ [[a, b]]) :
    IntervalIntegrable (fun x => x ^ n) μ a b :=
  (continuousOn_id.zpow₀ n fun _ hx => h.symm.imp (ne_of_mem_of_not_mem hx) id).intervalIntegrable


/-- See `intervalIntegrable_rpow'` for a version with a weaker hypothesis on `r`, but assuming the
measure is volume. -/
theorem intervalIntegrable_rpow {r : ℝ} (h : 0 ≤ r ∨ (0 : ℝ) ∉ [[a, b]]) :
    IntervalIntegrable (fun x => x ^ r) μ a b :=
  (continuousOn_id.rpow_const fun _ hx =>
    h.symm.imp (ne_of_mem_of_not_mem hx) id).intervalIntegrable


/-- See `intervalIntegrable_rpow` for a version applying to any locally finite measure, but with a
stronger hypothesis on `r`. -/
theorem intervalIntegrable_rpow' {r : ℝ} (h : -1 < r) :
    IntervalIntegrable (fun x => x ^ r) volume a b := by
  suffices ∀ c : ℝ, IntervalIntegrable (fun x => x ^ r) volume 0 c by
    exact IntervalIntegrable.trans (this a).symm (this b)
  have : ∀ c : ℝ, 0 ≤ c → IntervalIntegrable (fun x => x ^ r) volume 0 c := by
    intro c hc
    rw [intervalIntegrable_iff, uIoc_of_le hc]
    have hderiv : ∀ x ∈ Ioo 0 c, HasDerivAt (fun x : ℝ => x ^ (r + 1) / (r + 1)) (x ^ r) x := by
      intro x hx
      convert (Real.hasDerivAt_rpow_const (p := r + 1) (Or.inl hx.1.ne')).div_const (r + 1) using 1
      field_simp [(by linarith : r + 1 ≠ 0)]
    apply integrableOn_deriv_of_nonneg _ hderiv
    · intro x hx; apply rpow_nonneg hx.1.le
    · refine (continuousOn_id.rpow_const ?_).div_const _; intro x _; right; linarith
  /-
    a b r : Real
    h : LT.lt (-1) r
    this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
    ⊢ ∀ (c : Real), IntervalIntegrable (fun x => HPow.hPow x r) MeasureTheory.Meas …
  -/
  intro c; rcases le_total 0 c with (hc | hc)
    /-
      case inl
      a b r : Real
      h : LT.lt (-1) r
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
      c : Real
      hc : LE.le 0 c
      ⊢ IntervalIntegrable (fun x => HPow.hPow x r) MeasureTheory.MeasureSpace.volum …
    -/
  · exact this c hc
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b r : Real
      h : LT.lt (-1) r
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
      c : Real
      hc : LE.le c 0
      ⊢ IntervalIntegrable (fun x => HPow.hPow x r) MeasureTheory.MeasureSpace.volum …
    -/
  · rw [IntervalIntegrable.iff_comp_neg, neg_zero]
    /-
      case inr
      a b r : Real
      h : LT.lt (-1) r
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
      c : Real
      hc : LE.le c 0
      ⊢ IntervalIntegrable (fun x => HPow.hPow (Neg.neg x) r) MeasureTheory.MeasureS …
    -/
    have m := (this (-c) (by linarith)).smul (cos (r * π))
    /-
      case inr
      a b r : Real
      h : LT.lt (-1) r
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
      c : Real
      hc : LE.le c 0
      m : IntervalIntegrable (HSMul.hSMul (Real.cos (HMul.hMul r Real.pi)) fun x =>  …
      ⊢ IntervalIntegrable (fun x => HPow.hPow (Neg.neg x) r) MeasureTheory.MeasureS …
    -/
    rw [intervalIntegrable_iff] at m ⊢
    /-
      case inr
      a b r : Real
      h : LT.lt (-1) r
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (HSMul.hSMul (Real.cos (HMul.hMul r Real.pi)) f …
      ⊢ MeasureTheory.IntegrableOn (fun x => HPow.hPow (Neg.neg x) r) (Set.uIoc 0 (N …
    -/
    refine m.congr_fun ?_ measurableSet_Ioc; intro x hx
    /-
      case inr
      a b r : Real
      h : LT.lt (-1) r
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow x r) M …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (HSMul.hSMul (Real.cos (HMul.hMul r Real.pi)) f …
      x : Real
      hx : Membership.mem (Set.uIoc 0 (Neg.neg c)) x
      ⊢ Eq (HSMul.hSMul (Real.cos (HMul.hMul r Real.pi)) (fun x => HPow.hPow x r) x) …
    -/
    rw [uIoc_of_le (by linarith : 0 ≤ -c)] at hx
    simp only [Pi.smul_apply, Algebra.id.smul_eq_mul, log_neg_eq_log, mul_comm,
      rpow_def_of_pos hx.1, rpow_def_of_neg (by linarith [hx.1] : -x < 0)]


/-- The power function `x ↦ x^s` is integrable on `(0, t)` iff `-1 < s`. -/
lemma integrableOn_Ioo_rpow_iff {s t : ℝ} (ht : 0 < t) :
    /-
      a b : Real
      n : Nat
      f : Real → Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      c d s t : Real
      ht : LT.lt 0 t
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x ↦ x ^ s) (Ioo (0 : ℝ) t) ↔ -1 < s := by
    /-
      🎉 no goals
    -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simpa [intervalIntegrable_iff_integrableOn_Ioo_of_le ht.le]
    using intervalIntegrable_rpow' h (a := 0) (b := t)⟩
  /-
    s t : Real
    ht : LT.lt 0 t
    h : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioo 0 t) MeasureT …
    ⊢ LT.lt (-1) s
  -/
  contrapose! h
  /-
    s t : Real
    ht : LT.lt 0 t
    h : LE.le s (-1)
    ⊢ Not (MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioo 0 t) Measu …
  -/
  intro H
  /-
    s t : Real
    ht : LT.lt 0 t
    h : LE.le s (-1)
    H : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioo 0 t) MeasureT …
    ⊢ False
  -/
  have I : 0 < min 1 t := lt_min zero_lt_one ht
  have H' : IntegrableOn (fun x ↦ x ^ s) (Ioo 0 (min 1 t)) :=
    H.mono (Set.Ioo_subset_Ioo le_rfl (min_le_right _ _)) le_rfl
  have : IntegrableOn (fun x ↦ x⁻¹) (Ioo 0 (min 1 t)) := by
    apply H'.mono' measurable_inv.aestronglyMeasurable
    filter_upwards [ae_restrict_mem measurableSet_Ioo] with x hx
    simp only [norm_inv, Real.norm_eq_abs, abs_of_nonneg (le_of_lt hx.1)]
    rwa [← Real.rpow_neg_one x, Real.rpow_le_rpow_left_iff_of_base_lt_one hx.1]
    exact lt_of_lt_of_le hx.2 (min_le_left _ _)
  have : IntervalIntegrable (fun x ↦ x⁻¹) volume 0 (min 1 t) := by
    rwa [intervalIntegrable_iff_integrableOn_Ioo_of_le I.le]
  /-
    s t : Real
    ht : LT.lt 0 t
    h : LE.le s (-1)
    H : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioo 0 t) MeasureT …
    I : LT.lt 0 (Min.min 1 t)
    H' : MeasureTheory.IntegrableOn (fun x => HPow.hPow x s) (Set.Ioo 0 (Min.min 1 …
    this✝ : MeasureTheory.IntegrableOn (fun x => Inv.inv x) (Set.Ioo 0 (Min.min 1  …
    this : IntervalIntegrable (fun x => Inv.inv x) MeasureTheory.MeasureSpace.volu …
    ⊢ False
  -/
  simp [intervalIntegrable_inv_iff, I.ne] at this
  /-
    🎉 no goals
  -/


/-- See `intervalIntegrable_cpow'` for a version with a weaker hypothesis on `r`, but assuming the
measure is volume. -/
theorem intervalIntegrable_cpow {r : ℂ} (h : 0 ≤ r.re ∨ (0 : ℝ) ∉ [[a, b]]) :
    IntervalIntegrable (fun x : ℝ => (x : ℂ) ^ r) μ a b := by
  /-
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    r : Complex
    h : Or (LE.le 0 r.re) (Not (Membership.mem (Set.uIcc a b) 0))
    ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) μ a b
  -/
  by_cases h2 : (0 : ℝ) ∉ [[a, b]]
  · -- Easy case #1: 0 ∉ [a, b] -- use continuity.
    /-
      case pos
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : Or (LE.le 0 r.re) (Not (Membership.mem (Set.uIcc a b) 0))
      h2 : Not (Membership.mem (Set.uIcc a b) 0)
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) μ a b
    -/
    refine (continuousOn_of_forall_continuousAt fun x hx => ?_).intervalIntegrable
    /-
      case pos
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : Or (LE.le 0 r.re) (Not (Membership.mem (Set.uIcc a b) 0))
      h2 : Not (Membership.mem (Set.uIcc a b) 0)
      x : Real
      hx : Membership.mem (Set.uIcc a b) x
      ⊢ ContinuousAt (fun x => HPow.hPow (↑x) r) x
    -/
    exact Complex.continuousAt_ofReal_cpow_const _ _ (Or.inr <| ne_of_mem_of_not_mem hx h2)
    /-
      🎉 no goals
    -/
  /-
    case neg
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    r : Complex
    h : Or (LE.le 0 r.re) (Not (Membership.mem (Set.uIcc a b) 0))
    h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
    ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) μ a b
  -/
  rw [eq_false h2, or_false] at h
  /-
    case neg
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    r : Complex
    h : LE.le 0 r.re
    h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
    ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) μ a b
  -/
  rcases lt_or_eq_of_le h with (h' | h')
  · -- Easy case #2: 0 < re r -- again use continuity
    /-
      case neg.inl
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : LT.lt 0 r.re
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) μ a b
    -/
    exact (Complex.continuous_ofReal_cpow_const h').intervalIntegrable _ _
    /-
      🎉 no goals
    -/
  -- Now the hard case: re r = 0 and 0 is in the interval.
  /-
    case neg.inr
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    r : Complex
    h : LE.le 0 r.re
    h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
    h' : Eq 0 r.re
    ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) μ a b
  -/
  refine (IntervalIntegrable.intervalIntegrable_norm_iff ?_).mp ?_
    /-
      case neg.inr.refine_1
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HPow.hPow (↑x) r) (μ.restrict ( …
    -/
  · refine (measurable_of_continuousOn_compl_singleton (0 : ℝ) ?_).aestronglyMeasurable
    exact continuousOn_of_forall_continuousAt fun x hx =>
      Complex.continuousAt_ofReal_cpow_const x r (Or.inr hx)
  -- reduce to case of integral over `[0, c]`
  suffices ∀ c : ℝ, IntervalIntegrable (fun x : ℝ => ‖(x : ℂ) ^ r‖) μ 0 c from
    (this a).symm.trans (this b)
  /-
    case neg.inr.refine_2
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    r : Complex
    h : LE.le 0 r.re
    h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
    h' : Eq 0 r.re
    ⊢ ∀ (c : Real), IntervalIntegrable (fun x => Norm.norm (HPow.hPow (↑x) r)) μ 0 c
  -/
  intro c
  /-
    case neg.inr.refine_2
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    r : Complex
    h : LE.le 0 r.re
    h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
    h' : Eq 0 r.re
    c : Real
    ⊢ IntervalIntegrable (fun x => Norm.norm (HPow.hPow (↑x) r)) μ 0 c
  -/
  rcases le_or_lt 0 c with (hc | hc)
  · -- case `0 ≤ c`: integrand is identically 1
    /-
      case neg.inr.refine_2.inl
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LE.le 0 c
      ⊢ IntervalIntegrable (fun x => Norm.norm (HPow.hPow (↑x) r)) μ 0 c
    -/
    have : IntervalIntegrable (fun _ => 1 : ℝ → ℝ) μ 0 c := intervalIntegrable_const
    /-
      case neg.inr.refine_2.inl
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LE.le 0 c
      this : IntervalIntegrable (fun x => 1) μ 0 c
      ⊢ IntervalIntegrable (fun x => Norm.norm (HPow.hPow (↑x) r)) μ 0 c
    -/
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hc] at this ⊢
    /-
      case neg.inr.refine_2.inl
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LE.le 0 c
      this : MeasureTheory.IntegrableOn (fun x => 1) (Set.Ioc 0 c) μ
      ⊢ MeasureTheory.IntegrableOn (fun x => Norm.norm (HPow.hPow (↑x) r)) (Set.Ioc  …
    -/
    refine IntegrableOn.congr_fun this (fun x hx => ?_) measurableSet_Ioc
    /-
      case neg.inr.refine_2.inl
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LE.le 0 c
      this : MeasureTheory.IntegrableOn (fun x => 1) (Set.Ioc 0 c) μ
      x : Real
      hx : Membership.mem (Set.Ioc 0 c) x
      ⊢ Eq 1 (Norm.norm (HPow.hPow (↑x) r))
    -/
    dsimp only
    /-
      case neg.inr.refine_2.inl
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LE.le 0 c
      this : MeasureTheory.IntegrableOn (fun x => 1) (Set.Ioc 0 c) μ
      x : Real
      hx : Membership.mem (Set.Ioc 0 c) x
      ⊢ Eq 1 (Norm.norm (HPow.hPow (↑x) r))
    -/
    rw [Complex.norm_eq_abs, Complex.abs_cpow_eq_rpow_re_of_pos hx.1, ← h', rpow_zero]
    /-
      🎉 no goals
    -/
  · -- case `c < 0`: integrand is identically constant, *except* at `x = 0` if `r ≠ 0`.
    /-
      case neg.inr.refine_2.inr
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LT.lt c 0
      ⊢ IntervalIntegrable (fun x => Norm.norm (HPow.hPow (↑x) r)) μ 0 c
    -/
    apply IntervalIntegrable.symm
    /-
      case neg.inr.refine_2.inr.h
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LT.lt c 0
      ⊢ IntervalIntegrable (fun x => Norm.norm (HPow.hPow (↑x) r)) μ c 0
    -/
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hc.le]
    /-
      case neg.inr.refine_2.inr.h
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      r : Complex
      h : LE.le 0 r.re
      h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h' : Eq 0 r.re
      c : Real
      hc : LT.lt c 0
      ⊢ MeasureTheory.IntegrableOn (fun x => Norm.norm (HPow.hPow (↑x) r)) (Set.Ioc  …
    -/
    rw [← Ioo_union_right hc, integrableOn_union, and_comm]; constructor
      /-
        case neg.inr.refine_2.inr.h.left
        a b : Real
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        r : Complex
        h : LE.le 0 r.re
        h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h' : Eq 0 r.re
        c : Real
        hc : LT.lt c 0
        ⊢ MeasureTheory.IntegrableOn (fun x => Norm.norm (HPow.hPow (↑x) r)) (Singleto …
      -/
    · refine integrableOn_singleton_iff.mpr (Or.inr ?_)
      exact isFiniteMeasureOnCompacts_of_isLocallyFiniteMeasure.lt_top_of_isCompact
        isCompact_singleton
    · have : ∀ x : ℝ, x ∈ Ioo c 0 → ‖Complex.exp (↑π * Complex.I * r)‖ = ‖(x : ℂ) ^ r‖ := by
        intro x hx
        rw [Complex.ofReal_cpow_of_nonpos hx.2.le, norm_mul, ← Complex.ofReal_neg,
          Complex.norm_eq_abs (_ ^ _), Complex.abs_cpow_eq_rpow_re_of_pos (neg_pos.mpr hx.2), ← h',
          rpow_zero, one_mul]
      /-
        case neg.inr.refine_2.inr.h.right
        a b : Real
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        r : Complex
        h : LE.le 0 r.re
        h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h' : Eq 0 r.re
        c : Real
        hc : LT.lt c 0
        this : ∀ (x : Real), Membership.mem (Set.Ioo c 0) x → Eq (Norm.norm (Complex.e …
        ⊢ MeasureTheory.IntegrableOn (fun x => Norm.norm (HPow.hPow (↑x) r)) (Set.Ioo  …
      -/
      refine IntegrableOn.congr_fun ?_ this measurableSet_Ioo
      /-
        case neg.inr.refine_2.inr.h.right
        a b : Real
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        r : Complex
        h : LE.le 0 r.re
        h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h' : Eq 0 r.re
        c : Real
        hc : LT.lt c 0
        this : ∀ (x : Real), Membership.mem (Set.Ioo c 0) x → Eq (Norm.norm (Complex.e …
        ⊢ MeasureTheory.IntegrableOn (fun x => Norm.norm (Complex.exp (HMul.hMul (HMul …
      -/
      rw [integrableOn_const]
      /-
        case neg.inr.refine_2.inr.h.right
        a b : Real
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        r : Complex
        h : LE.le 0 r.re
        h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h' : Eq 0 r.re
        c : Real
        hc : LT.lt c 0
        this : ∀ (x : Real), Membership.mem (Set.Ioo c 0) x → Eq (Norm.norm (Complex.e …
        ⊢ Or (Eq (Norm.norm (Complex.exp (HMul.hMul (HMul.hMul (↑Real.pi) Complex.I) r …
      -/
      refine Or.inr ((measure_mono Set.Ioo_subset_Icc_self).trans_lt ?_)
      /-
        case neg.inr.refine_2.inr.h.right
        a b : Real
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        r : Complex
        h : LE.le 0 r.re
        h2 : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h' : Eq 0 r.re
        c : Real
        hc : LT.lt c 0
        this : ∀ (x : Real), Membership.mem (Set.Ioo c 0) x → Eq (Norm.norm (Complex.e …
        ⊢ LT.lt (μ (Set.Icc c 0)) Top.top
      -/
      exact isFiniteMeasureOnCompacts_of_isLocallyFiniteMeasure.lt_top_of_isCompact isCompact_Icc
      /-
        🎉 no goals
      -/


/-- See `intervalIntegrable_cpow` for a version applying to any locally finite measure, but with a
stronger hypothesis on `r`. -/
theorem intervalIntegrable_cpow' {r : ℂ} (h : -1 < r.re) :
    IntervalIntegrable (fun x : ℝ => (x : ℂ) ^ r) volume a b := by
  suffices ∀ c : ℝ, IntervalIntegrable (fun x => (x : ℂ) ^ r) volume 0 c by
    exact IntervalIntegrable.trans (this a).symm (this b)
  have : ∀ c : ℝ, 0 ≤ c → IntervalIntegrable (fun x => (x : ℂ) ^ r) volume 0 c := by
    intro c hc
    rw [← IntervalIntegrable.intervalIntegrable_norm_iff]
    · rw [intervalIntegrable_iff]
      apply IntegrableOn.congr_fun
      · rw [← intervalIntegrable_iff]; exact intervalIntegral.intervalIntegrable_rpow' h
      · intro x hx
        rw [uIoc_of_le hc] at hx
        dsimp only
        rw [Complex.norm_eq_abs, Complex.abs_cpow_eq_rpow_re_of_pos hx.1]
      · exact measurableSet_uIoc
    · refine ContinuousOn.aestronglyMeasurable ?_ measurableSet_uIoc
      refine continuousOn_of_forall_continuousAt fun x hx => ?_
      rw [uIoc_of_le hc] at hx
      refine (continuousAt_cpow_const (Or.inl ?_)).comp Complex.continuous_ofReal.continuousAt
      rw [Complex.ofReal_re]
      exact hx.1
  /-
    a b : Real
    r : Complex
    h : LT.lt (-1) r.re
    this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
    ⊢ ∀ (c : Real), IntervalIntegrable (fun x => HPow.hPow (↑x) r) MeasureTheory.M …
  -/
  intro c; rcases le_total 0 c with (hc | hc)
    /-
      case inl
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le 0 c
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) MeasureTheory.MeasureSpace.vo …
    -/
  · exact this c hc
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le c 0
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) r) MeasureTheory.MeasureSpace.vo …
    -/
  · rw [IntervalIntegrable.iff_comp_neg, neg_zero]
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le c 0
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑(Neg.neg x)) r) MeasureTheory.Measu …
    -/
    have m := (this (-c) (by linarith)).const_mul (Complex.exp (π * Complex.I * r))
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le c 0
      m : IntervalIntegrable (fun x => HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul  …
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑(Neg.neg x)) r) MeasureTheory.Measu …
    -/
    rw [intervalIntegrable_iff, uIoc_of_le (by linarith : 0 ≤ -c)] at m ⊢
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HM …
      ⊢ MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑(Neg.neg x)) r) (Set.Ioc 0  …
    -/
    refine m.congr_fun (fun x hx => ?_) measurableSet_Ioc
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 (Neg.neg c)) x
      ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (↑Real.pi) Complex.I) r)) ( …
    -/
    dsimp only
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x) r …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 (Neg.neg c)) x
      ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (↑Real.pi) Complex.I) r)) ( …
    -/
    have : -x ≤ 0 := by linarith [hx.1]
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this✝ : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x)  …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 (Neg.neg c)) x
      this : LE.le (Neg.neg x) 0
      ⊢ Eq (HMul.hMul (Complex.exp (HMul.hMul (HMul.hMul (↑Real.pi) Complex.I) r)) ( …
    -/
    rw [Complex.ofReal_cpow_of_nonpos this, mul_comm]
    /-
      case inr
      a b : Real
      r : Complex
      h : LT.lt (-1) r.re
      this✝ : ∀ (c : Real), LE.le 0 c → IntervalIntegrable (fun x => HPow.hPow (↑x)  …
      c : Real
      hc : LE.le c 0
      m : MeasureTheory.IntegrableOn (fun x => HMul.hMul (Complex.exp (HMul.hMul (HM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 (Neg.neg c)) x
      this : LE.le (Neg.neg x) 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑x) r) (Complex.exp (HMul.hMul (HMul.hMul (↑Real.p …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The complex power function `x ↦ x^s` is integrable on `(0, t)` iff `-1 < s.re`. -/
theorem integrableOn_Ioo_cpow_iff {s : ℂ} {t : ℝ} (ht : 0 < t) :
    /-
      a b : Real
      n : Nat
      f : Real → Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      c d : Real
      s : Complex
      t : Real
      ht : LT.lt 0 t
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x : ℝ ↦ (x : ℂ) ^ s) (Ioo (0 : ℝ) t) ↔ -1 < s.re := by
    /-
      🎉 no goals
    -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simpa [intervalIntegrable_iff_integrableOn_Ioo_of_le ht.le]
    using intervalIntegrable_cpow' h (a := 0) (b := t)⟩
  have B : IntegrableOn (fun a ↦ a ^ s.re) (Ioo 0 t) := by
    apply (integrableOn_congr_fun _ measurableSet_Ioo).1 h.norm
    intro a ha
    simp [Complex.abs_cpow_eq_rpow_re_of_pos ha.1]
  /-
    s : Complex
    t : Real
    ht : LT.lt 0 t
    h : MeasureTheory.IntegrableOn (fun x => HPow.hPow (↑x) s) (Set.Ioo 0 t) Measu …
    B : MeasureTheory.IntegrableOn (fun a => HPow.hPow a s.re) (Set.Ioo 0 t) Measu …
    ⊢ LT.lt (-1) s.re
  -/
  rwa [integrableOn_Ioo_rpow_iff ht] at B
  /-
    🎉 no goals
  -/


@[simp]
theorem intervalIntegrable_id : IntervalIntegrable (fun x => x) μ a b :=
  continuous_id.intervalIntegrable a b


theorem intervalIntegrable_const : IntervalIntegrable (fun _ => c) μ a b :=
  continuous_const.intervalIntegrable a b


theorem intervalIntegrable_one_div (h : ∀ x : ℝ, x ∈ [[a, b]] → f x ≠ 0)
    (hf : ContinuousOn f [[a, b]]) : IntervalIntegrable (fun x => 1 / f x) μ a b :=
  (continuousOn_const.div hf h).intervalIntegrable


@[simp]
theorem intervalIntegrable_inv (h : ∀ x : ℝ, x ∈ [[a, b]] → f x ≠ 0)
    (hf : ContinuousOn f [[a, b]]) : IntervalIntegrable (fun x => (f x)⁻¹) μ a b := by
  /-
    a b : Real
    f : Real → Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    h : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → Ne (f x) 0
    hf : ContinuousOn f (Set.uIcc a b)
    ⊢ IntervalIntegrable (fun x => Inv.inv (f x)) μ a b
  -/
  simpa only [one_div] using intervalIntegrable_one_div h hf
  /-
    🎉 no goals
  -/


@[simp]
theorem intervalIntegrable_exp : IntervalIntegrable exp μ a b :=
  continuous_exp.intervalIntegrable a b


@[simp]
theorem _root_.IntervalIntegrable.log (hf : ContinuousOn f [[a, b]])
    (h : ∀ x : ℝ, x ∈ [[a, b]] → f x ≠ 0) :
    IntervalIntegrable (fun x => log (f x)) μ a b :=
  (ContinuousOn.log hf h).intervalIntegrable


@[simp]
theorem intervalIntegrable_log (h : (0 : ℝ) ∉ [[a, b]]) : IntervalIntegrable log μ a b :=
  IntervalIntegrable.log continuousOn_id fun _ hx => ne_of_mem_of_not_mem hx h


@[simp]
theorem intervalIntegrable_sin : IntervalIntegrable sin μ a b :=
  continuous_sin.intervalIntegrable a b


@[simp]
theorem intervalIntegrable_cos : IntervalIntegrable cos μ a b :=
  continuous_cos.intervalIntegrable a b


theorem intervalIntegrable_one_div_one_add_sq :
    IntervalIntegrable (fun x : ℝ => 1 / (↑1 + x ^ 2)) μ a b := by
  /-
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ IntervalIntegrable (fun x => HDiv.hDiv 1 (HAdd.hAdd 1 (HPow.hPow x 2))) μ a b
  -/
  refine (continuous_const.div ?_ fun x => ?_).intervalIntegrable a b
    /-
      case refine_1
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      ⊢ Continuous fun x => HAdd.hAdd 1 (HPow.hPow x 2)
    -/
  · fun_prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      x : Real
      ⊢ Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    -/
  · nlinarith
    /-
      🎉 no goals
    -/


@[simp]
theorem intervalIntegrable_inv_one_add_sq :
    IntervalIntegrable (fun x : ℝ => (↑1 + x ^ 2)⁻¹) μ a b := by
  /-
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    ⊢ IntervalIntegrable (fun x => Inv.inv (HAdd.hAdd 1 (HPow.hPow x 2))) μ a b
  -/
  field_simp; exact mod_cast intervalIntegrable_one_div_one_add_sq
              /-
                🎉 no goals
              -/


@[simp]
theorem mul_integral_comp_mul_right : (c * ∫ x in a..b, f (x * c)) = ∫ x in a * c..b * c, f x :=
  smul_integral_comp_mul_right f c


@[simp]
theorem mul_integral_comp_mul_left : (c * ∫ x in a..b, f (c * x)) = ∫ x in c * a..c * b, f x :=
  smul_integral_comp_mul_left f c


@[simp]
theorem inv_mul_integral_comp_div : (c⁻¹ * ∫ x in a..b, f (x / c)) = ∫ x in a / c..b / c, f x :=
  inv_smul_integral_comp_div f c


@[simp]
theorem mul_integral_comp_mul_add :
    (c * ∫ x in a..b, f (c * x + d)) = ∫ x in c * a + d..c * b + d, f x :=
  smul_integral_comp_mul_add f c d


@[simp]
theorem mul_integral_comp_add_mul :
    (c * ∫ x in a..b, f (d + c * x)) = ∫ x in d + c * a..d + c * b, f x :=
  smul_integral_comp_add_mul f c d


@[simp]
theorem inv_mul_integral_comp_div_add :
    (c⁻¹ * ∫ x in a..b, f (x / c + d)) = ∫ x in a / c + d..b / c + d, f x :=
  inv_smul_integral_comp_div_add f c d


@[simp]
theorem inv_mul_integral_comp_add_div :
    (c⁻¹ * ∫ x in a..b, f (d + x / c)) = ∫ x in d + a / c..d + b / c, f x :=
  inv_smul_integral_comp_add_div f c d


@[simp]
theorem mul_integral_comp_mul_sub :
    (c * ∫ x in a..b, f (c * x - d)) = ∫ x in c * a - d..c * b - d, f x :=
  smul_integral_comp_mul_sub f c d


@[simp]
theorem mul_integral_comp_sub_mul :
    (c * ∫ x in a..b, f (d - c * x)) = ∫ x in d - c * b..d - c * a, f x :=
  smul_integral_comp_sub_mul f c d


@[simp]
theorem inv_mul_integral_comp_div_sub :
    (c⁻¹ * ∫ x in a..b, f (x / c - d)) = ∫ x in a / c - d..b / c - d, f x :=
  inv_smul_integral_comp_div_sub f c d


@[simp]
theorem inv_mul_integral_comp_sub_div :
    (c⁻¹ * ∫ x in a..b, f (d - x / c)) = ∫ x in d - b / c..d - a / c, f x :=
  inv_smul_integral_comp_sub_div f c d


theorem integral_cpow {r : ℂ} (h : -1 < r.re ∨ r ≠ -1 ∧ (0 : ℝ) ∉ [[a, b]]) :
    (∫ x : ℝ in a..b, (x : ℂ) ^ r) = ((b : ℂ) ^ (r + 1) - (a : ℂ) ^ (r + 1)) / (r + 1) := by
  /-
    a b : Real
    r : Complex
    h : Or (LT.lt (-1) r.re) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b)  …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (↑x) r) a b MeasureTheory.MeasureSp …
  -/
  rw [sub_div]
  have hr : r + 1 ≠ 0 := by
    cases' h with h h
    · apply_fun Complex.re
      rw [Complex.add_re, Complex.one_re, Complex.zero_re, Ne, add_eq_zero_iff_eq_neg]
      exact h.ne'
    · rw [Ne, ← add_eq_zero_iff_eq_neg] at h; exact h.1
  /-
    a b : Real
    r : Complex
    h : Or (LT.lt (-1) r.re) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b)  …
    hr : Ne (HAdd.hAdd r 1) 0
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (↑x) r) a b MeasureTheory.MeasureSp …
  -/
  by_cases hab : (0 : ℝ) ∉ [[a, b]]
  · apply integral_eq_sub_of_hasDerivAt (fun x hx => ?_)
      (intervalIntegrable_cpow (r := r) <| Or.inr hab)
    /-
      a b : Real
      r : Complex
      h : Or (LT.lt (-1) r.re) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b)  …
      hr : Ne (HAdd.hAdd r 1) 0
      hab : Not (Membership.mem (Set.uIcc a b) 0)
      x : Real
      hx : Membership.mem (Set.uIcc a b) x
      ⊢ HasDerivAt (fun {b} => HDiv.hDiv (HPow.hPow (↑b) (HAdd.hAdd r 1)) (HAdd.hAdd …
    -/
    refine hasDerivAt_ofReal_cpow (ne_of_mem_of_not_mem hx hab) ?_
    /-
      a b : Real
      r : Complex
      h : Or (LT.lt (-1) r.re) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b)  …
      hr : Ne (HAdd.hAdd r 1) 0
      hab : Not (Membership.mem (Set.uIcc a b) 0)
      x : Real
      hx : Membership.mem (Set.uIcc a b) x
      ⊢ Ne r (-1)
    -/
    contrapose! hr; rwa [add_eq_zero_iff_eq_neg]
                    /-
                      🎉 no goals
                    -/
  /-
    case neg
    a b : Real
    r : Complex
    h : Or (LT.lt (-1) r.re) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b)  …
    hr : Ne (HAdd.hAdd r 1) 0
    hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (↑x) r) a b MeasureTheory.MeasureSp …
  -/
  replace h : -1 < r.re := by tauto
  suffices ∀ c : ℝ, (∫ x : ℝ in (0)..c, (x : ℂ) ^ r) =
      (c : ℂ) ^ (r + 1) / (r + 1) - (0 : ℂ) ^ (r + 1) / (r + 1) by
    rw [← integral_add_adjacent_intervals (@intervalIntegrable_cpow' a 0 r h)
      (@intervalIntegrable_cpow' 0 b r h), integral_symm, this a, this b, Complex.zero_cpow hr]
    ring
  /-
    case neg
    a b : Real
    r : Complex
    hr : Ne (HAdd.hAdd r 1) 0
    hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
    h : LT.lt (-1) r.re
    ⊢ ∀ (c : Real), Eq (intervalIntegral (fun x => HPow.hPow (↑x) r) 0 c MeasureTh …
  -/
  intro c
  /-
    case neg
    a b : Real
    r : Complex
    hr : Ne (HAdd.hAdd r 1) 0
    hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
    h : LT.lt (-1) r.re
    c : Real
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (↑x) r) 0 c MeasureTheory.MeasureSp …
  -/
  apply integral_eq_sub_of_hasDeriv_right
    /-
      case neg.hcont
      a b : Real
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h : LT.lt (-1) r.re
      c : Real
      ⊢ ContinuousOn (fun c => HDiv.hDiv (HPow.hPow (↑c) (HAdd.hAdd r 1)) (HAdd.hAdd …
    -/
  · refine ((Complex.continuous_ofReal_cpow_const ?_).div_const _).continuousOn
    /-
      case neg.hcont
      a b : Real
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h : LT.lt (-1) r.re
      c : Real
      ⊢ LT.lt 0 (HAdd.hAdd r 1).re
    -/
    rwa [Complex.add_re, Complex.one_re, ← neg_lt_iff_pos_add]
    /-
      🎉 no goals
    -/
    /-
      case neg.hderiv
      a b : Real
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h : LT.lt (-1) r.re
      c : Real
      ⊢ ∀ (x : Real), Membership.mem (Set.Ioo (Min.min 0 c) (Max.max 0 c)) x → HasDe …
    -/
  · refine fun x hx => (hasDerivAt_ofReal_cpow ?_ ?_).hasDerivWithinAt
      /-
        case neg.hderiv.refine_1
        a b : Real
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h : LT.lt (-1) r.re
        c x : Real
        hx : Membership.mem (Set.Ioo (Min.min 0 c) (Max.max 0 c)) x
        ⊢ Ne x 0
      -/
    · rcases le_total c 0 with (hc | hc)
        /-
          case neg.hderiv.refine_1.inl
          a b : Real
          r : Complex
          hr : Ne (HAdd.hAdd r 1) 0
          hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
          h : LT.lt (-1) r.re
          c x : Real
          hx : Membership.mem (Set.Ioo (Min.min 0 c) (Max.max 0 c)) x
          hc : LE.le c 0
          ⊢ Ne x 0
        -/
      · rw [max_eq_left hc] at hx; exact hx.2.ne
                                   /-
                                     🎉 no goals
                                   -/
        /-
          case neg.hderiv.refine_1.inr
          a b : Real
          r : Complex
          hr : Ne (HAdd.hAdd r 1) 0
          hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
          h : LT.lt (-1) r.re
          c x : Real
          hx : Membership.mem (Set.Ioo (Min.min 0 c) (Max.max 0 c)) x
          hc : LE.le 0 c
          ⊢ Ne x 0
        -/
      · rw [min_eq_left hc] at hx; exact hx.1.ne'
                                   /-
                                     🎉 no goals
                                   -/
      /-
        case neg.hderiv.refine_2
        a b : Real
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
        h : LT.lt (-1) r.re
        c x : Real
        hx : Membership.mem (Set.Ioo (Min.min 0 c) (Max.max 0 c)) x
        ⊢ Ne r (-1)
      -/
    · contrapose! hr; rw [hr]; ring
                               /-
                                 🎉 no goals
                               -/
    /-
      case neg.hint
      a b : Real
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hab : Not (Not (Membership.mem (Set.uIcc a b) 0))
      h : LT.lt (-1) r.re
      c : Real
      ⊢ IntervalIntegrable (fun y => HPow.hPow (↑y) r) MeasureTheory.MeasureSpace.vo …
    -/
  · exact intervalIntegrable_cpow' h
    /-
      🎉 no goals
    -/


theorem integral_rpow {r : ℝ} (h : -1 < r ∨ r ≠ -1 ∧ (0 : ℝ) ∉ [[a, b]]) :
    ∫ x in a..b, x ^ r = (b ^ (r + 1) - a ^ (r + 1)) / (r + 1) := by
  have h' : -1 < (r : ℂ).re ∨ (r : ℂ) ≠ -1 ∧ (0 : ℝ) ∉ [[a, b]] := by
    cases h
    · left; rwa [Complex.ofReal_re]
    · right; rwa [← Complex.ofReal_one, ← Complex.ofReal_neg, Ne, Complex.ofReal_inj]
  have :
    (∫ x in a..b, (x : ℂ) ^ (r : ℂ)) = ((b : ℂ) ^ (r + 1 : ℂ) - (a : ℂ) ^ (r + 1 : ℂ)) / (r + 1) :=
    integral_cpow h'
  /-
    a b r : Real
    h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
    h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
    this : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Measu …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow x r) a b MeasureTheory.MeasureSpace …
  -/
  apply_fun Complex.re at this; convert this
    /-
      case h.e'_2
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Measu …
      ⊢ Eq (intervalIntegral (fun x => HPow.hPow x r) a b MeasureTheory.MeasureSpace …
    -/
  · simp_rw [intervalIntegral_eq_integral_uIoc, Complex.real_smul, Complex.re_ofReal_mul]
    -- Porting note: was `change ... with ...`
    /-
      case h.e'_2
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Measu …
      ⊢ Eq (HSMul.hSMul (ite (LE.le a b) 1 (-1)) (MeasureTheory.integral (MeasureThe …
    -/
    have : Complex.re = RCLike.re := rfl
    /-
      case h.e'_2
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this✝ : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Meas …
      this : Eq Complex.re ⇑RCLike.re
      ⊢ Eq (HSMul.hSMul (ite (LE.le a b) 1 (-1)) (MeasureTheory.integral (MeasureThe …
    -/
    rw [this, ← integral_re]
      /-
        case h.e'_2
        a b r : Real
        h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
        h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
        this✝ : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Meas …
        this : Eq Complex.re ⇑RCLike.re
        ⊢ Eq (HSMul.hSMul (ite (LE.le a b) 1 (-1)) (MeasureTheory.integral (MeasureThe …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case h.e'_2
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this✝ : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Meas …
      this : Eq Complex.re ⇑RCLike.re
      ⊢ MeasureTheory.Integrable (fun x => HPow.hPow ↑x ↑r) (MeasureTheory.MeasureSp …
    -/
    refine intervalIntegrable_iff.mp ?_
    /-
      case h.e'_2
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this✝ : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Meas …
      this : Eq Complex.re ⇑RCLike.re
      ⊢ IntervalIntegrable (fun x => HPow.hPow ↑x ↑r) MeasureTheory.MeasureSpace.vol …
    -/
    cases' h' with h' h'
      /-
        case h.e'_2.inl
        a b r : Real
        h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
        this✝ : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Meas …
        this : Eq Complex.re ⇑RCLike.re
        h' : LT.lt (-1) (↑r).re
        ⊢ IntervalIntegrable (fun x => HPow.hPow ↑x ↑r) MeasureTheory.MeasureSpace.vol …
      -/
    · exact intervalIntegrable_cpow' h'
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.inr
        a b r : Real
        h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
        this✝ : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Meas …
        this : Eq Complex.re ⇑RCLike.re
        h' : And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIcc a b) 0))
        ⊢ IntervalIntegrable (fun x => HPow.hPow ↑x ↑r) MeasureTheory.MeasureSpace.vol …
      -/
    · exact intervalIntegrable_cpow (Or.inr h'.2)
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Measu …
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HPow.hPow b (HAdd.hAdd r 1)) (HPow.hPow a (HAdd.hA …
    -/
  · rw [(by push_cast; rfl : (r : ℂ) + 1 = ((r + 1 : ℝ) : ℂ))]
    /-
      case h.e'_3
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Measu …
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HPow.hPow b (HAdd.hAdd r 1)) (HPow.hPow a (HAdd.hA …
    -/
    simp_rw [div_eq_inv_mul, ← Complex.ofReal_inv, Complex.re_ofReal_mul, Complex.sub_re]
    /-
      case h.e'_3
      a b r : Real
      h : Or (LT.lt (-1) r) (And (Ne r (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
      h' : Or (LT.lt (-1) (↑r).re) (And (Ne (↑r) (-1)) (Not (Membership.mem (Set.uIc …
      this : Eq (intervalIntegral (fun x => HPow.hPow ↑x ↑r) a b MeasureTheory.Measu …
      ⊢ Eq (HMul.hMul (Inv.inv (HAdd.hAdd r 1)) (HSub.hSub (HPow.hPow b (HAdd.hAdd r …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem integral_zpow {n : ℤ} (h : 0 ≤ n ∨ n ≠ -1 ∧ (0 : ℝ) ∉ [[a, b]]) :
    ∫ x in a..b, x ^ n = (b ^ (n + 1) - a ^ (n + 1)) / (n + 1) := by
  /-
    a b : Real
    n : Int
    h : Or (LE.le 0 n) (And (Ne n (-1)) (Not (Membership.mem (Set.uIcc a b) 0)))
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow x n) a b MeasureTheory.MeasureSpace …
  -/
  replace h : -1 < (n : ℝ) ∨ (n : ℝ) ≠ -1 ∧ (0 : ℝ) ∉ [[a, b]] := mod_cast h
  /-
    a b : Real
    n : Int
    h : Or (LT.lt (-1) ↑n) (And (Ne (↑n) (-1)) (Not (Membership.mem (Set.uIcc a b) …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow x n) a b MeasureTheory.MeasureSpace …
  -/
  exact mod_cast integral_rpow h
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_pow : ∫ x in a..b, x ^ n = (b ^ (n + 1) - a ^ (n + 1)) / (n + 1) := by
  /-
    a b : Real
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow x n) a b MeasureTheory.MeasureSpace …
  -/
  simpa only [← Int.ofNat_succ, zpow_natCast] using integral_zpow (Or.inl n.cast_nonneg)
  /-
    🎉 no goals
  -/


/-- Integral of `|x - a| ^ n` over `Ι a b`. This integral appears in the proof of the
Picard-Lindelöf/Cauchy-Lipschitz theorem. -/
theorem integral_pow_abs_sub_uIoc : ∫ x in Ι a b, |x - a| ^ n = |b - a| ^ (n + 1) / (n + 1) := by
  /-
    a b : Real
    n : Nat
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rcases le_or_lt a b with hab | hab
  · calc
      ∫ x in Ι a b, |x - a| ^ n = ∫ x in a..b, |x - a| ^ n := by
        rw [uIoc_of_le hab, ← integral_of_le hab]
      _ = ∫ x in (0)..(b - a), x ^ n := by
        simp only [integral_comp_sub_right fun x => |x| ^ n, sub_self]
        refine integral_congr fun x hx => congr_arg₂ Pow.pow (abs_of_nonneg <| ?_) rfl
        rw [uIcc_of_le (sub_nonneg.2 hab)] at hx
        exact hx.1
      _ = |b - a| ^ (n + 1) / (n + 1) := by simp [abs_of_nonneg (sub_nonneg.2 hab)]
  · calc
      ∫ x in Ι a b, |x - a| ^ n = ∫ x in b..a, |x - a| ^ n := by
        rw [uIoc_of_ge hab.le, ← integral_of_le hab.le]
      _ = ∫ x in b - a..0, (-x) ^ n := by
        simp only [integral_comp_sub_right fun x => |x| ^ n, sub_self]
        refine integral_congr fun x hx => congr_arg₂ Pow.pow (abs_of_nonpos <| ?_) rfl
        rw [uIcc_of_le (sub_nonpos.2 hab.le)] at hx
        exact hx.2
      _ = |b - a| ^ (n + 1) / (n + 1) := by
        simp [integral_comp_neg fun x => x ^ n, abs_of_neg (sub_neg.2 hab)]


@[simp]
theorem integral_id : ∫ x in a..b, x = (b ^ 2 - a ^ 2) / 2 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => x) a b MeasureTheory.MeasureSpace.volume) (HD …
  -/
  have := @integral_pow a b 1
  /-
    a b : Real
    this : Eq (intervalIntegral (fun x => HPow.hPow x 1) a b MeasureTheory.Measure …
    ⊢ Eq (intervalIntegral (fun x => x) a b MeasureTheory.MeasureSpace.volume) (HD …
  -/
  norm_num at this
  /-
    a b : Real
    this : Eq (intervalIntegral (fun x => x) a b MeasureTheory.MeasureSpace.volume …
    ⊢ Eq (intervalIntegral (fun x => x) a b MeasureTheory.MeasureSpace.volume) (HD …
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem integral_one : (∫ _ in a..b, (1 : ℝ)) = b - a := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => 1) a b MeasureTheory.MeasureSpace.volume) (HS …
  -/
  simp only [mul_one, smul_eq_mul, integral_const]
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         a b : Real
                                                                         ⊢ Eq (intervalIntegral (fun x => b) a (HAdd.hAdd a 1) MeasureTheory.MeasureSpa …
                                                                       -/
theorem integral_const_on_unit_interval : ∫ _ in a..a + 1, b = b := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem integral_inv (h : (0 : ℝ) ∉ [[a, b]]) : ∫ x in a..b, x⁻¹ = log (b / a) := by
  /-
    a b : Real
    h : Not (Membership.mem (Set.uIcc a b) 0)
    ⊢ Eq (intervalIntegral (fun x => Inv.inv x) a b MeasureTheory.MeasureSpace.vol …
  -/
  have h' := fun x (hx : x ∈ [[a, b]]) => ne_of_mem_of_not_mem hx h
  rw [integral_deriv_eq_sub' _ deriv_log' (fun x hx => differentiableAt_log (h' x hx))
      (continuousOn_inv₀.mono <| subset_compl_singleton_iff.mpr h),
    log_div (h' b right_mem_uIcc) (h' a left_mem_uIcc)]


@[simp]
theorem integral_inv_of_pos (ha : 0 < a) (hb : 0 < b) : ∫ x in a..b, x⁻¹ = log (b / a) :=
  integral_inv <| not_mem_uIcc_of_lt ha hb


@[simp]
theorem integral_inv_of_neg (ha : a < 0) (hb : b < 0) : ∫ x in a..b, x⁻¹ = log (b / a) :=
  integral_inv <| not_mem_uIcc_of_gt ha hb


theorem integral_one_div (h : (0 : ℝ) ∉ [[a, b]]) : ∫ x : ℝ in a..b, 1 / x = log (b / a) := by
  /-
    a b : Real
    h : Not (Membership.mem (Set.uIcc a b) 0)
    ⊢ Eq (intervalIntegral (fun x => HDiv.hDiv 1 x) a b MeasureTheory.MeasureSpace …
  -/
  simp only [one_div, integral_inv h]
  /-
    🎉 no goals
  -/


theorem integral_one_div_of_pos (ha : 0 < a) (hb : 0 < b) :
                                               /-
                                                 a b : Real
                                                 ha : LT.lt 0 a
                                                 hb : LT.lt 0 b
                                                 ⊢ Eq (intervalIntegral (fun x => HDiv.hDiv 1 x) a b MeasureTheory.MeasureSpace …
                                               -/
    ∫ x : ℝ in a..b, 1 / x = log (b / a) := by simp only [one_div, integral_inv_of_pos ha hb]
                                               /-
                                                 🎉 no goals
                                               -/


theorem integral_one_div_of_neg (ha : a < 0) (hb : b < 0) :
                                               /-
                                                 a b : Real
                                                 ha : LT.lt a 0
                                                 hb : LT.lt b 0
                                                 ⊢ Eq (intervalIntegral (fun x => HDiv.hDiv 1 x) a b MeasureTheory.MeasureSpace …
                                               -/
    ∫ x : ℝ in a..b, 1 / x = log (b / a) := by simp only [one_div, integral_inv_of_neg ha hb]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem integral_exp : ∫ x in a..b, exp x = exp b - exp a := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => Real.exp x) a b MeasureTheory.MeasureSpace.vo …
  -/
  rw [integral_deriv_eq_sub']
    /-
      case hderiv
      a b : Real
      ⊢ Eq (deriv Real.exp) Real.exp
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hdiff
      a b : Real
      ⊢ ∀ (x : Real), Membership.mem (Set.uIcc a b) x → DifferentiableAt Real Real.e …
    -/
  · exact fun _ _ => differentiableAt_exp
    /-
      🎉 no goals
    -/
    /-
      case hcont
      a b : Real
      ⊢ ContinuousOn Real.exp (Set.uIcc a b)
    -/
  · exact continuousOn_exp
    /-
      🎉 no goals
    -/


theorem integral_exp_mul_complex {c : ℂ} (hc : c ≠ 0) :
    (∫ x in a..b, Complex.exp (c * x)) = (Complex.exp (c * b) - Complex.exp (c * a)) / c := by
  have D : ∀ x : ℝ, HasDerivAt (fun y : ℝ => Complex.exp (c * y) / c) (Complex.exp (c * x)) x := by
    intro x
    conv => congr
    rw [← mul_div_cancel_right₀ (Complex.exp (c * x)) hc]
    apply ((Complex.hasDerivAt_exp _).comp x _).div_const c
    simpa only [mul_one] using ((hasDerivAt_id (x : ℂ)).const_mul _).comp_ofReal
  /-
    a b : Real
    c : Complex
    hc : Ne c 0
    D : ∀ (x : Real), HasDerivAt (fun y => HDiv.hDiv (Complex.exp (HMul.hMul c ↑y) …
    ⊢ Eq (intervalIntegral (fun x => Complex.exp (HMul.hMul c ↑x)) a b MeasureTheo …
  -/
  rw [integral_deriv_eq_sub' _ (funext fun x => (D x).deriv) fun x _ => (D x).differentiableAt]
    /-
      a b : Real
      c : Complex
      hc : Ne c 0
      D : ∀ (x : Real), HasDerivAt (fun y => HDiv.hDiv (Complex.exp (HMul.hMul c ↑y) …
      ⊢ Eq (HSub.hSub (HDiv.hDiv (Complex.exp (HMul.hMul c ↑b)) c) (HDiv.hDiv (Compl …
    -/
  · ring
    /-
      🎉 no goals
    -/
    /-
      a b : Real
      c : Complex
      hc : Ne c 0
      D : ∀ (x : Real), HasDerivAt (fun y => HDiv.hDiv (Complex.exp (HMul.hMul c ↑y) …
      ⊢ ContinuousOn (fun x => Complex.exp (HMul.hMul c ↑x)) (Set.uIcc a b)
    -/
  · fun_prop
    /-
      🎉 no goals
    -/


@[simp]
theorem integral_log (h : (0 : ℝ) ∉ [[a, b]]) :
    ∫ x in a..b, log x = b * log b - a * log a - b + a := by
  /-
    a b : Real
    h : Not (Membership.mem (Set.uIcc a b) 0)
    ⊢ Eq (intervalIntegral (fun x => Real.log x) a b MeasureTheory.MeasureSpace.vo …
  -/
  have h' := fun x (hx : x ∈ [[a, b]]) => ne_of_mem_of_not_mem hx h
  /-
    a b : Real
    h : Not (Membership.mem (Set.uIcc a b) 0)
    h' : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → Ne x 0
    ⊢ Eq (intervalIntegral (fun x => Real.log x) a b MeasureTheory.MeasureSpace.vo …
  -/
  have heq := fun x hx => mul_inv_cancel₀ (h' x hx)
  convert integral_mul_deriv_eq_deriv_mul (fun x hx => hasDerivAt_log (h' x hx))
    (fun x _ => hasDerivAt_id x) (continuousOn_inv₀.mono <|
      subset_compl_singleton_iff.mpr h).intervalIntegrable
        continuousOn_const.intervalIntegrable using 1 <;>
  /-
    case h.e'_2
    a b : Real
    h : Not (Membership.mem (Set.uIcc a b) 0)
    h' : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → Ne x 0
    heq : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → Eq (HMul.hMul x (Inv.inv …
    ⊢ Eq (intervalIntegral (fun x => Real.log x) a b MeasureTheory.MeasureSpace.vo …
  -/
  /-
    🎉 no goals
  -/
  simp [integral_congr heq, mul_comm, ← sub_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_log_of_pos (ha : 0 < a) (hb : 0 < b) :
    ∫ x in a..b, log x = b * log b - a * log a - b + a :=
  integral_log <| not_mem_uIcc_of_lt ha hb


@[simp]
theorem integral_log_of_neg (ha : a < 0) (hb : b < 0) :
    ∫ x in a..b, log x = b * log b - a * log a - b + a :=
  integral_log <| not_mem_uIcc_of_gt ha hb


@[simp]
theorem integral_sin : ∫ x in a..b, sin x = cos a - cos b := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => Real.sin x) a b MeasureTheory.MeasureSpace.vo …
  -/
  rw [integral_deriv_eq_sub' fun x => -cos x]
    /-
      a b : Real
      ⊢ Eq (HSub.hSub (Neg.neg (Real.cos b)) (Neg.neg (Real.cos a))) (HSub.hSub (Rea …
    -/
  · ring
    /-
      🎉 no goals
    -/
    /-
      case hderiv
      a b : Real
      ⊢ Eq (deriv fun x => Neg.neg (Real.cos x)) Real.sin
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case hdiff
      a b : Real
      ⊢ ∀ (x : Real), Membership.mem (Set.uIcc a b) x → DifferentiableAt Real (fun x …
    -/
  · simp only [differentiableAt_neg_iff, differentiableAt_cos, implies_true]
    /-
      🎉 no goals
    -/
    /-
      case hcont
      a b : Real
      ⊢ ContinuousOn Real.sin (Set.uIcc a b)
    -/
  · exact continuousOn_sin
    /-
      🎉 no goals
    -/


@[simp]
theorem integral_cos : ∫ x in a..b, cos x = sin b - sin a := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => Real.cos x) a b MeasureTheory.MeasureSpace.vo …
  -/
  rw [integral_deriv_eq_sub']
    /-
      case hderiv
      a b : Real
      ⊢ Eq (deriv Real.sin) Real.cos
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case hdiff
      a b : Real
      ⊢ ∀ (x : Real), Membership.mem (Set.uIcc a b) x → DifferentiableAt Real Real.s …
    -/
  · simp only [differentiableAt_sin, implies_true]
    /-
      🎉 no goals
    -/
    /-
      case hcont
      a b : Real
      ⊢ ContinuousOn Real.cos (Set.uIcc a b)
    -/
  · exact continuousOn_cos
    /-
      🎉 no goals
    -/


theorem integral_cos_mul_complex {z : ℂ} (hz : z ≠ 0) (a b : ℝ) :
    (∫ x in a..b, Complex.cos (z * x)) = Complex.sin (z * b) / z - Complex.sin (z * a) / z := by
  /-
    z : Complex
    hz : Ne z 0
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => Complex.cos (HMul.hMul z ↑x)) a b MeasureTheo …
  -/
  apply integral_eq_sub_of_hasDerivAt
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a b : Real
    ⊢ ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun b => HDiv.hD …
  -/
  swap
    /-
      case hint
      z : Complex
      hz : Ne z 0
      a b : Real
      ⊢ IntervalIntegrable (fun y => Complex.cos (HMul.hMul z ↑y)) MeasureTheory.Mea …
    -/
  · apply Continuous.intervalIntegrable
    /-
      case hint.hu
      z : Complex
      hz : Ne z 0
      a b : Real
      ⊢ Continuous fun y => Complex.cos (HMul.hMul z ↑y)
    -/
    exact Complex.continuous_cos.comp (continuous_const.mul Complex.continuous_ofReal)
    /-
      🎉 no goals
    -/
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a b : Real
    ⊢ ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun b => HDiv.hD …
  -/
  intro x _
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a b x : Real
    a✝ : Membership.mem (Set.uIcc a b) x
    ⊢ HasDerivAt (fun b => HDiv.hDiv (Complex.sin (HMul.hMul z ↑b)) z) (Complex.co …
  -/
  have a := Complex.hasDerivAt_sin (↑x * z)
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a✝¹ b x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    ⊢ HasDerivAt (fun b => HDiv.hDiv (Complex.sin (HMul.hMul z ↑b)) z) (Complex.co …
  -/
  have b : HasDerivAt (fun y => y * z : ℂ → ℂ) z ↑x := hasDerivAt_mul_const _
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a✝¹ b✝ x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b✝) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    b : HasDerivAt (fun y => HMul.hMul y z) z ↑x
    ⊢ HasDerivAt (fun b => HDiv.hDiv (Complex.sin (HMul.hMul z ↑b)) z) (Complex.co …
  -/
  have c : HasDerivAt (Complex.sin ∘ fun y : ℂ => (y * z)) _ ↑x := HasDerivAt.comp (𝕜 := ℂ) x a b
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a✝¹ b✝ x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b✝) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    b : HasDerivAt (fun y => HMul.hMul y z) z ↑x
    c : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y z) (HMul.hMul ( …
    ⊢ HasDerivAt (fun b => HDiv.hDiv (Complex.sin (HMul.hMul z ↑b)) z) (Complex.co …
  -/
  have d := HasDerivAt.comp_ofReal (c.div_const z)
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a✝¹ b✝ x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b✝) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    b : HasDerivAt (fun y => HMul.hMul y z) z ↑x
    c : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y z) (HMul.hMul ( …
    d : HasDerivAt (fun y => HDiv.hDiv (Function.comp Complex.sin (fun y => HMul.h …
    ⊢ HasDerivAt (fun b => HDiv.hDiv (Complex.sin (HMul.hMul z ↑b)) z) (Complex.co …
  -/
  simp only [mul_comm] at d
  /-
    case hderiv
    z : Complex
    hz : Ne z 0
    a✝¹ b✝ x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b✝) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    b : HasDerivAt (fun y => HMul.hMul y z) z ↑x
    c : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y z) (HMul.hMul ( …
    d : HasDerivAt (fun y => HDiv.hDiv (Function.comp Complex.sin (fun y => HMul.h …
    ⊢ HasDerivAt (fun b => HDiv.hDiv (Complex.sin (HMul.hMul z ↑b)) z) (Complex.co …
  -/
  convert d using 1
  /-
    case h.e'_9
    z : Complex
    hz : Ne z 0
    a✝¹ b✝ x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b✝) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    b : HasDerivAt (fun y => HMul.hMul y z) z ↑x
    c : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y z) (HMul.hMul ( …
    d : HasDerivAt (fun y => HDiv.hDiv (Function.comp Complex.sin (fun y => HMul.h …
    ⊢ Eq (Complex.cos (HMul.hMul z ↑x)) (HDiv.hDiv (HMul.hMul z (Complex.cos (HMul …
  -/
  conv_rhs => arg 1; rw [mul_comm]
  /-
    case h.e'_9
    z : Complex
    hz : Ne z 0
    a✝¹ b✝ x : Real
    a✝ : Membership.mem (Set.uIcc a✝¹ b✝) x
    a : HasDerivAt Complex.sin (Complex.cos (HMul.hMul (↑x) z)) (HMul.hMul (↑x) z)
    b : HasDerivAt (fun y => HMul.hMul y z) z ↑x
    c : HasDerivAt (Function.comp Complex.sin fun y => HMul.hMul y z) (HMul.hMul ( …
    d : HasDerivAt (fun y => HDiv.hDiv (Function.comp Complex.sin (fun y => HMul.h …
    ⊢ Eq (Complex.cos (HMul.hMul z ↑x)) (HDiv.hDiv (HMul.hMul (Complex.cos (HMul.h …
  -/
  rw [mul_div_cancel_right₀ _ hz]
  /-
    🎉 no goals
  -/


theorem integral_cos_sq_sub_sin_sq :
    ∫ x in a..b, cos x ^ 2 - sin x ^ 2 = sin b * cos b - sin a * cos a := by
  simpa only [sq, sub_eq_add_neg, neg_mul_eq_mul_neg] using
    integral_deriv_mul_eq_sub (fun x _ => hasDerivAt_sin x) (fun x _ => hasDerivAt_cos x)
      continuousOn_cos.intervalIntegrable continuousOn_sin.neg.intervalIntegrable


theorem integral_one_div_one_add_sq :
    (∫ x : ℝ in a..b, ↑1 / (↑1 + x ^ 2)) = arctan b - arctan a := by
  refine integral_deriv_eq_sub' _ Real.deriv_arctan (fun _ _ => differentiableAt_arctan _)
    (continuous_const.div ?_ fun x => ?_).continuousOn
    /-
      case refine_1
      a b : Real
      ⊢ Continuous fun x => HAdd.hAdd 1 (HPow.hPow x 2)
    -/
  · fun_prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b x : Real
      ⊢ Ne (HAdd.hAdd 1 (HPow.hPow x 2)) 0
    -/
  · nlinarith
    /-
      🎉 no goals
    -/


@[simp]
theorem integral_inv_one_add_sq : (∫ x : ℝ in a..b, (↑1 + x ^ 2)⁻¹) = arctan b - arctan a := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => Inv.inv (HAdd.hAdd 1 (HPow.hPow x 2))) a b Me …
  -/
  simp only [← one_div, integral_one_div_one_add_sq]
  /-
    🎉 no goals
  -/


theorem integral_mul_cpow_one_add_sq {t : ℂ} (ht : t ≠ -1) :
    (∫ x : ℝ in a..b, (x : ℂ) * ((1 : ℂ) + ↑x ^ 2) ^ t) =
      ((1 : ℂ) + (b : ℂ) ^ 2) ^ (t + 1) / (2 * (t + ↑1)) -
      ((1 : ℂ) + (a : ℂ) ^ 2) ^ (t + 1) / (2 * (t + ↑1)) := by
  /-
    a b : Real
    t : Complex
    ht : Ne t (-1)
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (↑x) (HPow.hPow (HAdd.hAdd 1 (HPow. …
  -/
  have : t + 1 ≠ 0 := by contrapose! ht; rwa [add_eq_zero_iff_eq_neg] at ht
  /-
    a b : Real
    t : Complex
    ht : Ne t (-1)
    this : Ne (HAdd.hAdd t 1) 0
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (↑x) (HPow.hPow (HAdd.hAdd 1 (HPow. …
  -/
  apply integral_eq_sub_of_hasDerivAt
    /-
      case hderiv
      a b : Real
      t : Complex
      ht : Ne t (-1)
      this : Ne (HAdd.hAdd t 1) 0
      ⊢ ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun {b} => HDiv. …
    -/
  · intro x _
    have f : HasDerivAt (fun y : ℂ => 1 + y ^ 2) (2 * x : ℂ) x := by
      convert (hasDerivAt_pow 2 (x : ℂ)).const_add 1
      simp
    have g :
      ∀ {z : ℂ}, 0 < z.re → HasDerivAt (fun z => z ^ (t + 1) / (2 * (t + 1))) (z ^ t / 2) z := by
      intro z hz
      convert (HasDerivAt.cpow_const (c := t + 1) (hasDerivAt_id _)
        (Or.inl hz)).div_const (2 * (t + 1)) using 1
      field_simp
      ring
    /-
      case hderiv
      a b : Real
      t : Complex
      ht : Ne t (-1)
      this : Ne (HAdd.hAdd t 1) 0
      x : Real
      a✝ : Membership.mem (Set.uIcc a b) x
      f : HasDerivAt (fun y => HAdd.hAdd 1 (HPow.hPow y 2)) (HMul.hMul 2 ↑x) ↑x
      g : ∀ {z : Complex}, LT.lt 0 z.re → HasDerivAt (fun z => HDiv.hDiv (HPow.hPow  …
      ⊢ HasDerivAt (fun {b} => HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (HPow.hPow (↑b) 2)) …
    -/
    convert (HasDerivAt.comp (↑x) (g _) f).comp_ofReal using 1
      /-
        case h.e'_9
        a b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        x : Real
        a✝ : Membership.mem (Set.uIcc a b) x
        f : HasDerivAt (fun y => HAdd.hAdd 1 (HPow.hPow y 2)) (HMul.hMul 2 ↑x) ↑x
        g : ∀ {z : Complex}, LT.lt 0 z.re → HasDerivAt (fun z => HDiv.hDiv (HPow.hPow  …
        ⊢ Eq (HMul.hMul (↑x) (HPow.hPow (HAdd.hAdd 1 (HPow.hPow (↑x) 2)) t)) (HMul.hMu …
      -/
    · field_simp; ring
                  /-
                    🎉 no goals
                  -/
      /-
        case hderiv
        a b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        x : Real
        a✝ : Membership.mem (Set.uIcc a b) x
        f : HasDerivAt (fun y => HAdd.hAdd 1 (HPow.hPow y 2)) (HMul.hMul 2 ↑x) ↑x
        g : ∀ {z : Complex}, LT.lt 0 z.re → HasDerivAt (fun z => HDiv.hDiv (HPow.hPow  …
        ⊢ LT.lt 0 (HAdd.hAdd 1 (HPow.hPow (↑x) 2)).re
      -/
    · exact mod_cast add_pos_of_pos_of_nonneg zero_lt_one (sq_nonneg x)
      /-
        🎉 no goals
      -/
    /-
      case hint
      a b : Real
      t : Complex
      ht : Ne t (-1)
      this : Ne (HAdd.hAdd t 1) 0
      ⊢ IntervalIntegrable (fun y => HMul.hMul (↑y) (HPow.hPow (HAdd.hAdd 1 (HPow.hP …
    -/
  · apply Continuous.intervalIntegrable
    /-
      case hint.hu
      a b : Real
      t : Complex
      ht : Ne t (-1)
      this : Ne (HAdd.hAdd t 1) 0
      ⊢ Continuous fun y => HMul.hMul (↑y) (HPow.hPow (HAdd.hAdd 1 (HPow.hPow (↑y) 2 …
    -/
    refine continuous_ofReal.mul ?_
    /-
      case hint.hu
      a b : Real
      t : Complex
      ht : Ne t (-1)
      this : Ne (HAdd.hAdd t 1) 0
      ⊢ Continuous fun y => HPow.hPow (HAdd.hAdd 1 (HPow.hPow (↑y) 2)) t
    -/
    apply Continuous.cpow
      /-
        case hint.hu.hf
        a b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        ⊢ Continuous fun x => HAdd.hAdd 1 (HPow.hPow (↑x) 2)
      -/
    · exact continuous_const.add (continuous_ofReal.pow 2)
      /-
        🎉 no goals
      -/
      /-
        case hint.hu.hg
        a b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        ⊢ Continuous fun x => t
      -/
    · exact continuous_const
      /-
        🎉 no goals
      -/
      /-
        case hint.hu.h0
        a b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        ⊢ ∀ (a : Real), Membership.mem Complex.slitPlane (HAdd.hAdd 1 (HPow.hPow (↑a)  …
      -/
    · intro a
      /-
        case hint.hu.h0
        a✝ b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        a : Real
        ⊢ Membership.mem Complex.slitPlane (HAdd.hAdd 1 (HPow.hPow (↑a) 2))
      -/
      norm_cast
      /-
        case hint.hu.h0
        a✝ b : Real
        t : Complex
        ht : Ne t (-1)
        this : Ne (HAdd.hAdd t 1) 0
        a : Real
        ⊢ Membership.mem Complex.slitPlane ↑(HAdd.hAdd 1 (HPow.hPow a 2))
      -/
      exact ofReal_mem_slitPlane.2 <| add_pos_of_pos_of_nonneg one_pos <| sq_nonneg a
      /-
        🎉 no goals
      -/


theorem integral_mul_rpow_one_add_sq {t : ℝ} (ht : t ≠ -1) :
    (∫ x : ℝ in a..b, x * (↑1 + x ^ 2) ^ t) =
      (↑1 + b ^ 2) ^ (t + 1) / (↑2 * (t + ↑1)) - (↑1 + a ^ 2) ^ (t + 1) / (↑2 * (t + ↑1)) := by
  have : ∀ x s : ℝ, (((↑1 + x ^ 2) ^ s : ℝ) : ℂ) = (1 + (x : ℂ) ^ 2) ^ (s : ℂ) := by
    intro x s
    norm_cast
    rw [ofReal_cpow, ofReal_add, ofReal_pow, ofReal_one]
    exact add_nonneg zero_le_one (sq_nonneg x)
  /-
    a b t : Real
    ht : Ne t (-1)
    this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul x (HPow.hPow (HAdd.hAdd 1 (HPow.hPo …
  -/
  rw [← ofReal_inj]
  /-
    a b t : Real
    ht : Ne t (-1)
    this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
    ⊢ Eq ↑(intervalIntegral (fun x => HMul.hMul x (HPow.hPow (HAdd.hAdd 1 (HPow.hP …
  -/
  convert integral_mul_cpow_one_add_sq (_ : (t : ℂ) ≠ -1)
    /-
      case h.e'_2
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      ⊢ Eq (↑(intervalIntegral (fun x => HMul.hMul x (HPow.hPow (HAdd.hAdd 1 (HPow.h …
    -/
  · rw [← intervalIntegral.integral_ofReal]
    /-
      case h.e'_2
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      ⊢ Eq (intervalIntegral (fun x => ↑(HMul.hMul x (HPow.hPow (HAdd.hAdd 1 (HPow.h …
    -/
    congr with x : 1
    /-
      case h.e'_2.e_f.h
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      x : Real
      ⊢ Eq (↑(HMul.hMul x (HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) t))) (HMul.hMul ( …
    -/
    rw [ofReal_mul, this x t]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      ⊢ Eq (↑(HSub.hSub (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (HPow.hPow b 2)) (HAdd.hA …
    -/
  · simp_rw [ofReal_sub, ofReal_div, this a (t + 1), this b (t + 1)]
    /-
      case h.e'_3
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      ⊢ Eq (HSub.hSub (HDiv.hDiv (HPow.hPow (HAdd.hAdd 1 (HPow.hPow (↑b) 2)) ↑(HAdd. …
    -/
    push_cast; rfl
               /-
                 🎉 no goals
               -/
    /-
      case convert_3
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      ⊢ Ne (↑t) (-1)
    -/
  · rw [← ofReal_one, ← ofReal_neg, Ne, ofReal_inj]
    /-
      case convert_3
      a b t : Real
      ht : Ne t (-1)
      this : ∀ (x s : Real), Eq (↑(HPow.hPow (HAdd.hAdd 1 (HPow.hPow x 2)) s)) (HPow …
      ⊢ Not (Eq t (-1))
    -/
    exact ht
    /-
      🎉 no goals
    -/


theorem integral_sin_pow_aux :
    (∫ x in a..b, sin x ^ (n + 2)) =
      (sin a ^ (n + 1) * cos a - sin b ^ (n + 1) * cos b + (↑n + 1) * ∫ x in a..b, sin x ^ n) -
        (↑n + 1) * ∫ x in a..b, sin x ^ (n + 2) := by
  /-
    a b : Real
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd n 2)) a b M …
  -/
  let C := sin a ^ (n + 1) * cos a - sin b ^ (n + 1) * cos b
  /-
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.sin a) (HAdd.hAdd n 1)) (Rea …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd n 2)) a b M …
  -/
  have h : ∀ α β γ : ℝ, β * α * γ * α = β * (α * α * γ) := fun α β γ => by ring
  have hu : ∀ x ∈ [[a, b]],
      HasDerivAt (fun y => sin y ^ (n + 1)) ((n + 1 : ℕ) * cos x * sin x ^ n) x :=
    fun x _ => by simpa only [mul_right_comm] using (hasDerivAt_sin x).pow (n + 1)
  have hv : ∀ x ∈ [[a, b]], HasDerivAt (-cos) (sin x) x := fun x _ => by
    simpa only [neg_neg] using (hasDerivAt_cos x).neg
  /-
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.sin a) (HAdd.hAdd n 1)) (Rea …
    h : ∀ (α β γ : Real), Eq (HMul.hMul (HMul.hMul (HMul.hMul β α) γ) α) (HMul.hMu …
    hu : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun y => HPow …
    hv : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (Neg.neg Real. …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd n 2)) a b M …
  -/
  have H := integral_mul_deriv_eq_deriv_mul hu hv ?_ ?_
  · calc
      (∫ x in a..b, sin x ^ (n + 2)) = ∫ x in a..b, sin x ^ (n + 1) * sin x := by
        simp only [_root_.pow_succ]
      _ = C + (↑n + 1) * ∫ x in a..b, cos x ^ 2 * sin x ^ n := by simp [H, h, sq]; ring
      _ = C + (↑n + 1) * ∫ x in a..b, sin x ^ n - sin x ^ (n + 2) := by
        simp [cos_sq', sub_mul, ← pow_add, add_comm]
      _ = (C + (↑n + 1) * ∫ x in a..b, sin x ^ n) - (↑n + 1) * ∫ x in a..b, sin x ^ (n + 2) := by
        rw [integral_sub, mul_sub, add_sub_assoc] <;>
          apply Continuous.intervalIntegrable <;> fun_prop
  /-
    case refine_1
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.sin a) (HAdd.hAdd n 1)) (Rea …
    h : ∀ (α β γ : Real), Eq (HMul.hMul (HMul.hMul (HMul.hMul β α) γ) α) (HMul.hMu …
    hu : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun y => HPow …
    hv : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (Neg.neg Real. …
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HMul.hMul (↑(HAdd.hAdd n 1)) (Real.c …
  -/
  all_goals apply Continuous.intervalIntegrable; fun_prop
  /-
    🎉 no goals
  -/


/-- The reduction formula for the integral of `sin x ^ n` for any natural `n ≥ 2`. -/
theorem integral_sin_pow :
    (∫ x in a..b, sin x ^ (n + 2)) =
      (sin a ^ (n + 1) * cos a - sin b ^ (n + 1) * cos b) / (n + 2) +
        (n + 1) / (n + 2) * ∫ x in a..b, sin x ^ n := by
  /-
    a b : Real
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd n 2)) a b M …
  -/
  field_simp
  /-
    a b : Real
    n : Nat
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd  …
  -/
  convert eq_sub_iff_add_eq.mp (integral_sin_pow_aux n) using 1
  /-
    case h.e'_2
    a b : Real
    n : Nat
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd  …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_sin_sq : ∫ x in a..b, sin x ^ 2 = (sin a * cos a - sin b * cos b + b - a) / 2 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) 2) a b MeasureTheory.M …
  -/
  field_simp [integral_sin_pow, add_sub_assoc]
  /-
    🎉 no goals
  -/


theorem integral_sin_pow_odd :
    (∫ x in (0)..π, sin x ^ (2 * n + 1)) = 2 * ∏ i ∈ range n, (2 * (i : ℝ) + 2) / (2 * i + 3) := by
  /-
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd (HMul.hMul  …
  -/
  induction' n with k ih; · norm_num
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n k : Nat
    ih : Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd (HMul.hM …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd (HMul.hMul  …
  -/
  rw [prod_range_succ_comm, mul_left_comm, ← ih, mul_succ, integral_sin_pow]
  /-
    case succ
    n k : Nat
    ih : Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd (HMul.hM …
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HMul.hMul (HPow.hPow (Real.sin 0) (HAdd …
  -/
  norm_cast
  /-
    case succ
    n k : Nat
    ih : Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd (HMul.hM …
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HMul.hMul (HPow.hPow (Real.sin 0) (HAdd …
  -/
  simp [-cast_add, field_simps]
  /-
    🎉 no goals
  -/


theorem integral_sin_pow_even :
    (∫ x in (0)..π, sin x ^ (2 * n)) = π * ∏ i ∈ range n, (2 * (i : ℝ) + 1) / (2 * i + 2) := by
  /-
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HMul.hMul 2 n)) 0 Rea …
  -/
  induction' n with k ih; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n k : Nat
    ih : Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HMul.hMul 2 k)) 0  …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HMul.hMul 2 (HAdd.hAd …
  -/
  rw [prod_range_succ_comm, mul_left_comm, ← ih, mul_succ, integral_sin_pow]
  /-
    case succ
    n k : Nat
    ih : Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HMul.hMul 2 k)) 0  …
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HMul.hMul (HPow.hPow (Real.sin 0) (HAdd …
  -/
  norm_cast
  /-
    case succ
    n k : Nat
    ih : Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HMul.hMul 2 k)) 0  …
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HMul.hMul (HPow.hPow (Real.sin 0) (HAdd …
  -/
  simp [-cast_add, field_simps]
  /-
    🎉 no goals
  -/


theorem integral_sin_pow_pos : 0 < ∫ x in (0)..π, sin x ^ n := by
  /-
    n : Nat
    ⊢ LT.lt 0 (intervalIntegral (fun x => HPow.hPow (Real.sin x) n) 0 Real.pi Meas …
  -/
  rcases even_or_odd' n with ⟨k, rfl | rfl⟩ <;>
  /-
    case intro.inl
    k : Nat
    ⊢ LT.lt 0 (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HMul.hMul 2 k))  …
  -/
  simp only [integral_sin_pow_even, integral_sin_pow_odd] <;>
  /-
    case intro.inl
    k : Nat
    ⊢ LT.lt 0 (HMul.hMul Real.pi ((Finset.range k).prod fun i => HDiv.hDiv (HAdd.h …
  -/
  refine mul_pos (by norm_num [pi_pos]) (prod_pos fun n _ => div_pos ?_ ?_) <;>
  /-
    case intro.inl.refine_1
    k n : Nat
    x✝ : Membership.mem (Finset.range k) n
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul 2 ↑n) 1)
  -/
  norm_cast <;>
  /-
    case intro.inl.refine_1
    k n : Nat
    x✝ : Membership.mem (Finset.range k) n
    ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul 2 n) 1)
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
  omega
  /-
    🎉 no goals
  -/


theorem integral_sin_pow_succ_le : (∫ x in (0)..π, sin x ^ (n + 1)) ≤ ∫ x in (0)..π, sin x ^ n := by
  /-
    n : Nat
    ⊢ LE.le (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd n 1)) 0  …
  -/
  let H x h := pow_le_pow_of_le_one (sin_nonneg_of_mem_Icc h) (sin_le_one x) (n.le_add_right 1)
  /-
    n : Nat
    H : ∀ (x : Real), Membership.mem (Set.Icc 0 Real.pi) x → LE.le (HPow.hPow (Rea …
    ⊢ LE.le (intervalIntegral (fun x => HPow.hPow (Real.sin x) (HAdd.hAdd n 1)) 0  …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  refine integral_mono_on pi_pos.le ?_ ?_ H <;> exact (continuous_sin.pow _).intervalIntegrable 0 π
                                                /-
                                                  🎉 no goals
                                                -/


theorem integral_sin_pow_antitone : Antitone fun n : ℕ => ∫ x in (0)..π, sin x ^ n :=
  antitone_nat_of_succ_le integral_sin_pow_succ_le


theorem integral_cos_pow_aux :
    (∫ x in a..b, cos x ^ (n + 2)) =
      (cos b ^ (n + 1) * sin b - cos a ^ (n + 1) * sin a + (n + 1) * ∫ x in a..b, cos x ^ n) -
        (n + 1) * ∫ x in a..b, cos x ^ (n + 2) := by
  /-
    a b : Real
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd n 2)) a b M …
  -/
  let C := cos b ^ (n + 1) * sin b - cos a ^ (n + 1) * sin a
  /-
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.cos b) (HAdd.hAdd n 1)) (Rea …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd n 2)) a b M …
  -/
  have h : ∀ α β γ : ℝ, β * α * γ * α = β * (α * α * γ) := fun α β γ => by ring
  have hu : ∀ x ∈ [[a, b]],
      HasDerivAt (fun y => cos y ^ (n + 1)) (-(n + 1 : ℕ) * sin x * cos x ^ n) x :=
    fun x _ => by
      simpa only [mul_right_comm, neg_mul, mul_neg] using (hasDerivAt_cos x).pow (n + 1)
  /-
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.cos b) (HAdd.hAdd n 1)) (Rea …
    h : ∀ (α β γ : Real), Eq (HMul.hMul (HMul.hMul (HMul.hMul β α) γ) α) (HMul.hMu …
    hu : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun y => HPow …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd n 2)) a b M …
  -/
  have hv : ∀ x ∈ [[a, b]], HasDerivAt sin (cos x) x := fun x _ => hasDerivAt_sin x
  /-
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.cos b) (HAdd.hAdd n 1)) (Rea …
    h : ∀ (α β γ : Real), Eq (HMul.hMul (HMul.hMul (HMul.hMul β α) γ) α) (HMul.hMu …
    hu : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun y => HPow …
    hv : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt Real.sin (Real …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd n 2)) a b M …
  -/
  have H := integral_mul_deriv_eq_deriv_mul hu hv ?_ ?_
  · calc
      (∫ x in a..b, cos x ^ (n + 2)) = ∫ x in a..b, cos x ^ (n + 1) * cos x := by
        simp only [_root_.pow_succ]
      _ = C + (n + 1) * ∫ x in a..b, sin x ^ 2 * cos x ^ n := by simp [C, H, h, sq, -neg_add_rev]
      _ = C + (n + 1) * ∫ x in a..b, cos x ^ n - cos x ^ (n + 2) := by
        simp [sin_sq, sub_mul, ← pow_add, add_comm]
      _ = (C + (n + 1) * ∫ x in a..b, cos x ^ n) - (n + 1) * ∫ x in a..b, cos x ^ (n + 2) := by
        rw [integral_sub, mul_sub, add_sub_assoc] <;>
          apply Continuous.intervalIntegrable <;> fun_prop
  /-
    case refine_1
    a b : Real
    n : Nat
    C : Real := HSub.hSub (HMul.hMul (HPow.hPow (Real.cos b) (HAdd.hAdd n 1)) (Rea …
    h : ∀ (α β γ : Real), Eq (HMul.hMul (HMul.hMul (HMul.hMul β α) γ) α) (HMul.hMu …
    hu : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt (fun y => HPow …
    hv : ∀ (x : Real), Membership.mem (Set.uIcc a b) x → HasDerivAt Real.sin (Real …
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HMul.hMul (Neg.neg ↑(HAdd.hAdd n 1)) …
  -/
  all_goals apply Continuous.intervalIntegrable; fun_prop
  /-
    🎉 no goals
  -/


/-- The reduction formula for the integral of `cos x ^ n` for any natural `n ≥ 2`. -/
theorem integral_cos_pow :
    (∫ x in a..b, cos x ^ (n + 2)) =
      (cos b ^ (n + 1) * sin b - cos a ^ (n + 1) * sin a) / (n + 2) +
        (n + 1) / (n + 2) * ∫ x in a..b, cos x ^ n := by
  /-
    a b : Real
    n : Nat
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd n 2)) a b M …
  -/
  field_simp
  /-
    a b : Real
    n : Nat
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd  …
  -/
  convert eq_sub_iff_add_eq.mp (integral_cos_pow_aux n) using 1
  /-
    case h.e'_2
    a b : Real
    n : Nat
    ⊢ Eq (HMul.hMul (intervalIntegral (fun x => HPow.hPow (Real.cos x) (HAdd.hAdd  …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_cos_sq : ∫ x in a..b, cos x ^ 2 = (cos b * sin b - cos a * sin a + b - a) / 2 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) 2) a b MeasureTheory.M …
  -/
  field_simp [integral_cos_pow, add_sub_assoc]
  /-
    🎉 no goals
  -/


/-- Simplification of the integral of `sin x ^ m * cos x ^ n`, case `n` is odd. -/
theorem integral_sin_pow_mul_cos_pow_odd (m n : ℕ) :
    (∫ x in a..b, sin x ^ m * cos x ^ (2 * n + 1)) = ∫ u in sin a..sin b, u^m * (↑1 - u ^ 2) ^ n :=
                                                                   /-
                                                                     a b : Real
                                                                     m n : Nat
                                                                     ⊢ Continuous fun u => HMul.hMul (HPow.hPow u m) (HPow.hPow (HSub.hSub 1 (HPow. …
                                                                   -/
  have hc : Continuous fun u : ℝ => u ^ m * (↑1 - u ^ 2) ^ n := by fun_prop
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  calc
    (∫ x in a..b, sin x ^ m * cos x ^ (2 * n + 1)) =
        ∫ x in a..b, sin x ^ m * (↑1 - sin x ^ 2) ^ n * cos x := by
      /-
        a b : Real
        m n : Nat
        hc : Continuous fun u => HMul.hMul (HPow.hPow u m) (HPow.hPow (HSub.hSub 1 (HP …
        ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) m) (HPow.hP …
      -/
      simp only [_root_.pow_zero, _root_.pow_succ, mul_assoc, pow_mul, one_mul]
      /-
        a b : Real
        m n : Nat
        hc : Continuous fun u => HMul.hMul (HPow.hPow u m) (HPow.hPow (HSub.hSub 1 (HP …
        ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) m) (HMul.hM …
      -/
      congr! 5
      /-
        case h.e'_4.h.h.e'_6.h.e'_5.h.e'_5
        a b : Real
        m n : Nat
        hc : Continuous fun u => HMul.hMul (HPow.hPow u m) (HPow.hPow (HSub.hSub 1 (HP …
        x✝ : Real
        ⊢ Eq (HMul.hMul (Real.cos x✝) (Real.cos x✝)) (HSub.hSub 1 (HMul.hMul (Real.sin …
      -/
      rw [← sq, ← sq, cos_sq']
      /-
        🎉 no goals
      -/
    _ = ∫ u in sin a..sin b, u ^ m * (1 - u ^ 2) ^ n := by
      -- Note(kmill): Didn't need `by exact`, but elaboration order seems to matter here.
      /-
        a b : Real
        m n : Nat
        hc : Continuous fun u => HMul.hMul (HPow.hPow u m) (HPow.hPow (HSub.hSub 1 (HP …
        ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HMul.hMul (HPow.hPow (Real.sin x)  …
      -/
      exact integral_comp_mul_deriv (fun x _ => hasDerivAt_sin x) continuousOn_cos hc
      /-
        🎉 no goals
      -/


/-- The integral of `sin x * cos x`, given in terms of sin².
  See `integral_sin_mul_cos₂` below for the integral given in terms of cos². -/
@[simp]
theorem integral_sin_mul_cos₁ : ∫ x in a..b, sin x * cos x = (sin b ^ 2 - sin a ^ 2) / 2 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Real.sin x) (Real.cos x)) a b Meas …
  -/
  simpa using integral_sin_pow_mul_cos_pow_odd 1 0
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_sin_sq_mul_cos :
    ∫ x in a..b, sin x ^ 2 * cos x = (sin b ^ 3 - sin a ^ 3) / 3 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) 2) (Real.co …
  -/
  have := @integral_sin_pow_mul_cos_pow_odd a b 2 0
  /-
    a b : Real
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) 2) (HP …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) 2) (Real.co …
  -/
  norm_num at this; exact this
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem integral_cos_pow_three :
    ∫ x in a..b, cos x ^ 3 = sin b - sin a - (sin b ^ 3 - sin a ^ 3) / 3 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) 3) a b MeasureTheory.M …
  -/
  have := @integral_sin_pow_mul_cos_pow_odd a b 0 1
  /-
    a b : Real
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) 0) (HP …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) 3) a b MeasureTheory.M …
  -/
  norm_num at this; exact this
                    /-
                      🎉 no goals
                    -/


/-- Simplification of the integral of `sin x ^ m * cos x ^ n`, case `m` is odd. -/
theorem integral_sin_pow_odd_mul_cos_pow (m n : ℕ) :
    (∫ x in a..b, sin x ^ (2 * m + 1) * cos x ^ n) = ∫ u in cos b..cos a, u^n * (↑1 - u ^ 2) ^ m :=
                                                                   /-
                                                                     a b : Real
                                                                     m n : Nat
                                                                     ⊢ Continuous fun u => HMul.hMul (HPow.hPow u n) (HPow.hPow (HSub.hSub 1 (HPow. …
                                                                   -/
  have hc : Continuous fun u : ℝ => u ^ n * (↑1 - u ^ 2) ^ m := by fun_prop
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  calc
    (∫ x in a..b, sin x ^ (2 * m + 1) * cos x ^ n) =
                                                            /-
                                                              a b : Real
                                                              m n : Nat
                                                              hc : Continuous fun u => HMul.hMul (HPow.hPow u n) (HPow.hPow (HSub.hSub 1 (HP …
                                                              ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) (HAdd.hAdd  …
                                                            -/
        -∫ x in b..a, sin x ^ (2 * m + 1) * cos x ^ n := by rw [integral_symm]
                                                            /-
                                                              🎉 no goals
                                                            -/
    _ = ∫ x in b..a, (↑1 - cos x ^ 2) ^ m * -sin x * cos x ^ n := by
      simp only [_root_.pow_succ, pow_mul, _root_.pow_zero, one_mul, mul_neg, neg_mul,
        integral_neg, neg_inj]
      /-
        a b : Real
        m n : Nat
        hc : Continuous fun u => HMul.hMul (HPow.hPow u n) (HPow.hPow (HSub.hSub 1 (HP …
        ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HMul.hMul (HPow.hPow (HMul.hMul (R …
      -/
      congr! 5
      /-
        case h.e'_4.h.h.e'_5.h.e'_5.h.e'_5
        a b : Real
        m n : Nat
        hc : Continuous fun u => HMul.hMul (HPow.hPow u n) (HPow.hPow (HSub.hSub 1 (HP …
        x✝ : Real
        ⊢ Eq (HMul.hMul (Real.sin x✝) (Real.sin x✝)) (HSub.hSub 1 (HMul.hMul (Real.cos …
      -/
      rw [← sq, ← sq, sin_sq]
      /-
        🎉 no goals
      -/
                                                                     /-
                                                                       a b : Real
                                                                       m n : Nat
                                                                       hc : Continuous fun u => HMul.hMul (HPow.hPow u n) (HPow.hPow (HSub.hSub 1 (HP …
                                                                       ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HMul.hMul (HPow.hPow (HSub.hSub 1  …
                                                                     -/
    _ = ∫ x in b..a, cos x ^ n * (↑1 - cos x ^ 2) ^ m * -sin x := by congr; ext; ring
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    _ = ∫ u in cos b..cos a, u ^ n * (↑1 - u ^ 2) ^ m :=
      integral_comp_mul_deriv (fun x _ => hasDerivAt_cos x) continuousOn_sin.neg hc


/-- The integral of `sin x * cos x`, given in terms of cos².
See `integral_sin_mul_cos₁` above for the integral given in terms of sin². -/
theorem integral_sin_mul_cos₂ : ∫ x in a..b, sin x * cos x = (cos a ^ 2 - cos b ^ 2) / 2 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Real.sin x) (Real.cos x)) a b Meas …
  -/
  simpa using integral_sin_pow_odd_mul_cos_pow 0 1
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_sin_mul_cos_sq :
    ∫ x in a..b, sin x * cos x ^ 2 = (cos a ^ 3 - cos b ^ 3) / 3 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Real.sin x) (HPow.hPow (Real.cos x …
  -/
  have := @integral_sin_pow_odd_mul_cos_pow a b 0 2
  /-
    a b : Real
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) (HAdd. …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (Real.sin x) (HPow.hPow (Real.cos x …
  -/
  norm_num at this; exact this
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem integral_sin_pow_three :
    ∫ x in a..b, sin x ^ 3 = cos a - cos b - (cos a ^ 3 - cos b ^ 3) / 3 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) 3) a b MeasureTheory.M …
  -/
  have := @integral_sin_pow_odd_mul_cos_pow a b 1 0
  /-
    a b : Real
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) (HAdd. …
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.sin x) 3) a b MeasureTheory.M …
  -/
  norm_num at this; exact this
                    /-
                      🎉 no goals
                    -/


/-- Simplification of the integral of `sin x ^ m * cos x ^ n`, case `m` and `n` are both even. -/
theorem integral_sin_pow_even_mul_cos_pow_even (m n : ℕ) :
    (∫ x in a..b, sin x ^ (2 * m) * cos x ^ (2 * n)) =
      ∫ x in a..b, ((1 - cos (2 * x)) / 2) ^ m * ((1 + cos (2 * x)) / 2) ^ n := by
  /-
    a b : Real
    m n : Nat
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) (HMul.hMul  …
  -/
  field_simp [pow_mul, sin_sq, cos_sq, ← sub_sub, (by ring : (2 : ℝ) - 1 = 1)]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_sin_sq_mul_cos_sq :
    ∫ x in a..b, sin x ^ 2 * cos x ^ 2 = (b - a) / 8 - (sin (4 * b) - sin (4 * a)) / 32 := by
  /-
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (Real.sin x) 2) (HPow.hP …
  -/
  convert integral_sin_pow_even_mul_cos_pow_even 1 1 using 1
  /-
    case h.e'_3
    a b : Real
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub b a) 8) (HDiv.hDiv (HSub.hSub (Real.sin  …
  -/
  have h1 : ∀ c : ℝ, (↑1 - c) / ↑2 * ((↑1 + c) / ↑2) = (↑1 - c ^ 2) / 4 := fun c => by ring
  /-
    case h.e'_3
    a b : Real
    h1 : ∀ (c : Real), Eq (HMul.hMul (HDiv.hDiv (HSub.hSub 1 c) 2) (HDiv.hDiv (HAd …
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub b a) 8) (HDiv.hDiv (HSub.hSub (Real.sin  …
  -/
  have h2 : Continuous fun x => cos (2 * x) ^ 2 := by fun_prop
  /-
    case h.e'_3
    a b : Real
    h1 : ∀ (c : Real), Eq (HMul.hMul (HDiv.hDiv (HSub.hSub 1 c) 2) (HDiv.hDiv (HAd …
    h2 : Continuous fun x => HPow.hPow (Real.cos (HMul.hMul 2 x)) 2
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub b a) 8) (HDiv.hDiv (HSub.hSub (Real.sin  …
  -/
  have h3 : ∀ x, cos x * sin x = sin (2 * x) / 2 := by intro; rw [sin_two_mul]; ring
  /-
    case h.e'_3
    a b : Real
    h1 : ∀ (c : Real), Eq (HMul.hMul (HDiv.hDiv (HSub.hSub 1 c) 2) (HDiv.hDiv (HAd …
    h2 : Continuous fun x => HPow.hPow (Real.cos (HMul.hMul 2 x)) 2
    h3 : ∀ (x : Real), Eq (HMul.hMul (Real.cos x) (Real.sin x)) (HDiv.hDiv (Real.s …
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub b a) 8) (HDiv.hDiv (HSub.hSub (Real.sin  …
  -/
  have h4 : ∀ d : ℝ, 2 * (2 * d) = 4 * d := fun d => by ring
  /-
    case h.e'_3
    a b : Real
    h1 : ∀ (c : Real), Eq (HMul.hMul (HDiv.hDiv (HSub.hSub 1 c) 2) (HDiv.hDiv (HAd …
    h2 : Continuous fun x => HPow.hPow (Real.cos (HMul.hMul 2 x)) 2
    h3 : ∀ (x : Real), Eq (HMul.hMul (Real.cos x) (Real.sin x)) (HDiv.hDiv (Real.s …
    h4 : ∀ (d : Real), Eq (HMul.hMul 2 (HMul.hMul 2 d)) (HMul.hMul 4 d)
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub b a) 8) (HDiv.hDiv (HSub.hSub (Real.sin  …
  -/
  simp [h1, h2.intervalIntegrable, integral_comp_mul_left fun x => cos x ^ 2, h3, h4]
  /-
    case h.e'_3
    a b : Real
    h1 : ∀ (c : Real), Eq (HMul.hMul (HDiv.hDiv (HSub.hSub 1 c) 2) (HDiv.hDiv (HAd …
    h2 : Continuous fun x => HPow.hPow (Real.cos (HMul.hMul 2 x)) 2
    h3 : ∀ (x : Real), Eq (HMul.hMul (Real.cos x) (Real.sin x)) (HDiv.hDiv (Real.s …
    h4 : ∀ (d : Real), Eq (HMul.hMul 2 (HMul.hMul 2 d)) (HMul.hMul 4 d)
    ⊢ Eq (HSub.hSub (HDiv.hDiv (HSub.hSub b a) 8) (HDiv.hDiv (HSub.hSub (Real.sin  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem integral_sqrt_one_sub_sq : ∫ x in (-1 : ℝ)..1, √(1 - x ^ 2 : ℝ) = π / 2 :=
  calc
                                                                    /-
                                                                      ⊢ Eq (intervalIntegral (fun x => (HSub.hSub 1 (HPow.hPow x 2)).sqrt) (-1) 1 Me …
                                                                    -/
    _ = ∫ x in sin (-(π / 2)).. sin (π / 2), √(1 - x ^ 2 : ℝ) := by rw [sin_neg, sin_pi_div_two]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    _ = ∫ x in (-(π / 2))..(π / 2), √(1 - sin x ^ 2 : ℝ) * cos x :=
          (integral_comp_mul_deriv (fun x _ => hasDerivAt_sin x) continuousOn_cos
                /-
                  ⊢ Continuous fun x => (HSub.hSub 1 (HPow.hPow x 2)).sqrt
                -/
            (by fun_prop)).symm
                /-
                  🎉 no goals
                -/
    _ = ∫ x in (-(π / 2))..(π / 2), cos x ^ 2 := by
          /-
            ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HSub.hSub 1 (HPow.hPow (Real.sin x …
          -/
          refine integral_congr_ae (MeasureTheory.ae_of_all _ fun _ h => ?_)
          /-
            x✝ : Real
            h : Membership.mem (Set.uIoc (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
            ⊢ Eq (HMul.hMul (HSub.hSub 1 (HPow.hPow (Real.sin x✝) 2)).sqrt (Real.cos x✝))  …
          -/
          rw [uIoc_of_le (neg_le_self (le_of_lt (half_pos Real.pi_pos))), Set.mem_Ioc] at h
          /-
            x✝ : Real
            h : And (LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x✝) (LE.le x✝ (HDiv.hDiv Real.p …
            ⊢ Eq (HMul.hMul (HSub.hSub 1 (HPow.hPow (Real.sin x✝) 2)).sqrt (Real.cos x✝))  …
          -/
          rw [← Real.cos_eq_sqrt_one_sub_sin_sq (le_of_lt h.1) h.2, pow_two]
          /-
            🎉 no goals
          -/
                    /-
                      ⊢ Eq (intervalIntegral (fun x => HPow.hPow (Real.cos x) 2) (Neg.neg (HDiv.hDiv …
                    -/
    _ = π / 2 := by simp
                    /-
                      🎉 no goals
                    -/

