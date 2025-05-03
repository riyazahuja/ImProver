/-- Asymptotic bound for the `Γ` function integrand. -/
theorem Gamma_integrand_isLittleO (s : ℝ) :
    (fun x : ℝ => exp (-x) * x ^ s) =o[atTop] fun x : ℝ => exp (-(1 / 2) * x) := by
  /-
    s : Real
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HMul.hMul (Real.exp (Neg.neg x) …
  -/
  refine isLittleO_of_tendsto (fun x hx => ?_) ?_
    /-
      case refine_1
      s x : Real
      hx : Eq (Real.exp (HMul.hMul (Neg.neg (1 / 2)) x)) 0
      ⊢ Eq (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x s)) 0
    -/
  · exfalso; exact (exp_pos (-(1 / 2) * x)).ne' hx
             /-
               🎉 no goals
             -/
  have : (fun x : ℝ => exp (-x) * x ^ s / exp (-(1 / 2) * x)) =
      (fun x : ℝ => exp (1 / 2 * x) / x ^ s)⁻¹ := by
    ext1 x
    field_simp [exp_ne_zero, exp_neg, ← Real.exp_add]
    left
    ring
  /-
    case refine_2
    s : Real
    this : Eq (fun x => HDiv.hDiv (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x s …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HMul.hMul (Real.exp (Neg.neg x)) (HPow.h …
  -/
  rw [this]
  /-
    case refine_2
    s : Real
    this : Eq (fun x => HDiv.hDiv (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x s …
    ⊢ Filter.Tendsto (Inv.inv fun x => HDiv.hDiv (Real.exp (HMul.hMul (1 / 2) x))  …
  -/
  exact (tendsto_exp_mul_div_rpow_atTop s (1 / 2) one_half_pos).inv_tendsto_atTop
  /-
    🎉 no goals
  -/


/-- The Euler integral for the `Γ` function converges for positive real `s`. -/
theorem GammaIntegral_convergent {s : ℝ} (h : 0 < s) :
    /-
      s : Real
      h : LT.lt 0 s
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x : ℝ => exp (-x) * x ^ (s - 1)) (Ioi 0) := by
    /-
      🎉 no goals
    -/
  /-
    s : Real
    h : LT.lt 0 s
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) (HPow. …
  -/
  rw [← Ioc_union_Ioi_eq_Ioi (@zero_le_one ℝ _ _ _ _), integrableOn_union]
  /-
    s : Real
    h : LT.lt 0 s
    ⊢ And (MeasureTheory.IntegrableOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) ( …
  -/
  constructor
    /-
      case left
      s : Real
      h : LT.lt 0 s
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) (HPow. …
    -/
  · rw [← integrableOn_Icc_iff_integrableOn_Ioc]
    /-
      case left
      s : Real
      h : LT.lt 0 s
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) (HPow. …
    -/
    refine IntegrableOn.continuousOn_mul continuousOn_id.neg.rexp ?_ isCompact_Icc
    /-
      case left
      s : Real
      h : LT.lt 0 s
      ⊢ MeasureTheory.IntegrableOn (fun x => HPow.hPow x (HSub.hSub s 1)) (Set.Icc 0 …
    -/
    refine (intervalIntegrable_iff_integrableOn_Icc_of_le zero_le_one).mp ?_
    /-
      case left
      s : Real
      h : LT.lt 0 s
      ⊢ IntervalIntegrable (fun x => HPow.hPow x (HSub.hSub s 1)) MeasureTheory.Meas …
    -/
    exact intervalIntegrable_rpow' (by linarith)
    /-
      🎉 no goals
    -/
    /-
      case right
      s : Real
      h : LT.lt 0 s
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) (HPow. …
    -/
  · refine integrable_of_isBigO_exp_neg one_half_pos ?_ (Gamma_integrand_isLittleO _).isBigO
    /-
      case right
      s : Real
      h : LT.lt 0 s
      ⊢ ContinuousOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HSub.h …
    -/
    refine continuousOn_id.neg.rexp.mul (continuousOn_id.rpow_const ?_)
    /-
      case right
      s : Real
      h : LT.lt 0 s
      ⊢ ∀ (x : Real), Membership.mem (Set.Ici 1) x → Or (Ne (id x) 0) (LE.le 0 (HSub …
    -/
    intro x hx
    /-
      case right
      s : Real
      h : LT.lt 0 s
      x : Real
      hx : Membership.mem (Set.Ici 1) x
      ⊢ Or (Ne (id x) 0) (LE.le 0 (HSub.hSub s 1))
    -/
    exact Or.inl ((zero_lt_one : (0 : ℝ) < 1).trans_le hx).ne'
    /-
      🎉 no goals
    -/


/-- The integral defining the `Γ` function converges for complex `s` with `0 < re s`.

This is proved by reduction to the real case. -/
theorem GammaIntegral_convergent {s : ℂ} (hs : 0 < s.re) :
    /-
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ MeasureTheory.Measure Real
    -/
    IntegrableOn (fun x => (-x).exp * x ^ (s - 1) : ℝ → ℂ) (Ioi 0) := by
    /-
      🎉 no goals
    -/
  /-
    s : Complex
    hs : LT.lt 0 s.re
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HP …
  -/
  constructor
    /-
      case left
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HMul.hMul (↑(Real.exp (Neg.neg  …
    -/
  · refine ContinuousOn.aestronglyMeasurable ?_ measurableSet_Ioi
    /-
      case left
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ ContinuousOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) ( …
    -/
    apply (continuous_ofReal.comp continuous_neg.rexp).continuousOn.mul
    /-
      case left
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ ContinuousOn (fun x => HPow.hPow (↑x) (HSub.hSub s 1)) (Set.Ioi 0)
    -/
    apply continuousOn_of_forall_continuousAt
    /-
      case left.hcont
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ ∀ (x : Real), Membership.mem (Set.Ioi 0) x → ContinuousAt (fun x => HPow.hPo …
    -/
    intro x hx
    have : ContinuousAt (fun x : ℂ => x ^ (s - 1)) ↑x :=
      continuousAt_cpow_const <| ofReal_mem_slitPlane.2 hx
    /-
      case left.hcont
      s : Complex
      hs : LT.lt 0 s.re
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      this : ContinuousAt (fun x => HPow.hPow x (HSub.hSub s 1)) ↑x
      ⊢ ContinuousAt (fun x => HPow.hPow (↑x) (HSub.hSub s 1)) x
    -/
    exact ContinuousAt.comp this continuous_ofReal.continuousAt
    /-
      🎉 no goals
    -/
    /-
      case right
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul (↑(Real.exp (Neg.neg x)) …
    -/
  · rw [← hasFiniteIntegral_norm_iff]
    /-
      case right
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ MeasureTheory.HasFiniteIntegral (fun a => Norm.norm (HMul.hMul (↑(Real.exp ( …
    -/
    refine HasFiniteIntegral.congr (Real.GammaIntegral_convergent hs).2 ?_
    /-
      case right
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioi 0))). …
    -/
    apply (ae_restrict_iff' measurableSet_Ioi).mpr
    /-
      case right
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → Eq ((fun x => HMu …
    -/
    filter_upwards with x hx
    rw [norm_eq_abs, map_mul, abs_of_nonneg <| le_of_lt <| exp_pos <| -x,
      abs_cpow_eq_rpow_re_of_pos hx _]
    /-
      case right.h
      s : Complex
      hs : LT.lt 0 s.re
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ Eq (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HSub.hSub s.re 1))) (HMul …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Euler's integral for the `Γ` function (of a complex variable `s`), defined as
`∫ x in Ioi 0, exp (-x) * x ^ (s - 1)`.

See `Complex.GammaIntegral_convergent` for a proof of the convergence of the integral for
`0 < re s`. -/
def GammaIntegral (s : ℂ) : ℂ :=
  ∫ x in Ioi (0 : ℝ), ↑(-x).exp * ↑x ^ (s - 1)


theorem GammaIntegral_conj (s : ℂ) : GammaIntegral (conj s) = conj (GammaIntegral s) := by
  /-
    s : Complex
    ⊢ Eq ((starRingEnd Complex) s).GammaIntegral ((starRingEnd Complex) s.GammaInt …
  -/
  rw [GammaIntegral, GammaIntegral, ← integral_conj]
  /-
    s : Complex
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioi fun x hx => ?_
  /-
    s : Complex
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) (HSub.hSub ((starRin …
  -/
  dsimp only
  rw [RingHom.map_mul, conj_ofReal, cpow_def_of_ne_zero (ofReal_ne_zero.mpr (ne_of_gt hx)),
    cpow_def_of_ne_zero (ofReal_ne_zero.mpr (ne_of_gt hx)), ← exp_conj, RingHom.map_mul, ←
    ofReal_log (le_of_lt hx), conj_ofReal, RingHom.map_sub, RingHom.map_one]


theorem GammaIntegral_ofReal (s : ℝ) :
    GammaIntegral ↑s = ↑(∫ x : ℝ in Ioi 0, Real.exp (-x) * x ^ (s - 1)) := by
  /-
    s : Real
    ⊢ Eq (↑s).GammaIntegral ↑(MeasureTheory.integral (MeasureTheory.MeasureSpace.v …
  -/
  have : ∀ r : ℝ, Complex.ofReal r = @RCLike.ofReal ℂ _ r := fun r => rfl
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    ⊢ Eq (↑s).GammaIntegral ↑(MeasureTheory.integral (MeasureTheory.MeasureSpace.v …
  -/
  rw [GammaIntegral]
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  conv_rhs => rw [this, ← _root_.integral_ofReal]
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioi ?_
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    ⊢ Set.EqOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) (HSub …
  -/
  intro x hx; dsimp only
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) (HSub.hSub (↑s) 1))) …
  -/
  conv_rhs => rw [← this]
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) (HSub.hSub (↑s) 1))) …
  -/
  rw [ofReal_mul, ofReal_cpow (mem_Ioi.mp hx).le]
  /-
    s : Real
    this : ∀ (r : Real), Eq ↑r ↑r
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) (HSub.hSub (↑s) 1))) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem GammaIntegral_one : GammaIntegral 1 = 1 := by
  simpa only [← ofReal_one, GammaIntegral_ofReal, ofReal_inj, sub_self, rpow_zero,
    mul_one] using integral_exp_neg_Ioi_zero


/-- The indefinite version of the `Γ` function, `Γ(s, X) = ∫ x ∈ 0..X, exp(-x) x ^ (s - 1)`. -/
def partialGamma (s : ℂ) (X : ℝ) : ℂ :=
  ∫ x in (0)..X, (-x).exp * x ^ (s - 1)


theorem tendsto_partialGamma {s : ℂ} (hs : 0 < s.re) :
    Tendsto (fun X : ℝ => partialGamma s X) atTop (𝓝 <| GammaIntegral s) :=
  intervalIntegral_tendsto_integral_Ioi 0 (GammaIntegral_convergent hs) tendsto_id


private theorem Gamma_integrand_intervalIntegrable (s : ℂ) {X : ℝ} (hs : 0 < s.re) (hX : 0 ≤ X) :
    IntervalIntegrable (fun x => (-x).exp * x ^ (s - 1) : ℝ → ℂ) volume 0 X := by
  /-
    s : Complex
    X : Real
    hs : LT.lt 0 s.re
    hX : LE.le 0 X
    ⊢ IntervalIntegrable (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow  …
  -/
  rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hX]
  /-
    s : Complex
    X : Real
    hs : LT.lt 0 s.re
    hX : LE.le 0 X
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HP …
  -/
  exact IntegrableOn.mono_set (GammaIntegral_convergent hs) Ioc_subset_Ioi_self
  /-
    🎉 no goals
  -/


private theorem Gamma_integrand_deriv_integrable_A {s : ℂ} (hs : 0 < s.re) {X : ℝ} (hX : 0 ≤ X) :
    IntervalIntegrable (fun x => -((-x).exp * x ^ s) : ℝ → ℂ) volume 0 X := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    ⊢ IntervalIntegrable (fun x => Neg.neg (HMul.hMul (↑(Real.exp (Neg.neg x))) (H …
  -/
  convert (Gamma_integrand_intervalIntegrable (s + 1) _ hX).neg
    /-
      case h.e'_3.h
      s : Complex
      hs : LT.lt 0 s.re
      X : Real
      hX : LE.le 0 X
      x✝ : Real
      ⊢ Eq (Neg.neg (HMul.hMul (↑(Real.exp (Neg.neg x✝))) (HPow.hPow (↑x✝) s))) (Neg …
    -/
  · simp only [ofReal_exp, ofReal_neg, add_sub_cancel_right]; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/
    /-
      s : Complex
      hs : LT.lt 0 s.re
      X : Real
      hX : LE.le 0 X
      ⊢ LT.lt 0 (HAdd.hAdd s 1).re
    -/
  · simp only [add_re, one_re]; linarith
                                /-
                                  🎉 no goals
                                -/


private theorem Gamma_integrand_deriv_integrable_B {s : ℂ} (hs : 0 < s.re) {Y : ℝ} (hY : 0 ≤ Y) :
    IntervalIntegrable (fun x : ℝ => (-x).exp * (s * x ^ (s - 1)) : ℝ → ℂ) volume 0 Y := by
  have : (fun x => (-x).exp * (s * x ^ (s - 1)) : ℝ → ℂ) =
      (fun x => s * ((-x).exp * x ^ (s - 1)) : ℝ → ℂ) := by ext1; ring
  /-
    s : Complex
    hs : LT.lt 0 s.re
    Y : Real
    hY : LE.le 0 Y
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ IntervalIntegrable (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul  …
  -/
  rw [this, intervalIntegrable_iff_integrableOn_Ioc_of_le hY]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    Y : Real
    hY : LE.le 0 Y
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul s (HMul.hMul (↑(Real.exp (Neg …
  -/
  constructor
    /-
      case left
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HMul.hMul s (HMul.hMul (↑(Real. …
    -/
  · refine (continuousOn_const.mul ?_).aestronglyMeasurable measurableSet_Ioc
    /-
      case left
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      ⊢ ContinuousOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (↑x) ( …
    -/
    apply (continuous_ofReal.comp continuous_neg.rexp).continuousOn.mul
    /-
      case left
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      ⊢ ContinuousOn (fun x => HPow.hPow (↑x) (HSub.hSub s 1)) (Set.Ioc 0 Y)
    -/
    apply continuousOn_of_forall_continuousAt
    /-
      case left.hcont
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      ⊢ ∀ (x : Real), Membership.mem (Set.Ioc 0 Y) x → ContinuousAt (fun x => HPow.h …
    -/
    intro x hx
    /-
      case left.hcont
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      x : Real
      hx : Membership.mem (Set.Ioc 0 Y) x
      ⊢ ContinuousAt (fun x => HPow.hPow (↑x) (HSub.hSub s 1)) x
    -/
    refine (?_ : ContinuousAt (fun x : ℂ => x ^ (s - 1)) _).comp continuous_ofReal.continuousAt
    /-
      case left.hcont
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      x : Real
      hx : Membership.mem (Set.Ioc 0 Y) x
      ⊢ ContinuousAt (fun x => HPow.hPow x (HSub.hSub s 1)) ↑x
    -/
    exact continuousAt_cpow_const <| ofReal_mem_slitPlane.2 hx.1
    /-
      🎉 no goals
    -/
  /-
    case right
    s : Complex
    hs : LT.lt 0 s.re
    Y : Real
    hY : LE.le 0 Y
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul s (HMul.hMul (↑(Real.exp …
  -/
  rw [← hasFiniteIntegral_norm_iff]
  /-
    case right
    s : Complex
    hs : LT.lt 0 s.re
    Y : Real
    hY : LE.le 0 Y
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ MeasureTheory.HasFiniteIntegral (fun a => Norm.norm (HMul.hMul s (HMul.hMul  …
  -/
  simp_rw [norm_eq_abs, map_mul]
  refine (((Real.GammaIntegral_convergent hs).mono_set
    Ioc_subset_Ioi_self).hasFiniteIntegral.congr ?_).const_mul _
  /-
    case right
    s : Complex
    hs : LT.lt 0 s.re
    Y : Real
    hY : LE.le 0 Y
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc 0 Y)) …
  -/
  rw [EventuallyEq, ae_restrict_iff']
    /-
      case right
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioc 0 Y) x → Eq (HMul.hMul ( …
    -/
  · filter_upwards with x hx
    /-
      case right.h
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      x : Real
      hx : Membership.mem (Set.Ioc 0 Y) x
      ⊢ Eq (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HSub.hSub s.re 1))) (HMul …
    -/
    rw [abs_of_nonneg (exp_pos _).le, abs_cpow_eq_rpow_re_of_pos hx.1]
    /-
      case right.h
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      x : Real
      hx : Membership.mem (Set.Ioc 0 Y) x
      ⊢ Eq (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HSub.hSub s.re 1))) (HMul …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case right
      s : Complex
      hs : LT.lt 0 s.re
      Y : Real
      hY : LE.le 0 Y
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      ⊢ MeasurableSet (Set.Ioc 0 Y)
    -/
  · exact measurableSet_Ioc
    /-
      🎉 no goals
    -/


/-- The recurrence relation for the indefinite version of the `Γ` function. -/
theorem partialGamma_add_one {s : ℂ} (hs : 0 < s.re) {X : ℝ} (hX : 0 ≤ X) :
    partialGamma (s + 1) X = s * partialGamma s X - (-X).exp * X ^ s := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    ⊢ Eq ((HAdd.hAdd s 1).partialGamma X) (HSub.hSub (HMul.hMul s (s.partialGamma  …
  -/
  rw [partialGamma, partialGamma, add_sub_cancel_right]
  have F_der_I : ∀ x : ℝ, x ∈ Ioo 0 X → HasDerivAt (fun x => (-x).exp * x ^ s : ℝ → ℂ)
      (-((-x).exp * x ^ s) + (-x).exp * (s * x ^ (s - 1))) x := by
    intro x hx
    have d1 : HasDerivAt (fun y : ℝ => (-y).exp) (-(-x).exp) x := by
      simpa using (hasDerivAt_neg x).exp
    have d2 : HasDerivAt (fun y : ℝ => (y : ℂ) ^ s) (s * x ^ (s - 1)) x := by
      have t := @HasDerivAt.cpow_const _ _ _ s (hasDerivAt_id ↑x) ?_
      · simpa only [mul_one] using t.comp_ofReal
      · exact ofReal_mem_slitPlane.2 hx.1
    simpa only [ofReal_neg, neg_mul] using d1.ofReal_comp.mul d2
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPo …
  -/
  have cont := (continuous_ofReal.comp continuous_neg.rexp).mul (continuous_ofReal_cpow_const hs)
  have der_ible :=
    (Gamma_integrand_deriv_integrable_A hs hX).add (Gamma_integrand_deriv_integrable_B hs hX)
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
    der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPo …
  -/
  have int_eval := integral_eq_sub_of_hasDerivAt_of_le hX cont.continuousOn F_der_I der_ible
  -- We are basically done here but manipulating the output into the right form is fiddly.
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
    der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
    int_eval : Eq (intervalIntegral (fun y => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Rea …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPo …
  -/
  apply_fun fun x : ℂ => -x at int_eval
  rw [intervalIntegral.integral_add (Gamma_integrand_deriv_integrable_A hs hX)
      (Gamma_integrand_deriv_integrable_B hs hX),
    intervalIntegral.integral_neg, neg_add, neg_neg] at int_eval
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
    der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
    int_eval : Eq (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Ne …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPo …
  -/
  rw [eq_sub_of_add_eq int_eval, sub_neg_eq_add, neg_sub, add_comm, add_sub]
  have : (fun x => (-x).exp * (s * x ^ (s - 1)) : ℝ → ℂ) =
      (fun x => s * (-x).exp * x ^ (s - 1) : ℝ → ℂ) := by ext1; ring
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
    der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
    int_eval : Eq (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Ne …
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp ( …
  -/
  rw [this]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
    der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
    int_eval : Eq (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Ne …
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (HMul.hMul s  …
  -/
  have t := @integral_const_mul 0 X volume _ _ s fun x : ℝ => (-x).exp * x ^ (s - 1)
  /-
    s : Complex
    hs : LT.lt 0 s.re
    X : Real
    hX : LE.le 0 X
    F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
    cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
    der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
    int_eval : Eq (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Ne …
    this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
    t : Eq (intervalIntegral (fun x => HMul.hMul s (HMul.hMul (↑(Real.exp (Neg.neg …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (HMul.hMul s  …
  -/
  rw [← t, ofReal_zero, zero_cpow]
    /-
      s : Complex
      hs : LT.lt 0 s.re
      X : Real
      hX : LE.le 0 X
      F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
      cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
      der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
      int_eval : Eq (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Ne …
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      t : Eq (intervalIntegral (fun x => HMul.hMul s (HMul.hMul (↑(Real.exp (Neg.neg …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (HMul.hMul s  …
    -/
  · rw [mul_zero, add_zero]; congr 2; ext1; ring
                                            /-
                                              🎉 no goals
                                            -/
    /-
      s : Complex
      hs : LT.lt 0 s.re
      X : Real
      hX : LE.le 0 X
      F_der_I : ∀ (x : Real), Membership.mem (Set.Ioo 0 X) x → HasDerivAt (fun x =>  …
      cont : Continuous fun x => HMul.hMul (Function.comp Complex.ofReal (fun y => R …
      der_ible : IntervalIntegrable (fun x => HAdd.hAdd (Neg.neg (HMul.hMul (↑(Real. …
      int_eval : Eq (HAdd.hAdd (intervalIntegral (fun x => HMul.hMul (↑(Real.exp (Ne …
      this : Eq (fun x => HMul.hMul (↑(Real.exp (Neg.neg x))) (HMul.hMul s (HPow.hPo …
      t : Eq (intervalIntegral (fun x => HMul.hMul s (HMul.hMul (↑(Real.exp (Neg.neg …
      ⊢ Ne s 0
    -/
  · contrapose! hs; rw [hs, zero_re]
                    /-
                      🎉 no goals
                    -/


/-- The recurrence relation for the `Γ` integral. -/
theorem GammaIntegral_add_one {s : ℂ} (hs : 0 < s.re) :
    GammaIntegral (s + 1) = s * GammaIntegral s := by
  suffices Tendsto (s + 1).partialGamma atTop (𝓝 <| s * GammaIntegral s) by
    refine tendsto_nhds_unique ?_ this
    apply tendsto_partialGamma; rw [add_re, one_re]; linarith
  have : (fun X : ℝ => s * partialGamma s X - X ^ s * (-X).exp) =ᶠ[atTop]
      (s + 1).partialGamma := by
    apply eventuallyEq_of_mem (Ici_mem_atTop (0 : ℝ))
    intro X hX
    rw [partialGamma_add_one hs (mem_Ici.mp hX)]
    ring_nf
  /-
    s : Complex
    hs : LT.lt 0 s.re
    this : Filter.atTop.EventuallyEq (fun X => HSub.hSub (HMul.hMul s (s.partialGa …
    ⊢ Filter.Tendsto (HAdd.hAdd s 1).partialGamma Filter.atTop (nhds (HMul.hMul s  …
  -/
  refine Tendsto.congr' this ?_
  suffices Tendsto (fun X => -X ^ s * (-X).exp : ℝ → ℂ) atTop (𝓝 0) by
    simpa using Tendsto.add (Tendsto.const_mul s (tendsto_partialGamma hs)) this
  /-
    s : Complex
    hs : LT.lt 0 s.re
    this : Filter.atTop.EventuallyEq (fun X => HSub.hSub (HMul.hMul s (s.partialGa …
    ⊢ Filter.Tendsto (fun X => HMul.hMul (Neg.neg (HPow.hPow (↑X) s)) ↑(Real.exp ( …
  -/
  rw [tendsto_zero_iff_norm_tendsto_zero]
  have :
      (fun e : ℝ => ‖-(e : ℂ) ^ s * (-e).exp‖) =ᶠ[atTop] fun e : ℝ => e ^ s.re * (-1 * e).exp := by
    refine eventuallyEq_of_mem (Ioi_mem_atTop 0) ?_
    intro x hx; dsimp only
    rw [norm_eq_abs, map_mul, abs.map_neg, abs_cpow_eq_rpow_re_of_pos hx,
      abs_of_nonneg (exp_pos (-x)).le, neg_mul, one_mul]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    this✝ : Filter.atTop.EventuallyEq (fun X => HSub.hSub (HMul.hMul s (s.partialG …
    this : Filter.atTop.EventuallyEq (fun e => Norm.norm (HMul.hMul (Neg.neg (HPow …
    ⊢ Filter.Tendsto (fun x => Norm.norm (HMul.hMul (Neg.neg (HPow.hPow (↑x) s)) ↑ …
  -/
  exact (tendsto_congr' this).mpr (tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero _ _ zero_lt_one)
  /-
    🎉 no goals
  -/


/-- The `n`th function in this family is `Γ(s)` if `-n < s.re`, and junk otherwise. -/
noncomputable def GammaAux : ℕ → ℂ → ℂ
  | 0 => GammaIntegral
  | n + 1 => fun s : ℂ => GammaAux n (s + 1) / s


theorem GammaAux_recurrence1 (s : ℂ) (n : ℕ) (h1 : -s.re < ↑n) :
    GammaAux n s = GammaAux n (s + 1) / s := by
  /-
    s : Complex
    n : Nat
    h1 : LT.lt (Neg.neg s.re) ↑n
    ⊢ Eq (Complex.GammaAux n s) (HDiv.hDiv (Complex.GammaAux n (HAdd.hAdd s 1)) s)
  -/
  induction' n with n hn generalizing s
    /-
      case zero
      s : Complex
      h1 : LT.lt (Neg.neg s.re) ↑0
      ⊢ Eq (Complex.GammaAux 0 s) (HDiv.hDiv (Complex.GammaAux 0 (HAdd.hAdd s 1)) s)
    -/
  · simp only [CharP.cast_eq_zero, Left.neg_neg_iff] at h1
    /-
      case zero
      s : Complex
      h1 : LT.lt 0 s.re
      ⊢ Eq (Complex.GammaAux 0 s) (HDiv.hDiv (Complex.GammaAux 0 (HAdd.hAdd s 1)) s)
    -/
    dsimp only [GammaAux]; rw [GammaIntegral_add_one h1]
    /-
      case zero
      s : Complex
      h1 : LT.lt 0 s.re
      ⊢ Eq s.GammaIntegral (HDiv.hDiv (HMul.hMul s s.GammaIntegral) s)
    -/
    rw [mul_comm, mul_div_cancel_right₀]; contrapose! h1; rw [h1]
    /-
      case zero.hb
      s : Complex
      h1 : Eq s 0
      ⊢ LE.le (Complex.re 0) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      hn : ∀ (s : Complex), LT.lt (Neg.neg s.re) ↑n → Eq (Complex.GammaAux n s) (HDi …
      s : Complex
      h1 : LT.lt (Neg.neg s.re) ↑(HAdd.hAdd n 1)
      ⊢ Eq (Complex.GammaAux (HAdd.hAdd n 1) s) (HDiv.hDiv (Complex.GammaAux (HAdd.h …
    -/
  · dsimp only [GammaAux]
    have hh1 : -(s + 1).re < n := by
      rw [Nat.cast_add, Nat.cast_one] at h1
      rw [add_re, one_re]; linarith
    /-
      case succ
      n : Nat
      hn : ∀ (s : Complex), LT.lt (Neg.neg s.re) ↑n → Eq (Complex.GammaAux n s) (HDi …
      s : Complex
      h1 : LT.lt (Neg.neg s.re) ↑(HAdd.hAdd n 1)
      hh1 : LT.lt (Neg.neg (HAdd.hAdd s 1).re) ↑n
      ⊢ Eq (HDiv.hDiv (Complex.GammaAux n (HAdd.hAdd s 1)) s) (HDiv.hDiv (HDiv.hDiv  …
    -/
    rw [← hn (s + 1) hh1]
    /-
      🎉 no goals
    -/


theorem GammaAux_recurrence2 (s : ℂ) (n : ℕ) (h1 : -s.re < ↑n) :
    GammaAux n s = GammaAux (n + 1) s := by
  /-
    s : Complex
    n : Nat
    h1 : LT.lt (Neg.neg s.re) ↑n
    ⊢ Eq (Complex.GammaAux n s) (Complex.GammaAux (HAdd.hAdd n 1) s)
  -/
  cases' n with n n
    /-
      case zero
      s : Complex
      h1 : LT.lt (Neg.neg s.re) ↑0
      ⊢ Eq (Complex.GammaAux 0 s) (Complex.GammaAux (HAdd.hAdd 0 1) s)
    -/
  · simp only [CharP.cast_eq_zero, Left.neg_neg_iff] at h1
    /-
      case zero
      s : Complex
      h1 : LT.lt 0 s.re
      ⊢ Eq (Complex.GammaAux 0 s) (Complex.GammaAux (HAdd.hAdd 0 1) s)
    -/
    dsimp only [GammaAux]
    /-
      case zero
      s : Complex
      h1 : LT.lt 0 s.re
      ⊢ Eq s.GammaIntegral (HDiv.hDiv (HAdd.hAdd s 1).GammaIntegral s)
    -/
    rw [GammaIntegral_add_one h1, mul_div_cancel_left₀]
    /-
      case zero.ha
      s : Complex
      h1 : LT.lt 0 s.re
      ⊢ Ne s 0
    -/
    rintro rfl
    /-
      case zero.ha
      h1 : LT.lt 0 (Complex.re 0)
      ⊢ False
    -/
    rw [zero_re] at h1
    /-
      case zero.ha
      h1 : LT.lt 0 0
      ⊢ False
    -/
    exact h1.false
    /-
      🎉 no goals
    -/
    /-
      case succ
      s : Complex
      n : Nat
      h1 : LT.lt (Neg.neg s.re) ↑(HAdd.hAdd n 1)
      ⊢ Eq (Complex.GammaAux (HAdd.hAdd n 1) s) (Complex.GammaAux (HAdd.hAdd (HAdd.h …
    -/
  · dsimp only [GammaAux]
    have : GammaAux n (s + 1 + 1) / (s + 1) = GammaAux n (s + 1) := by
      have hh1 : -(s + 1).re < n := by
        rw [Nat.cast_add, Nat.cast_one] at h1
        rw [add_re, one_re]; linarith
      rw [GammaAux_recurrence1 (s + 1) n hh1]
    /-
      case succ
      s : Complex
      n : Nat
      h1 : LT.lt (Neg.neg s.re) ↑(HAdd.hAdd n 1)
      this : Eq (HDiv.hDiv (Complex.GammaAux n (HAdd.hAdd (HAdd.hAdd s 1) 1)) (HAdd. …
      ⊢ Eq (HDiv.hDiv (Complex.GammaAux n (HAdd.hAdd s 1)) s) (HDiv.hDiv (HDiv.hDiv  …
    -/
    rw [this]
    /-
      🎉 no goals
    -/


/-- The `Γ` function (of a complex variable `s`). -/
@[pp_nodot]
irreducible_def Gamma (s : ℂ) : ℂ :=
  GammaAux ⌊1 - s.re⌋₊ s


theorem Gamma_eq_GammaAux (s : ℂ) (n : ℕ) (h1 : -s.re < ↑n) : Gamma s = GammaAux n s := by
  have u : ∀ k : ℕ, GammaAux (⌊1 - s.re⌋₊ + k) s = Gamma s := by
    intro k; induction' k with k hk
    · simp [Gamma]
    · rw [← hk, ← add_assoc]
      refine (GammaAux_recurrence2 s (⌊1 - s.re⌋₊ + k) ?_).symm
      rw [Nat.cast_add]
      have i0 := Nat.sub_one_lt_floor (1 - s.re)
      simp only [sub_sub_cancel_left] at i0
      refine lt_add_of_lt_of_nonneg i0 ?_
      rw [← Nat.cast_zero, Nat.cast_le]; exact Nat.zero_le k
  /-
    s : Complex
    n : Nat
    h1 : LT.lt (Neg.neg s.re) ↑n
    u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
    ⊢ Eq (Complex.Gamma s) (Complex.GammaAux n s)
  -/
  convert (u <| n - ⌊1 - s.re⌋₊).symm; rw [Nat.add_sub_of_le]
  /-
    case h.e'_3.h.e'_1
    s : Complex
    n : Nat
    h1 : LT.lt (Neg.neg s.re) ↑n
    u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
    ⊢ LE.le (Nat.floor (HSub.hSub 1 s.re)) n
  -/
  by_cases h : 0 ≤ 1 - s.re
    /-
      case pos
      s : Complex
      n : Nat
      h1 : LT.lt (Neg.neg s.re) ↑n
      u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
      h : LE.le 0 (HSub.hSub 1 s.re)
      ⊢ LE.le (Nat.floor (HSub.hSub 1 s.re)) n
    -/
  · apply Nat.le_of_lt_succ
    /-
      case pos.a
      s : Complex
      n : Nat
      h1 : LT.lt (Neg.neg s.re) ↑n
      u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
      h : LE.le 0 (HSub.hSub 1 s.re)
      ⊢ LT.lt (Nat.floor (HSub.hSub 1 s.re)) n.succ
    -/
    exact_mod_cast lt_of_le_of_lt (Nat.floor_le h) (by linarith : 1 - s.re < n + 1)
    /-
      🎉 no goals
    -/
    /-
      case neg
      s : Complex
      n : Nat
      h1 : LT.lt (Neg.neg s.re) ↑n
      u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
      h : Not (LE.le 0 (HSub.hSub 1 s.re))
      ⊢ LE.le (Nat.floor (HSub.hSub 1 s.re)) n
    -/
  · rw [Nat.floor_of_nonpos]
      /-
        case neg
        s : Complex
        n : Nat
        h1 : LT.lt (Neg.neg s.re) ↑n
        u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
        h : Not (LE.le 0 (HSub.hSub 1 s.re))
        ⊢ LE.le 0 n
      -/
    · omega
      /-
        🎉 no goals
      -/
      /-
        case neg
        s : Complex
        n : Nat
        h1 : LT.lt (Neg.neg s.re) ↑n
        u : ∀ (k : Nat), Eq (Complex.GammaAux (HAdd.hAdd (Nat.floor (HSub.hSub 1 s.re) …
        h : Not (LE.le 0 (HSub.hSub 1 s.re))
        ⊢ LE.le (HSub.hSub 1 s.re) 0
      -/
    · linarith
      /-
        🎉 no goals
      -/


/-- The recurrence relation for the `Γ` function. -/
theorem Gamma_add_one (s : ℂ) (h2 : s ≠ 0) : Gamma (s + 1) = s * Gamma s := by
  /-
    s : Complex
    h2 : Ne s 0
    ⊢ Eq (Complex.Gamma (HAdd.hAdd s 1)) (HMul.hMul s (Complex.Gamma s))
  -/
  let n := ⌊1 - s.re⌋₊
  /-
    s : Complex
    h2 : Ne s 0
    n : Nat := Nat.floor (HSub.hSub 1 s.re)
    ⊢ Eq (Complex.Gamma (HAdd.hAdd s 1)) (HMul.hMul s (Complex.Gamma s))
  -/
  have t1 : -s.re < n := by simpa only [sub_sub_cancel_left] using Nat.sub_one_lt_floor (1 - s.re)
  /-
    s : Complex
    h2 : Ne s 0
    n : Nat := Nat.floor (HSub.hSub 1 s.re)
    t1 : LT.lt (Neg.neg s.re) ↑n
    ⊢ Eq (Complex.Gamma (HAdd.hAdd s 1)) (HMul.hMul s (Complex.Gamma s))
  -/
  have t2 : -(s + 1).re < n := by rw [add_re, one_re]; linarith
  /-
    s : Complex
    h2 : Ne s 0
    n : Nat := Nat.floor (HSub.hSub 1 s.re)
    t1 : LT.lt (Neg.neg s.re) ↑n
    t2 : LT.lt (Neg.neg (HAdd.hAdd s 1).re) ↑n
    ⊢ Eq (Complex.Gamma (HAdd.hAdd s 1)) (HMul.hMul s (Complex.Gamma s))
  -/
  rw [Gamma_eq_GammaAux s n t1, Gamma_eq_GammaAux (s + 1) n t2, GammaAux_recurrence1 s n t1]
  /-
    s : Complex
    h2 : Ne s 0
    n : Nat := Nat.floor (HSub.hSub 1 s.re)
    t1 : LT.lt (Neg.neg s.re) ↑n
    t2 : LT.lt (Neg.neg (HAdd.hAdd s 1).re) ↑n
    ⊢ Eq (Complex.GammaAux n (HAdd.hAdd s 1)) (HMul.hMul s (HDiv.hDiv (Complex.Gam …
  -/
  field_simp
  /-
    🎉 no goals
  -/


theorem Gamma_eq_integral {s : ℂ} (hs : 0 < s.re) : Gamma s = GammaIntegral s :=
                            /-
                              s : Complex
                              hs : LT.lt 0 s.re
                              ⊢ LT.lt (Neg.neg s.re) ↑0
                            -/
  Gamma_eq_GammaAux s 0 (by norm_cast; linarith)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
                                      /-
                                        ⊢ Eq (Complex.Gamma 1) 1
                                      -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
theorem Gamma_one : Gamma 1 = 1 := by rw [Gamma_eq_integral] <;> simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem Gamma_nat_eq_factorial (n : ℕ) : Gamma (n + 1) = n ! := by
  induction n with
  | zero => simp
  | succ n hn =>
    rw [Gamma_add_one n.succ <| Nat.cast_ne_zero.mpr <| Nat.succ_ne_zero n]
    simp only [Nat.cast_succ, Nat.factorial_succ, Nat.cast_mul]
    congr


@[simp]
theorem Gamma_ofNat_eq_factorial (n : ℕ) [(n + 1).AtLeastTwo] :
    Gamma (no_index (OfNat.ofNat (n + 1) : ℂ)) = n ! :=
  mod_cast Gamma_nat_eq_factorial (n : ℕ)


/-- At `0` the Gamma function is undefined; by convention we assign it the value `0`. -/
@[simp]
theorem Gamma_zero : Gamma 0 = 0 := by
  /-
    ⊢ Eq (Complex.Gamma 0) 0
  -/
  simp_rw [Gamma, zero_re, sub_zero, Nat.floor_one, GammaAux, div_zero]
  /-
    🎉 no goals
  -/


/-- At `-n` for `n ∈ ℕ`, the Gamma function is undefined; by convention we assign it the value 0. -/
theorem Gamma_neg_nat_eq_zero (n : ℕ) : Gamma (-n) = 0 := by
  induction n with
  | zero => rw [Nat.cast_zero, neg_zero, Gamma_zero]
  | succ n IH =>
    have A : -(n.succ : ℂ) ≠ 0 := by
      rw [neg_ne_zero, Nat.cast_ne_zero]
      apply Nat.succ_ne_zero
    have : -(n : ℂ) = -↑n.succ + 1 := by simp
    rw [this, Gamma_add_one _ A] at IH
    contrapose! IH
    exact mul_ne_zero A IH


theorem Gamma_conj (s : ℂ) : Gamma (conj s) = conj (Gamma s) := by
  suffices ∀ (n : ℕ) (s : ℂ), GammaAux n (conj s) = conj (GammaAux n s) by
    simp [Gamma, this]
  /-
    s : Complex
    ⊢ ∀ (n : Nat) (s : Complex), Eq (Complex.GammaAux n ((starRingEnd Complex) s)) …
  -/
  intro n
  induction n with
  | zero => rw [GammaAux]; exact GammaIntegral_conj
  | succ n IH =>
    intro s
    rw [GammaAux]
    dsimp only
    rw [div_eq_mul_inv _ s, RingHom.map_mul, conj_inv, ← div_eq_mul_inv]
    suffices conj s + 1 = conj (s + 1) by rw [this, IH]
    rw [RingHom.map_add, RingHom.map_one]


/-- Expresses the integral over `Ioi 0` of `t ^ (a - 1) * exp (-(r * t))` in terms of the Gamma
function, for complex `a`. -/
lemma integral_cpow_mul_exp_neg_mul_Ioi {a : ℂ} {r : ℝ} (ha : 0 < a.re) (hr : 0 < r) :
    ∫ (t : ℝ) in Ioi 0, t ^ (a - 1) * exp (-(r * t)) = (1 / r) ^ a * Gamma a := by
  have aux : (1 / r : ℂ) ^ a = 1 / r * (1 / r) ^ (a - 1) := by
    nth_rewrite 2 [← cpow_one (1 / r : ℂ)]
    rw [← cpow_add _ _ (one_div_ne_zero <| ofReal_ne_zero.mpr hr.ne'), add_sub_cancel]
  calc
    _ = ∫ (t : ℝ) in Ioi 0, (1 / r) ^ (a - 1) * (r * t) ^ (a - 1) * exp (-(r * t)) := by
      refine MeasureTheory.setIntegral_congr_fun measurableSet_Ioi (fun x hx ↦ ?_)
      rw [mem_Ioi] at hx
      rw [mul_cpow_ofReal_nonneg hr.le hx.le, ← mul_assoc, one_div, ← ofReal_inv,
        ← mul_cpow_ofReal_nonneg (inv_pos.mpr hr).le hr.le, ← ofReal_mul r⁻¹,
        inv_mul_cancel₀ hr.ne', ofReal_one, one_cpow, one_mul]
    _ = 1 / r * ∫ (t : ℝ) in Ioi 0, (1 / r) ^ (a - 1) * t ^ (a - 1) * exp (-t) := by
      simp_rw [← ofReal_mul]
      rw [integral_comp_mul_left_Ioi (fun x ↦ _ * x ^ (a - 1) * exp (-x)) _ hr, mul_zero,
        real_smul, ← one_div, ofReal_div, ofReal_one]
    _ = 1 / r * (1 / r : ℂ) ^ (a - 1) * (∫ (t : ℝ) in Ioi 0, t ^ (a - 1) * exp (-t)) := by
      simp_rw [← integral_mul_left, mul_assoc]
    _ = (1 / r) ^ a * Gamma a := by
      rw [aux, Gamma_eq_integral ha]
      congr 2 with x
      rw [ofReal_exp, ofReal_neg, mul_comm]


/-- The `Γ` function (of a real variable `s`). -/
@[pp_nodot]
def Gamma (s : ℝ) : ℝ :=
  (Complex.Gamma s).re


theorem Gamma_eq_integral {s : ℝ} (hs : 0 < s) :
    Gamma s = ∫ x in Ioi 0, exp (-x) * x ^ (s - 1) := by
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Eq (Real.Gamma s) (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume …
  -/
  rw [Gamma, Complex.Gamma_eq_integral (by rwa [Complex.ofReal_re] : 0 < Complex.re s)]
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Eq (↑s).GammaIntegral.re (MeasureTheory.integral (MeasureTheory.MeasureSpace …
  -/
  dsimp only [Complex.GammaIntegral]
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  simp_rw [← Complex.ofReal_one, ← Complex.ofReal_sub]
  suffices ∫ x : ℝ in Ioi 0, ↑(exp (-x)) * (x : ℂ) ^ ((s - 1 : ℝ) : ℂ) =
      ∫ x : ℝ in Ioi 0, ((exp (-x) * x ^ (s - 1) : ℝ) : ℂ) by
    have cc : ∀ r : ℝ, Complex.ofReal r = @RCLike.ofReal ℂ _ r := fun r => rfl
    conv_lhs => rw [this]; enter [1, 2, x]; rw [cc]
    rw [_root_.integral_ofReal, ← cc, Complex.ofReal_re]
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioi fun x hx => ?_
  /-
    s : Real
    hs : LT.lt 0 s
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow ↑x ↑(HSub.hSub s 1))) ↑(H …
  -/
  push_cast
  /-
    s : Real
    hs : LT.lt 0 s
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (Complex.exp (Neg.neg ↑x)) (HPow.hPow (↑x) (HSub.hSub (↑s) 1)) …
  -/
  rw [Complex.ofReal_cpow (le_of_lt hx)]
  /-
    s : Real
    hs : LT.lt 0 s
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (HMul.hMul (Complex.exp (Neg.neg ↑x)) (HPow.hPow (↑x) (HSub.hSub (↑s) 1)) …
  -/
  push_cast; rfl
             /-
               🎉 no goals
             -/


theorem Gamma_add_one {s : ℝ} (hs : s ≠ 0) : Gamma (s + 1) = s * Gamma s := by
  /-
    s : Real
    hs : Ne s 0
    ⊢ Eq (Real.Gamma (HAdd.hAdd s 1)) (HMul.hMul s (Real.Gamma s))
  -/
  simp_rw [Gamma]
  /-
    s : Real
    hs : Ne s 0
    ⊢ Eq (Complex.Gamma ↑(HAdd.hAdd s 1)).re (HMul.hMul s (Complex.Gamma ↑s).re)
  -/
  rw [Complex.ofReal_add, Complex.ofReal_one, Complex.Gamma_add_one, Complex.re_ofReal_mul]
  /-
    case h2
    s : Real
    hs : Ne s 0
    ⊢ Ne (↑s) 0
  -/
  rwa [Complex.ofReal_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem Gamma_one : Gamma 1 = 1 := by
  /-
    ⊢ Eq (Real.Gamma 1) 1
  -/
  rw [Gamma, Complex.ofReal_one, Complex.Gamma_one, Complex.one_re]
  /-
    🎉 no goals
  -/


theorem _root_.Complex.Gamma_ofReal (s : ℝ) : Complex.Gamma (s : ℂ) = Gamma s := by
  /-
    s : Real
    ⊢ Eq (Complex.Gamma ↑s) ↑(Real.Gamma s)
  -/
  rw [Gamma, eq_comm, ← Complex.conj_eq_iff_re, ← Complex.Gamma_conj, Complex.conj_ofReal]
  /-
    🎉 no goals
  -/


theorem Gamma_nat_eq_factorial (n : ℕ) : Gamma (n + 1) = n ! := by
  rw [Gamma, Complex.ofReal_add, Complex.ofReal_natCast, Complex.ofReal_one,
    Complex.Gamma_nat_eq_factorial, ← Complex.ofReal_natCast, Complex.ofReal_re]


@[simp]
theorem Gamma_ofNat_eq_factorial (n : ℕ) [(n + 1).AtLeastTwo] :
    Gamma (no_index (OfNat.ofNat (n + 1) : ℝ)) = n ! :=
  mod_cast Gamma_nat_eq_factorial (n : ℕ)


/-- At `0` the Gamma function is undefined; by convention we assign it the value `0`. -/
@[simp]
theorem Gamma_zero : Gamma 0 = 0 := by
  simpa only [← Complex.ofReal_zero, Complex.Gamma_ofReal, Complex.ofReal_inj] using
    Complex.Gamma_zero


/-- At `-n` for `n ∈ ℕ`, the Gamma function is undefined; by convention we assign it the value `0`.
-/
theorem Gamma_neg_nat_eq_zero (n : ℕ) : Gamma (-n) = 0 := by
  simpa only [← Complex.ofReal_natCast, ← Complex.ofReal_neg, Complex.Gamma_ofReal,
    Complex.ofReal_eq_zero] using Complex.Gamma_neg_nat_eq_zero n


theorem Gamma_pos_of_pos {s : ℝ} (hs : 0 < s) : 0 < Gamma s := by
  /-
    s : Real
    hs : LT.lt 0 s
    ⊢ LT.lt 0 (Real.Gamma s)
  -/
  rw [Gamma_eq_integral hs]
  have : (Function.support fun x : ℝ => exp (-x) * x ^ (s - 1)) ∩ Ioi 0 = Ioi 0 := by
    rw [inter_eq_right]
    intro x hx
    rw [Function.mem_support]
    exact mul_ne_zero (exp_pos _).ne' (rpow_pos_of_pos hx _).ne'
  /-
    s : Real
    hs : LT.lt 0 s
    this : Eq (Inter.inter (Function.support fun x => HMul.hMul (Real.exp (Neg.neg …
    ⊢ LT.lt 0 (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
  -/
  rw [setIntegral_pos_iff_support_of_nonneg_ae]
    /-
      s : Real
      hs : LT.lt 0 s
      this : Eq (Inter.inter (Function.support fun x => HMul.hMul (Real.exp (Neg.neg …
      ⊢ LT.lt 0 (MeasureTheory.MeasureSpace.volume (Inter.inter (Function.support fu …
    -/
  · rw [this, volume_Ioi, ← ENNReal.ofReal_zero]
    /-
      s : Real
      hs : LT.lt 0 s
      this : Eq (Inter.inter (Function.support fun x => HMul.hMul (Real.exp (Neg.neg …
      ⊢ LT.lt (ENNReal.ofReal 0) Top.top
    -/
    exact ENNReal.ofReal_lt_top
    /-
      🎉 no goals
    -/
    /-
      case hf
      s : Real
      hs : LT.lt 0 s
      this : Eq (Inter.inter (Function.support fun x => HMul.hMul (Real.exp (Neg.neg …
      ⊢ (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioi 0))). …
    -/
  · refine eventually_of_mem (self_mem_ae_restrict measurableSet_Ioi) ?_
    /-
      case hf
      s : Real
      hs : LT.lt 0 s
      this : Eq (Inter.inter (Function.support fun x => HMul.hMul (Real.exp (Neg.neg …
      ⊢ ∀ (x : Real), Membership.mem (Set.Ioi 0) x → LE.le (0 x) ((fun x => HMul.hMu …
    -/
    exact fun x hx => (mul_pos (exp_pos _) (rpow_pos_of_pos hx _)).le
    /-
      🎉 no goals
    -/
    /-
      case hfi
      s : Real
      hs : LT.lt 0 s
      this : Eq (Inter.inter (Function.support fun x => HMul.hMul (Real.exp (Neg.neg …
      ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (Real.exp (Neg.neg x)) (HPow. …
    -/
  · exact GammaIntegral_convergent hs
    /-
      🎉 no goals
    -/


theorem Gamma_nonneg_of_nonneg {s : ℝ} (hs : 0 ≤ s) : 0 ≤ Gamma s := by
  /-
    s : Real
    hs : LE.le 0 s
    ⊢ LE.le 0 (Real.Gamma s)
  -/
  obtain rfl | h := eq_or_lt_of_le hs
    /-
      case inl
      hs : LE.le 0 0
      ⊢ LE.le 0 (Real.Gamma 0)
    -/
  · rw [Gamma_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Real
      hs : LE.le 0 s
      h : LT.lt 0 s
      ⊢ LE.le 0 (Real.Gamma s)
    -/
  · exact (Gamma_pos_of_pos h).le
    /-
      🎉 no goals
    -/


open Complex in
/-- Expresses the integral over `Ioi 0` of `t ^ (a - 1) * exp (-(r * t))`, for positive real `r`,
in terms of the Gamma function. -/
lemma integral_rpow_mul_exp_neg_mul_Ioi {a r : ℝ} (ha : 0 < a) (hr : 0 < r) :
    ∫ t : ℝ in Ioi 0, t ^ (a - 1) * exp (-(r * t)) = (1 / r) ^ a * Gamma a := by
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rw [← ofReal_inj, ofReal_mul, ← Gamma_ofReal, ofReal_cpow (by positivity), ofReal_div]
  /-
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    ⊢ Eq (↑(MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Se …
  -/
  convert integral_cpow_mul_exp_neg_mul_Ioi (by rwa [ofReal_re] : 0 < (a : ℂ).re) hr
  /-
    case h.e'_2
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    ⊢ Eq (↑(MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Se …
  -/
  refine integral_ofReal.symm.trans <| setIntegral_congr_fun measurableSet_Ioi (fun t ht ↦ ?_)
  /-
    case h.e'_2
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    t : Real
    ht : Membership.mem (Set.Ioi 0) t
    ⊢ Eq (↑(HMul.hMul (HPow.hPow t (HSub.hSub a 1)) (Real.exp (Neg.neg (HMul.hMul  …
  -/
  norm_cast
  /-
    case h.e'_2
    a r : Real
    ha : LT.lt 0 a
    hr : LT.lt 0 r
    t : Real
    ht : Membership.mem (Set.Ioi 0) t
    ⊢ Eq (↑(HMul.hMul (HPow.hPow t (HSub.hSub a 1)) (Real.exp (Neg.neg (HMul.hMul  …
  -/
  simp_rw [← ofReal_cpow ht.le, RCLike.ofReal_mul, coe_algebraMap]
  /-
    🎉 no goals
  -/


open Lean.Meta Qq Mathlib.Meta.Positivity in
/-- The `positivity` extension which identifies expressions of the form `Gamma a`. -/
@[positivity Gamma (_ : ℝ)]
def _root_.Mathlib.Meta.Positivity.evalGamma : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(Gamma $a) =>
    match ← core q(inferInstance) q(inferInstance) a with
    | .positive pa =>
      assertInstancesCommute
      pure (.positive q(Gamma_pos_of_pos $pa))
    | .nonnegative pa =>
      assertInstancesCommute
      pure (.nonnegative q(Gamma_nonneg_of_nonneg $pa))
    | _ => pure .none
  | _, _, _ => throwError "failed to match on Gamma application"


/-- The Gamma function does not vanish on `ℝ` (except at non-positive integers, where the function
is mathematically undefined and we set it to `0` by convention). -/
theorem Gamma_ne_zero {s : ℝ} (hs : ∀ m : ℕ, s ≠ -m) : Gamma s ≠ 0 := by
  suffices ∀ {n : ℕ}, -(n : ℝ) < s → Gamma s ≠ 0 by
    apply this
    swap
    · exact ⌊-s⌋₊ + 1
    rw [neg_lt, Nat.cast_add, Nat.cast_one]
    exact Nat.lt_floor_add_one _
  /-
    s : Real
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ ∀ {n : Nat}, LT.lt (Neg.neg ↑n) s → Ne (Real.Gamma s) 0
  -/
  intro n
  induction n generalizing s with
  | zero =>
    intro hs
    refine (Gamma_pos_of_pos ?_).ne'
    rwa [Nat.cast_zero, neg_zero] at hs
  | succ _ n_ih =>
    intro hs'
    have : Gamma (s + 1) ≠ 0 := by
      apply n_ih
      · intro m
        specialize hs (1 + m)
        contrapose! hs
        rw [← eq_sub_iff_add_eq] at hs
        rw [hs]
        push_cast
        ring
      · rw [Nat.cast_add, Nat.cast_one, neg_add] at hs'
        linarith
    rw [Gamma_add_one, mul_ne_zero_iff] at this
    · exact this.2
    · simpa using hs 0


theorem Gamma_eq_zero_iff (s : ℝ) : Gamma s = 0 ↔ ∃ m : ℕ, s = -m :=
      /-
        s : Real
        ⊢ Eq (Real.Gamma s) 0 → Exists fun m => Eq s (Neg.neg ↑m)
      -/
                   /-
                     🎉 no goals
                   -/
  ⟨by contrapose!; exact Gamma_ne_zero, by rintro ⟨m, rfl⟩; exact Gamma_neg_nat_eq_zero m⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


