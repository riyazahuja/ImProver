/-- The Beta function `Β (u, v)`, defined as `∫ x:ℝ in 0..1, x ^ (u - 1) * (1 - x) ^ (v - 1)`. -/
noncomputable def betaIntegral (u v : ℂ) : ℂ :=
  ∫ x : ℝ in (0)..1, (x : ℂ) ^ (u - 1) * (1 - (x : ℂ)) ^ (v - 1)


/-- Auxiliary lemma for `betaIntegral_convergent`, showing convergence at the left endpoint. -/
theorem betaIntegral_convergent_left {u : ℂ} (hu : 0 < re u) (v : ℂ) :
    IntervalIntegrable (fun x =>
      (x : ℂ) ^ (u - 1) * (1 - (x : ℂ)) ^ (v - 1) : ℝ → ℂ) volume 0 (1 / 2) := by
  /-
    u : Complex
    hu : LT.lt 0 u.re
    v : Complex
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub u 1)) (HPo …
  -/
  apply IntervalIntegrable.mul_continuousOn
    /-
      case hf
      u : Complex
      hu : LT.lt 0 u.re
      v : Complex
      ⊢ IntervalIntegrable (fun x => HPow.hPow (↑x) (HSub.hSub u 1)) MeasureTheory.M …
    -/
  · refine intervalIntegral.intervalIntegrable_cpow' ?_
    /-
      case hf
      u : Complex
      hu : LT.lt 0 u.re
      v : Complex
      ⊢ LT.lt (-1) (HSub.hSub u 1).re
    -/
    rwa [sub_re, one_re, ← zero_sub, sub_lt_sub_iff_right]
    /-
      🎉 no goals
    -/
    /-
      case hg
      u : Complex
      hu : LT.lt 0 u.re
      v : Complex
      ⊢ ContinuousOn (fun x => HPow.hPow (HSub.hSub 1 ↑x) (HSub.hSub v 1)) (Set.uIcc …
    -/
  · apply continuousOn_of_forall_continuousAt
    /-
      case hg.hcont
      u : Complex
      hu : LT.lt 0 u.re
      v : Complex
      ⊢ ∀ (x : Real), Membership.mem (Set.uIcc 0 (1 / 2)) x → ContinuousAt (fun x => …
    -/
    intro x hx
    /-
      case hg.hcont
      u : Complex
      hu : LT.lt 0 u.re
      v : Complex
      x : Real
      hx : Membership.mem (Set.uIcc 0 (1 / 2)) x
      ⊢ ContinuousAt (fun x => HPow.hPow (HSub.hSub 1 ↑x) (HSub.hSub v 1)) x
    -/
    rw [uIcc_of_le (by positivity : (0 : ℝ) ≤ 1 / 2)] at hx
    /-
      case hg.hcont
      u : Complex
      hu : LT.lt 0 u.re
      v : Complex
      x : Real
      hx : Membership.mem (Set.Icc 0 (1 / 2)) x
      ⊢ ContinuousAt (fun x => HPow.hPow (HSub.hSub 1 ↑x) (HSub.hSub v 1)) x
    -/
    apply ContinuousAt.cpow
      /-
        case hg.hcont.hf
        u : Complex
        hu : LT.lt 0 u.re
        v : Complex
        x : Real
        hx : Membership.mem (Set.Icc 0 (1 / 2)) x
        ⊢ ContinuousAt (fun x => HSub.hSub 1 ↑x) x
      -/
    · exact (continuous_const.sub continuous_ofReal).continuousAt
      /-
        🎉 no goals
      -/
      /-
        case hg.hcont.hg
        u : Complex
        hu : LT.lt 0 u.re
        v : Complex
        x : Real
        hx : Membership.mem (Set.Icc 0 (1 / 2)) x
        ⊢ ContinuousAt (fun x => HSub.hSub v 1) x
      -/
    · exact continuousAt_const
      /-
        🎉 no goals
      -/
      /-
        case hg.hcont.h0
        u : Complex
        hu : LT.lt 0 u.re
        v : Complex
        x : Real
        hx : Membership.mem (Set.Icc 0 (1 / 2)) x
        ⊢ Membership.mem Complex.slitPlane (HSub.hSub 1 ↑x)
      -/
    · norm_cast
      /-
        case hg.hcont.h0
        u : Complex
        hu : LT.lt 0 u.re
        v : Complex
        x : Real
        hx : Membership.mem (Set.Icc 0 (1 / 2)) x
        ⊢ Membership.mem Complex.slitPlane ↑(HSub.hSub 1 x)
      -/
      exact ofReal_mem_slitPlane.2 <| by linarith only [hx.2]
      /-
        🎉 no goals
      -/


/-- The Beta integral is convergent for all `u, v` of positive real part. -/
theorem betaIntegral_convergent {u v : ℂ} (hu : 0 < re u) (hv : 0 < re v) :
    IntervalIntegrable (fun x =>
      (x : ℂ) ^ (u - 1) * (1 - (x : ℂ)) ^ (v - 1) : ℝ → ℂ) volume 0 1 := by
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub u 1)) (HPo …
  -/
  refine (betaIntegral_convergent_left hu v).trans ?_
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub u 1)) (HPo …
  -/
  rw [IntervalIntegrable.iff_comp_neg]
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    ⊢ IntervalIntegrable (fun x => HMul.hMul (HPow.hPow (↑(Neg.neg x)) (HSub.hSub  …
  -/
  convert ((betaIntegral_convergent_left hv u).comp_add_right 1).symm using 1
    /-
      case h.e'_3
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      ⊢ Eq (fun x => HMul.hMul (HPow.hPow (↑(Neg.neg x)) (HSub.hSub u 1)) (HPow.hPow …
    -/
  · ext1 x
    /-
      case h.e'_3.h
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      x : Real
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Neg.neg x)) (HSub.hSub u 1)) (HPow.hPow (HSub.hS …
    -/
    conv_lhs => rw [mul_comm]
    /-
      case h.e'_3.h
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      x : Real
      ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub 1 ↑(Neg.neg x)) (HSub.hSub v 1)) (HPow.h …
    -/
                             /-
                               🎉 no goals
                             -/
    congr 2 <;> · push_cast; ring
                             /-
                               🎉 no goals
                             -/
    /-
      case h.e'_5
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      ⊢ Eq (Neg.neg (1 / 2)) (HSub.hSub (1 / 2) 1)
    -/
  · norm_num
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      ⊢ Eq (-1) (HSub.hSub 0 1)
    -/
  · norm_num
    /-
      🎉 no goals
    -/


theorem betaIntegral_symm (u v : ℂ) : betaIntegral v u = betaIntegral u v := by
  /-
    u v : Complex
    ⊢ Eq (v.betaIntegral u) (u.betaIntegral v)
  -/
  rw [betaIntegral, betaIntegral]
  have := intervalIntegral.integral_comp_mul_add (a := 0) (b := 1) (c := -1)
    (fun x : ℝ => (x : ℂ) ^ (u - 1) * (1 - (x : ℂ)) ^ (v - 1)) neg_one_lt_zero.ne 1
  /-
    u v : Complex
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑(HAdd.hAdd (HMul. …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub v 1)) (H …
  -/
  rw [inv_neg, inv_one, neg_one_smul, ← intervalIntegral.integral_symm] at this
  simp? at this says
    simp only [neg_mul, one_mul, ofReal_add, ofReal_neg, ofReal_one, sub_add_cancel_right, neg_neg,
      mul_one, neg_add_cancel, mul_zero, zero_add] at this
  /-
    u v : Complex
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (HAdd.hAdd (Neg.neg …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub v 1)) (H …
  -/
  conv_lhs at this => arg 1; intro x; rw [add_comm, ← sub_eq_add_neg, mul_comm]
  /-
    u v : Complex
    this : Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub v 1 …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub v 1)) (H …
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem betaIntegral_eval_one_right {u : ℂ} (hu : 0 < re u) : betaIntegral u 1 = 1 / u := by
  /-
    u : Complex
    hu : LT.lt 0 u.re
    ⊢ Eq (u.betaIntegral 1) (HDiv.hDiv 1 u)
  -/
  simp_rw [betaIntegral, sub_self, cpow_zero, mul_one]
  /-
    u : Complex
    hu : LT.lt 0 u.re
    ⊢ Eq (intervalIntegral (fun x => HPow.hPow (↑x) (HSub.hSub u 1)) 0 1 MeasureTh …
  -/
  rw [integral_cpow (Or.inl _)]
    /-
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HPow.hPow (↑1) (HAdd.hAdd (HSub.hSub u 1) 1)) (HPo …
    -/
  · rw [ofReal_zero, ofReal_one, one_cpow, zero_cpow, sub_zero, sub_add_cancel]
    /-
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ Ne (HAdd.hAdd (HSub.hSub u 1) 1) 0
    -/
    rw [sub_add_cancel]
    /-
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ Ne u 0
    -/
    contrapose! hu; rw [hu, zero_re]
                    /-
                      🎉 no goals
                    -/
    /-
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ LT.lt (-1) (HSub.hSub u 1).re
    -/
  · rwa [sub_re, one_re, ← sub_pos, sub_neg_eq_add, sub_add_cancel]
    /-
      🎉 no goals
    -/


theorem betaIntegral_scaled (s t : ℂ) {a : ℝ} (ha : 0 < a) :
    ∫ x in (0)..a, (x : ℂ) ^ (s - 1) * ((a : ℂ) - x) ^ (t - 1) =
    (a : ℂ) ^ (s + t - 1) * betaIntegral s t := by
  /-
    s t : Complex
    a : Real
    ha : LT.lt 0 a
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) (H …
  -/
  have ha' : (a : ℂ) ≠ 0 := ofReal_ne_zero.mpr ha.ne'
  /-
    s t : Complex
    a : Real
    ha : LT.lt 0 a
    ha' : Ne (↑a) 0
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) (H …
  -/
  rw [betaIntegral]
  have A : (a : ℂ) ^ (s + t - 1) = a * ((a : ℂ) ^ (s - 1) * (a : ℂ) ^ (t - 1)) := by
    rw [(by abel : s + t - 1 = 1 + (s - 1) + (t - 1)), cpow_add _ _ ha', cpow_add 1 _ ha', cpow_one,
      mul_assoc]
  rw [A, mul_assoc, ← intervalIntegral.integral_const_mul, ← real_smul, ← zero_div a, ←
    div_self ha.ne', ← intervalIntegral.integral_comp_div _ ha.ne', zero_div]
  /-
    s t : Complex
    a : Real
    ha : LT.lt 0 a
    ha' : Ne (↑a) 0
    A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) (H …
  -/
  simp_rw [intervalIntegral.integral_of_le ha.le]
  /-
    s t : Complex
    a : Real
    ha : LT.lt 0 a
    ha' : Ne (↑a) 0
    A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioc fun x hx => ?_
  /-
    s t : Complex
    a : Real
    ha : LT.lt 0 a
    ha' : Ne (↑a) 0
    A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
    x : Real
    hx : Membership.mem (Set.Ioc 0 a) x
    ⊢ Eq (HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) (HPow.hPow (HSub.hSub ↑a ↑x)  …
  -/
  rw [mul_mul_mul_comm]
  /-
    s t : Complex
    a : Real
    ha : LT.lt 0 a
    ha' : Ne (↑a) 0
    A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
    x : Real
    hx : Membership.mem (Set.Ioc 0 a) x
    ⊢ Eq (HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) (HPow.hPow (HSub.hSub ↑a ↑x)  …
  -/
  congr 1
    /-
      case e_a
      s t : Complex
      a : Real
      ha : LT.lt 0 a
      ha' : Ne (↑a) 0
      A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 a) x
      ⊢ Eq (HPow.hPow (↑x) (HSub.hSub s 1)) (HMul.hMul (HPow.hPow (↑a) (HSub.hSub s  …
    -/
  · rw [← mul_cpow_ofReal_nonneg ha.le (div_pos hx.1 ha).le, ofReal_div, mul_div_cancel₀ _ ha']
    /-
      🎉 no goals
    -/
  · rw [(by norm_cast : (1 : ℂ) - ↑(x / a) = ↑(1 - x / a)), ←
      mul_cpow_ofReal_nonneg ha.le (sub_nonneg.mpr <| (div_le_one ha).mpr hx.2)]
    /-
      case e_a
      s t : Complex
      a : Real
      ha : LT.lt 0 a
      ha' : Ne (↑a) 0
      A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 a) x
      ⊢ Eq (HPow.hPow (HSub.hSub ↑a ↑x) (HSub.hSub t 1)) (HPow.hPow (HMul.hMul ↑a ↑( …
    -/
    push_cast
    /-
      case e_a
      s t : Complex
      a : Real
      ha : LT.lt 0 a
      ha' : Ne (↑a) 0
      A : Eq (HPow.hPow (↑a) (HSub.hSub (HAdd.hAdd s t) 1)) (HMul.hMul (↑a) (HMul.hM …
      x : Real
      hx : Membership.mem (Set.Ioc 0 a) x
      ⊢ Eq (HPow.hPow (HSub.hSub ↑a ↑x) (HSub.hSub t 1)) (HPow.hPow (HMul.hMul (↑a)  …
    -/
    rw [mul_sub, mul_one, mul_div_cancel₀ _ ha']
    /-
      🎉 no goals
    -/


/-- Relation between Beta integral and Gamma function. -/
theorem Gamma_mul_Gamma_eq_betaIntegral {s t : ℂ} (hs : 0 < re s) (ht : 0 < re t) :
    Gamma s * Gamma t = Gamma (s + t) * betaIntegral s t := by
  -- Note that we haven't proved (yet) that the Gamma function has no zeroes, so we can't formulate
  -- this as a formula for the Beta function.
  have conv_int := integral_posConvolution
    (GammaIntegral_convergent hs) (GammaIntegral_convergent ht) (ContinuousLinearMap.mul ℝ ℂ)
  /-
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    ⊢ Eq (HMul.hMul (Complex.Gamma s) (Complex.Gamma t)) (HMul.hMul (Complex.Gamma …
  -/
  simp_rw [ContinuousLinearMap.mul_apply'] at conv_int
  /-
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    ⊢ Eq (HMul.hMul (Complex.Gamma s) (Complex.Gamma t)) (HMul.hMul (Complex.Gamma …
  -/
  have hst : 0 < re (s + t) := by rw [add_re]; exact add_pos hs ht
  rw [Gamma_eq_integral hs, Gamma_eq_integral ht, Gamma_eq_integral hst, GammaIntegral,
    GammaIntegral, GammaIntegral, ← conv_int, ← integral_mul_right (betaIntegral _ _)]
  /-
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    hst : LT.lt 0 (HAdd.hAdd s t).re
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioi fun x hx => ?_
  /-
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    hst : LT.lt 0 (HAdd.hAdd s t).re
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (intervalIntegral (fun t_1 => HMul.hMul (HMul.hMul (↑(Real.exp (Neg.neg t …
  -/
  rw [mul_assoc, ← betaIntegral_scaled s t hx, ← intervalIntegral.integral_const_mul]
  /-
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    hst : LT.lt 0 (HAdd.hAdd s t).re
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    ⊢ Eq (intervalIntegral (fun t_1 => HMul.hMul (HMul.hMul (↑(Real.exp (Neg.neg t …
  -/
  congr 1 with y : 1
  /-
    case e_f.h
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    hst : LT.lt 0 (HAdd.hAdd s t).re
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul (↑(Real.exp (Neg.neg y))) (HPow.hPow (↑y) (HSub.hSu …
  -/
  push_cast
  /-
    case e_f.h
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    hst : LT.lt 0 (HAdd.hAdd s t).re
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    ⊢ Eq (HMul.hMul (HMul.hMul (Complex.exp (Neg.neg ↑y)) (HPow.hPow (↑y) (HSub.hS …
  -/
  suffices Complex.exp (-x) = Complex.exp (-y) * Complex.exp (-(x - y)) by rw [this]; ring
  /-
    case e_f.h
    s t : Complex
    hs : LT.lt 0 s.re
    ht : LT.lt 0 t.re
    conv_int : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restr …
    hst : LT.lt 0 (HAdd.hAdd s t).re
    x : Real
    hx : Membership.mem (Set.Ioi 0) x
    y : Real
    ⊢ Eq (Complex.exp (Neg.neg ↑x)) (HMul.hMul (Complex.exp (Neg.neg ↑y)) (Complex …
  -/
                                   /-
                                     🎉 no goals
                                   -/
  rw [← Complex.exp_add]; congr 1; abel
                                   /-
                                     🎉 no goals
                                   -/


/-- Recurrence formula for the Beta function. -/
theorem betaIntegral_recurrence {u v : ℂ} (hu : 0 < re u) (hv : 0 < re v) :
    u * betaIntegral u (v + 1) = v * betaIntegral (u + 1) v := by
  -- NB: If we knew `Gamma (u + v + 1) ≠ 0` this would be an easy consequence of
  -- `Gamma_mul_Gamma_eq_betaIntegral`; but we don't know that yet. We will prove it later, but
  -- this lemma is needed in the proof. So we give a (somewhat laborious) direct argument.
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    ⊢ Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd v 1))) (HMul.hMul v ((HAdd.hAdd u …
  -/
  let F : ℝ → ℂ := fun x => (x : ℂ) ^ u * (1 - (x : ℂ)) ^ v
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
    ⊢ Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd v 1))) (HMul.hMul v ((HAdd.hAdd u …
  -/
  have hu' : 0 < re (u + 1) := by rw [add_re, one_re]; positivity
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
    hu' : LT.lt 0 (HAdd.hAdd u 1).re
    ⊢ Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd v 1))) (HMul.hMul v ((HAdd.hAdd u …
  -/
  have hv' : 0 < re (v + 1) := by rw [add_re, one_re]; positivity
  have hc : ContinuousOn F (Icc 0 1) := by
    refine (continuousOn_of_forall_continuousAt fun x hx => ?_).mul
        (continuousOn_of_forall_continuousAt fun x hx => ?_)
    · refine (continuousAt_cpow_const_of_re_pos (Or.inl ?_) hu).comp continuous_ofReal.continuousAt
      rw [ofReal_re]; exact hx.1
    · refine (continuousAt_cpow_const_of_re_pos (Or.inl ?_) hv).comp
        (continuous_const.sub continuous_ofReal).continuousAt
      rw [sub_re, one_re, ofReal_re, sub_nonneg]
      exact hx.2
  have hder : ∀ x : ℝ, x ∈ Ioo (0 : ℝ) 1 →
      HasDerivAt F (u * ((x : ℂ) ^ (u - 1) * (1 - (x : ℂ)) ^ v) -
        v * ((x : ℂ) ^ u * (1 - (x : ℂ)) ^ (v - 1))) x := by
    intro x hx
    have U : HasDerivAt (fun y : ℂ => y ^ u) (u * (x : ℂ) ^ (u - 1)) ↑x := by
      have := @HasDerivAt.cpow_const _ _ _ u (hasDerivAt_id (x : ℂ)) (Or.inl ?_)
      · simp only [id_eq, mul_one] at this
        exact this
      · rw [id_eq, ofReal_re]; exact hx.1
    have V : HasDerivAt (fun y : ℂ => (1 - y) ^ v) (-v * (1 - (x : ℂ)) ^ (v - 1)) ↑x := by
      have A := @HasDerivAt.cpow_const _ _ _ v (hasDerivAt_id (1 - (x : ℂ))) (Or.inl ?_)
      swap; · rw [id, sub_re, one_re, ofReal_re, sub_pos]; exact hx.2
      simp_rw [id] at A
      have B : HasDerivAt (fun y : ℂ => 1 - y) (-1) ↑x := by
        apply HasDerivAt.const_sub; apply hasDerivAt_id
      convert HasDerivAt.comp (↑x) A B using 1
      ring
    convert (U.mul V).comp_ofReal using 1
    ring
  have h_int := ((betaIntegral_convergent hu hv').const_mul u).sub
    ((betaIntegral_convergent hu' hv).const_mul v)
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
    hu' : LT.lt 0 (HAdd.hAdd u 1).re
    hv' : LT.lt 0 (HAdd.hAdd v 1).re
    hc : ContinuousOn F (Set.Icc 0 1)
    hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
    h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
    ⊢ Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd v 1))) (HMul.hMul v ((HAdd.hAdd u …
  -/
  rw [add_sub_cancel_right, add_sub_cancel_right] at h_int
  /-
    u v : Complex
    hu : LT.lt 0 u.re
    hv : LT.lt 0 v.re
    F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
    hu' : LT.lt 0 (HAdd.hAdd u 1).re
    hv' : LT.lt 0 (HAdd.hAdd v 1).re
    hc : ContinuousOn F (Set.Icc 0 1)
    hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
    h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
    ⊢ Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd v 1))) (HMul.hMul v ((HAdd.hAdd u …
  -/
  have int_ev := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le zero_le_one hc hder h_int
  have hF0 : F 0 = 0 := by
    simp only [F, mul_eq_zero, ofReal_zero, cpow_eq_zero_iff, eq_self_iff_true, Ne,
      true_and, sub_zero, one_cpow, one_ne_zero, or_false]
    contrapose! hu; rw [hu, zero_re]
  have hF1 : F 1 = 0 := by
    simp only [F, mul_eq_zero, ofReal_one, one_cpow, one_ne_zero, sub_self, cpow_eq_zero_iff,
      eq_self_iff_true, Ne, true_and, false_or]
    contrapose! hv; rw [hv, zero_re]
  rw [hF0, hF1, sub_zero, intervalIntegral.integral_sub, intervalIntegral.integral_const_mul,
    intervalIntegral.integral_const_mul] at int_ev
    /-
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
      hu' : LT.lt 0 (HAdd.hAdd u 1).re
      hv' : LT.lt 0 (HAdd.hAdd v 1).re
      hc : ContinuousOn F (Set.Icc 0 1)
      hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
      h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
      int_ev : Eq (HSub.hSub (HMul.hMul u (intervalIntegral (fun x => HMul.hMul (HPo …
      hF0 : Eq (F 0) 0
      hF1 : Eq (F 1) 0
      ⊢ Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd v 1))) (HMul.hMul v ((HAdd.hAdd u …
    -/
  · rw [betaIntegral, betaIntegral, ← sub_eq_zero]
    /-
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
      hu' : LT.lt 0 (HAdd.hAdd u 1).re
      hv' : LT.lt 0 (HAdd.hAdd v 1).re
      hc : ContinuousOn F (Set.Icc 0 1)
      hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
      h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
      int_ev : Eq (HSub.hSub (HMul.hMul u (intervalIntegral (fun x => HMul.hMul (HPo …
      hF0 : Eq (F 0) 0
      hF1 : Eq (F 1) 0
      ⊢ Eq (HSub.hSub (HMul.hMul u (intervalIntegral (fun x => HMul.hMul (HPow.hPow  …
    -/
                       /-
                         🎉 no goals
                       -/
    convert int_ev <;> ring
                       /-
                         🎉 no goals
                       -/
    /-
      case hf
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
      hu' : LT.lt 0 (HAdd.hAdd u 1).re
      hv' : LT.lt 0 (HAdd.hAdd v 1).re
      hc : ContinuousOn F (Set.Icc 0 1)
      hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
      h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
      int_ev : Eq (intervalIntegral (fun y => HSub.hSub (HMul.hMul u (HMul.hMul (HPo …
      hF0 : Eq (F 0) 0
      hF1 : Eq (F 1) 0
      ⊢ IntervalIntegrable (fun y => HMul.hMul u (HMul.hMul (HPow.hPow (↑y) (HSub.hS …
    -/
  · apply IntervalIntegrable.const_mul
    /-
      case hf.hf
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
      hu' : LT.lt 0 (HAdd.hAdd u 1).re
      hv' : LT.lt 0 (HAdd.hAdd v 1).re
      hc : ContinuousOn F (Set.Icc 0 1)
      hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
      h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
      int_ev : Eq (intervalIntegral (fun y => HSub.hSub (HMul.hMul u (HMul.hMul (HPo …
      hF0 : Eq (F 0) 0
      hF1 : Eq (F 1) 0
      ⊢ IntervalIntegrable (fun x => HMul.hMul (HPow.hPow (↑x) (HSub.hSub u 1)) (HPo …
    -/
    convert betaIntegral_convergent hu hv'; ring
                                            /-
                                              🎉 no goals
                                            -/
    /-
      case hg
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
      hu' : LT.lt 0 (HAdd.hAdd u 1).re
      hv' : LT.lt 0 (HAdd.hAdd v 1).re
      hc : ContinuousOn F (Set.Icc 0 1)
      hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
      h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
      int_ev : Eq (intervalIntegral (fun y => HSub.hSub (HMul.hMul u (HMul.hMul (HPo …
      hF0 : Eq (F 0) 0
      hF1 : Eq (F 1) 0
      ⊢ IntervalIntegrable (fun y => HMul.hMul v (HMul.hMul (HPow.hPow (↑y) u) (HPow …
    -/
  · apply IntervalIntegrable.const_mul
    /-
      case hg.hf
      u v : Complex
      hu : LT.lt 0 u.re
      hv : LT.lt 0 v.re
      F : Real → Complex := fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
      hu' : LT.lt 0 (HAdd.hAdd u 1).re
      hv' : LT.lt 0 (HAdd.hAdd v 1).re
      hc : ContinuousOn F (Set.Icc 0 1)
      hder : ∀ (x : Real), Membership.mem (Set.Ioo 0 1) x → HasDerivAt F (HSub.hSub  …
      h_int : IntervalIntegrable (fun x => HSub.hSub (HMul.hMul u (HMul.hMul (HPow.h …
      int_ev : Eq (intervalIntegral (fun y => HSub.hSub (HMul.hMul u (HMul.hMul (HPo …
      hF0 : Eq (F 0) 0
      hF1 : Eq (F 1) 0
      ⊢ IntervalIntegrable (fun x => HMul.hMul (HPow.hPow (↑x) u) (HPow.hPow (HSub.h …
    -/
    convert betaIntegral_convergent hu' hv; ring
                                            /-
                                              🎉 no goals
                                            -/


/-- Explicit formula for the Beta function when second argument is a positive integer. -/
theorem betaIntegral_eval_nat_add_one_right {u : ℂ} (hu : 0 < re u) (n : ℕ) :
    betaIntegral u (n + 1) = n ! / ∏ j ∈ Finset.range (n + 1), (u + j) := by
  /-
    u : Complex
    hu : LT.lt 0 u.re
    n : Nat
    ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (HDiv.hDiv (↑n.factorial) ((Finset.ra …
  -/
  induction' n with n IH generalizing u
    /-
      case zero
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑0) 1)) (HDiv.hDiv (↑(Nat.factorial 0)) ((Fin …
    -/
  · rw [Nat.cast_zero, zero_add, betaIntegral_eval_one_right hu, Nat.factorial_zero, Nat.cast_one]
    /-
      case zero
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ Eq (HDiv.hDiv 1 u) (HDiv.hDiv 1 ((Finset.range (HAdd.hAdd 0 1)).prod fun j = …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑(HAdd.hAdd n 1)) 1)) (HDiv.hDiv (↑(HAdd.hAdd …
    -/
  · have := betaIntegral_recurrence hu (?_ : 0 < re n.succ)
    /-
      case succ.refine_2
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd (↑n.succ) 1))) (HMul.hMul (↑ …
      ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑(HAdd.hAdd n 1)) 1)) (HDiv.hDiv (↑(HAdd.hAdd …
    -/
    swap; · rw [← ofReal_natCast, ofReal_re]; positivity
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case succ.refine_2
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (HMul.hMul u (u.betaIntegral (HAdd.hAdd (↑n.succ) 1))) (HMul.hMul (↑ …
      ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑(HAdd.hAdd n 1)) 1)) (HDiv.hDiv (↑(HAdd.hAdd …
    -/
    rw [mul_comm u _, ← eq_div_iff] at this
    /-
      case succ.refine_2
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (u.betaIntegral (HAdd.hAdd (↑n.succ) 1)) (HDiv.hDiv (HMul.hMul (↑n.s …
      ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑(HAdd.hAdd n 1)) 1)) (HDiv.hDiv (↑(HAdd.hAdd …
    -/
    swap; · contrapose! hu; rw [hu, zero_re]
                            /-
                              🎉 no goals
                            -/
    /-
      case succ.refine_2
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (u.betaIntegral (HAdd.hAdd (↑n.succ) 1)) (HDiv.hDiv (HMul.hMul (↑n.s …
      ⊢ Eq (u.betaIntegral (HAdd.hAdd (↑(HAdd.hAdd n 1)) 1)) (HDiv.hDiv (↑(HAdd.hAdd …
    -/
    rw [this, Finset.prod_range_succ', Nat.cast_succ, IH]
    /-
      case succ.refine_2
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (u.betaIntegral (HAdd.hAdd (↑n.succ) 1)) (HDiv.hDiv (HMul.hMul (↑n.s …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑n) 1) (HDiv.hDiv (↑n.factorial) ((Fins …
    -/
    swap; · rw [add_re, one_re]; positivity
                                 /-
                                   🎉 no goals
                                 -/
    rw [Nat.factorial_succ, Nat.cast_mul, Nat.cast_add, Nat.cast_one, Nat.cast_zero, add_zero, ←
      mul_div_assoc, ← div_div]
    /-
      case succ.refine_2
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (u.betaIntegral (HAdd.hAdd (↑n.succ) 1)) (HDiv.hDiv (HMul.hMul (↑n.s …
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HMul.hMul (HAdd.hAdd (↑n) 1) ↑n.factorial) ((Finse …
    -/
    congr 3 with j : 1
    /-
      case succ.refine_2.e_a.e_a.e_f.h
      n : Nat
      IH : ∀ {u : Complex}, LT.lt 0 u.re → Eq (u.betaIntegral (HAdd.hAdd (↑n) 1)) (H …
      u : Complex
      hu : LT.lt 0 u.re
      this : Eq (u.betaIntegral (HAdd.hAdd (↑n.succ) 1)) (HDiv.hDiv (HMul.hMul (↑n.s …
      j : Nat
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd u 1) ↑j) (HAdd.hAdd u ↑(HAdd.hAdd j 1))
    -/
               /-
                 🎉 no goals
               -/
    push_cast; abel
               /-
                 🎉 no goals
               -/


/-- The sequence with `n`-th term `n ^ s * n! / (s * (s + 1) * ... * (s + n))`, for complex `s`.
We will show that this tends to `Γ(s)` as `n → ∞`. -/
noncomputable def GammaSeq (s : ℂ) (n : ℕ) :=
  (n : ℂ) ^ s * n ! / ∏ j ∈ Finset.range (n + 1), (s + j)


theorem GammaSeq_eq_betaIntegral_of_re_pos {s : ℂ} (hs : 0 < re s) (n : ℕ) :
    GammaSeq s n = (n : ℂ) ^ s * betaIntegral s (n + 1) := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    ⊢ Eq (s.GammaSeq n) (HMul.hMul (HPow.hPow (↑n) s) (s.betaIntegral (HAdd.hAdd ( …
  -/
  rw [GammaSeq, betaIntegral_eval_nat_add_one_right hs n, ← mul_div_assoc]
  /-
    🎉 no goals
  -/


theorem GammaSeq_add_one_left (s : ℂ) {n : ℕ} (hn : n ≠ 0) :
    GammaSeq (s + 1) n / s = n / (n + 1 + s) * GammaSeq s n := by
  /-
    s : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HDiv.hDiv ((HAdd.hAdd s 1).GammaSeq n) s) (HMul.hMul (HDiv.hDiv (↑n) (HA …
  -/
  conv_lhs => rw [GammaSeq, Finset.prod_range_succ, div_div]
  conv_rhs =>
    rw [GammaSeq, Finset.prod_range_succ', Nat.cast_zero, add_zero, div_mul_div_comm, ← mul_assoc,
      ← mul_assoc, mul_comm _ (Finset.prod _ _)]
  /-
    s : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul (HPow.hPow (↑n) (HAdd.hAdd s 1)) ↑n.factorial) (HMu …
  -/
  congr 3
    /-
      case e_a.e_a
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (HPow.hPow (↑n) (HAdd.hAdd s 1)) (HMul.hMul (↑n) (HPow.hPow (↑n) s))
    -/
  · rw [cpow_add _ _ (Nat.cast_ne_zero.mpr hn), cpow_one, mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_a.e_a
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq ((Finset.range n).prod fun x => HAdd.hAdd (HAdd.hAdd s 1) ↑x) ((Finset.ra …
    -/
  · refine Finset.prod_congr (by rfl) fun x _ => ?_
    /-
      case e_a.e_a.e_a
      s : Complex
      n : Nat
      hn : Ne n 0
      x : Nat
      x✝ : Membership.mem (Finset.range n) x
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd s 1) ↑x) (HAdd.hAdd s ↑(HAdd.hAdd x 1))
    -/
    push_cast; ring
               /-
                 🎉 no goals
               -/
    /-
      case e_a.e_a.e_a
      s : Complex
      n : Nat
      hn : Ne n 0
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd s 1) ↑n) (HAdd.hAdd (HAdd.hAdd (↑n) 1) s)
    -/
    /-
      🎉 no goals
    -/
  · abel
    /-
      🎉 no goals
    -/


theorem GammaSeq_eq_approx_Gamma_integral {s : ℂ} (hs : 0 < re s) {n : ℕ} (hn : n ≠ 0) :
    GammaSeq s n = ∫ x : ℝ in (0)..n, ↑((1 - x / n) ^ n) * (x : ℂ) ^ (s - 1) := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    ⊢ Eq (s.GammaSeq n) (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.h …
  -/
  have : ∀ x : ℝ, x = x / n * n := by intro x; rw [div_mul_cancel₀]; exact Nat.cast_ne_zero.mpr hn
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    ⊢ Eq (s.GammaSeq n) (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.h …
  -/
  conv_rhs => enter [1, x, 2, 1]; rw [this x]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    ⊢ Eq (s.GammaSeq n) (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.h …
  -/
  rw [GammaSeq_eq_betaIntegral_of_re_pos hs]
  have := intervalIntegral.integral_comp_div (a := 0) (b := n)
    (fun x => ↑((1 - x) ^ n) * ↑(x * ↑n) ^ (s - 1) : ℝ → ℂ) (Nat.cast_ne_zero.mpr hn)
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => (fun x => HMul.hMul (↑(HPow.hPow (HSub.h …
    ⊢ Eq (HMul.hMul (HPow.hPow (↑n) s) (s.betaIntegral (HAdd.hAdd (↑n) 1))) (inter …
  -/
  dsimp only at this
  rw [betaIntegral, this, real_smul, zero_div, div_self, add_sub_cancel_right,
    ← intervalIntegral.integral_const_mul, ← intervalIntegral.integral_const_mul]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 (HDi …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑n) s) (HMul.hMul (HPow …
  -/
  swap; · exact Nat.cast_ne_zero.mpr hn
          /-
            🎉 no goals
          -/
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 (HDi …
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (HPow.hPow (↑n) s) (HMul.hMul (HPow …
  -/
  simp_rw [intervalIntegral.integral_of_le zero_le_one]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 (HDi …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  refine setIntegral_congr_fun measurableSet_Ioc fun x hx => ?_
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 (HDi …
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    ⊢ Eq (HMul.hMul (HPow.hPow (↑n) s) (HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) …
  -/
  push_cast
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 (HDi …
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    ⊢ Eq (HMul.hMul (HPow.hPow (↑n) s) (HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) …
  -/
  have hn' : (n : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  have A : (n : ℂ) ^ s = (n : ℂ) ^ (s - 1) * n := by
    conv_lhs => rw [(by ring : s = s - 1 + 1), cpow_add _ _ hn']
    simp
  have B : ((x : ℂ) * ↑n) ^ (s - 1) = (x : ℂ) ^ (s - 1) * (n : ℂ) ^ (s - 1) := by
    rw [← ofReal_natCast,
      mul_cpow_ofReal_nonneg hx.1.le (Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn)).le]
  /-
    s : Complex
    hs : LT.lt 0 s.re
    n : Nat
    hn : Ne n 0
    this✝ : ∀ (x : Real), Eq x (HMul.hMul (HDiv.hDiv x ↑n) ↑n)
    this : Eq (intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 (HDi …
    x : Real
    hx : Membership.mem (Set.Ioc 0 1) x
    hn' : Ne (↑n) 0
    A : Eq (HPow.hPow (↑n) s) (HMul.hMul (HPow.hPow (↑n) (HSub.hSub s 1)) ↑n)
    B : Eq (HPow.hPow (HMul.hMul ↑x ↑n) (HSub.hSub s 1)) (HMul.hMul (HPow.hPow (↑x …
    ⊢ Eq (HMul.hMul (HPow.hPow (↑n) s) (HMul.hMul (HPow.hPow (↑x) (HSub.hSub s 1)) …
  -/
  rw [A, B, cpow_natCast]; ring
                           /-
                             🎉 no goals
                           -/


/-- The main technical lemma for `GammaSeq_tendsto_Gamma`, expressing the integral defining the
Gamma function for `0 < re s` as the limit of a sequence of integrals over finite intervals. -/
theorem approx_Gamma_integral_tendsto_Gamma_integral {s : ℂ} (hs : 0 < re s) :
    Tendsto (fun n : ℕ => ∫ x : ℝ in (0)..n, ((1 - x / n) ^ n : ℝ) * (x : ℂ) ^ (s - 1)) atTop
      (𝓝 <| Gamma s) := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    ⊢ Filter.Tendsto (fun n => intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow ( …
  -/
  rw [Gamma_eq_integral hs]
  -- We apply dominated convergence to the following function, which we will show is uniformly
  -- bounded above by the Gamma integrand `exp (-x) * x ^ (re s - 1)`.
  let f : ℕ → ℝ → ℂ := fun n =>
    indicator (Ioc 0 (n : ℝ)) fun x : ℝ => ((1 - x / n) ^ n : ℝ) * (x : ℂ) ^ (s - 1)
  -- integrability of f
  have f_ible : ∀ n : ℕ, Integrable (f n) (volume.restrict (Ioi 0)) := by
    intro n
    rw [integrable_indicator_iff (measurableSet_Ioc : MeasurableSet (Ioc (_ : ℝ) _)), IntegrableOn,
      Measure.restrict_restrict_of_subset Ioc_subset_Ioi_self, ← IntegrableOn, ←
      intervalIntegrable_iff_integrableOn_Ioc_of_le (by positivity : (0 : ℝ) ≤ n)]
    apply IntervalIntegrable.continuousOn_mul
    · refine intervalIntegral.intervalIntegrable_cpow' ?_
      rwa [sub_re, one_re, ← zero_sub, sub_lt_sub_iff_right]
    · apply Continuous.continuousOn
      exact RCLike.continuous_ofReal.comp -- Porting note: was `continuity`
        ((continuous_const.sub (continuous_id'.div_const (n : ℝ))).pow n)
  -- pointwise limit of f
  have f_tends : ∀ x : ℝ, x ∈ Ioi (0 : ℝ) →
      Tendsto (fun n : ℕ => f n x) atTop (𝓝 <| ↑(Real.exp (-x)) * (x : ℂ) ^ (s - 1)) := by
    intro x hx
    apply Tendsto.congr'
    · show ∀ᶠ n : ℕ in atTop, ↑((1 - x / n) ^ n) * (x : ℂ) ^ (s - 1) = f n x
      filter_upwards [eventually_ge_atTop ⌈x⌉₊] with n hn
      rw [Nat.ceil_le] at hn
      dsimp only [f]
      rw [indicator_of_mem]
      exact ⟨hx, hn⟩
    · simp_rw [mul_comm]
      refine (Tendsto.comp (continuous_ofReal.tendsto _) ?_).const_mul _
      convert tendsto_one_plus_div_pow_exp (-x) using 1
      ext1 n
      rw [neg_div, ← sub_eq_add_neg]
  -- let `convert` identify the remaining goals
  convert tendsto_integral_of_dominated_convergence _ (fun n => (f_ible n).1)
    (Real.GammaIntegral_convergent hs) _
    ((ae_restrict_iff' measurableSet_Ioi).mpr (ae_of_all _ f_tends)) using 1
  -- limit of f is the integrand we want
    /-
      case h.e'_3
      s : Complex
      hs : LT.lt 0 s.re
      f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
      f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
      f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
      ⊢ Eq (fun n => intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1  …
    -/
  · ext1 n
    rw [MeasureTheory.integral_indicator (measurableSet_Ioc : MeasurableSet (Ioc (_ : ℝ) _)),
      intervalIntegral.integral_of_le (by positivity : 0 ≤ (n : ℝ)),
      Measure.restrict_restrict_of_subset Ioc_subset_Ioi_self]
  -- f is uniformly bounded by the Gamma integrand
    /-
      s : Complex
      hs : LT.lt 0 s.re
      f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
      f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
      f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
      ⊢ ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (f n a)) (HMul.hMu …
    -/
  · intro n
    /-
      s : Complex
      hs : LT.lt 0 s.re
      f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
      f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
      f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
      n : Nat
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f n a)) (HMul.hMul (Real.exp ( …
    -/
    rw [ae_restrict_iff' measurableSet_Ioi]
    /-
      s : Complex
      hs : LT.lt 0 s.re
      f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
      f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
      f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
      n : Nat
      ⊢ Filter.Eventually (fun x => Membership.mem (Set.Ioi 0) x → LE.le (Norm.norm  …
    -/
    filter_upwards with x hx
    /-
      case h
      s : Complex
      hs : LT.lt 0 s.re
      f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
      f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
      f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
      n : Nat
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ LE.le (Norm.norm (f n x)) (HMul.hMul (Real.exp (Neg.neg x)) (HPow.hPow x (HS …
    -/
    dsimp only [f]
    /-
      case h
      s : Complex
      hs : LT.lt 0 s.re
      f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
      f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
      f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
      n : Nat
      x : Real
      hx : Membership.mem (Set.Ioi 0) x
      ⊢ LE.le (Norm.norm ((Set.Ioc 0 ↑n).indicator (fun x => HMul.hMul (↑(HPow.hPow  …
    -/
    rcases lt_or_le (n : ℝ) x with (hxn | hxn)
    · rw [indicator_of_not_mem (not_mem_Ioc_of_gt hxn), norm_zero,
        mul_nonneg_iff_right_nonneg_of_pos (exp_pos _)]
      /-
        case h.inl
        s : Complex
        hs : LT.lt 0 s.re
        f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
        f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
        f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
        n : Nat
        x : Real
        hx : Membership.mem (Set.Ioi 0) x
        hxn : LT.lt (↑n) x
        ⊢ LE.le 0 (HPow.hPow x (HSub.hSub s.re 1))
      -/
      exact rpow_nonneg (le_of_lt hx) _
      /-
        🎉 no goals
      -/
    · rw [indicator_of_mem (mem_Ioc.mpr ⟨mem_Ioi.mp hx, hxn⟩), norm_mul, Complex.norm_eq_abs,
        Complex.abs_of_nonneg
          (pow_nonneg (sub_nonneg.mpr <| div_le_one_of_le₀ hxn <| by positivity) _),
        Complex.norm_eq_abs, abs_cpow_eq_rpow_re_of_pos hx, sub_re, one_re,
        mul_le_mul_right (rpow_pos_of_pos hx _)]
      /-
        case h.inr
        s : Complex
        hs : LT.lt 0 s.re
        f : Nat → Real → Complex := fun n => (Set.Ioc 0 ↑n).indicator fun x => HMul.hM …
        f_ible : ∀ (n : Nat), MeasureTheory.Integrable (f n) (MeasureTheory.MeasureSpa …
        f_tends : ∀ (x : Real), Membership.mem (Set.Ioi 0) x → Filter.Tendsto (fun n = …
        n : Nat
        x : Real
        hx : Membership.mem (Set.Ioi 0) x
        hxn : LE.le x ↑n
        ⊢ LE.le (HPow.hPow (HSub.hSub 1 (HDiv.hDiv x ↑n)) n) (Real.exp (Neg.neg x))
      -/
      exact one_sub_div_pow_le_exp_neg hxn
      /-
        🎉 no goals
      -/


/-- Euler's limit formula for the complex Gamma function. -/
theorem GammaSeq_tendsto_Gamma (s : ℂ) : Tendsto (GammaSeq s) atTop (𝓝 <| Gamma s) := by
  suffices ∀ m : ℕ, -↑m < re s → Tendsto (GammaSeq s) atTop (𝓝 <| GammaAux m s) by
    rw [Gamma]
    apply this
    rw [neg_lt]
    rcases lt_or_le 0 (re s) with (hs | hs)
    · exact (neg_neg_of_pos hs).trans_le (Nat.cast_nonneg _)
    · refine (Nat.lt_floor_add_one _).trans_le ?_
      rw [sub_eq_neg_add, Nat.floor_add_one (neg_nonneg.mpr hs), Nat.cast_add_one]
  /-
    s : Complex
    ⊢ ∀ (m : Nat), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filter.atTo …
  -/
  intro m
  /-
    s : Complex
    m : Nat
    ⊢ LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Comp …
  -/
  induction' m with m IH generalizing s
  · -- Base case: `0 < re s`, so Gamma is given by the integral formula
    /-
      case zero
      s : Complex
      ⊢ LT.lt (Neg.neg ↑0) s.re → Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Comp …
    -/
    intro hs
    /-
      case zero
      s : Complex
      hs : LT.lt (Neg.neg ↑0) s.re
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Complex.GammaAux 0 s))
    -/
    rw [Nat.cast_zero, neg_zero] at hs
    /-
      case zero
      s : Complex
      hs : LT.lt 0 s.re
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Complex.GammaAux 0 s))
    -/
    rw [← Gamma_eq_GammaAux]
      /-
        case zero
        s : Complex
        hs : LT.lt 0 s.re
        ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Complex.Gamma s))
      -/
    · refine Tendsto.congr' ?_ (approx_Gamma_integral_tendsto_Gamma_integral hs)
      /-
        case zero
        s : Complex
        hs : LT.lt 0 s.re
        ⊢ Filter.atTop.EventuallyEq (fun n => intervalIntegral (fun x => HMul.hMul (↑( …
      -/
      refine (eventually_ne_atTop 0).mp (Eventually.of_forall fun n hn => ?_)
      /-
        case zero
        s : Complex
        hs : LT.lt 0 s.re
        n : Nat
        hn : Ne n 0
        ⊢ Eq ((fun n => intervalIntegral (fun x => HMul.hMul (↑(HPow.hPow (HSub.hSub 1 …
      -/
      exact (GammaSeq_eq_approx_Gamma_integral hs hn).symm
      /-
        🎉 no goals
      -/
      /-
        case zero.h1
        s : Complex
        hs : LT.lt 0 s.re
        ⊢ LT.lt (Neg.neg s.re) ↑0
      -/
    · rwa [Nat.cast_zero, neg_lt_zero]
      /-
        🎉 no goals
      -/
  · -- Induction step: use recurrence formulae in `s` for Gamma and GammaSeq
    /-
      case succ
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      ⊢ LT.lt (Neg.neg ↑(HAdd.hAdd m 1)) s.re → Filter.Tendsto s.GammaSeq Filter.atT …
    -/
    intro hs
    /-
      case succ
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑(HAdd.hAdd m 1)) s.re
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Complex.GammaAux (HAdd.hAdd m  …
    -/
    rw [Nat.cast_succ, neg_add, ← sub_eq_add_neg, sub_lt_iff_lt_add, ← one_re, ← add_re] at hs
    /-
      case succ
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑m) (HAdd.hAdd s 1).re
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds (Complex.GammaAux (HAdd.hAdd m  …
    -/
    rw [GammaAux]
    have := @Tendsto.congr' _ _ _ ?_ _ _
      ((eventually_ne_atTop 0).mp (Eventually.of_forall fun n hn => ?_)) ((IH _ hs).div_const s)
    /-
      case succ.refine_3
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑m) (HAdd.hAdd s 1).re
      this : Filter.Tendsto ?succ.refine_1 Filter.atTop (nhds (HDiv.hDiv (Complex.Ga …
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds ((fun s => HDiv.hDiv (Complex.G …
    -/
    pick_goal 3; · exact GammaSeq_add_one_left s hn -- doesn't work if inlined?
                   /-
                     🎉 no goals
                   -/
    /-
      case succ.refine_3
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑m) (HAdd.hAdd s 1).re
      this : Filter.Tendsto (fun n => HMul.hMul (HDiv.hDiv (↑n) (HAdd.hAdd (HAdd.hAd …
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds ((fun s => HDiv.hDiv (Complex.G …
    -/
    conv at this => arg 1; intro n; rw [mul_comm]
    /-
      case succ.refine_3
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑m) (HAdd.hAdd s 1).re
      this : Filter.Tendsto (fun n => HMul.hMul (s.GammaSeq n) (HDiv.hDiv (↑n) (HAdd …
      ⊢ Filter.Tendsto s.GammaSeq Filter.atTop (nhds ((fun s => HDiv.hDiv (Complex.G …
    -/
    rwa [← mul_one (GammaAux m (s + 1) / s), tendsto_mul_iff_of_ne_zero _ (one_ne_zero' ℂ)] at this
    /-
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑m) (HAdd.hAdd s 1).re
      this : Filter.Tendsto (fun n => HMul.hMul (s.GammaSeq n) (HDiv.hDiv (↑n) (HAdd …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑n) (HAdd.hAdd (HAdd.hAdd (↑n) 1) s)) Fi …
    -/
    simp_rw [add_assoc]
    /-
      m : Nat
      IH : ∀ (s : Complex), LT.lt (Neg.neg ↑m) s.re → Filter.Tendsto s.GammaSeq Filt …
      s : Complex
      hs : LT.lt (Neg.neg ↑m) (HAdd.hAdd s 1).re
      this : Filter.Tendsto (fun n => HMul.hMul (s.GammaSeq n) (HDiv.hDiv (↑n) (HAdd …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑n) (HAdd.hAdd (↑n) (HAdd.hAdd 1 s))) Fi …
    -/
    exact tendsto_natCast_div_add_atTop (1 + s)
    /-
      🎉 no goals
    -/


theorem GammaSeq_mul (z : ℂ) {n : ℕ} (hn : n ≠ 0) :
    GammaSeq z n * GammaSeq (1 - z) n =
      n / (n + ↑1 - z) * (↑1 / (z * ∏ j ∈ Finset.range n, (↑1 - z ^ 2 / ((j : ℂ) + 1) ^ 2))) := by
  -- also true for n = 0 but we don't need it
  /-
    z : Complex
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HMul.hMul (z.GammaSeq n) ((HSub.hSub 1 z).GammaSeq n)) (HMul.hMul (HDiv. …
  -/
  have aux : ∀ a b c d : ℂ, a * b * (c * d) = a * c * (b * d) := by intros; ring
  /-
    z : Complex
    n : Nat
    hn : Ne n 0
    aux : ∀ (a b c d : Complex), Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul c d)) (H …
    ⊢ Eq (HMul.hMul (z.GammaSeq n) ((HSub.hSub 1 z).GammaSeq n)) (HMul.hMul (HDiv. …
  -/
  rw [GammaSeq, GammaSeq, div_mul_div_comm, aux, ← pow_two]
  have : (n : ℂ) ^ z * (n : ℂ) ^ (1 - z) = n := by
    rw [← cpow_add _ _ (Nat.cast_ne_zero.mpr hn), add_sub_cancel, cpow_one]
  rw [this, Finset.prod_range_succ', Finset.prod_range_succ, aux, ← Finset.prod_mul_distrib,
    Nat.cast_zero, add_zero, add_comm (1 - z) n, ← add_sub_assoc]
  have : ∀ j : ℕ, (z + ↑(j + 1)) * (↑1 - z + ↑j) =
      ((j + 1) ^ 2 :) * (↑1 - z ^ 2 / ((j : ℂ) + 1) ^ 2) := by
    intro j
    push_cast
    have : (j : ℂ) + 1 ≠ 0 := by rw [← Nat.cast_succ, Nat.cast_ne_zero]; exact Nat.succ_ne_zero j
    field_simp; ring
  /-
    z : Complex
    n : Nat
    hn : Ne n 0
    aux : ∀ (a b c d : Complex), Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul c d)) (H …
    this✝ : Eq (HMul.hMul (HPow.hPow (↑n) z) (HPow.hPow (↑n) (HSub.hSub 1 z))) ↑n
    this : ∀ (j : Nat), Eq (HMul.hMul (HAdd.hAdd z ↑(HAdd.hAdd j 1)) (HAdd.hAdd (H …
    ⊢ Eq (HDiv.hDiv (HMul.hMul (↑n) (HPow.hPow (↑n.factorial) 2)) (HMul.hMul ((Fin …
  -/
  simp_rw [this]
  rw [Finset.prod_mul_distrib, ← Nat.cast_prod, Finset.prod_pow,
    Finset.prod_range_add_one_eq_factorial, Nat.cast_pow,
    (by intros; ring : ∀ a b c d : ℂ, a * b * (c * d) = a * (d * (b * c))), ← div_div,
    mul_div_cancel_right₀, ← div_div, mul_comm z _, mul_one_div]
  /-
    case hb
    z : Complex
    n : Nat
    hn : Ne n 0
    aux : ∀ (a b c d : Complex), Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul c d)) (H …
    this✝ : Eq (HMul.hMul (HPow.hPow (↑n) z) (HPow.hPow (↑n) (HSub.hSub 1 z))) ↑n
    this : ∀ (j : Nat), Eq (HMul.hMul (HAdd.hAdd z ↑(HAdd.hAdd j 1)) (HAdd.hAdd (H …
    ⊢ Ne (HPow.hPow (↑n.factorial) 2) 0
  -/
  exact pow_ne_zero 2 (Nat.cast_ne_zero.mpr <| Nat.factorial_ne_zero n)
  /-
    🎉 no goals
  -/


/-- Euler's reflection formula for the complex Gamma function. -/
theorem Gamma_mul_Gamma_one_sub (z : ℂ) : Gamma z * Gamma (1 - z) = π / sin (π * z) := by
  /-
    z : Complex
    ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) (HDiv.hDiv  …
  -/
  have pi_ne : (π : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr pi_ne_zero
  /-
    z : Complex
    pi_ne : Ne (↑Real.pi) 0
    ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) (HDiv.hDiv  …
  -/
  by_cases hs : sin (↑π * z) = 0
  · -- first deal with silly case z = integer
    /-
      case pos
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0
      ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) (HDiv.hDiv  …
    -/
    rw [hs, div_zero]
    /-
      case pos
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0
      ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) 0
    -/
    rw [← neg_eq_zero, ← Complex.sin_neg, ← mul_neg, Complex.sin_eq_zero_iff, mul_comm] at hs
    /-
      case pos
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Exists fun k => Eq (HMul.hMul (Neg.neg z) ↑Real.pi) (HMul.hMul ↑k ↑Real.pi)
      ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) 0
    -/
    obtain ⟨k, hk⟩ := hs
    rw [mul_eq_mul_right_iff, eq_false (ofReal_ne_zero.mpr pi_pos.ne'), or_false,
      neg_eq_iff_eq_neg] at hk
    /-
      case pos.intro
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      k : Int
      hk : Eq z (Neg.neg ↑k)
      ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) 0
    -/
    rw [hk]
    /-
      case pos.intro
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      k : Int
      hk : Eq z (Neg.neg ↑k)
      ⊢ Eq (HMul.hMul (Complex.Gamma (Neg.neg ↑k)) (Complex.Gamma (HSub.hSub 1 (Neg. …
    -/
    cases k
      /-
        case pos.intro.ofNat
        z : Complex
        pi_ne : Ne (↑Real.pi) 0
        a✝ : Nat
        hk : Eq z (Neg.neg ↑(Int.ofNat a✝))
        ⊢ Eq (HMul.hMul (Complex.Gamma (Neg.neg ↑(Int.ofNat a✝))) (Complex.Gamma (HSub …
      -/
    · rw [Int.ofNat_eq_coe, Int.cast_natCast, Complex.Gamma_neg_nat_eq_zero, zero_mul]
      /-
        🎉 no goals
      -/
    · rw [Int.cast_negSucc, neg_neg, Nat.cast_add, Nat.cast_one, add_comm, sub_add_cancel_left,
        Complex.Gamma_neg_nat_eq_zero, mul_zero]
  /-
    case neg
    z : Complex
    pi_ne : Ne (↑Real.pi) 0
    hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
    ⊢ Eq (HMul.hMul (Complex.Gamma z) (Complex.Gamma (HSub.hSub 1 z))) (HDiv.hDiv  …
  -/
  refine tendsto_nhds_unique ((GammaSeq_tendsto_Gamma z).mul (GammaSeq_tendsto_Gamma <| 1 - z)) ?_
  /-
    case neg
    z : Complex
    pi_ne : Ne (↑Real.pi) 0
    hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (z.GammaSeq x) ((HSub.hSub 1 z).GammaSeq  …
  -/
  have : ↑π / sin (↑π * z) = 1 * (π / sin (π * z)) := by rw [one_mul]
  convert Tendsto.congr' ((eventually_ne_atTop 0).mp (Eventually.of_forall fun n hn =>
    (GammaSeq_mul z hn).symm)) (Tendsto.mul _ _)
    /-
      case neg.convert_3
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
      this : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HMul. …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑n) (HSub.hSub (HAdd.hAdd (↑n) 1) z)) Fi …
    -/
  · convert tendsto_natCast_div_add_atTop (1 - z) using 1; ext1 n; rw [add_sub_assoc]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    /-
      case neg.convert_4
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
      this : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HMul. …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv 1 (HMul.hMul z ((Finset.range n).prod fun …
    -/
  · have : ↑π / sin (↑π * z) = 1 / (sin (π * z) / π) := by field_simp
    /-
      case neg.convert_4
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
      this✝ : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HMul …
      this : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HDiv. …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv 1 (HMul.hMul z ((Finset.range n).prod fun …
    -/
    convert tendsto_const_nhds.div _ (div_ne_zero hs pi_ne)
    /-
      case neg.convert_4.convert_5
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
      this✝ : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HMul …
      this : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HDiv. …
      ⊢ Filter.Tendsto (fun x => HMul.hMul z ((Finset.range x).prod fun j => HSub.hS …
    -/
    rw [← tendsto_mul_iff_of_ne_zero tendsto_const_nhds pi_ne, div_mul_cancel₀ _ pi_ne]
    /-
      case neg.convert_4.convert_5
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
      this✝ : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HMul …
      this : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HDiv. …
      ⊢ Filter.Tendsto (fun n => HMul.hMul (HMul.hMul z ((Finset.range n).prod fun j …
    -/
    convert tendsto_euler_sin_prod z using 1
    /-
      case h.e'_3
      z : Complex
      pi_ne : Ne (↑Real.pi) 0
      hs : Not (Eq (Complex.sin (HMul.hMul (↑Real.pi) z)) 0)
      this✝ : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HMul …
      this : Eq (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) z))) (HDiv. …
      ⊢ Eq (fun n => HMul.hMul (HMul.hMul z ((Finset.range n).prod fun j => HSub.hSu …
    -/
    ext1 n; rw [mul_comm, ← mul_assoc]
            /-
              🎉 no goals
            -/


/-- The Gamma function does not vanish on `ℂ` (except at non-positive integers, where the function
is mathematically undefined and we set it to `0` by convention). -/
theorem Gamma_ne_zero {s : ℂ} (hs : ∀ m : ℕ, s ≠ -m) : Gamma s ≠ 0 := by
  /-
    s : Complex
    hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
    ⊢ Ne (Complex.Gamma s) 0
  -/
  by_cases h_im : s.im = 0
  · have : s = ↑s.re := by
      conv_lhs => rw [← Complex.re_add_im s]
      rw [h_im, ofReal_zero, zero_mul, add_zero]
    /-
      case pos
      s : Complex
      hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      h_im : Eq s.im 0
      this : Eq s ↑s.re
      ⊢ Ne (Complex.Gamma s) 0
    -/
    rw [this, Gamma_ofReal, ofReal_ne_zero]
    /-
      case pos
      s : Complex
      hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      h_im : Eq s.im 0
      this : Eq s ↑s.re
      ⊢ Ne (Real.Gamma s.re) 0
    -/
    refine Real.Gamma_ne_zero fun n => ?_
    /-
      case pos
      s : Complex
      hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      h_im : Eq s.im 0
      this : Eq s ↑s.re
      n : Nat
      ⊢ Ne s.re (Neg.neg ↑n)
    -/
    specialize hs n
    /-
      case pos
      s : Complex
      h_im : Eq s.im 0
      this : Eq s ↑s.re
      n : Nat
      hs : Ne s (Neg.neg ↑n)
      ⊢ Ne s.re (Neg.neg ↑n)
    -/
    contrapose! hs
    /-
      case pos
      s : Complex
      h_im : Eq s.im 0
      this : Eq s ↑s.re
      n : Nat
      hs : Eq s.re (Neg.neg ↑n)
      ⊢ Eq s (Neg.neg ↑n)
    -/
    rwa [this, ← ofReal_natCast, ← ofReal_neg, ofReal_inj]
    /-
      🎉 no goals
    -/
  · have : sin (↑π * s) ≠ 0 := by
      rw [Complex.sin_ne_zero_iff]
      intro k
      apply_fun im
      rw [im_ofReal_mul, ← ofReal_intCast, ← ofReal_mul, ofReal_im]
      exact mul_ne_zero Real.pi_pos.ne' h_im
    /-
      case neg
      s : Complex
      hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      h_im : Not (Eq s.im 0)
      this : Ne (Complex.sin (HMul.hMul (↑Real.pi) s)) 0
      ⊢ Ne (Complex.Gamma s) 0
    -/
    have A := div_ne_zero (ofReal_ne_zero.mpr Real.pi_pos.ne') this
    /-
      case neg
      s : Complex
      hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      h_im : Not (Eq s.im 0)
      this : Ne (Complex.sin (HMul.hMul (↑Real.pi) s)) 0
      A : Ne (HDiv.hDiv (↑Real.pi) (Complex.sin (HMul.hMul (↑Real.pi) s))) 0
      ⊢ Ne (Complex.Gamma s) 0
    -/
    rw [← Complex.Gamma_mul_Gamma_one_sub s, mul_ne_zero_iff] at A
    /-
      case neg
      s : Complex
      hs : ∀ (m : Nat), Ne s (Neg.neg ↑m)
      h_im : Not (Eq s.im 0)
      this : Ne (Complex.sin (HMul.hMul (↑Real.pi) s)) 0
      A : And (Ne (Complex.Gamma s) 0) (Ne (Complex.Gamma (HSub.hSub 1 s)) 0)
      ⊢ Ne (Complex.Gamma s) 0
    -/
    exact A.1
    /-
      🎉 no goals
    -/


theorem Gamma_eq_zero_iff (s : ℂ) : Gamma s = 0 ↔ ∃ m : ℕ, s = -m := by
  /-
    s : Complex
    ⊢ Iff (Eq (Complex.Gamma s) 0) (Exists fun m => Eq s (Neg.neg ↑m))
  -/
  constructor
    /-
      case mp
      s : Complex
      ⊢ Eq (Complex.Gamma s) 0 → Exists fun m => Eq s (Neg.neg ↑m)
    -/
  · contrapose!; exact Gamma_ne_zero
                 /-
                   🎉 no goals
                 -/
    /-
      case mpr
      s : Complex
      ⊢ (Exists fun m => Eq s (Neg.neg ↑m)) → Eq (Complex.Gamma s) 0
    -/
  · rintro ⟨m, rfl⟩; exact Gamma_neg_nat_eq_zero m
                     /-
                       🎉 no goals
                     -/


/-- A weaker, but easier-to-apply, version of `Complex.Gamma_ne_zero`. -/
theorem Gamma_ne_zero_of_re_pos {s : ℂ} (hs : 0 < re s) : Gamma s ≠ 0 := by
  /-
    s : Complex
    hs : LT.lt 0 s.re
    ⊢ Ne (Complex.Gamma s) 0
  -/
  refine Gamma_ne_zero fun m => ?_
  /-
    s : Complex
    hs : LT.lt 0 s.re
    m : Nat
    ⊢ Ne s (Neg.neg ↑m)
  -/
  contrapose! hs
  /-
    s : Complex
    m : Nat
    hs : Eq s (Neg.neg ↑m)
    ⊢ LE.le s.re 0
  -/
  simpa only [hs, neg_re, ← ofReal_natCast, ofReal_re, neg_nonpos] using Nat.cast_nonneg _
  /-
    🎉 no goals
  -/


/-- The sequence with `n`-th term `n ^ s * n! / (s * (s + 1) * ... * (s + n))`, for real `s`. We
will show that this tends to `Γ(s)` as `n → ∞`. -/
noncomputable def GammaSeq (s : ℝ) (n : ℕ) :=
  (n : ℝ) ^ s * n ! / ∏ j ∈ Finset.range (n + 1), (s + j)


/-- Euler's limit formula for the real Gamma function. -/
theorem GammaSeq_tendsto_Gamma (s : ℝ) : Tendsto (GammaSeq s) atTop (𝓝 <| Gamma s) := by
  suffices Tendsto ((↑) ∘ GammaSeq s : ℕ → ℂ) atTop (𝓝 <| Complex.Gamma s) by
    exact (Complex.continuous_re.tendsto (Complex.Gamma ↑s)).comp this
  /-
    s : Real
    ⊢ Filter.Tendsto (Function.comp Complex.ofReal s.GammaSeq) Filter.atTop (nhds  …
  -/
  convert Complex.GammaSeq_tendsto_Gamma s
  /-
    case h.e'_3
    s : Real
    ⊢ Eq (Function.comp Complex.ofReal s.GammaSeq) (↑s).GammaSeq
  -/
  ext1 n
  /-
    case h.e'_3.h
    s : Real
    n : Nat
    ⊢ Eq (Function.comp Complex.ofReal s.GammaSeq n) ((↑s).GammaSeq n)
  -/
  dsimp only [GammaSeq, Function.comp_apply, Complex.GammaSeq]
  /-
    case h.e'_3.h
    s : Real
    n : Nat
    ⊢ Eq (↑(HDiv.hDiv (HMul.hMul (HPow.hPow (↑n) s) ↑n.factorial) ((Finset.range ( …
  -/
  push_cast
  /-
    case h.e'_3.h
    s : Real
    n : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul ↑(HPow.hPow (↑n) s) ↑n.factorial) ((Finset.range (H …
  -/
  rw [Complex.ofReal_cpow n.cast_nonneg, Complex.ofReal_natCast]
  /-
    🎉 no goals
  -/


/-- Euler's reflection formula for the real Gamma function. -/
theorem Gamma_mul_Gamma_one_sub (s : ℝ) : Gamma s * Gamma (1 - s) = π / sin (π * s) := by
  simp_rw [← Complex.ofReal_inj, Complex.ofReal_div, Complex.ofReal_sin, Complex.ofReal_mul, ←
    Complex.Gamma_ofReal, Complex.ofReal_sub, Complex.ofReal_one]
  /-
    s : Real
    ⊢ Eq (HMul.hMul (Complex.Gamma ↑s) (Complex.Gamma (HSub.hSub 1 ↑s))) (HDiv.hDi …
  -/
  exact Complex.Gamma_mul_Gamma_one_sub s
  /-
    🎉 no goals
  -/


/-- A reformulation of the Gamma recurrence relation which is true for `s = 0` as well. -/
theorem one_div_Gamma_eq_self_mul_one_div_Gamma_add_one (s : ℂ) :
    (Gamma s)⁻¹ = s * (Gamma (s + 1))⁻¹ := by
  /-
    s : Complex
    ⊢ Eq (Inv.inv (Complex.Gamma s)) (HMul.hMul s (Inv.inv (Complex.Gamma (HAdd.hA …
  -/
  rcases ne_or_eq s 0 with (h | rfl)
    /-
      case inl
      s : Complex
      h : Ne s 0
      ⊢ Eq (Inv.inv (Complex.Gamma s)) (HMul.hMul s (Inv.inv (Complex.Gamma (HAdd.hA …
    -/
  · rw [Gamma_add_one s h, mul_inv, mul_inv_cancel_left₀ h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ⊢ Eq (Inv.inv (Complex.Gamma 0)) (HMul.hMul 0 (Inv.inv (Complex.Gamma (HAdd.hA …
    -/
  · rw [zero_add, Gamma_zero, inv_zero, zero_mul]
    /-
      🎉 no goals
    -/


/-- The reciprocal of the Gamma function is differentiable everywhere
(including the points where Gamma itself is not). -/
theorem differentiable_one_div_Gamma : Differentiable ℂ fun s : ℂ => (Gamma s)⁻¹ := fun s ↦ by
  /-
    s : Complex
    ⊢ DifferentiableAt Complex (fun s => Inv.inv (Complex.Gamma s)) s
  -/
  rcases exists_nat_gt (-s.re) with ⟨n, hs⟩
  induction n generalizing s with
  | zero =>
    rw [Nat.cast_zero, neg_lt_zero] at hs
    suffices ∀ m : ℕ, s ≠ -↑m from (differentiableAt_Gamma _ this).inv (Gamma_ne_zero this)
    rintro m rfl
    apply hs.not_le
    simp
  | succ n ihn =>
    rw [funext one_div_Gamma_eq_self_mul_one_div_Gamma_add_one]
    specialize ihn (s + 1) (by rwa [add_re, one_re, neg_add', sub_lt_iff_lt_add, ← Nat.cast_succ])
    exact differentiableAt_id.mul (ihn.comp s (f := fun s => s + 1) <|
      differentiableAt_id.add_const (1 : ℂ))


theorem Gamma_mul_Gamma_add_half (s : ℂ) :
    Gamma s * Gamma (s + 1 / 2) = Gamma (2 * s) * (2 : ℂ) ^ (1 - 2 * s) * ↑(√π) := by
  suffices (fun z => (Gamma z)⁻¹ * (Gamma (z + 1 / 2))⁻¹) = fun z =>
      (Gamma (2 * z))⁻¹ * (2 : ℂ) ^ (2 * z - 1) / ↑(√π) by
    convert congr_arg Inv.inv (congr_fun this s) using 1
    · rw [mul_inv, inv_inv, inv_inv]
    · rw [div_eq_mul_inv, mul_inv, mul_inv, inv_inv, inv_inv, ← cpow_neg, neg_sub]
  have h1 : AnalyticOnNhd ℂ (fun z : ℂ => (Gamma z)⁻¹ * (Gamma (z + 1 / 2))⁻¹) univ := by
    refine DifferentiableOn.analyticOnNhd ?_ isOpen_univ
    refine (differentiable_one_div_Gamma.mul ?_).differentiableOn
    exact differentiable_one_div_Gamma.comp (differentiable_id.add (differentiable_const _))
  have h2 : AnalyticOnNhd ℂ
      (fun z => (Gamma (2 * z))⁻¹ * (2 : ℂ) ^ (2 * z - 1) / ↑(√π)) univ := by
    refine DifferentiableOn.analyticOnNhd ?_ isOpen_univ
    refine (Differentiable.mul ?_ (differentiable_const _)).differentiableOn
    apply Differentiable.mul
    · exact differentiable_one_div_Gamma.comp (differentiable_id'.const_mul _)
    · refine fun t => DifferentiableAt.const_cpow ?_ (Or.inl two_ne_zero)
      exact DifferentiableAt.sub_const (differentiableAt_id.const_mul _) _
  have h3 : Tendsto ((↑) : ℝ → ℂ) (𝓝[≠] 1) (𝓝[≠] 1) := by
    rw [tendsto_nhdsWithin_iff]; constructor
    · exact tendsto_nhdsWithin_of_tendsto_nhds continuous_ofReal.continuousAt
    · exact eventually_nhdsWithin_iff.mpr (Eventually.of_forall fun t ht => ofReal_ne_one.mpr ht)
  /-
    s : Complex
    h1 : AnalyticOnNhd Complex (fun z => HMul.hMul (Inv.inv (Complex.Gamma z)) (In …
    h2 : AnalyticOnNhd Complex (fun z => HDiv.hDiv (HMul.hMul (Inv.inv (Complex.Ga …
    h3 : Filter.Tendsto Complex.ofReal (nhdsWithin 1 (HasCompl.compl (Singleton.si …
    ⊢ Eq (fun z => HMul.hMul (Inv.inv (Complex.Gamma z)) (Inv.inv (Complex.Gamma ( …
  -/
  refine AnalyticOnNhd.eq_of_frequently_eq h1 h2 (h3.frequently ?_)
  /-
    s : Complex
    h1 : AnalyticOnNhd Complex (fun z => HMul.hMul (Inv.inv (Complex.Gamma z)) (In …
    h2 : AnalyticOnNhd Complex (fun z => HDiv.hDiv (HMul.hMul (Inv.inv (Complex.Ga …
    h3 : Filter.Tendsto Complex.ofReal (nhdsWithin 1 (HasCompl.compl (Singleton.si …
    ⊢ Filter.Frequently (fun x => Eq (HMul.hMul (Inv.inv (Complex.Gamma ↑x)) (Inv. …
  -/
  refine ((Eventually.filter_mono nhdsWithin_le_nhds) ?_).frequently
  /-
    s : Complex
    h1 : AnalyticOnNhd Complex (fun z => HMul.hMul (Inv.inv (Complex.Gamma z)) (In …
    h2 : AnalyticOnNhd Complex (fun z => HDiv.hDiv (HMul.hMul (Inv.inv (Complex.Ga …
    h3 : Filter.Tendsto Complex.ofReal (nhdsWithin 1 (HasCompl.compl (Singleton.si …
    ⊢ Filter.Eventually (fun x => Eq (HMul.hMul (Inv.inv (Complex.Gamma ↑x)) (Inv. …
  -/
  refine (eventually_gt_nhds zero_lt_one).mp (Eventually.of_forall fun t ht => ?_)
  rw [← mul_inv, Gamma_ofReal, (by norm_num : (t : ℂ) + 1 / 2 = ↑(t + 1 / 2)), Gamma_ofReal, ←
    ofReal_mul, Gamma_mul_Gamma_add_half_of_pos ht, ofReal_mul, ofReal_mul, ← Gamma_ofReal,
    mul_inv, mul_inv, (by norm_num : 2 * (t : ℂ) = ↑(2 * t)), Gamma_ofReal,
    ofReal_cpow zero_le_two, show (2 : ℝ) = (2 : ℂ) by norm_cast, ← cpow_neg, ofReal_sub,
    ofReal_one, neg_sub, ← div_eq_mul_inv]


theorem Gamma_mul_Gamma_add_half (s : ℝ) :
    Gamma s * Gamma (s + 1 / 2) = Gamma (2 * s) * (2 : ℝ) ^ (1 - 2 * s) * √π := by
  /-
    s : Real
    ⊢ Eq (HMul.hMul (Real.Gamma s) (Real.Gamma (HAdd.hAdd s (1 / 2)))) (HMul.hMul  …
  -/
  rw [← ofReal_inj]
  simpa only [← Gamma_ofReal, ofReal_cpow zero_le_two, ofReal_mul, ofReal_add, ofReal_div,
    ofReal_sub] using Complex.Gamma_mul_Gamma_add_half ↑s


