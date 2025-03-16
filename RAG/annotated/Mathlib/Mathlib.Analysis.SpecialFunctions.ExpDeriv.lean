/-- `exp` is entire -/
theorem analyticOnNhd_cexp : AnalyticOnNhd ℂ exp univ := by
  /-
    ⊢ AnalyticOnNhd Complex Complex.exp Set.univ
  -/
  rw [Complex.exp_eq_exp_ℂ]
  /-
    ⊢ AnalyticOnNhd Complex (NormedSpace.exp Complex) Set.univ
  -/
  exact fun x _ ↦ NormedSpace.exp_analytic x
  /-
    🎉 no goals
  -/


theorem analyticOn_cexp : AnalyticOn ℂ exp univ := analyticOnNhd_cexp.analyticOn


/-- `exp` is analytic at any point -/
theorem analyticAt_cexp : AnalyticAt ℂ exp z :=
  analyticOnNhd_cexp z (mem_univ _)


/-- `exp ∘ f` is analytic -/
theorem AnalyticAt.cexp (fa : AnalyticAt ℂ f x) : AnalyticAt ℂ (fun z ↦ exp (f z)) x :=
  analyticAt_cexp.comp fa


theorem AnalyticWithinAt.cexp (fa : AnalyticWithinAt ℂ f s x) :
    AnalyticWithinAt ℂ (fun z ↦ exp (f z)) s x :=
  analyticAt_cexp.comp_analyticWithinAt fa


/-- `exp ∘ f` is analytic -/
theorem AnalyticOnNhd.cexp (fs : AnalyticOnNhd ℂ f s) : AnalyticOnNhd ℂ (fun z ↦ exp (f z)) s :=
  fun z n ↦ analyticAt_cexp.comp (fs z n)


theorem AnalyticOn.cexp (fs : AnalyticOn ℂ f s) : AnalyticOn ℂ (fun z ↦ exp (f z)) s :=
  analyticOnNhd_cexp.comp_analyticOn fs (mapsTo_univ _ _)


/-- The complex exponential is everywhere differentiable, with the derivative `exp x`. -/
theorem hasDerivAt_exp (x : ℂ) : HasDerivAt exp (exp x) x := by
  /-
    x : Complex
    ⊢ HasDerivAt Complex.exp (Complex.exp x) x
  -/
  rw [hasDerivAt_iff_isLittleO_nhds_zero]
  /-
    x : Complex
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun h => HSub.hSub (HSub.hSub (Complex.exp ( …
  -/
  have : (1 : ℕ) < 2 := by norm_num
  /-
    x : Complex
    this : LT.lt 1 2
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun h => HSub.hSub (HSub.hSub (Complex.exp ( …
  -/
  refine (IsBigO.of_bound ‖exp x‖ ?_).trans_isLittleO (isLittleO_pow_id this)
  /-
    x : Complex
    this : LT.lt 1 2
    ⊢ Filter.Eventually (fun x_1 => LE.le (Norm.norm (HSub.hSub (HSub.hSub (Comple …
  -/
  filter_upwards [Metric.ball_mem_nhds (0 : ℂ) zero_lt_one]
  /-
    case h
    x : Complex
    this : LT.lt 1 2
    ⊢ ∀ (a : Complex), Membership.mem (Metric.ball 0 1) a → LE.le (Norm.norm (HSub …
  -/
  simp only [Metric.mem_ball, dist_zero_right, norm_pow]
  /-
    case h
    x : Complex
    this : LT.lt 1 2
    ⊢ ∀ (a : Complex), LT.lt (Norm.norm a) 1 → LE.le (Norm.norm (HSub.hSub (HSub.h …
  -/
  exact fun z hz => exp_bound_sq x z hz.le
  /-
    🎉 no goals
  -/


@[simp]
theorem differentiable_exp : Differentiable 𝕜 exp := fun x =>
  (hasDerivAt_exp x).differentiableAt.restrictScalars 𝕜


@[simp]
theorem differentiableAt_exp {x : ℂ} : DifferentiableAt 𝕜 exp x :=
  differentiable_exp x


@[simp]
theorem deriv_exp : deriv exp = exp :=
  funext fun x => (hasDerivAt_exp x).deriv


@[simp]
theorem iter_deriv_exp : ∀ n : ℕ, deriv^[n] exp = exp
  | 0 => rfl
                /-
                  n : Nat
                  ⊢ Eq (Nat.iterate deriv (HAdd.hAdd n 1) Complex.exp) Complex.exp
                -/
  | n + 1 => by rw [iterate_succ_apply, deriv_exp, iter_deriv_exp n]
                /-
                  🎉 no goals
                -/


theorem contDiff_exp {n : WithTop ℕ∞} : ContDiff 𝕜 n exp :=
  analyticOnNhd_cexp.restrictScalars.contDiff


theorem hasStrictDerivAt_exp (x : ℂ) : HasStrictDerivAt exp (exp x) x :=
  contDiff_exp.contDiffAt.hasStrictDerivAt' (hasDerivAt_exp x) le_rfl


theorem hasStrictFDerivAt_exp_real (x : ℂ) : HasStrictFDerivAt exp (exp x • (1 : ℂ →L[ℝ] ℂ)) x :=
  (hasStrictDerivAt_exp x).complexToReal_fderiv


theorem HasStrictDerivAt.cexp (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun x => Complex.exp (f x)) (Complex.exp (f x) * f') x :=
  (Complex.hasStrictDerivAt_exp (f x)).comp x hf


theorem HasDerivAt.cexp (hf : HasDerivAt f f' x) :
    HasDerivAt (fun x => Complex.exp (f x)) (Complex.exp (f x) * f') x :=
  (Complex.hasDerivAt_exp (f x)).comp x hf


theorem HasDerivWithinAt.cexp (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun x => Complex.exp (f x)) (Complex.exp (f x) * f') s x :=
  (Complex.hasDerivAt_exp (f x)).comp_hasDerivWithinAt x hf


theorem derivWithin_cexp (hf : DifferentiableWithinAt 𝕜 f s x) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x => Complex.exp (f x)) s x = Complex.exp (f x) * derivWithin f s x :=
  hf.hasDerivWithinAt.cexp.derivWithin hxs


@[simp]
theorem deriv_cexp (hc : DifferentiableAt 𝕜 f x) :
    deriv (fun x => Complex.exp (f x)) x = Complex.exp (f x) * deriv f x :=
  hc.hasDerivAt.cexp.deriv


theorem HasStrictFDerivAt.cexp (hf : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (fun x => Complex.exp (f x)) (Complex.exp (f x) • f') x :=
  (Complex.hasStrictDerivAt_exp (f x)).comp_hasStrictFDerivAt x hf


theorem HasFDerivWithinAt.cexp (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (fun x => Complex.exp (f x)) (Complex.exp (f x) • f') s x :=
  (Complex.hasDerivAt_exp (f x)).comp_hasFDerivWithinAt x hf


theorem HasFDerivAt.cexp (hf : HasFDerivAt f f' x) :
    HasFDerivAt (fun x => Complex.exp (f x)) (Complex.exp (f x) • f') x :=
  hasFDerivWithinAt_univ.1 <| hf.hasFDerivWithinAt.cexp


theorem DifferentiableWithinAt.cexp (hf : DifferentiableWithinAt 𝕜 f s x) :
    DifferentiableWithinAt 𝕜 (fun x => Complex.exp (f x)) s x :=
  hf.hasFDerivWithinAt.cexp.differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.cexp (hc : DifferentiableAt 𝕜 f x) :
    DifferentiableAt 𝕜 (fun x => Complex.exp (f x)) x :=
  hc.hasFDerivAt.cexp.differentiableAt


theorem DifferentiableOn.cexp (hc : DifferentiableOn 𝕜 f s) :
    DifferentiableOn 𝕜 (fun x => Complex.exp (f x)) s := fun x h => (hc x h).cexp


@[simp, fun_prop]
theorem Differentiable.cexp (hc : Differentiable 𝕜 f) :
    Differentiable 𝕜 fun x => Complex.exp (f x) := fun x => (hc x).cexp


theorem ContDiff.cexp {n} (h : ContDiff 𝕜 n f) : ContDiff 𝕜 n fun x => Complex.exp (f x) :=
  Complex.contDiff_exp.comp h


theorem ContDiffAt.cexp {n} (hf : ContDiffAt 𝕜 n f x) :
    ContDiffAt 𝕜 n (fun x => Complex.exp (f x)) x :=
  Complex.contDiff_exp.contDiffAt.comp x hf


theorem ContDiffOn.cexp {n} (hf : ContDiffOn 𝕜 n f s) :
    ContDiffOn 𝕜 n (fun x => Complex.exp (f x)) s :=
  Complex.contDiff_exp.comp_contDiffOn hf


theorem ContDiffWithinAt.cexp {n} (hf : ContDiffWithinAt 𝕜 n f s x) :
    ContDiffWithinAt 𝕜 n (fun x => Complex.exp (f x)) s x :=
  Complex.contDiff_exp.contDiffAt.comp_contDiffWithinAt x hf


open Complex in
@[simp]
theorem iteratedDeriv_cexp_const_mul (n : ℕ) (c : ℂ) :
    (iteratedDeriv n fun s : ℂ => exp (c * s)) = fun s => c ^ n * exp (c * s) := by
  /-
    n : Nat
    c : Complex
    ⊢ Eq (iteratedDeriv n fun s => Complex.exp (HMul.hMul c s)) fun s => HMul.hMul …
  -/
  rw [iteratedDeriv_comp_const_mul contDiff_exp, iteratedDeriv_eq_iterate, iter_deriv_exp]
  /-
    🎉 no goals
  -/


/-- `exp` is entire -/
theorem analyticOnNhd_rexp : AnalyticOnNhd ℝ exp univ := by
  /-
    ⊢ AnalyticOnNhd Real Real.exp Set.univ
  -/
  rw [Real.exp_eq_exp_ℝ]
  /-
    ⊢ AnalyticOnNhd Real (NormedSpace.exp Real) Set.univ
  -/
  exact fun x _ ↦ NormedSpace.exp_analytic x
  /-
    🎉 no goals
  -/


theorem analyticOn_rexp : AnalyticOn ℝ exp univ := analyticOnNhd_rexp.analyticOn


/-- `exp` is analytic at any point -/
theorem analyticAt_rexp : AnalyticAt ℝ exp x :=
  analyticOnNhd_rexp x (mem_univ _)


/-- `exp ∘ f` is analytic -/
theorem AnalyticAt.rexp {x : E} (fa : AnalyticAt ℝ f x) : AnalyticAt ℝ (fun z ↦ exp (f z)) x :=
  analyticAt_rexp.comp fa


theorem AnalyticWithinAt.rexp {x : E} (fa : AnalyticWithinAt ℝ f s x) :
    AnalyticWithinAt ℝ (fun z ↦ exp (f z)) s x :=
  analyticAt_rexp.comp_analyticWithinAt fa


/-- `exp ∘ f` is analytic -/
theorem AnalyticOnNhd.rexp {s : Set E} (fs : AnalyticOnNhd ℝ f s) :
    AnalyticOnNhd ℝ (fun z ↦ exp (f z)) s :=
  fun z n ↦ analyticAt_rexp.comp (fs z n)


theorem AnalyticOn.rexp (fs : AnalyticOn ℝ f s) : AnalyticOn ℝ (fun z ↦ exp (f z)) s :=
  analyticOnNhd_rexp.comp_analyticOn fs (mapsTo_univ _ _)


theorem hasStrictDerivAt_exp (x : ℝ) : HasStrictDerivAt exp (exp x) x :=
  (Complex.hasStrictDerivAt_exp x).real_of_complex


theorem hasDerivAt_exp (x : ℝ) : HasDerivAt exp (exp x) x :=
  (Complex.hasDerivAt_exp x).real_of_complex


theorem contDiff_exp {n : WithTop ℕ∞} : ContDiff ℝ n exp :=
  Complex.contDiff_exp.real_of_complex


@[simp]
theorem differentiable_exp : Differentiable ℝ exp := fun x => (hasDerivAt_exp x).differentiableAt


@[simp]
theorem differentiableAt_exp {x : ℝ} : DifferentiableAt ℝ exp x :=
  differentiable_exp x


@[simp]
theorem iter_deriv_exp : ∀ n : ℕ, deriv^[n] exp = exp
  | 0 => rfl
                /-
                  n : Nat
                  ⊢ Eq (Nat.iterate deriv (HAdd.hAdd n 1) Real.exp) Real.exp
                -/
  | n + 1 => by rw [iterate_succ_apply, deriv_exp, iter_deriv_exp n]
                /-
                  🎉 no goals
                -/


theorem HasStrictDerivAt.exp (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun x => Real.exp (f x)) (Real.exp (f x) * f') x :=
  (Real.hasStrictDerivAt_exp (f x)).comp x hf


theorem HasDerivAt.exp (hf : HasDerivAt f f' x) :
    HasDerivAt (fun x => Real.exp (f x)) (Real.exp (f x) * f') x :=
  (Real.hasDerivAt_exp (f x)).comp x hf


theorem HasDerivWithinAt.exp (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun x => Real.exp (f x)) (Real.exp (f x) * f') s x :=
  (Real.hasDerivAt_exp (f x)).comp_hasDerivWithinAt x hf


theorem derivWithin_exp (hf : DifferentiableWithinAt ℝ f s x) (hxs : UniqueDiffWithinAt ℝ s x) :
    derivWithin (fun x => Real.exp (f x)) s x = Real.exp (f x) * derivWithin f s x :=
  hf.hasDerivWithinAt.exp.derivWithin hxs


@[simp]
theorem deriv_exp (hc : DifferentiableAt ℝ f x) :
    deriv (fun x => Real.exp (f x)) x = Real.exp (f x) * deriv f x :=
  hc.hasDerivAt.exp.deriv


theorem ContDiff.exp {n} (hf : ContDiff ℝ n f) : ContDiff ℝ n fun x => Real.exp (f x) :=
  Real.contDiff_exp.comp hf


theorem ContDiffAt.exp {n} (hf : ContDiffAt ℝ n f x) : ContDiffAt ℝ n (fun x => Real.exp (f x)) x :=
  Real.contDiff_exp.contDiffAt.comp x hf


theorem ContDiffOn.exp {n} (hf : ContDiffOn ℝ n f s) : ContDiffOn ℝ n (fun x => Real.exp (f x)) s :=
  Real.contDiff_exp.comp_contDiffOn hf


theorem ContDiffWithinAt.exp {n} (hf : ContDiffWithinAt ℝ n f s x) :
    ContDiffWithinAt ℝ n (fun x => Real.exp (f x)) s x :=
  Real.contDiff_exp.contDiffAt.comp_contDiffWithinAt x hf


theorem HasFDerivWithinAt.exp (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (fun x => Real.exp (f x)) (Real.exp (f x) • f') s x :=
  (Real.hasDerivAt_exp (f x)).comp_hasFDerivWithinAt x hf


theorem HasFDerivAt.exp (hf : HasFDerivAt f f' x) :
    HasFDerivAt (fun x => Real.exp (f x)) (Real.exp (f x) • f') x :=
  (Real.hasDerivAt_exp (f x)).comp_hasFDerivAt x hf


theorem HasStrictFDerivAt.exp (hf : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (fun x => Real.exp (f x)) (Real.exp (f x) • f') x :=
  (Real.hasStrictDerivAt_exp (f x)).comp_hasStrictFDerivAt x hf


theorem DifferentiableWithinAt.exp (hf : DifferentiableWithinAt ℝ f s x) :
    DifferentiableWithinAt ℝ (fun x => Real.exp (f x)) s x :=
  hf.hasFDerivWithinAt.exp.differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.exp (hc : DifferentiableAt ℝ f x) :
    DifferentiableAt ℝ (fun x => Real.exp (f x)) x :=
  hc.hasFDerivAt.exp.differentiableAt


theorem DifferentiableOn.exp (hc : DifferentiableOn ℝ f s) :
    DifferentiableOn ℝ (fun x => Real.exp (f x)) s := fun x h => (hc x h).exp


@[simp, fun_prop]
theorem Differentiable.exp (hc : Differentiable ℝ f) : Differentiable ℝ fun x => Real.exp (f x) :=
  fun x => (hc x).exp


theorem fderivWithin_exp (hf : DifferentiableWithinAt ℝ f s x) (hxs : UniqueDiffWithinAt ℝ s x) :
    fderivWithin ℝ (fun x => Real.exp (f x)) s x = Real.exp (f x) • fderivWithin ℝ f s x :=
  hf.hasFDerivWithinAt.exp.fderivWithin hxs


@[simp]
theorem fderiv_exp (hc : DifferentiableAt ℝ f x) :
    fderiv ℝ (fun x => Real.exp (f x)) x = Real.exp (f x) • fderiv ℝ f x :=
  hc.hasFDerivAt.exp.fderiv


open Real in
@[simp]
theorem iteratedDeriv_exp_const_mul (n : ℕ) (c : ℝ) :
    (iteratedDeriv n fun s => exp (c * s)) = fun s => c ^ n * exp (c * s) := by
  /-
    n : Nat
    c : Real
    ⊢ Eq (iteratedDeriv n fun s => Real.exp (HMul.hMul c s)) fun s => HMul.hMul (H …
  -/
  rw [iteratedDeriv_comp_const_mul contDiff_exp, iteratedDeriv_eq_iterate, iter_deriv_exp]
  /-
    🎉 no goals
  -/

