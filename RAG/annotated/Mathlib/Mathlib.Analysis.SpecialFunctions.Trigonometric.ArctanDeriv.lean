theorem hasStrictDerivAt_tan {x : ℝ} (h : cos x ≠ 0) : HasStrictDerivAt tan (1 / cos x ^ 2) x :=
                                             /-
                                               x : Real
                                               h : Ne (Real.cos x) 0
                                               ⊢ Ne (Complex.cos ↑x) 0
                                             -/
  mod_cast (Complex.hasStrictDerivAt_tan (by exact mod_cast h)).real_of_complex
                                             /-
                                               🎉 no goals
                                             -/


theorem hasDerivAt_tan {x : ℝ} (h : cos x ≠ 0) : HasDerivAt tan (1 / cos x ^ 2) x :=
                                       /-
                                         x : Real
                                         h : Ne (Real.cos x) 0
                                         ⊢ Ne (Complex.cos ↑x) 0
                                       -/
  mod_cast (Complex.hasDerivAt_tan (by exact mod_cast h)).real_of_complex
                                       /-
                                         🎉 no goals
                                       -/


theorem tendsto_abs_tan_of_cos_eq_zero {x : ℝ} (hx : cos x = 0) :
    Tendsto (fun x => abs (tan x)) (𝓝[≠] x) atTop := by
  /-
    x : Real
    hx : Eq (Real.cos x) 0
    ⊢ Filter.Tendsto (fun x => abs (Real.tan x)) (nhdsWithin x (HasCompl.compl (Si …
  -/
  have hx : Complex.cos x = 0 := mod_cast hx
  /-
    x : Real
    hx✝ : Eq (Real.cos x) 0
    hx : Eq (Complex.cos ↑x) 0
    ⊢ Filter.Tendsto (fun x => abs (Real.tan x)) (nhdsWithin x (HasCompl.compl (Si …
  -/
  simp only [← Complex.abs_ofReal, Complex.ofReal_tan]
  /-
    x : Real
    hx✝ : Eq (Real.cos x) 0
    hx : Eq (Complex.cos ↑x) 0
    ⊢ Filter.Tendsto (fun x => Complex.abs (Complex.tan ↑x)) (nhdsWithin x (HasCom …
  -/
  refine (Complex.tendsto_abs_tan_of_cos_eq_zero hx).comp ?_
  /-
    x : Real
    hx✝ : Eq (Real.cos x) 0
    hx : Eq (Complex.cos ↑x) 0
    ⊢ Filter.Tendsto Complex.ofReal (nhdsWithin x (HasCompl.compl (Singleton.singl …
  -/
  refine Tendsto.inf Complex.continuous_ofReal.continuousAt ?_
  /-
    x : Real
    hx✝ : Eq (Real.cos x) 0
    hx : Eq (Complex.cos ↑x) 0
    ⊢ Filter.Tendsto Complex.ofReal (Filter.principal (HasCompl.compl (Singleton.s …
  -/
  exact tendsto_principal_principal.2 fun y => mt Complex.ofReal_inj.1
  /-
    🎉 no goals
  -/


theorem tendsto_abs_tan_atTop (k : ℤ) :
    Tendsto (fun x => abs (tan x)) (𝓝[≠] ((2 * k + 1) * π / 2)) atTop :=
  tendsto_abs_tan_of_cos_eq_zero <| cos_eq_zero_iff.2 ⟨k, rfl⟩


theorem continuousAt_tan {x : ℝ} : ContinuousAt tan x ↔ cos x ≠ 0 := by
  /-
    x : Real
    ⊢ Iff (ContinuousAt Real.tan x) (Ne (Real.cos x) 0)
  -/
  refine ⟨fun hc h₀ => ?_, fun h => (hasDerivAt_tan h).continuousAt⟩
  exact not_tendsto_nhds_of_tendsto_atTop (tendsto_abs_tan_of_cos_eq_zero h₀) _
    (hc.norm.tendsto.mono_left inf_le_left)


theorem differentiableAt_tan {x : ℝ} : DifferentiableAt ℝ tan x ↔ cos x ≠ 0 :=
  ⟨fun h => continuousAt_tan.1 h.continuousAt, fun h => (hasDerivAt_tan h).differentiableAt⟩


@[simp]
theorem deriv_tan (x : ℝ) : deriv tan x = 1 / cos x ^ 2 :=
  if h : cos x = 0 then by
    /-
      x : Real
      h : Eq (Real.cos x) 0
      ⊢ Eq (deriv Real.tan x) (HDiv.hDiv 1 (HPow.hPow (Real.cos x) 2))
    -/
    have : ¬DifferentiableAt ℝ tan x := mt differentiableAt_tan.1 (Classical.not_not.2 h)
    /-
      x : Real
      h : Eq (Real.cos x) 0
      this : Not (DifferentiableAt Real Real.tan x)
      ⊢ Eq (deriv Real.tan x) (HDiv.hDiv 1 (HPow.hPow (Real.cos x) 2))
    -/
    simp [deriv_zero_of_not_differentiableAt this, h, sq]
    /-
      🎉 no goals
    -/
  else (hasDerivAt_tan h).deriv


@[simp]
theorem contDiffAt_tan {n : WithTop ℕ∞} {x : ℝ} : ContDiffAt ℝ n tan x ↔ cos x ≠ 0 :=
  ⟨fun h => continuousAt_tan.1 h.continuousAt, fun h =>
    (Complex.contDiffAt_tan.2 <| mod_cast h).real_of_complex⟩


theorem hasDerivAt_tan_of_mem_Ioo {x : ℝ} (h : x ∈ Ioo (-(π / 2) : ℝ) (π / 2)) :
    HasDerivAt tan (1 / cos x ^ 2) x :=
  hasDerivAt_tan (cos_pos_of_mem_Ioo h).ne'


theorem differentiableAt_tan_of_mem_Ioo {x : ℝ} (h : x ∈ Ioo (-(π / 2) : ℝ) (π / 2)) :
    DifferentiableAt ℝ tan x :=
  (hasDerivAt_tan_of_mem_Ioo h).differentiableAt


theorem hasStrictDerivAt_arctan (x : ℝ) : HasStrictDerivAt arctan (1 / (1 + x ^ 2)) x := by
  /-
    x : Real
    ⊢ HasStrictDerivAt Real.arctan (HDiv.hDiv 1 (HAdd.hAdd 1 (HPow.hPow x 2))) x
  -/
  have A : cos (arctan x) ≠ 0 := (cos_arctan_pos x).ne'
  simpa [cos_sq_arctan] using
    tanPartialHomeomorph.hasStrictDerivAt_symm trivial (by simpa) (hasStrictDerivAt_tan A)


theorem hasDerivAt_arctan (x : ℝ) : HasDerivAt arctan (1 / (1 + x ^ 2)) x :=
  (hasStrictDerivAt_arctan x).hasDerivAt


theorem hasDerivAt_arctan' (x : ℝ) : HasDerivAt arctan (1 + x ^ 2)⁻¹ x :=
  one_div (1 + x ^ 2) ▸ hasDerivAt_arctan x


theorem differentiableAt_arctan (x : ℝ) : DifferentiableAt ℝ arctan x :=
  (hasDerivAt_arctan x).differentiableAt


theorem differentiable_arctan : Differentiable ℝ arctan :=
  differentiableAt_arctan


@[simp]
theorem deriv_arctan : deriv arctan = fun (x : ℝ) => 1 / (1 + x ^ 2) :=
  funext fun x => (hasDerivAt_arctan x).deriv


theorem contDiff_arctan {n : WithTop ℕ∞} : ContDiff ℝ n arctan :=
  contDiff_iff_contDiffAt.2 fun x =>
    have : cos (arctan x) ≠ 0 := (cos_arctan_pos x).ne'
                                                   /-
                                                     n : WithTop ENat
                                                     x : Real
                                                     this : Ne (Real.cos (Real.arctan x)) 0
                                                     ⊢ Ne (HDiv.hDiv 1 (HPow.hPow (Real.cos (↑Real.tanPartialHomeomorph.symm x)) 2) …
                                                   -/
    tanPartialHomeomorph.contDiffAt_symm_deriv (by simpa) trivial (hasDerivAt_tan this)
                                                   /-
                                                     🎉 no goals
                                                   -/
      (contDiffAt_tan.2 this)


theorem HasStrictDerivAt.arctan (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun x => arctan (f x)) (1 / (1 + f x ^ 2) * f') x :=
  (Real.hasStrictDerivAt_arctan (f x)).comp x hf


theorem HasDerivAt.arctan (hf : HasDerivAt f f' x) :
    HasDerivAt (fun x => arctan (f x)) (1 / (1 + f x ^ 2) * f') x :=
  (Real.hasDerivAt_arctan (f x)).comp x hf


theorem HasDerivWithinAt.arctan (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun x => arctan (f x)) (1 / (1 + f x ^ 2) * f') s x :=
  (Real.hasDerivAt_arctan (f x)).comp_hasDerivWithinAt x hf


theorem derivWithin_arctan (hf : DifferentiableWithinAt ℝ f s x) (hxs : UniqueDiffWithinAt ℝ s x) :
    derivWithin (fun x => arctan (f x)) s x = 1 / (1 + f x ^ 2) * derivWithin f s x :=
  hf.hasDerivWithinAt.arctan.derivWithin hxs


@[simp]
theorem deriv_arctan (hc : DifferentiableAt ℝ f x) :
    deriv (fun x => arctan (f x)) x = 1 / (1 + f x ^ 2) * deriv f x :=
  hc.hasDerivAt.arctan.deriv


theorem HasStrictFDerivAt.arctan (hf : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (fun x => arctan (f x)) ((1 / (1 + f x ^ 2)) • f') x :=
  (hasStrictDerivAt_arctan (f x)).comp_hasStrictFDerivAt x hf


theorem HasFDerivAt.arctan (hf : HasFDerivAt f f' x) :
    HasFDerivAt (fun x => arctan (f x)) ((1 / (1 + f x ^ 2)) • f') x :=
  (hasDerivAt_arctan (f x)).comp_hasFDerivAt x hf


theorem HasFDerivWithinAt.arctan (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (fun x => arctan (f x)) ((1 / (1 + f x ^ 2)) • f') s x :=
  (hasDerivAt_arctan (f x)).comp_hasFDerivWithinAt x hf


theorem fderivWithin_arctan (hf : DifferentiableWithinAt ℝ f s x) (hxs : UniqueDiffWithinAt ℝ s x) :
    fderivWithin ℝ (fun x => arctan (f x)) s x = (1 / (1 + f x ^ 2)) • fderivWithin ℝ f s x :=
  hf.hasFDerivWithinAt.arctan.fderivWithin hxs


@[simp]
theorem fderiv_arctan (hc : DifferentiableAt ℝ f x) :
    fderiv ℝ (fun x => arctan (f x)) x = (1 / (1 + f x ^ 2)) • fderiv ℝ f x :=
  hc.hasFDerivAt.arctan.fderiv


theorem DifferentiableWithinAt.arctan (hf : DifferentiableWithinAt ℝ f s x) :
    DifferentiableWithinAt ℝ (fun x => Real.arctan (f x)) s x :=
  hf.hasFDerivWithinAt.arctan.differentiableWithinAt


@[simp]
theorem DifferentiableAt.arctan (hc : DifferentiableAt ℝ f x) :
    DifferentiableAt ℝ (fun x => arctan (f x)) x :=
  hc.hasFDerivAt.arctan.differentiableAt


theorem DifferentiableOn.arctan (hc : DifferentiableOn ℝ f s) :
    DifferentiableOn ℝ (fun x => arctan (f x)) s := fun x h => (hc x h).arctan


@[simp]
theorem Differentiable.arctan (hc : Differentiable ℝ f) : Differentiable ℝ fun x => arctan (f x) :=
  fun x => (hc x).arctan


theorem ContDiffAt.arctan (h : ContDiffAt ℝ n f x) : ContDiffAt ℝ n (fun x => arctan (f x)) x :=
  contDiff_arctan.contDiffAt.comp x h


theorem ContDiff.arctan (h : ContDiff ℝ n f) : ContDiff ℝ n fun x => arctan (f x) :=
  contDiff_arctan.comp h


theorem ContDiffWithinAt.arctan (h : ContDiffWithinAt ℝ n f s x) :
    ContDiffWithinAt ℝ n (fun x => arctan (f x)) s x :=
  contDiff_arctan.comp_contDiffWithinAt h


theorem ContDiffOn.arctan (h : ContDiffOn ℝ n f s) : ContDiffOn ℝ n (fun x => arctan (f x)) s :=
  contDiff_arctan.comp_contDiffOn h


