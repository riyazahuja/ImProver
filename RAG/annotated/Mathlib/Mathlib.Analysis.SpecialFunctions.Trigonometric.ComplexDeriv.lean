theorem hasStrictDerivAt_tan {x : ℂ} (h : cos x ≠ 0) : HasStrictDerivAt tan (1 / cos x ^ 2) x := by
  /-
    x : Complex
    h : Ne (Complex.cos x) 0
    ⊢ HasStrictDerivAt Complex.tan (HDiv.hDiv 1 (HPow.hPow (Complex.cos x) 2)) x
  -/
  convert (hasStrictDerivAt_sin x).div (hasStrictDerivAt_cos x) h using 1
  /-
    case h.e'_9
    x : Complex
    h : Ne (Complex.cos x) 0
    ⊢ Eq (HDiv.hDiv 1 (HPow.hPow (Complex.cos x) 2)) (HDiv.hDiv (HSub.hSub (HMul.h …
  -/
  rw_mod_cast [← sin_sq_add_cos_sq x]
  /-
    case h.e'_9
    x : Complex
    h : Not (Eq (Complex.cos x) 0)
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HPow.hPow (Complex.sin x) 2) (HPow.hPow (Complex.c …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem hasDerivAt_tan {x : ℂ} (h : cos x ≠ 0) : HasDerivAt tan (1 / cos x ^ 2) x :=
  (hasStrictDerivAt_tan h).hasDerivAt


theorem tendsto_abs_tan_of_cos_eq_zero {x : ℂ} (hx : cos x = 0) :
    Tendsto (fun x => abs (tan x)) (𝓝[≠] x) atTop := by
  /-
    x : Complex
    hx : Eq (Complex.cos x) 0
    ⊢ Filter.Tendsto (fun x => Complex.abs (Complex.tan x)) (nhdsWithin x (HasComp …
  -/
  simp only [tan_eq_sin_div_cos, ← norm_eq_abs, norm_div]
  /-
    x : Complex
    hx : Eq (Complex.cos x) 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (Norm.norm (Complex.sin x)) (Norm.norm (C …
  -/
  have A : sin x ≠ 0 := fun h => by simpa [*, sq] using sin_sq_add_cos_sq x
  have B : Tendsto cos (𝓝[≠] x) (𝓝[≠] 0) :=
    hx ▸ (hasDerivAt_cos x).tendsto_punctured_nhds (neg_ne_zero.2 A)
  exact continuous_sin.continuousWithinAt.norm.mul_atTop (norm_pos_iff.2 A)
    (tendsto_norm_nhdsNE_zero.comp B).inv_tendsto_nhdsGT_zero


theorem tendsto_abs_tan_atTop (k : ℤ) :
    Tendsto (fun x => abs (tan x)) (𝓝[≠] ((2 * k + 1) * π / 2 : ℂ)) atTop :=
  tendsto_abs_tan_of_cos_eq_zero <| cos_eq_zero_iff.2 ⟨k, rfl⟩


@[simp]
theorem continuousAt_tan {x : ℂ} : ContinuousAt tan x ↔ cos x ≠ 0 := by
  /-
    x : Complex
    ⊢ Iff (ContinuousAt Complex.tan x) (Ne (Complex.cos x) 0)
  -/
  refine ⟨fun hc h₀ => ?_, fun h => (hasDerivAt_tan h).continuousAt⟩
  exact not_tendsto_nhds_of_tendsto_atTop (tendsto_abs_tan_of_cos_eq_zero h₀) _
    (hc.norm.tendsto.mono_left inf_le_left)


@[simp]
theorem differentiableAt_tan {x : ℂ} : DifferentiableAt ℂ tan x ↔ cos x ≠ 0 :=
  ⟨fun h => continuousAt_tan.1 h.continuousAt, fun h => (hasDerivAt_tan h).differentiableAt⟩


@[simp]
theorem deriv_tan (x : ℂ) : deriv tan x = 1 / cos x ^ 2 :=
  if h : cos x = 0 then by
    /-
      x : Complex
      h : Eq (Complex.cos x) 0
      ⊢ Eq (deriv Complex.tan x) (HDiv.hDiv 1 (HPow.hPow (Complex.cos x) 2))
    -/
    have : ¬DifferentiableAt ℂ tan x := mt differentiableAt_tan.1 (Classical.not_not.2 h)
    /-
      x : Complex
      h : Eq (Complex.cos x) 0
      this : Not (DifferentiableAt Complex Complex.tan x)
      ⊢ Eq (deriv Complex.tan x) (HDiv.hDiv 1 (HPow.hPow (Complex.cos x) 2))
    -/
    simp [deriv_zero_of_not_differentiableAt this, h, sq]
    /-
      🎉 no goals
    -/
  else (hasDerivAt_tan h).deriv


@[simp]
theorem contDiffAt_tan {x : ℂ} {n : WithTop ℕ∞} : ContDiffAt ℂ n tan x ↔ cos x ≠ 0 :=
  ⟨fun h => continuousAt_tan.1 h.continuousAt, contDiff_sin.contDiffAt.div contDiff_cos.contDiffAt⟩


