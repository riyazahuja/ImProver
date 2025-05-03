/-- If a complex function is differentiable at a real point, then the induced real function is also
differentiable at this point, with a derivative equal to the real part of the complex derivative. -/
theorem HasStrictDerivAt.real_of_complex (h : HasStrictDerivAt e e' z) :
    HasStrictDerivAt (fun x : ℝ => (e x).re) e'.re z := by
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasStrictDerivAt e e' ↑z
    ⊢ HasStrictDerivAt (fun x => (e ↑x).re) e'.re z
  -/
  have A : HasStrictFDerivAt ((↑) : ℝ → ℂ) ofRealCLM z := ofRealCLM.hasStrictFDerivAt
  have B :
    HasStrictFDerivAt e ((ContinuousLinearMap.smulRight 1 e' : ℂ →L[ℂ] ℂ).restrictScalars ℝ)
      (ofRealCLM z) :=
    h.hasStrictFDerivAt.restrictScalars ℝ
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasStrictDerivAt e e' ↑z
    A : HasStrictFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasStrictFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousL …
    ⊢ HasStrictDerivAt (fun x => (e ↑x).re) e'.re z
  -/
  have C : HasStrictFDerivAt re reCLM (e (ofRealCLM z)) := reCLM.hasStrictFDerivAt
  -- Porting note: this should be by:
  -- simpa using (C.comp z (B.comp z A)).hasStrictDerivAt
  -- but for some reason simp can not use `ContinuousLinearMap.comp_apply`
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasStrictDerivAt e e' ↑z
    A : HasStrictFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasStrictFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousL …
    C : HasStrictFDerivAt Complex.re Complex.reCLM (e (Complex.ofRealCLM z))
    ⊢ HasStrictDerivAt (fun x => (e ↑x).re) e'.re z
  -/
  convert (C.comp z (B.comp z A)).hasStrictDerivAt
  /-
    case h.e'_9
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasStrictDerivAt e e' ↑z
    A : HasStrictFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasStrictFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousL …
    C : HasStrictFDerivAt Complex.re Complex.reCLM (e (Complex.ofRealCLM z))
    ⊢ Eq e'.re ((Complex.reCLM.comp ((ContinuousLinearMap.restrictScalars Real (Co …
  -/
  rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.comp_apply]
  /-
    case h.e'_9
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasStrictDerivAt e e' ↑z
    A : HasStrictFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasStrictFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousL …
    C : HasStrictFDerivAt Complex.re Complex.reCLM (e (Complex.ofRealCLM z))
    ⊢ Eq e'.re (Complex.reCLM ((ContinuousLinearMap.restrictScalars Real (Continuo …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If a complex function `e` is differentiable at a real point, then the function `ℝ → ℝ` given by
the real part of `e` is also differentiable at this point, with a derivative equal to the real part
of the complex derivative. -/
theorem HasDerivAt.real_of_complex (h : HasDerivAt e e' z) :
    HasDerivAt (fun x : ℝ => (e x).re) e'.re z := by
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasDerivAt e e' ↑z
    ⊢ HasDerivAt (fun x => (e ↑x).re) e'.re z
  -/
  have A : HasFDerivAt ((↑) : ℝ → ℂ) ofRealCLM z := ofRealCLM.hasFDerivAt
  have B :
    HasFDerivAt e ((ContinuousLinearMap.smulRight 1 e' : ℂ →L[ℂ] ℂ).restrictScalars ℝ)
      (ofRealCLM z) :=
    h.hasFDerivAt.restrictScalars ℝ
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasDerivAt e e' ↑z
    A : HasFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousLinearM …
    ⊢ HasDerivAt (fun x => (e ↑x).re) e'.re z
  -/
  have C : HasFDerivAt re reCLM (e (ofRealCLM z)) := reCLM.hasFDerivAt
  -- Porting note: this should be by:
  -- simpa using (C.comp z (B.comp z A)).hasStrictDerivAt
  -- but for some reason simp can not use `ContinuousLinearMap.comp_apply`
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasDerivAt e e' ↑z
    A : HasFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousLinearM …
    C : HasFDerivAt Complex.re Complex.reCLM (e (Complex.ofRealCLM z))
    ⊢ HasDerivAt (fun x => (e ↑x).re) e'.re z
  -/
  convert (C.comp z (B.comp z A)).hasDerivAt
  /-
    case h.e'_9
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasDerivAt e e' ↑z
    A : HasFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousLinearM …
    C : HasFDerivAt Complex.re Complex.reCLM (e (Complex.ofRealCLM z))
    ⊢ Eq e'.re ((Complex.reCLM.comp ((ContinuousLinearMap.restrictScalars Real (Co …
  -/
  rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.comp_apply]
  /-
    case h.e'_9
    e : Complex → Complex
    e' : Complex
    z : Real
    h : HasDerivAt e e' ↑z
    A : HasFDerivAt Complex.ofReal Complex.ofRealCLM z
    B : HasFDerivAt e (ContinuousLinearMap.restrictScalars Real (ContinuousLinearM …
    C : HasFDerivAt Complex.re Complex.reCLM (e (Complex.ofRealCLM z))
    ⊢ Eq e'.re (Complex.reCLM ((ContinuousLinearMap.restrictScalars Real (Continuo …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ContDiffAt.real_of_complex {n : WithTop ℕ∞} (h : ContDiffAt ℂ n e z) :
    ContDiffAt ℝ n (fun x : ℝ => (e x).re) z := by
  /-
    e : Complex → Complex
    z : Real
    n : WithTop ENat
    h : ContDiffAt Complex n e ↑z
    ⊢ ContDiffAt Real n (fun x => (e ↑x).re) z
  -/
  have A : ContDiffAt ℝ n ((↑) : ℝ → ℂ) z := ofRealCLM.contDiff.contDiffAt
  /-
    e : Complex → Complex
    z : Real
    n : WithTop ENat
    h : ContDiffAt Complex n e ↑z
    A : ContDiffAt Real n Complex.ofReal z
    ⊢ ContDiffAt Real n (fun x => (e ↑x).re) z
  -/
  have B : ContDiffAt ℝ n e z := h.restrict_scalars ℝ
  /-
    e : Complex → Complex
    z : Real
    n : WithTop ENat
    h : ContDiffAt Complex n e ↑z
    A : ContDiffAt Real n Complex.ofReal z
    B : ContDiffAt Real n e ↑z
    ⊢ ContDiffAt Real n (fun x => (e ↑x).re) z
  -/
  have C : ContDiffAt ℝ n re (e z) := reCLM.contDiff.contDiffAt
  /-
    e : Complex → Complex
    z : Real
    n : WithTop ENat
    h : ContDiffAt Complex n e ↑z
    A : ContDiffAt Real n Complex.ofReal z
    B : ContDiffAt Real n e ↑z
    C : ContDiffAt Real n Complex.re (e ↑z)
    ⊢ ContDiffAt Real n (fun x => (e ↑x).re) z
  -/
  exact C.comp z (B.comp z A)
  /-
    🎉 no goals
  -/


theorem ContDiff.real_of_complex {n : WithTop ℕ∞} (h : ContDiff ℂ n e) :
    ContDiff ℝ n fun x : ℝ => (e x).re :=
  contDiff_iff_contDiffAt.2 fun _ => h.contDiffAt.real_of_complex


theorem HasStrictDerivAt.complexToReal_fderiv' {f : ℂ → E} {x : ℂ} {f' : E}
    (h : HasStrictDerivAt f f' x) :
    HasStrictFDerivAt f (reCLM.smulRight f' + I • imCLM.smulRight f') x := by
  simpa only [Complex.restrictScalars_one_smulRight'] using
    h.hasStrictFDerivAt.restrictScalars ℝ


theorem HasDerivAt.complexToReal_fderiv' {f : ℂ → E} {x : ℂ} {f' : E} (h : HasDerivAt f f' x) :
    HasFDerivAt f (reCLM.smulRight f' + I • imCLM.smulRight f') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Complex → E
    x : Complex
    f' : E
    h : HasDerivAt f f' x
    ⊢ HasFDerivAt f (HAdd.hAdd (Complex.reCLM.smulRight f') (HSMul.hSMul Complex.I …
  -/
  simpa only [Complex.restrictScalars_one_smulRight'] using h.hasFDerivAt.restrictScalars ℝ
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.complexToReal_fderiv' {f : ℂ → E} {s : Set ℂ} {x : ℂ} {f' : E}
    (h : HasDerivWithinAt f f' s x) :
    HasFDerivWithinAt f (reCLM.smulRight f' + I • imCLM.smulRight f') s x := by
  simpa only [Complex.restrictScalars_one_smulRight'] using
    h.hasFDerivWithinAt.restrictScalars ℝ


theorem HasStrictDerivAt.complexToReal_fderiv {f : ℂ → ℂ} {f' x : ℂ} (h : HasStrictDerivAt f f' x) :
    HasStrictFDerivAt f (f' • (1 : ℂ →L[ℝ] ℂ)) x := by
  /-
    f : Complex → Complex
    f' x : Complex
    h : HasStrictDerivAt f f' x
    ⊢ HasStrictFDerivAt f (HSMul.hSMul f' 1) x
  -/
  simpa only [Complex.restrictScalars_one_smulRight] using h.hasStrictFDerivAt.restrictScalars ℝ
  /-
    🎉 no goals
  -/


theorem HasDerivAt.complexToReal_fderiv {f : ℂ → ℂ} {f' x : ℂ} (h : HasDerivAt f f' x) :
    HasFDerivAt f (f' • (1 : ℂ →L[ℝ] ℂ)) x := by
  /-
    f : Complex → Complex
    f' x : Complex
    h : HasDerivAt f f' x
    ⊢ HasFDerivAt f (HSMul.hSMul f' 1) x
  -/
  simpa only [Complex.restrictScalars_one_smulRight] using h.hasFDerivAt.restrictScalars ℝ
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.complexToReal_fderiv {f : ℂ → ℂ} {s : Set ℂ} {f' x : ℂ}
    (h : HasDerivWithinAt f f' s x) : HasFDerivWithinAt f (f' • (1 : ℂ →L[ℝ] ℂ)) s x := by
  /-
    f : Complex → Complex
    s : Set Complex
    f' x : Complex
    h : HasDerivWithinAt f f' s x
    ⊢ HasFDerivWithinAt f (HSMul.hSMul f' 1) s x
  -/
  simpa only [Complex.restrictScalars_one_smulRight] using h.hasFDerivWithinAt.restrictScalars ℝ
  /-
    🎉 no goals
  -/


/-- If a complex function `e` is differentiable at a real point, then its restriction to `ℝ` is
differentiable there as a function `ℝ → ℂ`, with the same derivative. -/
theorem HasDerivAt.comp_ofReal (hf : HasDerivAt e e' ↑z) : HasDerivAt (fun y : ℝ => e ↑y) e' z := by
  /-
    e : Complex → Complex
    e' : Complex
    z : Real
    hf : HasDerivAt e e' ↑z
    ⊢ HasDerivAt (fun y => e ↑y) e' z
  -/
  simpa only [ofRealCLM_apply, ofReal_one, mul_one] using hf.comp z ofRealCLM.hasDerivAt
  /-
    🎉 no goals
  -/


/-- If a function `f : ℝ → ℝ` is differentiable at a (real) point `x`, then it is also
differentiable as a function `ℝ → ℂ`. -/
theorem HasDerivAt.ofReal_comp {f : ℝ → ℝ} {u : ℝ} (hf : HasDerivAt f u z) :
    HasDerivAt (fun y : ℝ => ↑(f y) : ℝ → ℂ) u z := by
  simpa only [ofRealCLM_apply, ofReal_one, real_smul, mul_one] using
    ofRealCLM.hasDerivAt.scomp z hf


