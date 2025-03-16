theorem contDiffAt_abs {x : ℝ} (hx : x ≠ 0) : ContDiffAt ℝ n (|·|) x := contDiffAt_norm ℝ hx


theorem ContDiffAt.abs (hf : ContDiffAt ℝ n f x) (h₀ : f x ≠ 0) :
    ContDiffAt ℝ n (fun x ↦ |f x|) x := hf.norm ℝ h₀


theorem contDiffWithinAt_abs {x : ℝ} (hx : x ≠ 0) (s : Set ℝ) :
    ContDiffWithinAt ℝ n (|·|) s x := (contDiffAt_abs hx).contDiffWithinAt


theorem ContDiffWithinAt.abs (hf : ContDiffWithinAt ℝ n f s x) (h₀ : f x ≠ 0) :
    ContDiffWithinAt ℝ n (fun y ↦ |f y|) s x :=
  (contDiffAt_abs h₀).comp_contDiffWithinAt x hf


theorem contDiffOn_abs {s : Set ℝ} (hs : ∀ x ∈ s, x ≠ 0) :
    ContDiffOn ℝ n (|·|) s := fun x hx ↦ contDiffWithinAt_abs (hs x hx) s


theorem ContDiffOn.abs (hf : ContDiffOn ℝ n f s) (h₀ : ∀ x ∈ s, f x ≠ 0) :
    ContDiffOn ℝ n (fun y ↦ |f y|) s := fun x hx ↦ (hf x hx).abs (h₀ x hx)


theorem ContDiff.abs (hf : ContDiff ℝ n f) (h₀ : ∀ x, f x ≠ 0) : ContDiff ℝ n fun y ↦ |f y| :=
  contDiff_iff_contDiffAt.2 fun x ↦ hf.contDiffAt.abs (h₀ x)


theorem hasStrictDerivAt_abs_neg {x : ℝ} (hx : x < 0) :
    HasStrictDerivAt (|·|) (-1) x :=
  (hasStrictDerivAt_neg x).congr_of_eventuallyEq <|
    EqOn.eventuallyEq_of_mem (fun _ hy ↦ (abs_of_neg (mem_Iio.1 hy)).symm) (Iio_mem_nhds hx)


theorem hasDerivAt_abs_neg {x : ℝ} (hx : x < 0) :
    HasDerivAt (|·|) (-1) x := (hasStrictDerivAt_abs_neg hx).hasDerivAt


theorem hasStrictDerivAt_abs_pos {x : ℝ} (hx : 0 < x) :
    HasStrictDerivAt (|·|) 1 x :=
  (hasStrictDerivAt_id x).congr_of_eventuallyEq <|
    EqOn.eventuallyEq_of_mem (fun _ hy ↦ (abs_of_pos (mem_Iio.1 hy)).symm) (Ioi_mem_nhds hx)


theorem hasDerivAt_abs_pos {x : ℝ} (hx : 0 < x) :
    HasDerivAt (|·|) 1 x := (hasStrictDerivAt_abs_pos hx).hasDerivAt


theorem hasStrictDerivAt_abs {x : ℝ} (hx : x ≠ 0) :
    HasStrictDerivAt (|·|) (SignType.sign x : ℝ) x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ HasStrictDerivAt (fun x => abs x) (↑(SignType.sign x)) x
  -/
  obtain hx | hx := hx.lt_or_lt
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt x 0
      ⊢ HasStrictDerivAt (fun x => abs x) (↑(SignType.sign x)) x
    -/
  · simpa [hx] using hasStrictDerivAt_abs_neg hx
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt 0 x
      ⊢ HasStrictDerivAt (fun x => abs x) (↑(SignType.sign x)) x
    -/
  · simpa [hx] using hasStrictDerivAt_abs_pos hx
    /-
      🎉 no goals
    -/


theorem hasDerivAt_abs {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (|·|) (SignType.sign x : ℝ) x := (hasStrictDerivAt_abs hx).hasDerivAt


theorem HasStrictFDerivAt.abs_of_neg (hf : HasStrictFDerivAt f f' x)
    (h₀ : f x < 0) : HasStrictFDerivAt (fun x ↦ |f x|) (-f') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : LT.lt (f x) 0
    ⊢ HasStrictFDerivAt (fun x => abs (f x)) (Neg.neg f') x
  -/
  convert (hasStrictDerivAt_abs_neg h₀).hasStrictFDerivAt.comp x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : LT.lt (f x) 0
    ⊢ Eq (Neg.neg f') ((ContinuousLinearMap.smulRight 1 (-1)).comp f')
  -/
  ext y
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : LT.lt (f x) 0
    y : E
    ⊢ Eq ((Neg.neg f') y) (((ContinuousLinearMap.smulRight 1 (-1)).comp f') y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.abs_of_neg (hf : HasFDerivAt f f' x)
    (h₀ : f x < 0) : HasFDerivAt (fun x ↦ |f x|) (-f') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : LT.lt (f x) 0
    ⊢ HasFDerivAt (fun x => abs (f x)) (Neg.neg f') x
  -/
  convert (hasDerivAt_abs_neg h₀).hasFDerivAt.comp x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : LT.lt (f x) 0
    ⊢ Eq (Neg.neg f') ((ContinuousLinearMap.smulRight 1 (-1)).comp f')
  -/
  ext y
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : LT.lt (f x) 0
    y : E
    ⊢ Eq ((Neg.neg f') y) (((ContinuousLinearMap.smulRight 1 (-1)).comp f') y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.abs_of_pos (hf : HasStrictFDerivAt f f' x)
    (h₀ : 0 < f x) : HasStrictFDerivAt (fun x ↦ |f x|) f' x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : LT.lt 0 (f x)
    ⊢ HasStrictFDerivAt (fun x => abs (f x)) f' x
  -/
  convert (hasStrictDerivAt_abs_pos h₀).hasStrictFDerivAt.comp x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : LT.lt 0 (f x)
    ⊢ Eq f' ((ContinuousLinearMap.smulRight 1 1).comp f')
  -/
  ext y
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : LT.lt 0 (f x)
    y : E
    ⊢ Eq (f' y) (((ContinuousLinearMap.smulRight 1 1).comp f') y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.abs_of_pos (hf : HasFDerivAt f f' x)
    (h₀ : 0 < f x) : HasFDerivAt (fun x ↦ |f x|) f' x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : LT.lt 0 (f x)
    ⊢ HasFDerivAt (fun x => abs (f x)) f' x
  -/
  convert (hasDerivAt_abs_pos h₀).hasFDerivAt.comp x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : LT.lt 0 (f x)
    ⊢ Eq f' ((ContinuousLinearMap.smulRight 1 1).comp f')
  -/
  ext y
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : LT.lt 0 (f x)
    y : E
    ⊢ Eq (f' y) (((ContinuousLinearMap.smulRight 1 1).comp f') y)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.abs (hf : HasStrictFDerivAt f f' x)
    (h₀ : f x ≠ 0) : HasStrictFDerivAt (fun x ↦ |f x|) ((SignType.sign (f x) : ℝ) • f') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : Ne (f x) 0
    ⊢ HasStrictFDerivAt (fun x => _root_.abs (f x)) (HSMul.hSMul (↑(SignType.sign  …
  -/
  convert (hasStrictDerivAt_abs h₀).hasStrictFDerivAt.comp x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : Ne (f x) 0
    ⊢ Eq (HSMul.hSMul (↑(SignType.sign (f x))) f') ((ContinuousLinearMap.smulRight …
  -/
  ext y
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasStrictFDerivAt f f' x
    h₀ : Ne (f x) 0
    y : E
    ⊢ Eq ((HSMul.hSMul (↑(SignType.sign (f x))) f') y) (((ContinuousLinearMap.smul …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.abs (hf : HasFDerivAt f f' x)
    (h₀ : f x ≠ 0) : HasFDerivAt (fun x ↦ |f x|) ((SignType.sign (f x) : ℝ) • f') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : Ne (f x) 0
    ⊢ HasFDerivAt (fun x => _root_.abs (f x)) (HSMul.hSMul (↑(SignType.sign (f x)) …
  -/
  convert (hasDerivAt_abs h₀).hasFDerivAt.comp x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : Ne (f x) 0
    ⊢ Eq (HSMul.hSMul (↑(SignType.sign (f x))) f') ((ContinuousLinearMap.smulRight …
  -/
  ext y
  /-
    case h.e'_12.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    h₀ : Ne (f x) 0
    y : E
    ⊢ Eq ((HSMul.hSMul (↑(SignType.sign (f x))) f') y) (((ContinuousLinearMap.smul …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem hasDerivWithinAt_abs_neg (s : Set ℝ) {x : ℝ} (hx : x < 0) :
    HasDerivWithinAt (|·|) (-1) s x := (hasDerivAt_abs_neg hx).hasDerivWithinAt


theorem hasDerivWithinAt_abs_pos (s : Set ℝ) {x : ℝ} (hx : 0 < x) :
    HasDerivWithinAt (|·|) 1 s x := (hasDerivAt_abs_pos hx).hasDerivWithinAt


theorem hasDerivWithinAt_abs (s : Set ℝ) {x : ℝ} (hx : x ≠ 0) :
    HasDerivWithinAt (|·|) (SignType.sign x : ℝ) s x := (hasDerivAt_abs hx).hasDerivWithinAt


theorem HasFDerivWithinAt.abs_of_neg (hf : HasFDerivWithinAt f f' s x)
    (h₀ : f x < 0) : HasFDerivWithinAt (fun x ↦ |f x|) (-f') s x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    x : E
    hf : HasFDerivWithinAt f f' s x
    h₀ : LT.lt (f x) 0
    ⊢ HasFDerivWithinAt (fun x => abs (f x)) (Neg.neg f') s x
  -/
  convert (hasDerivAt_abs_neg h₀).comp_hasFDerivWithinAt x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    x : E
    hf : HasFDerivWithinAt f f' s x
    h₀ : LT.lt (f x) 0
    ⊢ Eq (Neg.neg f') (HSMul.hSMul (-1) f')
  -/
  simp
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.abs_of_pos (hf : HasFDerivWithinAt f f' s x)
    (h₀ : 0 < f x) : HasFDerivWithinAt (fun x ↦ |f x|) f' s x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    x : E
    hf : HasFDerivWithinAt f f' s x
    h₀ : LT.lt 0 (f x)
    ⊢ HasFDerivWithinAt (fun x => abs (f x)) f' s x
  -/
  convert (hasDerivAt_abs_pos h₀).comp_hasFDerivWithinAt x hf using 1
  /-
    case h.e'_12
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    x : E
    hf : HasFDerivWithinAt f f' s x
    h₀ : LT.lt 0 (f x)
    ⊢ Eq f' (HSMul.hSMul 1 f')
  -/
  simp
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.abs (hf : HasFDerivWithinAt f f' s x)
    (h₀ : f x ≠ 0) : HasFDerivWithinAt (fun x ↦ |f x|) ((SignType.sign (f x) : ℝ) • f') s x :=
  (hasDerivAt_abs h₀).comp_hasFDerivWithinAt x hf


theorem differentiableAt_abs_neg {x : ℝ} (hx : x < 0) :
    DifferentiableAt ℝ (|·|) x := (hasDerivAt_abs_neg hx).differentiableAt


theorem differentiableAt_abs_pos {x : ℝ} (hx : 0 < x) :
    DifferentiableAt ℝ (|·|) x := (hasDerivAt_abs_pos hx).differentiableAt


theorem differentiableAt_abs {x : ℝ} (hx : x ≠ 0) :
    DifferentiableAt ℝ (|·|) x := (hasDerivAt_abs hx).differentiableAt


theorem DifferentiableAt.abs_of_neg (hf : DifferentiableAt ℝ f x) (h₀ : f x < 0) :
    DifferentiableAt ℝ (fun x ↦ |f x|) x := (differentiableAt_abs_neg h₀).comp x hf


theorem DifferentiableAt.abs_of_pos (hf : DifferentiableAt ℝ f x) (h₀ : 0 < f x) :
    DifferentiableAt ℝ (fun x ↦ |f x|) x := (differentiableAt_abs_pos h₀).comp x hf


theorem DifferentiableAt.abs (hf : DifferentiableAt ℝ f x) (h₀ : f x ≠ 0) :
    DifferentiableAt ℝ (fun x ↦ |f x|) x := (differentiableAt_abs h₀).comp x hf


theorem differentiableWithinAt_abs_neg (s : Set ℝ) {x : ℝ} (hx : x < 0) :
    DifferentiableWithinAt ℝ (|·|) s x := (differentiableAt_abs_neg hx).differentiableWithinAt


theorem differentiableWithinAt_abs_pos (s : Set ℝ) {x : ℝ} (hx : 0 < x) :
    DifferentiableWithinAt ℝ (|·|) s x := (differentiableAt_abs_pos hx).differentiableWithinAt


theorem differentiableWithinAt_abs (s : Set ℝ) {x : ℝ} (hx : x ≠ 0) :
    DifferentiableWithinAt ℝ (|·|) s x := (differentiableAt_abs hx).differentiableWithinAt


theorem DifferentiableWithinAt.abs_of_neg (hf : DifferentiableWithinAt ℝ f s x) (h₀ : f x < 0) :
    DifferentiableWithinAt ℝ (fun x ↦ |f x|) s x :=
  (differentiableAt_abs_neg h₀).comp_differentiableWithinAt x hf


theorem DifferentiableWithinAt.abs_of_pos (hf : DifferentiableWithinAt ℝ f s x) (h₀ : 0 < f x) :
    DifferentiableWithinAt ℝ (fun x ↦ |f x|) s x :=
  (differentiableAt_abs_pos h₀).comp_differentiableWithinAt x hf


theorem DifferentiableWithinAt.abs (hf : DifferentiableWithinAt ℝ f s x) (h₀ : f x ≠ 0) :
    DifferentiableWithinAt ℝ (fun x ↦ |f x|) s x :=
  (differentiableAt_abs h₀).comp_differentiableWithinAt x hf


theorem differentiableOn_abs {s : Set ℝ} (hs : ∀ x ∈ s, x ≠ 0) : DifferentiableOn ℝ (|·|) s :=
  fun x hx ↦ differentiableWithinAt_abs s (hs x hx)


theorem DifferentiableOn.abs (hf : DifferentiableOn ℝ f s) (h₀ : ∀ x ∈ s, f x ≠ 0) :
    DifferentiableOn ℝ (fun x ↦ |f x|) s :=
  fun x hx ↦ (hf x hx).abs (h₀ x hx)


theorem Differentiable.abs (hf : Differentiable ℝ f) (h₀ : ∀ x, f x ≠ 0) :
    Differentiable ℝ (fun x ↦ |f x|) := fun x ↦ (hf x).abs (h₀ x)


theorem not_differentiableAt_abs_zero : ¬ DifferentiableAt ℝ (abs : ℝ → ℝ) 0 := by
  /-
    ⊢ Not (DifferentiableAt Real abs 0)
  -/
  intro h
  have h₁ : deriv abs (0 : ℝ) = 1 :=
    (uniqueDiffOn_Ici _ _ Set.left_mem_Ici).eq_deriv _ h.hasDerivAt.hasDerivWithinAt <|
      (hasDerivWithinAt_id _ _).congr_of_mem (fun _ h ↦ abs_of_nonneg h) Set.left_mem_Ici
  have h₂ : deriv abs (0 : ℝ) = -1 :=
    (uniqueDiffOn_Iic _ _ Set.right_mem_Iic).eq_deriv _ h.hasDerivAt.hasDerivWithinAt <|
      (hasDerivWithinAt_neg _ _).congr_of_mem (fun _ h ↦ abs_of_nonpos h) Set.right_mem_Iic
  /-
    h : DifferentiableAt Real abs 0
    h₁ : Eq (deriv abs 0) 1
    h₂ : Eq (deriv abs 0) (-1)
    ⊢ False
  -/
  linarith
  /-
    🎉 no goals
  -/


theorem deriv_abs_neg {x : ℝ} (hx : x < 0) : deriv (|·|) x = -1 := (hasDerivAt_abs_neg hx).deriv


theorem deriv_abs_pos {x : ℝ} (hx : 0 < x) : deriv (|·|) x = 1 := (hasDerivAt_abs_pos hx).deriv


theorem deriv_abs_zero : deriv (|·|) (0 : ℝ) = 0 :=
  deriv_zero_of_not_differentiableAt not_differentiableAt_abs_zero


theorem deriv_abs (x : ℝ) : deriv (|·|) x = SignType.sign x := by
  /-
    x : Real
    ⊢ Eq (deriv (fun x => abs x) x) ↑(SignType.sign x)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      ⊢ Eq (deriv (fun x => abs x) 0) ↑(SignType.sign 0)
    -/
  · simpa using deriv_abs_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      hx : Ne x 0
      ⊢ Eq (deriv (fun x => abs x) x) ↑(SignType.sign x)
    -/
  · simpa [hx] using (hasDerivAt_abs hx).deriv
    /-
      🎉 no goals
    -/

