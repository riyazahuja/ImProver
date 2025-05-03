/-- Local homeomorph between `(0, +∞)` and `(0, +∞)` with `toFun = (· ^ 2)` and
`invFun = Real.sqrt`. -/
noncomputable def sqPartialHomeomorph : PartialHomeomorph ℝ ℝ where
  toFun x := x ^ 2
  invFun := (√·)
  source := Ioi 0
  target := Ioi 0
  map_source' _ h := mem_Ioi.2 (pow_pos (mem_Ioi.1 h) _)
  map_target' _ h := mem_Ioi.2 (sqrt_pos.2 h)
  left_inv' _ h := sqrt_sq (le_of_lt h)
  right_inv' _ h := sq_sqrt (le_of_lt h)
  open_source := isOpen_Ioi
  open_target := isOpen_Ioi
  continuousOn_toFun := (continuous_pow 2).continuousOn
  continuousOn_invFun := continuousOn_id.sqrt


theorem deriv_sqrt_aux {x : ℝ} (hx : x ≠ 0) :
    HasStrictDerivAt (√·) (1 / (2 * √x)) x ∧ ∀ n, ContDiffAt ℝ n (√·) x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ And (HasStrictDerivAt (fun x => x.sqrt) (HDiv.hDiv 1 (HMul.hMul 2 x.sqrt)) x …
  -/
  cases' hx.lt_or_lt with hx hx
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt x 0
      ⊢ And (HasStrictDerivAt (fun x => x.sqrt) (HDiv.hDiv 1 (HMul.hMul 2 x.sqrt)) x …
    -/
  · rw [sqrt_eq_zero_of_nonpos hx.le, mul_zero, div_zero]
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt x 0
      ⊢ And (HasStrictDerivAt (fun x => x.sqrt) 0 x) (∀ (n : WithTop ENat), ContDiff …
    -/
    have : (√·) =ᶠ[𝓝 x] fun _ => 0 := (gt_mem_nhds hx).mono fun x hx => sqrt_eq_zero_of_nonpos hx.le
    exact
      ⟨(hasStrictDerivAt_const x (0 : ℝ)).congr_of_eventuallyEq this.symm, fun n =>
        contDiffAt_const.congr_of_eventuallyEq this⟩
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt 0 x
      ⊢ And (HasStrictDerivAt (fun x => x.sqrt) (HDiv.hDiv 1 (HMul.hMul 2 x.sqrt)) x …
    -/
  · have : ↑2 * √x ^ (2 - 1) ≠ 0 := by simp [(sqrt_pos.2 hx).ne', @two_ne_zero ℝ]
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt 0 x
      this : Ne (HMul.hMul 2 (HPow.hPow x.sqrt (HSub.hSub 2 1))) 0
      ⊢ And (HasStrictDerivAt (fun x => x.sqrt) (HDiv.hDiv 1 (HMul.hMul 2 x.sqrt)) x …
    -/
    constructor
      /-
        case inr.left
        x : Real
        hx✝ : Ne x 0
        hx : LT.lt 0 x
        this : Ne (HMul.hMul 2 (HPow.hPow x.sqrt (HSub.hSub 2 1))) 0
        ⊢ HasStrictDerivAt (fun x => x.sqrt) (HDiv.hDiv 1 (HMul.hMul 2 x.sqrt)) x
      -/
    · simpa using sqPartialHomeomorph.hasStrictDerivAt_symm hx this (hasStrictDerivAt_pow 2 _)
      /-
        🎉 no goals
      -/
    · exact fun n => sqPartialHomeomorph.contDiffAt_symm_deriv this hx (hasDerivAt_pow 2 (√x))
        (contDiffAt_id.pow 2)


theorem hasStrictDerivAt_sqrt {x : ℝ} (hx : x ≠ 0) : HasStrictDerivAt (√·) (1 / (2 * √x)) x :=
  (deriv_sqrt_aux hx).1


theorem contDiffAt_sqrt {x : ℝ} {n : WithTop ℕ∞} (hx : x ≠ 0) : ContDiffAt ℝ n (√·) x :=
  (deriv_sqrt_aux hx).2 n


theorem hasDerivAt_sqrt {x : ℝ} (hx : x ≠ 0) : HasDerivAt (√·) (1 / (2 * √x)) x :=
  (hasStrictDerivAt_sqrt hx).hasDerivAt


theorem HasDerivWithinAt.sqrt (hf : HasDerivWithinAt f f' s x) (hx : f x ≠ 0) :
    HasDerivWithinAt (fun y => √(f y)) (f' / (2 * √(f x))) s x := by
  simpa only [(· ∘ ·), div_eq_inv_mul, mul_one] using
    (hasDerivAt_sqrt hx).comp_hasDerivWithinAt x hf


theorem HasDerivAt.sqrt (hf : HasDerivAt f f' x) (hx : f x ≠ 0) :
    HasDerivAt (fun y => √(f y)) (f' / (2 * √(f x))) x := by
  /-
    f : Real → Real
    f' x : Real
    hf : HasDerivAt f f' x
    hx : Ne (f x) 0
    ⊢ HasDerivAt (fun y => (f y).sqrt) (HDiv.hDiv f' (HMul.hMul 2 (f x).sqrt)) x
  -/
  simpa only [(· ∘ ·), div_eq_inv_mul, mul_one] using (hasDerivAt_sqrt hx).comp x hf
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.sqrt (hf : HasStrictDerivAt f f' x) (hx : f x ≠ 0) :
    HasStrictDerivAt (fun t => √(f t)) (f' / (2 * √(f x))) x := by
  /-
    f : Real → Real
    f' x : Real
    hf : HasStrictDerivAt f f' x
    hx : Ne (f x) 0
    ⊢ HasStrictDerivAt (fun t => (f t).sqrt) (HDiv.hDiv f' (HMul.hMul 2 (f x).sqrt …
  -/
  simpa only [(· ∘ ·), div_eq_inv_mul, mul_one] using (hasStrictDerivAt_sqrt hx).comp x hf
  /-
    🎉 no goals
  -/


theorem derivWithin_sqrt (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0)
    (hxs : UniqueDiffWithinAt ℝ s x) :
    derivWithin (fun x => √(f x)) s x = derivWithin f s x / (2 * √(f x)) :=
  (hf.hasDerivWithinAt.sqrt hx).derivWithin hxs


@[simp]
theorem deriv_sqrt (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    deriv (fun x => √(f x)) x = deriv f x / (2 * √(f x)) :=
  (hf.hasDerivAt.sqrt hx).deriv


theorem HasFDerivAt.sqrt (hf : HasFDerivAt f f' x) (hx : f x ≠ 0) :
    HasFDerivAt (fun y => √(f y)) ((1 / (2 * √(f x))) • f') x :=
  (hasDerivAt_sqrt hx).comp_hasFDerivAt x hf


theorem HasStrictFDerivAt.sqrt (hf : HasStrictFDerivAt f f' x) (hx : f x ≠ 0) :
    HasStrictFDerivAt (fun y => √(f y)) ((1 / (2 * √(f x))) • f') x :=
  (hasStrictDerivAt_sqrt hx).comp_hasStrictFDerivAt x hf


theorem HasFDerivWithinAt.sqrt (hf : HasFDerivWithinAt f f' s x) (hx : f x ≠ 0) :
    HasFDerivWithinAt (fun y => √(f y)) ((1 / (2 * √(f x))) • f') s x :=
  (hasDerivAt_sqrt hx).comp_hasFDerivWithinAt x hf


theorem DifferentiableWithinAt.sqrt (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0) :
    DifferentiableWithinAt ℝ (fun y => √(f y)) s x :=
  (hf.hasFDerivWithinAt.sqrt hx).differentiableWithinAt


theorem DifferentiableAt.sqrt (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    DifferentiableAt ℝ (fun y => √(f y)) x :=
  (hf.hasFDerivAt.sqrt hx).differentiableAt


theorem DifferentiableOn.sqrt (hf : DifferentiableOn ℝ f s) (hs : ∀ x ∈ s, f x ≠ 0) :
    DifferentiableOn ℝ (fun y => √(f y)) s := fun x hx => (hf x hx).sqrt (hs x hx)


theorem Differentiable.sqrt (hf : Differentiable ℝ f) (hs : ∀ x, f x ≠ 0) :
    Differentiable ℝ fun y => √(f y) := fun x => (hf x).sqrt (hs x)


theorem fderivWithin_sqrt (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0)
    (hxs : UniqueDiffWithinAt ℝ s x) :
    fderivWithin ℝ (fun x => √(f x)) s x = (1 / (2 * √(f x))) • fderivWithin ℝ f s x :=
  (hf.hasFDerivWithinAt.sqrt hx).fderivWithin hxs


@[simp]
theorem fderiv_sqrt (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    fderiv ℝ (fun x => √(f x)) x = (1 / (2 * √(f x))) • fderiv ℝ f x :=
  (hf.hasFDerivAt.sqrt hx).fderiv


theorem ContDiffAt.sqrt (hf : ContDiffAt ℝ n f x) (hx : f x ≠ 0) :
    ContDiffAt ℝ n (fun y => √(f y)) x :=
  (contDiffAt_sqrt hx).comp x hf


theorem ContDiffWithinAt.sqrt (hf : ContDiffWithinAt ℝ n f s x) (hx : f x ≠ 0) :
    ContDiffWithinAt ℝ n (fun y => √(f y)) s x :=
  (contDiffAt_sqrt hx).comp_contDiffWithinAt x hf


theorem ContDiffOn.sqrt (hf : ContDiffOn ℝ n f s) (hs : ∀ x ∈ s, f x ≠ 0) :
    ContDiffOn ℝ n (fun y => √(f y)) s := fun x hx => (hf x hx).sqrt (hs x hx)


theorem ContDiff.sqrt (hf : ContDiff ℝ n f) (h : ∀ x, f x ≠ 0) : ContDiff ℝ n fun y => √(f y) :=
  contDiff_iff_contDiffAt.2 fun x => hf.contDiffAt.sqrt (h x)


