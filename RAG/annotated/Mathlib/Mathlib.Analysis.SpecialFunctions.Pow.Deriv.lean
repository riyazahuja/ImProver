theorem hasStrictFDerivAt_cpow {p : ℂ × ℂ} (hp : p.1 ∈ slitPlane) :
    HasStrictFDerivAt (fun x : ℂ × ℂ => x.1 ^ x.2)
      ((p.2 * p.1 ^ (p.2 - 1)) • ContinuousLinearMap.fst ℂ ℂ ℂ +
        (p.1 ^ p.2 * log p.1) • ContinuousLinearMap.snd ℂ ℂ ℂ) p := by
  /-
    p : Prod Complex Complex
    hp : Membership.mem Complex.slitPlane p.1
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow x.1 x.2) (HAdd.hAdd (HSMul.hSMul (HMul …
  -/
  have A : p.1 ≠ 0 := slitPlane_ne_zero hp
  have : (fun x : ℂ × ℂ => x.1 ^ x.2) =ᶠ[𝓝 p] fun x => exp (log x.1 * x.2) :=
    ((isOpen_ne.preimage continuous_fst).eventually_mem A).mono fun p hp =>
      cpow_def_of_ne_zero hp _
  /-
    p : Prod Complex Complex
    hp : Membership.mem Complex.slitPlane p.1
    A : Ne p.1 0
    this : (nhds p).EventuallyEq (fun x => HPow.hPow x.1 x.2) fun x => Complex.exp …
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow x.1 x.2) (HAdd.hAdd (HSMul.hSMul (HMul …
  -/
  rw [cpow_sub _ _ A, cpow_one, mul_div_left_comm, mul_smul, mul_smul]
  /-
    p : Prod Complex Complex
    hp : Membership.mem Complex.slitPlane p.1
    A : Ne p.1 0
    this : (nhds p).EventuallyEq (fun x => HPow.hPow x.1 x.2) fun x => Complex.exp …
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow x.1 x.2) (HAdd.hAdd (HSMul.hSMul (HPow …
  -/
  refine HasStrictFDerivAt.congr_of_eventuallyEq ?_ this.symm
  simpa only [cpow_def_of_ne_zero A, div_eq_mul_inv, mul_smul, add_comm, smul_add] using
    ((hasStrictFDerivAt_fst.clog hp).mul hasStrictFDerivAt_snd).cexp


theorem hasStrictFDerivAt_cpow' {x y : ℂ} (hp : x ∈ slitPlane) :
    HasStrictFDerivAt (fun x : ℂ × ℂ => x.1 ^ x.2)
      ((y * x ^ (y - 1)) • ContinuousLinearMap.fst ℂ ℂ ℂ +
        (x ^ y * log x) • ContinuousLinearMap.snd ℂ ℂ ℂ) (x, y) :=
  @hasStrictFDerivAt_cpow (x, y) hp


theorem hasStrictDerivAt_const_cpow {x y : ℂ} (h : x ≠ 0 ∨ y ≠ 0) :
    HasStrictDerivAt (fun y => x ^ y) (x ^ y * log x) y := by
  /-
    x y : Complex
    h : Or (Ne x 0) (Ne y 0)
    ⊢ HasStrictDerivAt (fun y => HPow.hPow x y) (HMul.hMul (HPow.hPow x y) (Comple …
  -/
  rcases em (x = 0) with (rfl | hx)
    /-
      case inl
      y : Complex
      h : Or (Ne 0 0) (Ne y 0)
      ⊢ HasStrictDerivAt (fun y => HPow.hPow 0 y) (HMul.hMul (HPow.hPow 0 y) (Comple …
    -/
  · replace h := h.neg_resolve_left rfl
    /-
      case inl
      y : Complex
      h : Ne y 0
      ⊢ HasStrictDerivAt (fun y => HPow.hPow 0 y) (HMul.hMul (HPow.hPow 0 y) (Comple …
    -/
    rw [log_zero, mul_zero]
    /-
      case inl
      y : Complex
      h : Ne y 0
      ⊢ HasStrictDerivAt (fun y => HPow.hPow 0 y) 0 y
    -/
    refine (hasStrictDerivAt_const y 0).congr_of_eventuallyEq ?_
    /-
      case inl
      y : Complex
      h : Ne y 0
      ⊢ (nhds y).EventuallyEq (fun x => 0) fun y => HPow.hPow 0 y
    -/
    exact (isOpen_ne.eventually_mem h).mono fun y hy => (zero_cpow hy).symm
    /-
      🎉 no goals
    -/
  · simpa only [cpow_def_of_ne_zero hx, mul_one] using
      ((hasStrictDerivAt_id y).const_mul (log x)).cexp


theorem hasFDerivAt_cpow {p : ℂ × ℂ} (hp : p.1 ∈ slitPlane) :
    HasFDerivAt (fun x : ℂ × ℂ => x.1 ^ x.2)
      ((p.2 * p.1 ^ (p.2 - 1)) • ContinuousLinearMap.fst ℂ ℂ ℂ +
        (p.1 ^ p.2 * log p.1) • ContinuousLinearMap.snd ℂ ℂ ℂ) p :=
  (hasStrictFDerivAt_cpow hp).hasFDerivAt


theorem HasStrictFDerivAt.cpow (hf : HasStrictFDerivAt f f' x) (hg : HasStrictFDerivAt g g' x)
    (h0 : f x ∈ slitPlane) : HasStrictFDerivAt (fun x => f x ^ g x)
      ((g x * f x ^ (g x - 1)) • f' + (f x ^ g x * Complex.log (f x)) • g') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : E → Complex
    f' g' : ContinuousLinearMap (RingHom.id Complex) E Complex
    x : E
    hf : HasStrictFDerivAt f f' x
    hg : HasStrictFDerivAt g g' x
    h0 : Membership.mem Complex.slitPlane (f x)
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HSMul.hSMul ( …
  -/
  convert (@hasStrictFDerivAt_cpow ((fun x => (f x, g x)) x) h0).comp x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.const_cpow (hf : HasStrictFDerivAt f f' x) (h0 : c ≠ 0 ∨ f x ≠ 0) :
    HasStrictFDerivAt (fun x => c ^ f x) ((c ^ f x * Complex.log c) • f') x :=
  (hasStrictDerivAt_const_cpow h0).comp_hasStrictFDerivAt x hf


theorem HasFDerivAt.cpow (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x)
    (h0 : f x ∈ slitPlane) : HasFDerivAt (fun x => f x ^ g x)
      ((g x * f x ^ (g x - 1)) • f' + (f x ^ g x * Complex.log (f x)) • g') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f g : E → Complex
    f' g' : ContinuousLinearMap (RingHom.id Complex) E Complex
    x : E
    hf : HasFDerivAt f f' x
    hg : HasFDerivAt g g' x
    h0 : Membership.mem Complex.slitPlane (f x)
    ⊢ HasFDerivAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HSMul.hSMul (HMul.h …
  -/
  convert (@Complex.hasFDerivAt_cpow ((fun x => (f x, g x)) x) h0).comp x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem HasFDerivAt.const_cpow (hf : HasFDerivAt f f' x) (h0 : c ≠ 0 ∨ f x ≠ 0) :
    HasFDerivAt (fun x => c ^ f x) ((c ^ f x * Complex.log c) • f') x :=
  (hasStrictDerivAt_const_cpow h0).hasDerivAt.comp_hasFDerivAt x hf


theorem HasFDerivWithinAt.cpow (hf : HasFDerivWithinAt f f' s x) (hg : HasFDerivWithinAt g g' s x)
    (h0 : f x ∈ slitPlane) : HasFDerivWithinAt (fun x => f x ^ g x)
      ((g x * f x ^ (g x - 1)) • f' + (f x ^ g x * Complex.log (f x)) • g') s x := by
  convert
    (@Complex.hasFDerivAt_cpow ((fun x => (f x, g x)) x) h0).comp_hasFDerivWithinAt x (hf.prod hg)


theorem HasFDerivWithinAt.const_cpow (hf : HasFDerivWithinAt f f' s x) (h0 : c ≠ 0 ∨ f x ≠ 0) :
    HasFDerivWithinAt (fun x => c ^ f x) ((c ^ f x * Complex.log c) • f') s x :=
  (hasStrictDerivAt_const_cpow h0).hasDerivAt.comp_hasFDerivWithinAt x hf


theorem DifferentiableAt.cpow (hf : DifferentiableAt ℂ f x) (hg : DifferentiableAt ℂ g x)
    (h0 : f x ∈ slitPlane) : DifferentiableAt ℂ (fun x => f x ^ g x) x :=
  (hf.hasFDerivAt.cpow hg.hasFDerivAt h0).differentiableAt


theorem DifferentiableAt.const_cpow (hf : DifferentiableAt ℂ f x) (h0 : c ≠ 0 ∨ f x ≠ 0) :
    DifferentiableAt ℂ (fun x => c ^ f x) x :=
  (hf.hasFDerivAt.const_cpow h0).differentiableAt


theorem DifferentiableWithinAt.cpow (hf : DifferentiableWithinAt ℂ f s x)
    (hg : DifferentiableWithinAt ℂ g s x) (h0 : f x ∈ slitPlane) :
    DifferentiableWithinAt ℂ (fun x => f x ^ g x) s x :=
  (hf.hasFDerivWithinAt.cpow hg.hasFDerivWithinAt h0).differentiableWithinAt


theorem DifferentiableWithinAt.const_cpow (hf : DifferentiableWithinAt ℂ f s x)
    (h0 : c ≠ 0 ∨ f x ≠ 0) : DifferentiableWithinAt ℂ (fun x => c ^ f x) s x :=
  (hf.hasFDerivWithinAt.const_cpow h0).differentiableWithinAt


theorem DifferentiableOn.cpow (hf : DifferentiableOn ℂ f s) (hg : DifferentiableOn ℂ g s)
    (h0 : Set.MapsTo f s slitPlane) : DifferentiableOn ℂ (fun x ↦ f x ^ g x) s :=
  fun x hx ↦ (hf x hx).cpow (hg x hx) (h0 hx)


theorem DifferentiableOn.const_cpow (hf : DifferentiableOn ℂ f s)
    (h0 : c ≠ 0 ∨ ∀ x ∈ s, f x ≠ 0) : DifferentiableOn ℂ (fun x ↦ c ^ f x) s :=
  fun x hx ↦ (hf x hx).const_cpow (h0.imp_right fun h ↦ h x hx)


theorem Differentiable.cpow (hf : Differentiable ℂ f) (hg : Differentiable ℂ g)
    (h0 : ∀ x, f x ∈ slitPlane) : Differentiable ℂ (fun x ↦ f x ^ g x) :=
  fun x ↦ (hf x).cpow (hg x) (h0 x)


theorem Differentiable.const_cpow (hf : Differentiable ℂ f)
    (h0 : c ≠ 0 ∨ ∀ x, f x ≠ 0) : Differentiable ℂ (fun x ↦ c ^ f x) :=
  fun x ↦ (hf x).const_cpow (h0.imp_right fun h ↦ h x)


@[fun_prop]
lemma differentiable_const_cpow_of_neZero (z : ℂ) [NeZero z] :
    Differentiable ℂ fun s : ℂ ↦ z ^ s :=
  differentiable_id.const_cpow (.inl <| NeZero.ne z)


@[fun_prop]
lemma differentiableAt_const_cpow_of_neZero (z : ℂ) [NeZero z] (t : ℂ) :
    DifferentiableAt ℂ (fun s : ℂ ↦ z ^ s) t :=
  differentiableAt_id.const_cpow (.inl <| NeZero.ne z)


/-- A private lemma that rewrites the output of lemmas like `HasFDerivAt.cpow` to the form
expected by lemmas like `HasDerivAt.cpow`. -/
private theorem aux : ((g x * f x ^ (g x - 1)) • (1 : ℂ →L[ℂ] ℂ).smulRight f' +
    (f x ^ g x * log (f x)) • (1 : ℂ →L[ℂ] ℂ).smulRight g') 1 =
      g x * f x ^ (g x - 1) * f' + f x ^ g x * log (f x) * g' := by
  simp only [Algebra.id.smul_eq_mul, one_mul, ContinuousLinearMap.one_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.add_apply, Pi.smul_apply,
    ContinuousLinearMap.coe_smul']


nonrec theorem HasStrictDerivAt.cpow (hf : HasStrictDerivAt f f' x) (hg : HasStrictDerivAt g g' x)
    (h0 : f x ∈ slitPlane) : HasStrictDerivAt (fun x => f x ^ g x)
      (g x * f x ^ (g x - 1) * f' + f x ^ g x * Complex.log (f x) * g') x := by
  /-
    f g : Complex → Complex
    f' g' x : Complex
    hf : HasStrictDerivAt f f' x
    hg : HasStrictDerivAt g g' x
    h0 : Membership.mem Complex.slitPlane (f x)
    ⊢ HasStrictDerivAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HMul.hMul (HMu …
  -/
  simpa using (hf.cpow hg h0).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.const_cpow (hf : HasStrictDerivAt f f' x) (h : c ≠ 0 ∨ f x ≠ 0) :
    HasStrictDerivAt (fun x => c ^ f x) (c ^ f x * Complex.log c * f') x :=
  (hasStrictDerivAt_const_cpow h).comp x hf


theorem Complex.hasStrictDerivAt_cpow_const (h : x ∈ slitPlane) :
    HasStrictDerivAt (fun z : ℂ => z ^ c) (c * x ^ (c - 1)) x := by
  simpa only [mul_zero, add_zero, mul_one] using
    (hasStrictDerivAt_id x).cpow (hasStrictDerivAt_const x c) h


theorem HasStrictDerivAt.cpow_const (hf : HasStrictDerivAt f f' x)
    (h0 : f x ∈ slitPlane) :
    HasStrictDerivAt (fun x => f x ^ c) (c * f x ^ (c - 1) * f') x :=
  (Complex.hasStrictDerivAt_cpow_const h0).comp x hf


theorem HasDerivAt.cpow (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x)
    (h0 : f x ∈ slitPlane) : HasDerivAt (fun x => f x ^ g x)
      (g x * f x ^ (g x - 1) * f' + f x ^ g x * Complex.log (f x) * g') x := by
  /-
    f g : Complex → Complex
    f' g' x : Complex
    hf : HasDerivAt f f' x
    hg : HasDerivAt g g' x
    h0 : Membership.mem Complex.slitPlane (f x)
    ⊢ HasDerivAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HMul.hMul (HMul.hMul …
  -/
  simpa only [aux] using (hf.hasFDerivAt.cpow hg h0).hasDerivAt
  /-
    🎉 no goals
  -/


theorem HasDerivAt.const_cpow (hf : HasDerivAt f f' x) (h0 : c ≠ 0 ∨ f x ≠ 0) :
    HasDerivAt (fun x => c ^ f x) (c ^ f x * Complex.log c * f') x :=
  (hasStrictDerivAt_const_cpow h0).hasDerivAt.comp x hf


theorem HasDerivAt.cpow_const (hf : HasDerivAt f f' x) (h0 : f x ∈ slitPlane) :
    HasDerivAt (fun x => f x ^ c) (c * f x ^ (c - 1) * f') x :=
  (Complex.hasStrictDerivAt_cpow_const h0).hasDerivAt.comp x hf


theorem HasDerivWithinAt.cpow (hf : HasDerivWithinAt f f' s x) (hg : HasDerivWithinAt g g' s x)
    (h0 : f x ∈ slitPlane) : HasDerivWithinAt (fun x => f x ^ g x)
      (g x * f x ^ (g x - 1) * f' + f x ^ g x * Complex.log (f x) * g') s x := by
  /-
    f g : Complex → Complex
    s : Set Complex
    f' g' x : Complex
    hf : HasDerivWithinAt f f' s x
    hg : HasDerivWithinAt g g' s x
    h0 : Membership.mem Complex.slitPlane (f x)
    ⊢ HasDerivWithinAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HMul.hMul (HMu …
  -/
  simpa only [aux] using (hf.hasFDerivWithinAt.cpow hg h0).hasDerivWithinAt
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.const_cpow (hf : HasDerivWithinAt f f' s x) (h0 : c ≠ 0 ∨ f x ≠ 0) :
    HasDerivWithinAt (fun x => c ^ f x) (c ^ f x * Complex.log c * f') s x :=
  (hasStrictDerivAt_const_cpow h0).hasDerivAt.comp_hasDerivWithinAt x hf


theorem HasDerivWithinAt.cpow_const (hf : HasDerivWithinAt f f' s x)
    (h0 : f x ∈ slitPlane) :
    HasDerivWithinAt (fun x => f x ^ c) (c * f x ^ (c - 1) * f') s x :=
  (Complex.hasStrictDerivAt_cpow_const h0).hasDerivAt.comp_hasDerivWithinAt x hf


/-- Although `fun x => x ^ r` for fixed `r` is *not* complex-differentiable along the negative real
line, it is still real-differentiable, and the derivative is what one would formally expect. -/
theorem hasDerivAt_ofReal_cpow {x : ℝ} (hx : x ≠ 0) {r : ℂ} (hr : r ≠ -1) :
    HasDerivAt (fun y : ℝ => (y : ℂ) ^ (r + 1) / (r + 1)) (x ^ r) x := by
  /-
    x : Real
    hx : Ne x 0
    r : Complex
    hr : Ne r (-1)
    ⊢ HasDerivAt (fun y => HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r 1)) (HAdd.hAdd r …
  -/
  rw [Ne, ← add_eq_zero_iff_eq_neg, ← Ne] at hr
  /-
    x : Real
    hx : Ne x 0
    r : Complex
    hr : Ne (HAdd.hAdd r 1) 0
    ⊢ HasDerivAt (fun y => HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r 1)) (HAdd.hAdd r …
  -/
  rcases lt_or_gt_of_ne hx.symm with (hx | hx)
  · -- easy case : `0 < x`
    -- Porting note: proof used to be
    -- convert (((hasDerivAt_id (x : ℂ)).cpow_const _).div_const (r + 1)).comp_ofReal using 1
    -- · rw [add_sub_cancel, id.def, mul_one, mul_comm, mul_div_cancel _ hr]
    -- · rw [id.def, ofReal_re]; exact Or.inl hx
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : LT.lt 0 x
      ⊢ HasDerivAt (fun y => HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r 1)) (HAdd.hAdd r …
    -/
    apply HasDerivAt.comp_ofReal (e := fun y => (y : ℂ) ^ (r + 1) / (r + 1))
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : LT.lt 0 x
      ⊢ HasDerivAt (fun y => HDiv.hDiv (HPow.hPow y (HAdd.hAdd r 1)) (HAdd.hAdd r 1) …
    -/
    convert HasDerivAt.div_const (𝕜 := ℂ) ?_ (r + 1) using 1
      /-
        case h.e'_9
        x : Real
        hx✝ : Ne x 0
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hx : LT.lt 0 x
        ⊢ Eq (HPow.hPow (↑x) r) (HDiv.hDiv ?inl.convert_3 (HAdd.hAdd r 1))
      -/
    · exact (mul_div_cancel_right₀ _ hr).symm
      /-
        🎉 no goals
      -/
      /-
        case inl.convert_4
        x : Real
        hx✝ : Ne x 0
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hx : LT.lt 0 x
        ⊢ HasDerivAt (fun y => HPow.hPow y (HAdd.hAdd r 1)) (HMul.hMul (HPow.hPow (↑x) …
      -/
    · convert HasDerivAt.cpow_const ?_ ?_ using 1
        /-
          case h.e'_9
          x : Real
          hx✝ : Ne x 0
          r : Complex
          hr : Ne (HAdd.hAdd r 1) 0
          hx : LT.lt 0 x
          ⊢ Eq (HMul.hMul (HPow.hPow (↑x) r) (HAdd.hAdd r 1)) (HMul.hMul (HMul.hMul (HAd …
        -/
      · rw [add_sub_cancel_right, mul_comm]; exact (mul_one _).symm
                                             /-
                                               🎉 no goals
                                             -/
        /-
          case inl.convert_4.convert_5
          x : Real
          hx✝ : Ne x 0
          r : Complex
          hr : Ne (HAdd.hAdd r 1) 0
          hx : LT.lt 0 x
          ⊢ HasDerivAt (fun y => y) 1 ↑x
        -/
      · exact hasDerivAt_id (x : ℂ)
        /-
          🎉 no goals
        -/
        /-
          case inl.convert_4.convert_6
          x : Real
          hx✝ : Ne x 0
          r : Complex
          hr : Ne (HAdd.hAdd r 1) 0
          hx : LT.lt 0 x
          ⊢ Membership.mem Complex.slitPlane ↑x
        -/
      · simp [hx]
        /-
          🎉 no goals
        -/
  · -- harder case : `x < 0`
    have : ∀ᶠ y : ℝ in 𝓝 x,
        (y : ℂ) ^ (r + 1) / (r + 1) = (-y : ℂ) ^ (r + 1) * exp (π * I * (r + 1)) / (r + 1) := by
      refine Filter.eventually_of_mem (Iio_mem_nhds hx) fun y hy => ?_
      rw [ofReal_cpow_of_nonpos (le_of_lt hy)]
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : GT.gt 0 x
      this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
      ⊢ HasDerivAt (fun y => HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r 1)) (HAdd.hAdd r …
    -/
    refine HasDerivAt.congr_of_eventuallyEq ?_ this
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : GT.gt 0 x
      this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
      ⊢ HasDerivAt (fun x => HDiv.hDiv (HMul.hMul (HPow.hPow (Neg.neg ↑x) (HAdd.hAdd …
    -/
    rw [ofReal_cpow_of_nonpos (le_of_lt hx)]
    suffices HasDerivAt (fun y : ℝ => (-↑y) ^ (r + 1) * exp (↑π * I * (r + 1)))
        ((r + 1) * (-↑x) ^ r * exp (↑π * I * r)) x by
      convert this.div_const (r + 1) using 1
      conv_rhs => rw [mul_assoc, mul_comm, mul_div_cancel_right₀ _ hr]
    rw [mul_add ((π : ℂ) * _), mul_one, exp_add, exp_pi_mul_I, mul_comm (_ : ℂ) (-1 : ℂ),
      neg_one_mul]
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : GT.gt 0 x
      this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
      ⊢ HasDerivAt (fun y => HMul.hMul (HPow.hPow (Neg.neg ↑y) (HAdd.hAdd r 1)) (Neg …
    -/
    simp_rw [mul_neg, ← neg_mul, ← ofReal_neg]
    suffices HasDerivAt (fun y : ℝ => (↑(-y) : ℂ) ^ (r + 1)) (-(r + 1) * ↑(-x) ^ r) x by
      convert this.neg.mul_const _ using 1; ring
    suffices HasDerivAt (fun y : ℝ => (y : ℂ) ^ (r + 1)) ((r + 1) * ↑(-x) ^ r) (-x) by
      convert @HasDerivAt.scomp ℝ _ ℂ _ _ x ℝ _ _ _ _ _ _ _ _ this (hasDerivAt_neg x) using 1
      rw [real_smul, ofReal_neg 1, ofReal_one]; ring
    suffices HasDerivAt (fun y : ℂ => y ^ (r + 1)) ((r + 1) * ↑(-x) ^ r) ↑(-x) by
      exact this.comp_ofReal
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : GT.gt 0 x
      this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
      ⊢ HasDerivAt (fun y => HPow.hPow y (HAdd.hAdd r 1)) (HMul.hMul (HAdd.hAdd r 1) …
    -/
    conv in ↑_ ^ _ => rw [(by ring : r = r + 1 - 1)]
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      r : Complex
      hr : Ne (HAdd.hAdd r 1) 0
      hx : GT.gt 0 x
      this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
      ⊢ HasDerivAt (fun y => HPow.hPow y (HAdd.hAdd (HSub.hSub (HAdd.hAdd r 1) 1) 1) …
    -/
    convert HasDerivAt.cpow_const ?_ ?_ using 1
      /-
        case h.e'_9
        x : Real
        hx✝ : Ne x 0
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hx : GT.gt 0 x
        this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
        ⊢ Eq (HMul.hMul (HAdd.hAdd r 1) (HPow.hPow (↑(Neg.neg x)) r)) (HMul.hMul (HMul …
      -/
    · rw [add_sub_cancel_right, add_sub_cancel_right]; exact (mul_one _).symm
                                                       /-
                                                         🎉 no goals
                                                       -/
      /-
        case inr.convert_5
        x : Real
        hx✝ : Ne x 0
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hx : GT.gt 0 x
        this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
        ⊢ HasDerivAt (fun y => y) 1 ↑(Neg.neg x)
      -/
    · exact hasDerivAt_id ((-x : ℝ) : ℂ)
      /-
        🎉 no goals
      -/
      /-
        case inr.convert_6
        x : Real
        hx✝ : Ne x 0
        r : Complex
        hr : Ne (HAdd.hAdd r 1) 0
        hx : GT.gt 0 x
        this : Filter.Eventually (fun y => Eq (HDiv.hDiv (HPow.hPow (↑y) (HAdd.hAdd r  …
        ⊢ Membership.mem Complex.slitPlane ↑(Neg.neg x)
      -/
    · simp [hx]
      /-
        🎉 no goals
      -/


/-- `(x, y) ↦ x ^ y` is strictly differentiable at `p : ℝ × ℝ` such that `0 < p.fst`. -/
theorem hasStrictFDerivAt_rpow_of_pos (p : ℝ × ℝ) (hp : 0 < p.1) :
    HasStrictFDerivAt (fun x : ℝ × ℝ => x.1 ^ x.2)
      ((p.2 * p.1 ^ (p.2 - 1)) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (p.1 ^ p.2 * log p.1) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  have : (fun x : ℝ × ℝ => x.1 ^ x.2) =ᶠ[𝓝 p] fun x => exp (log x.1 * x.2) :=
    (continuousAt_fst.eventually (lt_mem_nhds hp)).mono fun p hp => rpow_def_of_pos hp _
  /-
    p : Prod Real Real
    hp : LT.lt 0 p.1
    this : (nhds p).EventuallyEq (fun x => HPow.hPow x.1 x.2) fun x => Real.exp (H …
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow x.1 x.2) (HAdd.hAdd (HSMul.hSMul (HMul …
  -/
  refine HasStrictFDerivAt.congr_of_eventuallyEq ?_ this.symm
  /-
    p : Prod Real Real
    hp : LT.lt 0 p.1
    this : (nhds p).EventuallyEq (fun x => HPow.hPow x.1 x.2) fun x => Real.exp (H …
    ⊢ HasStrictFDerivAt (fun x => Real.exp (HMul.hMul (Real.log x.1) x.2)) (HAdd.h …
  -/
  convert ((hasStrictFDerivAt_fst.log hp.ne').mul hasStrictFDerivAt_snd).exp using 1
  rw [rpow_sub_one hp.ne', ← rpow_def_of_pos hp, smul_add, smul_smul, mul_div_left_comm,
    div_eq_mul_inv, smul_smul, smul_smul, mul_assoc, add_comm]


/-- `(x, y) ↦ x ^ y` is strictly differentiable at `p : ℝ × ℝ` such that `p.fst < 0`. -/
theorem hasStrictFDerivAt_rpow_of_neg (p : ℝ × ℝ) (hp : p.1 < 0) :
    HasStrictFDerivAt (fun x : ℝ × ℝ => x.1 ^ x.2)
      ((p.2 * p.1 ^ (p.2 - 1)) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (p.1 ^ p.2 * log p.1 - exp (log p.1 * p.2) * sin (p.2 * π) * π) •
          ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  have : (fun x : ℝ × ℝ => x.1 ^ x.2) =ᶠ[𝓝 p] fun x => exp (log x.1 * x.2) * cos (x.2 * π) :=
    (continuousAt_fst.eventually (gt_mem_nhds hp)).mono fun p hp => rpow_def_of_neg hp _
  /-
    p : Prod Real Real
    hp : LT.lt p.1 0
    this : (nhds p).EventuallyEq (fun x => HPow.hPow x.1 x.2) fun x => HMul.hMul ( …
    ⊢ HasStrictFDerivAt (fun x => HPow.hPow x.1 x.2) (HAdd.hAdd (HSMul.hSMul (HMul …
  -/
  refine HasStrictFDerivAt.congr_of_eventuallyEq ?_ this.symm
  convert ((hasStrictFDerivAt_fst.log hp.ne).mul hasStrictFDerivAt_snd).exp.mul
    (hasStrictFDerivAt_snd.mul_const π).cos using 1
  simp_rw [rpow_sub_one hp.ne, smul_add, ← add_assoc, smul_smul, ← add_smul, ← mul_assoc,
    mul_comm (cos _), ← rpow_def_of_neg hp]
  /-
    case h.e'_12
    p : Prod Real Real
    hp : LT.lt p.1 0
    this : (nhds p).EventuallyEq (fun x => HPow.hPow x.1 x.2) fun x => HMul.hMul ( …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul p.2 (HDiv.hDiv (HPow.hPow p.1 p.2) p.1 …
  -/
                                             /-
                                               🎉 no goals
                                             -/
  rw [div_eq_mul_inv, add_comm]; congr 2 <;> ring
                                             /-
                                               🎉 no goals
                                             -/


/-- The function `fun (x, y) => x ^ y` is infinitely smooth at `(x, y)` unless `x = 0`. -/
theorem contDiffAt_rpow_of_ne (p : ℝ × ℝ) (hp : p.1 ≠ 0) {n : WithTop ℕ∞} :
    ContDiffAt ℝ n (fun p : ℝ × ℝ => p.1 ^ p.2) p := by
  /-
    p : Prod Real Real
    hp : Ne p.1 0
    n : WithTop ENat
    ⊢ ContDiffAt Real n (fun p => HPow.hPow p.1 p.2) p
  -/
  cases' hp.lt_or_lt with hneg hpos
  exacts
    [(((contDiffAt_fst.log hneg.ne).mul contDiffAt_snd).exp.mul
          (contDiffAt_snd.mul contDiffAt_const).cos).congr_of_eventuallyEq
      ((continuousAt_fst.eventually (gt_mem_nhds hneg)).mono fun p hp => rpow_def_of_neg hp _),
    ((contDiffAt_fst.log hpos.ne').mul contDiffAt_snd).exp.congr_of_eventuallyEq
      ((continuousAt_fst.eventually (lt_mem_nhds hpos)).mono fun p hp => rpow_def_of_pos hp _)]


theorem differentiableAt_rpow_of_ne (p : ℝ × ℝ) (hp : p.1 ≠ 0) :
    DifferentiableAt ℝ (fun p : ℝ × ℝ => p.1 ^ p.2) p :=
  (contDiffAt_rpow_of_ne p hp).differentiableAt le_rfl


theorem _root_.HasStrictDerivAt.rpow {f g : ℝ → ℝ} {f' g' : ℝ} (hf : HasStrictDerivAt f f' x)
    (hg : HasStrictDerivAt g g' x) (h : 0 < f x) : HasStrictDerivAt (fun x => f x ^ g x)
      (f' * g x * f x ^ (g x - 1) + g' * f x ^ g x * Real.log (f x)) x := by
  convert (hasStrictFDerivAt_rpow_of_pos ((fun x => (f x, g x)) x) h).comp_hasStrictDerivAt x
    (hf.prod hg) using 1
  /-
    case h.e'_9
    x : Real
    f g : Real → Real
    f' g' : Real
    hf : HasStrictDerivAt f f' x
    hg : HasStrictDerivAt g g' x
    h : LT.lt 0 (f x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul f' (g x)) (HPow.hPow (f x) (HSub.hSub (g …
  -/
  simp [mul_assoc, mul_comm, mul_left_comm]
  /-
    🎉 no goals
  -/


theorem hasStrictDerivAt_rpow_const_of_ne {x : ℝ} (hx : x ≠ 0) (p : ℝ) :
    HasStrictDerivAt (fun x => x ^ p) (p * x ^ (p - 1)) x := by
  /-
    x : Real
    hx : Ne x 0
    p : Real
    ⊢ HasStrictDerivAt (fun x => HPow.hPow x p) (HMul.hMul p (HPow.hPow x (HSub.hS …
  -/
  cases' hx.lt_or_lt with hx hx
  · have := (hasStrictFDerivAt_rpow_of_neg (x, p) hx).comp_hasStrictDerivAt x
      ((hasStrictDerivAt_id x).prod (hasStrictDerivAt_const x p))
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      p : Real
      hx : LT.lt x 0
      this : HasStrictDerivAt (Function.comp (fun x => HPow.hPow x.1 x.2) fun x => { …
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x p) (HMul.hMul p (HPow.hPow x (HSub.hS …
    -/
    convert this using 1; simp
                          /-
                            🎉 no goals
                          -/
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      p : Real
      hx : LT.lt 0 x
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x p) (HMul.hMul p (HPow.hPow x (HSub.hS …
    -/
  · simpa using (hasStrictDerivAt_id x).rpow (hasStrictDerivAt_const x p) hx
    /-
      🎉 no goals
    -/


theorem hasStrictDerivAt_const_rpow {a : ℝ} (ha : 0 < a) (x : ℝ) :
    HasStrictDerivAt (fun x => a ^ x) (a ^ x * log a) x := by
  /-
    a : Real
    ha : LT.lt 0 a
    x : Real
    ⊢ HasStrictDerivAt (fun x => HPow.hPow a x) (HMul.hMul (HPow.hPow a x) (Real.l …
  -/
  simpa using (hasStrictDerivAt_const _ _).rpow (hasStrictDerivAt_id x) ha
  /-
    🎉 no goals
  -/


lemma differentiableAt_rpow_const_of_ne (p : ℝ) {x : ℝ} (hx : x ≠ 0) :
    DifferentiableAt ℝ (fun x => x ^ p) x :=
  (hasStrictDerivAt_rpow_const_of_ne hx p).differentiableAt


lemma differentiableOn_rpow_const (p : ℝ) :
    DifferentiableOn ℝ (fun x => (x : ℝ) ^ p) {0}ᶜ :=
  fun _ hx => (Real.differentiableAt_rpow_const_of_ne p hx).differentiableWithinAt


/-- This lemma says that `fun x => a ^ x` is strictly differentiable for `a < 0`. Note that these
values of `a` are outside of the "official" domain of `a ^ x`, and we may redefine `a ^ x`
for negative `a` if some other definition will be more convenient. -/
theorem hasStrictDerivAt_const_rpow_of_neg {a x : ℝ} (ha : a < 0) :
    HasStrictDerivAt (fun x => a ^ x) (a ^ x * log a - exp (log a * x) * sin (x * π) * π) x := by
  simpa using (hasStrictFDerivAt_rpow_of_neg (a, x) ha).comp_hasStrictDerivAt x
    ((hasStrictDerivAt_const _ _).prod (hasStrictDerivAt_id _))


theorem hasDerivAt_rpow_const {x p : ℝ} (h : x ≠ 0 ∨ 1 ≤ p) :
    HasDerivAt (fun x => x ^ p) (p * x ^ (p - 1)) x := by
  /-
    x p : Real
    h : Or (Ne x 0) (LE.le 1 p)
    ⊢ HasDerivAt (fun x => HPow.hPow x p) (HMul.hMul p (HPow.hPow x (HSub.hSub p 1 …
  -/
  rcases ne_or_eq x 0 with (hx | rfl)
    /-
      case inl
      x p : Real
      h : Or (Ne x 0) (LE.le 1 p)
      hx : Ne x 0
      ⊢ HasDerivAt (fun x => HPow.hPow x p) (HMul.hMul p (HPow.hPow x (HSub.hSub p 1 …
    -/
  · exact (hasStrictDerivAt_rpow_const_of_ne hx _).hasDerivAt
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    h : Or (Ne 0 0) (LE.le 1 p)
    ⊢ HasDerivAt (fun x => HPow.hPow x p) (HMul.hMul p (HPow.hPow 0 (HSub.hSub p 1 …
  -/
  replace h : 1 ≤ p := h.neg_resolve_left rfl
  apply hasDerivAt_of_hasDerivAt_of_ne fun x hx =>
    (hasStrictDerivAt_rpow_const_of_ne hx p).hasDerivAt
  exacts [continuousAt_id.rpow_const (Or.inr (zero_le_one.trans h)),
    continuousAt_const.mul (continuousAt_id.rpow_const (Or.inr (sub_nonneg.2 h)))]


theorem differentiable_rpow_const {p : ℝ} (hp : 1 ≤ p) : Differentiable ℝ fun x : ℝ => x ^ p :=
  fun _ => (hasDerivAt_rpow_const (Or.inr hp)).differentiableAt


theorem deriv_rpow_const {x p : ℝ} (h : x ≠ 0 ∨ 1 ≤ p) :
    deriv (fun x : ℝ => x ^ p) x = p * x ^ (p - 1) :=
  (hasDerivAt_rpow_const h).deriv


theorem deriv_rpow_const' {p : ℝ} (h : 1 ≤ p) :
    (deriv fun x : ℝ => x ^ p) = fun x => p * x ^ (p - 1) :=
  funext fun _ => deriv_rpow_const (Or.inr h)


theorem contDiffAt_rpow_const_of_ne {x p : ℝ} {n : WithTop ℕ∞} (h : x ≠ 0) :
    ContDiffAt ℝ n (fun x => x ^ p) x :=
  (contDiffAt_rpow_of_ne (x, p) h).comp x (contDiffAt_id.prod contDiffAt_const)


theorem contDiff_rpow_const_of_le {p : ℝ} {n : ℕ} (h : ↑n ≤ p) :
    ContDiff ℝ n fun x : ℝ => x ^ p := by
  /-
    p : Real
    n : Nat
    h : LE.le (↑n) p
    ⊢ ContDiff Real ↑n fun x => HPow.hPow x p
  -/
  induction' n with n ihn generalizing p
    /-
      case zero
      p : Real
      h : LE.le (↑0) p
      ⊢ ContDiff Real ↑0 fun x => HPow.hPow x p
    -/
  · exact contDiff_zero.2 (continuous_id.rpow_const fun x => Or.inr <| by simpa using h)
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ihn : ∀ {p : Real}, LE.le (↑n) p → ContDiff Real ↑n fun x => HPow.hPow x p
      p : Real
      h : LE.le (↑(HAdd.hAdd n 1)) p
      ⊢ ContDiff Real ↑(HAdd.hAdd n 1) fun x => HPow.hPow x p
    -/
  · have h1 : 1 ≤ p := le_trans (by simp) h
    /-
      case succ
      n : Nat
      ihn : ∀ {p : Real}, LE.le (↑n) p → ContDiff Real ↑n fun x => HPow.hPow x p
      p : Real
      h : LE.le (↑(HAdd.hAdd n 1)) p
      h1 : LE.le 1 p
      ⊢ ContDiff Real ↑(HAdd.hAdd n 1) fun x => HPow.hPow x p
    -/
    rw [Nat.cast_succ, ← le_sub_iff_add_le] at h
    rw [show ((n + 1 : ℕ) : WithTop ℕ∞) = n + 1 from rfl,
      contDiff_succ_iff_deriv, deriv_rpow_const' h1]
    /-
      case succ
      n : Nat
      ihn : ∀ {p : Real}, LE.le (↑n) p → ContDiff Real ↑n fun x => HPow.hPow x p
      p : Real
      h : LE.le (↑n) (HSub.hSub p 1)
      h1 : LE.le 1 p
      ⊢ And (Differentiable Real fun x => HPow.hPow x p) (And (Eq (↑n) Top.top → Ana …
    -/
    simp only [WithTop.natCast_ne_top, analyticOn_univ, IsEmpty.forall_iff, true_and]
    /-
      case succ
      n : Nat
      ihn : ∀ {p : Real}, LE.le (↑n) p → ContDiff Real ↑n fun x => HPow.hPow x p
      p : Real
      h : LE.le (↑n) (HSub.hSub p 1)
      h1 : LE.le 1 p
      ⊢ And (Differentiable Real fun x => HPow.hPow x p) (ContDiff Real ↑n fun x =>  …
    -/
    exact ⟨differentiable_rpow_const h1, contDiff_const.mul (ihn h)⟩
    /-
      🎉 no goals
    -/


theorem contDiffAt_rpow_const_of_le {x p : ℝ} {n : ℕ} (h : ↑n ≤ p) :
    ContDiffAt ℝ n (fun x : ℝ => x ^ p) x :=
  (contDiff_rpow_const_of_le h).contDiffAt


theorem contDiffAt_rpow_const {x p : ℝ} {n : ℕ} (h : x ≠ 0 ∨ ↑n ≤ p) :
    ContDiffAt ℝ n (fun x : ℝ => x ^ p) x :=
  h.elim contDiffAt_rpow_const_of_ne contDiffAt_rpow_const_of_le


theorem hasStrictDerivAt_rpow_const {x p : ℝ} (hx : x ≠ 0 ∨ 1 ≤ p) :
    HasStrictDerivAt (fun x => x ^ p) (p * x ^ (p - 1)) x :=
                                                          /-
                                                            x p : Real
                                                            hx : Or (Ne x 0) (LE.le 1 p)
                                                            ⊢ Or (Ne x 0) (LE.le (↑One.one) p)
                                                          -/
  ContDiffAt.hasStrictDerivAt' (contDiffAt_rpow_const (by rwa [← Nat.cast_one] at hx))
                                                          /-
                                                            🎉 no goals
                                                          -/
    (hasDerivAt_rpow_const hx) le_rfl


theorem HasFDerivWithinAt.rpow (hf : HasFDerivWithinAt f f' s x) (hg : HasFDerivWithinAt g g' s x)
    (h : 0 < f x) : HasFDerivWithinAt (fun x => f x ^ g x)
      ((g x * f x ^ (g x - 1)) • f' + (f x ^ g x * Real.log (f x)) • g') s x := by
  exact (hasStrictFDerivAt_rpow_of_pos (f x, g x) h).hasFDerivAt.comp_hasFDerivWithinAt x
    (hf.prod hg)


theorem HasFDerivAt.rpow (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x) (h : 0 < f x) :
    HasFDerivAt (fun x => f x ^ g x)
      ((g x * f x ^ (g x - 1)) • f' + (f x ^ g x * Real.log (f x)) • g') x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : E → Real
    f' g' : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    hf : HasFDerivAt f f' x
    hg : HasFDerivAt g g' x
    h : LT.lt 0 (f x)
    ⊢ HasFDerivAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HSMul.hSMul (HMul.h …
  -/
  exact (hasStrictFDerivAt_rpow_of_pos (f x, g x) h).hasFDerivAt.comp x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem HasStrictFDerivAt.rpow (hf : HasStrictFDerivAt f f' x) (hg : HasStrictFDerivAt g g' x)
    (h : 0 < f x) : HasStrictFDerivAt (fun x => f x ^ g x)
      ((g x * f x ^ (g x - 1)) • f' + (f x ^ g x * Real.log (f x)) • g') x :=
  (hasStrictFDerivAt_rpow_of_pos (f x, g x) h).comp x (hf.prod hg)


theorem DifferentiableWithinAt.rpow (hf : DifferentiableWithinAt ℝ f s x)
    (hg : DifferentiableWithinAt ℝ g s x) (h : f x ≠ 0) :
    DifferentiableWithinAt ℝ (fun x => f x ^ g x) s x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : E → Real
    x : E
    s : Set E
    hf : DifferentiableWithinAt Real f s x
    hg : DifferentiableWithinAt Real g s x
    h : Ne (f x) 0
    ⊢ DifferentiableWithinAt Real (fun x => HPow.hPow (f x) (g x)) s x
  -/
  exact (differentiableAt_rpow_of_ne (f x, g x) h).comp_differentiableWithinAt x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem DifferentiableAt.rpow (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (h : f x ≠ 0) : DifferentiableAt ℝ (fun x => f x ^ g x) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : E → Real
    x : E
    hf : DifferentiableAt Real f x
    hg : DifferentiableAt Real g x
    h : Ne (f x) 0
    ⊢ DifferentiableAt Real (fun x => HPow.hPow (f x) (g x)) x
  -/
  exact (differentiableAt_rpow_of_ne (f x, g x) h).comp x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem DifferentiableOn.rpow (hf : DifferentiableOn ℝ f s) (hg : DifferentiableOn ℝ g s)
    (h : ∀ x ∈ s, f x ≠ 0) : DifferentiableOn ℝ (fun x => f x ^ g x) s := fun x hx =>
  (hf x hx).rpow (hg x hx) (h x hx)


theorem Differentiable.rpow (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) (h : ∀ x, f x ≠ 0) :
    Differentiable ℝ fun x => f x ^ g x := fun x => (hf x).rpow (hg x) (h x)


theorem HasFDerivWithinAt.rpow_const (hf : HasFDerivWithinAt f f' s x) (h : f x ≠ 0 ∨ 1 ≤ p) :
    HasFDerivWithinAt (fun x => f x ^ p) ((p * f x ^ (p - 1)) • f') s x :=
  (hasDerivAt_rpow_const h).comp_hasFDerivWithinAt x hf


theorem HasFDerivAt.rpow_const (hf : HasFDerivAt f f' x) (h : f x ≠ 0 ∨ 1 ≤ p) :
    HasFDerivAt (fun x => f x ^ p) ((p * f x ^ (p - 1)) • f') x :=
  (hasDerivAt_rpow_const h).comp_hasFDerivAt x hf


theorem HasStrictFDerivAt.rpow_const (hf : HasStrictFDerivAt f f' x) (h : f x ≠ 0 ∨ 1 ≤ p) :
    HasStrictFDerivAt (fun x => f x ^ p) ((p * f x ^ (p - 1)) • f') x :=
  (hasStrictDerivAt_rpow_const h).comp_hasStrictFDerivAt x hf


theorem DifferentiableWithinAt.rpow_const (hf : DifferentiableWithinAt ℝ f s x)
    (h : f x ≠ 0 ∨ 1 ≤ p) : DifferentiableWithinAt ℝ (fun x => f x ^ p) s x :=
  (hf.hasFDerivWithinAt.rpow_const h).differentiableWithinAt


@[simp]
theorem DifferentiableAt.rpow_const (hf : DifferentiableAt ℝ f x) (h : f x ≠ 0 ∨ 1 ≤ p) :
    DifferentiableAt ℝ (fun x => f x ^ p) x :=
  (hf.hasFDerivAt.rpow_const h).differentiableAt


theorem DifferentiableOn.rpow_const (hf : DifferentiableOn ℝ f s) (h : ∀ x ∈ s, f x ≠ 0 ∨ 1 ≤ p) :
    DifferentiableOn ℝ (fun x => f x ^ p) s := fun x hx => (hf x hx).rpow_const (h x hx)


theorem Differentiable.rpow_const (hf : Differentiable ℝ f) (h : ∀ x, f x ≠ 0 ∨ 1 ≤ p) :
    Differentiable ℝ fun x => f x ^ p := fun x => (hf x).rpow_const (h x)


theorem HasFDerivWithinAt.const_rpow (hf : HasFDerivWithinAt f f' s x) (hc : 0 < c) :
    HasFDerivWithinAt (fun x => c ^ f x) ((c ^ f x * Real.log c) • f') s x :=
  (hasStrictDerivAt_const_rpow hc (f x)).hasDerivAt.comp_hasFDerivWithinAt x hf


theorem HasFDerivAt.const_rpow (hf : HasFDerivAt f f' x) (hc : 0 < c) :
    HasFDerivAt (fun x => c ^ f x) ((c ^ f x * Real.log c) • f') x :=
  (hasStrictDerivAt_const_rpow hc (f x)).hasDerivAt.comp_hasFDerivAt x hf


theorem HasStrictFDerivAt.const_rpow (hf : HasStrictFDerivAt f f' x) (hc : 0 < c) :
    HasStrictFDerivAt (fun x => c ^ f x) ((c ^ f x * Real.log c) • f') x :=
  (hasStrictDerivAt_const_rpow hc (f x)).comp_hasStrictFDerivAt x hf


theorem ContDiffWithinAt.rpow (hf : ContDiffWithinAt ℝ n f s x) (hg : ContDiffWithinAt ℝ n g s x)
    (h : f x ≠ 0) : ContDiffWithinAt ℝ n (fun x => f x ^ g x) s x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : E → Real
    x : E
    s : Set E
    n : WithTop ENat
    hf : ContDiffWithinAt Real n f s x
    hg : ContDiffWithinAt Real n g s x
    h : Ne (f x) 0
    ⊢ ContDiffWithinAt Real n (fun x => HPow.hPow (f x) (g x)) s x
  -/
  exact (contDiffAt_rpow_of_ne (f x, g x) h).comp_contDiffWithinAt x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem ContDiffAt.rpow (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) (h : f x ≠ 0) :
    ContDiffAt ℝ n (fun x => f x ^ g x) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : E → Real
    x : E
    n : WithTop ENat
    hf : ContDiffAt Real n f x
    hg : ContDiffAt Real n g x
    h : Ne (f x) 0
    ⊢ ContDiffAt Real n (fun x => HPow.hPow (f x) (g x)) x
  -/
  exact (contDiffAt_rpow_of_ne (f x, g x) h).comp x (hf.prod hg)
  /-
    🎉 no goals
  -/


theorem ContDiffOn.rpow (hf : ContDiffOn ℝ n f s) (hg : ContDiffOn ℝ n g s) (h : ∀ x ∈ s, f x ≠ 0) :
    ContDiffOn ℝ n (fun x => f x ^ g x) s := fun x hx => (hf x hx).rpow (hg x hx) (h x hx)


theorem ContDiff.rpow (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (h : ∀ x, f x ≠ 0) :
    ContDiff ℝ n fun x => f x ^ g x :=
  contDiff_iff_contDiffAt.mpr fun x => hf.contDiffAt.rpow hg.contDiffAt (h x)


theorem ContDiffWithinAt.rpow_const_of_ne (hf : ContDiffWithinAt ℝ n f s x) (h : f x ≠ 0) :
    ContDiffWithinAt ℝ n (fun x => f x ^ p) s x :=
  hf.rpow contDiffWithinAt_const h


theorem ContDiffAt.rpow_const_of_ne (hf : ContDiffAt ℝ n f x) (h : f x ≠ 0) :
    ContDiffAt ℝ n (fun x => f x ^ p) x :=
  hf.rpow contDiffAt_const h


theorem ContDiffOn.rpow_const_of_ne (hf : ContDiffOn ℝ n f s) (h : ∀ x ∈ s, f x ≠ 0) :
    ContDiffOn ℝ n (fun x => f x ^ p) s := fun x hx => (hf x hx).rpow_const_of_ne (h x hx)


theorem ContDiff.rpow_const_of_ne (hf : ContDiff ℝ n f) (h : ∀ x, f x ≠ 0) :
    ContDiff ℝ n fun x => f x ^ p :=
  hf.rpow contDiff_const h


theorem ContDiffWithinAt.rpow_const_of_le (hf : ContDiffWithinAt ℝ m f s x) (h : ↑m ≤ p) :
    ContDiffWithinAt ℝ m (fun x => f x ^ p) s x :=
  (contDiffAt_rpow_const_of_le h).comp_contDiffWithinAt x hf


theorem ContDiffAt.rpow_const_of_le (hf : ContDiffAt ℝ m f x) (h : ↑m ≤ p) :
    ContDiffAt ℝ m (fun x => f x ^ p) x := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    x : E
    p : Real
    m : Nat
    hf : ContDiffAt Real (↑m) f x
    h : LE.le (↑m) p
    ⊢ ContDiffAt Real (↑m) (fun x => HPow.hPow (f x) p) x
  -/
  rw [← contDiffWithinAt_univ] at *; exact hf.rpow_const_of_le h
                                     /-
                                       🎉 no goals
                                     -/


theorem ContDiffOn.rpow_const_of_le (hf : ContDiffOn ℝ m f s) (h : ↑m ≤ p) :
    ContDiffOn ℝ m (fun x => f x ^ p) s := fun x hx => (hf x hx).rpow_const_of_le h


theorem ContDiff.rpow_const_of_le (hf : ContDiff ℝ m f) (h : ↑m ≤ p) :
    ContDiff ℝ m fun x => f x ^ p :=
  contDiff_iff_contDiffAt.mpr fun _ => hf.contDiffAt.rpow_const_of_le h


theorem HasDerivWithinAt.rpow (hf : HasDerivWithinAt f f' s x) (hg : HasDerivWithinAt g g' s x)
    (h : 0 < f x) : HasDerivWithinAt (fun x => f x ^ g x)
      (f' * g x * f x ^ (g x - 1) + g' * f x ^ g x * Real.log (f x)) s x := by
  /-
    f g : Real → Real
    f' g' x : Real
    s : Set Real
    hf : HasDerivWithinAt f f' s x
    hg : HasDerivWithinAt g g' s x
    h : LT.lt 0 (f x)
    ⊢ HasDerivWithinAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HMul.hMul (HMu …
  -/
  convert (hf.hasFDerivWithinAt.rpow hg.hasFDerivWithinAt h).hasDerivWithinAt using 1
  /-
    case h.e'_9
    f g : Real → Real
    f' g' x : Real
    s : Set Real
    hf : HasDerivWithinAt f f' s x
    hg : HasDerivWithinAt g g' s x
    h : LT.lt 0 (f x)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul f' (g x)) (HPow.hPow (f x) (HSub.hSub (g …
  -/
  dsimp; ring
         /-
           🎉 no goals
         -/


theorem HasDerivAt.rpow (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x) (h : 0 < f x) :
    HasDerivAt (fun x => f x ^ g x)
      (f' * g x * f x ^ (g x - 1) + g' * f x ^ g x * Real.log (f x)) x := by
  /-
    f g : Real → Real
    f' g' x : Real
    hf : HasDerivAt f f' x
    hg : HasDerivAt g g' x
    h : LT.lt 0 (f x)
    ⊢ HasDerivAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HMul.hMul (HMul.hMul …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    f g : Real → Real
    f' g' x : Real
    hf : HasDerivWithinAt f f' Set.univ x
    hg : HasDerivWithinAt g g' Set.univ x
    h : LT.lt 0 (f x)
    ⊢ HasDerivWithinAt (fun x => HPow.hPow (f x) (g x)) (HAdd.hAdd (HMul.hMul (HMu …
  -/
  exact hf.rpow hg h
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.rpow_const (hf : HasDerivWithinAt f f' s x) (hx : f x ≠ 0 ∨ 1 ≤ p) :
    HasDerivWithinAt (fun y => f y ^ p) (f' * p * f x ^ (p - 1)) s x := by
  /-
    f : Real → Real
    f' x p : Real
    s : Set Real
    hf : HasDerivWithinAt f f' s x
    hx : Or (Ne (f x) 0) (LE.le 1 p)
    ⊢ HasDerivWithinAt (fun y => HPow.hPow (f y) p) (HMul.hMul (HMul.hMul f' p) (H …
  -/
  convert (hasDerivAt_rpow_const hx).comp_hasDerivWithinAt x hf using 1
  /-
    case h.e'_9
    f : Real → Real
    f' x p : Real
    s : Set Real
    hf : HasDerivWithinAt f f' s x
    hx : Or (Ne (f x) 0) (LE.le 1 p)
    ⊢ Eq (HMul.hMul (HMul.hMul f' p) (HPow.hPow (f x) (HSub.hSub p 1))) (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem HasDerivAt.rpow_const (hf : HasDerivAt f f' x) (hx : f x ≠ 0 ∨ 1 ≤ p) :
    HasDerivAt (fun y => f y ^ p) (f' * p * f x ^ (p - 1)) x := by
  /-
    f : Real → Real
    f' x p : Real
    hf : HasDerivAt f f' x
    hx : Or (Ne (f x) 0) (LE.le 1 p)
    ⊢ HasDerivAt (fun y => HPow.hPow (f y) p) (HMul.hMul (HMul.hMul f' p) (HPow.hP …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    f : Real → Real
    f' x p : Real
    hf : HasDerivWithinAt f f' Set.univ x
    hx : Or (Ne (f x) 0) (LE.le 1 p)
    ⊢ HasDerivWithinAt (fun y => HPow.hPow (f y) p) (HMul.hMul (HMul.hMul f' p) (H …
  -/
  exact hf.rpow_const hx
  /-
    🎉 no goals
  -/


theorem derivWithin_rpow_const (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0 ∨ 1 ≤ p)
    (hxs : UniqueDiffWithinAt ℝ s x) :
    derivWithin (fun x => f x ^ p) s x = derivWithin f s x * p * f x ^ (p - 1) :=
  (hf.hasDerivWithinAt.rpow_const hx).derivWithin hxs


@[simp]
theorem deriv_rpow_const (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0 ∨ 1 ≤ p) :
    deriv (fun x => f x ^ p) x = deriv f x * p * f x ^ (p - 1) :=
  (hf.hasDerivAt.rpow_const hx).deriv


lemma isTheta_deriv_rpow_const_atTop {p : ℝ} (hp : p ≠ 0) :
    deriv (fun (x : ℝ) => x ^ p) =Θ[atTop] fun x => x ^ (p-1) := by
  calc deriv (fun (x : ℝ) => x ^ p) =ᶠ[atTop] fun x => p * x ^ (p - 1) := by
              filter_upwards [eventually_ne_atTop 0] with x hx
              rw [Real.deriv_rpow_const (Or.inl hx)]
       _ =Θ[atTop] fun x => x ^ (p-1) :=
              Asymptotics.IsTheta.const_mul_left hp Asymptotics.isTheta_rfl


lemma isBigO_deriv_rpow_const_atTop (p : ℝ) :
    deriv (fun (x : ℝ) => x ^ p) =O[atTop] fun x => x ^ (p-1) := by
  /-
    p : Real
    ⊢ Asymptotics.IsBigO Filter.atTop (deriv fun x => HPow.hPow x p) fun x => HPow …
  -/
  rcases eq_or_ne p 0 with rfl | hp
  case inl =>
    simp [zero_sub, Real.rpow_neg_one, Real.rpow_zero, deriv_const', Asymptotics.isBigO_zero]
  case inr =>
    exact (isTheta_deriv_rpow_const_atTop hp).1


/-- The function `(1 + t/x) ^ x` tends to `exp t` at `+∞`. -/
theorem tendsto_one_plus_div_rpow_exp (t : ℝ) :
    Tendsto (fun x : ℝ => (1 + t / x) ^ x) atTop (𝓝 (exp t)) := by
  /-
    t : Real
    ⊢ Filter.Tendsto (fun x => HPow.hPow (HAdd.hAdd 1 (HDiv.hDiv t x)) x) Filter.a …
  -/
  apply ((Real.continuous_exp.tendsto _).comp (tendsto_mul_log_one_plus_div_atTop t)).congr' _
  /-
    t : Real
    ⊢ Filter.atTop.EventuallyEq (Function.comp Real.exp fun x => HMul.hMul x (Real …
  -/
  have h₁ : (1 : ℝ) / 2 < 1 := by norm_num
  have h₂ : Tendsto (fun x : ℝ => 1 + t / x) atTop (𝓝 1) := by
    simpa using (tendsto_inv_atTop_zero.const_mul t).const_add 1
  /-
    t : Real
    h₁ : LT.lt (1 / 2) 1
    h₂ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv t x)) Filter.atTop (nhds 1)
    ⊢ Filter.atTop.EventuallyEq (Function.comp Real.exp fun x => HMul.hMul x (Real …
  -/
  refine (h₂.eventually_const_le h₁).mono fun x hx => ?_
  /-
    t : Real
    h₁ : LT.lt (1 / 2) 1
    h₂ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv t x)) Filter.atTop (nhds 1)
    x : Real
    hx : LE.le (1 / 2) (HAdd.hAdd 1 (HDiv.hDiv t x))
    ⊢ Eq (Function.comp Real.exp (fun x => HMul.hMul x (Real.log (HAdd.hAdd 1 (HDi …
  -/
  have hx' : 0 < 1 + t / x := by linarith
  /-
    t : Real
    h₁ : LT.lt (1 / 2) 1
    h₂ : Filter.Tendsto (fun x => HAdd.hAdd 1 (HDiv.hDiv t x)) Filter.atTop (nhds 1)
    x : Real
    hx : LE.le (1 / 2) (HAdd.hAdd 1 (HDiv.hDiv t x))
    hx' : LT.lt 0 (HAdd.hAdd 1 (HDiv.hDiv t x))
    ⊢ Eq (Function.comp Real.exp (fun x => HMul.hMul x (Real.log (HAdd.hAdd 1 (HDi …
  -/
  simp [mul_comm x, exp_mul, exp_log hx']
  /-
    🎉 no goals
  -/


/-- The function `(1 + t/x) ^ x` tends to `exp t` at `+∞` for naturals `x`. -/
theorem tendsto_one_plus_div_pow_exp (t : ℝ) :
    Tendsto (fun x : ℕ => (1 + t / (x : ℝ)) ^ x) atTop (𝓝 (Real.exp t)) :=
                                                                                 /-
                                                                                   t : Real
                                                                                   ⊢ ∀ (x : Nat), Eq (Function.comp (fun x => HPow.hPow (HAdd.hAdd 1 (HDiv.hDiv t …
                                                                                 -/
  ((tendsto_one_plus_div_rpow_exp t).comp tendsto_natCast_atTop_atTop).congr (by simp)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


