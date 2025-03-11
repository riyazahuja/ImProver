theorem hasStrictDerivAt_log_of_pos (hx : 0 < x) : HasStrictDerivAt log x⁻¹ x := by
  have : HasStrictDerivAt log (exp <| log x)⁻¹ x :=
    (hasStrictDerivAt_exp <| log x).of_local_left_inverse (continuousAt_log hx.ne')
        (ne_of_gt <| exp_pos _) <|
      Eventually.mono (lt_mem_nhds hx) @exp_log
  /-
    x : Real
    hx : LT.lt 0 x
    this : HasStrictDerivAt Real.log (Inv.inv (Real.exp (Real.log x))) x
    ⊢ HasStrictDerivAt Real.log (Inv.inv x) x
  -/
  rwa [exp_log hx] at this
  /-
    🎉 no goals
  -/


theorem hasStrictDerivAt_log (hx : x ≠ 0) : HasStrictDerivAt log x⁻¹ x := by
  /-
    x : Real
    hx : Ne x 0
    ⊢ HasStrictDerivAt Real.log (Inv.inv x) x
  -/
  cases' hx.lt_or_lt with hx hx
    /-
      case inl
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt x 0
      ⊢ HasStrictDerivAt Real.log (Inv.inv x) x
    -/
  · convert (hasStrictDerivAt_log_of_pos (neg_pos.mpr hx)).comp x (hasStrictDerivAt_neg x) using 1
      /-
        case h.e'_8
        x : Real
        hx✝ : Ne x 0
        hx : LT.lt x 0
        ⊢ Eq Real.log (Function.comp Real.log Neg.neg)
      -/
    · ext y; exact (log_neg_eq_log y).symm
             /-
               🎉 no goals
             -/
      /-
        case h.e'_9
        x : Real
        hx✝ : Ne x 0
        hx : LT.lt x 0
        ⊢ Eq (Inv.inv x) (HMul.hMul (Inv.inv (Neg.neg x)) (-1))
      -/
    · field_simp [hx.ne]
      /-
        🎉 no goals
      -/
    /-
      case inr
      x : Real
      hx✝ : Ne x 0
      hx : LT.lt 0 x
      ⊢ HasStrictDerivAt Real.log (Inv.inv x) x
    -/
  · exact hasStrictDerivAt_log_of_pos hx
    /-
      🎉 no goals
    -/


theorem hasDerivAt_log (hx : x ≠ 0) : HasDerivAt log x⁻¹ x :=
  (hasStrictDerivAt_log hx).hasDerivAt


@[fun_prop] theorem differentiableAt_log (hx : x ≠ 0) : DifferentiableAt ℝ log x :=
  (hasDerivAt_log hx).differentiableAt


theorem differentiableOn_log : DifferentiableOn ℝ log {0}ᶜ := fun _x hx =>
  (differentiableAt_log hx).differentiableWithinAt


@[simp]
theorem differentiableAt_log_iff : DifferentiableAt ℝ log x ↔ x ≠ 0 :=
  ⟨fun h => continuousAt_log_iff.1 h.continuousAt, differentiableAt_log⟩


theorem deriv_log (x : ℝ) : deriv log x = x⁻¹ :=
  if hx : x = 0 then by
    /-
      x : Real
      hx : Eq x 0
      ⊢ Eq (deriv Real.log x) (Inv.inv x)
    -/
    rw [deriv_zero_of_not_differentiableAt (differentiableAt_log_iff.not_left.2 hx), hx, inv_zero]
    /-
      🎉 no goals
    -/
  else (hasDerivAt_log hx).deriv


@[simp]
theorem deriv_log' : deriv log = Inv.inv :=
  funext deriv_log


theorem contDiffAt_log {n : WithTop ℕ∞} {x : ℝ} : ContDiffAt ℝ n log x ↔ x ≠ 0 := by
  /-
    n : WithTop ENat
    x : Real
    ⊢ Iff (ContDiffAt Real n Real.log x) (Ne x 0)
  -/
  refine ⟨fun h ↦ continuousAt_log_iff.1 h.continuousAt, fun hx ↦ ?_⟩
  have A y (hy : 0 < y) : ContDiffAt ℝ n log y := by
    apply expPartialHomeomorph.contDiffAt_symm_deriv (f₀' := y) hy.ne' (by simpa)
    · convert hasDerivAt_exp (log y)
      rw [exp_log hy]
    · exact analyticAt_rexp.contDiffAt
  /-
    n : WithTop ENat
    x : Real
    hx : Ne x 0
    A : ∀ (y : Real), LT.lt 0 y → ContDiffAt Real n Real.log y
    ⊢ ContDiffAt Real n Real.log x
  -/
  rcases hx.lt_or_lt with hx | hx
  · have : ContDiffAt ℝ n (log ∘ (fun y ↦ -y)) x := by
      apply ContDiffAt.comp
      apply A _ (Left.neg_pos_iff.mpr hx)
      apply contDiffAt_id.neg
    /-
      case inl
      n : WithTop ENat
      x : Real
      hx✝ : Ne x 0
      A : ∀ (y : Real), LT.lt 0 y → ContDiffAt Real n Real.log y
      hx : LT.lt x 0
      this : ContDiffAt Real n (Function.comp Real.log fun y => Neg.neg y) x
      ⊢ ContDiffAt Real n Real.log x
    -/
    convert this
    /-
      case h.e'_10
      n : WithTop ENat
      x : Real
      hx✝ : Ne x 0
      A : ∀ (y : Real), LT.lt 0 y → ContDiffAt Real n Real.log y
      hx : LT.lt x 0
      this : ContDiffAt Real n (Function.comp Real.log fun y => Neg.neg y) x
      ⊢ Eq Real.log (Function.comp Real.log fun y => Neg.neg y)
    -/
    ext x
    /-
      case h.e'_10.h
      n : WithTop ENat
      x✝ : Real
      hx✝ : Ne x✝ 0
      A : ∀ (y : Real), LT.lt 0 y → ContDiffAt Real n Real.log y
      hx : LT.lt x✝ 0
      this : ContDiffAt Real n (Function.comp Real.log fun y => Neg.neg y) x✝
      x : Real
      ⊢ Eq (Real.log x) (Function.comp Real.log (fun y => Neg.neg y) x)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : WithTop ENat
      x : Real
      hx✝ : Ne x 0
      A : ∀ (y : Real), LT.lt 0 y → ContDiffAt Real n Real.log y
      hx : LT.lt 0 x
      ⊢ ContDiffAt Real n Real.log x
    -/
  · exact A x hx
    /-
      🎉 no goals
    -/


theorem contDiffOn_log {n : WithTop ℕ∞} : ContDiffOn ℝ n log {0}ᶜ := by
  /-
    n : WithTop ENat
    ⊢ ContDiffOn Real n Real.log (HasCompl.compl (Singleton.singleton 0))
  -/
  intro x hx
  /-
    n : WithTop ENat
    x : Real
    hx : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
    ⊢ ContDiffWithinAt Real n Real.log (HasCompl.compl (Singleton.singleton 0)) x
  -/
  simp only [mem_compl_iff, mem_singleton_iff] at hx
  /-
    n : WithTop ENat
    x : Real
    hx : Not (Eq x 0)
    ⊢ ContDiffWithinAt Real n Real.log (HasCompl.compl (Singleton.singleton 0)) x
  -/
  exact (contDiffAt_log.2 hx).contDiffWithinAt
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.log (hf : HasDerivWithinAt f f' s x) (hx : f x ≠ 0) :
    HasDerivWithinAt (fun y => log (f y)) (f' / f x) s x := by
  /-
    f : Real → Real
    x f' : Real
    s : Set Real
    hf : HasDerivWithinAt f f' s x
    hx : Ne (f x) 0
    ⊢ HasDerivWithinAt (fun y => Real.log (f y)) (HDiv.hDiv f' (f x)) s x
  -/
  rw [div_eq_inv_mul]
  /-
    f : Real → Real
    x f' : Real
    s : Set Real
    hf : HasDerivWithinAt f f' s x
    hx : Ne (f x) 0
    ⊢ HasDerivWithinAt (fun y => Real.log (f y)) (HMul.hMul (Inv.inv (f x)) f') s x
  -/
  exact (hasDerivAt_log hx).comp_hasDerivWithinAt x hf
  /-
    🎉 no goals
  -/


theorem HasDerivAt.log (hf : HasDerivAt f f' x) (hx : f x ≠ 0) :
    HasDerivAt (fun y => log (f y)) (f' / f x) x := by
  /-
    f : Real → Real
    x f' : Real
    hf : HasDerivAt f f' x
    hx : Ne (f x) 0
    ⊢ HasDerivAt (fun y => Real.log (f y)) (HDiv.hDiv f' (f x)) x
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    f : Real → Real
    x f' : Real
    hf : HasDerivWithinAt f f' Set.univ x
    hx : Ne (f x) 0
    ⊢ HasDerivWithinAt (fun y => Real.log (f y)) (HDiv.hDiv f' (f x)) Set.univ x
  -/
  exact hf.log hx
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.log (hf : HasStrictDerivAt f f' x) (hx : f x ≠ 0) :
    HasStrictDerivAt (fun y => log (f y)) (f' / f x) x := by
  /-
    f : Real → Real
    x f' : Real
    hf : HasStrictDerivAt f f' x
    hx : Ne (f x) 0
    ⊢ HasStrictDerivAt (fun y => Real.log (f y)) (HDiv.hDiv f' (f x)) x
  -/
  rw [div_eq_inv_mul]
  /-
    f : Real → Real
    x f' : Real
    hf : HasStrictDerivAt f f' x
    hx : Ne (f x) 0
    ⊢ HasStrictDerivAt (fun y => Real.log (f y)) (HMul.hMul (Inv.inv (f x)) f') x
  -/
  exact (hasStrictDerivAt_log hx).comp x hf
  /-
    🎉 no goals
  -/


theorem derivWithin.log (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0)
    (hxs : UniqueDiffWithinAt ℝ s x) :
    derivWithin (fun x => log (f x)) s x = derivWithin f s x / f x :=
  (hf.hasDerivWithinAt.log hx).derivWithin hxs


@[simp]
theorem deriv.log (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    deriv (fun x => log (f x)) x = deriv f x / f x :=
  (hf.hasDerivAt.log hx).deriv


/-- The derivative of `log ∘ f` is the logarithmic derivative provided `f` is differentiable and
`f x  ≠ 0`. -/
lemma Real.deriv_log_comp_eq_logDeriv {f : ℝ → ℝ} {x : ℝ} (h₁ : DifferentiableAt ℝ f x)
    (h₂ : f x ≠ 0) : deriv (log ∘ f) x = logDeriv f x := by
  /-
    f : Real → Real
    x : Real
    h₁ : DifferentiableAt Real f x
    h₂ : Ne (f x) 0
    ⊢ Eq (deriv (Function.comp Real.log f) x) (logDeriv f x)
  -/
  simp only [ne_eq, logDeriv, Pi.div_apply, ← deriv.log h₁ h₂]
  /-
    f : Real → Real
    x : Real
    h₁ : DifferentiableAt Real f x
    h₂ : Ne (f x) 0
    ⊢ Eq (deriv (Function.comp Real.log f) x) (deriv (fun x => Real.log (f x)) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem HasFDerivWithinAt.log (hf : HasFDerivWithinAt f f' s x) (hx : f x ≠ 0) :
    HasFDerivWithinAt (fun x => log (f x)) ((f x)⁻¹ • f') s x :=
  (hasDerivAt_log hx).comp_hasFDerivWithinAt x hf


theorem HasFDerivAt.log (hf : HasFDerivAt f f' x) (hx : f x ≠ 0) :
    HasFDerivAt (fun x => log (f x)) ((f x)⁻¹ • f') x :=
  (hasDerivAt_log hx).comp_hasFDerivAt x hf


theorem HasStrictFDerivAt.log (hf : HasStrictFDerivAt f f' x) (hx : f x ≠ 0) :
    HasStrictFDerivAt (fun x => log (f x)) ((f x)⁻¹ • f') x :=
  (hasStrictDerivAt_log hx).comp_hasStrictFDerivAt x hf


theorem DifferentiableWithinAt.log (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0) :
    DifferentiableWithinAt ℝ (fun x => log (f x)) s x :=
  (hf.hasFDerivWithinAt.log hx).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.log (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    DifferentiableAt ℝ (fun x => log (f x)) x :=
  (hf.hasFDerivAt.log hx).differentiableAt


theorem ContDiffAt.log {n} (hf : ContDiffAt ℝ n f x) (hx : f x ≠ 0) :
    ContDiffAt ℝ n (fun x => log (f x)) x :=
  (contDiffAt_log.2 hx).comp x hf


theorem ContDiffWithinAt.log {n} (hf : ContDiffWithinAt ℝ n f s x) (hx : f x ≠ 0) :
    ContDiffWithinAt ℝ n (fun x => log (f x)) s x :=
  (contDiffAt_log.2 hx).comp_contDiffWithinAt x hf


theorem ContDiffOn.log {n} (hf : ContDiffOn ℝ n f s) (hs : ∀ x ∈ s, f x ≠ 0) :
    ContDiffOn ℝ n (fun x => log (f x)) s := fun x hx => (hf x hx).log (hs x hx)


theorem ContDiff.log {n} (hf : ContDiff ℝ n f) (h : ∀ x, f x ≠ 0) :
    ContDiff ℝ n fun x => log (f x) :=
  contDiff_iff_contDiffAt.2 fun x => hf.contDiffAt.log (h x)


@[fun_prop]
theorem DifferentiableOn.log (hf : DifferentiableOn ℝ f s) (hx : ∀ x ∈ s, f x ≠ 0) :
    DifferentiableOn ℝ (fun x => log (f x)) s := fun x h => (hf x h).log (hx x h)


@[simp, fun_prop]
theorem Differentiable.log (hf : Differentiable ℝ f) (hx : ∀ x, f x ≠ 0) :
    Differentiable ℝ fun x => log (f x) := fun x => (hf x).log (hx x)


theorem fderivWithin.log (hf : DifferentiableWithinAt ℝ f s x) (hx : f x ≠ 0)
    (hxs : UniqueDiffWithinAt ℝ s x) :
    fderivWithin ℝ (fun x => log (f x)) s x = (f x)⁻¹ • fderivWithin ℝ f s x :=
  (hf.hasFDerivWithinAt.log hx).fderivWithin hxs


@[simp]
theorem fderiv.log (hf : DifferentiableAt ℝ f x) (hx : f x ≠ 0) :
    fderiv ℝ (fun x => log (f x)) x = (f x)⁻¹ • fderiv ℝ f x :=
  (hf.hasFDerivAt.log hx).fderiv


/-- The function `x * log (1 + t / x)` tends to `t` at `+∞`. -/
theorem tendsto_mul_log_one_plus_div_atTop (t : ℝ) :
    Tendsto (fun x => x * log (1 + t / x)) atTop (𝓝 t) := by
  have h₁ : Tendsto (fun h => h⁻¹ * log (1 + t * h)) (𝓝[≠] 0) (𝓝 t) := by
    simpa [hasDerivAt_iff_tendsto_slope, slope_fun_def] using
      (((hasDerivAt_id (0 : ℝ)).const_mul t).const_add 1).log (by simp)
  have h₂ : Tendsto (fun x : ℝ => x⁻¹) atTop (𝓝[≠] 0) :=
    tendsto_inv_atTop_nhdsGT_zero.mono_right (nhdsGT_le_nhdsNE _)
  /-
    t : Real
    h₁ : Filter.Tendsto (fun h => HMul.hMul (Inv.inv h) (Real.log (HAdd.hAdd 1 (HM …
    h₂ : Filter.Tendsto (fun x => Inv.inv x) Filter.atTop (nhdsWithin 0 (HasCompl. …
    ⊢ Filter.Tendsto (fun x => HMul.hMul x (Real.log (HAdd.hAdd 1 (HDiv.hDiv t x)) …
  -/
  simpa only [Function.comp_def, inv_inv] using h₁.comp h₂
  /-
    🎉 no goals
  -/


/-- A crude lemma estimating the difference between `log (1-x)` and its Taylor series at `0`,
where the main point of the bound is that it tends to `0`. The goal is to deduce the series
expansion of the logarithm, in `hasSum_pow_div_log_of_abs_lt_1`.

Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: use one of generic theorems about Taylor's series
to prove this estimate.
-/
theorem abs_log_sub_add_sum_range_le {x : ℝ} (h : |x| < 1) (n : ℕ) :
    |(∑ i ∈ range n, x ^ (i + 1) / (i + 1)) + log (1 - x)| ≤ |x| ^ (n + 1) / (1 - |x|) := by
  /- For the proof, we show that the derivative of the function to be estimated is small,
    and then apply the mean value inequality. -/
  /-
    x : Real
    h : LT.lt (abs x) 1
    n : Nat
    ⊢ LE.le (abs (HAdd.hAdd ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow x  …
  -/
  let F : ℝ → ℝ := fun x => (∑ i ∈ range n, x ^ (i + 1) / (i + 1)) + log (1 - x)
  /-
    x : Real
    h : LT.lt (abs x) 1
    n : Nat
    F : Real → Real := fun x => HAdd.hAdd ((Finset.range n).sum fun i => HDiv.hDiv …
    ⊢ LE.le (abs (HAdd.hAdd ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow x  …
  -/
  let F' : ℝ → ℝ := fun x ↦ -x ^ n / (1 - x)
  -- Porting note: In `mathlib3`, the proof used `deriv`/`DifferentiableAt`. `simp` failed to
  -- compute `deriv`, so I changed the proof to use `HasDerivAt` instead
  -- First step: compute the derivative of `F`
  have A : ∀ y ∈ Ioo (-1 : ℝ) 1, HasDerivAt F (F' y) y := fun y hy ↦ by
    have : HasDerivAt F ((∑ i ∈ range n, ↑(i + 1) * y ^ i / (↑i + 1)) + (-1) / (1 - y)) y :=
      .add (.sum fun i _ ↦ (hasDerivAt_pow (i + 1) y).div_const ((i : ℝ) + 1))
        (((hasDerivAt_id y).const_sub _).log <| sub_ne_zero.2 hy.2.ne')
    convert this using 1
    calc
      -y ^ n / (1 - y) = ∑ i ∈ Finset.range n, y ^ i + -1 / (1 - y) := by
        field_simp [geom_sum_eq hy.2.ne, sub_ne_zero.2 hy.2.ne, sub_ne_zero.2 hy.2.ne']
        ring
      _ = ∑ i ∈ Finset.range n, ↑(i + 1) * y ^ i / (↑i + 1) + -1 / (1 - y) := by
        congr with i
        rw [Nat.cast_succ, mul_div_cancel_left₀ _ (Nat.cast_add_one_pos i).ne']
  -- second step: show that the derivative of `F` is small
  have B : ∀ y ∈ Icc (-|x|) |x|, |F' y| ≤ |x| ^ n / (1 - |x|) := fun y hy ↦
    calc
      |F' y| = |y| ^ n / |1 - y| := by simp [F', abs_div]
      _ ≤ |x| ^ n / (1 - |x|) := by
        have : |y| ≤ |x| := abs_le.2 hy
        have : 1 - |x| ≤ |1 - y| := le_trans (by linarith [hy.2]) (le_abs_self _)
        gcongr
        exact sub_pos.2 h
  -- third step: apply the mean value inequality
  have C : ‖F x - F 0‖ ≤ |x| ^ n / (1 - |x|) * ‖x - 0‖ := by
    refine Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (fun y hy ↦ (A _ ?_).hasDerivWithinAt) B (convex_Icc _ _) ?_ ?_
    · exact Icc_subset_Ioo (neg_lt_neg h) h hy
    · simp
    · simp [le_abs_self x, neg_le.mp (neg_le_abs x)]
  -- fourth step: conclude by massaging the inequality of the third step
  /-
    x : Real
    h : LT.lt (abs x) 1
    n : Nat
    F : Real → Real := fun x => HAdd.hAdd ((Finset.range n).sum fun i => HDiv.hDiv …
    F' : Real → Real := fun x => HDiv.hDiv (Neg.neg (HPow.hPow x n)) (HSub.hSub 1 x)
    A : ∀ (y : Real), Membership.mem (Set.Ioo (-1) 1) y → HasDerivAt F (F' y) y
    B : ∀ (y : Real), Membership.mem (Set.Icc (Neg.neg (abs x)) (abs x)) y → LE.le …
    C : LE.le (Norm.norm (HSub.hSub (F x) (F 0))) (HMul.hMul (HDiv.hDiv (HPow.hPow …
    ⊢ LE.le (abs (HAdd.hAdd ((Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow x  …
  -/
  simpa [F, div_mul_eq_mul_div, pow_succ] using C
  /-
    🎉 no goals
  -/


/-- Power series expansion of the logarithm around `1`. -/
theorem hasSum_pow_div_log_of_abs_lt_one {x : ℝ} (h : |x| < 1) :
    HasSum (fun n : ℕ => x ^ (n + 1) / (n + 1)) (-log (1 - x)) := by
  /-
    x : Real
    h : LT.lt (abs x) 1
    ⊢ HasSum (fun n => HDiv.hDiv (HPow.hPow x (HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)) …
  -/
  rw [Summable.hasSum_iff_tendsto_nat]
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow  …
    -/
  · show Tendsto (fun n : ℕ => ∑ i ∈ range n, x ^ (i + 1) / (i + 1)) atTop (𝓝 (-log (1 - x)))
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HDiv.hDiv (HPow.hPow  …
    -/
    rw [tendsto_iff_norm_sub_tendsto_zero]
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub ((Finset.range e).sum fun i => …
    -/
    simp only [norm_eq_abs, sub_neg_eq_add]
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (fun e => abs (HAdd.hAdd ((Finset.range e).sum fun i => HDiv. …
    -/
    refine squeeze_zero (fun n => abs_nonneg _) (abs_log_sub_add_sum_range_le h) ?_
    suffices Tendsto (fun t : ℕ => |x| ^ (t + 1) / (1 - |x|)) atTop (𝓝 (|x| * 0 / (1 - |x|))) by
      simpa
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (fun t => HDiv.hDiv (HPow.hPow (abs x) (HAdd.hAdd t 1)) (HSub …
    -/
    simp only [pow_succ']
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (fun t => HDiv.hDiv (HMul.hMul (abs x) (HPow.hPow (abs x) t)) …
    -/
    refine (tendsto_const_nhds.mul ?_).div_const _
    /-
      x : Real
      h : LT.lt (abs x) 1
      ⊢ Filter.Tendsto (HPow.hPow (abs x)) Filter.atTop (nhds 0)
    -/
    exact tendsto_pow_atTop_nhds_zero_of_lt_one (abs_nonneg _) h
    /-
      🎉 no goals
    -/
  /-
    x : Real
    h : LT.lt (abs x) 1
    ⊢ Summable fun n => HDiv.hDiv (HPow.hPow x (HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)
  -/
  show Summable fun n : ℕ => x ^ (n + 1) / (n + 1)
  /-
    x : Real
    h : LT.lt (abs x) 1
    ⊢ Summable fun n => HDiv.hDiv (HPow.hPow x (HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)
  -/
  refine .of_norm_bounded _ (summable_geometric_of_lt_one (abs_nonneg _) h) fun i => ?_
  calc
    ‖x ^ (i + 1) / (i + 1)‖ = |x| ^ (i + 1) / (i + 1) := by
      have : (0 : ℝ) ≤ i + 1 := le_of_lt (Nat.cast_add_one_pos i)
      rw [norm_eq_abs, abs_div, ← pow_abs, abs_of_nonneg this]
    _ ≤ |x| ^ (i + 1) / (0 + 1) := by
      gcongr
      exact i.cast_nonneg
    _ ≤ |x| ^ i := by
      simpa [pow_succ] using mul_le_of_le_one_right (pow_nonneg (abs_nonneg x) i) (le_of_lt h)


/-- Power series expansion of `log(1 + x) - log(1 - x)` for `|x| < 1`. -/
theorem hasSum_log_sub_log_of_abs_lt_one {x : ℝ} (h : |x| < 1) :
    HasSum (fun k : ℕ => (2 : ℝ) * (1 / (2 * k + 1)) * x ^ (2 * k + 1))
      (log (1 + x) - log (1 - x)) := by
  /-
    x : Real
    h : LT.lt (abs x) 1
    ⊢ HasSum (fun k => HMul.hMul (HMul.hMul 2 (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 …
  -/
  set term := fun n : ℕ => -1 * ((-x) ^ (n + 1) / ((n : ℝ) + 1)) + x ^ (n + 1) / (n + 1)
  have h_term_eq_goal :
      term ∘ (2 * ·) = fun k : ℕ => 2 * (1 / (2 * k + 1)) * x ^ (2 * k + 1) := by
    ext n
    dsimp only [term, (· ∘ ·)]
    rw [Odd.neg_pow (⟨n, rfl⟩ : Odd (2 * n + 1)) x]
    push_cast
    ring_nf
  /-
    x : Real
    h : LT.lt (abs x) 1
    term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
    h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
    ⊢ HasSum (fun k => HMul.hMul (HMul.hMul 2 (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 …
  -/
  rw [← h_term_eq_goal, (mul_right_injective₀ (two_ne_zero' ℕ)).hasSum_iff]
    /-
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      ⊢ HasSum term (HSub.hSub (Real.log (HAdd.hAdd 1 x)) (Real.log (HSub.hSub 1 x)))
    -/
  · have h₁ := (hasSum_pow_div_log_of_abs_lt_one (Eq.trans_lt (abs_neg x) h)).mul_left (-1)
    /-
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      h₁ : HasSum (fun i => HMul.hMul (-1) (HDiv.hDiv (HPow.hPow (Neg.neg x) (HAdd.h …
      ⊢ HasSum term (HSub.hSub (Real.log (HAdd.hAdd 1 x)) (Real.log (HSub.hSub 1 x)))
    -/
    convert h₁.add (hasSum_pow_div_log_of_abs_lt_one h) using 1
    /-
      case h.e'_6
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      h₁ : HasSum (fun i => HMul.hMul (-1) (HDiv.hDiv (HPow.hPow (Neg.neg x) (HAdd.h …
      ⊢ Eq (HSub.hSub (Real.log (HAdd.hAdd 1 x)) (Real.log (HSub.hSub 1 x))) (HAdd.h …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
    /-
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      ⊢ ∀ (x : Nat), Not (Membership.mem (Set.range fun x => HMul.hMul 2 x) x) → Eq  …
    -/
  · intro m hm
    /-
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      m : Nat
      hm : Not (Membership.mem (Set.range fun x => HMul.hMul 2 x) m)
      ⊢ Eq (term m) 0
    -/
    rw [range_two_mul, Set.mem_setOf_eq, ← Nat.even_add_one] at hm
    /-
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      m : Nat
      hm : Even (HAdd.hAdd m 1)
      ⊢ Eq (term m) 0
    -/
    dsimp [term]
    /-
      x : Real
      h : LT.lt (abs x) 1
      term : Nat → Real := fun n => HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow  …
      h_term_eq_goal : Eq (Function.comp term fun x => HMul.hMul 2 x) fun k => HMul. …
      m : Nat
      hm : Even (HAdd.hAdd m 1)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (-1) (HDiv.hDiv (HPow.hPow (Neg.neg x) (HAdd.hAdd m …
    -/
    rw [Even.neg_pow hm, neg_one_mul, neg_add_cancel]
    /-
      🎉 no goals
    -/


/-- Expansion of `log (1 + a⁻¹)` as a series in powers of `1 / (2 * a + 1)`. -/
theorem hasSum_log_one_add_inv {a : ℝ} (h : 0 < a) :
    HasSum (fun k : ℕ => (2 : ℝ) * (1 / (2 * k + 1)) * (1 / (2 * a + 1)) ^ (2 * k + 1))
      (log (1 + a⁻¹)) := by
  have h₁ : |1 / (2 * a + 1)| < 1 := by
    rw [abs_of_pos, div_lt_one]
    · linarith
    · linarith
    · exact div_pos one_pos (by linarith)
  /-
    a : Real
    h : LT.lt 0 a
    h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
    ⊢ HasSum (fun k => HMul.hMul (HMul.hMul 2 (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 …
  -/
  convert hasSum_log_sub_log_of_abs_lt_one h₁ using 1
  /-
    case h.e'_6
    a : Real
    h : LT.lt 0 a
    h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
    ⊢ Eq (Real.log (HAdd.hAdd 1 (Inv.inv a))) (HSub.hSub (Real.log (HAdd.hAdd 1 (H …
  -/
  have h₂ : (2 : ℝ) * a + 1 ≠ 0 := by linarith
  /-
    case h.e'_6
    a : Real
    h : LT.lt 0 a
    h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
    h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
    ⊢ Eq (Real.log (HAdd.hAdd 1 (Inv.inv a))) (HSub.hSub (Real.log (HAdd.hAdd 1 (H …
  -/
  have h₃ := h.ne'
  /-
    case h.e'_6
    a : Real
    h : LT.lt 0 a
    h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
    h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
    h₃ : Ne a 0
    ⊢ Eq (Real.log (HAdd.hAdd 1 (Inv.inv a))) (HSub.hSub (Real.log (HAdd.hAdd 1 (H …
  -/
  rw [← log_div]
    /-
      case h.e'_6
      a : Real
      h : LT.lt 0 a
      h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
      h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
      h₃ : Ne a 0
      ⊢ Eq (Real.log (HAdd.hAdd 1 (Inv.inv a))) (Real.log (HDiv.hDiv (HAdd.hAdd 1 (H …
    -/
  · congr
    /-
      case h.e'_6.e_x
      a : Real
      h : LT.lt 0 a
      h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
      h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
      h₃ : Ne a 0
      ⊢ Eq (HAdd.hAdd 1 (Inv.inv a)) (HDiv.hDiv (HAdd.hAdd 1 (HDiv.hDiv 1 (HAdd.hAdd …
    -/
    field_simp
    /-
      case h.e'_6.e_x
      a : Real
      h : LT.lt 0 a
      h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
      h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
      h₃ : Ne a 0
      ⊢ Eq (HMul.hMul (HAdd.hAdd a 1) (HMul.hMul 2 a)) (HMul.hMul (HAdd.hAdd (HAdd.h …
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6.hx
      a : Real
      h : LT.lt 0 a
      h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
      h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
      h₃ : Ne a 0
      ⊢ Ne (HAdd.hAdd 1 (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 0
    -/
  · field_simp
    /-
      case h.e'_6.hx
      a : Real
      h : LT.lt 0 a
      h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
      h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
      h₃ : Ne a 0
      ⊢ Not (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 a) 1) 1) 0)
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6.hy
      a : Real
      h : LT.lt 0 a
      h₁ : LT.lt (abs (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 1
      h₂ : Ne (HAdd.hAdd (HMul.hMul 2 a) 1) 0
      h₃ : Ne a 0
      ⊢ Ne (HSub.hSub 1 (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul 2 a) 1))) 0
    -/
  · field_simp
    /-
      🎉 no goals
    -/


