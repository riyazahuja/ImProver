/-- `arsinh` is defined using a logarithm, `arsinh x = log (x + sqrt(1 + x^2))`. -/
@[pp_nodot]
def arsinh (x : ℝ) :=
  log (x + √(1 + x ^ 2))


theorem exp_arsinh (x : ℝ) : exp (arsinh x) = x + √(1 + x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.exp (Real.arsinh x)) (HAdd.hAdd x (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt)
  -/
  apply exp_log
  /-
    case hx
    x : Real
    ⊢ LT.lt 0 (HAdd.hAdd x (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt)
  -/
  rw [← neg_lt_iff_pos_add']
  /-
    case hx
    x : Real
    ⊢ LT.lt (Neg.neg x) (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt
  -/
  apply lt_sqrt_of_sq_lt
  /-
    case hx.h
    x : Real
    ⊢ LT.lt (HPow.hPow (Neg.neg x) 2) (HAdd.hAdd 1 (HPow.hPow x 2))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
                                         /-
                                           ⊢ Eq (Real.arsinh 0) 0
                                         -/
theorem arsinh_zero : arsinh 0 = 0 := by simp [arsinh]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem arsinh_neg (x : ℝ) : arsinh (-x) = -arsinh x := by
  /-
    x : Real
    ⊢ Eq (Real.arsinh (Neg.neg x)) (Neg.neg (Real.arsinh x))
  -/
  rw [← exp_eq_exp, exp_arsinh, exp_neg, exp_arsinh]
  /-
    x : Real
    ⊢ Eq (HAdd.hAdd (Neg.neg x) (HAdd.hAdd 1 (HPow.hPow (Neg.neg x) 2)).sqrt) (Inv …
  -/
  apply eq_inv_of_mul_eq_one_left
  /-
    case h
    x : Real
    ⊢ Eq (HMul.hMul (HAdd.hAdd (Neg.neg x) (HAdd.hAdd 1 (HPow.hPow (Neg.neg x) 2)) …
  -/
  rw [neg_sq, neg_add_eq_sub, add_comm x, mul_comm, ← sq_sub_sq, sq_sqrt, add_sub_cancel_right]
  /-
    case h
    x : Real
    ⊢ LE.le 0 (HAdd.hAdd 1 (HPow.hPow x 2))
  -/
  exact add_nonneg zero_le_one (sq_nonneg _)
  /-
    🎉 no goals
  -/


/-- `arsinh` is the right inverse of `sinh`. -/
@[simp]
theorem sinh_arsinh (x : ℝ) : sinh (arsinh x) = x := by
  /-
    x : Real
    ⊢ Eq (Real.sinh (Real.arsinh x)) x
  -/
  rw [sinh_eq, ← arsinh_neg, exp_arsinh, exp_arsinh, neg_sq]; field_simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem cosh_arsinh (x : ℝ) : cosh (arsinh x) = √(1 + x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.cosh (Real.arsinh x)) (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt
  -/
  rw [← sqrt_sq (cosh_pos _).le, cosh_sq', sinh_arsinh]
  /-
    🎉 no goals
  -/


/-- `sinh` is surjective, `∀ b, ∃ a, sinh a = b`. In this case, we use `a = arsinh b`. -/
theorem sinh_surjective : Surjective sinh :=
  LeftInverse.surjective sinh_arsinh


/-- `sinh` is bijective, both injective and surjective. -/
theorem sinh_bijective : Bijective sinh :=
  ⟨sinh_injective, sinh_surjective⟩


/-- `arsinh` is the left inverse of `sinh`. -/
@[simp]
theorem arsinh_sinh (x : ℝ) : arsinh (sinh x) = x :=
  rightInverse_of_injective_of_leftInverse sinh_injective sinh_arsinh x


/-- `Real.sinh` as an `Equiv`. -/
@[simps]
def sinhEquiv : ℝ ≃ ℝ where
  toFun := sinh
  invFun := arsinh
  left_inv := arsinh_sinh
  right_inv := sinh_arsinh


/-- `Real.sinh` as an `OrderIso`. -/
@[simps! (config := .asFn)]
def sinhOrderIso : ℝ ≃o ℝ where
  toEquiv := sinhEquiv
  map_rel_iff' := @sinh_le_sinh


/-- `Real.sinh` as a `Homeomorph`. -/
@[simps! (config := .asFn)]
def sinhHomeomorph : ℝ ≃ₜ ℝ :=
  sinhOrderIso.toHomeomorph


theorem arsinh_bijective : Bijective arsinh :=
  sinhEquiv.symm.bijective


theorem arsinh_injective : Injective arsinh :=
  sinhEquiv.symm.injective


theorem arsinh_surjective : Surjective arsinh :=
  sinhEquiv.symm.surjective


theorem arsinh_strictMono : StrictMono arsinh :=
  sinhOrderIso.symm.strictMono


@[simp]
theorem arsinh_inj : arsinh x = arsinh y ↔ x = y :=
  arsinh_injective.eq_iff


@[simp]
theorem arsinh_le_arsinh : arsinh x ≤ arsinh y ↔ x ≤ y :=
  sinhOrderIso.symm.le_iff_le


@[gcongr] protected alias ⟨_, GCongr.arsinh_le_arsinh⟩ := arsinh_le_arsinh


@[simp]
theorem arsinh_lt_arsinh : arsinh x < arsinh y ↔ x < y :=
  sinhOrderIso.symm.lt_iff_lt


@[simp]
theorem arsinh_eq_zero_iff : arsinh x = 0 ↔ x = 0 :=
  arsinh_injective.eq_iff' arsinh_zero


@[simp]
                                                       /-
                                                         x : Real
                                                         ⊢ Iff (LE.le 0 (Real.arsinh x)) (LE.le 0 x)
                                                       -/
theorem arsinh_nonneg_iff : 0 ≤ arsinh x ↔ 0 ≤ x := by rw [← sinh_le_sinh, sinh_zero, sinh_arsinh]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
                                                       /-
                                                         x : Real
                                                         ⊢ Iff (LE.le (Real.arsinh x) 0) (LE.le x 0)
                                                       -/
theorem arsinh_nonpos_iff : arsinh x ≤ 0 ↔ x ≤ 0 := by rw [← sinh_le_sinh, sinh_zero, sinh_arsinh]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem arsinh_pos_iff : 0 < arsinh x ↔ 0 < x :=
  lt_iff_lt_of_le_iff_le arsinh_nonpos_iff


@[simp]
theorem arsinh_neg_iff : arsinh x < 0 ↔ x < 0 :=
  lt_iff_lt_of_le_iff_le arsinh_nonneg_iff


theorem hasStrictDerivAt_arsinh (x : ℝ) : HasStrictDerivAt arsinh (√(1 + x ^ 2))⁻¹ x := by
  convert sinhHomeomorph.toPartialHomeomorph.hasStrictDerivAt_symm (mem_univ x) (cosh_pos _).ne'
    (hasStrictDerivAt_sinh _) using 2
  /-
    case h.e'_9.h.e'_3
    x : Real
    ⊢ Eq (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt (Real.cosh (↑Real.sinhHomeomorph.toPar …
  -/
  exact (cosh_arsinh _).symm
  /-
    🎉 no goals
  -/


theorem hasDerivAt_arsinh (x : ℝ) : HasDerivAt arsinh (√(1 + x ^ 2))⁻¹ x :=
  (hasStrictDerivAt_arsinh x).hasDerivAt


theorem differentiable_arsinh : Differentiable ℝ arsinh := fun x =>
  (hasDerivAt_arsinh x).differentiableAt


theorem contDiff_arsinh {n : ℕ∞} : ContDiff ℝ n arsinh :=
  sinhHomeomorph.contDiff_symm_deriv (fun x => (cosh_pos x).ne') hasDerivAt_sinh contDiff_sinh


@[continuity]
theorem continuous_arsinh : Continuous arsinh :=
  sinhHomeomorph.symm.continuous


theorem Filter.Tendsto.arsinh {α : Type*} {l : Filter α} {f : α → ℝ} {a : ℝ}
    (h : Tendsto f l (𝓝 a)) : Tendsto (fun x => arsinh (f x)) l (𝓝 (arsinh a)) :=
  (continuous_arsinh.tendsto _).comp h


nonrec theorem ContinuousAt.arsinh (h : ContinuousAt f a) :
    ContinuousAt (fun x => arsinh (f x)) a :=
  h.arsinh


nonrec theorem ContinuousWithinAt.arsinh (h : ContinuousWithinAt f s a) :
    ContinuousWithinAt (fun x => arsinh (f x)) s a :=
  h.arsinh


theorem ContinuousOn.arsinh (h : ContinuousOn f s) : ContinuousOn (fun x => arsinh (f x)) s :=
  fun x hx => (h x hx).arsinh


theorem Continuous.arsinh (h : Continuous f) : Continuous fun x => arsinh (f x) :=
  continuous_arsinh.comp h


theorem HasStrictFDerivAt.arsinh (hf : HasStrictFDerivAt f f' a) :
    HasStrictFDerivAt (fun x => arsinh (f x)) ((√(1 + f a ^ 2))⁻¹ • f') a :=
  (hasStrictDerivAt_arsinh _).comp_hasStrictFDerivAt a hf


theorem HasFDerivAt.arsinh (hf : HasFDerivAt f f' a) :
    HasFDerivAt (fun x => arsinh (f x)) ((√(1 + f a ^ 2))⁻¹ • f') a :=
  (hasDerivAt_arsinh _).comp_hasFDerivAt a hf


theorem HasFDerivWithinAt.arsinh (hf : HasFDerivWithinAt f f' s a) :
    HasFDerivWithinAt (fun x => arsinh (f x)) ((√(1 + f a ^ 2))⁻¹ • f') s a :=
  (hasDerivAt_arsinh _).comp_hasFDerivWithinAt a hf


theorem DifferentiableAt.arsinh (h : DifferentiableAt ℝ f a) :
    DifferentiableAt ℝ (fun x => arsinh (f x)) a :=
  (differentiable_arsinh _).comp a h


theorem DifferentiableWithinAt.arsinh (h : DifferentiableWithinAt ℝ f s a) :
    DifferentiableWithinAt ℝ (fun x => arsinh (f x)) s a :=
  (differentiable_arsinh _).comp_differentiableWithinAt a h


theorem DifferentiableOn.arsinh (h : DifferentiableOn ℝ f s) :
    DifferentiableOn ℝ (fun x => arsinh (f x)) s := fun x hx => (h x hx).arsinh


theorem Differentiable.arsinh (h : Differentiable ℝ f) : Differentiable ℝ fun x => arsinh (f x) :=
  differentiable_arsinh.comp h


theorem ContDiffAt.arsinh (h : ContDiffAt ℝ n f a) : ContDiffAt ℝ n (fun x => arsinh (f x)) a :=
  contDiff_arsinh.contDiffAt.comp a h


theorem ContDiffWithinAt.arsinh (h : ContDiffWithinAt ℝ n f s a) :
    ContDiffWithinAt ℝ n (fun x => arsinh (f x)) s a :=
  contDiff_arsinh.contDiffAt.comp_contDiffWithinAt a h


theorem ContDiff.arsinh (h : ContDiff ℝ n f) : ContDiff ℝ n fun x => arsinh (f x) :=
  contDiff_arsinh.comp h


theorem ContDiffOn.arsinh (h : ContDiffOn ℝ n f s) : ContDiffOn ℝ n (fun x => arsinh (f x)) s :=
  fun x hx => (h x hx).arsinh


theorem HasStrictDerivAt.arsinh (hf : HasStrictDerivAt f f' a) :
    HasStrictDerivAt (fun x => arsinh (f x)) ((√(1 + f a ^ 2))⁻¹ • f') a :=
  (hasStrictDerivAt_arsinh _).comp a hf


theorem HasDerivAt.arsinh (hf : HasDerivAt f f' a) :
    HasDerivAt (fun x => arsinh (f x)) ((√(1 + f a ^ 2))⁻¹ • f') a :=
  (hasDerivAt_arsinh _).comp a hf


theorem HasDerivWithinAt.arsinh (hf : HasDerivWithinAt f f' s a) :
    HasDerivWithinAt (fun x => arsinh (f x)) ((√(1 + f a ^ 2))⁻¹ • f') s a :=
  (hasDerivAt_arsinh _).comp_hasDerivWithinAt a hf


