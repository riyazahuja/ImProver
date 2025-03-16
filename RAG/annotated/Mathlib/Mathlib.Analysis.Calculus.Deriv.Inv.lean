theorem hasStrictDerivAt_inv (hx : x ≠ 0) : HasStrictDerivAt Inv.inv (-(x ^ 2)⁻¹) x := by
  suffices
    (fun p : 𝕜 × 𝕜 => (p.1 - p.2) * ((x * x)⁻¹ - (p.1 * p.2)⁻¹)) =o[𝓝 (x, x)] fun p =>
      (p.1 - p.2) * 1 by
    refine .of_isLittleO <| this.congr' ?_ (Eventually.of_forall fun _ => mul_one _)
    refine Eventually.mono ((isOpen_ne.prod isOpen_ne).mem_nhds ⟨hx, hx⟩) ?_
    rintro ⟨y, z⟩ ⟨hy, hz⟩
    simp only [mem_setOf_eq] at hy hz
    -- hy : y ≠ 0, hz : z ≠ 0
    field_simp [hx, hy, hz]
    ring
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    hx : Ne x 0
    ⊢ Asymptotics.IsLittleO (nhds { fst := x, snd := x }) (fun p => HMul.hMul (HSu …
  -/
  refine (isBigO_refl (fun p : 𝕜 × 𝕜 => p.1 - p.2) _).mul_isLittleO ((isLittleO_one_iff 𝕜).2 ?_)
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    hx : Ne x 0
    ⊢ Filter.Tendsto (fun p => HSub.hSub (Inv.inv (HMul.hMul x x)) (Inv.inv (HMul. …
  -/
  rw [← sub_self (x * x)⁻¹]
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    hx : Ne x 0
    ⊢ Filter.Tendsto (fun p => HSub.hSub (Inv.inv (HMul.hMul x x)) (Inv.inv (HMul. …
  -/
  exact tendsto_const_nhds.sub ((continuous_mul.tendsto (x, x)).inv₀ <| mul_ne_zero hx hx)
  /-
    🎉 no goals
  -/


theorem hasDerivAt_inv (x_ne_zero : x ≠ 0) : HasDerivAt (fun y => y⁻¹) (-(x ^ 2)⁻¹) x :=
  (hasStrictDerivAt_inv x_ne_zero).hasDerivAt


theorem hasDerivWithinAt_inv (x_ne_zero : x ≠ 0) (s : Set 𝕜) :
    HasDerivWithinAt (fun x => x⁻¹) (-(x ^ 2)⁻¹) s x :=
  (hasDerivAt_inv x_ne_zero).hasDerivWithinAt


theorem differentiableAt_inv_iff : DifferentiableAt 𝕜 (fun x => x⁻¹) x ↔ x ≠ 0 :=
  ⟨fun H => NormedField.continuousAt_inv.1 H.continuousAt, fun H =>
    (hasDerivAt_inv H).differentiableAt⟩


theorem deriv_inv : deriv (fun x => x⁻¹) x = -(x ^ 2)⁻¹ := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    ⊢ Eq (deriv (fun x => Inv.inv x) x) (Neg.neg (Inv.inv (HPow.hPow x 2)))
  -/
  rcases eq_or_ne x 0 with (rfl | hne)
    /-
      case inl
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      ⊢ Eq (deriv (fun x => Inv.inv x) 0) (Neg.neg (Inv.inv (HPow.hPow 0 2)))
    -/
  · simp [deriv_zero_of_not_differentiableAt (mt differentiableAt_inv_iff.1 (not_not.2 rfl))]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      x : 𝕜
      hne : Ne x 0
      ⊢ Eq (deriv (fun x => Inv.inv x) x) (Neg.neg (Inv.inv (HPow.hPow x 2)))
    -/
  · exact (hasDerivAt_inv hne).deriv
    /-
      🎉 no goals
    -/


@[simp]
theorem deriv_inv' : (deriv fun x : 𝕜 => x⁻¹) = fun x => -(x ^ 2)⁻¹ :=
  funext fun _ => deriv_inv


theorem derivWithin_inv (x_ne_zero : x ≠ 0) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x => x⁻¹) s x = -(x ^ 2)⁻¹ := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    x_ne_zero : Ne x 0
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (derivWithin (fun x => Inv.inv x) s x) (Neg.neg (Inv.inv (HPow.hPow x 2)))
  -/
  rw [DifferentiableAt.derivWithin (differentiableAt_inv x_ne_zero) hxs]
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    x_ne_zero : Ne x 0
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (deriv Inv.inv x) (Neg.neg (Inv.inv (HPow.hPow x 2)))
  -/
  exact deriv_inv
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_inv (x_ne_zero : x ≠ 0) :
    HasFDerivAt (fun x => x⁻¹) (smulRight (1 : 𝕜 →L[𝕜] 𝕜) (-(x ^ 2)⁻¹) : 𝕜 →L[𝕜] 𝕜) x :=
  hasDerivAt_inv x_ne_zero


theorem hasStrictFDerivAt_inv (x_ne_zero : x ≠ 0) :
    HasStrictFDerivAt (fun x => x⁻¹) (smulRight (1 : 𝕜 →L[𝕜] 𝕜) (-(x ^ 2)⁻¹) : 𝕜 →L[𝕜] 𝕜) x :=
  hasStrictDerivAt_inv x_ne_zero


theorem hasFDerivWithinAt_inv (x_ne_zero : x ≠ 0) :
    HasFDerivWithinAt (fun x => x⁻¹) (smulRight (1 : 𝕜 →L[𝕜] 𝕜) (-(x ^ 2)⁻¹) : 𝕜 →L[𝕜] 𝕜) s x :=
  (hasFDerivAt_inv x_ne_zero).hasFDerivWithinAt


theorem fderiv_inv : fderiv 𝕜 (fun x => x⁻¹) x = smulRight (1 : 𝕜 →L[𝕜] 𝕜) (-(x ^ 2)⁻¹) := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    ⊢ Eq (fderiv 𝕜 (fun x => Inv.inv x) x) (ContinuousLinearMap.smulRight 1 (Neg.n …
  -/
  rw [← deriv_fderiv, deriv_inv]
  /-
    🎉 no goals
  -/


theorem fderivWithin_inv (x_ne_zero : x ≠ 0) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun x => x⁻¹) s x = smulRight (1 : 𝕜 →L[𝕜] 𝕜) (-(x ^ 2)⁻¹) := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    x_ne_zero : Ne x 0
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (fun x => Inv.inv x) s x) (ContinuousLinearMap.smulRight  …
  -/
  rw [DifferentiableAt.fderivWithin (differentiableAt_inv x_ne_zero) hxs]
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    x_ne_zero : Ne x 0
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderiv 𝕜 Inv.inv x) (ContinuousLinearMap.smulRight 1 (Neg.neg (Inv.inv ( …
  -/
  exact fderiv_inv
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.inv (hc : HasDerivWithinAt c c' s x) (hx : c x ≠ 0) :
    HasDerivWithinAt (fun y => (c y)⁻¹) (-c' / c x ^ 2) s x := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    c : 𝕜 → 𝕜
    c' : 𝕜
    hc : HasDerivWithinAt c c' s x
    hx : Ne (c x) 0
    ⊢ HasDerivWithinAt (fun y => Inv.inv (c y)) (HDiv.hDiv (Neg.neg c') (HPow.hPow …
  -/
  convert (hasDerivAt_inv hx).comp_hasDerivWithinAt x hc using 1
  /-
    case h.e'_9
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    c : 𝕜 → 𝕜
    c' : 𝕜
    hc : HasDerivWithinAt c c' s x
    hx : Ne (c x) 0
    ⊢ Eq (HDiv.hDiv (Neg.neg c') (HPow.hPow (c x) 2)) (HMul.hMul (Neg.neg (Inv.inv …
  -/
  field_simp
  /-
    🎉 no goals
  -/


theorem HasDerivAt.inv (hc : HasDerivAt c c' x) (hx : c x ≠ 0) :
    HasDerivAt (fun y => (c y)⁻¹) (-c' / c x ^ 2) x := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    c : 𝕜 → 𝕜
    c' : 𝕜
    hc : HasDerivAt c c' x
    hx : Ne (c x) 0
    ⊢ HasDerivAt (fun y => Inv.inv (c y)) (HDiv.hDiv (Neg.neg c') (HPow.hPow (c x) …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    c : 𝕜 → 𝕜
    c' : 𝕜
    hc : HasDerivWithinAt c c' Set.univ x
    hx : Ne (c x) 0
    ⊢ HasDerivWithinAt (fun y => Inv.inv (c y)) (HDiv.hDiv (Neg.neg c') (HPow.hPow …
  -/
  exact hc.inv hx
  /-
    🎉 no goals
  -/


theorem derivWithin_inv' (hc : DifferentiableWithinAt 𝕜 c s x) (hx : c x ≠ 0)
    (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x => (c x)⁻¹) s x = -derivWithin c s x / c x ^ 2 :=
  (hc.hasDerivWithinAt.inv hx).derivWithin hxs


@[simp]
theorem deriv_inv'' (hc : DifferentiableAt 𝕜 c x) (hx : c x ≠ 0) :
    deriv (fun x => (c x)⁻¹) x = -deriv c x / c x ^ 2 :=
  (hc.hasDerivAt.inv hx).deriv


theorem HasDerivWithinAt.div (hc : HasDerivWithinAt c c' s x) (hd : HasDerivWithinAt d d' s x)
    (hx : d x ≠ 0) :
    HasDerivWithinAt (fun y => c y / d y) ((c' * d x - c x * d') / d x ^ 2) s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c d : 𝕜 → 𝕜'
    c' d' : 𝕜'
    hc : HasDerivWithinAt c c' s x
    hd : HasDerivWithinAt d d' s x
    hx : Ne (d x) 0
    ⊢ HasDerivWithinAt (fun y => HDiv.hDiv (c y) (d y)) (HDiv.hDiv (HSub.hSub (HMu …
  -/
  convert hc.mul ((hasDerivAt_inv hx).comp_hasDerivWithinAt x hd) using 1
    /-
      case h.e'_8
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      s : Set 𝕜
      𝕜' : Type u_1
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      c d : 𝕜 → 𝕜'
      c' d' : 𝕜'
      hc : HasDerivWithinAt c c' s x
      hd : HasDerivWithinAt d d' s x
      hx : Ne (d x) 0
      ⊢ Eq (fun y => HDiv.hDiv (c y) (d y)) fun y => HMul.hMul (c y) (Function.comp  …
    -/
  · simp only [div_eq_mul_inv, (· ∘ ·)]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_9
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      s : Set 𝕜
      𝕜' : Type u_1
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      c d : 𝕜 → 𝕜'
      c' d' : 𝕜'
      hc : HasDerivWithinAt c c' s x
      hd : HasDerivWithinAt d d' s x
      hx : Ne (d x) 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HMul.hMul c' (d x)) (HMul.hMul (c x) d')) (HPow.hP …
    -/
  · field_simp
    /-
      case h.e'_9
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      s : Set 𝕜
      𝕜' : Type u_1
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      c d : 𝕜 → 𝕜'
      c' d' : 𝕜'
      hc : HasDerivWithinAt c c' s x
      hd : HasDerivWithinAt d d' s x
      hx : Ne (d x) 0
      ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul c' (d x)) (HMul.hMul (c x) d')) (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem HasStrictDerivAt.div (hc : HasStrictDerivAt c c' x) (hd : HasStrictDerivAt d d' x)
    (hx : d x ≠ 0) : HasStrictDerivAt (fun y => c y / d y) ((c' * d x - c x * d') / d x ^ 2) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c d : 𝕜 → 𝕜'
    c' d' : 𝕜'
    hc : HasStrictDerivAt c c' x
    hd : HasStrictDerivAt d d' x
    hx : Ne (d x) 0
    ⊢ HasStrictDerivAt (fun y => HDiv.hDiv (c y) (d y)) (HDiv.hDiv (HSub.hSub (HMu …
  -/
  convert hc.mul ((hasStrictDerivAt_inv hx).comp x hd) using 1
    /-
      case h.e'_8
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      𝕜' : Type u_1
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      c d : 𝕜 → 𝕜'
      c' d' : 𝕜'
      hc : HasStrictDerivAt c c' x
      hd : HasStrictDerivAt d d' x
      hx : Ne (d x) 0
      ⊢ Eq (fun y => HDiv.hDiv (c y) (d y)) fun y => HMul.hMul (c y) (Function.comp  …
    -/
  · simp only [div_eq_mul_inv, (· ∘ ·)]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_9
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      𝕜' : Type u_1
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      c d : 𝕜 → 𝕜'
      c' d' : 𝕜'
      hc : HasStrictDerivAt c c' x
      hd : HasStrictDerivAt d d' x
      hx : Ne (d x) 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HMul.hMul c' (d x)) (HMul.hMul (c x) d')) (HPow.hP …
    -/
  · field_simp
    /-
      case h.e'_9
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      𝕜' : Type u_1
      inst✝¹ : NontriviallyNormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      c d : 𝕜 → 𝕜'
      c' d' : 𝕜'
      hc : HasStrictDerivAt c c' x
      hd : HasStrictDerivAt d d' x
      hx : Ne (d x) 0
      ⊢ Eq (HMul.hMul (HSub.hSub (HMul.hMul c' (d x)) (HMul.hMul (c x) d')) (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem HasDerivAt.div (hc : HasDerivAt c c' x) (hd : HasDerivAt d d' x) (hx : d x ≠ 0) :
    HasDerivAt (fun y => c y / d y) ((c' * d x - c x * d') / d x ^ 2) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c d : 𝕜 → 𝕜'
    c' d' : 𝕜'
    hc : HasDerivAt c c' x
    hd : HasDerivAt d d' x
    hx : Ne (d x) 0
    ⊢ HasDerivAt (fun y => HDiv.hDiv (c y) (d y)) (HDiv.hDiv (HSub.hSub (HMul.hMul …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_1
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c d : 𝕜 → 𝕜'
    c' d' : 𝕜'
    hc : HasDerivWithinAt c c' Set.univ x
    hd : HasDerivWithinAt d d' Set.univ x
    hx : Ne (d x) 0
    ⊢ HasDerivWithinAt (fun y => HDiv.hDiv (c y) (d y)) (HDiv.hDiv (HSub.hSub (HMu …
  -/
  exact hc.div hd hx
  /-
    🎉 no goals
  -/


theorem DifferentiableWithinAt.div (hc : DifferentiableWithinAt 𝕜 c s x)
    (hd : DifferentiableWithinAt 𝕜 d s x) (hx : d x ≠ 0) :
    DifferentiableWithinAt 𝕜 (fun x => c x / d x) s x :=
  (hc.hasDerivWithinAt.div hd.hasDerivWithinAt hx).differentiableWithinAt


@[simp]
theorem DifferentiableAt.div (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x)
    (hx : d x ≠ 0) : DifferentiableAt 𝕜 (fun x => c x / d x) x :=
  (hc.hasDerivAt.div hd.hasDerivAt hx).differentiableAt


theorem DifferentiableOn.div (hc : DifferentiableOn 𝕜 c s) (hd : DifferentiableOn 𝕜 d s)
    (hx : ∀ x ∈ s, d x ≠ 0) : DifferentiableOn 𝕜 (fun x => c x / d x) s := fun x h =>
  (hc x h).div (hd x h) (hx x h)


@[simp]
theorem Differentiable.div (hc : Differentiable 𝕜 c) (hd : Differentiable 𝕜 d) (hx : ∀ x, d x ≠ 0) :
    Differentiable 𝕜 fun x => c x / d x := fun x => (hc x).div (hd x) (hx x)


theorem derivWithin_div (hc : DifferentiableWithinAt 𝕜 c s x) (hd : DifferentiableWithinAt 𝕜 d s x)
    (hx : d x ≠ 0) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x => c x / d x) s x =
      (derivWithin c s x * d x - c x * derivWithin d s x) / d x ^ 2 :=
  (hc.hasDerivWithinAt.div hd.hasDerivWithinAt hx).derivWithin hxs


@[simp]
theorem deriv_div (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x) (hx : d x ≠ 0) :
    deriv (fun x => c x / d x) x = (deriv c x * d x - c x * deriv d x) / d x ^ 2 :=
  (hc.hasDerivAt.div hd.hasDerivAt hx).deriv


