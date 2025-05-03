local notation "|" x "|" => Complex.abs x


/-- An element of the unit circle defines a `LinearIsometryEquiv` from `ℂ` to itself, by
rotation. -/
def rotation : Circle →* ℂ ≃ₗᵢ[ℝ] ℂ where
  toFun a :=
    { DistribMulAction.toLinearEquiv ℝ ℂ a with
                                                  /-
                                                    a : Circle
                                                    x : Complex
                                                    ⊢ Eq (Complex.abs (HMul.hMul (↑a) x)) (Complex.abs x)
                                                  -/
      norm_map' := fun x => show |a * x| = |x| by rw [map_mul, Circle.abs_coe, one_mul] }
                                                  /-
                                                    🎉 no goals
                                                  -/
                                            /-
                                              ⊢ ∀ (x : Complex),
                                                  Eq
                                                    (((fun a =>
                                                          let __src := DistribMulAction.toLinearEquiv Real Complex a;
                                                          { toLinearEquiv := __src, norm_map' := ⋯ })
                                                        1)
                                                      x)
                                                    (1 x)
                                            -/
  map_one' := LinearIsometryEquiv.ext <| by simp
                                            /-
                                              🎉 no goals
                                            -/
  map_mul' a b := LinearIsometryEquiv.ext <| mul_smul a b


@[simp]
theorem rotation_apply (a : Circle) (z : ℂ) : rotation a z = a * z :=
  rfl


@[simp]
theorem rotation_symm (a : Circle) : (rotation a).symm = rotation a⁻¹ :=
  LinearIsometryEquiv.ext fun _ => rfl


@[simp]
theorem rotation_trans (a b : Circle) : (rotation a).trans (rotation b) = rotation (b * a) := by
  /-
    a b : Circle
    ⊢ Eq ((rotation a).trans (rotation b)) (rotation (HMul.hMul b a))
  -/
  ext1
  /-
    case h
    a b : Circle
    x✝ : Complex
    ⊢ Eq (((rotation a).trans (rotation b)) x✝) ((rotation (HMul.hMul b a)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem rotation_ne_conjLIE (a : Circle) : rotation a ≠ conjLIE := by
  /-
    a : Circle
    ⊢ Ne (rotation a) Complex.conjLIE
  -/
  intro h
  /-
    a : Circle
    h : Eq (rotation a) Complex.conjLIE
    ⊢ False
  -/
  have h1 : rotation a 1 = conj 1 := LinearIsometryEquiv.congr_fun h 1
  /-
    a : Circle
    h : Eq (rotation a) Complex.conjLIE
    h1 : Eq ((rotation a) 1) ((starRingEnd Complex) 1)
    ⊢ False
  -/
  have hI : rotation a I = conj I := LinearIsometryEquiv.congr_fun h I
  /-
    a : Circle
    h : Eq (rotation a) Complex.conjLIE
    h1 : Eq ((rotation a) 1) ((starRingEnd Complex) 1)
    hI : Eq ((rotation a) Complex.I) ((starRingEnd Complex) Complex.I)
    ⊢ False
  -/
  rw [rotation_apply, RingHom.map_one, mul_one] at h1
  /-
    a : Circle
    h : Eq (rotation a) Complex.conjLIE
    h1 : Eq (↑a) 1
    hI : Eq ((rotation a) Complex.I) ((starRingEnd Complex) Complex.I)
    ⊢ False
  -/
  rw [rotation_apply, conj_I, ← neg_one_mul, mul_left_inj' I_ne_zero, h1, eq_neg_self_iff] at hI
  /-
    a : Circle
    h : Eq (rotation a) Complex.conjLIE
    h1 : Eq (↑a) 1
    hI : Eq 1 0
    ⊢ False
  -/
  exact one_ne_zero hI
  /-
    🎉 no goals
  -/


/-- Takes an element of `ℂ ≃ₗᵢ[ℝ] ℂ` and checks if it is a rotation, returns an element of the
unit circle. -/
@[simps]
def rotationOf (e : ℂ ≃ₗᵢ[ℝ] ℂ) : Circle :=
                               /-
                                 e : LinearIsometryEquiv (RingHom.id Real) Complex Complex
                                 ⊢ Membership.mem (Submonoid.unitSphere Complex) (HDiv.hDiv (e 1) ↑(Complex.abs …
                               -/
  ⟨e 1 / Complex.abs (e 1), by simp [Submonoid.unitSphere, ← Complex.norm_eq_abs]⟩
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem rotationOf_rotation (a : Circle) : rotationOf (rotation a) = a :=
                    /-
                      a : Circle
                      ⊢ Eq ↑(rotationOf (rotation a)) ↑a
                    -/
  Subtype.ext <| by simp
                    /-
                      🎉 no goals
                    -/


theorem rotation_injective : Function.Injective rotation :=
  Function.LeftInverse.injective rotationOf_rotation


theorem LinearIsometry.re_apply_eq_re_of_add_conj_eq (f : ℂ →ₗᵢ[ℝ] ℂ)
    (h₃ : ∀ z, z + conj z = f z + conj (f z)) (z : ℂ) : (f z).re = z.re := by
  simpa [Complex.ext_iff, add_re, add_im, conj_re, conj_im, ← two_mul,
    show (2 : ℝ) ≠ 0 by simp [two_ne_zero]] using (h₃ z).symm


theorem LinearIsometry.im_apply_eq_im_or_neg_of_re_apply_eq_re {f : ℂ →ₗᵢ[ℝ] ℂ}
    (h₂ : ∀ z, (f z).re = z.re) (z : ℂ) : (f z).im = z.im ∨ (f z).im = -z.im := by
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h₂ : ∀ (z : Complex), Eq (f z).re z.re
    z : Complex
    ⊢ Or (Eq (f z).im z.im) (Eq (f z).im (Neg.neg z.im))
  -/
  have h₁ := f.norm_map z
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h₂ : ∀ (z : Complex), Eq (f z).re z.re
    z : Complex
    h₁ : Eq (Norm.norm (f z)) (Norm.norm z)
    ⊢ Or (Eq (f z).im z.im) (Eq (f z).im (Neg.neg z.im))
  -/
  simp only [Complex.abs_def, norm_eq_abs] at h₁
  rwa [Real.sqrt_inj (normSq_nonneg _) (normSq_nonneg _), normSq_apply (f z), normSq_apply z,
    h₂, add_left_cancel_iff, mul_self_eq_mul_self_iff] at h₁


theorem LinearIsometry.im_apply_eq_im {f : ℂ →ₗᵢ[ℝ] ℂ} (h : f 1 = 1) (z : ℂ) :
    z + conj z = f z + conj (f z) := by
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  have : ‖f z - 1‖ = ‖z - 1‖ := by rw [← f.norm_map (z - 1), f.map_sub, h]
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (Norm.norm (HSub.hSub (f z) 1)) (Norm.norm (HSub.hSub z 1))
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  apply_fun fun x => x ^ 2 at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HPow.hPow (Norm.norm (HSub.hSub (f z) 1)) 2) (HPow.hPow (Norm.norm  …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  simp only [norm_eq_abs, ← normSq_eq_abs] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (Complex.normSq (HSub.hSub (f z) 1)) (Complex.normSq (HSub.hSub z 1))
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  rw [← ofReal_inj, ← mul_conj, ← mul_conj] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HMul.hMul (HSub.hSub (f z) 1) ((starRingEnd Complex) (HSub.hSub (f  …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  rw [RingHom.map_sub, RingHom.map_sub] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HMul.hMul (HSub.hSub (f z) 1) (HSub.hSub ((starRingEnd Complex) (f  …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  simp only [sub_mul, mul_sub, one_mul, mul_one] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HSub.hSub (HSub.hSub (HMul.hMul (f z) ((starRingEnd Complex) (f z)) …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  rw [mul_conj, normSq_eq_abs, ← norm_eq_abs, LinearIsometry.norm_map] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HSub.hSub (HSub.hSub (↑(HPow.hPow (Norm.norm z) 2)) ((starRingEnd C …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  rw [mul_conj, normSq_eq_abs, ← norm_eq_abs] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HSub.hSub (HSub.hSub (↑(HPow.hPow (Norm.norm z) 2)) ((starRingEnd C …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  simp only [sub_sub, sub_right_inj, mul_one, ofReal_pow, RingHom.map_one, norm_eq_abs] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HAdd.hAdd ((starRingEnd Complex) (f z)) (HSub.hSub (f z) 1)) (HAdd. …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  simp only [add_sub, sub_left_inj] at this
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    this : Eq (HAdd.hAdd ((starRingEnd Complex) (f z)) (f z)) (HAdd.hAdd ((starRin …
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  rw [add_comm, ← this, add_comm]
  /-
    🎉 no goals
  -/


theorem LinearIsometry.re_apply_eq_re {f : ℂ →ₗᵢ[ℝ] ℂ} (h : f 1 = 1) (z : ℂ) : (f z).re = z.re := by
  /-
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    ⊢ Eq (f z).re z.re
  -/
  apply LinearIsometry.re_apply_eq_re_of_add_conj_eq
  /-
    case h₃
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z : Complex
    ⊢ ∀ (z : Complex), Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) …
  -/
  intro z
  /-
    case h₃
    f : LinearIsometry (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    z✝ z : Complex
    ⊢ Eq (HAdd.hAdd z ((starRingEnd Complex) z)) (HAdd.hAdd (f z) ((starRingEnd Co …
  -/
  apply LinearIsometry.im_apply_eq_im h
  /-
    🎉 no goals
  -/


theorem linear_isometry_complex_aux {f : ℂ ≃ₗᵢ[ℝ] ℂ} (h : f 1 = 1) :
    f = LinearIsometryEquiv.refl ℝ ℂ ∨ f = conjLIE := by
  have h0 : f I = I ∨ f I = -I := by
    simp only [Complex.ext_iff, ← and_or_left, neg_re, I_re, neg_im, neg_zero]
    constructor
    · rw [← I_re]
      exact @LinearIsometry.re_apply_eq_re f.toLinearIsometry h I
    · apply @LinearIsometry.im_apply_eq_im_or_neg_of_re_apply_eq_re f.toLinearIsometry
      intro z
      rw [@LinearIsometry.re_apply_eq_re f.toLinearIsometry h]
  /-
    f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
    h : Eq (f 1) 1
    h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
    ⊢ Or (Eq f (LinearIsometryEquiv.refl Real Complex)) (Eq f Complex.conjLIE)
  -/
  refine h0.imp (fun h' : f I = I => ?_) fun h' : f I = -I => ?_ <;>
      /-
        case refine_1
        f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
        h : Eq (f 1) 1
        h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
        h' : Eq (f Complex.I) Complex.I
        ⊢ Eq f (LinearIsometryEquiv.refl Real Complex)
      -/
      /-
        case refine_1.a
        f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
        h : Eq (f 1) 1
        h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
        h' : Eq (f Complex.I) Complex.I
        ⊢ Eq f.toLinearEquiv (LinearIsometryEquiv.refl Real Complex).toLinearEquiv
      -/
      /-
        case refine_1.a
        f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
        h : Eq (f 1) 1
        h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
        h' : Eq (f Complex.I) Complex.I
        ⊢ ∀ (i : Fin 2), Eq (f.toLinearEquiv (Complex.basisOneI i)) ((LinearIsometryEq …
      -/
      /-
        case refine_1.a
        f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
        h : Eq (f 1) 1
        h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
        h' : Eq (f Complex.I) Complex.I
        i : Fin 2
        ⊢ Eq (f.toLinearEquiv (Complex.basisOneI i)) ((LinearIsometryEquiv.refl Real C …
      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
      /-
        case refine_2.a
        f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
        h : Eq (f 1) 1
        h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
        h' : Eq (f Complex.I) (Neg.neg Complex.I)
        ⊢ ∀ (i : Fin 2), Eq (f.toLinearEquiv (Complex.basisOneI i)) (Complex.conjLIE.t …
      -/
      intro i
      /-
        case refine_2.a
        f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
        h : Eq (f 1) 1
        h0 : Or (Eq (f Complex.I) Complex.I) (Eq (f Complex.I) (Neg.neg Complex.I))
        h' : Eq (f Complex.I) (Neg.neg Complex.I)
        i : Fin 2
        ⊢ Eq (f.toLinearEquiv (Complex.basisOneI i)) (Complex.conjLIE.toLinearEquiv (C …
      -/
                      /-
                        🎉 no goals
                      -/
      fin_cases i <;> simp [h, h']
                      /-
                        🎉 no goals
                      -/


theorem linear_isometry_complex (f : ℂ ≃ₗᵢ[ℝ] ℂ) :
    ∃ a : Circle, f = rotation a ∨ f = conjLIE.trans (rotation a) := by
  /-
    f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
    ⊢ Exists fun a => Or (Eq f (rotation a)) (Eq f (Complex.conjLIE.trans (rotatio …
  -/
  let a : Circle := ⟨f 1, by simp [Submonoid.unitSphere, ← Complex.norm_eq_abs, f.norm_map]⟩
  /-
    f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
    a : Circle := ⟨f 1, ⋯⟩
    ⊢ Exists fun a => Or (Eq f (rotation a)) (Eq f (Complex.conjLIE.trans (rotatio …
  -/
  use a
  /-
    case h
    f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
    a : Circle := ⟨f 1, ⋯⟩
    ⊢ Or (Eq f (rotation a)) (Eq f (Complex.conjLIE.trans (rotation a)))
  -/
  have : (f.trans (rotation a).symm) 1 = 1 := by simpa [a] using rotation_apply a⁻¹ (f 1)
  /-
    case h
    f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
    a : Circle := ⟨f 1, ⋯⟩
    this : Eq ((f.trans (rotation a).symm) 1) 1
    ⊢ Or (Eq f (rotation a)) (Eq f (Complex.conjLIE.trans (rotation a)))
  -/
  refine (linear_isometry_complex_aux this).imp (fun h₁ => ?_) fun h₂ => ?_
    /-
      case h.refine_1
      f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
      a : Circle := ⟨f 1, ⋯⟩
      this : Eq ((f.trans (rotation a).symm) 1) 1
      h₁ : Eq (f.trans (rotation a).symm) (LinearIsometryEquiv.refl Real Complex)
      ⊢ Eq f (rotation a)
    -/
  · simpa using eq_mul_of_inv_mul_eq h₁
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      f : LinearIsometryEquiv (RingHom.id Real) Complex Complex
      a : Circle := ⟨f 1, ⋯⟩
      this : Eq ((f.trans (rotation a).symm) 1) 1
      h₂ : Eq (f.trans (rotation a).symm) Complex.conjLIE
      ⊢ Eq f (Complex.conjLIE.trans (rotation a))
    -/
  · exact eq_mul_of_inv_mul_eq h₂
    /-
      🎉 no goals
    -/


/-- The matrix representation of `rotation a` is equal to the conformal matrix
`!![re a, -im a; im a, re a]`. -/
theorem toMatrix_rotation (a : Circle) :
    LinearMap.toMatrix basisOneI basisOneI (rotation a).toLinearEquiv =
                                                    /-
                                                      a : Circle
                                                      ⊢ Ne (HAdd.hAdd (HPow.hPow (↑a).re 2) (HPow.hPow (↑a).im 2)) 0
                                                    -/
      Matrix.planeConformalMatrix (re a) (im a) (by simp [pow_two, ← normSq_apply]) := by
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    a : Circle
    ⊢ Eq ((LinearMap.toMatrix Complex.basisOneI Complex.basisOneI) ↑(rotation a).t …
  -/
  ext i j
  simp only [LinearMap.toMatrix_apply, coe_basisOneI, LinearEquiv.coe_coe,
    LinearIsometryEquiv.coe_toLinearEquiv, rotation_apply, coe_basisOneI_repr, mul_re, mul_im,
    Matrix.val_planeConformalMatrix, Matrix.of_apply, Matrix.cons_val', Matrix.empty_val',
    Matrix.cons_val_fin_one]
  /-
    case a
    a : Circle
    i j : Fin 2
    ⊢ Eq (Matrix.vecCons (HSub.hSub (HMul.hMul (↑a).re (Matrix.vecCons 1 (Matrix.v …
  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  fin_cases i <;> fin_cases j <;> simp
                                  /-
                                    🎉 no goals
                                  -/


/-- The determinant of `rotation` (as a linear map) is equal to `1`. -/
@[simp]
theorem det_rotation (a : Circle) : LinearMap.det ((rotation a).toLinearEquiv : ℂ →ₗ[ℝ] ℂ) = 1 := by
  /-
    a : Circle
    ⊢ Eq (LinearMap.det ↑(rotation a).toLinearEquiv) 1
  -/
  rw [← LinearMap.det_toMatrix basisOneI, toMatrix_rotation, Matrix.det_fin_two]
  /-
    a : Circle
    ⊢ Eq (HSub.hSub (HMul.hMul (↑(Matrix.planeConformalMatrix (↑a).re (↑a).im ⋯) 0 …
  -/
  simp [← normSq_apply]
  /-
    🎉 no goals
  -/


/-- The determinant of `rotation` (as a linear equiv) is equal to `1`. -/
@[simp]
theorem linearEquiv_det_rotation (a : Circle) : LinearEquiv.det (rotation a).toLinearEquiv = 1 := by
  /-
    a : Circle
    ⊢ Eq (LinearEquiv.det (rotation a).toLinearEquiv) 1
  -/
  rw [← Units.eq_iff, LinearEquiv.coe_det, det_rotation, Units.val_one]
  /-
    🎉 no goals
  -/

