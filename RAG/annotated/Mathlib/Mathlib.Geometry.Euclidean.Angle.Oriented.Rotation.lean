local notation "J" => o.rightAngleRotation


/-- Auxiliary construction to build a rotation by the oriented angle `θ`. -/
def rotationAux (θ : Real.Angle) : V →ₗᵢ[ℝ] V :=
  LinearMap.isometryOfInner
    (Real.Angle.cos θ • LinearMap.id +
      Real.Angle.sin θ • (LinearIsometryEquiv.toLinearEquiv J).toLinearMap)
    (by
      /-
        V : Type u_1
        V' : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : NormedAddCommGroup V'
        inst✝³ : InnerProductSpace Real V
        inst✝² : InnerProductSpace Real V'
        inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
        inst✝ : Fact (Eq (Module.finrank Real V') 2)
        o : Orientation Real V (Fin 2)
        θ : Real.Angle
        ⊢ ∀ (x y : V), Eq (Inner.inner ((HAdd.hAdd (HSMul.hSMul θ.cos LinearMap.id) (H …
      -/
      intro x y
      simp only [RCLike.conj_to_real, id, LinearMap.smul_apply, LinearMap.add_apply,
        LinearMap.id_coe, LinearEquiv.coe_coe, LinearIsometryEquiv.coe_toLinearEquiv,
        Orientation.areaForm_rightAngleRotation_left, Orientation.inner_rightAngleRotation_left,
        Orientation.inner_rightAngleRotation_right, inner_add_left, inner_smul_left,
        inner_add_right, inner_smul_right]
      /-
        V : Type u_1
        V' : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : NormedAddCommGroup V'
        inst✝³ : InnerProductSpace Real V
        inst✝² : InnerProductSpace Real V'
        inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
        inst✝ : Fact (Eq (Module.finrank Real V') 2)
        o : Orientation Real V (Fin 2)
        θ : Real.Angle
        x y : V
        ⊢ Eq (HAdd.hAdd (HMul.hMul θ.cos (HAdd.hAdd (HMul.hMul θ.cos (Inner.inner x y) …
      -/
      linear_combination inner (𝕜 := ℝ) x y * θ.cos_sq_add_sin_sq)
      /-
        🎉 no goals
      -/


@[simp]
theorem rotationAux_apply (θ : Real.Angle) (x : V) :
    o.rotationAux θ x = Real.Angle.cos θ • x + Real.Angle.sin θ • J x :=
  rfl


/-- A rotation by the oriented angle `θ`. -/
def rotation (θ : Real.Angle) : V ≃ₗᵢ[ℝ] V :=
  LinearIsometryEquiv.ofLinearIsometry (o.rotationAux θ)
    (Real.Angle.cos θ • LinearMap.id -
      Real.Angle.sin θ • (LinearIsometryEquiv.toLinearEquiv J).toLinearMap)
    (by
      /-
        V : Type u_1
        V' : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : NormedAddCommGroup V'
        inst✝³ : InnerProductSpace Real V
        inst✝² : InnerProductSpace Real V'
        inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
        inst✝ : Fact (Eq (Module.finrank Real V') 2)
        o : Orientation Real V (Fin 2)
        θ : Real.Angle
        ⊢ Eq ((o.rotationAux θ).comp (HSub.hSub (HSMul.hSMul θ.cos LinearMap.id) (HSMu …
      -/
      ext x
      /-
        case h
        V : Type u_1
        V' : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : NormedAddCommGroup V'
        inst✝³ : InnerProductSpace Real V
        inst✝² : InnerProductSpace Real V'
        inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
        inst✝ : Fact (Eq (Module.finrank Real V') 2)
        o : Orientation Real V (Fin 2)
        θ : Real.Angle
        x : V
        ⊢ Eq (((o.rotationAux θ).comp (HSub.hSub (HSMul.hSMul θ.cos LinearMap.id) (HSM …
      -/
      convert congr_arg (fun t : ℝ => t • x) θ.cos_sq_add_sin_sq using 1
      · simp only [o.rightAngleRotation_rightAngleRotation, o.rotationAux_apply,
          Function.comp_apply, id, LinearEquiv.coe_coe, LinearIsometry.coe_toLinearMap,
          LinearIsometryEquiv.coe_toLinearEquiv, map_smul, map_sub, LinearMap.coe_comp,
          LinearMap.id_coe, LinearMap.smul_apply, LinearMap.sub_apply]
        /-
          case h.e'_2
          V : Type u_1
          V' : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : NormedAddCommGroup V'
          inst✝³ : InnerProductSpace Real V
          inst✝² : InnerProductSpace Real V'
          inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
          inst✝ : Fact (Eq (Module.finrank Real V') 2)
          o : Orientation Real V (Fin 2)
          θ : Real.Angle
          x : V
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul θ.cos (HSub.hSub (HSMul.hSMul θ.cos x) (HSMul.hSM …
        -/
        module
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3
          V : Type u_1
          V' : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : NormedAddCommGroup V'
          inst✝³ : InnerProductSpace Real V
          inst✝² : InnerProductSpace Real V'
          inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
          inst✝ : Fact (Eq (Module.finrank Real V') 2)
          o : Orientation Real V (Fin 2)
          θ : Real.Angle
          x : V
          ⊢ Eq (LinearMap.id x) (HSMul.hSMul 1 x)
        -/
      · simp)
        /-
          🎉 no goals
        -/
    (by
      /-
        V : Type u_1
        V' : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : NormedAddCommGroup V'
        inst✝³ : InnerProductSpace Real V
        inst✝² : InnerProductSpace Real V'
        inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
        inst✝ : Fact (Eq (Module.finrank Real V') 2)
        o : Orientation Real V (Fin 2)
        θ : Real.Angle
        ⊢ Eq ((HSub.hSub (HSMul.hSMul θ.cos LinearMap.id) (HSMul.hSMul θ.sin ↑o.rightA …
      -/
      ext x
      /-
        case h
        V : Type u_1
        V' : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : NormedAddCommGroup V'
        inst✝³ : InnerProductSpace Real V
        inst✝² : InnerProductSpace Real V'
        inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
        inst✝ : Fact (Eq (Module.finrank Real V') 2)
        o : Orientation Real V (Fin 2)
        θ : Real.Angle
        x : V
        ⊢ Eq (((HSub.hSub (HSMul.hSMul θ.cos LinearMap.id) (HSMul.hSMul θ.sin ↑o.right …
      -/
      convert congr_arg (fun t : ℝ => t • x) θ.cos_sq_add_sin_sq using 1
      · simp only [o.rightAngleRotation_rightAngleRotation, o.rotationAux_apply,
          Function.comp_apply, id, LinearEquiv.coe_coe, LinearIsometry.coe_toLinearMap,
          LinearIsometryEquiv.coe_toLinearEquiv, map_add, map_smul, LinearMap.coe_comp,
          LinearMap.id_coe, LinearMap.smul_apply, LinearMap.sub_apply]
        /-
          case h.e'_2
          V : Type u_1
          V' : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : NormedAddCommGroup V'
          inst✝³ : InnerProductSpace Real V
          inst✝² : InnerProductSpace Real V'
          inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
          inst✝ : Fact (Eq (Module.finrank Real V') 2)
          o : Orientation Real V (Fin 2)
          θ : Real.Angle
          x : V
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul θ.cos (HSub.hSub (HSMul.hSMul θ.cos x) (HSMul.hSM …
        -/
        module
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3
          V : Type u_1
          V' : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : NormedAddCommGroup V'
          inst✝³ : InnerProductSpace Real V
          inst✝² : InnerProductSpace Real V'
          inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
          inst✝ : Fact (Eq (Module.finrank Real V') 2)
          o : Orientation Real V (Fin 2)
          θ : Real.Angle
          x : V
          ⊢ Eq (LinearMap.id x) (HSMul.hSMul 1 x)
        -/
      · simp)
        /-
          🎉 no goals
        -/


theorem rotation_apply (θ : Real.Angle) (x : V) :
    o.rotation θ x = Real.Angle.cos θ • x + Real.Angle.sin θ • J x :=
  rfl


theorem rotation_symm_apply (θ : Real.Angle) (x : V) :
    (o.rotation θ).symm x = Real.Angle.cos θ • x - Real.Angle.sin θ • J x :=
  rfl


theorem rotation_eq_matrix_toLin (θ : Real.Angle) {x : V} (hx : x ≠ 0) :
    (o.rotation θ).toLinearMap =
      Matrix.toLin (o.basisRightAngleRotation x hx) (o.basisRightAngleRotation x hx)
        !![θ.cos, -θ.sin; θ.sin, θ.cos] := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    x : V
    hx : Ne x 0
    ⊢ Eq (↑(o.rotation θ).toLinearEquiv) ((Matrix.toLin (o.basisRightAngleRotation …
  -/
  apply (o.basisRightAngleRotation x hx).ext
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    x : V
    hx : Ne x 0
    ⊢ ∀ (i : Fin 2), Eq (↑(o.rotation θ).toLinearEquiv ((o.basisRightAngleRotation …
  -/
  intro i
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    x : V
    hx : Ne x 0
    i : Fin 2
    ⊢ Eq (↑(o.rotation θ).toLinearEquiv ((o.basisRightAngleRotation x hx) i)) (((M …
  -/
  fin_cases i
    /-
      case «0»
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      θ : Real.Angle
      x : V
      hx : Ne x 0
      ⊢ Eq (↑(o.rotation θ).toLinearEquiv ((o.basisRightAngleRotation x hx) ((fun i  …
    -/
  · rw [Matrix.toLin_self]
    /-
      case «0»
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      θ : Real.Angle
      x : V
      hx : Ne x 0
      ⊢ Eq (↑(o.rotation θ).toLinearEquiv ((o.basisRightAngleRotation x hx) ((fun i  …
    -/
    simp [rotation_apply, Fin.sum_univ_succ]
    /-
      🎉 no goals
    -/
    /-
      case «1»
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      θ : Real.Angle
      x : V
      hx : Ne x 0
      ⊢ Eq (↑(o.rotation θ).toLinearEquiv ((o.basisRightAngleRotation x hx) ((fun i  …
    -/
  · rw [Matrix.toLin_self]
    /-
      case «1»
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      θ : Real.Angle
      x : V
      hx : Ne x 0
      ⊢ Eq (↑(o.rotation θ).toLinearEquiv ((o.basisRightAngleRotation x hx) ((fun i  …
    -/
    simp [rotation_apply, Fin.sum_univ_succ, add_comm]
    /-
      🎉 no goals
    -/


/-- The determinant of `rotation` (as a linear map) is equal to `1`. -/
@[simp]
theorem det_rotation (θ : Real.Angle) : LinearMap.det (o.rotation θ).toLinearMap = 1 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    ⊢ Eq (LinearMap.det ↑(o.rotation θ).toLinearEquiv) 1
  -/
  haveI : Nontrivial V := nontrivial_of_finrank_eq_succ (@Fact.out (finrank ℝ V = 2) _)
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    this : Nontrivial V
    ⊢ Eq (LinearMap.det ↑(o.rotation θ).toLinearEquiv) 1
  -/
  obtain ⟨x, hx⟩ : ∃ x, x ≠ (0 : V) := exists_ne (0 : V)
  /-
    case intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Eq (LinearMap.det ↑(o.rotation θ).toLinearEquiv) 1
  -/
  rw [o.rotation_eq_matrix_toLin θ hx]
  /-
    case intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Eq (LinearMap.det ((Matrix.toLin (o.basisRightAngleRotation x hx) (o.basisRi …
  -/
  simpa [sq] using θ.cos_sq_add_sin_sq
  /-
    🎉 no goals
  -/


/-- The determinant of `rotation` (as a linear equiv) is equal to `1`. -/
@[simp]
theorem linearEquiv_det_rotation (θ : Real.Angle) :
    LinearEquiv.det (o.rotation θ).toLinearEquiv = 1 :=
  Units.ext <| by
    -- Porting note: Lean can't see through `LinearEquiv.coe_det` and needed the rewrite
    -- in mathlib3 this was just `units.ext <| o.det_rotation θ`
    /-
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      θ : Real.Angle
      ⊢ Eq ↑(LinearEquiv.det (o.rotation θ).toLinearEquiv) ↑1
    -/
    simpa only [LinearEquiv.coe_det, Units.val_one] using o.det_rotation θ
    /-
      🎉 no goals
    -/


/-- The inverse of `rotation` is rotation by the negation of the angle. -/
@[simp]
theorem rotation_symm (θ : Real.Angle) : (o.rotation θ).symm = o.rotation (-θ) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    ⊢ Eq (o.rotation θ).symm (o.rotation (Neg.neg θ))
  -/
  ext; simp [o.rotation_apply, o.rotation_symm_apply, sub_eq_add_neg]
       /-
         🎉 no goals
       -/


/-- Rotation by 0 is the identity. -/
@[simp]
                                                                          /-
                                                                            V : Type u_1
                                                                            inst✝² : NormedAddCommGroup V
                                                                            inst✝¹ : InnerProductSpace Real V
                                                                            inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                                            o : Orientation Real V (Fin 2)
                                                                            ⊢ Eq (o.rotation 0) (LinearIsometryEquiv.refl Real V)
                                                                          -/
theorem rotation_zero : o.rotation 0 = LinearIsometryEquiv.refl ℝ V := by ext; simp [rotation]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- Rotation by π is negation. -/
@[simp]
theorem rotation_pi : o.rotation π = LinearIsometryEquiv.neg ℝ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    ⊢ Eq (o.rotation ↑Real.pi) (LinearIsometryEquiv.neg Real)
  -/
  ext x
  /-
    case h
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq ((o.rotation ↑Real.pi) x) ((LinearIsometryEquiv.neg Real) x)
  -/
  simp [rotation]
  /-
    🎉 no goals
  -/


/-- Rotation by π is negation. -/
                                                              /-
                                                                V : Type u_1
                                                                inst✝² : NormedAddCommGroup V
                                                                inst✝¹ : InnerProductSpace Real V
                                                                inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                                                o : Orientation Real V (Fin 2)
                                                                x : V
                                                                ⊢ Eq ((o.rotation ↑Real.pi) x) (Neg.neg x)
                                                              -/
theorem rotation_pi_apply (x : V) : o.rotation π x = -x := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Rotation by π / 2 is the "right-angle-rotation" map `J`. -/
theorem rotation_pi_div_two : o.rotation (π / 2 : ℝ) = J := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    ⊢ Eq (o.rotation ↑(HDiv.hDiv Real.pi 2)) o.rightAngleRotation
  -/
  ext x
  /-
    case h
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x) (o.rightAngleRotation x)
  -/
  simp [rotation]
  /-
    🎉 no goals
  -/


/-- Rotating twice is equivalent to rotating by the sum of the angles. -/
@[simp]
theorem rotation_rotation (θ₁ θ₂ : Real.Angle) (x : V) :
    o.rotation θ₁ (o.rotation θ₂ x) = o.rotation (θ₁ + θ₂) x := by
  simp only [o.rotation_apply, Real.Angle.cos_add, Real.Angle.sin_add, LinearIsometryEquiv.map_add,
    LinearIsometryEquiv.trans_apply, map_smul, rightAngleRotation_rightAngleRotation]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ₁ θ₂ : Real.Angle
    x : V
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul θ₁.cos (HAdd.hAdd (HSMul.hSMul θ₂.cos x) (HSMul.h …
  -/
  module
  /-
    🎉 no goals
  -/


/-- Rotating twice is equivalent to rotating by the sum of the angles. -/
@[simp]
theorem rotation_trans (θ₁ θ₂ : Real.Angle) :
    (o.rotation θ₁).trans (o.rotation θ₂) = o.rotation (θ₂ + θ₁) :=
                                      /-
                                        V : Type u_1
                                        inst✝² : NormedAddCommGroup V
                                        inst✝¹ : InnerProductSpace Real V
                                        inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                        o : Orientation Real V (Fin 2)
                                        θ₁ θ₂ : Real.Angle
                                        x✝ : V
                                        ⊢ Eq (((o.rotation θ₁).trans (o.rotation θ₂)) x✝) ((o.rotation (HAdd.hAdd θ₂ θ …
                                      -/
  LinearIsometryEquiv.ext fun _ => by rw [← rotation_rotation, LinearIsometryEquiv.trans_apply]
                                      /-
                                        🎉 no goals
                                      -/


/-- Rotating the first of two vectors by `θ` scales their Kahler form by `cos θ - sin θ * I`. -/
@[simp]
theorem kahler_rotation_left (x y : V) (θ : Real.Angle) :
    o.kahler (o.rotation θ x) y = conj (θ.toCircle : ℂ) * o.kahler x y := by
  -- Porting note: this needed the `Complex.conj_ofReal` instead of `RCLike.conj_ofReal`;
  -- I believe this is because the respective coercions are no longer defeq, and
  -- `Real.Angle.coe_toCircle` uses the `Complex` version.
  simp only [o.rotation_apply, map_add, map_mul, LinearMap.map_smulₛₗ, RingHom.id_apply,
    LinearMap.add_apply, LinearMap.smul_apply, real_smul, kahler_rightAngleRotation_left,
    Real.Angle.coe_toCircle, Complex.conj_ofReal, conj_I]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑θ.cos) ((o.kahler x) y)) (HMul.hMul (↑θ.sin) (HMu …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Negating a rotation is equivalent to rotation by π plus the angle. -/
theorem neg_rotation (θ : Real.Angle) (x : V) : -o.rotation θ x = o.rotation (π + θ) x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    x : V
    ⊢ Eq (Neg.neg ((o.rotation θ) x)) ((o.rotation (HAdd.hAdd (↑Real.pi) θ)) x)
  -/
  rw [← o.rotation_pi_apply, rotation_rotation]
  /-
    🎉 no goals
  -/


/-- Negating a rotation by -π / 2 is equivalent to rotation by π / 2. -/
@[simp]
theorem neg_rotation_neg_pi_div_two (x : V) :
    -o.rotation (-π / 2 : ℝ) x = o.rotation (π / 2 : ℝ) x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (Neg.neg ((o.rotation ↑(HDiv.hDiv (Neg.neg Real.pi) 2)) x)) ((o.rotation  …
  -/
  rw [neg_rotation, ← Real.Angle.coe_add, neg_div, ← sub_eq_add_neg, sub_half]
  /-
    🎉 no goals
  -/


/-- Negating a rotation by π / 2 is equivalent to rotation by -π / 2. -/
theorem neg_rotation_pi_div_two (x : V) : -o.rotation (π / 2 : ℝ) x = o.rotation (-π / 2 : ℝ) x :=
  (neg_eq_iff_eq_neg.mp <| o.neg_rotation_neg_pi_div_two _).symm


/-- Rotating the first of two vectors by `θ` scales their Kahler form by `cos (-θ) + sin (-θ) * I`.
-/
theorem kahler_rotation_left' (x y : V) (θ : Real.Angle) :
    o.kahler (o.rotation θ x) y = (-θ).toCircle * o.kahler x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    θ : Real.Angle
    ⊢ Eq ((o.kahler ((o.rotation θ) x)) y) (HMul.hMul (↑(Neg.neg θ).toCircle) ((o. …
  -/
  simp only [Real.Angle.toCircle_neg, Circle.coe_inv_eq_conj, kahler_rotation_left]
  /-
    🎉 no goals
  -/


/-- Rotating the second of two vectors by `θ` scales their Kahler form by `cos θ + sin θ * I`. -/
@[simp]
theorem kahler_rotation_right (x y : V) (θ : Real.Angle) :
    o.kahler x (o.rotation θ y) = θ.toCircle * o.kahler x y := by
  simp only [o.rotation_apply, map_add, LinearMap.map_smulₛₗ, RingHom.id_apply, real_smul,
    kahler_rightAngleRotation_right, Real.Angle.coe_toCircle]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    θ : Real.Angle
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑θ.cos) ((o.kahler x) y)) (HMul.hMul (↑θ.sin) (HMu …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Rotating the first vector by `θ` subtracts `θ` from the angle between two vectors. -/
@[simp]
theorem oangle_rotation_left {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) (θ : Real.Angle) :
    o.oangle (o.rotation θ x) y = o.oangle x y - θ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    ⊢ Eq (o.oangle ((o.rotation θ) x) y) (HSub.hSub (o.oangle x y) θ)
  -/
  simp only [oangle, o.kahler_rotation_left']
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    ⊢ Eq (↑(HMul.hMul (↑(Neg.neg θ).toCircle) ((o.kahler x) y)).arg) (HSub.hSub (↑ …
  -/
  rw [Complex.arg_mul_coe_angle, Real.Angle.arg_toCircle]
    /-
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Eq (HAdd.hAdd (Neg.neg θ) ↑((o.kahler x) y).arg) (HSub.hSub (↑((o.kahler x)  …
    -/
    /-
      🎉 no goals
    -/
  · abel
    /-
      🎉 no goals
    -/
    /-
      case hx
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Ne (↑(Neg.neg θ).toCircle) 0
    -/
  · exact Circle.coe_ne_zero _
    /-
      🎉 no goals
    -/
    /-
      case hy
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Ne ((o.kahler x) y) 0
    -/
  · exact o.kahler_ne_zero hx hy
    /-
      🎉 no goals
    -/


/-- Rotating the second vector by `θ` adds `θ` to the angle between two vectors. -/
@[simp]
theorem oangle_rotation_right {x y : V} (hx : x ≠ 0) (hy : y ≠ 0) (θ : Real.Angle) :
    o.oangle x (o.rotation θ y) = o.oangle x y + θ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    ⊢ Eq (o.oangle x ((o.rotation θ) y)) (HAdd.hAdd (o.oangle x y) θ)
  -/
  simp only [oangle, o.kahler_rotation_right]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    ⊢ Eq (↑(HMul.hMul (↑θ.toCircle) ((o.kahler x) y)).arg) (HAdd.hAdd (↑((o.kahler …
  -/
  rw [Complex.arg_mul_coe_angle, Real.Angle.arg_toCircle]
    /-
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Eq (HAdd.hAdd θ ↑((o.kahler x) y).arg) (HAdd.hAdd (↑((o.kahler x) y).arg) θ)
    -/
    /-
      🎉 no goals
    -/
  · abel
    /-
      🎉 no goals
    -/
    /-
      case hx
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Ne (↑θ.toCircle) 0
    -/
  · exact Circle.coe_ne_zero _
    /-
      🎉 no goals
    -/
    /-
      case hy
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Ne ((o.kahler x) y) 0
    -/
  · exact o.kahler_ne_zero hx hy
    /-
      🎉 no goals
    -/


/-- The rotation of a vector by `θ` has an angle of `-θ` from that vector. -/
@[simp]
theorem oangle_rotation_self_left {x : V} (hx : x ≠ 0) (θ : Real.Angle) :
                                           /-
                                             V : Type u_1
                                             inst✝² : NormedAddCommGroup V
                                             inst✝¹ : InnerProductSpace Real V
                                             inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                             o : Orientation Real V (Fin 2)
                                             x : V
                                             hx : Ne x 0
                                             θ : Real.Angle
                                             ⊢ Eq (o.oangle ((o.rotation θ) x) x) (Neg.neg θ)
                                           -/
    o.oangle (o.rotation θ x) x = -θ := by simp [hx]
                                           /-
                                             🎉 no goals
                                           -/


/-- A vector has an angle of `θ` from the rotation of that vector by `θ`. -/
@[simp]
theorem oangle_rotation_self_right {x : V} (hx : x ≠ 0) (θ : Real.Angle) :
                                          /-
                                            V : Type u_1
                                            inst✝² : NormedAddCommGroup V
                                            inst✝¹ : InnerProductSpace Real V
                                            inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                            o : Orientation Real V (Fin 2)
                                            x : V
                                            hx : Ne x 0
                                            θ : Real.Angle
                                            ⊢ Eq (o.oangle x ((o.rotation θ) x)) θ
                                          -/
    o.oangle x (o.rotation θ x) = θ := by simp [hx]
                                          /-
                                            🎉 no goals
                                          -/


/-- Rotating the first vector by the angle between the two vectors results in an angle of 0. -/
@[simp]
theorem oangle_rotation_oangle_left (x y : V) : o.oangle (o.rotation (o.oangle x y) x) y = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle ((o.rotation (o.oangle x y)) x) y) 0
  -/
  by_cases hx : x = 0
    /-
      case pos
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Eq x 0
      ⊢ Eq (o.oangle ((o.rotation (o.oangle x y)) x) y) 0
    -/
  · simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Not (Eq x 0)
      ⊢ Eq (o.oangle ((o.rotation (o.oangle x y)) x) y) 0
    -/
  · by_cases hy : y = 0
      /-
        case pos
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Eq (o.oangle ((o.rotation (o.oangle x y)) x) y) 0
      -/
    · simp [hy]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Eq (o.oangle ((o.rotation (o.oangle x y)) x) y) 0
      -/
    · simp [hx, hy]
      /-
        🎉 no goals
      -/


/-- Rotating the first vector by the angle between the two vectors and swapping the vectors
results in an angle of 0. -/
@[simp]
theorem oangle_rotation_oangle_right (x y : V) : o.oangle y (o.rotation (o.oangle x y) x) = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (o.oangle y ((o.rotation (o.oangle x y)) x)) 0
  -/
  rw [oangle_rev]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Eq (Neg.neg (o.oangle ((o.rotation (o.oangle x y)) x) y)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Rotating both vectors by the same angle does not change the angle between those vectors. -/
@[simp]
theorem oangle_rotation (x y : V) (θ : Real.Angle) :
    o.oangle (o.rotation θ x) (o.rotation θ y) = o.oangle x y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    θ : Real.Angle
    ⊢ Eq (o.oangle ((o.rotation θ) x) ((o.rotation θ) y)) (o.oangle x y)
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
  by_cases hx : x = 0 <;> by_cases hy : y = 0 <;> simp [hx, hy]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- A rotation of a nonzero vector equals that vector if and only if the angle is zero. -/
@[simp]
theorem rotation_eq_self_iff_angle_eq_zero {x : V} (hx : x ≠ 0) (θ : Real.Angle) :
    o.rotation θ x = x ↔ θ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    hx : Ne x 0
    θ : Real.Angle
    ⊢ Iff (Eq ((o.rotation θ) x) x) (Eq θ 0)
  -/
  constructor
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      hx : Ne x 0
      θ : Real.Angle
      ⊢ Eq ((o.rotation θ) x) x → Eq θ 0
    -/
  · intro h
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      hx : Ne x 0
      θ : Real.Angle
      h : Eq ((o.rotation θ) x) x
      ⊢ Eq θ 0
    -/
    rw [eq_comm]
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      hx : Ne x 0
      θ : Real.Angle
      h : Eq ((o.rotation θ) x) x
      ⊢ Eq 0 θ
    -/
    simpa [hx, h] using o.oangle_rotation_right hx hx θ
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      hx : Ne x 0
      θ : Real.Angle
      ⊢ Eq θ 0 → Eq ((o.rotation θ) x) x
    -/
  · intro h
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      hx : Ne x 0
      θ : Real.Angle
      h : Eq θ 0
      ⊢ Eq ((o.rotation θ) x) x
    -/
    simp [h]
    /-
      🎉 no goals
    -/


/-- A nonzero vector equals a rotation of that vector if and only if the angle is zero. -/
@[simp]
theorem eq_rotation_self_iff_angle_eq_zero {x : V} (hx : x ≠ 0) (θ : Real.Angle) :
                                     /-
                                       V : Type u_1
                                       inst✝² : NormedAddCommGroup V
                                       inst✝¹ : InnerProductSpace Real V
                                       inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                       o : Orientation Real V (Fin 2)
                                       x : V
                                       hx : Ne x 0
                                       θ : Real.Angle
                                       ⊢ Iff (Eq x ((o.rotation θ) x)) (Eq θ 0)
                                     -/
    x = o.rotation θ x ↔ θ = 0 := by rw [← o.rotation_eq_self_iff_angle_eq_zero hx, eq_comm]
                                     /-
                                       🎉 no goals
                                     -/


/-- A rotation of a vector equals that vector if and only if the vector or the angle is zero. -/
theorem rotation_eq_self_iff (x : V) (θ : Real.Angle) : o.rotation θ x = x ↔ x = 0 ∨ θ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    θ : Real.Angle
    ⊢ Iff (Eq ((o.rotation θ) x) x) (Or (Eq x 0) (Eq θ 0))
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : x = 0 <;> simp [h]
                         /-
                           🎉 no goals
                         -/


/-- A vector equals a rotation of that vector if and only if the vector or the angle is zero. -/
theorem eq_rotation_self_iff (x : V) (θ : Real.Angle) : x = o.rotation θ x ↔ x = 0 ∨ θ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    θ : Real.Angle
    ⊢ Iff (Eq x ((o.rotation θ) x)) (Or (Eq x 0) (Eq θ 0))
  -/
  rw [← rotation_eq_self_iff, eq_comm]
  /-
    🎉 no goals
  -/


/-- Rotating a vector by the angle to another vector gives the second vector if and only if the
norms are equal. -/
@[simp]
theorem rotation_oangle_eq_iff_norm_eq (x y : V) : o.rotation (o.oangle x y) x = y ↔ ‖x‖ = ‖y‖ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq ((o.rotation (o.oangle x y)) x) y) (Eq (Norm.norm x) (Norm.norm y))
  -/
  constructor
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      ⊢ Eq ((o.rotation (o.oangle x y)) x) y → Eq (Norm.norm x) (Norm.norm y)
    -/
  · intro h
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq ((o.rotation (o.oangle x y)) x) y
      ⊢ Eq (Norm.norm x) (Norm.norm y)
    -/
    rw [← h, LinearIsometryEquiv.norm_map]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      ⊢ Eq (Norm.norm x) (Norm.norm y) → Eq ((o.rotation (o.oangle x y)) x) y
    -/
  · intro h
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Eq (Norm.norm x) (Norm.norm y)
      ⊢ Eq ((o.rotation (o.oangle x y)) x) y
    -/
                                                /-
                                                  🎉 no goals
                                                -/
    rw [o.eq_iff_oangle_eq_zero_of_norm_eq] <;> simp [h]
                                                /-
                                                  🎉 no goals
                                                -/


/-- The angle between two nonzero vectors is `θ` if and only if the second vector is the first
rotated by `θ` and scaled by the ratio of the norms. -/
theorem oangle_eq_iff_eq_norm_div_norm_smul_rotation_of_ne_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0)
    (θ : Real.Angle) : o.oangle x y = θ ↔ y = (‖y‖ / ‖x‖) • o.rotation θ x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    ⊢ Iff (Eq (o.oangle x y) θ) (Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm. …
  -/
  have hp := div_pos (norm_pos_iff.2 hy) (norm_pos_iff.2 hx)
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    hp : LT.lt 0 (HDiv.hDiv (Norm.norm y) (Norm.norm x))
    ⊢ Iff (Eq (o.oangle x y) θ) (Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm. …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      hp : LT.lt 0 (HDiv.hDiv (Norm.norm y) (Norm.norm x))
      ⊢ Eq (o.oangle x y) θ → Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm.norm  …
    -/
  · rintro rfl
    rw [← LinearIsometryEquiv.map_smul, ← o.oangle_smul_left_of_pos x y hp, eq_comm,
      rotation_oangle_eq_iff_norm_eq, norm_smul, Real.norm_of_nonneg hp.le,
      div_mul_cancel₀ _ (norm_ne_zero_iff.2 hx)]
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      hp : LT.lt 0 (HDiv.hDiv (Norm.norm y) (Norm.norm x))
      ⊢ Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm.norm x)) ((o.rotation θ) x) …
    -/
  · intro hye
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      hp : LT.lt 0 (HDiv.hDiv (Norm.norm y) (Norm.norm x))
      hye : Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm.norm x)) ((o.rotation θ …
      ⊢ Eq (o.oangle x y) θ
    -/
    rw [hye, o.oangle_smul_right_of_pos _ _ hp, o.oangle_rotation_self_right hx]
    /-
      🎉 no goals
    -/


/-- The angle between two nonzero vectors is `θ` if and only if the second vector is the first
rotated by `θ` and scaled by a positive real. -/
theorem oangle_eq_iff_eq_pos_smul_rotation_of_ne_zero {x y : V} (hx : x ≠ 0) (hy : y ≠ 0)
    (θ : Real.Angle) : o.oangle x y = θ ↔ ∃ r : ℝ, 0 < r ∧ y = r • o.rotation θ x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    hx : Ne x 0
    hy : Ne y 0
    θ : Real.Angle
    ⊢ Iff (Eq (o.oangle x y) θ) (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMu …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ Eq (o.oangle x y) θ → Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r ( …
    -/
  · intro h
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      h : Eq (o.oangle x y) θ
      ⊢ Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r ((o.rotation θ) x)))
    -/
    rw [o.oangle_eq_iff_eq_norm_div_norm_smul_rotation_of_ne_zero hx hy] at h
    /-
      case mp
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      h : Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm.norm x)) ((o.rotation θ)  …
      ⊢ Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r ((o.rotation θ) x)))
    -/
    exact ⟨‖y‖ / ‖x‖, div_pos (norm_pos_iff.2 hy) (norm_pos_iff.2 hx), h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      hx : Ne x 0
      hy : Ne y 0
      θ : Real.Angle
      ⊢ (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r ((o.rotation θ) x))))  …
    -/
  · rintro ⟨r, hr, rfl⟩
    /-
      case mpr.intro.intro
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x : V
      hx : Ne x 0
      θ : Real.Angle
      r : Real
      hr : LT.lt 0 r
      hy : Ne (HSMul.hSMul r ((o.rotation θ) x)) 0
      ⊢ Eq (o.oangle x (HSMul.hSMul r ((o.rotation θ) x))) θ
    -/
    rw [o.oangle_smul_right_of_pos _ _ hr, o.oangle_rotation_self_right hx]
    /-
      🎉 no goals
    -/


/-- The angle between two vectors is `θ` if and only if they are nonzero and the second vector
is the first rotated by `θ` and scaled by the ratio of the norms, or `θ` and at least one of the
vectors are zero. -/
theorem oangle_eq_iff_eq_norm_div_norm_smul_rotation_or_eq_zero {x y : V} (θ : Real.Angle) :
    o.oangle x y = θ ↔
      x ≠ 0 ∧ y ≠ 0 ∧ y = (‖y‖ / ‖x‖) • o.rotation θ x ∨ θ = 0 ∧ (x = 0 ∨ y = 0) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    θ : Real.Angle
    ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Eq y (HSMul.hSMul …
  -/
  by_cases hx : x = 0
    /-
      case pos
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      θ : Real.Angle
      hx : Eq x 0
      ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Eq y (HSMul.hSMul …
    -/
  · simp [hx, eq_comm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      θ : Real.Angle
      hx : Not (Eq x 0)
      ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Eq y (HSMul.hSMul …
    -/
  · by_cases hy : y = 0
      /-
        case pos
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        θ : Real.Angle
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Eq y (HSMul.hSMul …
      -/
    · simp [hy, eq_comm]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        θ : Real.Angle
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Eq y (HSMul.hSMul …
      -/
    · rw [o.oangle_eq_iff_eq_norm_div_norm_smul_rotation_of_ne_zero hx hy]
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        θ : Real.Angle
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Iff (Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm.norm x)) ((o.rotation  …
      -/
      simp [hx, hy]
      /-
        🎉 no goals
      -/


/-- The angle between two vectors is `θ` if and only if they are nonzero and the second vector
is the first rotated by `θ` and scaled by a positive real, or `θ` and at least one of the
vectors are zero. -/
theorem oangle_eq_iff_eq_pos_smul_rotation_or_eq_zero {x y : V} (θ : Real.Angle) :
    o.oangle x y = θ ↔
      (x ≠ 0 ∧ y ≠ 0 ∧ ∃ r : ℝ, 0 < r ∧ y = r • o.rotation θ x) ∨ θ = 0 ∧ (x = 0 ∨ y = 0) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    θ : Real.Angle
    ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Exists fun r => A …
  -/
  by_cases hx : x = 0
    /-
      case pos
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      θ : Real.Angle
      hx : Eq x 0
      ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Exists fun r => A …
    -/
  · simp [hx, eq_comm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      θ : Real.Angle
      hx : Not (Eq x 0)
      ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Exists fun r => A …
    -/
  · by_cases hy : y = 0
      /-
        case pos
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        θ : Real.Angle
        hx : Not (Eq x 0)
        hy : Eq y 0
        ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Exists fun r => A …
      -/
    · simp [hy, eq_comm]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        θ : Real.Angle
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Iff (Eq (o.oangle x y) θ) (Or (And (Ne x 0) (And (Ne y 0) (Exists fun r => A …
      -/
    · rw [o.oangle_eq_iff_eq_pos_smul_rotation_of_ne_zero hx hy]
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x y : V
        θ : Real.Angle
        hx : Not (Eq x 0)
        hy : Not (Eq y 0)
        ⊢ Iff (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r ((o.rotation θ) x) …
      -/
      simp [hx, hy]
      /-
        🎉 no goals
      -/


/-- Any linear isometric equivalence in `V` with positive determinant is `rotation`. -/
theorem exists_linearIsometryEquiv_eq_of_det_pos {f : V ≃ₗᵢ[ℝ] V}
    (hd : 0 < LinearMap.det (f.toLinearEquiv : V →ₗ[ℝ] V)) :
    ∃ θ : Real.Angle, f = o.rotation θ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    ⊢ Exists fun θ => Eq f (o.rotation θ)
  -/
  haveI : Nontrivial V := nontrivial_of_finrank_eq_succ (@Fact.out (finrank ℝ V = 2) _)
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    ⊢ Exists fun θ => Eq f (o.rotation θ)
  -/
  obtain ⟨x, hx⟩ : ∃ x, x ≠ (0 : V) := exists_ne (0 : V)
  /-
    case intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Exists fun θ => Eq f (o.rotation θ)
  -/
  use o.oangle x (f x)
  /-
    case h
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Eq f (o.rotation (o.oangle x (f x)))
  -/
  apply LinearIsometryEquiv.toLinearEquiv_injective
  /-
    case h.a
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Eq f.toLinearEquiv (o.rotation (o.oangle x (f x))).toLinearEquiv
  -/
  apply LinearEquiv.toLinearMap_injective
  /-
    case h.a.a
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Eq ↑f.toLinearEquiv ↑(o.rotation (o.oangle x (f x))).toLinearEquiv
  -/
  apply (o.basisRightAngleRotation x hx).ext
  /-
    case h.a.a
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ ∀ (i : Fin 2), Eq (↑f.toLinearEquiv ((o.basisRightAngleRotation x hx) i)) (↑ …
  -/
  intro i
  /-
    case h.a.a
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    i : Fin 2
    ⊢ Eq (↑f.toLinearEquiv ((o.basisRightAngleRotation x hx) i)) (↑(o.rotation (o. …
  -/
  symm
  /-
    case h.a.a
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this : Nontrivial V
    x : V
    hx : Ne x 0
    i : Fin 2
    ⊢ Eq (↑(o.rotation (o.oangle x (f x))).toLinearEquiv ((o.basisRightAngleRotati …
  -/
  fin_cases i
    /-
      case h.a.a.«0»
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      f : LinearIsometryEquiv (RingHom.id Real) V V
      hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
      this : Nontrivial V
      x : V
      hx : Ne x 0
      ⊢ Eq (↑(o.rotation (o.oangle x (f x))).toLinearEquiv ((o.basisRightAngleRotati …
    -/
  · simp
    /-
      🎉 no goals
    -/
  have : o.oangle (J x) (f (J x)) = o.oangle x (f x) := by
    simp only [oangle, o.linearIsometryEquiv_comp_rightAngleRotation f hd,
      o.kahler_comp_rightAngleRotation]
  /-
    case h.a.a.«1»
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    f : LinearIsometryEquiv (RingHom.id Real) V V
    hd : LT.lt 0 (LinearMap.det ↑f.toLinearEquiv)
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Eq (o.oangle (o.rightAngleRotation x) (f (o.rightAngleRotation x))) (o. …
    ⊢ Eq (↑(o.rotation (o.oangle x (f x))).toLinearEquiv ((o.basisRightAngleRotati …
  -/
  simp [← this]
  /-
    🎉 no goals
  -/


theorem rotation_map (θ : Real.Angle) (f : V ≃ₗᵢ[ℝ] V') (x : V') :
    (Orientation.map (Fin 2) f.toLinearEquiv o).rotation θ x = f (o.rotation θ (f.symm x)) := by
  /-
    V : Type u_1
    V' : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : NormedAddCommGroup V'
    inst✝³ : InnerProductSpace Real V
    inst✝² : InnerProductSpace Real V'
    inst✝¹ : Fact (Eq (Module.finrank Real V) 2)
    inst✝ : Fact (Eq (Module.finrank Real V') 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    f : LinearIsometryEquiv (RingHom.id Real) V V'
    x : V'
    ⊢ Eq ((((Orientation.map (Fin 2) f.toLinearEquiv) o).rotation θ) x) (f ((o.rot …
  -/
  simp [rotation_apply, o.rightAngleRotation_map]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem _root_.Complex.rotation (θ : Real.Angle) (z : ℂ) :
    Complex.orientation.rotation θ z = θ.toCircle * z := by
  /-
    θ : Real.Angle
    z : Complex
    ⊢ Eq ((Complex.orientation.rotation θ) z) (HMul.hMul (↑θ.toCircle) z)
  -/
  simp only [rotation_apply, Complex.rightAngleRotation, Real.Angle.coe_toCircle, real_smul]
  /-
    θ : Real.Angle
    z : Complex
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑θ.cos) z) (HMul.hMul (↑θ.sin) (HMul.hMul Complex. …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Rotation in an oriented real inner product space of dimension 2 can be evaluated in terms of a
complex-number representation of the space. -/
theorem rotation_map_complex (θ : Real.Angle) (f : V ≃ₗᵢ[ℝ] ℂ)
    (hf : Orientation.map (Fin 2) f.toLinearEquiv o = Complex.orientation) (x : V) :
    f (o.rotation θ x) = θ.toCircle * f x := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    θ : Real.Angle
    f : LinearIsometryEquiv (RingHom.id Real) V Complex
    hf : Eq ((Orientation.map (Fin 2) f.toLinearEquiv) o) Complex.orientation
    x : V
    ⊢ Eq (f ((o.rotation θ) x)) (HMul.hMul (↑θ.toCircle) (f x))
  -/
  rw [← Complex.rotation, ← hf, o.rotation_map, LinearIsometryEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- Negating the orientation negates the angle in `rotation`. -/
theorem rotation_neg_orientation_eq_neg (θ : Real.Angle) : (-o).rotation θ = o.rotation (-θ) :=
                                /-
                                  V : Type u_1
                                  inst✝² : NormedAddCommGroup V
                                  inst✝¹ : InnerProductSpace Real V
                                  inst✝ : Fact (Eq (Module.finrank Real V) 2)
                                  o : Orientation Real V (Fin 2)
                                  θ : Real.Angle
                                  ⊢ ∀ (x : V), Eq (((Neg.neg o).rotation θ) x) ((o.rotation (Neg.neg θ)) x)
                                -/
  LinearIsometryEquiv.ext <| by simp [rotation_apply]
                                /-
                                  🎉 no goals
                                -/


/-- The inner product between a `π / 2` rotation of a vector and that vector is zero. -/
@[simp]
theorem inner_rotation_pi_div_two_left (x : V) : ⟪o.rotation (π / 2 : ℝ) x, x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (Inner.inner ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x) x) 0
  -/
  rw [rotation_pi_div_two, inner_rightAngleRotation_self]
  /-
    🎉 no goals
  -/


/-- The inner product between a vector and a `π / 2` rotation of that vector is zero. -/
@[simp]
theorem inner_rotation_pi_div_two_right (x : V) : ⟪x, o.rotation (π / 2 : ℝ) x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    ⊢ Eq (Inner.inner x ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) 0
  -/
  rw [real_inner_comm, inner_rotation_pi_div_two_left]
  /-
    🎉 no goals
  -/


/-- The inner product between a multiple of a `π / 2` rotation of a vector and that vector is
zero. -/
@[simp]
theorem inner_smul_rotation_pi_div_two_left (x : V) (r : ℝ) :
    ⟪r • o.rotation (π / 2 : ℝ) x, x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) x) 0
  -/
  rw [inner_smul_left, inner_rotation_pi_div_two_left, mul_zero]
  /-
    🎉 no goals
  -/


/-- The inner product between a vector and a multiple of a `π / 2` rotation of that vector is
zero. -/
@[simp]
theorem inner_smul_rotation_pi_div_two_right (x : V) (r : ℝ) :
    ⟪x, r • o.rotation (π / 2 : ℝ) x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    ⊢ Eq (Inner.inner x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))) 0
  -/
  rw [real_inner_comm, inner_smul_rotation_pi_div_two_left]
  /-
    🎉 no goals
  -/


/-- The inner product between a `π / 2` rotation of a vector and a multiple of that vector is
zero. -/
@[simp]
theorem inner_rotation_pi_div_two_left_smul (x : V) (r : ℝ) :
    ⟪o.rotation (π / 2 : ℝ) x, r • x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    ⊢ Eq (Inner.inner ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x) (HSMul.hSMul r x)) 0
  -/
  rw [inner_smul_right, inner_rotation_pi_div_two_left, mul_zero]
  /-
    🎉 no goals
  -/


/-- The inner product between a multiple of a vector and a `π / 2` rotation of that vector is
zero. -/
@[simp]
theorem inner_rotation_pi_div_two_right_smul (x : V) (r : ℝ) :
    ⟪r • x, o.rotation (π / 2 : ℝ) x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul r x) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) 0
  -/
  rw [real_inner_comm, inner_rotation_pi_div_two_left_smul]
  /-
    🎉 no goals
  -/


/-- The inner product between a multiple of a `π / 2` rotation of a vector and a multiple of
that vector is zero. -/
@[simp]
theorem inner_smul_rotation_pi_div_two_smul_left (x : V) (r₁ r₂ : ℝ) :
    ⟪r₁ • o.rotation (π / 2 : ℝ) x, r₂ • x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r₁ r₂ : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul r₁ ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) (HS …
  -/
  rw [inner_smul_right, inner_smul_rotation_pi_div_two_left, mul_zero]
  /-
    🎉 no goals
  -/


/-- The inner product between a multiple of a vector and a multiple of a `π / 2` rotation of
that vector is zero. -/
@[simp]
theorem inner_smul_rotation_pi_div_two_smul_right (x : V) (r₁ r₂ : ℝ) :
    ⟪r₂ • x, r₁ • o.rotation (π / 2 : ℝ) x⟫ = 0 := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x : V
    r₁ r₂ : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul r₂ x) (HSMul.hSMul r₁ ((o.rotation ↑(HDiv.hDiv  …
  -/
  rw [real_inner_comm, inner_smul_rotation_pi_div_two_smul_left]
  /-
    🎉 no goals
  -/


/-- The inner product between two vectors is zero if and only if the first vector is zero or
the second is a multiple of a `π / 2` rotation of that vector. -/
theorem inner_eq_zero_iff_eq_zero_or_eq_smul_rotation_pi_div_two {x y : V} :
    ⟪x, y⟫ = 0 ↔ x = 0 ∨ ∃ r : ℝ, r • o.rotation (π / 2 : ℝ) x = y := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Eq (Inner.inner x y) 0) (Or (Eq x 0) (Exists fun r => Eq (HSMul.hSMul r …
  -/
  rw [← o.eq_zero_or_oangle_eq_iff_inner_eq_zero]
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : Fact (Eq (Module.finrank Real V) 2)
    o : Orientation Real V (Fin 2)
    x y : V
    ⊢ Iff (Or (Eq x 0) (Or (Eq y 0) (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Or (Eq x 0) (Or (Eq y 0) (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) (E …
      ⊢ Or (Eq x 0) (Exists fun r => Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real …
    -/
  · rcases h with (rfl | rfl | h | h)
      /-
        case refine_1.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        y : V
        ⊢ Or (Eq 0 0) (Exists fun r => Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real …
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        ⊢ Or (Eq x 0) (Exists fun r => Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real …
      -/
    · exact Or.inr ⟨0, zero_smul _ _⟩
      /-
        🎉 no goals
      -/
    · obtain ⟨r, _, rfl⟩ :=
        (o.oangle_eq_iff_eq_pos_smul_rotation_of_ne_zero (o.left_ne_zero_of_oangle_eq_pi_div_two h)
          (o.right_ne_zero_of_oangle_eq_pi_div_two h) _).1 h
      /-
        case refine_1.inr.inr.inl.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        left✝ : LT.lt 0 r
        h : Eq (o.oangle x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))) ↑( …
        ⊢ Or (Eq x 0) (Exists fun r_1 => Eq (HSMul.hSMul r_1 ((o.rotation ↑(HDiv.hDiv  …
      -/
      exact Or.inr ⟨r, rfl⟩
      /-
        🎉 no goals
      -/
    · obtain ⟨r, _, rfl⟩ :=
        (o.oangle_eq_iff_eq_pos_smul_rotation_of_ne_zero
          (o.left_ne_zero_of_oangle_eq_neg_pi_div_two h)
          (o.right_ne_zero_of_oangle_eq_neg_pi_div_two h) _).1 h
      /-
        case refine_1.inr.inr.inr.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        left✝ : LT.lt 0 r
        h : Eq (o.oangle x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv (Neg.neg Real.pi) 2 …
        ⊢ Or (Eq x 0) (Exists fun r_1 => Eq (HSMul.hSMul r_1 ((o.rotation ↑(HDiv.hDiv  …
      -/
      refine Or.inr ⟨-r, ?_⟩
      /-
        case refine_1.inr.inr.inr.intro.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        left✝ : LT.lt 0 r
        h : Eq (o.oangle x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv (Neg.neg Real.pi) 2 …
        ⊢ Eq (HSMul.hSMul (Neg.neg r) ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x)) (HSMul. …
      -/
      rw [neg_smul, ← smul_neg, o.neg_rotation_pi_div_two]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      V : Type u_1
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : Fact (Eq (Module.finrank Real V) 2)
      o : Orientation Real V (Fin 2)
      x y : V
      h : Or (Eq x 0) (Exists fun r => Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Re …
      ⊢ Or (Eq x 0) (Or (Eq y 0) (Or (Eq (o.oangle x y) ↑(HDiv.hDiv Real.pi 2)) (Eq  …
    -/
  · rcases h with (rfl | ⟨r, rfl⟩)
      /-
        case refine_2.inl
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        y : V
        ⊢ Or (Eq 0 0) (Or (Eq y 0) (Or (Eq (o.oangle 0 y) ↑(HDiv.hDiv Real.pi 2)) (Eq  …
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        ⊢ Or (Eq x 0) (Or (Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))  …
      -/
    · by_cases hx : x = 0; · exact Or.inl hx
                             /-
                               🎉 no goals
                             -/
      /-
        case neg
        V : Type u_1
        inst✝² : NormedAddCommGroup V
        inst✝¹ : InnerProductSpace Real V
        inst✝ : Fact (Eq (Module.finrank Real V) 2)
        o : Orientation Real V (Fin 2)
        x : V
        r : Real
        hx : Not (Eq x 0)
        ⊢ Or (Eq x 0) (Or (Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))  …
      -/
      rcases lt_trichotomy r 0 with (hr | rfl | hr)
        /-
          case neg.inl
          V : Type u_1
          inst✝² : NormedAddCommGroup V
          inst✝¹ : InnerProductSpace Real V
          inst✝ : Fact (Eq (Module.finrank Real V) 2)
          o : Orientation Real V (Fin 2)
          x : V
          r : Real
          hx : Not (Eq x 0)
          hr : LT.lt r 0
          ⊢ Or (Eq x 0) (Or (Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))  …
        -/
      · refine Or.inr (Or.inr (Or.inr ?_))
        rw [o.oangle_smul_right_of_neg _ _ hr, o.neg_rotation_pi_div_two,
          o.oangle_rotation_self_right hx]
        /-
          case neg.inr.inl
          V : Type u_1
          inst✝² : NormedAddCommGroup V
          inst✝¹ : InnerProductSpace Real V
          inst✝ : Fact (Eq (Module.finrank Real V) 2)
          o : Orientation Real V (Fin 2)
          x : V
          hx : Not (Eq x 0)
          ⊢ Or (Eq x 0) (Or (Eq (HSMul.hSMul 0 ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))  …
        -/
      · exact Or.inr (Or.inl (zero_smul _ _))
        /-
          🎉 no goals
        -/
        /-
          case neg.inr.inr
          V : Type u_1
          inst✝² : NormedAddCommGroup V
          inst✝¹ : InnerProductSpace Real V
          inst✝ : Fact (Eq (Module.finrank Real V) 2)
          o : Orientation Real V (Fin 2)
          x : V
          r : Real
          hx : Not (Eq x 0)
          hr : LT.lt 0 r
          ⊢ Or (Eq x 0) (Or (Eq (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))  …
        -/
      · refine Or.inr (Or.inr (Or.inl ?_))
        /-
          case neg.inr.inr
          V : Type u_1
          inst✝² : NormedAddCommGroup V
          inst✝¹ : InnerProductSpace Real V
          inst✝ : Fact (Eq (Module.finrank Real V) 2)
          o : Orientation Real V (Fin 2)
          x : V
          r : Real
          hx : Not (Eq x 0)
          hr : LT.lt 0 r
          ⊢ Eq (o.oangle x (HSMul.hSMul r ((o.rotation ↑(HDiv.hDiv Real.pi 2)) x))) ↑(HD …
        -/
        rw [o.oangle_smul_right_of_pos _ _ hr, o.oangle_rotation_self_right hx]
        /-
          🎉 no goals
        -/


