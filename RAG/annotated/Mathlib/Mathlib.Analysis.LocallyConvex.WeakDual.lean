/-- Construct a seminorm from a linear form `f : E →ₗ[𝕜] 𝕜` over a normed field `𝕜` by
`fun x => ‖f x‖` -/
def toSeminorm (f : E →ₗ[𝕜] 𝕜) : Seminorm 𝕜 E :=
  (normSeminorm 𝕜 𝕜).comp f


theorem coe_toSeminorm {f : E →ₗ[𝕜] 𝕜} : ⇑f.toSeminorm = fun x => ‖f x‖ :=
  rfl


@[simp]
theorem toSeminorm_apply {f : E →ₗ[𝕜] 𝕜} {x : E} : f.toSeminorm x = ‖f x‖ :=
  rfl


theorem toSeminorm_ball_zero {f : E →ₗ[𝕜] 𝕜} {r : ℝ} :
    Seminorm.ball f.toSeminorm 0 r = { x : E | ‖f x‖ < r } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E 𝕜
    r : Real
    ⊢ Eq (f.toSeminorm.ball 0 r) (setOf fun x => LT.lt (Norm.norm (f x)) r)
  -/
  simp only [Seminorm.ball_zero_eq, toSeminorm_apply]
  /-
    🎉 no goals
  -/


theorem toSeminorm_comp (f : F →ₗ[𝕜] 𝕜) (g : E →ₗ[𝕜] F) :
    f.toSeminorm.comp g = (f.comp g).toSeminorm := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    f : LinearMap (RingHom.id 𝕜) F 𝕜
    g : LinearMap (RingHom.id 𝕜) E F
    ⊢ Eq (f.toSeminorm.comp g) (f.comp g).toSeminorm
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    f : LinearMap (RingHom.id 𝕜) F 𝕜
    g : LinearMap (RingHom.id 𝕜) E F
    x✝ : E
    ⊢ Eq ((f.toSeminorm.comp g) x✝) ((f.comp g).toSeminorm x✝)
  -/
  simp only [Seminorm.comp_apply, toSeminorm_apply, coe_comp, Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- Construct a family of seminorms from a bilinear form. -/
def toSeminormFamily (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) : SeminormFamily 𝕜 E F := fun y =>
  (B.flip y).toSeminorm


@[simp]
theorem toSeminormFamily_apply {B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜} {x y} : (B.toSeminormFamily y) x = ‖B x y‖ :=
  rfl


theorem LinearMap.weakBilin_withSeminorms (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) :
    WithSeminorms (LinearMap.toSeminormFamily B : F → Seminorm 𝕜 (WeakBilin B)) :=
  let e : F ≃ (Σ _ : F, Fin 1) := .symm <| .sigmaUnique _ _
  have : Nonempty (Σ _ : F, Fin 1) := e.symm.nonempty
  withSeminorms_induced (withSeminorms_pi (fun _ ↦ norm_withSeminorms 𝕜 𝕜))
    (LinearMap.ltoFun 𝕜 F 𝕜 ∘ₗ B : (WeakBilin B) →ₗ[𝕜] (F → 𝕜)) |>.congr_equiv e


theorem LinearMap.hasBasis_weakBilin (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) :
    (𝓝 (0 : WeakBilin B)).HasBasis B.toSeminormFamily.basisSets _root_.id :=
  LinearMap.weakBilin_withSeminorms B |>.hasBasis


instance WeakBilin.locallyConvexSpace {B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜} :
    LocallyConvexSpace ℝ (WeakBilin B) :=
  B.weakBilin_withSeminorms.toLocallyConvexSpace


