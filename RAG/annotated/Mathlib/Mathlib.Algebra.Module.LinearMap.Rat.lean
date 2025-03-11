/-- Reinterpret an additive homomorphism as a `ℚ`-linear map. -/
def AddMonoidHom.toRatLinearMap [AddCommGroup M] [Module ℚ M] [AddCommGroup M₂] [Module ℚ M₂]
    (f : M →+ M₂) : M →ₗ[ℚ] M₂ :=
  { f with map_smul' := map_rat_smul f }


theorem AddMonoidHom.toRatLinearMap_injective [AddCommGroup M] [Module ℚ M] [AddCommGroup M₂]
    [Module ℚ M₂] : Function.Injective (@AddMonoidHom.toRatLinearMap M M₂ _ _ _ _) := by
  /-
    M : Type u_1
    M₂ : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module Rat M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module Rat M₂
    ⊢ Function.Injective AddMonoidHom.toRatLinearMap
  -/
  intro f g h
  /-
    M : Type u_1
    M₂ : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module Rat M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module Rat M₂
    f g : AddMonoidHom M M₂
    h : Eq f.toRatLinearMap g.toRatLinearMap
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    M : Type u_1
    M₂ : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module Rat M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module Rat M₂
    f g : AddMonoidHom M M₂
    h : Eq f.toRatLinearMap g.toRatLinearMap
    x : M
    ⊢ Eq (f x) (g x)
  -/
  exact LinearMap.congr_fun h x
  /-
    🎉 no goals
  -/


@[simp]
theorem AddMonoidHom.coe_toRatLinearMap [AddCommGroup M] [Module ℚ M] [AddCommGroup M₂]
    [Module ℚ M₂] (f : M →+ M₂) : ⇑f.toRatLinearMap = f :=
  rfl

