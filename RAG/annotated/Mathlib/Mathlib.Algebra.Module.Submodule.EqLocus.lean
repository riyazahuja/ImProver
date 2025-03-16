/-- A linear map version of `AddMonoidHom.eqLocusM` -/
def eqLocus (f g : F) : Submodule R M :=
  { (f : M →+ M₂).eqLocusM g with
    carrier := { x | f x = g x }
    smul_mem' := fun {r} {x} (hx : _ = _) => show _ = _ by
      -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 changed `map_smulₛₗ` into `map_smulₛₗ _`
      /-
        R : Type u_1
        R₂ : Type u_2
        M : Type u_3
        M₂ : Type u_4
        inst✝⁷ : Semiring R
        inst✝⁶ : Semiring R₂
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : Module R M
        inst✝² : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        F : Type u_5
        inst✝¹ : FunLike F M M₂
        inst✝ : SemilinearMapClass F τ₁₂ M M₂
        f g : F
        r : R
        x : M
        hx : Eq (f x) (g x)
        ⊢ Eq (f (HSMul.hSMul r x)) (g (HSMul.hSMul r x))
      -/
      simpa only [map_smulₛₗ _] using congr_arg (τ₁₂ r • ·) hx }
      /-
        🎉 no goals
      -/


@[simp]
theorem mem_eqLocus {x : M} {f g : F} : x ∈ eqLocus f g ↔ f x = g x :=
  Iff.rfl


theorem eqLocus_toAddSubmonoid (f g : F) :
    (eqLocus f g).toAddSubmonoid = (f : M →+ M₂).eqLocusM g :=
  rfl


@[simp]
theorem eqLocus_eq_top {f g : F} : eqLocus f g = ⊤ ↔ f = g := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_3
    M₂ : Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_5
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f g : F
    ⊢ Iff (Eq (LinearMap.eqLocus f g) Top.top) (Eq f g)
  -/
  simp [SetLike.ext_iff, DFunLike.ext_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem eqLocus_same (f : F) : eqLocus f f = ⊤ := eqLocus_eq_top.2 rfl


theorem le_eqLocus {f g : F} {S : Submodule R M} : S ≤ eqLocus f g ↔ Set.EqOn f g S := Iff.rfl


include τ₁₂ in
theorem eqOn_sup {f g : F} {S T : Submodule R M} (hS : Set.EqOn f g S) (hT : Set.EqOn f g T) :
    Set.EqOn f g ↑(S ⊔ T) := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_3
    M₂ : Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_5
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f g : F
    S T : Submodule R M
    hS : Set.EqOn ⇑f ⇑g ↑S
    hT : Set.EqOn ⇑f ⇑g ↑T
    ⊢ Set.EqOn ⇑f ⇑g ↑(Max.max S T)
  -/
  rw [← le_eqLocus] at hS hT ⊢
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_3
    M₂ : Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring R₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    F : Type u_5
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f g : F
    S T : Submodule R M
    hS : LE.le S (LinearMap.eqLocus f g)
    hT : LE.le T (LinearMap.eqLocus f g)
    ⊢ LE.le (Max.max S T) (LinearMap.eqLocus f g)
  -/
  exact sup_le hS hT
  /-
    🎉 no goals
  -/


include τ₁₂ in
theorem ext_on_codisjoint {f g : F} {S T : Submodule R M} (hST : Codisjoint S T)
    (hS : Set.EqOn f g S) (hT : Set.EqOn f g T) : f = g :=
  DFunLike.ext _ _ fun _ ↦ eqOn_sup hS hT <| hST.eq_top.symm ▸ trivial


theorem eqLocus_eq_ker_sub (f g : M →ₛₗ[τ₁₂] M₂) : eqLocus f g = ker (f - g) :=
  SetLike.ext fun _ => sub_eq_zero.symm


